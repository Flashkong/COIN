import os
import shutil
import torch
from detectron2.config import CfgNode, LazyConfig
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import _try_get_key, _highlight
import argparse
import sys
import datetime
import numpy as np
import random
from detectron2.structures import Instances
from typing import Any, List, Union, Optional
import itertools
from detectron2.utils.events import JSONWriter, TensorboardXWriter, EventWriter, get_event_storage
import time
import supervision as sv
import cv2
from PIL import Image
from detectron2.structures import pairwise_iou
from detectron2.utils.memory import retry_if_cuda_oom


def copy_codes(s,t,recursion=True):
    for root, dirs, files in os.walk(s) :
        if not recursion:
            dirs[:] = []
        for file in files:
            if file.endswith(".py") or file.endswith(".sh"):
                src = os.path.join(root, file)
                rel_path = src[len(s):].lstrip(os.sep)
                dst = os.path.join(t, rel_path)
                try:
                    os.makedirs(os.path.dirname(dst))
                except OSError as e:
                    if e.errno != 17: # 17 means 'File exists'
                        raise
                shutil.copyfile(src, dst)


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                PathManager.open(args.config_file, "r", encoding="utf-8").read()
            )
        )

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            logger.info("Running with full config:\n{}".format(cfg.dump(), ".yaml"))
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed = _try_get_key(cfg, "SEED", "train.seed", default=-1)
    seed_all_rng(cfg, None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )


def seed_all_rng(cfg, seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--info", default='')
    parser.add_argument("--test_model_role", default='targetdet')
    return parser


class MyInstances(Instances):

    def set(self, name: str, value: Any, check_len=True) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        if check_len:
            data_len = len(value)
            if len(self._fields):
                assert (
                    len(self) == data_len
                ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value
    
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = MyInstances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v, check_len=False)
        return ret
    
    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = MyInstances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item], check_len=False)
        return ret
    
    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        if not isinstance(image_size, torch.Tensor):  # could be a tensor in tracing
            for i in instance_lists[1:]:
                assert i.image_size == image_size
        ret = MyInstances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values, check_len=False)
        return ret
    

class CommonMetricPrinter(EventWriter):
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    It also applies smoothing using a window of 20 elements.

    It's meant to print common metrics in common ways.
    To print something in more customized ways, please implement a similar printer by yourself.
    """

    def __init__(self, output_dir, max_iter: Optional[int] = None, window_size: int = 20):
        """
        Args:
            max_iter: the maximum number of iterations to train.
                Used to compute ETA. If not given, ETA will not be printed.
            window_size (int): the losses will be median-smoothed by this window size
        """
        self.logger = setup_logger(output_dir, name = __name__, distributed_rank = comm.get_rank())
        self._max_iter = max_iter
        self._window_size = window_size
        self._last_write = None  # (step, time) of last call to write(). Used to compute ETA

    def _get_eta(self, storage) -> Optional[str]:
        if self._max_iter is None:
            return ""
        iteration = storage.iter
        try:
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration - 1)
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            return str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            # estimate eta on our own - more noisy
            eta_string = None
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                    iteration - self._last_write[0]
                )
                eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())
            return eta_string

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None
        try:
            # iter_time = storage.history("time").global_avg()
            iter_time = storage.history("time").latest()
        except KeyError:
            iter_time = None
        try:
            lr = "{:.5g}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        try:
            merge_lr = "{:.5g}".format(storage.history("merge_lr").latest())
        except KeyError:
            merge_lr = "N/A"
        
        try:
            WEIGHT_FOR_BOX_A = "{:.5g}".format(storage.history("WEIGHT_FOR_BOX_A").latest())
        except KeyError:
            WEIGHT_FOR_BOX_A = "N/A"

        eta_string = self._get_eta(storage)

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            " {eta}iter: {iter}  {losses}  {time}{data_time}lr: {lr}  merge_lr: {merge_lr}  WEIGHT_FOR_BOX_A: {WEIGHT_FOR_BOX_A}  {memory}".format(
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                losses="  ".join(
                    [
                        "{}: {:.4g}".format(k, v.median(self._window_size))
                        for k, v in storage.histories().items()
                        if "loss" in k
                    ]
                ),
                # losses="  ".join(
                #     [
                #         "{}: {:.4g}".format(k, v.latest())
                #         for k, v in storage.histories().items()
                #         if "loss" in k
                #     ]
                # ),
                time="time: {:.4f}  ".format(iter_time) if iter_time is not None else "",
                data_time="data_time: {:.4f}  ".format(data_time) if data_time is not None else "",
                lr=lr,
                merge_lr=merge_lr,
                WEIGHT_FOR_BOX_A=WEIGHT_FOR_BOX_A,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )

def default_writers(output_dir: str, max_iter: Optional[int] = None, window_size=20):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(output_dir, max_iter,window_size=window_size),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]

def draw(instances, batched_inputs, save_dir, type):
    os.makedirs(save_dir, exist_ok=True)
    for i, (instance, batched_input) in enumerate(zip(instances, batched_inputs)):
        if instance.has('gt_boxes'):
            boxes = instance.gt_boxes
        else: boxes = instance.pred_boxes
        boxes = boxes.tensor.cpu()
        if instance.has('gt_classes'): pass
        else: instance.gt_classes = instance.pred_classes
        try:
            logits = instance.scores.cpu()
        except:
            logits = torch.ones_like(instance.gt_classes).cpu()
        phrases = instance.gt_classes.cpu().tolist()
        phrases = [str(i) for i in phrases]

        detections = sv.Detections(xyxy=boxes.numpy())
        labels = [
            f"{phrase} {logit:.2f}"
            for phrase, logit
            in zip(phrases, logits)
        ]
        box_annotator = sv.BoxAnnotator()

        a = batched_input["image"].permute((1,2,0)).contiguous().cpu().numpy()
        img_pil = Image.fromarray(a)
        annotated_frame = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        file_name = '.'.join((batched_input['file_name'].split('/')[-1]).split('.')[:-1]) + '_' + str(i) + '_' + type + '.' + batched_input['file_name'].split('.')[-1]
        save_name = os.path.join(save_dir, file_name)
        cv2.imwrite(save_name, annotated_frame)

def delete_duplicate_boxes(instances, return_split=False):
    # Here we randomly select one, both box a and box b are
    boxes_sum = instances.gt_boxes.tensor.sum(1)
    unique = torch.unique(boxes_sum)
    matrix = torch.eq(unique.unsqueeze(1), boxes_sum)
    mask = matrix.sum(1)!=1
    same_box_matrix = matrix[mask]
    outs = []
    for i in range(same_box_matrix.size(0)):
        m = same_box_matrix[i]
        # Situations where the boxes are different but the same.
        if (instances[m].gt_boxes.tensor - instances[m].gt_boxes.tensor[0] ).sum()==0:
            same_box_instance = instances[m]
            if return_split:
                outs.append(same_box_instance)
            else:
                outs.append(same_box_instance[random.randint(0,len(same_box_instance)-1)])
        else:
            ids = same_box_matrix[i].nonzero()[:,0]
            same_box_matrix[i][ids] = False
    if return_split:
        return instances[(same_box_matrix.sum(0)==0).nonzero()[:,0]], outs
    instances = Instances.cat([instances[(same_box_matrix.sum(0)==0).nonzero()[:,0]]]+outs)
    return instances

def find_same(sets, ups, i):
    for j in sets[i]:
        if j != i and j not in ups:
            if sets[j]-sets[i] == set(): pass
            else: sets[i] = sets[i] | find_same(sets, ups+[i], j)
    return sets[i]

def filter_result(result, thresh):
    boxes = result.gt_boxes
    iou_matrix = retry_if_cuda_oom(pairwise_iou)(boxes, boxes) >= thresh
    sets = []
    for i in range(len(boxes)):
        sets.append(set(iou_matrix[i].nonzero()[:,0].tolist()))
    for i in range(len(sets)):
        for j in sets[i]:
            ups = [] 
            if j != i:
                sets[i] = sets[i] | find_same(sets, ups+[i], j)
        for j in sets[i]:
            if j != i:
                sets[j] = set()
    sets = [i for i in sets if len(i)!=0]
    self_match_result = [result[list(i)] for i in sets if len(i)!=1]
    return self_match_result

def online_boxes_merging(instances, common_instances_offline, common_instances_online):
    self_match_result = filter_result(instances, 0.95)
    if len(self_match_result)!=0:
        for result in self_match_result:
            assert result.gt_classes.unique().size(0) != 1
            box = result.gt_boxes.tensor
            comman_box = common_instances_online.gt_boxes.tensor
            matrix = torch.eq(box.unsqueeze(1), comman_box).sum(-1) == 4
            comman_index = torch.unique(matrix.nonzero()[:,1])
            a = torch.ones(len(common_instances_online))
            a[comman_index] = 0
            other_index = a.nonzero()[:,0]
            s = matrix[0].nonzero()[:,0] # The self-matching threshold is 0.95, which is very high, so we assume that if a box matches the first self-matching box, it will definitely match the second one as well.
            if common_instances_offline.gt_classes[s].unique().size(0) == 1: # If the predictions of the categories for the matched offline boxes are all consistent
                mask = common_instances_online[comman_index].gt_classes == common_instances_offline.gt_classes[s].unique()
                if mask.sum()!=0:   comman_index = comman_index[mask]
                else:   pass
            else:
                # If the predictions from multiple offline boxes are inconsistent with each other, then this box should be treated as a B box.
                mask = common_instances_online[comman_index].gt_classes != common_instances_offline.gt_classes[comman_index]
                comman_index = comman_index[mask]
            common_instances_online = Instances.cat([common_instances_online[other_index], common_instances_online[comman_index]])
            common_instances_offline = Instances.cat([common_instances_offline[other_index], common_instances_offline[comman_index]])
    return common_instances_offline, common_instances_online

