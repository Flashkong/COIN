import os
import torch
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
import detectron2.utils.comm as comm
from detectron2.engine import DefaultTrainer, TrainerBase
from detectron2.evaluation import COCOEvaluator,DatasetEvaluator,inference_on_dataset
from detectron2.utils.logger import setup_logger
from coin.modeling.meta_arch import build_model
from coin.data.build import build_detection_test_loader
from coin.evaluation import print_csv_format
from coin.evaluation import Cloud_PascalVOCDetectionEvaluator
from coin.data.dataset_mapper import GDINOMapper,TESTMapper
from groundingdino.util.misc import clean_state_dict


# GDINO test Trainer
class GDINOTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = self.auto_scale_workers(cfg, comm.get_world_size())
        # create GDINO model
        model = self.build_model(cfg)
        TrainerBase.__init__(self)
        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
    
    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        return model
        
    @classmethod
    def load_model(cls, cfg, model):
        checkpoint = torch.load(cfg.MODEL.TEACHER_CLOUD.WEIGHT, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        del checkpoint
        logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
        if cfg.MODEL.TEACHER_CLOUD.META_ARCHITECTURE != 'GDINO1_5_API':
            logger.info("GDINO Model Loaded from: {}".format(cfg.MODEL.TEACHER_CLOUD.WEIGHT))
        model.eval()
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if cfg.TEST.EVALUATOR == "COCOeval":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif cfg.TEST.EVALUATOR == "VOCeval":
            return Cloud_PascalVOCDetectionEvaluator(cfg, dataset_name)
        else:
            raise ValueError("Unknown test evaluator.")
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name,mapper=GDINOMapper(cfg))
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = setup_logger(cfg.OUTPUT_DIR, name = __name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(cfg, results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

class CLIPTrainer(GDINOTrainer):
    @classmethod
    def build_cloud_loader(cls,cfg):
        loaders = {}
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            if cfg.MODEL.TEACHER_CLOUD.PROCESSOR_ARCHITECTURE in ['GDINO', 'GDINO_CLASSONLY', 'GDINO1_5_API']:
                mapper = GDINOMapper(cfg)
            elif cfg.MODEL.TEACHER_CLOUD.PROCESSOR_ARCHITECTURE == 'GLIP':
                mapper = TESTMapper(cfg, False)
            else:
                raise NotImplementedError
            data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
            loaders[dataset_name] = data_loader
        return loaders

    @classmethod
    def build_clip_loader(cls,cfg):
        loaders = {}
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = build_detection_test_loader(cfg, dataset_name)
            loaders[dataset_name] = data_loader
        return loaders
    
    @classmethod
    def build_model(cls, cfg):
        model_CLOUD, model_CLIP = build_model(cfg)
        model_CLIP.eval()
        model_CLOUD.set('dataloader',cls.build_cloud_loader(cfg))
        model_CLIP.set('dataloader',cls.build_clip_loader(cfg))
        return model_CLOUD, model_CLIP
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)
    
    @classmethod
    def collect_results(cls, model_CLOUD, model_CLIP):
        model_CLOUD.collect()
        model_CLIP.collect(model_CLOUD.get_results())

class GLIPTrainer(GDINOTrainer):

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)
    
    @classmethod
    def load_model(cls, cfg, model):
        # The model weights have been loaded when building the model
        return model
