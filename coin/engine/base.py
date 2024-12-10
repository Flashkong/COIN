import os
import torch
import numpy as np
from collections import OrderedDict
from torch.cuda.amp import autocast
import detectron2.utils.comm as comm
from detectron2.engine import DefaultTrainer
from detectron2.utils.events import EventStorage
from detectron2.structures import Boxes
from detectron2.evaluation import COCOEvaluator,DatasetEvaluator,inference_on_dataset
from detectron2.utils.logger import setup_logger
from coin.solver.build import build_lr_scheduler,build_optimizer
from coin.data.build import build_detection_test_loader
from coin.evaluation import verify_results,print_csv_format
from coin.evaluation import Cloud_PascalVOCDetectionEvaluator
from coin.data.dataset_mapper import DatasetMapperUnsupervised
from coin.data.build import build_detection_unsupervised_train_loader
from coin.utils.util import MyInstances, default_writers
import copy

class BASE_Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if cfg.TEST.EVALUATOR == "VOCeval":
            return Cloud_PascalVOCDetectionEvaluator(cfg, dataset_name)
        else:
            raise ValueError("Unknown test evaluator.")

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperUnsupervised(cfg, True)
        return build_detection_unsupervised_train_loader(cfg, mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)
    
    @classmethod
    def build_optimizer(cls, cfg, model, name):
        return build_optimizer(cfg, model, name)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        self.logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    # with torch.autograd.set_detect_anomaly(True):
                    self.run_step()
                    self.after_step()
            except Exception:
                self.logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
                self.log_final_accs()

    def log_final_accs(self,):
        pass

    def process(self, instances,old_size,new_size,random_flip,thresh=None,keep_name=False):
            img_height,img_width = old_size
            net_height,net_width = new_size
            newinstances = MyInstances((net_height,net_width))
            newinstances._fields = copy.deepcopy(instances.get_fields())
            if newinstances.has('pred_boxes'):
                boxes = newinstances.pred_boxes
            else:
                boxes = newinstances.gt_boxes
            boxes.scale(net_width/img_width,net_height/img_height)
            if random_flip=='horizontal':
                boxes_flipped = boxes.tensor.clone() # copy
                boxes_flipped[:, 0] = net_width - boxes.tensor[:, 2] # Calculate xmin after flipping
                boxes_flipped[:, 2] = net_width - boxes.tensor[:, 0] # Calculate xmax after flipping
                boxes = Boxes(boxes_flipped)
            elif random_flip=='vertical':
                boxes_flipped = boxes.tensor.clone() # copy
                boxes_flipped[:, 1] = net_height - boxes.tensor[:, 3] # Calculate ymin after flipping
                boxes_flipped[:, 3] = net_height - boxes.tensor[:, 1] # Calculate ymax after flipping
                boxes = Boxes(boxes_flipped)
            elif random_flip=='no':
                pass
            else:
                raise NotImplementedError
            if newinstances.has('pred_boxes'):
                if keep_name:
                    newinstances.set('pred_boxes', boxes)
                else:
                    newinstances.remove('pred_boxes')
                    newinstances.set('gt_boxes', boxes)
            else:
                newinstances.set('gt_boxes', boxes)
            if not keep_name:
                newinstances.set('gt_classes', newinstances.get('pred_classes'))
                newinstances.remove('pred_classes')
            
            if thresh is not None:
                index = instances.scores >= thresh # 这里要用instances，用之前的scores筛选
                thresh_instances = newinstances[index]
                # Ensure that those not exceeding the threshold, although not trained, are not used as background either.
                # Not used.
                # thresh_instances.set('no_thresh_boxes',newinstances[~index].gt_boxes, check_len=False)
                del newinstances, instances
                return thresh_instances
            else:
                del instances
                return newinstances
    
    def preprocess_results(self, results, new_image_size, random_flip, thresh=None):
        results['RCNN'] = self.process(results['RCNN']['instances'],(results['height'],results['width']),new_image_size,random_flip,thresh)
        if 'RPN_AUG' in results.keys():
            del results['RPN']
            results['RPN'] = self.process(results['RPN_AUG']['instances'],(results['height'],results['width']),new_image_size,random_flip,thresh)
            del results['RPN_AUG']
        else:
            results['RPN'] = self.process(results['RPN']['instances'],(results['height'],results['width']),new_image_size,random_flip,thresh)
        return results
    
    def clip_gradient(self, model, clip_norm):
        """Computes a gradient clipping coefficient based on gradient norm."""
        totalnorm = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                modulenorm = p.grad.norm()
                totalnorm += modulenorm ** 2
        totalnorm = torch.sqrt(totalnorm).item()
        norm = (clip_norm / max(totalnorm, clip_norm))
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.mul_(norm)

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
        logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
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
            # if True:
            with autocast():
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
    
    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                # k: np.mean([x[k] if k in x.keys() else 0.0 for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)
    

    def build_writers(self, window_size):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter, window_size)
