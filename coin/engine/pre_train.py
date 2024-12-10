import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
from torch.cuda.amp import autocast, GradScaler
import detectron2.utils.comm as comm
from detectron2.engine import SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer

from detectron2.engine import hooks
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from coin.engine.hooks import MyPeriodicCheckpointer, MyLRScheduler, MyEvalHook

from coin.modeling.meta_arch import build_model
from coin.data.build import build_detection_test_loader
from coin.data.dataset_mapper import COLLECTMapper
from coin.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from groundingdino.util.misc import clean_state_dict
import copy
import gc
from coin.engine.base import BASE_Trainer

class PRETrainer(BASE_Trainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = self.auto_scale_workers(cfg, comm.get_world_size())
        # cfg = self.auto_scale_workers(cfg, cfg.SOLVER.IMG_PER_BATCH_LABEL)
        self.cfg = cfg

        data_loader = self.build_train_loader(cfg)
        # create an model
        if self.cfg.CLOUD.PRE_TRAIN_NAME == 'CLIP':
            if self.cfg.MODEL.WEIGHTS=='':
                model, collect_model, model_CLOUD = self.build_model(cfg)
                if cfg.MODEL.TEACHER_CLOUD.META_ARCHITECTURE in ['GDINO', 'GDINO_CLASSONLY', 'GDINO1_5_API']:
                    model_CLOUD.set('dataloader',self.build_gdino_loader(cfg))
                    model_CLOUD = self.load_gdino_model(cfg, model_CLOUD)
                elif self.cfg.MODEL.TEACHER_CLOUD.META_ARCHITECTURE == 'GLIP':
                    model_CLOUD.set('dataloader',self.build_glip_loader(cfg))
                else:
                    raise NotImplementedError
                self.model_CLOUD = model_CLOUD
            else:
                model, collect_model = self.build_model(cfg)
            collect_model.set('dataloader',self.build_clip_loader(cfg))
            collect_model.eval()
        else: raise NotImplementedError

        self.collect_model = collect_model
        optimizer = self.build_optimizer(cfg, model, name='all')
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer=None)
        self.optimizer = optimizer
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        self.class_num = len(MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes)
        
        self.checkpointer = DetectionTSCheckpointer(
            self.model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.ap_50 = {}
        
        self.register_hooks(self.build_hooks())
        self.scaler = GradScaler()

        # merlin to save memeory
        def inplace_relu(m):
            classname = m.__class__.__name__
            if classname.find('ReLU') != -1:
                m.inplace = True

        self.model.apply(inplace_relu)
        self.logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
    
    @classmethod
    def build_model(cls, cfg):
        models = build_model(cfg)
        return models
    
    @classmethod
    def load_gdino_model(cls, cfg, model):
        checkpoint = torch.load(cfg.MODEL.TEACHER_CLOUD.WEIGHT, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        del checkpoint
        logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
        if cfg.MODEL.TEACHER_CLOUD.META_ARCHITECTURE != 'GDINO1_5_API':
            logger.info("GDINO Model Loaded from: {}".format(cfg.MODEL.TEACHER_CLOUD.WEIGHT))
        model.eval()
        return model
    
    @classmethod
    def build_gdino_loader(cls,cfg):
        loaders = {}
        for idx, dataset_name in enumerate(cfg.DATASETS.TRAIN_UNLABEL):
            data_loader = build_detection_test_loader(cfg, dataset_name, mapper=COLLECTMapper(cfg))
            loaders[dataset_name] = data_loader
        return loaders

    @classmethod
    def build_glip_loader(cls,cfg):
        loaders = {}
        for idx, dataset_name in enumerate(cfg.DATASETS.TRAIN_UNLABEL):
            data_loader = build_detection_test_loader(cfg, dataset_name)
            loaders[dataset_name] = data_loader
        return loaders

    @classmethod
    def build_clip_loader(cls,cfg):
        loaders = {}
        for idx, dataset_name in enumerate(cfg.DATASETS.TRAIN_UNLABEL):
            data_loader = build_detection_test_loader(cfg, dataset_name)
            loaders[dataset_name] = data_loader
        return loaders
    
    def delete_model(self):
        self.collect_model.delete_model()
    
    def save(self,iteration, load_models, model_name='model'):
        additional_state = {
            "iteration": iteration, 
            "results": self.collect_model.get_results()}
        additional_state.update({"load_models":load_models} if not load_models else {})

        self.checkpointer.save(
            "{}_{:07d}".format(model_name, iteration), **additional_state
        )
    
    def collect_results(self):
        # collect
        if self.cfg.CLOUD.PRE_TRAIN_NAME == 'CLIP':
            self.model_CLOUD.collect()
            online_results = {'results': self.model_CLOUD.get_results()}
            torch.save(online_results, os.path.join(self.cfg.OUTPUT_DIR,'GDINO_collect.pth'))
            self.collect_model.collect(self.model_CLOUD.get_results())
            del self.model_CLOUD
            torch.cuda.empty_cache()
        else:
            raise NotImplementedError
        
        self.delete_model()
        self.save(iteration=-1, load_models=False, model_name='' + self.cfg.CLOUD.PRE_TRAIN_NAME)

    def log_final_accs(self,):
        self.logger.info('student acc: ')
        accs = []
        for k,v in self.ap_50.items():
            accs.append(f'{k}:{v}')
        self.logger.info('\n'+'\n'.join(accs))
    
    def after_step(self):
        if self.iter == self.cfg.SOLVER.MAX_ITER -1:
            self.save(iteration = self.iter, load_models=True, model_name = 'pre_train_'+self.cfg.CLOUD.PRE_TRAIN_NAME)
        super().after_step()

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================
    def run_step(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[PTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        unlabel_data_s, unlabel_data_w = data
        data_time = time.perf_counter() - start

        # pre-train-stage
        if self.iter < self.cfg.SOLVER.MAX_ITER:
            [unlabel_data_s, unlabel_data_w] = self.set_boxes([unlabel_data_s, unlabel_data_w], thresh=0.5 if self.cfg.DATASETS.TRAIN_UNLABEL==("cliparttrain",) else None)
            
            # input both strong and weak supervised data into model
            unlabel_data_s.extend(unlabel_data_w)

            if self.cfg.CLOUD.PROTOTYPE_UPDATE_START == -1: update_prototype = False
            else: update_prototype = True if self.iter >= self.cfg.CLOUD.PROTOTYPE_UPDATE_START else False
            with autocast():
                record_dict = self.model(unlabel_data_s, branch="pre_train", update_prototype=update_prototype)

            losses = sum(record_dict.values())
            self.optimizer.zero_grad()
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        del record_dict
        del losses
        torch.cuda.empty_cache()
        gc.collect()

    def set_boxes(self, unlabel_datas, thresh=None):
        for unlabel_data in unlabel_datas:
            for data_dict in unlabel_data:
                name = data_dict['file_name']
                random_flip = data_dict['random_flip']
                collect_result = self.collect_model(name)
                assert collect_result['height']==data_dict['height']
                assert collect_result['width']==data_dict['width']
                assert collect_result['image_id'] == data_dict['image_id']
                collect_result = self.preprocess_results(collect_result,data_dict['image'].size()[1:],random_flip,thresh=thresh)
                RCNN_instances = collect_result['RCNN']
                RCNN_instances.gt_classes_offline = RCNN_instances.gt_classes
                RCNN_instances.gt_probs_offline = RCNN_instances.probs
                RCNN_instances.gt_scores_offline = RCNN_instances.scores
                RCNN_instances.remove('scores')
                RCNN_instances.remove('gt_classes')
                RCNN_instances.remove('probs')
                data_dict['RCNN'] = RCNN_instances
                RPN_instances = collect_result['RPN']
                RPN_instances.remove('scores')
                RPN_instances.remove('probs')
                data_dict['RPN'] = RPN_instances
                del collect_result
        return unlabel_datas
    
    def resume_or_load(self, resume=False):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        model_stat = copy.deepcopy(self.model.state_dict())
        if resume:
            checkpoint = self.checkpointer.load(
                # The scheduler loaded here will cause setting a new learning rate to be useless because the old learning rate already exists in the scheduler
                self.cfg.MODEL.WEIGHTS, checkpointables=['optimizer', 'scheduler']
            )
        else:
            # only load model parameters
            checkpoint = self.checkpointer.load(
                self.cfg.MODEL.WEIGHTS, checkpointables=[]
            )
        if self.cfg.MODEL.WEIGHTS!='':
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # load collected results
            self.collect_model.set_results(checkpoint.get('results'))
            self.ap_50 = checkpoint.get('ap_50', {})
            # In order to ensure the current random seed, the current random parameters can be generated instead of the previous random parameters.
            if 'load_models' in checkpoint.keys():
                if not checkpoint['load_models']:
                    self.model.load_state_dict(model_stat)
            del model_stat
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]
        torch.cuda.empty_cache()

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            MyLRScheduler(self.optimizer, self.scheduler, 'lr'),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model) 
            else None,
        ]

        # do test before save checkpoint to save ap_50 into the saved checkpoint
        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            self.logger.info('test pre-train model complete! For iter: {}'.format(self.iter))
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            self.ap_50[self.iter] = self._last_eval_results_student['bbox']['AP50']
            return _last_eval_results_student

        ret.append(MyEvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_student))

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(MyPeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, file_prefix = self.cfg.CLOUD.PRE_TRAIN_NAME,
                    results=self.collect_model, ap_50=self.ap_50))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            period = 20
            ret.append(hooks.PeriodicWriter(self.build_writers(window_size=period), period=period))
        return ret
