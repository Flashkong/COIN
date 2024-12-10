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
from detectron2.utils.logger import setup_logger

from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
)

from coin.engine.hooks import MyPeriodicCheckpointer, MyLRScheduler, MyEvalHook
from coin.modeling.meta_arch import build_model
from coin.checkpoint.detection_checkpoint import DetectionTSCheckpointer
import gc
from coin.engine.base import BASE_Trainer

class OracleTrainer(BASE_Trainer):
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
        model = self.build_model(cfg)
        
        optimizer = self.build_optimizer(cfg, model, name='cls')
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
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg)
    
    @classmethod
    def build_model(cls, cfg):
        models = build_model(cfg)
        return models
    
    def save(self, iteration, model_name='model'):
        additional_state = {
            "iteration": iteration}
        self.checkpointer.save(
            "{}_{:07d}".format(model_name, iteration), **additional_state
        )
    
    def log_final_accs(self,):
        self.logger.info('oracle acc: ')
        accs = []
        for k,v in self.ap_50.items():
            accs.append(f'{k}:{v}')
        self.logger.info('\n'+'\n'.join(accs))
    
    def after_step(self):
        if self.iter == self.cfg.SOLVER.MAX_ITER -1:
            self.save(iteration = self.iter, model_name = 'oracle_'+self.cfg.CLOUD.PRE_TRAIN_NAME)
        super().after_step()

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================
    def run_step(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[PTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        
        data_time = time.perf_counter() - start

        # oracle stage
        if self.iter < self.cfg.SOLVER.MAX_ITER:
            with autocast():
                loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

            self.optimizer.zero_grad()
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        del loss_dict
        del losses
        torch.cuda.empty_cache()
        gc.collect()
    
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
        if resume:
            checkpoint = self.checkpointer.load(
                self.cfg.MODEL.WEIGHTS, checkpointables=['optimizer', 'scheduler']
            )
        else:
            checkpoint = self.checkpointer.load(
                self.cfg.MODEL.WEIGHTS, checkpointables=[]
            )
        if self.cfg.MODEL.WEIGHTS!='':
            self.start_iter = checkpoint.get("iteration", -1) + 1
            self.ap_50 = checkpoint.get('ap_50', {})
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]
        del checkpoint
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
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model) # not used by default
            else None,
        ]

        # do test before save checkpoint to save ap_50 into the saved checkpoint
        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            self.logger.info('test oracle model complete! For iter: {}'.format(self.iter))
            _last_eval_results = {
                k + "_student": self._last_eval_results[k]
                for k in self._last_eval_results.keys()
            }
            self.ap_50[self.iter] = self._last_eval_results['bbox']['AP50']
            return _last_eval_results

        ret.append(MyEvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(MyPeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, file_prefix = 'oracle',
            ap_50=self.ap_50))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            period = 20
            ret.append(hooks.PeriodicWriter(self.build_writers(window_size=period), period=period))
        return ret
