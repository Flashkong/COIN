# Copyright (c) Shuaifeng Li at UESTC. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import torch
import copy
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
from torch.cuda.amp import autocast, GradScaler
import detectron2.utils.comm as comm
from detectron2.engine import SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.engine import hooks
from detectron2.structures import pairwise_iou, Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from coin.engine.hooks import MyPeriodicCheckpointer, MyLRScheduler, MyEvalHook
from coin.modeling.meta_arch import build_model,EnsembleTSModel
from coin.modeling.merge import build_merge
from coin.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from coin.utils.losses import gradient_discrepancy_loss
import gc
from coin.engine.base import BASE_Trainer
from coin.utils.util import delete_duplicate_boxes, online_boxes_merging
import random
from detectron2.utils.memory import retry_if_cuda_oom
from coin.layers.nms import weighted_box_fusion_split

class CoinTrainer(BASE_Trainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = self.auto_scale_workers(cfg, comm.get_world_size())
        # cfg = self.auto_scale_workers(cfg, cfg.SOLVER.IMG_PER_BATCH_LABEL)
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model, offline_teacher, model_CLOUD = self.build_model(cfg)
        merge = self.build_merge(cfg)
        model_CLOUD.delete_model()
        for p in offline_teacher.parameters():
            p.requires_grad = False
        self.offline_teacher = offline_teacher
        self.model_CLOUD = model_CLOUD
        self.merge = merge
        optimizer = self.build_optimizer(cfg, model, name='all')
        optimizer_merge = self.build_optimizer(cfg, merge, name='all')

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
            merge = DistributedDataParallel(
                merge, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer=None)
        self.optimizer = optimizer
        self.optimizer_merge = optimizer_merge
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        self.scheduler_merge = self.build_lr_scheduler(cfg, self.optimizer_merge)
        self.class_num = len(MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes)
        # Ensemble teacher and student model is for model saving and loading
        self.ensem_ts_model = EnsembleTSModel(offline_teacher, model_CLOUD, model, merge, cfg.OUTPUT_DIR)
        
        self.checkpointer = DetectionTSCheckpointer(
            self.ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            optimizer_merge=self.optimizer_merge,
            scheduler=self.scheduler,
            scheduler_merge=self.scheduler_merge
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.ap_50_student = {}
        self.ap_50_offline_teacher = {}
        
        self.register_hooks(self.build_hooks())
        self.scaler = GradScaler()

        # merlin to save memeory
        def inplace_relu(m):
            classname = m.__class__.__name__
            if classname.find('ReLU') != -1:
                m.inplace = True

        self.offline_teacher.apply(inplace_relu)
        self.model.apply(inplace_relu)
        self.logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
        self.WEIGHT_FOR_BOX_A = 1.0
    
    @classmethod
    def build_model(cls, cfg):
        return build_model(cfg)

    @classmethod
    def build_merge(cls, cfg):
        merge = build_merge(cfg)
        return merge
    
    def save(self,iteration, model_name='model'):
        additional_state = {
            "iteration": iteration, 
            "ap_50_student": self.ap_50_student,
            "ap_50_offline_teacher": self.ap_50_offline_teacher,
            "online_results": self.model_CLOUD.get_results()}
        
        self.checkpointer.save(
            "{}_{:07d}".format(model_name, iteration), **additional_state
        )

    def log_final_accs(self,):
        self.logger.info('student acc: ')
        stu = []
        for k,v in self.ap_50_student.items():
            stu.append(f'{k}:{v}')
        self.logger.info('\n'+'\n'.join(stu))
        self.logger.info('teacher acc: ')
        tea = []
        for k,v in self.ap_50_offline_teacher.items():
            tea.append(f'{k}:{v}')
        self.logger.info('\n'+'\n'.join(tea))
    
    def before_step(self):
        super().before_step()

        
    def after_step(self):
        if self.iter == self.cfg.CLOUD.BURN_UP_STEP -1:
            self.save(iteration = self.iter, model_name = 'burn_up')
        if self.iter >= self.cfg.CLOUD.BURN_UP_STEP:
            self.WEIGHT_FOR_BOX_A = 0.5
        self.storage.put_scalar("WEIGHT_FOR_BOX_A", self.WEIGHT_FOR_BOX_A)
        super().after_step()

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================
    def run_step(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[PTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # strong augmentation, weak augmentation
        unlabel_data_s, unlabel_data_w = data
        data_time = time.perf_counter() - start

        # Teacher model parameters are not updated until after BURN_UP_STEP steps
        if (self.iter >= self.cfg.CLOUD.BURN_UP_STEP) \
            and (self.iter - self.cfg.CLOUD.BURN_UP_STEP) % self.cfg.CLOUD.OFFLINE_TEACHER_UPDATE_ITER == 0:
                self.ensem_ts_model.update_params(keep_rate=self.cfg.CLOUD.EMA_KEEP_RATE_OFFLINE, name = 'offline')

        with torch.no_grad():
            with autocast():
                self.offline_teacher.eval()
                offline_results = self.offline_teacher(unlabel_data_w, branch = 'test')
                self.offline_teacher.train()
            dual_teacher_instances = self.match_boxes(unlabel_data_w, offline_results)
            
        if self.cfg.CLOUD.PROTOTYPE_UPDATE_START == -1: update_prototype = False
        else: update_prototype = True if self.iter >= self.cfg.CLOUD.PROTOTYPE_UPDATE_START else False
        grad_loss_scale = 1e4

        input_data = unlabel_data_s
        branch = "step_one" if self.iter < self.cfg.CLOUD.BURN_UP_STEP else "step_two"
        with autocast():
            record_dict = self.model(input_data, self.merge, dual_teacher_instances, branch=branch, update_prototype=update_prototype)
            
        self.optimizer.zero_grad()
        self.optimizer_merge.zero_grad()
        if 'loss_merge_a' in record_dict:
            record_dict['loss_merge_grad'] = gradient_discrepancy_loss(self.model, grad_loss_scale * record_dict['loss_merge_a'], grad_loss_scale * record_dict['loss_merge_b'])
            self.optimizer.zero_grad()
            self.optimizer_merge.zero_grad()
            self.scaler.scale(record_dict['loss_merge_grad'] + record_dict['loss_merge_base']).backward(retain_graph=True)
            self.scaler.step(self.optimizer_merge)

        self.optimizer.zero_grad()
        self.optimizer_merge.zero_grad()
        losses = 0
        loss_list = ['loss_merge_grad','loss_merge_a','loss_merge_b', 'loss_merge_base'] if self.iter >= self.cfg.CLOUD.BURN_UP_STEP \
            else ['loss_merge_grad','loss_merge_a','loss_merge_b', 'loss_merge_base', 'loss_cls_b']
        for k,v in record_dict.items():
            if k not in loss_list:
                losses += v
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
    
    def resume_or_load(self, resume=False):
        assert self.cfg.MODEL.WEIGHTS != '', "pretrain models must be loaded!"
        pretrain_model_paths = self.cfg.MODEL.WEIGHTS.split('+')
        if len(pretrain_model_paths)==2:
            if resume: assert False, "resume need only one model."
            checkpoint_offline = torch.load(pretrain_model_paths[0])
            try:
                self.offline_teacher.load_state_dict(checkpoint_offline['model'], strict = False)
            except:
                # for loading model zoo
                self.offline_teacher.load_state_dict(checkpoint_offline, strict = False)
            checkpoint_online = torch.load(pretrain_model_paths[1])
            self.model_CLOUD.set_results(checkpoint_online['results'])
            del checkpoint_offline, checkpoint_online
            self.logger.info('loaded offline pre train model from: '+ pretrain_model_paths[0])
            self.logger.info('loaded online results from: '+ pretrain_model_paths[1])
        elif len(pretrain_model_paths)==1:
            if resume:
                checkpointables = ['optimizer', 'optimizer_merge', 'scheduler', 'scheduler_merge']
                checkpoint = self.checkpointer.load(
                    self.cfg.MODEL.WEIGHTS, checkpointables=checkpointables
                )
                self.logger.info('loaded model with '+ str(checkpointables))
            else:
                checkpoint = self.checkpointer.load(
                    self.cfg.MODEL.WEIGHTS, checkpointables=[]
                )
                self.scheduler.last_epoch = checkpoint.get("iteration", -1)
                self.scheduler_merge.last_epoch = checkpoint.get("iteration", -1)
                self.logger.info('loaded model only. optimizer, optimizer_merge, scheduler, scheduler_merge is not loaded.')
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # load results from checkpoints
            self.model_CLOUD.set_results(checkpoint.get('online_results'))
            self.load_aps(checkpoint)
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
            if isinstance(self.model, DistributedDataParallel):
                # broadcast loaded data/model from the first rank, because other
                # machines may not have access to the checkpoint file
                if TORCH_VERSION >= (1, 7):
                    self.model._sync_params_and_buffers()
                self.start_iter = comm.all_gather(self.start_iter)[0]
        else:
            if resume: assert False, "resume need only one model."
            assert False ,"pretrain models path should be two paths split by '+'. "
        torch.cuda.empty_cache()
    
    def load_aps(self, checkpoint):
        temp = checkpoint.get('ap_50_student', {})
        for k, v in temp.items():
            self.ap_50_student[k] = v
        temp = checkpoint.get('ap_50_offline_teacher', {})
        for k, v in temp.items():
            self.ap_50_offline_teacher[k] = v

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            MyLRScheduler(self.optimizer, self.scheduler, 'lr'), 
            MyLRScheduler(self.optimizer_merge, self.scheduler_merge, 'merge_lr'),
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

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            self.logger.info('test student complete! For iter: {}'.format(self.iter))
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            self.ap_50_student[self.iter] = self._last_eval_results_student['bbox']['AP50']
            if self.iter == self.cfg.TEST.EVAL_PERIOD-1:
                test_and_save_results_teacher()
            elif self.iter <= self.cfg.CLOUD.BURN_UP_STEP:
                self.ap_50_offline_teacher[self.iter] = self.ap_50_offline_teacher[self.iter-self.cfg.TEST.EVAL_PERIOD]
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(self.cfg, self.offline_teacher)
            self.logger.info('test teacher complete! For iter: {}'.format(self.iter))
            self.ap_50_offline_teacher[self.iter] = self._last_eval_results_teacher['bbox']['AP50']
            return self._last_eval_results_teacher

        ret.append(MyEvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_student))
        teacher_test_start = self.cfg.CLOUD.BURN_UP_STEP if self.cfg.CLOUD.EMA_KEEP_RATE_OFFLINE != 1.0 else 10000000000
        ret.append(MyEvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_teacher, teacher_test_start))

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                MyPeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, online_results=self.model_CLOUD,
                    ap_50_student=self.ap_50_student, ap_50_offline_teacher=self.ap_50_offline_teacher
                )
            )

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            period = 20
            ret.append(hooks.PeriodicWriter(self.build_writers(window_size=period), period=period))
        return ret
    
    def match_dual_teacher(self, online_result,offline_result,tag,device):
        """
        In order to simplify the notation, we use A, B, and C to represent consistent, inconsistent, and private, respectively.
        """
        with torch.no_grad():
            if len(online_result[tag])==0 and len(offline_result)==0:
                common_instances_online = online_result[tag]
                common_instances_offline = offline_result
                offline_only_instances = offline_result
                online_only_instances = online_result[tag]
            else:
                if len(online_result[tag])==0:
                    # only consistent and private 
                    fg_mask = offline_result.scores > 0.8
                    common_instances_online = offline_result[fg_mask]
                    common_instances_offline = offline_result[fg_mask]
                    offline_only_instances = offline_result[~fg_mask]
                    online_only_instances = online_result[tag]
                elif len(offline_result)==0:
                    # only consistent, no private
                    common_instances_online = online_result[tag]
                    common_instances_offline = online_result[tag]
                    offline_only_instances = offline_result
                    online_only_instances = offline_result
                else:
                    no_duplicate_result, duplicate_result = delete_duplicate_boxes(offline_result, return_split=True)
                    iou_matrix = retry_if_cuda_oom(pairwise_iou)(online_result[tag].gt_boxes, no_duplicate_result.gt_boxes) 
                    match_mask = iou_matrix >= self.cfg.CLOUD.MATCHER.IOU_THRESHOLDS
                    match_inds = match_mask.nonzero()
                    common_instances_online = [online_result[tag][match_inds[:,0]]]
                    common_instances_offline = [no_duplicate_result[match_inds[:,1]]]
                    offline_only_instances = [no_duplicate_result[torch.LongTensor(list(set([i for i in range(len(no_duplicate_result))]) - set(match_inds[:,1].tolist()))).to(device=match_inds.device)]]
                    common_instances_online_list = match_inds[:,0].tolist()
                    # Handling duplicate boxes in offline predictions
                    for r in duplicate_result:
                        iou_matrix = retry_if_cuda_oom(pairwise_iou)(online_result[tag].gt_boxes, r.gt_boxes) 
                        match_mask = iou_matrix >= self.cfg.CLOUD.MATCHER.IOU_THRESHOLDS
                        match_inds = match_mask.nonzero()
                        if match_inds.size(0) != 0:
                            # Here, the first matched box is selected.
                            same_label_index = r.gt_classes == online_result[tag].gt_classes[match_inds[0,0]]
                            common_instances_online.append(online_result[tag][match_inds[0,0].item()])
                            common_instances_online_list.append(match_inds[0,0].item())
                            if same_label_index.sum()>=1:
                                # this box will be A box in the following code
                                common_instances_offline.append(r[same_label_index])
                            else:
                                common_instances_offline.append(r[random.randint(0,len(r)-1)]) # this box will be B box in the following code
                        else:
                            offline_only_instances.append(r[random.randint(0,len(r)-1)]) # If multiple offline boxes do not match any online box, we simply select one arbitrarily
                    common_instances_offline = Instances.cat(common_instances_offline)
                    common_instances_online = Instances.cat(common_instances_online)
                    common_instances_offline, common_instances_online = online_boxes_merging(online_result[tag], common_instances_offline, common_instances_online)
                    online_only_instances = online_result[tag][torch.LongTensor(list(set([i for i in range(len(online_result[tag]))]) - set(common_instances_online_list))).to(device=match_inds.device)]

            offline_only_instances = [offline_only_instances] if type(offline_only_instances)!= list else offline_only_instances

            c_instances = Instances.cat(offline_only_instances + [online_only_instances]).to(device)
            c_instances.gt_scores = c_instances.scores
            c_instances.gt_probs = c_instances.probs
            c_instances.remove('scores')
            c_instances.remove('probs')
            
            if tag=='RCNN':
                same_label_index = common_instances_offline.gt_classes==common_instances_online.gt_classes
                a_instances = common_instances_offline[same_label_index]
                a_instances.gt_scores_online = common_instances_online[same_label_index].scores
                a_instances.gt_scores_offline = common_instances_offline[same_label_index].scores
                a_instances.remove('scores')
                a_instances.gt_probs_online = common_instances_online[same_label_index].probs
                a_instances.gt_probs_offline = common_instances_offline[same_label_index].probs
                a_instances.remove('probs')
                a_instances.gt_boxes.tensor = self.merge_boxes(
                    common_instances_online[same_label_index].gt_boxes.tensor,
                    common_instances_offline[same_label_index].gt_boxes.tensor,
                    a_instances.gt_scores_online,
                    a_instances.gt_scores_offline)
                a_instances = delete_duplicate_boxes(a_instances)

                b_instances = common_instances_offline[~same_label_index]
                b_instances.gt_classes_offline = b_instances.gt_classes
                b_instances.gt_classes_online = common_instances_online[~same_label_index].gt_classes
                b_instances.remove('gt_classes')
                b_instances.gt_scores_online = common_instances_online[~same_label_index].scores
                b_instances.gt_scores_offline = common_instances_offline[~same_label_index].scores
                b_instances.remove('scores')
                b_instances.gt_probs_online = common_instances_online[~same_label_index].probs
                b_instances.gt_probs_offline = common_instances_offline[~same_label_index].probs
                b_instances.remove('probs')
                b_instances.gt_boxes.tensor = self.merge_boxes(
                    common_instances_online[~same_label_index].gt_boxes.tensor,
                    common_instances_offline[~same_label_index].gt_boxes.tensor,
                    b_instances.gt_scores_online,
                    b_instances.gt_scores_offline)
                b_instances = delete_duplicate_boxes(b_instances)

                # For handling identical boxes
                b_boxes = b_instances.gt_boxes.tensor
                a_boxes = a_instances.gt_boxes.tensor
                matrix = torch.eq(b_boxes.unsqueeze(1), a_boxes).sum(-1) == 4
                mask = matrix.sum(1)==0
                b_instances = b_instances[mask]
                
            elif tag=='RPN':
                a_instances = copy.deepcopy(common_instances_offline)
                a_instances.gt_scores_online = common_instances_online.scores
                a_instances.gt_scores_offline = common_instances_offline.scores
                a_instances.remove('scores')
                a_instances.gt_probs_online = common_instances_online.probs
                a_instances.gt_probs_offline = common_instances_offline.probs
                a_instances.remove('probs')
                a_instances.gt_boxes.tensor = self.merge_boxes(
                    common_instances_online.gt_boxes.tensor,
                    common_instances_offline.gt_boxes.tensor,
                    a_instances.gt_scores_online,
                    a_instances.gt_scores_offline)
                a_instances = delete_duplicate_boxes(a_instances)
                b_instances = None
            
        a_instances = a_instances.to(device)
        b_instances = b_instances.to(device) if b_instances!=None else None
        c_instances = c_instances.to(device)
        
        return a_instances, b_instances, c_instances

    def match_boxes(self,batched_input,offline_results):
        rcnn, rpn = [], []
        for data_dict, offline_result in zip(batched_input,offline_results):
            device = offline_result['instances'].pred_boxes.tensor.device
            online_result = self.model_CLOUD(data_dict['file_name'])
            # set filp as 'no'
            offline_result = self.process(offline_result['instances'].to('cpu'), (data_dict['height'],data_dict['width']), data_dict['image'].size()[1:], 'no')
            random_flip = data_dict['random_flip']
            assert online_result['height']==data_dict['height']
            assert online_result['width']==data_dict['width']
            assert online_result['image_id'] == data_dict['image_id']
            # set thresh as None
            online_result = self.preprocess_results(online_result, data_dict['image'].size()[1:], random_flip, thresh=None)
            rcnn.append(self.match_dual_teacher(online_result, offline_result, 'RCNN', device))
            rpn.append(self.match_dual_teacher(online_result, offline_result, 'RPN', device))
        return rcnn, rpn

    def merge_boxes(self, onlinebox, offlinebox, onlinescores, offlinescores):
        # Setting WEIGHT_FOR_BOX_A to always be 1.0 or 0.5 may produce better results
        if self.WEIGHT_FOR_BOX_A != 1.0:
            return weighted_box_fusion_split(onlinebox, offlinebox, onlinescores, offlinescores)
        else:
            return onlinebox

