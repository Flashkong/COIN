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
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb
from PIL import Image
from detectron2.config import configurable
from detectron2.modeling.proposal_generator import build_proposal_generator
from typing import Dict, List, Optional
from detectron2.structures import ImageList, Instances
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.layers import batched_nms
import os
from detectron2.utils.logger import setup_logger
from detectron2.modeling.backbone import build_backbone
import copy
from torchvision.transforms import ToTensor, Normalize, Compose
from coin.modeling.roi_heads import build_roi_heads
import cv2
import numpy as np
import supervision as sv
from detectron2.utils import comm
__all__ = ["CLIP, OpenVocabularyRCNN"]

@META_ARCH_REGISTRY.register()
class CLIP(nn.Module):
    """
    image features are extracted by CLIP's visual encoder.
    boxes are from CLOUD model
    ROI head is used to extract instance features
    """
    @configurable
    def __init__(
        self,
        *,
        backbone,
        roi_heads,
        pixel_mean,
        pixel_std,
        device,
    ):
        super().__init__()
        self.backbone = backbone
        self.roi_heads = roi_heads
        self.target_device = device
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.save_path = 'clip_labels'
        os.makedirs(self.save_path,exist_ok=True)
        self.draw_result = False


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        # CLIP also predicts whether a box (from cloud) is background 
        roi_heads = build_roi_heads(cfg, backbone.output_shape(), backgroud = True, name = cfg.MODEL.ROI_HEADS.TEACHER_OFFLINE)
        return {
            "backbone": backbone,
            "roi_heads": roi_heads,
            "pixel_mean": cfg.INPUT.TEACHER_OFFLINE.PIXEL_MEAN,  # use CLIP's mean and std
            "pixel_std": cfg.INPUT.TEACHER_OFFLINE.PIXEL_STD,
            "device": cfg.MODEL.DEVICE,
        }

    @property
    def device(self):
        return self.target_device
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], pre_result):
        images = [x["image"].to(self.device) for x in batched_inputs]
        bs = len(images)
        assert bs == 1
        assert pre_result['file_name'] == batched_inputs[0]['file_name']
        assert pre_result['height'] == batched_inputs[0]['height']
        assert pre_result['width'] == batched_inputs[0]['width']
        assert pre_result['image_id'] == batched_inputs[0]['image_id']
        with torch.no_grad():
            images = self.preprocess_image(batched_inputs)
            image_features = self.backbone(images.tensor)
            output = copy.deepcopy(pre_result) 
            with autocast():
                output = self.get_clip_result(image_features,batched_inputs,output)
            self.draw(output,'RCNN',batched_inputs,os.path.join(self.save_path,batched_inputs[0]['file_name'].split("/")[-1].split('.')[0]+'_clip_rcnn.jpg'))
        return output

    def get_clip_result(self, image_features, batched_inputs, output):
        def process(name):
            instances = output[name]['instances']
            if len(instances)!=0:
                instances.proposal_boxes = self.preprocess_boxes(instances.pred_boxes,batched_inputs)
                instances = instances.to(self.device)
                assert self.backbone.attnpool is not None
                probs = self.roi_heads(image_features,[instances], self.backbone.layer4, self.backbone.attnpool)
                max_probs, labels  = probs.max(1)
                instances.remove('proposal_boxes')
                instances.remove('pred_classes')
                instances.remove('scores')
                instances.remove('probs')
                instances.set('pred_classes', labels)
                instances.set('scores', max_probs)
                instances.set('probs', probs)
                mask = labels != (probs.size(1)-1) # Filter out the background
                instances = instances[mask]
            return instances

        output['RCNN'] = {'instances': process('RCNN')}
        if 'RPN' in output:
            output['RPN'] = {'instances': process('RPN')}
        if 'RPN_AUG' in output:
            output['RPN_AUG'] = {'instances': process('RPN_AUG')}
        # no need to nms
        return output
    
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        try : pre_process = self.pre_process
        except: 
            self.pre_process = Compose([ToTensor(),Normalize(self.pixel_mean.tolist(),self.pixel_std.tolist())])
            pre_process = self.pre_process
        images = [pre_process(x["image"].permute((1,2,0)).contiguous().cpu().numpy()) for x in batched_inputs]
        images = [x.to(self.device) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def preprocess_boxes(self,boxes,batched_inputs):
        new_boxes = boxes.clone()
        img_height,img_width = batched_inputs[0]['height'],batched_inputs[0]['width']
        net_height,net_width = batched_inputs[0]['image'].size()[1:]
        new_boxes.scale(net_width/img_width,net_height/img_height)
        return new_boxes
    
    def nms(self,outputs,names='all'):
        for name, output in outputs.items():
            if names!='all' and name not in names:
                continue
            boxes = output['instances'].pred_boxes
            scores = output['instances'].scores
            labels = output['instances'].pred_classes
            keep = batched_nms(boxes.tensor, scores, labels, self.COLLECT_NMS_THRESH)
            outputs[name] = {"instances": output['instances'][keep]}
        return outputs
        
    def draw(self,outputs,name,batched_inputs,save_name):
        if not self.draw_result:
            return
        boxes=self.preprocess_boxes(outputs[name]["instances"].pred_boxes,batched_inputs)
        boxes = boxes.tensor.cpu()
        logits=outputs[name]["instances"].scores.cpu()
        phrases=outputs[name]["instances"].pred_classes.cpu().tolist()
        phrases = [str(i) for i in phrases]

        detections = sv.Detections(xyxy=boxes.numpy())
        labels = [
            f"{phrase} {logit:.2f}"
            for phrase, logit
            in zip(phrases, logits)
        ]
        box_annotator = sv.BoxAnnotator()
        img_pil = Image.fromarray(batched_inputs[0]["image"].permute((1,2,0)).contiguous().cpu().numpy())
        annotated_frame = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(save_name, annotated_frame)


@META_ARCH_REGISTRY.register()
class OpenVocabularyRCNN(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        backbone,
        proposal_generator: nn.Module,
        roi_heads,
        pixel_mean,
        pixel_std,
        device,
        vis_period: int = 0,
        input_format,
        logger
    ):
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.vis_period = vis_period
        self.target_device = device
        # If set to False, it will not be written to state_dict
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.logger = logger
        self.input_format = input_format
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"
        self.draw_gt = False


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        if cfg.MODEL.ROI_HEADS.POOLING_TYPE != 'attnpool':
            backbone.del_attnpool()
        roi_heads = build_roi_heads(cfg, backbone.output_shape(), 
                                    backgroud = True,
                                    name = cfg.MODEL.ROI_HEADS.NAME)
        return {
            "backbone": backbone,
            "roi_heads": roi_heads,
            "pixel_mean": cfg.INPUT.TEACHER_OFFLINE.PIXEL_MEAN, # use CLIP's mean and std
            "pixel_std": cfg.INPUT.TEACHER_OFFLINE.PIXEL_STD,
            "device": cfg.MODEL.DEVICE,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "proposal_generator":build_proposal_generator(cfg, backbone.output_shape()),
            "logger":setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
        }

    @property
    def device(self):
        return self.target_device
    
    def to(self, device, *args, **kwargs):
        result = super().to(device, *args, **kwargs)
        self.target_device = device
        return result
    
    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        try : pre_process = self.pre_process
        except: 
            self.pre_process = Compose([ToTensor(),Normalize(self.pixel_mean.tolist(),self.pixel_std.tolist())])
            pre_process = self.pre_process
        images = [pre_process(x["image"].permute((1,2,0)).contiguous().cpu().numpy()) for x in batched_inputs]
        images = [x.to(self.device) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def draw(self, instances, batched_inputs, save_dir, type):
        if not self.draw_gt:
            return
        os.makedirs(save_dir, exist_ok=True)
        for i, (instance, batched_input) in enumerate(zip(instances, batched_inputs)):
            boxes = instance.gt_boxes
            boxes = boxes.tensor.cpu()
            logits = instance.scores.cpu()
            phrases = instance.gt_classes.cpu().tolist()
            phrases = [str(i) for i in phrases]

            detections = sv.Detections(xyxy=boxes.numpy())
            labels = [
                f"{phrase} {logit:.2f}"
                for phrase, logit
                in zip(phrases, logits)
            ]
            box_annotator = sv.BoxAnnotator()

            img_pil = Image.fromarray(batched_input["image"].permute((1,2,0)).contiguous().cpu().numpy())
            annotated_frame = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            file_name = '.'.join((batched_input['file_name'].split('/')[-1]).split('.')[:-1]) + '_' + str(i) + '_' + type + '.' + batched_input['file_name'].split('.')[-1]
            save_name = os.path.join(save_dir, file_name)
            cv2.imwrite(save_name, annotated_frame)

    def forward(self, batched_inputs, merge_module=None, dual_teacher_instances = None, branch=None, step_two_data=None, update_prototype=False):
        
        if not self.training or branch=='test':
            return self.inference(batched_inputs, branch=branch)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if branch=='pre_train':
            rcnn_instances = [x["RCNN"].to(self.device) for x in batched_inputs]
            rpn_instances = [x["RPN"].to(self.device) for x in batched_inputs]
            # self.draw(rcnn_instances, batched_inputs, 'pre_train', 'rcnn')
            # self.draw(rpn_instances, batched_inputs, 'pre_train', 'rpn')

            if self.proposal_generator is not None:
                proposals, proposal_losses = self.proposal_generator(images, features, rpn_instances, branch=branch)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}
                
            _, detector_losses = self.roi_heads(images, features, proposals,
                    self.backbone.layer4 ,self.backbone.attnpool, branch=branch,
                    targets = rcnn_instances, update_prototype=update_prototype)
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals)

        elif branch=='step_one' or branch=='step_two':
            assert dual_teacher_instances!=None, 'dual_teacher_instances must not be None when brach is step_one and step_two'
            rcnn_instances,rpn_instances = dual_teacher_instances
            if self.proposal_generator is not None:
                proposals, proposal_losses = self.proposal_generator(images, features, rpn_instances, branch=branch)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}

            _, detector_losses = self.roi_heads(images, features, proposals,
                    self.backbone.layer4 ,self.backbone.attnpool, branch=branch, merge_module=merge_module,
                    targets = rcnn_instances, update_prototype=update_prototype)
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals)
                    
        else:
            raise NotImplementedError
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
        
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        branch = None,
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert (not self.training) or branch=='test'

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None, branch)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, \
                    self.backbone.layer4 ,self.backbone.attnpool, branch=branch, \
                    targets = None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
