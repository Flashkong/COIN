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
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb
import torch
import torch.nn as nn
from PIL import Image
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances
from detectron2.modeling.proposal_generator import build_proposal_generator
from typing import Dict, List, Optional

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
import os
from detectron2.utils.logger import setup_logger
from detectron2.modeling.backbone import build_backbone
from torchvision.transforms import ToTensor, Normalize, Compose
from coin.modeling.roi_heads import build_roi_heads
import cv2
import numpy as np
import supervision as sv
from detectron2.utils import comm
__all__ = ["CLIP, OpenVocabularyRCNN"]

@META_ARCH_REGISTRY.register()
class OpenVocabularyOracleRCNN(nn.Module):

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
            "pixel_mean": cfg.INPUT.TEACHER_OFFLINE.PIXEL_MEAN,
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

    def forward(self, batched_inputs):
        
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, self.backbone.layer4 ,self.backbone.attnpool, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
        
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
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
        assert (not self.training)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, \
                    self.backbone.layer4 ,self.backbone.attnpool, \
                    targets = None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
