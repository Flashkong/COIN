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
from typing import Dict, List
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.config import configurable
import torch
from torch import nn
import torch.nn.functional as F
import os
import groundingdino
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
import os

__all__ = ["GDINO"]

@META_ARCH_REGISTRY.register()
class GDINO_CLASSONLY(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        type: str = "",
        device: str = "",
        classes: tuple = "",
        use_dino_type_filter: bool,
        test_threshold: float = 0.25,
        per_class_test: bool = False,
        logger
    ):
        super().__init__()
        self.type = type
        if self.type=="swinT":
            model_config_path = os.path.join(groundingdino.__path__[0],"config/GroundingDINO_SwinT_OGC.py")
        elif self.type =="swinB":
            model_config_path = os.path.join(groundingdino.__path__[0],"config/GroundingDINO_SwinB_cfg.py")
        else:
            raise NotImplementedError
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        self.test_threshold = test_threshold
        self.use_dino_type_filter = use_dino_type_filter
        self.target_device = device
        self.model = build_model(args)
        self.classes = classes
        self.per_class_test = per_class_test
        if self.per_class_test:
            per_class_prompt = {}
            for class_name in self.classes:
                captions, cat2tokenspan = build_captions_and_token_span([class_name], True)
                tokenspanlist = [cat2tokenspan[cat] for cat in [class_name]]
                assert([captions[i[0][0]:i[0][1]]  for i in tokenspanlist]==[class_name])
                per_class_prompt[class_name] = {}
                per_class_prompt[class_name]['captions'] = captions
                per_class_prompt[class_name]['token_spans'] = tokenspanlist
            self.per_class_prompt = per_class_prompt
        else:
            captions, cat2tokenspan = build_captions_and_token_span(self.classes, True)
            tokenspanlist = [cat2tokenspan[cat] for cat in self.classes]
            assert([captions[i[0][0]:i[0][1]]  for i in tokenspanlist]==self.classes)
            self.caption = captions
            self.token_spans = tokenspanlist
        logger.info("GDINO Model Type: {}".format(type))
        if per_class_test:
            logger.info("Using per class detection for GDINO")
            for k,v in self.per_class_prompt.items():
                logger.info("Input text prompt for {} is: '{}'".format(k,v['captions']))
        else:
            logger.info("Input text prompt:{}".format(self.caption))
        


    @classmethod
    def from_config(cls, cfg):
        assert cfg.MODEL.TEACHER_CLOUD.USE_DINO_TYPE_FILTER!=None
        return {
            "type": cfg.MODEL.TEACHER_CLOUD.TYPE,
            "device": cfg.MODEL.DEVICE,
            "classes":MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes,
            "test_threshold":cfg.MODEL.TEACHER_CLOUD.TEST_THRESHOLD,
            "per_class_test":cfg.MODEL.TEACHER_CLOUD.PER_CLASS_TEST,
            "use_dino_type_filter":cfg.MODEL.TEACHER_CLOUD.USE_DINO_TYPE_FILTER,
            "logger":setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
        }

    @property
    def device(self):
        return self.target_device
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = [x["image"].to(self.device) for x in batched_inputs]
        bs = len(images)
        assert bs == 1
        with torch.no_grad():
            if self.per_class_test:
                boxes,labels,scores,probs=[],[],[]
                for k,v in self.per_class_prompt.items():
                    box, label, score, prob = self.get_grounding_output(images[0],v['captions'],v['token_spans'])
                    assert box.size(0)==label.size(0)==prob.size(0)
                    if box.size(0)!=0:
                        boxes.append(box)
                        labels.append(label)
                        scores.append(score)
                        probs.append(prob)
                boxes = torch.cat(boxes,dim=0)
                labels = torch.cat(labels,dim=0)
                scores = torch.cat(scores,dim=0)
                probs = torch.cat(probs,dim=0)
            else:
                boxes, labels, scores, probs = self.get_grounding_output(images[0],self.caption,self.token_spans)
                assert boxes.size(0)==labels.size(0)==scores.size(0)==probs.size(0)
            del probs, scores
            probs = F.one_hot(labels, num_classes=len(self.classes)+1).float()
            scores = probs.max(1)[0]
            print('using one hot')

        pred_dict = {
            "boxes": boxes,
            "size": [batched_inputs[0]['height'], batched_inputs[0]['width']],  # H,W
        }
        boxes = self.resize_boxes(pred_dict)
        assert len(boxes) == len(labels), "boxes and labels must have same length"
        assert len(scores) == len(labels), "scores and labels must have same length"
        size = (batched_inputs[0]['height'], batched_inputs[0]['width'])
        output = Instances(size)
        bo = Boxes(boxes)
        bo.clip(size)
        output.pred_boxes = bo
        output.scores = scores
        output.pred_classes = labels
        output.set('probs',probs)
        outputs = {"instances": output}
        return [outputs]
    
    def resize_boxes(self,tgt):
        H, W = tgt["size"]
        boxes = tgt["boxes"]
        new_boxes = []
        # draw boxes and masks
        for box in boxes:
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H]).to(box.device)
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            new_boxes.append(box)
        if len(new_boxes)!=0:
            new_boxes = torch.stack(new_boxes)
            return new_boxes
        else:
            return boxes
    
    def get_grounding_output(self, image, caption,token_spans, with_logits=False):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        with torch.no_grad():
            outputs = self.model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            self.model.tokenizer(caption),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq

        if not self.use_dino_type_filter:
            probs = logits_for_phrases.T
            
            max_probs,_ = probs.max(1)
            filt_mask = max_probs > self.test_threshold
            probs = probs[filt_mask]
            # Since the sum of all categories is not 1.0 (may greater than 1.0), we use this method to normalize.
            # By applying log to the probabilities and then softmax, the impact on the final result is minimal.
            probs = torch.cat((probs,torch.zeros(probs.size(0),1).to(probs.device)),dim=1)
            probs = F.softmax(torch.log(probs),dim=1)
            max_probs,labels = probs.max(1)
            boxes = boxes[filt_mask]
            return boxes, labels, max_probs, probs

        probs = logits_for_phrases.T
        filt_mask = probs > self.test_threshold

        inds = filt_mask.nonzero()
        boxes = boxes[inds[:,0]]
        labels = inds[:,1]
        probs = torch.cat((probs,torch.zeros(probs.size(0),1).to(probs.device)),dim=1)
        probs = F.softmax(torch.log(probs),dim=1)
        max_probs = probs[inds[:,0], inds[:,1]]
        probs = probs[inds[:,0]]
        return boxes, labels, max_probs, probs

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        self.model.load_state_dict(state_dict,strict=strict)
