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
import os
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
import os
import pickle

__all__ = ["GDINO1_5_API"]

@META_ARCH_REGISTRY.register()
class GDINO1_5_API(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        classes: tuple = "",
        device,
        test_threshold: float = 0.25,
        TOKEN,
        logger
    ):
        super().__init__()
        
        from gdino import GroundingDINOAPIWrapper, visualize
        self.target_device = device
        self.test_threshold = test_threshold
        self.classes = classes
        self.captions = '.'.join(classes)
        self.TOKEN = TOKEN
        self.model = GroundingDINOAPIWrapper(self.TOKEN)
        logger.info("GDINO Model: Grounding DINO 1.5 API")
        logger.info("Input text prompt:{}".format(self.captions))

    @classmethod
    def from_config(cls, cfg):
        assert cfg.MODEL.TEACHER_CLOUD.USE_DINO_TYPE_FILTER!=None
        return {
            "classes":MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes,
            "device": cfg.MODEL.DEVICE,
            "test_threshold":cfg.MODEL.TEACHER_CLOUD.TEST_THRESHOLD,
            "TOKEN": cfg.MODEL.TEACHER_CLOUD.TOKEN,
            "logger":setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
        }
    
    @property
    def device(self):
        return self.target_device

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = [x["image"] for x in batched_inputs]
        bs = len(images)
        assert bs == 1
        image_paths = [x["file_name"] for x in batched_inputs]
        with torch.no_grad():
            image_path = image_paths[0]
            prompts = dict(image=image_path, prompt=self.captions)            

            results = self.model.inference(prompts)

            boxes = torch.FloatTensor(results['boxes']).to(self.device)
            scores = torch.FloatTensor(results['scores']).to(self.device)
            labels = [self.classes.index(t) for t in results['categorys']]
            labels = torch.LongTensor(labels).to(self.device)
            probs = torch.zeros((len(boxes), len(self.classes)+1))
            for i in range(len(boxes)):
                probs[i, :-1] = (1.0 - scores[i]) / (len(self.classes)-1)
                probs[i, labels[i]] = scores[i]
            probs = probs.to(self.device)
            assert boxes.size(0)==labels.size(0)==scores.size(0)==probs.size(0)
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
    
    def load_state_dict(self, state_dict,
                        strict: bool = True):
        pass
