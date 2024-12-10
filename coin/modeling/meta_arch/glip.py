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
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
import numpy as np
from .glip_model import GLIPModel
from maskrcnn_benchmark.config import cfg as glip_cfg
from PIL import Image

__all__ = ["GLIP"]

@META_ARCH_REGISTRY.register()
class GLIP(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        type: str = "",
        config: str = "",
        weight: str = "",
        device: str = "",
        classes: tuple = "",
        test_threshold: float = 0.2,
        logger,
        min_image_size,

    ):
        super().__init__()
        self.type = type
        self.config = config
        self.test_threshold = test_threshold
        self.target_device = device
        self.weight = weight

        glip_cfg.merge_from_file(self.config)
        # Load the weights when building the model
        glip_cfg.merge_from_list(["MODEL.WEIGHT", weight])
        glip_cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

        self.model = GLIPModel(
            glip_cfg,
            min_image_size=min_image_size,
            confidence_threshold=self.test_threshold,
            show_mask_heatmaps=False
        )

        self.classes = classes

        logger.info("GLIP Model Type: {}".format(type))


    @classmethod
    def from_config(cls, cfg):
        return {
            "type": cfg.MODEL.TEACHER_CLOUD.TYPE,
            "config": cfg.MODEL.TEACHER_CLOUD.CONFIG_PATH,
            "weight": cfg.MODEL.TEACHER_CLOUD.WEIGHT,
            "device": cfg.MODEL.DEVICE,
            "classes":MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes,
            "test_threshold":cfg.MODEL.TEACHER_CLOUD.TEST_THRESHOLD,
            "logger":setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank()),
            "min_image_size": cfg.INPUT.TEACHER_CLOUD.MIN_SIZE_TEST
        }

    @property
    def device(self):
        return self.target_device
    
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        assert len(batched_inputs) == 1

        image_paths = [x["file_name"] for x in batched_inputs]
        image_path = image_paths[0]
        # Load the image directly according to the image path
        pil_image = Image.open(image_path).convert("RGB")
        # convert to BGR format
        image = np.array(pil_image)[:, :, [2, 1, 0]]
        # GLIP has its transforms, which contains cfg.INPUT.TEACHER_CLOUD.MIN_SIZE_TEST 
        with torch.no_grad():
            predictions = self.model.run_on_local_image(image, self.classes, thresh=self.test_threshold)
            boxes = predictions.bbox
            scores = predictions.extra_fields['scores']
            labels = predictions.extra_fields['labels']-1
            probs = torch.zeros((len(boxes), len(self.classes)+1))
            for i in range(len(boxes)):
                probs[i, :-1] = (1.0 - scores[i]) / (len(self.classes)-1)
                probs[i, labels[i]] = scores[i]

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

