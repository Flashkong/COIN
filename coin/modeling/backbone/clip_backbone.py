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
from typing import Dict, List, Tuple
from detectron2.config import configurable
import torch
from torch import nn
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
import numpy as np
import os
from typing import Any, Union, List
from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from ..utils import _MODELS, _download, convert_weights, available_models, ModifiedResNet, AttentionPool2d

__all__ = ["CLIP"]


class IMAGE_ENCODER(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 out_features,
                 freeze_at,
                 depth
                 ):
        super().__init__()

        vision_heads = vision_width * 32 // 64
        self.visual = ModifiedResNet(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width,
            out_features=out_features, 
            freeze_at=freeze_at,
            depth=depth,
        )
        self.attnpool = AttentionPool2d(image_resolution // 32, vision_width * 32, vision_heads, embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.visual, ModifiedResNet):
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)
    
    @classmethod
    def build_model(cls, state_dict, out_features, depth, freeze_at, region_clip=False):
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
        embed_dim = state_dict["text_projection"].shape[1]

        encoder = IMAGE_ENCODER(
            embed_dim,image_resolution, vision_layers, vision_width,
            out_features=out_features, freeze_at=freeze_at, depth=depth,
        )

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]
        for k in list(state_dict.keys()):
            if 'transformer' in k:
                del state_dict[k]
        for k in ["positional_embedding", "text_projection", "logit_scale", "token_embedding.weight", "ln_final.weight", "ln_final.bias"]:
            if k in state_dict:
                del state_dict[k]
        visual_state_dict = {}
        for k in list(state_dict.keys()):
            if 'visual' in k:
                visual_state_dict[k[7:]] = state_dict.pop(k)
        for i in list(visual_state_dict.keys()):
            if 'layer1' in i and 'num_batches_tracked' in i:
                del visual_state_dict[i]
        del visual_state_dict['bn1.num_batches_tracked']
        del visual_state_dict['bn2.num_batches_tracked']
        del visual_state_dict['bn3.num_batches_tracked']
        atten_state_dict = {}
        for k in list(visual_state_dict.keys()):
            if 'attnpool' in k:
                atten_state_dict[k[9:]] = visual_state_dict.pop(k) 

        if region_clip:
            # You can load the parameters of RegionCLIP. 
            # The model download address is: https://drive.google.com/drive/folders/1YTiJJ4sAqHGj-_viiUyuADOdvlHsoub9. 
            # download file: regionclip/regionclip_pretrained-cc_rn50.pth
            # official github: https://github.com/microsoft/RegionCLIP
            # After downloading, place it in the cloud_models folder.
            region_state_dict = torch.load('cloud_models/regionclip_pretrained-cc_rn50.pth')
            region_state_dict = region_state_dict['model']
            visual_state_dict = {}
            for k in list(region_state_dict.keys()):
                prefix = 'backbone.'
                # prefix = 'teacher_backbone.'
                if k[:len(prefix)]==prefix:
                    visual_state_dict[k[len(prefix):]] = region_state_dict.pop(k)
            atten_state_dict = {}
            for k in list(visual_state_dict.keys()):
                if 'attnpool' in k:
                    atten_state_dict[k[9:]] = visual_state_dict.pop(k) 

        convert_weights(encoder)
        # The types of the entries in the state_dict are all in float16
        try:
            encoder.visual.load_state_dict(visual_state_dict)
        except:
            encoder.visual.load_state_dict(visual_state_dict, strict=False)
        encoder.attnpool.load_state_dict(atten_state_dict)
        del state_dict
        if region_clip:
            del region_state_dict
        del visual_state_dict
        del atten_state_dict
        torch.cuda.empty_cache()
        return encoder.eval()

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, img):
        x = self.visual(img)
        return x

class CLIP_IMAGE(Backbone):
    @configurable
    def __init__(
        self,
        *,
        type: str,
        device: str,
        out_features,
        depth,
        freeze_at,
        update_backbone,
        freeze_attnpool,
        logger,
        region_clip
    ):
        super().__init__()
        assert type in _MODELS ,RuntimeError(f"Model {type} not found; available models = {available_models()}")
        self.type = type
        self.target_device = device
        self.region_clip = region_clip
        self.encoder = self.load(type, out_features, depth, freeze_at)
        self.encoder.to(self.target_device)
        logger.info("Using CLIP backbone: {}".format(type))
        logger.info("Update CLIP's backbone is: {}".format(update_backbone))
        if region_clip:
            logger.info("Using CLIP paramters from region CLIP.")
        changes = []
        if not update_backbone: self.freeze_backbone() 
        else: changes.append('backbone')
        if freeze_attnpool: self.freeze_attn()
        else: changes.append('attnpool')
        self.change_to_float32(changes)
        self.update_backbone = update_backbone

    @classmethod
    def from_config(cls, cfg):
        return {
            "type": cfg.MODEL.TEACHER_OFFLINE.TYPE,
            "device": cfg.MODEL.DEVICE,
            "out_features": cfg.MODEL.RESNETS.OUT_FEATURES,
            "depth": cfg.MODEL.RESNETS.DEPTH,
            "freeze_at": cfg.MODEL.BACKBONE.FREEZE_AT,
            "update_backbone": cfg.CLOUD.UPDATE_BACKBONE,
            "freeze_attnpool": True if cfg.MODEL.ROI_HEADS.POOLING_TYPE!='attnpool' else False,
            "logger":setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank()),
            "region_clip": cfg.MODEL.REGION_CLIP,
        }
    
    def del_attnpool(self,):
        del self.encoder.attnpool
        self.encoder.attnpool = None
        torch.cuda.empty_cache()
    
    def change_to_float32(self, module):
        if 'backbone' in module:
            for n,v in self.encoder.visual.named_parameters():
                v.data = v.data.float()
        if 'attnpool' in module:
            for n,v in self.encoder.attnpool.named_parameters():
                v.data = v.data.float()
    
    def freeze_backbone(self,):
        for n,v in self.encoder.visual.named_parameters():
            if 'layer4' not in n:
                v.requires_grad=False
        # While keeping the backbone fixed, upgrade the type of res5.
        for n,v in self.encoder.visual.layer4.named_parameters():
                v.data = v.data.float()
    
    def freeze_attn(self,):
        for n,v in self.encoder.attnpool.named_parameters():
            v.requires_grad=False
    
    def train(self, mode: bool = True):
        if self.update_backbone:
            self.training = mode
            for module in self.children():
                module.train(mode)
        else:
            self.training = False
            self.encoder.training = False
            for module in self.encoder.children():
                module.eval()
            self.encoder.visual.layer4.train(mode)
        return self

    @property
    def device(self):
        return self.target_device
    
    @property
    def attnpool(self):
        return self.encoder.attnpool
    
    @property
    def _out_features(self):
        return self.encoder.visual._out_features
    
    @property
    def _out_feature_channels(self):
        return self.encoder.visual._out_feature_channels
    
    @property
    def _out_feature_strides(self):
        return self.encoder.visual._out_feature_strides
    
    @property
    def output_shape(self):
        return self.encoder.visual.output_shape
    
    @property
    def layer4(self):
        return self.encoder.visual.layer4
    
    @property
    def dtype(self):
        return self.encoder.dtype
    
    def forward(self, image: List[Dict[str, torch.Tensor]]):
        return self.encoder(image.type(self.dtype))

    def load(self, backbone_name: str, out_features, depth, freeze_at, download_root: str = None):
        # load model without jit mode
        url = _MODELS[backbone_name]
        model_path = _download(url, download_root or os.path.expanduser("~/.cache/clip"))
        try:
            # loading JIT archive.
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        # The model is in float16.
        model = IMAGE_ENCODER.build_model(state_dict or model.state_dict(),out_features, depth, freeze_at, self.region_clip)
        return model
        
@BACKBONE_REGISTRY.register()
def build_clip_image_backbone(cfg, input_shape):
    return CLIP_IMAGE(cfg)