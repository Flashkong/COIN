import hashlib
import os
import urllib
import warnings
from typing import List
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from detectron2.modeling.backbone import Backbone
from detectron2.layers.batch_norm import FrozenBatchNorm2d
from detectron2.layers import ShapeSpec

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    tensors = [tensor for tensor in tensors if tensor is not None]
    return torch.cat(tensors, dim)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm_type='default'):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn1 = FrozenBatchNorm2d(planes) # nn.BatchNorm2d(planes)
        elif norm_type == 'SyncBN':
            self.bn1 = nn.SyncBatchNorm(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn2 = FrozenBatchNorm2d(planes) # nn.BatchNorm2d(planes)
        elif norm_type == 'SyncBN':
            self.bn2 = nn.SyncBatchNorm(planes)
        else:
            self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn3 = FrozenBatchNorm2d(planes * self.expansion) # nn.BatchNorm2d(planes * self.expansion)
        elif norm_type == 'SyncBN':
            self.bn3 = nn.SyncBatchNorm(planes * self.expansion)
        else:
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            if norm_type == 'FronzenBN':
                this_norm = FrozenBatchNorm2d(planes * self.expansion) #("1", nn.BatchNorm2d(planes * self.expansion))
            elif norm_type == 'SyncBN':
                this_norm = nn.SyncBatchNorm(planes * self.expansion)
            else:
                this_norm = nn.BatchNorm2d(planes * self.expansion)
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", this_norm), #("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(Backbone):
    """
    modified from from region clip 
    
    Extended from CLIP implementation. It contains following changes:
    1. change all nn.BatchNorm2d() to FrozenBatchNorm2d(), due to small batch size of detection training
    2. add self._out_feature_strides according to standard ResNet
    2. modify forward() to be compatible with Detectron2
    3. add freeze() and output_shape() to be compatible with Detectron2
    4. add build_clip_resnet_backbone() to build this ModifiedResNet

    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64,
                 out_features=None, freeze_at=0, depth=None, norm_type='default'):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.norm_type = norm_type

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn1 = FrozenBatchNorm2d(width // 2)  # nn.BatchNorm2d(width // 2)
        elif norm_type == 'SyncBN':
            self.bn1 = nn.SyncBatchNorm(width // 2)
        else:
            self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn2 = FrozenBatchNorm2d(width // 2)  # nn.BatchNorm2d(width // 2)
        elif norm_type == 'SyncBN':
            self.bn2 = nn.SyncBatchNorm(width // 2)
        else:
            self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn3 = FrozenBatchNorm2d(width) # nn.BatchNorm2d(width)
        elif norm_type == 'SyncBN':
            self.bn3 = nn.SyncBatchNorm(width)
        else:
            self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        if 'res5' in out_features:  # FPN
            self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        else:  # C4, layer4 created here won't be used in backbone, but used in roi_head
            self.layer4 = self._make_layer(width * 8, layers[3], stride=2) # None

        # attention layer在外部创建
        self._out_features = out_features if out_features else []
        if depth in [50,101]: # resnet50 or resnet 101
            # FPN: ["res2", "res3", "res4", "res5"]; C4: ["res4"]
            self._out_feature_channels = {'stem': 64, 'res2': 256, 'res3': 512, 'res4': 1024, 'res5': 2048} if 'res5' in self._out_features \
                else {'stem': 64, 'res2': 256, 'res3': 512, 'res4': 1024}
            self._out_feature_strides = {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32} if 'res5' in self._out_features \
                else  {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16}  # anti-aliasing strided conv???        
        elif depth in [200]: # resnet50x4
            # FPN: ["res2", "res3", "res4", "res5"]; C4: ["res4"]
            self._out_feature_channels = {'stem': 80, 'res2': 320, 'res3': 640, 'res4': 1280, 'res5': 2560} if 'res5' in self._out_features \
                else {'stem': 80, 'res2': 320, 'res3': 640, 'res4': 1280}
            self._out_feature_strides = {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32} if 'res5' in self._out_features \
                else  {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16}  # anti-aliasing strided conv???  
        elif depth in [800]: # resnet50x16
            # FPN: ["res2", "res3", "res4", "res5"]; C4: ["res4"]
            self._out_feature_channels = {'stem': 80, 'res2': 384, 'res3': 768, 'res4': 1536, 'res5': 3072} if 'res5' in self._out_features \
                else {'stem': 80, 'res2': 384, 'res3': 768, 'res4': 1536}
            self._out_feature_strides = {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32} if 'res5' in self._out_features \
                else  {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16}  # anti-aliasing strided conv???        
        self.freeze(freeze_at)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride, norm_type=self.norm_type)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, norm_type=self.norm_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = x.type(self.conv1.weight.dtype) # det2 resnet50: [3, 800, 1216]; CLIP resnet50: [3, 224, 224]
        x = stem(x) # det2 resnet50: [64, 200, 304]; CLIP resnet50: [64, 56, 56]
        if "stem" in self._out_features:
            outputs["stem"] = x
        x = self.layer1(x) # det2 resnet50: [256, 200, 304]; CLIP resnet50: [256, 56, 56]
        outputs['res2'] = x if "res2" in self._out_features else None
        x = self.layer2(x) # det2 resnet50: [512, 100, 152]; CLIP resnet50: [512, 28, 28]
        outputs['res3'] = x if "res3" in self._out_features else None
        x = self.layer3(x) # det2 resnet50: [1024, 50, 76]; CLIP resnet50: [1024, 14, 14]
        outputs['res4'] = x if "res4" in self._out_features else None
        x = self.layer4(x)  if "res5" in self._out_features else x # det2 resnet50: [2048, 25, 38]; CLIP resnet50: [2048, 7, 7]
        outputs['res5'] = x if "res5" in self._out_features else None

        return outputs

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        def cnnblockbase_freeze(nn_module):
            """
            Make this block not trainable.
            This method sets all parameters to `requires_grad=False`,
            and convert all BatchNorm layers to FrozenBatchNorm

            Returns:
                the block itself
            """
            for p in nn_module.parameters():
                p.requires_grad = False
            return FrozenBatchNorm2d.convert_frozen_batchnorm(nn_module)
        
        if freeze_at >= 1: # stem
            self.conv1 = cnnblockbase_freeze(self.conv1)
            self.bn1 = cnnblockbase_freeze(self.bn1)
            self.conv2 = cnnblockbase_freeze(self.conv2)
            self.bn2 = cnnblockbase_freeze(self.bn2)
            self.conv3 = cnnblockbase_freeze(self.conv3)
            self.bn3 = cnnblockbase_freeze(self.bn3)
        # each stage is a torch.nn.modules.container.Sequential
        for idx, stage in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], start=2): 
            if freeze_at >= idx:
                for block in stage.children():  # each block is a Bottleneck
                    cnnblockbase_freeze(block)  
        return self

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def _download(url: str, root: str):
        os.makedirs(root, exist_ok=True)
        filename = os.path.basename(url)

        expected_sha256 = url.split("/")[-2]
        download_target = os.path.join(root, filename)

        if os.path.exists(download_target) and not os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")

        if os.path.isfile(download_target):
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
            raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

        return download_target

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


MODIFIED_REGION_CLIP_TEMPLATES = [
    '{} {}.',
    'a {} photo of a {}.',
    'a {} bad photo of a {}.',
    'a {} photo of many {}.',
    'a {} sculpture of a {}.',
    'a {} photo of the hard to see {}.',
    'a {} low resolution photo of the {}.',
    'a {} rendering of a {}.',
    '{} graffiti of a {}.',
    'a {} bad photo of the {}.',
    'a {} cropped photo of the {}.',
    'a {} tattoo of a {}.',
    'the {} embroidered {}.',
    'a {} photo of a hard to see {}.',
    'a {} bright photo of a {}.',
    'a {} photo of a clean {}.',
    'a {} photo of a dirty {}.',
    'a {} dark photo of the {}.',
    'a {} drawing of a {}.',
    'a {} photo of my {}.',
    'the {} plastic {}.',
    'a {} photo of the cool {}.',
    'a {} close-up photo of a {}.',
    'a {} black and white photo of the {}.',
    'a {} painting of the {}.',
    'a {} painting of a {}.',
    'a {} pixelated photo of the {}.',
    'a {} sculpture of the {}.',
    'a {} bright photo of the {}.',
    'a {} cropped photo of a {}.',
    'a {} plastic {}.',
    'a {} photo of the dirty {}.',
    'a {} jpeg corrupted photo of a {}.',
    'a {} blurry photo of the {}.',
    'a {} photo of the {}.',
    'a {} good photo of the {}.',
    'a {} rendering of the {}.',
    'a {} {} in a video game.',
    'a {} photo of one {}.',
    'a {} doodle of a {}.',
    'a {} close-up photo of the {}.',
    'the {} origami {}.',
    'the {} {} in a video game.',
    'a {} sketch of a {}.',
    'a {} doodle of the {}.',
    'a {} origami {}.',
    'a {} low resolution photo of a {}.',
    'the {} toy {}.',
    'a {} rendition of the {}.',
    'a {} photo of the clean {}.',
    'a {} photo of a large {}.',
    'a {} rendition of a {}.',
    'a {} photo of a nice {}.',
    'a {} photo of a weird {}.',
    'a {} blurry photo of a {}.',
    'a {} cartoon {}.',
    '{} art of a {}.',
    'a {} sketch of the {}.',
    'a {} embroidered {}.',
    'a {} pixelated photo of a {}.',
    '{} itap of the {}.',
    'a {} jpeg corrupted photo of the {}.',
    'a {} good photo of a {}.',
    'a {} plushie {}.',
    'a {} photo of the nice {}.',
    'a {} photo of the small {}.',
    'a {} photo of the weird {}.',
    'the {} cartoon {}.',
    '{} art of the {}.',
    'a {} drawing of the {}.',
    'a {} photo of the large {}.',
    'a {} black and white photo of a {}.',
    'the {} plushie {}.',
    'a {} dark photo of a {}.',
    '{} itap of a {}.',
    '{} graffiti of the {}.',
    'a {} toy {}.',
    '{} itap of my {}.',
    'a {} photo of a cool {}.',
    'a {} photo of a small {}.',
    'a {} tattoo of the {}.',
]