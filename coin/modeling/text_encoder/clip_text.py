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
from typing import List
from detectron2.config import configurable
import torch
from torch import nn
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
import numpy as np
import os
from torch.cuda.amp import autocast
from typing import Union, List
from pkg_resources import packaging
from coin.modeling.text_encoder.build import TEXT_ENCODER_REGISTRY
import torch
from detectron2.data import MetadataCatalog
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from ..utils import _MODELS, MODIFIED_REGION_CLIP_TEMPLATES, _download, convert_weights, available_models, Transformer, LayerNorm
import copy
__all__ = ["CLIP_TEXT"]

class TEXT_ENCODER(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 prompt_info: tuple,
                 ):
        super().__init__()

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.tokenized_prompts, self.prompt_tmp_len, self.add_prompt_num = prompt_info

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def freeze_encoder(self,):
        for v in self.token_embedding.parameters():
            v.requires_grad=False
        self.positional_embedding.requires_grad=False
        for v in self.ln_final.parameters():
            v.requires_grad=False
        self.text_projection.requires_grad=False
        self.logit_scale.requires_grad=False
        for v in self.transformer.parameters():
            v.requires_grad=False
    
    @classmethod
    def build_model(cls,state_dict, prompt_info, region_clip=False):
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]  
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

        encoder = TEXT_ENCODER(
            embed_dim,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            prompt_info
        )

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]
        for k in list(state_dict.keys()):
            if 'visual' in k:
                del state_dict[k]

        if region_clip:
            # You can load the parameters of RegionCLIP. 
            # The model download address is: https://drive.google.com/drive/folders/1YTiJJ4sAqHGj-_viiUyuADOdvlHsoub9. 
            # download file: regionclip/regionclip_pretrained-cc_rn50.pth
            # official github: https://github.com/microsoft/RegionCLIP
            # After downloading, place it in the cloud_models folder.
            region_state_dict = torch.load('cloud_models/regionclip_pretrained-cc_rn50.pth')
            region_state_dict = region_state_dict['model']
            text_state_dict = {}
            for k in list(region_state_dict.keys()):
                prefix = 'lang_encoder.'
                if k[:len(prefix)]==prefix:
                    text_state_dict[k[len(prefix):]] = region_state_dict.pop(k)
            text_state_dict['logit_scale'] = state_dict['logit_scale']

        convert_weights(encoder)
        if region_clip:
            encoder.load_state_dict(text_state_dict)
            del text_state_dict
        else:
            encoder.load_state_dict(state_dict)
            del state_dict
        encoder.eval()
        encoder.load_embedding(transformer_width)
        return encoder
    
    def load_embedding(self,transformer_width):
        with torch.no_grad():
            embedding = self.token_embedding(self.tokenized_prompts).type(self.dtype)
        # the template is: a photo of a X X X X [cls].
        self.sos = nn.Parameter(embedding[0, :1, :], requires_grad=False) # SOS 
        self.embedding_tmp = nn.Parameter(embedding[ 0, 1 :1 + self.prompt_tmp_len, :].type(torch.float32), requires_grad=True) # a photo of a
        self.register_buffer("embedding_class", embedding[:, 1 + self.prompt_tmp_len + self.add_prompt_num : 2 + self.prompt_tmp_len + self.add_prompt_num, :])  # cls
        self.eos = nn.Parameter(embedding[0, 2 + self.prompt_tmp_len + self.add_prompt_num :, :], requires_grad=False) # EOS 

        add_in_embedding_vectors = torch.empty(self.add_prompt_num, transformer_width, dtype=torch.float32) # X X X X
        nn.init.normal_(add_in_embedding_vectors, std=0.02)
        self.add_in_embedding = nn.Parameter(add_in_embedding_vectors, requires_grad=True)

    @property
    def dtype(self):
        return self.transformer.resblocks[0].attn.in_proj_weight.dtype

    def forward(self, text, add):
        if not add:
            token = text
            x = self.token_embedding(token).type(self.dtype)  # [batch_size, n_ctx, d_model]
        else:
            add_in_embedding = self.add_in_embedding
            if add_in_embedding.dim() == 2:
                add_in_embedding = add_in_embedding.unsqueeze(0).expand(self.embedding_class.size(0), -1, -1)
            embedding_tmp = self.embedding_tmp
            if embedding_tmp.dim() == 2:
                embedding_tmp = embedding_tmp.unsqueeze(0).expand(self.embedding_class.size(0), -1, -1)
            sos = self.sos
            if sos.dim() == 2:
                sos = sos.unsqueeze(0).expand(self.embedding_class.size(0), -1, -1)
            eos = self.eos
            if eos.dim() == 2:
                eos = eos.unsqueeze(0).expand(self.embedding_class.size(0), -1, -1)
            x = torch.cat(
                [sos, 
                embedding_tmp,  # (n_cls, 1+self.prompt_tmp_len, dim)
                add_in_embedding,     # (n_cls, self.add_prompt_num, dim)
                self.embedding_class,
                eos],              # (n_cls, *, dim)
                dim=1,
            )

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if not add:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        else:
            x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        # we add norm here.
        x /= torch.norm(x, dim=-1, keepdim=True)
        return x
    
_tokenizer = _Tokenizer()

@TEXT_ENCODER_REGISTRY.register()
class CLIP_TEXT(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        type: str,
        device: str,
        prompt_tmp: str,
        classes,
        dataset_style,
        add_prompt_num: int,
        logger,
        region_clip
    ):
        super().__init__()
        assert type in _MODELS ,RuntimeError(f"Model {type} not found; available models = {available_models()}")
        self.type = type
        self.target_device = device
        self.prompt_tmp = prompt_tmp
        self.classes = classes
        self.dataset_style = dataset_style
        self.add_prompt_num = add_prompt_num
        state_dict = self.load(type)
        self.region_clip = region_clip
        self.encoder = TEXT_ENCODER.build_model(state_dict,self.get_token(prompt_tmp,classes,add_prompt_num), region_clip)
        self.freeze_encoder()
        self.encoder.to(self.target_device)

        self.load_embedding()
        logger.info("Using prompt template for CLIP_TEXT: '{}'".format(prompt_tmp))
        logger.info("Using {} prompt templates for prompt without adding.".format(len(MODIFIED_REGION_CLIP_TEMPLATES)))
        logger.info('Added prompt num: {}'.format(add_prompt_num))
        del state_dict

    @classmethod
    def from_config(cls, cfg, backgroud=False):
        classes = copy.deepcopy(MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes)
        # background is at last in faster rcnn.
        if backgroud:
            classes.append('backgroud')
        return {
            "type": cfg.MODEL.TEACHER_OFFLINE.TYPE,
            "device": cfg.MODEL.DEVICE,
            "prompt_tmp": "a photo of a {}.", # a {} takes up the entire photo.
            "classes":classes,
            "dataset_style": cfg.DATASETS.STYLE_NAME,
            "add_prompt_num": cfg.CLOUD.ADD_PROMPT_NUM,
            'logger':setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank()),
            "region_clip": cfg.MODEL.REGION_CLIP,
        }
    
    def load_embedding(self,):
        # if True:
        with autocast():
            with torch.no_grad():
                per_class_feat = []
                for cls_name in self.classes:
                    cls_strs = []
                    for t in MODIFIED_REGION_CLIP_TEMPLATES:
                        cls_strs.append(t.format(self.dataset_style, cls_name))
                    tokens = self.tokenize(cls_strs).to(self.device)
                    cls_temp_feat = self.encoder(tokens, add=False)
                    mean_cls_feat = cls_temp_feat.mean(0,keepdim=True)
                    per_class_feat.append(mean_cls_feat)
                per_class_feat = torch.cat(per_class_feat,dim=0)
        per_class_feat = per_class_feat / per_class_feat.norm(dim=1, keepdim=True)
        self.register_buffer("per_class_feat",per_class_feat)
        self.register_buffer("prototype_b_online",per_class_feat.clone())
        self.register_buffer("prototype_b_offline",per_class_feat.clone())

    def get_token(self,prompt_tmp,classes,add_prompt_num):
        for name in classes:
            assert ' ' not in name,  'only one word class name is supported.'
        assert type(prompt_tmp) == str
        prompt_tmp = prompt_tmp.replace("_", " ")
        prompt_tmp_len = len(prompt_tmp.split("{")[0][:-1].split(" "))
        # It is ok to use X here instead, because it is randomly initialized later.
        temp = prompt_tmp.split("{")[0] + " ".join(["X"]*add_prompt_num) + " {" + prompt_tmp.split("{")[1]
        prompts = [temp.format(name) for name in classes]
        tokenized_prompts = torch.cat([self.tokenize(p) for p in prompts])
        return tokenized_prompts,prompt_tmp_len,add_prompt_num
    
    def freeze_encoder(self,):
        self.encoder.freeze_encoder()
    
    def train(self, mode: bool = True):
        mode = False
        # Keep the text encoder in evaluation mode.
        self.training = mode
        for module in self.children():
            module.eval()
        return self

    @property
    def device(self):
        return self.target_device

    @property
    def num_classes(self):
        return len(self.classes)
    
    @property
    def logit_scale(self):
        return self.encoder.logit_scale

    @property
    def dtype(self):
        return self.encoder.dtype

    @property
    def prototype(self):
        return self.per_class_feat
    
    def forward(self, added):
        if not added:
            return self.per_class_feat
        return self.encoder(None, add=True)

    def load(self, backbone_name: str, download_root: str = None):
        # load model without jit mode
        url = _MODELS[backbone_name]
        model_path = _download(url, download_root or os.path.expanduser("~/.cache/clip"))
        try:
            # loading JIT archive.
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        return state_dict or model.state_dict()

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = _tokenizer.encoder["<|startoftext|>"]
        eot_token = _tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result