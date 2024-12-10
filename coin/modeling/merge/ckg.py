from typing import Dict, List
from detectron2.config import configurable
import torch
from torch import nn
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
import numpy as np
import os
from torch.cuda.amp import autocast
from typing import Any, Union, List
from pkg_resources import packaging
from coin.modeling.merge.build import MERGE_REGISTRY
import torch
from detectron2.data import MetadataCatalog
import copy
from math import sqrt
import torch.nn.functional as F

__all__ = ["CKGNet"]
class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class CROSS_ATTENTION(nn.Module):
    def __init__(
        self,
        hidden_size,
        all_head_size,
        num_classes, 
        logger,
        head_num=8
    ):
        super().__init__()
        self.hidden_size    = hidden_size
        self.all_head_size  = all_head_size
        self.num_heads      = head_num
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, num_classes)

        # normalization
        self.norm = sqrt(all_head_size)

        # init
        self.linear_q.apply(weight_init)
        self.linear_k.apply(weight_init)
        self.linear_v.apply(weight_init)
        self.linear_output.apply(weight_init)
    
    def forward(self,x,y):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q和k的输入，y作为v的输入
        """
        q_s = self.linear_q(x).view(1, -1, self.num_heads, self.h_size).transpose(1,2)

        k_s = self.linear_k(y).view(1, -1, self.num_heads, self.h_size).transpose(1,2)

        v_s = self.linear_v(y).view(1, -1, self.num_heads, self.h_size).transpose(1,2)

        attention = CalculateAttention()(q_s,k_s,v_s)
        attention = attention.transpose(1, 2).contiguous().view(-1, self.num_heads * self.h_size)

        output = self.linear_output(attention)
        return output
    

@MERGE_REGISTRY.register()
class CKGNet(nn.Module):

    @configurable
    def __init__(
        self,
        hidden_size,
        all_head_size,
        num_classes, 
        logger,
        head_num=8
    ):
        super().__init__()
        self.cross_offline = CROSS_ATTENTION(hidden_size, all_head_size, num_classes, logger, head_num)
        self.cross_online = CROSS_ATTENTION(hidden_size, all_head_size, num_classes, logger, head_num)

    @classmethod
    def from_config(cls, cfg,):
        classes = copy.deepcopy(MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes)
        return {
            "hidden_size": cfg.MODEL.MERGE_DIM,
            "all_head_size": cfg.MODEL.MERGE_DIM,
            "num_classes": len(classes) + 1,
            'logger':setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
        }

    def forward(self, x, prototype_offline, prototype_online, probs_offline, probs_online):
        weight_offline = self.cross_offline(x, prototype_offline)
        weight_online = self.cross_online(x, prototype_online)
        final_probs = weight_offline * probs_offline + weight_online * probs_online
        final_probs = F.softmax(final_probs,dim=1)
        return final_probs