# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
import torch
MERGE_REGISTRY = Registry("MERGE")
MERGE_REGISTRY.__doc__ = """
Registry for merge, which encode image for clip model

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`Backbone`.
"""


def build_merge(cfg):

    merge_name = cfg.MODEL.MERGE
    merge = MERGE_REGISTRY.get(merge_name)(cfg)
    merge.to(torch.device(cfg.MODEL.DEVICE))
    logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
    logger.info("modeled.MODEL.MERGE: " + merge_name)

    return merge
