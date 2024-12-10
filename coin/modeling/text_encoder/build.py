# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm

TEXT_ENCODER_REGISTRY = Registry("TEXT_ENCODER")
TEXT_ENCODER_REGISTRY.__doc__ = """
Registry for CLIP, which encode image for clip model

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`Backbone`.
"""


def build_text_encoder(cfg, backgroud):

    text_encoder_name = cfg.MODEL.TEACHER_OFFLINE.TEXT_ENCODER
    text_encoder = TEXT_ENCODER_REGISTRY.get(text_encoder_name)(cfg, backgroud)
    logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
    logger.info("modeled.TEACHER_OFFLINE.TEXT_ENCODER: " + text_encoder_name)

    return text_encoder
