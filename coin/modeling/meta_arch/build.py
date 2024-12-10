# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from detectron2.utils.logger import setup_logger
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils import comm

def build_model(cfg):
    logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
    if cfg.CLOUD.Trainer == 'CoinTrainer':
        meta_arch = cfg.MODEL.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model.to(torch.device(cfg.MODEL.DEVICE))
        logger.info("modeled.META_ARCHITECTURE. Student model: " + meta_arch)
        model_offline_teacher = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model_offline_teacher.to(torch.device(cfg.MODEL.DEVICE))
        logger.info("modeled.META_ARCHITECTURE. Teacher model: " + meta_arch)
        meta_arch = cfg.MODEL.TEACHER_CLOUD.COLLECT_ARCHITECTURE
        model_CLOUD = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model_CLOUD.to(torch.device(cfg.MODEL.DEVICE))
        logger.info("modeled.TEACHER_CLOUD TEACHER MODEL: " + meta_arch)
        return model,model_offline_teacher,model_CLOUD
    elif cfg.CLOUD.Trainer == 'GDINO':
        meta_arch = cfg.MODEL.TEACHER_CLOUD.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model.to(torch.device(cfg.MODEL.DEVICE))
        logger.info("modeled.TEACHER_CLOUD: " + meta_arch)
        return model
    elif cfg.CLOUD.Trainer == 'GLIP':
        meta_arch = cfg.MODEL.TEACHER_CLOUD.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model.to(torch.device(cfg.MODEL.DEVICE))
        logger.info("modeled.TEACHER_CLOUD: " + meta_arch)
        return model
    elif cfg.CLOUD.Trainer == 'CLIP':
        meta_arch = cfg.MODEL.TEACHER_CLOUD.COLLECT_ARCHITECTURE
        model_CLOUD = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model_CLOUD.to(torch.device(cfg.MODEL.DEVICE))
        logger.info("modeled.TEACHER_CLOUD: " + meta_arch)

        meta_arch = cfg.MODEL.TEACHER_OFFLINE.COLLECT_ARCHITECTURE
        model_CLIP = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model_CLIP.to(torch.device(cfg.MODEL.DEVICE))
        logger.info("modeled.TEACHER_OFFLINE: " + meta_arch)
        return model_CLOUD, model_CLIP
    elif cfg.CLOUD.Trainer == 'PRETRAIN':
        meta_arch = cfg.MODEL.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model.to(torch.device(cfg.MODEL.DEVICE))
        logger.info("modeled.META_ARCHITECTURE. pre-train model: " + meta_arch)
        if cfg.CLOUD.PRE_TRAIN_NAME == 'CLIP':
            meta_arch = cfg.MODEL.TEACHER_OFFLINE.COLLECT_ARCHITECTURE
            model_CLIP = META_ARCH_REGISTRY.get(meta_arch)(cfg)
            model_CLIP.to(torch.device(cfg.MODEL.DEVICE))
            logger.info("modeled.TEACHER_OFFLINE COLLECT_ARCHITECTURE: " + meta_arch)
            if cfg.MODEL.WEIGHTS=='':
                meta_arch = cfg.MODEL.TEACHER_CLOUD.COLLECT_ARCHITECTURE
                model_CLOUD = META_ARCH_REGISTRY.get(meta_arch)(cfg)
                model_CLOUD.to(torch.device(cfg.MODEL.DEVICE))
                logger.info("cfg.MODEL.WEIGHTS is None. modeled.TEACHER_CLOUD TEACHER MODEL: " + meta_arch)
                return model, model_CLIP, model_CLOUD
            else:
                logger.info(f"Will load collected results from {cfg.MODEL.WEIGHTS}. No need to build modeled.TEACHER_CLOUD TEACHER MODEL: " + meta_arch)
                return model, model_CLIP
        else : raise NotImplementedError
    elif cfg.CLOUD.Trainer == 'ORACLE':
        meta_arch = cfg.MODEL.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model.to(torch.device(cfg.MODEL.DEVICE))
        logger.info("modeled.META_ARCHITECTURE. oracle model: " + meta_arch)
        return model
    elif cfg.CLOUD.Trainer == 'ModelZoo_test':
        meta_arch = cfg.MODEL.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model.to(torch.device(cfg.MODEL.DEVICE))
        logger.info("modeled.META_ARCHITECTURE. Student model: " + meta_arch)
        return model
    else:
        raise NotImplementedError
