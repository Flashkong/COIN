from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY

def build_roi_heads(cfg, input_shape, backgroud=False, name=None):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    if name==None:
        name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape, backgroud)