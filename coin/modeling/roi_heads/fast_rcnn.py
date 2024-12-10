# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Tuple
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, nonzero_tuple
from coin.modeling.utils import cat
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
import copy
from coin.utils.losses import MILCrossEntropy, MILFocalLoss
__all__ = ["fast_rcnn_inference", "FastRCNNOutputLayers"]

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """

    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    probs = copy.deepcopy(scores)

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()

    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    probs = probs[filter_inds[:,0]]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    probs = probs[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.probs = probs
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]
    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
            
class FastRCNNOutputLayers(nn.Module):
    """
    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        text_encoder,
        pooling_type,
        box2box_transform,
        text_dim: int,
        classes_weight,
        loss_type,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Dict[str, float] = 1.0,
        batch_size_per_image,
        cls_b_thresh,
        dataset,
        prototype_update_rate
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.text_dim = text_dim
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        if pooling_type=='attnpool' or pooling_type=='meanpool':
            # Mapping from features that can only be classified to features that can be classified and located
            self.trans = nn.Sequential(
                nn.Linear(input_size, input_size//2),
                nn.LeakyReLU(),
                nn.Linear(input_size//2, input_size//2),
                nn.LeakyReLU(),
                nn.Linear(input_size//2, input_size),
            )
        else: self.trans = None
        self.cls_score = nn.Linear(input_size, text_dim) # to text vector dim
        self.logit_scale = nn.Parameter(torch.FloatTensor([0.01]),requires_grad=False)

        self.num_classes = text_encoder.num_classes-1
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else self.num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        if self.trans:
            self.trans.apply(weight_init)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        self.loss_weight = loss_weight
        self.text_encoder = text_encoder
        self.MILCrossEntropy = MILCrossEntropy()
        self.MILFocalLoss = MILFocalLoss(self.num_classes+1, torch.FloatTensor(classes_weight))
        self.L1_loss = nn.L1Loss(reduction='mean')
        self.L1_loss_none = nn.L1Loss(reduction='none')
        self.KL_loss = nn.KLDivLoss(reduction='mean')
        self.loss_type = loss_type
        self.classes_weight = classes_weight
        self.batch_size_per_image = batch_size_per_image
        self.cls_b_thresh = cls_b_thresh
        self.dataset = dataset
        self.prototype_update_rate = prototype_update_rate

    @classmethod
    def from_config(cls, cfg, text_encoder, input_shape):
        dims = {"RN50":1024, "RN101":512, "RN50x4":640, "RN50x16":768}
        text_dim = dims[cfg.MODEL.TEACHER_OFFLINE.TYPE]
        cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        loss_weight = {
            'loss_box_reg': cfg.CLOUD.LOSS_BOX_REG_WEIGHT,
            'loss_box_reg_offline': cfg.CLOUD.LOSS_BOX_REG_OFFLINE_WEIGHT,
            'loss_box_reg_online': cfg.CLOUD.LOSS_BOX_REG_ONLINE_WEIGHT, 
            'loss_cls': cfg.CLOUD.LOSS_CLS_WEIGHT,
            'loss_text_align': cfg.CLOUD.LOSS_TEXT_ALIGN_WEIGHT,
            'loss_distillation': cfg.CLOUD.LOSS_DISTILLATION_WEIGHT,
            'loss_cls_b': cfg.CLOUD.LOSS_CLS_B_WEIGHT,
        }
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "text_dim"           : text_dim,
            "cls_agnostic_bbox_reg" : cls_agnostic_bbox_reg,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : loss_weight,
            "text_encoder"          : text_encoder,
            "pooling_type"          : cfg.MODEL.ROI_HEADS.POOLING_TYPE,
            "classes_weight"        : cfg.CLOUD.CLASSES_WEIGHT,
            "loss_type"             : cfg.CLOUD.LOSS_TYPE,
            "batch_size_per_image"  : cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "cls_b_thresh"          : cfg.CLOUD.CLS_B_THRESH,
            "dataset"               : cfg.DATASETS.TRAIN_UNLABEL,
            "prototype_update_rate" : cfg.CLOUD.PROTOTYPE_UPDATE_WEIGHT
            # fmt: on
        }

    def forward(self, x, branch, return_feats = True):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        if self.trans:
            x = self.trans(x)
        class_feats = self.cls_score(x)
        scores = self.do_classify(class_feats, branch)
        proposal_deltas = self.bbox_pred(x)
        if return_feats and self.training and branch!='test':
            return scores, proposal_deltas, class_feats
        return scores, proposal_deltas
    
    def do_classify(self,image_features, branch):
        text_features = self.text_encoder(added=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        scores = (image_features @ text_features.t()) / self.logit_scale
        if self.training and branch!='test':
            clip_text = self.text_encoder(added=False).detach()
            clip_text = clip_text / clip_text.norm(dim=1, keepdim=True)

            loss_text_align = self.L1_loss(text_features, clip_text)
            return scores, loss_text_align
        else: return scores

    def losses(self, predictions, proposals, merge_module, branch, update_prototype=False):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        if branch=='pre_train':
            (scores, loss_text_align), proposal_deltas, img_features = predictions
            losses = {'loss_text_align': loss_text_align}
            bs = len(proposals)
            per_image_proposals_num = [sum([len(x[0]),len(x[1])]) for x in proposals]
            scores = torch.split(scores,per_image_proposals_num,dim=0)
            proposal_deltas = torch.split(proposal_deltas,per_image_proposals_num,dim=0)
            
            # ========= Classification Loss ==================
            per_image_fg_nums = [len(proposals[i][0]) for i in range(bs)]
            scores_fg = cat([scores[i][:per_image_fg_nums[i],:] for i in range(bs)],dim=0)

            gt_classes_fg_offline = cat([proposals[i][0].gt_classes_offline for i in range(bs)],dim=0)
            gt_probs_fg_offline = cat([proposals[i][0].gt_probs_offline for i in range(bs)],dim=0)
            assert scores_fg.size(0)==gt_classes_fg_offline.size(0)==gt_probs_fg_offline.size(0)

            if sum(per_image_fg_nums)!=0:
                scores_bg = cat([scores[i][-len(proposals[i][1]):,:] for i in range(bs)],dim=0)
                gt_classes_bg = cat([proposals[i][1].gt_classes for i in range(bs)],dim=0)
                assert scores_bg.size(0)==gt_classes_bg.size(0)
            else:
                scores_bg = None
                gt_classes_bg = None

            losses.update({
                "loss_cls": ( self.class_cross_loss(cat([scores_fg,scores_bg], dim=0), \
                    gt_classes_fg_offline, gt_classes_bg, gt_probs_fg_offline) if self.dataset!=("cliparttrain",) else \
                        self.class_cross_loss1(cat([scores_fg,scores_bg], dim=0), \
                    gt_classes_fg_offline, gt_classes_bg, gt_probs_fg_offline) ) if sum(per_image_fg_nums)!=0 \
                        else torch.zeros_like(loss_text_align)
            })

            # ========= update prototype ==================
            if update_prototype and sum(per_image_fg_nums)!=0:
                img_features = img_features / img_features.norm(dim=1, keepdim=True)
                update_rate = self.prototype_update_rate
                img_features = torch.split(img_features, per_image_proposals_num, dim=0)
                img_features_fg = cat([img_features[i][:per_image_fg_nums[i],:] for i in range(bs)],dim=0)
                img_features_bg = cat([img_features[i][-len(proposals[i][1]):,:] for i in range(bs)],dim=0)
                one_hot_fg = F.one_hot(gt_classes_fg_offline, num_classes = self.num_classes+1)
                one_hot_bg = F.one_hot(gt_classes_bg, num_classes = self.num_classes+1)
                img_features = cat([img_features_fg, img_features_bg])
                one_hot = cat([one_hot_fg, one_hot_bg]).float()
                new_prototype = self.text_encoder.prototype.data.clone().float()
                mask = one_hot.sum(0) != 0
                new_prototype[mask] = (one_hot.T @ img_features.float() / (one_hot.sum(0)).unsqueeze(1))[mask]
                self.text_encoder.prototype.data = self.text_encoder.prototype.data * update_rate + (1-update_rate) * new_prototype

            _log_classification_stats(cat([scores_fg, scores_bg], dim=0), cat([gt_classes_fg_offline, gt_classes_bg], dim=0))

            # ========= Regression Loss ==================
            gt_classes_for_all_offline = cat([cat([x[0].gt_classes_offline, x[1].gt_classes],dim=0) for x in proposals],dim=0)
            # parse box regression outputs
            if len(proposals):
                proposal_boxes = \
                    cat([cat([x[0].proposal_boxes.tensor, x[1].proposal_boxes.tensor],dim=0) for x in proposals],dim=0)
                assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
                # If "gt_boxes" does not exist, the proposals must be all negative and
                # should not be included in regression loss computation.
                # Here we just use proposal_boxes as an arbitrary placeholder because its
                # value won't be used in self.box_reg_loss().
                gt_boxes = cat([cat([x[0].gt_boxes.tensor, x[1].proposal_boxes.tensor],dim=0) for x in proposals],dim=0)
            else:
                proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

            proposal_deltas = cat([proposal_deltas[i] for i in range(bs)],dim=0)
            assert proposal_deltas.size(0)==proposal_boxes.size(0)==gt_boxes.size(0)==gt_classes_for_all_offline.size(0)
            losses['loss_box_reg'] = self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes_for_all_offline)

            for v in losses.values():
                assert not torch.any(torch.isnan(v))

            return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

        elif branch=='step_one' or branch=='step_two':
            ((scores, loss_text_align), proposal_deltas, img_features), ((scores_c, loss_text_align_c), proposal_deltas_c) = predictions
            proposals, gt_instances_c = proposals
            del loss_text_align_c, proposal_deltas_c
            
            bs = len(proposals)
            per_image_proposals_num = [sum([len(x[0]),len(x[1]),len(x[2])]) for x in proposals]
            scores = torch.split(scores,per_image_proposals_num,dim=0)
            a_proposals_nums = [len(proposals[i][0]) for i in range(bs)]
            b_proposals_nums = [len(proposals[i][1]) for i in range(bs)]
            bg_proposals_nums = [len(proposals[i][2]) for i in range(bs)]
            calc_bg = sum(bg_proposals_nums)!=0
            losses = {'loss_text_align': loss_text_align}

            # ========= A boxes consistent ==================
            scores_a = cat([scores[i][:a_proposals_nums[i],:] for i in range(bs)],dim=0)
            gt_classes_a = cat([proposals[i][0].gt_classes for i in range(bs)],dim=0)

            scores_bg = cat([scores[i][a_proposals_nums[i]+b_proposals_nums[i]:,:] for i in range(bs)],dim=0)
            gt_classes_bg = cat([proposals[i][2].gt_classes for i in range(bs)],dim=0)

            if self.loss_type=='MILCrossEntropy':
                weights = cat([torch.ones_like(gt_classes_a), torch.ones_like(gt_classes_bg)*self.classes_weight[-1]], dim=0)
                losses["loss_cls"] = self.MILCrossEntropy(
                    cat([scores_a,scores_bg],dim=0),
                    cat([F.one_hot(gt_classes_a, num_classes=self.num_classes+1), \
                            F.one_hot(gt_classes_bg, num_classes=self.num_classes+1)],dim=0),\
                    weights=weights, avg_positives=True, reduction='mean')
            elif self.loss_type=='MILFocalLoss':
                losses["loss_cls"] = self.MILFocalLoss(
                    cat([scores_a,scores_bg],dim=0),
                    cat([F.one_hot(gt_classes_a, num_classes=self.num_classes+1), \
                            F.one_hot(gt_classes_bg, num_classes=self.num_classes+1)],dim=0),\
                        avg_positives=True)
            else:
                raise NotImplementedError
            
            # ========= update prototype ==================
            if update_prototype:
                img_features = img_features / img_features.norm(dim=1, keepdim=True)
                update_rate = self.prototype_update_rate
                img_features = torch.split(img_features, per_image_proposals_num, dim=0)
                # --------a and bg -------------
                img_features_a = cat([img_features[i][:a_proposals_nums[i],:] for i in range(bs)],dim=0)
                img_features_bg = cat([img_features[i][a_proposals_nums[i]+b_proposals_nums[i]:,:] for i in range(bs)],dim=0)
                one_hot_a = F.one_hot(gt_classes_a, num_classes = self.num_classes+1)
                one_hot_bg = F.one_hot(gt_classes_bg, num_classes = self.num_classes+1)
                img_features_a_bg = cat([img_features_a, img_features_bg])
                one_hot_a_bg = cat([one_hot_a, one_hot_bg]).float()
                new_prototype = self.text_encoder.prototype.data.clone().float()
                mask = one_hot_a_bg.sum(0) != 0
                new_prototype[mask] = (one_hot_a_bg.T @ img_features_a_bg.float() / (one_hot_a_bg.sum(0)).unsqueeze(1))[mask]
                self.text_encoder.prototype.data = self.text_encoder.prototype.data * update_rate + (1-update_rate) * new_prototype
                
                if sum(b_proposals_nums)!=0:
                # -------- online b -------------
                    img_features_b = cat([img_features[i][a_proposals_nums[i]:a_proposals_nums[i]+b_proposals_nums[i],:] for i in range(bs)],dim=0)
                    gt_probs_b_online = cat([proposals[i][1].gt_probs_online for i in range(bs)],dim=0)
                    gt_classes_b_online = cat([proposals[i][1].gt_classes_online for i in range(bs)],dim=0)
                    assert img_features_b.size(0)==gt_probs_b_online.size(0)==gt_classes_b_online.size(0)
                    one_hot_online_b = F.one_hot(gt_classes_b_online, num_classes = self.num_classes+1)
                    img_features_ab_bg_online = cat([img_features_a, img_features_b, img_features_bg])
                    one_hot_ab_bg_online = cat([one_hot_a, one_hot_online_b, one_hot_bg]).float()
                    new_prototype = self.text_encoder.prototype_b_online.data.clone().float()
                    mask = one_hot_ab_bg_online.sum(0) != 0
                    new_prototype[mask] = (one_hot_ab_bg_online.T @ img_features_ab_bg_online.float() / (one_hot_ab_bg_online.sum(0)).unsqueeze(1))[mask]
                    self.text_encoder.prototype_b_online.data = self.text_encoder.prototype_b_online.data * update_rate + (1-update_rate) * new_prototype
                    # -------- offline b -------------
                    gt_probs_b_offline = cat([proposals[i][1].gt_probs_offline for i in range(bs)],dim=0)
                    gt_classes_b_offline = cat([proposals[i][1].gt_classes_offline for i in range(bs)],dim=0)
                    assert img_features_b.size(0)==gt_probs_b_offline.size(0)==gt_classes_b_offline.size(0)
                    one_hot_offline_b = F.one_hot(gt_classes_b_offline, num_classes = self.num_classes+1)
                    img_features_ab_bg_offline = cat([img_features_a, img_features_b, img_features_bg])
                    one_hot_ab_bg_offline = cat([one_hot_a, one_hot_offline_b, one_hot_bg]).float()
                    new_prototype = self.text_encoder.prototype_b_offline.data.clone().float()
                    mask = one_hot_ab_bg_offline.sum(0) != 0
                    new_prototype[mask] = (one_hot_ab_bg_offline.T @ img_features_ab_bg_offline.float() / (one_hot_ab_bg_offline.sum(0)).unsqueeze(1))[mask]
                    self.text_encoder.prototype_b_offline.data = self.text_encoder.prototype_b_offline.data * update_rate + (1-update_rate) * new_prototype

                # ========= B boxes inconsistent==================
                if sum(b_proposals_nums)!=0:
                    
                    gt_probs_online_a = cat([proposals[i][0].gt_probs_online for i in range(bs)],dim=0)
                    gt_probs_offline_a = cat([proposals[i][0].gt_probs_offline for i in range(bs)],dim=0)
                    merge_probs = merge_module(img_features_a.detach(), self.text_encoder.prototype_b_offline.data, self.text_encoder.prototype_b_online.data, 
                                    gt_probs_offline_a, gt_probs_online_a)
                    losses['loss_merge_base'] = self.KL_loss(torch.log(merge_probs+1e-7), one_hot_a.float().detach())
                    merge_probs = merge_module(img_features_b.detach(), self.text_encoder.prototype_b_offline.data, self.text_encoder.prototype_b_online.data, 
                                    gt_probs_b_offline, gt_probs_b_online)
                    scores_b = cat([scores[i][a_proposals_nums[i]:a_proposals_nums[i]+b_proposals_nums[i],:] for i in range(bs)],dim=0)
                    p_b = F.softmax(scores_b, dim=1)
                    p_a = F.softmax(scores_a, dim=1)
                    losses["loss_merge_b"] = F.mse_loss(p_b, merge_probs)
                    losses["loss_merge_a"] = F.mse_loss(p_a, one_hot_a.float().detach())

                    if branch=='step_two':
                        mask = (merge_probs.max(1)[0] >= self.cls_b_thresh).detach()
                        if mask.sum() > 0:
                            losses["loss_cls_b"] = self.KL_loss(torch.log(p_b[mask]+1e-7), merge_probs[mask].detach())

            # ========= C boxes private ==================
            if scores_c is not None: # None means no C boxes
                p = F.softmax(scores_c, dim=1)
                q = cat([gt_instances_c[i].gt_probs for i in range(bs)], dim=0)
                objectness_loss = self.KL_loss(torch.log(p+1e-7), q)
                losses['loss_distillation'] = objectness_loss

            # ========= regression ==================
            gt_classes_for_all_online = cat([cat([x[0].gt_classes, x[1].gt_classes_online, x[2].gt_classes], dim=0) for x in proposals], dim=0)
            gt_classes_for_all_offline = cat([cat([x[0].gt_classes, x[1].gt_classes_offline, x[2].gt_classes], dim=0) for x in proposals], dim=0)
            # parse box regression outputs
            if len(proposals):
                proposal_boxes = cat([
                    cat([x[0].proposal_boxes.tensor, x[1].proposal_boxes.tensor, x[2].proposal_boxes.tensor], dim=0) \
                    for x in proposals],dim=0)
                assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
                # If "gt_boxes" does not exist, the proposals must be all negative and
                # should not be included in regression loss computation.
                # Here we just use proposal_boxes as an arbitrary placeholder because its
                # value won't be used in self.box_reg_loss().
                gt_boxes = cat([
                    cat([x[0].gt_boxes.tensor, x[1].gt_boxes.tensor, x[2].proposal_boxes.tensor], dim=0) \
                    for x in proposals],dim=0)
            else:
                proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)
            if self.cls_agnostic_bbox_reg:
                losses['loss_box_reg'] = self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes_for_all_online, normalizer = None if calc_bg else self.batch_size_per_image * bs)
            else:
                losses['loss_box_reg_online'] = self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes_for_all_online, normalizer =  None if calc_bg else self.batch_size_per_image * bs)
                losses['loss_box_reg_offline'] = self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes_for_all_offline, normalizer = None if calc_bg else self.batch_size_per_image * bs)

            return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
    
    def class_cross_loss(self,scores, classes_offline, classes_bg, probs_offline):
        one_hot_offline = F.one_hot(classes_offline, num_classes=self.num_classes+1)
        one_hot_fg = one_hot_offline
        one_hot_bg = F.one_hot(classes_bg, num_classes=self.num_classes+1)
        one_hot = cat([one_hot_fg,one_hot_bg],dim=0)
        if self.loss_type=='MILCrossEntropy':
            weights = cat([torch.ones_like(classes_offline), torch.ones_like(classes_bg)*self.classes_weight[-1]], dim=0)
            contrastive_loss = self.MILCrossEntropy(scores, one_hot, weights=weights, avg_positives=True)
        elif self.loss_type=='MILFocalLoss':
            contrastive_loss = self.MILFocalLoss(scores, one_hot, avg_positives=True)
        else:
            raise NotImplementedError
        return contrastive_loss

    def class_cross_loss1(self,scores, classes_offline, classes_bg, probs_offline):
        one_hot_offline = F.one_hot(classes_offline, num_classes=self.num_classes+1) * probs_offline.max(1)[0].unsqueeze(1)
        one_hot_fg = one_hot_offline
        one_hot_bg = F.one_hot(classes_bg, num_classes=self.num_classes+1)
        one_hot = cat([one_hot_fg,one_hot_bg],dim=0)
        if self.loss_type=='MILCrossEntropy':
            weights = cat([torch.ones_like(classes_offline), torch.ones_like(classes_bg)*self.classes_weight[-1]], dim=0)
            contrastive_loss = self.MILCrossEntropy(scores, one_hot, weights=weights, avg_positives=False)
        elif self.loss_type=='MILFocalLoss':
            contrastive_loss = self.MILFocalLoss(scores, one_hot, avg_positives=True)
        else:
            raise NotImplementedError
        return contrastive_loss

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, normalizer=None):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        if normalizer is not None:
            return loss_box_reg / normalizer
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty
    
    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        scores, proposal_deltas = predictions
        predictions = (scores, proposal_deltas)
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh, # 0.05  A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
            self.test_nms_thresh, # 0.5
            self.test_topk_per_image, # 100
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)

        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)
