import numpy as np
from typing import List, Optional
import torch
from torch import nn
from detectron2.modeling.matcher import Matcher
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, select_foreground_proposals, ROIHeads
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from coin.modeling.text_encoder import build_text_encoder
from .fast_rcnn import FastRCNNOutputLayers

@ROI_HEADS_REGISTRY.register()
class CLIPRes5ROIHeads(nn.Module):
    """
    Using the proposal input from the outside, the classification result is obtained through res5, ROIPooler and attentionPooling.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        text_encoder,
    ):
        super().__init__()
        self.in_features = in_features
        self.pooler = pooler
        self.text_encoder = text_encoder

    @classmethod
    def from_config(cls, cfg, input_shape, backgroud):
        # fmt: off
        ret = {}
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        text_encoder = build_text_encoder(cfg, backgroud)
        ret['text_encoder'] = text_encoder
        return ret

    def _shared_roi_transform(self, features, boxes, backbone_res5):
        x = self.pooler(features, boxes)
        return backbone_res5(x)

    def forward(self, features, proposals, res5=None, attnpool=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        proposal_boxes = [x.proposal_boxes for x in proposals] # object proposals
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes, res5
        )
        if attnpool:  # att pooling
            att_feats = attnpool(box_features)
            region_feats = att_feats
        else: # mean pooling
            region_feats = box_features.mean(dim=[2, 3])
        probs = self.do_classify(region_feats)
        return probs

    def do_classify(self,image_features):
        text_features = self.text_encoder(added=False)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.text_encoder.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image.softmax(dim=-1)


@ROI_HEADS_REGISTRY.register()
class OpenVocabularyRes5ROIHeads(ROIHeads):

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        box_predictor: nn.Module,
        pooling_type,
        mask_head: Optional[nn.Module] = None,
        logger,
        BG_TRAIN,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pooling_type = pooling_type
        self.in_features = in_features
        self.pooler = pooler
        self.box_predictor = box_predictor
        self.mask_on = mask_head is not None
        if self.mask_on:
            self.mask_head = mask_head
        self.BG_TRAIN = BG_TRAIN
        logger.info("Using pooling type: {}".format(self.pooling_type))

    @classmethod
    def from_config(cls, cfg, input_shape, backgroud):
        # fmt: off
        # ret = super().from_config(cfg)
        ret = {
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),
        }
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        mask_on           = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        text_encoder = build_text_encoder(cfg, backgroud)
        ret["num_classes"]= len(text_encoder.classes)-1 if backgroud else len(text_encoder.classes)
        pooling_type = cfg.MODEL.ROI_HEADS.POOLING_TYPE
        res_dims = {'stem': 64, 'res2': 256, 'res3': 512, 'res4': 1024, 'res5': 2048}
        dims = {"RN50":1024, "RN101":512, "RN50x4":640}
        
        ret["box_predictor"] = FastRCNNOutputLayers(
            cfg, text_encoder, ShapeSpec(channels=
                                         res_dims['res5'] if pooling_type!='attnpool' else dims[cfg.MODEL.TEACHER_OFFLINE.TYPE], 
                                         height=1, width=1)
        )
        ret["pooling_type"]= pooling_type

        if mask_on:
            ret["mask_head"] = build_mask_head(
                cfg,
                ShapeSpec(channels=
                          res_dims['res5'] if pooling_type!='attnpool' else dims[cfg.MODEL.TEACHER_OFFLINE.TYPE], 
                          width=pooler_resolution, height=pooler_resolution),
            )
        ret['logger'] = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
        ret['BG_TRAIN'] = cfg.CLOUD.BG_TRAIN
        return ret

    def _shared_roi_transform(self, features, boxes, backbone_res5):
        x = self.pooler(features, boxes)
        final = backbone_res5(x)
        del x
        return final
    
    def forward(self, images, features, proposals,  res5, attnpool, branch, merge_module=None, targets=None, update_prototype=False):

        del images

        if self.training and branch!='test':
            assert targets

            if branch=='pre_train':
                proposals = self.label_and_sample_proposals(proposals, targets, branch=branch)
                proposal_boxes = [Boxes.cat([x[0].proposal_boxes,x[1].proposal_boxes]) for x in proposals]
            elif branch=='step_one' or branch=='step_two':
                gt_instances_a = [i[0] for i in targets]
                gt_instances_b = [i[1] for i in targets]
                gt_instances_c = [i[2] for i in targets]
                proposals = self.label_and_sample_proposals(proposals, [gt_instances_a,gt_instances_b,gt_instances_c], branch=branch)
                proposal_boxes = [Boxes.cat([x[0].proposal_boxes,x[1].proposal_boxes,x[2].proposal_boxes]) for x in proposals]
                del gt_instances_a, gt_instances_b

        del targets

        if branch=='test' or not self.training:
            proposal_boxes = [x.proposal_boxes for x in proposals]

        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes, res5
        )
        del proposal_boxes
        if self.pooling_type=='attnpool':
            predictions = self.box_predictor(attnpool(box_features), branch=branch)
        elif self.pooling_type=='meanpool':
            predictions = self.box_predictor(box_features.mean(dim=[2, 3]), branch=branch)
        else:
            raise NotImplementedError

        if self.training and branch!='test':
            if branch=='step_one' or branch=='step_two':
                c_boxes = [x.gt_boxes for x in gt_instances_c]
                nums = [len(x) for x in gt_instances_c]
                if sum(nums)!=0:
                    c_box_features = self._shared_roi_transform(
                        [features[f] for f in self.in_features], c_boxes, res5
                    )
                    del c_boxes
                    if self.pooling_type=='attnpool':
                        c_predictions = self.box_predictor(attnpool(c_box_features), branch=branch, return_feats = False)
                    elif self.pooling_type=='meanpool':
                        c_predictions = self.box_predictor(c_box_features.mean(dim=[2, 3]), branch=branch, return_feats = False)
                    else:
                        raise NotImplementedError
                    predictions = (predictions, c_predictions)
                    proposals = (proposals, gt_instances_c)
                else:
                    predictions = (predictions, ((None, None), None))
                    proposals = (proposals, None)
            del features
            if not self.mask_on:
                if branch=='step_one':
                    del box_features
                if branch=='step_one' or branch=='step_two':
                    if sum(nums)!=0:
                        del c_box_features
            losses = self.box_predictor.losses(predictions, proposals, merge_module, branch=branch, update_prototype=update_prototype)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            if self.mask_on:
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances
        
    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals, targets, branch
    ):
        if branch=='pre_train':
            no_thresh_boxes = [x.no_thresh_boxes for x in targets if x.has('no_thresh_boxes')]
            if self.proposal_append_gt:
                if len(no_thresh_boxes)==len(targets):
                    for x in targets:
                        x.remove('no_thresh_boxes')
                proposals = add_ground_truth_to_proposals(targets, proposals)
            proposals_with_gt = []

            num_fg_samples = []
            num_bg_samples = []
            for i in range(len(targets)):
                proposals_per_image, target = proposals[i], targets[i]
                has_gt = len(target)
                if len(no_thresh_boxes)==len(targets):
                    match_quality_matrix = pairwise_iou(
                        Boxes.cat([target.gt_boxes,no_thresh_boxes[i]]), proposals_per_image.proposal_boxes
                    )
                    matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
                    mask_no_thresh = (matched_idxs >= len(target.gt_boxes)) & (matched_idxs < len(target.gt_boxes) + len(no_thresh_boxes[i]))
                    mask_bg = (matched_labels==0)
                    mask = mask_no_thresh & (~mask_bg)
                    matched_labels[mask] = -1
                    matched_idxs[mask_no_thresh] = 0
                else:
                    match_quality_matrix = pairwise_iou(
                        target.gt_boxes, proposals_per_image.proposal_boxes
                    )
                    matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
                
                sampled_idxs, temp_labels = self._sample_proposals(
                    matched_idxs, matched_labels, target.gt_classes_offline
                )
                matched_idxs = matched_idxs[sampled_idxs]
                mask_bg = temp_labels==self.num_classes
                mask_fg = ~mask_bg
                proposals_per_image_fg = proposals_per_image[sampled_idxs[mask_fg]]
                proposals_per_image_bg = proposals_per_image[sampled_idxs[mask_bg]]
                proposals_per_image_bg.gt_classes = temp_labels[mask_bg]
                
                sampled_targets = matched_idxs
                for (trg_name, trg_value) in target.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image_fg.has(trg_name):
                        proposals_per_image_fg.set(trg_name, trg_value[sampled_targets[mask_fg]])

                num_bg_samples.append((temp_labels == self.num_classes).sum().item())
                num_fg_samples.append(temp_labels.numel() - num_bg_samples[-1])
                proposals_with_gt.append((proposals_per_image_fg, proposals_per_image_bg))
            
            # Log the number of fg/bg samples that are selected for training ROI heads
            storage = get_event_storage()
            storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
            storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

            return proposals_with_gt
        
        elif branch=='step_one' or branch=='step_two':
            a_targets, b_targets, c_targets = targets
            if self.proposal_append_gt:
                proposals = add_ground_truth_to_proposals(a_targets, proposals)
                proposals = add_ground_truth_to_proposals(b_targets, proposals)
            proposals_with_gt = []

            num_fg_samples = []
            num_bg_samples = []
            for proposals_per_image, a_target, b_target, c_target in zip(proposals, a_targets, b_targets, c_targets):
                len_a, len_b, len_c = len(a_target), len(b_target), len(c_target)
                match_quality_matrix = pairwise_iou(
                    Boxes.cat([a_target.gt_boxes,b_target.gt_boxes,c_target.gt_boxes]), proposals_per_image.proposal_boxes
                )
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
                
                mask_c = (matched_idxs >= len_a+len_b) & (matched_idxs < len_a+len_b+len_c)
                mask_bg = (matched_labels==0)
                
                mask_fg_c = mask_c & (~mask_bg)
                matched_labels[mask_fg_c] = -1
                sampled_idxs, temp_labels = self._sample_proposals(
                    matched_idxs, matched_labels, torch.cat([a_target.gt_classes,b_target.gt_classes_online,c_target.gt_classes])
                )
                matched_idxs = matched_idxs[sampled_idxs]
                mask_a = (matched_idxs >= 0) & (matched_idxs < len_a)
                mask_b = (matched_idxs >= len_a) & (matched_idxs < len_a+len_b)
                mask_bg = temp_labels==self.num_classes
                mask_a = mask_a & (~mask_bg)
                mask_b = mask_b & (~mask_bg)
                proposals_per_image_a = proposals_per_image[sampled_idxs[mask_a]]
                proposals_per_image_b = proposals_per_image[sampled_idxs[mask_b]]
                # if self.BG_TRAIN:
                proposals_per_image_bg = proposals_per_image[sampled_idxs[mask_bg]]
                proposals_per_image_bg.gt_classes = temp_labels[mask_bg]
                if not self.BG_TRAIN:
                    proposals_per_image_bg = proposals_per_image_bg[0:0]
                
                sampled_targets = matched_idxs
                for (trg_name, trg_value) in a_target.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image_a.has(trg_name):
                        proposals_per_image_a.set(trg_name, trg_value[sampled_targets[mask_a]])

                sampled_targets = matched_idxs
                for (trg_name, trg_value) in b_target.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image_b.has(trg_name):
                        proposals_per_image_b.set(trg_name, trg_value[sampled_targets[mask_b]-len_a])

                num_bg_samples.append((temp_labels == self.num_classes).sum().item())
                num_fg_samples.append(temp_labels.numel() - num_bg_samples[-1])
                proposals_with_gt.append((proposals_per_image_a,proposals_per_image_b,proposals_per_image_bg))
            
            # Log the number of fg/bg samples that are selected for training ROI heads
            storage = get_event_storage()
            storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
            storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

            return proposals_with_gt