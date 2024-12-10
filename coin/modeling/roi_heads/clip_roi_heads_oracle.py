import numpy as np
from typing import List, Optional
import torch
from torch import nn
from detectron2.modeling.matcher import Matcher
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from .fast_rcnn_oracle import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, select_foreground_proposals, ROIHeads
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from coin.modeling.text_encoder import build_text_encoder


@ROI_HEADS_REGISTRY.register()
class OpenVocabularyRes5ROIHeads_Oracle(ROIHeads):

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
        return ret

    def _shared_roi_transform(self, features, boxes, backbone_res5):
        x = self.pooler(features, boxes)
        final = backbone_res5(x)
        del x
        return final
    
    def forward(self, images, features, proposals,  res5, attnpool, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes, res5
        )

        del proposal_boxes
        if self.pooling_type=='attnpool':
            predictions = self.box_predictor(attnpool(box_features))
        elif self.pooling_type=='meanpool':
            predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
        else:
            raise NotImplementedError

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
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
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).

        # true by default
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            # 计算IOU，这个方法在rpn的部分也用到了，这里每张图片的gt都不一致，所以用遍历来实现，而rpn那里则用直接全部一起计算
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            """ 
                N proposals and M groundtruth
                matched_idxs (Tensor): a vector of length N, each is the best-matched
                    gt index in [0, M) for each proposal.
                matched_labels (Tensor): a vector of length N, the matcher's label
                    (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            """
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label. 因为在self._sample_proposals中，设置了如果没有gt的话，那么self.num_classes就是gt，也就是背景，然后背景就会参与训练

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt