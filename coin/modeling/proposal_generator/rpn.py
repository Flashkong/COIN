from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator import RPN
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
import copy

KL_loss = nn.KLDivLoss(reduction='mean')
@PROPOSAL_GENERATOR_REGISTRY.register()
class DualTeacherRPN(RPN):

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_distillation": cfg.CLOUD.LOSS_DISTILLATION_WEIGHT
            },
            "BG_TRAIN": cfg.CLOUD.BG_TRAIN
        })
        return ret
    
    @configurable
    def __init__(
        self,
        BG_TRAIN,
        **kwarg
    ):
        super().__init__(**kwarg)
        self.BG_TRAIN = BG_TRAIN

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
        branch = None
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]

        anchors = self.anchor_generator(features)
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training and branch!='test':
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            if branch=='pre_train':
                gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances, branch)
                losses = self.losses(
                    anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
                )
            elif branch=='step_one' or branch=='step_two':
                gt_instances_a = [i[0] for i in gt_instances]
                gt_instances_c = [i[2] for i in gt_instances]
                
                # For the c (private) boxes, it should be to directly get the corresponding features, similar to RCNN
                # However, since the logic of RPN is different from that of RCNN, we cannot get that feature, so we use ​​box matching to find boxes that matched to private boxes.
                gt_labels, gt_boxes, all_matched_idxs, distillation_labels  = self.label_and_sample_anchors(anchors, [gt_instances_a, gt_instances_c], branch)
                
                teacher_probs = [(gt_instances_c[i].gt_probs)[:,:-1].sum(1)[all_matched_idxs[i]] \
                                 if len(gt_instances_c[i])!=0 \
                                 else torch.zeros_like(all_matched_idxs[i]) \
                                    for i in range(len(gt_instances_c))]
                losses1 = self.losses(
                    anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes, calc_bg=self.BG_TRAIN
                )
                losses2 = self.losses(
                    anchors, pred_objectness_logits, distillation_labels, None, None,\
                          teacher_probs=teacher_probs, only_distillation=True
                )
                losses = {}
                losses.update(losses1)
                losses.update(losses2)
                
            else: raise NotImplementedError
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses
    
    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances], branch
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)
        if branch=='pre_train':
            gt_boxes = [x.gt_boxes for x in gt_instances]
            if gt_instances[0].has('no_thresh_boxes'):
                no_thresh_boxes = [x.no_thresh_boxes for x in gt_instances]
            else: no_thresh_boxes = None
            image_sizes = [x.image_size for x in gt_instances]
            del gt_instances

            gt_labels = []
            matched_gt_boxes = []
            for i in range(len(gt_boxes)):
                image_size_i, gt_boxes_i = image_sizes[i], gt_boxes[i]
                
                """
                image_size_i: (h, w) for the i-th image
                gt_boxes_i: ground-truth boxes for i-th image
                """
                if no_thresh_boxes is not None:
                    # These boxes will not be used as foreground. And they will not be used as background either.
                    temp_boxes = Boxes.cat([gt_boxes_i,no_thresh_boxes[i]])
                    match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(temp_boxes, anchors)
                    matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
                    mask_no_thresh = (matched_idxs >= len(gt_boxes_i)) & (matched_idxs < len(temp_boxes))
                    mask_bg = (gt_labels_i==0)
                    mask = mask_no_thresh & (~mask_bg)
                    gt_labels_i[mask] = -1
                    matched_idxs[mask_no_thresh] = 0
                    # Matching is memory-expensive and may result in CPU tensors. But the result is small
                    gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
                else:
                    match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
                    matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
                    # Matching is memory-expensive and may result in CPU tensors. But the result is small
                    gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
                del match_quality_matrix

                if self.anchor_boundary_thresh >= 0:
                    # Discard anchors that go out of the boundaries of the image
                    # NOTE: This is legacy functionality that is turned off by default in Detectron2
                    anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                    gt_labels_i[~anchors_inside_image] = -1

                # A vector of labels (-1, 0, 1) for each anchor
                gt_labels_i = self._subsample_labels(gt_labels_i)

                if len(gt_boxes_i) == 0:
                    # These values won't be used anyway since the anchor is labeled as background
                    matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                    if no_thresh_boxes is None:
                        gt_labels_i[:] = -1
                    else:
                        mask = mask_no_thresh & (mask_bg)
                        gt_labels_i[~mask] = -1
                else:
                    matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

                gt_labels.append(gt_labels_i)  # N,AHW
                matched_gt_boxes.append(matched_gt_boxes_i)
            return gt_labels, matched_gt_boxes

        elif branch=='step_two' or branch=='step_one':
            gt_boxes_a = [x.gt_boxes for x in gt_instances[0]]
            gt_boxes_c = [x.gt_boxes for x in gt_instances[1]]
            image_sizes = [x.image_size for x in gt_instances[0]]
            del gt_instances

            gt_labels = []
            matched_gt_boxes = []
            all_matched_idxs = []
            distillation_labels = []
            for image_size_i, gt_boxes_a_i, gt_boxes_c_i in zip(image_sizes, gt_boxes_a, gt_boxes_c):
                
                temp_boxes = Boxes.cat([gt_boxes_a_i,gt_boxes_c_i])
                match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(temp_boxes, anchors)
                matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
                distillation_idxs = copy.deepcopy(matched_idxs)

                mask_c = (matched_idxs >= len(gt_boxes_a_i)) & (matched_idxs < len(temp_boxes))
                mask_bg = (gt_labels_i==0)
                mask_fg_c = mask_c & (~mask_bg)
                gt_labels_i[mask_fg_c] = -1
                matched_idxs[mask_c] = 0 

                distillation_idxs = distillation_idxs - len(gt_boxes_a_i)
                distillation_idxs[~mask_fg_c] = 0  
                all_matched_idxs.append(distillation_idxs) 
                distillation_labels_i = copy.deepcopy(gt_labels_i)
                distillation_labels_i[mask_fg_c] = 1
                distillation_labels_i[~mask_fg_c] = 0
                distillation_labels.append(distillation_labels_i)

                # Matching is memory-expensive and may result in CPU tensors. But the result is small
                gt_labels_i = gt_labels_i.to(device=gt_boxes_a_i.device)
                distillation_labels_i = distillation_labels_i.to(device=gt_boxes_a_i.device)
                del match_quality_matrix

                if self.anchor_boundary_thresh >= 0:
                    # Discard anchors that go out of the boundaries of the image
                    # NOTE: This is legacy functionality that is turned off by default in Detectron2
                    anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                    gt_labels_i[~anchors_inside_image] = -1

                # A vector of labels (-1, 0, 1) for each anchor
                gt_labels_i = self._subsample_labels(gt_labels_i)

                if len(gt_boxes_a_i) == 0:
                    # These values won't be used anyway since the anchor is labeled as background
                    matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                    mask_bg_c = mask_c & (mask_bg)
                    gt_labels_i[~mask_bg_c] = -1
                else:
                    matched_gt_boxes_i = gt_boxes_a_i[matched_idxs].tensor

                gt_labels.append(gt_labels_i)  # N,AHW
                matched_gt_boxes.append(matched_gt_boxes_i)
            return gt_labels, matched_gt_boxes, all_matched_idxs, distillation_labels
        

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        teacher_probs = None,
        only_distillation = False,
        calc_bg = True
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        if not only_distillation:
            # Log the number of positive/negative anchors per-image that's used in training
            pos_mask = gt_labels == 1
            num_pos_anchors = pos_mask.sum().item()
            num_neg_anchors = (gt_labels == 0).sum().item()
            storage = get_event_storage()
            storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
            storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

            localization_loss = _dense_box_regression_loss(
                anchors,
                self.box2box_transform,
                pred_anchor_deltas,
                gt_boxes,
                pos_mask,
                box_reg_loss_type=self.box_reg_loss_type,
                smooth_l1_beta=self.smooth_l1_beta,
            )
            if calc_bg:
                valid_mask = gt_labels >= 0
            else:
                valid_mask = gt_labels >= 1
            objectness_loss = F.binary_cross_entropy_with_logits(
                cat(pred_objectness_logits, dim=1)[valid_mask],
                gt_labels[valid_mask].to(torch.float32),
                reduction="sum",
            )
            # self._subsample_labels already specifies the total number of training images in a picture, so valid_mask is always equal to self.batch_size_per_image=512
            normalizer = self.batch_size_per_image * num_images
            losses = {
                "loss_rpn_cls": objectness_loss / (normalizer if calc_bg else max(valid_mask.sum().item(), 1.0)),
                # The original Faster R-CNN paper uses a slightly different normalizer
                # for loc loss. But it doesn't matter in practice
                "loss_rpn_loc": localization_loss / normalizer,
            }
        else:
            losses = {}
            assert teacher_probs!=None, 'distillation need teacher probs'
            teacher_probs = torch.stack(teacher_probs)
            valid_mask = gt_labels > 0
            p = torch.sigmoid(cat(pred_objectness_logits, dim=1)[valid_mask])
            p = torch.cat((p.unsqueeze(1),1-p.unsqueeze(1)),dim=1)
            q = teacher_probs[valid_mask]
            q = torch.cat((q.unsqueeze(1),1-q.unsqueeze(1)),dim=1)
            objectness_loss = KL_loss(torch.log(p+1e-7),q)
            normalizer = valid_mask.float().sum()
            if normalizer!=0:
                losses.update({
                    "loss_rpn_distillation": objectness_loss,
                })

        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        for v in losses.values():
                assert not torch.any(torch.isnan(v))
        return losses