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
from typing import Dict, List
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.config import configurable
import torch
from torch import nn
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.logger import setup_logger
from detectron2.modeling.matcher import Matcher
from coin.layers.nms import mynms, merge_probs_split, weighted_box_fusion_split, merge_probs_split_bayesian
import cv2
import numpy as np
import os
import supervision as sv
from detectron2.utils import comm

__all__ = ["GDINO_PROCESSOR"]

@META_ARCH_REGISTRY.register()
class GDINO_PROCESSOR(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        model: nn.Module = None,
        RPN_SEPARATE_COLLECT: bool,
        RPN_THRESH: float,
        RCNN_THRESH: float,
        COLLECT_AUG: str,
        zoom_matcher: Matcher,
        zoom_matcher2: Matcher,
        ZOOM_MATCHER_THRESH: float,
        COLLECT_NMS_THRESH: float,
        logger
    ):
        super().__init__()
        self.target_device = model.device
        self.model = model
        self.matcher = zoom_matcher
        self.matcher2 = zoom_matcher2
        self.classes = model.classes
        self.COLLECT_AUG = COLLECT_AUG
        if not RPN_SEPARATE_COLLECT:
            RCNN_THRESH = RPN_THRESH
        self.RPN_THRESH = RPN_THRESH
        self.RCNN_THRESH = RCNN_THRESH
        self.ZOOM_MATCHER_THRESH = ZOOM_MATCHER_THRESH
        self.COLLECT_NMS_THRESH = COLLECT_NMS_THRESH
        
        assert hasattr(self.model,"test_threshold")
        self.model.test_threshold = self.RPN_THRESH
        logger.info('Using {} for collecting dino.'.format(self.COLLECT_AUG))
        self.pre_draw()

        assert self.check_support(), f'The current cloud model does not support cfg.INPUT.TEACHER_CLOUD.COLLECT_AUG:{self.COLLECT_AUG}'

    @classmethod
    def from_config(cls, cfg):
        meta_arch = cfg.MODEL.TEACHER_CLOUD.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model.to(torch.device(cfg.MODEL.DEVICE))
        return {
            "model": model,
            "RPN_SEPARATE_COLLECT": cfg.CLOUD.TEACHER_CLOUD.RPN_SEPARATE_COLLECT,
            "RPN_THRESH": cfg.CLOUD.TEACHER_CLOUD.RPN_THRESH,
            "RCNN_THRESH": cfg.CLOUD.TEACHER_CLOUD.RCNN_THRESH,
            "COLLECT_AUG": cfg.INPUT.TEACHER_CLOUD.COLLECT_AUG,
            "zoom_matcher":  Matcher(
                [cfg.CLOUD.TEACHER_CLOUD.ZOOM_MATCHER_THRESH], [0,1], allow_low_quality_matches=False
            ),
            "zoom_matcher2":  Matcher(
                [0.96], [0,1], allow_low_quality_matches=False
            ),
            "ZOOM_MATCHER_THRESH":cfg.CLOUD.TEACHER_CLOUD.ZOOM_MATCHER_THRESH,
            "COLLECT_NMS_THRESH": cfg.CLOUD.TEACHER_CLOUD.COLLECT_NMS_THRESH,
            "logger":setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
        }
    
    def pre_draw(self):
        self.save_path = 'gdino_labels'
        os.makedirs(self.save_path,exist_ok=True)
        self.draw_result = False
    
    def check_support(self):
        # GDINO supports ZOOM and AUG.
        return True

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        self.model.load_state_dict(state_dict,strict=strict)

    @property
    def device(self):
        return self.target_device
    
    def get_zomm_border(self, zoom_instances, temp_size, zoom_point, batched_inputs):
        X1, Y1 = zoom_point
        temp = zoom_instances.pred_boxes.tensor - torch.Tensor([X1+temp_size, Y1+temp_size, X1+temp_size, Y1+temp_size]).to(self.device)
        temp = Boxes(temp)
        temp.clip((batched_inputs[0]['zoom_info']['crop_size'][1]-2*temp_size,batched_inputs[0]['zoom_info']['crop_size'][0]-2*temp_size))
        temp.tensor = temp.tensor + torch.Tensor([X1+temp_size, Y1+temp_size, X1+temp_size, Y1+temp_size]).to(self.device)
        zomm_border_idxs = ((temp.tensor != zoom_instances.pred_boxes.tensor).sum(1)  >= 1).nonzero()[:,0]
        return zomm_border_idxs
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = [x["image"].to(self.device) for x in batched_inputs]
        bs = len(images)
        assert bs == 1
        outputs={}
        outputs["ORI"] = self.model(batched_inputs)[0]

        # ZOOM and AUG are not used in the paper.
        if self.COLLECT_AUG=="ZOOM":
            batched_inputs[0]["image"] = batched_inputs[0]["image_zoom"]
            outputs["ZOOM"] = self.model(batched_inputs)[0]
        elif self.COLLECT_AUG=="AUG":
            batched_inputs[0]["image"] = batched_inputs[0]["image_aug"]
            outputs["AUG"] = self.model(batched_inputs)[0]
        elif self.COLLECT_AUG=="ZOOM&AUG":
            batched_inputs[0]["image"] = batched_inputs[0]["image_zoom"]
            outputs["ZOOM"] = self.model(batched_inputs)[0]
            batched_inputs[0]["image"] = batched_inputs[0]["image_aug"]
            outputs["AUG"] = self.model(batched_inputs)[0]
        elif self.COLLECT_AUG == "":
            pass
        else: NotImplementedError
        if "ZOOM" in outputs.keys():
            zoom_info = batched_inputs[0]['zoom_info']
            W1,H1 = batched_inputs[0]['width'], batched_inputs[0]['height']
            W2,H2 = zoom_info["crop_size"]
            zoom_output = outputs["ZOOM"]["instances"]
            zoom_output.pred_boxes.scale(1/W1,1/H1)
            zoom_output.pred_boxes.scale(W2,H2)
            X1,Y1 = zoom_info["leftup"]
            zoom_output.pred_boxes.tensor += torch.Tensor([X1, Y1, X1, Y1]).to(self.device)

        outputs = self.nms(outputs,reutrn_new_dict=False)
        self.draw_gt(batched_inputs,self.save_path)
        self.draw(outputs,'ORI',batched_inputs,os.path.join(self.save_path,batched_inputs[0]['file_name'].split("/")[-1].split('.')[0]+'_ori.jpg'))
        if "ZOOM" in outputs.keys():
            self.draw(outputs,'ZOOM',batched_inputs,os.path.join(self.save_path,batched_inputs[0]['file_name'].split("/")[-1].split('.')[0]+'_zoom.jpg'))
        if "AUG" in outputs.keys():
            self.draw(outputs,'AUG',batched_inputs,os.path.join(self.save_path,batched_inputs[0]['file_name'].split("/")[-1].split('.')[0]+'_aug.jpg'))

        # merge all results
        outputs = self.post_process(outputs,batched_inputs)
        for k in ['file_name', 'image_id', 'height', 'width']:
            outputs[k] = batched_inputs[0][k]
        if 'zoom_info' in batched_inputs[0].keys():
            outputs['zoom_info'] = batched_inputs[0]['zoom_info']
        return outputs
    
    def nms(self,outputs,names='all',reutrn_new_dict=False):
        out = {} if reutrn_new_dict else outputs
        for name, output in outputs.items():
            if names!='all' and name not in names:
                continue
            boxes = output['instances'].pred_boxes
            scores = output['instances'].scores
            labels = output['instances'].pred_classes
            probs = output['instances'].probs
            _, boxes, scores, probs, labels = \
                mynms.nms(boxes.tensor, scores, probs, labels, self.COLLECT_NMS_THRESH)
            nms_instance = Instances(output['instances'].image_size)
            nms_instance.pred_boxes = Boxes(boxes)
            nms_instance.scores = scores
            nms_instance.pred_classes = labels
            nms_instance.probs = probs
            del output['instances']
            out[name] = {"instances": nms_instance}
        return out
    
    def post_process(self,outputs,batched_inputs):
        """
        The results of ZOOM and ORI are combined, and then the results of RPN and RCNN are obtained. 
        The results of AUG are then combined with RPN to form RPN_AUG.

        Please note that we DO NOT use ZOOM and AUG (we set cfg.INPUT.TEACHER_CLOUD.COLLECT_AUG='') in our paper.
        """
        if "ZOOM" in outputs.keys():
            ORI_instances = outputs['ORI']['instances']
            zoom_instances = outputs['ZOOM']['instances']
            if len(zoom_instances)!=0:
                ori_cp = ORI_instances.pred_boxes.clone()
                # First move to the upper left corner of the zoom, then clip to get the corresponding box predicted by ori
                X1,Y1 = batched_inputs[0]['zoom_info']["leftup"]
                ori_cp.tensor = ori_cp.tensor - torch.Tensor([X1, Y1, X1, Y1]).to(self.device)
                ori_cp.clip((batched_inputs[0]['zoom_info']['crop_size'][1],batched_inputs[0]['zoom_info']['crop_size'][0]))
                index = ori_cp.nonempty()
                # Index represents the number of ORI boxes within the zoom range.
                if index.sum() != 0:
                    ori_cp.tensor = ori_cp.tensor[index]
                    ori_cp.tensor = ori_cp.tensor + torch.Tensor([X1, Y1, X1, Y1]).to(self.device)
                    """
                    For the ZOOM area:
                    1. Add boxes that are present in ZOOM but not in ORI.
                    2. For boxes that match between ZOOM and ORI:
                        - If the categories match, then merge the scores and boxes according to the principles of ps.
                        - If the categories do not match, prioritize the results from ZOOM.
                    3. Remove boxes that are present in ORI but not in ZOOM.
                    """
                    # keep represents the ORI boxes outside the ZOOM range (excluding the boundary boxes), 
                    # change represents the boxes within the ZOOM range that need to be modified.
                    # border_instances represent the boxes that are half within the ZOOM range and half outside the ZOOM range
                    keep_instances = ORI_instances[~index]
                    border_mask = (ori_cp.tensor != ORI_instances[index].pred_boxes.tensor).sum(1)  >= 1
                    boarder_insances = ORI_instances[index][border_mask]
                    
                    match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(zoom_instances.pred_boxes, ori_cp[border_mask])
                    matched_idxs, match_label_ori = retry_if_cuda_oom(self.matcher2)(match_quality_matrix)
                    mask = (match_label_ori==1).nonzero()[:,0]
                    merged_probs, merged_scores = merge_probs_split_bayesian(
                            zoom_instances.probs[matched_idxs[[mask]]],
                            boarder_insances.probs[mask])
                    mask1 = boarder_insances.pred_classes[mask] == merged_probs.max(1)[1]
                    boarder_insances.scores[mask[mask1]] = merged_scores[mask1]
                    boarder_insances.probs[mask[mask1]] = merged_probs[mask1]

                    # The bounding box is not processed, which avoids problems such as matching the border boxes and ZOOM boxes
                    # (the entire car and the front of the car).
                    change_instances = ORI_instances[index][~border_mask]
                    change_instances.pred_boxes = ori_cp[~border_mask]
                    
                    match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(zoom_instances.pred_boxes, change_instances.pred_boxes) 
                    # ORI matches ZOOM
                    matched_idxs, match_label_ori = retry_if_cuda_oom(self.matcher)(match_quality_matrix)

                    # 3. Remove boxes that are present in ORI but not in ZOOM.
                    error_mask = (match_label_ori!=1)
                    change_instances = change_instances[~error_mask]

                    # 2. For boxes that match between ZOOM and ORI:
                    #    - If the categories match, then merge the scores and boxes according to the principles of ps.
                    #    - If the categories do not match, prioritize the results from ZOOM.
                    matched_idxs = matched_idxs[match_label_ori.nonzero()[:,0]]
                    if len(matched_idxs) > 0:
                        same_label_mask = zoom_instances.pred_classes[matched_idxs] == change_instances.pred_classes
                        change_instances.pred_classes = zoom_instances.pred_classes[matched_idxs]
                        change_instances.scores[~same_label_mask] = zoom_instances.scores[matched_idxs][~same_label_mask]
                        change_instances.probs[~same_label_mask] = zoom_instances.probs[matched_idxs][~same_label_mask]
                        change_instances.pred_boxes.tensor[~same_label_mask] = zoom_instances.pred_boxes.tensor[matched_idxs][~same_label_mask]

                        # Same category
                        change_instances.pred_boxes.tensor[same_label_mask] = weighted_box_fusion_split(
                            zoom_instances.pred_boxes.tensor[matched_idxs[same_label_mask]],
                            change_instances.pred_boxes.tensor[same_label_mask],
                            zoom_instances.scores[matched_idxs[same_label_mask]],
                            change_instances.scores[same_label_mask])
                        merged_probs, merged_scores = merge_probs_split(
                            zoom_instances.probs[matched_idxs[same_label_mask]],
                            change_instances.probs[same_label_mask])
                        change_instances.scores[same_label_mask] = merged_scores
                        change_instances.probs[same_label_mask] = merged_probs
                        if len(change_instances.pred_classes[same_label_mask])!=0:
                            assert (torch.argmax(merged_probs,dim=1) == change_instances.pred_classes[same_label_mask]).float().mean() == 1
                        
                    # 1. Add the boxes unique to the ZOOM range, considering these as newly predicted, but do not add the border boxes unique to the ZOOM range.
                    zomm_border_idxs = self.get_zomm_border(zoom_instances, 5, (X1,Y1), batched_inputs)
                    match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(zoom_instances[zomm_border_idxs].pred_boxes, ori_cp[border_mask])
                    a = (match_quality_matrix > 0.1).sum(1) > 0
                    added = set(torch.cat((matched_idxs, zomm_border_idxs[a]),dim=0).cpu().tolist())
                    add_index = list(set([i for i in range(len(zoom_instances))])  - added)
                    if len(add_index)>0:
                        add_instances = Instances.cat([zoom_instances[i] for i in add_index])
                        outputs['ORI']['instances'] = Instances.cat([keep_instances,change_instances,boarder_insances,add_instances])
                    else:
                        outputs['ORI']['instances'] = Instances.cat([keep_instances,change_instances,boarder_insances])
                else:
                    # If ORI does not predict any objects in the ZOOM area, add all zoom boxes.
                    outputs['ORI']['instances'] = Instances.cat([ORI_instances,zoom_instances])
            self.draw(outputs,'ORI',batched_inputs,os.path.join(self.save_path,batched_inputs[0]['file_name'].split("/")[-1].split('.')[0]+'_ori+zoom.jpg'))
            del outputs['ZOOM']
            assert(outputs['ORI']['instances'].scores == outputs['ORI']['instances'].probs.max(1)[0]).sum()==len(outputs['ORI']['instances'])
            assert(outputs['ORI']['instances'].pred_classes == outputs['ORI']['instances'].probs.max(1)[1]).sum()==len(outputs['ORI']['instances'])

        RCNN_index = outputs['ORI']['instances'].scores >= self.RCNN_THRESH
        RPN_index = outputs['ORI']['instances'].scores >= self.RPN_THRESH
        outputs['RCNN']={'instances':outputs['ORI']['instances'][RCNN_index]}
        outputs['RPN']={'instances':outputs['ORI']['instances'][RPN_index]}
        del outputs['ORI']
        
        outputs = self.nms(outputs,['RCNN','RPN'])
        self.draw(outputs,'RCNN',batched_inputs,os.path.join(self.save_path,batched_inputs[0]['file_name'].split("/")[-1].split('.')[0]+'_RCNN.jpg'))
        self.draw(outputs,'RPN',batched_inputs,os.path.join(self.save_path,batched_inputs[0]['file_name'].split("/")[-1].split('.')[0]+'_RPN.jpg'))

        if 'AUG' in outputs.keys():
            outputs['RPN_AUG'] = {'instances':Instances.cat([outputs['RPN']['instances'],outputs['AUG']['instances']])}
            outputs = self.nms(outputs,['RPN_AUG'])
            self.draw(outputs,'RPN_AUG',batched_inputs,os.path.join(self.save_path,batched_inputs[0]['file_name'].split("/")[-1].split('.')[0]+'_RPN_AUG.jpg'))
            del outputs['AUG']
        return outputs

    def draw(self,outputs,name,batched_inputs,save_name):
        if not self.draw_result:
            return
        boxes=outputs[name]["instances"].pred_boxes.tensor.cpu()
        logits=outputs[name]["instances"].scores.cpu()
        phrases=outputs[name]["instances"].pred_classes.cpu().tolist()
        phrases = [str(i) for i in phrases]

        detections = sv.Detections(xyxy=boxes.numpy())
        labels = [
            f"{phrase} {logit:.2f}"
            for phrase, logit
            in zip(phrases, logits)
        ]
        box_annotator = sv.BoxAnnotator()
        annotated_frame = cv2.cvtColor(np.asarray(batched_inputs[0]["img_pil"]), cv2.COLOR_RGB2BGR)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(save_name, annotated_frame)

    def draw_gt(self,batched_inputs,dir):
        if not self.draw_result:
            return
        labels = []
        boxes = []
        for annotation in batched_inputs[0]['annotations']:
            labels.append(annotation['category_id'])
            boxes.append(annotation['bbox'])
        labels = torch.LongTensor(labels)
        boxes = torch.FloatTensor(boxes)
        scores = torch.ones_like(labels)
        size = (batched_inputs[0]['height'], batched_inputs[0]['width'])
        output = {"instances": Instances(size)}
        output["instances"].pred_boxes = Boxes(boxes)
        output["instances"].scores = scores
        output["instances"].pred_classes = labels
        outputs = {'GT':output}
        self.draw(outputs,'GT',batched_inputs,os.path.join(dir,batched_inputs[0]['file_name'].split("/")[-1].split('.')[0]+'_gt.jpg'))
