from typing import List
import torch
from detectron2.layers import batched_nms
import torch.nn.functional as F

def merge_probs_split_bayesian(probsa, probsb):
    probsa = torch.log(probsa)
    probsb = torch.log(probsb)
    sum_logits = (probsa + probsb)/2
    probs = F.softmax(sum_logits,dim=-1)
    scores = probs.max(1)[0]
    return probs, scores

def merge_probs_split(probsa, probsb):
    # max
    scoresa = torch.max(probsa,dim=1,keepdim=False)[0]
    scoresb = torch.max(probsb,dim=1,keepdim=False)[0]
    mask = (scoresa > scoresb).float()
    probs = probsa*(mask.unsqueeze(1)) + probsb*(1-mask.unsqueeze(1))
    scores = scoresa*mask + scoresb*(1-mask)

    return probs, scores

def weighted_box_fusion_split(bboxa, bboxb, scorea, scoreb):
    score = torch.cat((scorea.unsqueeze(1),scoreb.unsqueeze(1)),dim=1)
    weight = score / torch.sum(score, dim=1, keepdim=True)
    bboxa = bboxa * weight[:,0:1]
    bboxb = bboxb * weight[:,1:]
    out_bbox = bboxa + bboxb

    return out_bbox

def weighted_box_fusion(bbox, score):
    weight = score / torch.sum(score)
    out_bbox = bbox * weight[:,None]
    out_bbox = torch.sum(out_bbox, dim=0)    
    return out_bbox

def bayesian_fusion_multiclass(match_score_vec, pred_class):
    assert (match_score_vec.max(1)[1]==pred_class).float().mean()==1
    log_scores = torch.log(match_score_vec)
    sum_logits = torch.sum(log_scores, dim=0)
    exp_logits = torch.exp(sum_logits)
    score_norm = exp_logits / torch.sum(exp_logits)

    pred_class = torch.unique(pred_class)
    assert len(pred_class)==1
    out_score = score_norm[pred_class]
    return out_score[0], score_norm, pred_class[0]

def avg_bbox_fusion(match_bbox_vec):
    avg_bboxs = torch.sum(match_bbox_vec,dim=0) / len(match_bbox_vec)
    return avg_bboxs

class MyNMS:
    def __init__(self, method):
        self.method = method
        if self.method is not None:
            self.update_cfg()
    
    def update_cfg(self):
        if self.method !='nms':
            assert len(self.method)==2
            if self.method[0]=='p':
                self.score_method = 'probEn'
            elif self.method[0]=='a':
                self.score_method = 'avg'
            elif self.method[0]=='m':
                self.score_method = 'max'
            else:
                raise NotImplementedError
            if self.method[1]=='s':
                self.box_method = 's-avg'
            elif self.method[1]=='a':
                self.box_method = 'avg'
            elif self.method[1]=='m':
                self.box_method = 'max'
            else:
                raise NotImplementedError
            
            if self.score_method == 'max' and self.box_method=='max':
                self.method = 'nms'
        
    def nms_bayesian(self, boxes, probs, labels, iou_threshold):
        boxes_for_nms, boxes = boxes
        x1 = boxes_for_nms[:, 0]
        y1 = boxes_for_nms[:, 1]
        x2 = boxes_for_nms[:, 2]
        y2 = boxes_for_nms[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # Sort First
        scores = probs[torch.arange(probs.size(0)),labels]
        order = scores.argsort(descending=True)
        
        keep = []
        out_classes = []
        match_scores = []
        match_probs = []
        match_bboxs = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            w = torch.max(torch.tensor(0.0), xx2 - xx1 + 1)
            h = torch.max(torch.tensor(0.0), yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = torch.where(ovr <= iou_threshold)[0]
            match = torch.where(ovr > iou_threshold)[0]
            
            match_ind = order[match+1]

            match_prob = list(probs[match_ind])
            match_score = list(scores[match_ind])
            match_label = list(labels[match_ind])
            
            match_bbox = list(boxes[match_ind][:,:4])
            original_prob = probs[i]
            original_score = scores[i]
            original_bbox = boxes[i][:4]
            
            # If some boxes are matched
            if len(match_score)>0:
                match_score += [original_score]
                match_prob += [original_prob]
                match_bbox += [original_bbox]
                match_label += [labels[i]]
                match_prob = torch.stack(match_prob)
                match_score = torch.stack(match_score)
                match_bbox = torch.stack(match_bbox)
                match_label = torch.stack(match_label)
                
                # score fusion
                if self.score_method == "probEn":
                    final_score, final_prob, out_class = bayesian_fusion_multiclass(match_prob, match_label)
                    out_classes.append(out_class)
                elif self.score_method == 'avg':
                    final_prob  = torch.mean(match_prob, dim=0)
                    final_score = torch.mean(match_score)
                    # All matches are of the same category, there will be no second category
                    final_class = torch.unique(match_label)
                    assert len(final_class)==1
                    out_classes.append(final_class[0])
                elif self.score_method == 'max':
                    max_score_id = torch.argmax(match_score)
                    final_prob = match_prob[max_score_id]
                    final_score = match_score[max_score_id]
                    # All matches are of the same category, there will be no second category
                    final_class = torch.unique(match_label)
                    assert len(final_class)==1
                    out_classes.append(final_class[0])
                
                # box fusion
                if self.box_method == 's-avg':
                    final_bbox = weighted_box_fusion(match_bbox, match_score)
                elif self.box_method == 'avg':                
                    final_bbox = avg_bbox_fusion(match_bbox)
                elif self.box_method == 'max':                                
                    max_score_id = torch.argmax(match_score)
                    final_bbox = match_bbox[max_score_id]              
                
                match_scores.append(final_score)
                match_probs.append(final_prob)
                match_bboxs.append(final_bbox)
            else:
                match_probs.append(original_prob)
                match_scores.append(original_score)
                match_bboxs.append(original_bbox)
                out_classes.append(labels[i])

            order = order[inds + 1]

            
        assert len(keep)==len(match_scores)
        assert len(keep)==len(match_bboxs)
        assert len(keep)==len(out_classes)
        assert len(keep)==len(match_probs)

        match_bboxs = torch.stack(match_bboxs)
        match_scores = torch.stack(match_scores)
        match_classes = torch.stack(out_classes)
        match_probs = torch.stack(match_probs)
        keep = torch.stack(keep)

        new_sort = match_scores.argsort(descending=True)

        return keep[new_sort], match_bboxs[new_sort], match_scores[new_sort], match_probs[new_sort], match_classes[new_sort]
        
    def batch_nms_bayesian(self, boxes, scores, probs, labels, iou_threshold):
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device), boxes, scores, probs, labels
        max_coordinate = boxes.max()
        offsets = labels.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
        result = self.nms_bayesian((boxes_for_nms,boxes), probs, labels, iou_threshold)
        return result
    
    def nms(self, boxes: torch.Tensor, scores: torch.Tensor, probs: torch.Tensor, idxs: torch.Tensor, iou_threshold: float):
        if self.method == 'nms':
            keep = batched_nms(boxes, scores, idxs, iou_threshold)
            result = (keep, boxes[keep], scores[keep], probs[keep], idxs[keep])
            return result
        else:
            return self.Probabilistic_Fusion(boxes, scores, probs, idxs, iou_threshold)

    def Probabilistic_Fusion(
        self, boxes: torch.Tensor, scores: torch.Tensor, probs: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
    ):
        
        assert boxes.shape[-1] == 4
        if len(boxes) < 40000:
            # fp16 does not have enough range for batched NMS
            return self.batch_nms_bayesian(boxes.float(), scores, probs, idxs, iou_threshold)

        result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        boxes_keeps, scores_keeps, probs_keeps, labels_keeps = [], [], [], []
        for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
            mask = (idxs == id).nonzero().view(-1)
            keep, boxes_keep, scores_keep, probs_keep, labels_keep= self.nms_bayesian((boxes[mask],boxes[mask]), probs[mask], idxs[mask], iou_threshold)
            boxes_keeps.append(boxes_keep)
            scores_keeps.append(scores_keep)
            probs_keeps.append(probs_keep)
            labels_keeps.append(labels_keep)
            result_mask[mask[keep]] = True
        boxes_keeps = torch.cat(boxes_keeps,dim=0)
        scores_keeps = torch.cat(scores_keeps,dim=0)
        probs_keeps = torch.cat(probs_keeps,dim=0)
        labels_keeps = torch.cat(labels_keeps,dim=0)
        keep = result_mask.nonzero().view(-1)
        new_sort = scores_keeps.argsort(descending=True)
        return keep[new_sort], boxes_keeps[new_sort], scores_keeps[new_sort], probs_keeps[new_sort], labels_keeps[new_sort]
    
    def update(self, method):
        self.method = method
        self.update_cfg()
    
mynms = MyNMS(method=None)