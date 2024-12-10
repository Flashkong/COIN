import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
import torch.nn.functional as F
class MILCrossEntropy(nn.Module):
    """
    Multi-instance learning loss
    """
    def __init__(self):
        super(MILCrossEntropy, self).__init__()

    def forward(self, x, target, dim=-1, weights=None, avg_positives=False, reduction='mean'):
        # for numerical stability
        # logits_max, _ = torch.max(x, dim=1, keepdim=True)
        # logits = x - logits_max.detach()
        logits = x 
        exp_logits = torch.exp(logits)

        # get non-zero entries off-diagonal
        # identity = torch.eye(target.shape[0]).type_as(target)
        # laplacian = 1 - (target - identity)
        probs = exp_logits / (exp_logits).sum(dim=dim, keepdim=True)
        if avg_positives:  # average the logits over positive targets
            loss = -torch.log(torch.sum(target * probs, dim=dim) / (torch.sum(target, dim=dim) + 1e-6))
        else:  # sum the logits over positive targets
            loss = -torch.log(torch.sum(target * probs, dim=dim))
        if weights is not None:
            loss = loss * weights
        
        if reduction=='mean':
            return loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction=='sum':
            return loss.sum()
    
class MILFocalLoss(nn.Module):
    """
    Multi-instance focal loss
    """
    def __init__(self, class_num, alpha=None, gamma=1.5, size_average=True):
        super(MILFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, x, target, dim=-1, avg_positives=True):
        # for numerical stability
        # logits_max, _ = torch.max(x, dim=1, keepdim=True)
        # logits = x - logits_max.detach()
        logits = x 
        exp_logits = torch.exp(logits)
        probs = exp_logits / (exp_logits).sum(dim=dim, keepdim=True)

        if x.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        
        alpha = ( target * (self.alpha.view(1,-1)) ).sum(1) / (torch.sum(target, dim=dim) + 1e-6)

        if avg_positives:  # average the logits over positive targets
            probs = torch.sum(target * probs, dim=dim) / (torch.sum(target, dim=dim) + 1e-6)
            loss = -alpha*(torch.pow((1-probs), self.gamma))*(probs.log())
        else:  # sum the logits over positive targets
            probs = torch.sum(target * probs, dim=dim)
            loss = -alpha*(torch.pow((1-probs), self.gamma))*(probs.log())

        return loss.mean()

def gradient_discrepancy_loss(model, lossa, lossb):
    grad_losses = []
    for n, p in model.roi_heads.box_predictor.trans.named_parameters():
        if p.requires_grad==False:
            continue
        grad_a = grad([lossa],
            [p],
            create_graph=True,
            only_inputs=True)[0]
        grad_b = grad([lossb],
            [p],
            create_graph=True,
            only_inputs=True)[0]

        if len(p.shape) > 1:
            _cossim = F.cosine_similarity(grad_a.detach(), grad_b, dim=1).mean()
        else:
            _cossim = F.cosine_similarity(grad_a.detach(), grad_b, dim=0)
        grad_losses.append(_cossim)
    
    grad_losses = torch.stack(grad_losses)
    return (1.0 - grad_losses).mean()


def gradient_similarity(model, lossa, lossb):
    grad_losses = []
    for n, p in model.roi_heads.box_predictor.trans.named_parameters():
        if p.requires_grad==False:
            continue
        grad_a = grad([lossa],
            [p],
            create_graph=True,
            only_inputs=True)[0]
        grad_b = grad([lossb],
            [p],
            create_graph=True,
            only_inputs=True)[0]

        if len(p.shape) > 1:
            _cossim = F.cosine_similarity(grad_a, grad_b, dim=1).mean()
        else:
            _cossim = F.cosine_similarity(grad_a, grad_b, dim=0)
        grad_losses.append(_cossim)
    
    grad_losses = torch.stack(grad_losses)
    return grad_losses.mean()