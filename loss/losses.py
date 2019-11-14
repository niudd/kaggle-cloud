import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_bce(logit, truth, weight=[0.2, 0.8]):
    logit = logit.view(-1)
    truth = truth.view(-1)
    assert(logit.shape==truth.shape)
    
    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    if 0:
        loss = loss.mean()
    if 1:
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (weight[1]*pos*loss/pos_weight + weight[0]*neg*loss/neg_weight).sum()
    return loss

def soft_dice_loss(logit, truth, weight=[0.2,0.8]):#weight for 0-1
    batch_size = logit.size()[0]
    logit = logit.view(batch_size,-1)
    truth = truth.view(batch_size,-1)
    assert(logit.shape==truth.shape)

    p = torch.sigmoid(logit)
    t = truth
    w = truth.detach()
    w = w*(weight[1]-weight[0])+weight[0]

    p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
    t = w*(t*2-1)

    intersection = (p * t).sum(-1)
    union =  (p * p).sum(-1) + (t * t).sum(-1)
    dice  = 1 - 2*intersection/union

    loss = dice.mean()
    return loss

def dice_loss(logit, target, nonempty_only=False):
    logit = torch.sigmoid(logit)
    
    if nonempty_only:
        n,c = logit.size()[:2]
        iflat = logit.view([n,c,-1])
        tflat = target.view([n,c,-1])
        indexing = tflat.sum(dim=[2])>0
        iflat = iflat[indexing]
        tflat = tflat[indexing]
        
        intersection = (iflat * tflat).sum()
        
        return 1 - ((2. * intersection) /
                  (iflat.sum() + tflat.sum()))
    else:
        smooth = 1.

        iflat = logit.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
    
        return 1 - ((2. * intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth))

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2):
#         super().__init__()
#         self.gamma = gamma
        
#     def forward(self, input, target):
#         if not (target.size() == input.size()):
#             raise ValueError("Target size ({}) must be the same as input size ({})"
#                              .format(target.size(), input.size()))

#         max_val = (-input).clamp(min=0)
#         loss = input - input * target + max_val + \
#             ((-max_val).exp() + (-input - max_val).exp()).log()

#         invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
#         loss = (invprobs * self.gamma).exp() * loss
        
#         return loss.sum(dim=1).mean()

def f1_loss(logits, labels):
    __small_value=1e-6
    beta = 1
    batch_size = logits.size()[0]
    p = F.sigmoid(logits)
    l = labels
    num_pos = torch.sum(p, 1) + __small_value
    num_pos_hat = torch.sum(l, 1) + __small_value
    tp = torch.sum(l * p, 1)
    precise = tp / num_pos
    recall = tp / num_pos_hat
    fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + __small_value)
    loss = fs.sum() / batch_size
    return (1 - loss)