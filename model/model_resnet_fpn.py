import sys
sys.path.append('../')

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import segmentation_models_pytorch as smp

import loss.lovasz_losses as L
from loss.losses import dice_loss, FocalLoss, weighted_bce, soft_dice_loss
from utils.metrics import iou_pytorch, dice


class FPNResNet34(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.FPN = smp.FPN(encoder_name='resnet34', encoder_weights='imagenet', 
                           classes=4, activation=None, final_upsampling=2)#default final_upsampling=4
        self.debug = debug
    
    def forward(self, x):
        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[0])/std[0],
            (x-mean[1])/std[1],
            (x-mean[2])/std[2],
        ],1)
        if self.debug:
            print('input: ', x.size())
        
        logit = self.FPN(x)
        if self.debug:
            print('logit: ', logit.size())
        
        
        logit_clf = F.adaptive_max_pool2d(logit, 1).view(logit.size()[0], -1)
        if self.debug:
            print('logit: ', logit_clf.size())
        return logit, logit_clf
        
        #return logit, None
    
    ##-----------------------------------------------------------------

    def criterion(self, logit, truth, nonempty_only=False, logit_clf=None):
        """Define the (customized) loss function here.""" 
        if logit_clf is not None:
            Loss_FUNC_mask = nn.BCEWithLogitsLoss()
            loss_mask = Loss_FUNC_mask(logit, truth)
            #loss_mask = 0.5 * dice_loss(logit, truth) + 0.5 * Loss_FUNC_mask(logit, truth)

#             p = torch.clamp(torch.sigmoid(logit_clf), 1e-9, 1-1e-9)
#             t = (truth.sum(dim=[2,3])>0).float()
#             loss_label = - t*torch.log(p) - 2*(1-t)*torch.log(1-p)
#             loss_label = loss_label.mean()
            
            return loss_mask#0.5 * loss_mask + 0.5 * loss_label
        
        Loss_FUNC = nn.BCEWithLogitsLoss()
        #Loss_FUNC = FocalLoss(alpha=1, gamma=2, logits=True, reduce=True)
        bce_loss = Loss_FUNC(logit, truth)
        #loss = Loss_FUNC(logit, truth)
        
        #loss = L.lovasz_hinge(logit, truth, ignore=None)#255
        #loss = L.symmetric_lovasz(logit, truth)
        loss = 0.5 * dice_loss(logit, truth) + 0.5 * bce_loss
        #loss = 0.5 * soft_dice_loss(logit, truth, weight=[0.75, 0.25]) + 0.5 * bce_loss
        #loss = 0.1* L.lovasz_hinge(logit, truth, ignore=None) + 0.9 * bce_loss
        #loss = dice_loss(logit, truth)
        #loss = weighted_bce(logit, truth)
        return loss

    def metric(self, logit, truth, nonempty_only=False, logit_clf=None):
        """Define metrics for evaluation especially for early stoppping."""
        #return iou_pytorch(logit, truth)
        return dice(logit, truth, nonempty_only=nonempty_only, logit_clf=logit_clf)

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


def predict_proba(net, test_dl, device, multi_gpu=False, mode='test', tta=True):
    if tta:
        print("use TTA")
    else:
        print("not use TTA")
    y_pred = None
    y_pred_clf = None
    if multi_gpu:
        net.module.set_mode('test')
    else:
        net.set_mode('test')
    with torch.no_grad():
        if mode=='valid':
            for i, (image, masks) in enumerate(test_dl):
                input_data = image.to(device=device, dtype=torch.float)
                logit, logit_clf = net(input_data)
                logit = logit.cpu().numpy()
                logit_clf = logit_clf.cpu().numpy()
                if tta:#horizontal flip
                    input_data_flip = torch.flip(image, [3]).to(device=device, dtype=torch.float)
                    logit_flip, logit_clf_flip = net(input_data_flip)
                    logit_flip = logit_flip.cpu().numpy()[:,:,:,::-1]#vertical: [:,:,::-1,:]
                    logit_clf_flip = logit_clf_flip.cpu().numpy()
                    logit = (logit + logit_flip) / 2
                    logit_clf = (logit_clf + logit_clf_flip) / 2
                if y_pred is None:
                    y_pred = logit
                    y_pred_clf = logit_clf
                else:
                    y_pred = np.concatenate([y_pred, logit], axis=0)
                    y_pred_clf = np.concatenate([y_pred_clf, logit_clf], axis=0)
        elif mode=='test':
            for i, image in enumerate(test_dl):
                input_data = image.to(device=device, dtype=torch.float)
                logit, logit_clf = net(input_data)
                logit = logit.cpu().numpy()
                logit_clf = logit_clf.cpu().numpy()
                if tta:#horizontal flip
                    input_data_flip = torch.flip(image, [3]).to(device=device, dtype=torch.float)
                    logit_flip, logit_clf_flip = net(input_data_flip)
                    logit_flip = logit_flip.cpu().numpy()[:,:,:,::-1]#vertical: [:,:,::-1,:]
                    logit_clf_flip = logit_clf_flip.cpu().numpy()
                    logit = (logit + logit_flip) / 2
                    logit_clf = (logit_clf + logit_clf_flip) / 2
                if y_pred is None:
                    y_pred = logit
                    y_pred_clf = logit_clf
                else:
                    y_pred = np.concatenate([y_pred, logit], axis=0)
                    y_pred_clf = np.concatenate([y_pred_clf, logit_clf], axis=0)
    h,w = y_pred.shape[2], y_pred.shape[3]
    return y_pred.reshape(-1, 4, h, w), y_pred_clf.reshape(-1, 4)

