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


class UNetResNet34(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.Unet = smp.Unet(encoder_name='resnet34', 
                             encoder_weights='imagenet', 
                             classes=4,
                             activation=None,
                             center=True)
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
        
        logit = self.Unet(x)
        if self.debug:
            print('logit: ', logit.size())
        
        return logit
    
    ##-----------------------------------------------------------------

    def criterion(self, logit, truth, nonempty_only=False):
        """Define the (customized) loss function here."""        
        Loss_FUNC = nn.BCEWithLogitsLoss()
        #Loss_FUNC = FocalLoss(alpha=1, gamma=2, logits=True, reduce=True)
        bce_loss = Loss_FUNC(logit, truth)
        #loss = Loss_FUNC(logit, truth)
        
        #loss = L.lovasz_hinge(logit, truth, ignore=None)#255
        #loss = L.symmetric_lovasz(logit, truth)
        loss = 0.5 * dice_loss(logit, truth) + 0.5 * bce_loss
        #loss = dice_loss(logit, truth)
        #loss = weighted_bce(logit, truth)
        return loss

    def metric(self, logit, truth, nonempty_only=False):
        """Define metrics for evaluation especially for early stoppping."""
        #return iou_pytorch(logit, truth)
        return dice(logit, truth, nonempty_only)

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
    if multi_gpu:
        net.module.set_mode('test')
    else:
        net.set_mode('test')
    with torch.no_grad():
        if mode=='valid':
            for i, (image, masks) in enumerate(test_dl):
                input_data = image.to(device=device, dtype=torch.float)
                logit = net(input_data).cpu().numpy()
                if tta:#horizontal flip
                    input_data_flip = torch.flip(image, [3]).to(device=device, dtype=torch.float)
                    logit_flip = net(input_data_flip).cpu().numpy()[:,:,:,::-1]#vertical: [:,:,::-1,:]
                    logit = (logit + logit_flip) / 2
                if y_pred is None:
                    y_pred = logit
                else:
                    y_pred = np.concatenate([y_pred, logit], axis=0)
        elif mode=='test':
            for i, image in enumerate(test_dl):
                input_data = image.to(device=device, dtype=torch.float)
                logit = net(input_data).cpu().numpy()
                if tta:#horizontal flip
                    input_data_flip = torch.flip(image, [3]).to(device=device, dtype=torch.float)
                    logit_flip = net(input_data_flip).cpu().numpy()[:,:,:,::-1]
                    logit = (logit + logit_flip) / 2
                if y_pred is None:
                    y_pred = logit
                else:
                    y_pred = np.concatenate([y_pred, logit], axis=0)
    h,w = y_pred.shape[2], y_pred.shape[3]
    return y_pred.reshape(-1, 4, h, w)#Nx4x256x1600

