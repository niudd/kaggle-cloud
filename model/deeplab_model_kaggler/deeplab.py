import sys
sys.path.insert(0,'../..')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .aspp import build_aspp
from .decoder import build_decoder
from .backbone import build_backbone

import loss.lovasz_losses as L
from loss.losses import dice_loss, FocalLoss, weighted_bce, soft_dice_loss
from utils.metrics import iou_pytorch, dice


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, debug=False, clf_path=False):
        super(DeepLab, self).__init__()
        self.debug = debug
        self.clf_path = clf_path

        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        input = torch.cat([
            (input-mean[0])/std[0],
            (input-mean[1])/std[1],
            (input-mean[2])/std[2],
        ],1)
        if self.debug:
            print('input:', input.size())
        x, low_level_feat = self.backbone(input)
        if self.debug:
            print('backbone--x:', x.size())
            print('backbone--low_level_feat:', low_level_feat.size())
        x = self.aspp(x)
        if self.debug:
            print('aspp:', x.size())
        x = self.decoder(x, low_level_feat)
        if self.debug:
            print('decoder:', x.size())
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        if self.debug:
            print('interpolate:', x.size())

        if self.clf_path:
            logit_clf = F.adaptive_max_pool2d(x, 1).view(x.size()[0], -1)
            if self.debug:
                print('logit_clf: ', logit_clf.size())
            return x, logit_clf
        else:
            return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    ##-----------------------------------------------------------------

    def criterion(self, logit, truth, nonempty_only=False):
        """Define the (customized) loss function here."""
        if nonempty_only:
            Loss_FUNC = nn.BCEWithLogitsLoss(reduction='none')
            full_bce_loss = Loss_FUNC(logit, truth)
            bce_loss = full_bce_loss.mean(dim=[2,3])[truth.sum(dim=[2,3])>0].mean()
            
            #d_loss = dice_loss(logit, truth, nonempty_only)
            
            #loss = 0.5 * d_loss + 0.5 * bce_loss
            return bce_loss#loss
        
        if self.clf_path:
            Loss_FUNC = nn.BCEWithLogitsLoss()
            loss = Loss_FUNC(logit, truth)
            return loss
        
        Loss_FUNC = nn.BCEWithLogitsLoss()
        #Loss_FUNC2 = FocalLoss(alpha=1, gamma=2, logits=True, reduce=True)
        bce_loss = Loss_FUNC(logit, truth)
        #focal_loss = Loss_FUNC2(logit, truth)
        #loss = Loss_FUNC(logit, truth)
        
        #lovasz_loss = L.lovasz_hinge(logit, truth, ignore=None)#255
        #loss = L.symmetric_lovasz(logit, truth)
        
        loss = 0.5 * dice_loss(logit, truth) + 0.5 * bce_loss
        #loss = 0.5 * soft_dice_loss(logit, truth, weight=[0.95, 0.05]) + 0.5 * weighted_bce(logit, truth, weight=[0.95, 0.05])
        #loss = 0.5 * lovasz_loss + 0.5 * bce_loss
                
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


############## with clf path
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

############## only segment
# def predict_proba(net, test_dl, device, multi_gpu=False, mode='test', tta=True):
#     if tta:
#         print("use TTA")
#     else:
#         print("not use TTA")
#     y_pred = None
#     if multi_gpu:
#         net.module.set_mode('test')
#     else:
#         net.set_mode('test')
#     with torch.no_grad():
#         if mode=='valid':
#             for i, (image, masks) in enumerate(test_dl):
#                 input_data = image.to(device=device, dtype=torch.float)
#                 logit = net(input_data).cpu().numpy()
#                 if tta:#horizontal flip
#                     input_data_flip = torch.flip(image, [3]).to(device=device, dtype=torch.float)
#                     logit_flip = net(input_data_flip).cpu().numpy()[:,:,:,::-1]#vertical: [:,:,::-1,:]
#                     logit = (logit + logit_flip) / 2
#                 if y_pred is None:
#                     y_pred = logit
#                 else:
#                     y_pred = np.concatenate([y_pred, logit], axis=0)
#         elif mode=='test':
#             for i, image in enumerate(test_dl):
#                 input_data = image.to(device=device, dtype=torch.float)
#                 logit = net(input_data).cpu().numpy()
#                 if tta:#horizontal flip
#                     input_data_flip = torch.flip(image, [3]).to(device=device, dtype=torch.float)
#                     logit_flip = net(input_data_flip).cpu().numpy()[:,:,:,::-1]
#                     logit = (logit + logit_flip) / 2
#                 if y_pred is None:
#                     y_pred = logit
#                 else:
#                     y_pred = np.concatenate([y_pred, logit], axis=0)
#     h,w = y_pred.shape[2], y_pred.shape[3]
#     return y_pred.reshape(-1, 4, h, w)#Nx4x256x1600


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


