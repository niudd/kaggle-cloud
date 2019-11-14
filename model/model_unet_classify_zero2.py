import sys
sys.path.append('../')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, confusion_matrix
import copy

from model.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback, patch_replication_callback


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        #self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        self.bn = SynchronizedBatchNorm2d(out_channels, eps=1e-5, affine=False)
        #self.bn = SynchronizedBatchNorm2d(out_channels)


    def forward(self, z):
        x = self.conv(z)
        #x = self.dropout(x)
        x = self.bn(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 =  ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.spa_cha_gate = SCSE(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)#False
        if e is not None:
            x = torch.cat([x, e], 1)
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        x = self.spa_cha_gate(x)
        return x

class SCSE(nn.Module):
    def __init__(self, in_ch):
        super(SCSE, self).__init__()
        self.spatial_gate = SpatialGate2d(in_ch, 16)#16
        self.channel_gate = ChannelGate2d(in_ch)
    
    def forward(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1 + g2 #x = g1*x + g2*x
        return x

class SpatialGate2d(nn.Module):
    def __init__(self, in_ch, r=16):
        super(SpatialGate2d, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]),-1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.sigmoid(x)

        x = input_x * x

        return x

class ChannelGate2d(nn.Module):
    def __init__(self, in_ch):
        super(ChannelGate2d, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x

class UNetResNet34(nn.Module):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.
    def load_pretrain(self, pretrain_file):
        self.encoder.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

    def __init__(self, pretrained=True, debug=False):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        self.debug = debug

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            #self.resnet.maxpool,
        )# 64
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.logit_clf = nn.Sequential(
            #nn.BatchNorm1d(512),
            #nn.Dropout(0.5),
            nn.Linear(512, 4), #block.expansion=1, num_classes=28
            #nn.Sigmoid()
        )

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

        x = self.conv1(x)
        if self.debug:
            print('e1',x.size())
        e2 = self.encoder2(x)
        if self.debug:
            print('e2',e2.size())
        e3 = self.encoder3(e2)
        if self.debug:
            print('e3',e3.size())
        e4 = self.encoder4(e3)
        if self.debug:
            print('e4',e4.size())
        e5 = self.encoder5(e4)
        if self.debug:
            print('e5',e5.size())
        
        f = self.avgpool(e5)
        if self.debug:
            print('avgpool',f.size())
        
        f = F.dropout(f, p=0.40, training=self.training)
        
        f = f.view(f.size(0), -1)
        if self.debug:
            print('reshape',f.size())
        
        logit_clf = self.logit_clf(f)
        if self.debug:
            print('logit_clf', logit_clf.size())
        return logit_clf
    
    ##-----------------------------------------------------------------

    def criterion(self, logit, truth):
        """Define the (customized) loss function here."""
        #empty mask: weight 1.0, non-empty mask: weight 0.75
        #Loss_FUNC = nn.BCEWithLogitsLoss(reduction='none')
        #loss = Loss_FUNC(logit, truth)
        #return (loss*(0.75+truth*0.25)).mean()
        Loss_FUNC = nn.BCEWithLogitsLoss()
        loss = Loss_FUNC(logit, truth)
        return loss

#     def metric(self, logit, truth):
#         """
#         AUC score as metric
#         """
#         pred = self.sigmoid(logit.cpu().detach().numpy())
#         truth = truth.cpu().detach().numpy()
#         ##
#         THRESHOLD_candidate = np.arange(0.01, 0.99, 0.01)
#         N = len(THRESHOLD_candidate)
#         best_threshold = [0.01, 0.01, 0.01, 0.01]
#         best_score = -1
#         tn, fp, fn, tp, pos_percent = 0, 0, 0, 0, 0.0

#         for ch in range(4):
#             for i in range(N):
#                 THRESHOLD = copy.deepcopy(best_threshold)
#                 THRESHOLD[ch] = THRESHOLD_candidate[i]
#                 _pred = pred>THRESHOLD
#                 _pred, truth = _pred.reshape(-1, 1), truth.reshape(-1, 1)

#                 _tn, _fp, _fn, _tp = confusion_matrix(truth, _pred).ravel()
#                 _auc = round(roc_auc_score(truth, _pred), 5)
#                 if _tn+_fn==0:
#                     fn_rate = 9999
#                 else:
#                     fn_rate = round(_fn/(_tn+_fn), 5)
#                 _pos_percent = (_tp+_fp)/(_tp+_fp+_tn+_fn)

#                 if _auc > best_score:
#                     best_threshold = copy.deepcopy(THRESHOLD)
#                     best_score = _auc
#                     tn, fp, fn, tp, pos_percent = _tn, _fp, _fn, _tp, _pos_percent
#         return np.round(best_threshold, 2), best_score, tn, fp, fn, tp, pos_percent
    
    def metric(self, logit, truth):
        """
        AUC score as metric
        """
        pred = self.sigmoid(logit.cpu().detach().numpy())
        truth = truth.cpu().detach().numpy()
        ##
        THRESHOLD = [0.5, 0.5, 0.5, 0.5]

        _pred = pred>THRESHOLD
        _pred, truth = _pred.reshape(-1, 1), truth.reshape(-1, 1)

        tn, fp, fn, tp = confusion_matrix(truth, _pred).ravel()
        auc = round(roc_auc_score(truth, _pred), 5)
        pos_percent = (tp+fp)/(tp+fp+tn+fn)
        return THRESHOLD, auc, tn, fp, fn, tp, pos_percent

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

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
            for i, (images, masks) in enumerate(test_dl):
                input_data = images.to(device=device, dtype=torch.float)
                logit = net(input_data).cpu().numpy()
                if tta:#horizontal flip
                    input_data_flip = torch.flip(images, [3]).to(device=device, dtype=torch.float)
                    logit_flip = net(input_data_flip).cpu().numpy()
                    logit = (logit + logit_flip) / 2
                if y_pred is None:
                    y_pred = logit
                else:
                    y_pred = np.concatenate([y_pred, logit], axis=0)
        elif mode=='test':
            for i, images in enumerate(test_dl):
                input_data = images.to(device=device, dtype=torch.float)
                logit = net(input_data).cpu().numpy()
                if tta:#horizontal flip
                    input_data_flip = torch.flip(images, [3]).to(device=device, dtype=torch.float)
                    logit_flip = net(input_data_flip).cpu().numpy()
                    logit = (logit + logit_flip) / 2
                if y_pred is None:
                    y_pred = logit
                else:
                    y_pred = np.concatenate([y_pred, logit], axis=0)
    #IMG_SIZE = y_pred.shape[-1]
    return y_pred.reshape(-1, 4)



