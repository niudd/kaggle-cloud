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

import loss.lovasz_losses as L
from loss.losses import dice_loss, FocalLoss, weighted_bce, soft_dice_loss
from utils.metrics import iou_pytorch, dice

from model.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback, patch_replication_callback


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        #self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        #self.bn = nn.BatchNorm2d(out_channels)
        self.bn = SynchronizedBatchNorm2d(out_channels, eps=1e-5, affine=False)
        #self.bn = DataParallelWithCallback(self.bn, device_ids=[0, 1])
        #self.bn = nn.DataParallel(self.bn, device_ids=[0, 1])
        #patch_replication_callback(self.bn)  # monkey-patching

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
        self.encoder2 = nn.Sequential(self.resnet.layer1, SCSE(64))
        self.encoder3 = nn.Sequential(self.resnet.layer2, SCSE(128))
        self.encoder4 = nn.Sequential(self.resnet.layer3, SCSE(256))
        self.encoder5 = nn.Sequential(self.resnet.layer4, SCSE(512))

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = Decoder(512+256, 512, 64)
        self.decoder4 = Decoder(256+64, 256, 64)
        self.decoder3 = Decoder(128+64, 128,  64)
        self.decoder2 = Decoder( 64+ 64, 64, 64)
        self.decoder1 = Decoder(64    , 32,  64)

        self.logit    = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  4, kernel_size=1, padding=0),
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

        f = self.center(e5)
        if self.debug:
            print('center',f.size())

        d5 = self.decoder5(f,e5)
        if self.debug:
            print('d5',d5.size())
        d4 = self.decoder4(d5, e4)
        if self.debug:
            print('d4',d4.size())
        d3 = self.decoder3(d4,e3)
        if self.debug:
            print('d3',d3.size())
        d2 = self.decoder2(d3,e2)
        if self.debug:
            print('d2',d2.size())
        d1 = self.decoder1(d2)
        if self.debug:
            print('d1',d1.size())

        f = torch.cat((
            d1,
            F.upsample(d2,scale_factor= 2, mode='bilinear',align_corners=False),
            F.upsample(d3,scale_factor= 4, mode='bilinear',align_corners=False),
            F.upsample(d4,scale_factor= 8, mode='bilinear',align_corners=False),
            F.upsample(d5,scale_factor=16, mode='bilinear',align_corners=False),
        ),1)
        if self.debug:
            print('hypercolum', f.size())
        f = F.dropout(f, p=0.40, training=self.training)
        logit = self.logit(f)
        if self.debug:
            print('logit', logit.size())
        #try:
        #    print(logit.squeeze(1).shape, 'no problem')
        #except:
        #    print(x.shape, x, 'problem')
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



