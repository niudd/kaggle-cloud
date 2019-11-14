import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
import pickle
import os
import logging
import time
from IPython.core.debugger import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import save_checkpoint, load_checkpoint, set_logger
from utils.gpu_utils import set_n_get_device

from dataset.dataset import prepare_trainset
from model.model_unet import UNetResNet34, predict_proba
from model.deeplab_model_kaggler.lr_scheduler import LR_Scheduler

#from sync_batchnorm import convert_model

import argparse
#from pprint import pprint


parser = argparse.ArgumentParser(description='====Model Parameters====')

#parser.add_argument('--use_cutie2', type=bool, default=False) # True to read image from doc_path
#parser.add_argument('--doc_path', type=str, default='data/pru_column_data') #input data (json) #sroie2019_data
parser.add_argument('--SEED', type=int, default=1234)
#parser.add_argument('--weight_decay', type=float, default=0.0005)

params = parser.parse_args()

SEED = params.SEED
print('SEED=%d'%SEED)

#pprint(params)
#print(parser.parse_args())

######### Define the training process #########
def run_check_net(train_dl, val_dl, multi_gpu=[0, 1], nonempty_only_loss=False):
    set_logger(LOG_PATH)
    logging.info('\n\n')
    #---
    if MODEL == 'UNetResNet34':
        net = UNetResNet34(debug=False).cuda(device=device)
    #elif MODEL == 'RESNET18':
    #    net = AtlasResNet18(debug=False).cuda(device=device)

#     for param in net.named_parameters():
#         if param[0][:8] in ['decoder5']:#'decoder5', 'decoder4', 'decoder3', 'decoder2'
#             param[1].requires_grad = False

    # dummy sgd to see if it can converge ...
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
    #                  lr=LearningRate, momentum=0.9, weight_decay=0.0001)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.045)#LearningRate
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
#                                                            factor=0.5, patience=4,#4 resnet34 
#                                                            verbose=False, threshold=0.0001, 
#                                                            threshold_mode='rel', cooldown=0, 
#                                                            min_lr=0, eps=1e-08)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.9, last_epoch=-1)
    
    #train_params = filter(lambda p: p.requires_grad, net.parameters())
    # 1x lr for encoder, 10x lr for other
    enc_params = [p[1] for p in net.named_parameters() if ('resnet' in p[0] or 'encoder' in p[0])]
    other_params = [p[1] for p in net.named_parameters() if ('resnet' not in p[0] and 'encoder' not in p[0])]
    train_params = [{'params': enc_params, 'lr': LearningRate},
                    {'params': other_params, 'lr': LearningRate * 10}]
    
    optimizer = torch.optim.SGD(train_params, momentum=0.9, weight_decay=0.0001)#lr=LearningRate
    scheduler = LR_Scheduler('poly', LearningRate, NUM_EPOCHS, len(train_dl))#lr_scheduler=['poly', 'step', 'cos']
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LR, last_epoch=-1)
    
    if warm_start:
        logging.info('warm_start: '+last_checkpoint_path)
        net, _ = load_checkpoint(last_checkpoint_path, net)
    
    # using multi GPU
    if multi_gpu is not None:
        net = nn.DataParallel(net, device_ids=multi_gpu)
    
    #use sync_batchnorm
    #net = convert_model(net)

    diff = 0
    best_val_metric = -0.1
    optimizer.zero_grad()
    
    #seed = get_seed()
    #seed = SEED
    #logging.info('aug seed: '+str(seed))
    #ia.imgaug.seed(seed)
    #np.random.seed(seed)
    
    for i_epoch in range(NUM_EPOCHS):
        ### adjust learning rate
        #scheduler.step(epoch=i_epoch)
        #print('lr: %f'%scheduler.get_lr()[0])
        
        t0 = time.time()
        # iterate through trainset
        if multi_gpu is not None:
            net.module.set_mode('train')
        else:
            net.set_mode('train')
        train_loss_list, train_metric_list = [], []
        #for seed in [1]:#[1, SEED]:#augment raw data with a duplicate one (augmented)
        #seed = get_seed()
        #np.random.seed(seed)
        #ia.imgaug.seed(i//10)
        for i, (image, masks) in enumerate(train_dl):
            ## adjust learning rate
            scheduler(optimizer, i, i_epoch, best_val_metric)
            
            input_data = image.to(device=device, dtype=torch.float)
            truth = masks.to(device=device, dtype=torch.float)
            #set_trace()
            logit = net(input_data)#[:, :3, :, :]
            
            if multi_gpu is not None:
                _train_loss  = net.module.criterion(logit, truth, nonempty_only_loss)
                _train_metric  = net.module.metric(logit, truth, nonempty_only_loss)#device='gpu'
            else:
                _train_loss  = net.criterion(logit, truth, nonempty_only_loss)
                _train_metric  = net.metric(logit, truth, nonempty_only_loss)#device='gpu'
            train_loss_list.append(_train_loss.item())
            train_metric_list.append(_train_metric.item())#.detach()

            #grandient accumulation step=2
            acc_step = 2
            _train_loss = _train_loss / acc_step
            _train_loss.backward()
            if (i+1)%acc_step==0:
                optimizer.step()
                optimizer.zero_grad()

        train_loss = np.mean(train_loss_list)
        train_metric = np.mean(train_metric_list)

        # compute valid loss & metrics (concatenate valid set in cpu, then compute loss, metrics on full valid set)
        net.module.set_mode('valid')
        with torch.no_grad():
            val_loss_list, val_metric_list = [], []
            for i, (image, masks) in enumerate(val_dl):
                input_data = image.to(device=device, dtype=torch.float)
                truth = masks.to(device=device, dtype=torch.float)
                logit = net(input_data)
                
                if multi_gpu is not None:
                    _val_loss  = net.module.criterion(logit, truth, nonempty_only_loss)
                    _val_metric  = net.module.metric(logit, truth, nonempty_only_loss)#device='gpu'
                else:
                    _val_loss  = net.criterion(logit, truth, nonempty_only_loss)
                    _val_metric  = net.metric(logit, truth, nonempty_only_loss)#device='gpu'
                val_loss_list.append(_val_loss.item())
                val_metric_list.append(_val_metric.item())#.detach()

            val_loss = np.mean(val_loss_list)
            val_metric = np.mean(val_metric_list)
            
#             logit_valid, truth_valid = None, None
#             for j, (image, masks) in enumerate(val_dl):
#                 input_data = image.to(device=device, dtype=torch.float)
#                 logit = net(input_data).cpu().float()
#                 truth = masks.cpu().float()
#                 if logit_valid is None:
#                     logit_valid = logit
#                     truth_valid = truth
#                 else:
#                     logit_valid = torch.cat((logit_valid, logit), dim=0)
#                     truth_valid = torch.cat((truth_valid, truth), dim=0)
#             if multi_gpu is not None:
#                 val_loss = net.module.criterion(logit_valid, truth_valid)
#                 val_metric = net.module.metric(logit_valid, truth_valid)
#             else:
#                 val_loss = net.criterion(logit_valid, truth_valid)
#                 val_metric = net.metric(logit_valid, truth_valid)

        # Adjust learning_rate
        #scheduler.step(val_metric)
        
        #for 1024 trainging is harder, sometimes too early stop, force to at least train 40 epochs
        if i_epoch>=-1:
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                is_best = True
                diff = 0
            else:
                is_best = False
                diff += 1
                if diff > early_stopping_round:
                    logging.info('Early Stopping: val_metric does not increase %d rounds'%early_stopping_round)
                    #print('Early Stopping: val_iou does not increase %d rounds'%early_stopping_round)
                    break
        else:
            is_best = False

        #save checkpoint
        checkpoint_dict = \
        {
            'epoch': i,
            'state_dict': net.module.state_dict() if multi_gpu is not None else net.state_dict(),
            'optim_dict' : optimizer.state_dict(),
            'metrics': {'train_loss': train_loss, 'val_loss': val_loss, 
                        'train_metric': train_metric, 'val_metric': val_metric}
        }
        save_checkpoint(checkpoint_dict, is_best=is_best, checkpoint=checkpoint_path)

        #if i_epoch%20==0:
        if i_epoch>-1:
            logging.info('[EPOCH %05d]train_loss, train_metric: %0.5f, %0.5f; val_loss, val_metric: %0.5f, %0.5f; time elapsed: %0.1f min'%(i_epoch, train_loss.item(), train_metric.item(), val_loss.item(), val_metric.item(), (time.time()-t0)/60))


#
def seed_everything(seed):
    #random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    #np.random.seed(seed)
    #torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #tf.set_random_seed(seed)
seed_everything(SEED)


######### Config the training process #########
#device = set_n_get_device("0, 1, 2, 3", data_device_id="cuda:0")#0, 1, 2, 3, IMPORTANT: data_device_id is set to free gpu for storing the model, e.g."cuda:1"
MODEL = 'UNetResNet34'#'RESNET34', 'RESNET18', 'INCEPTION_V3', 'BNINCEPTION', 'SEResnet50'
#AUX_LOGITS = True#False, only for 'INCEPTION_V3'
print('====MODEL ACHITECTURE: %s===='%MODEL)

device = set_n_get_device("0,1,2,3", data_device_id="cuda:0")#0, 1, 2, 3, IMPORTANT: data_device_id is set to free gpu for storing the model, e.g."cuda:1"
multi_gpu = [0,1,2,3]#use 2 gpus

#SEED = 1234#5678#4567#3456#2345#1234
debug = False # if True, load 100 samples, False
IMG_SIZE = (512, 768) #(1024, 1536)
BATCH_SIZE = 8#16
NUM_WORKERS = 24
warm_start, last_checkpoint_path = False, 'checkpoint/%s_%s_v1_seed%s/best.pth.tar'%(MODEL, IMG_SIZE, SEED)
checkpoint_path = '../checkpoint/%s_%dx%d_v3_seed%s'%(MODEL, IMG_SIZE[0], IMG_SIZE[1], SEED)
LOG_PATH = '../logging/%s_%dx%d_v3_seed%s.log'%(MODEL, IMG_SIZE[0], IMG_SIZE[1], SEED)#
#torch.cuda.manual_seed_all(SEED)

NUM_EPOCHS = 50
early_stopping_round = 10 #500#50
LearningRate = 0.02 #0.002
#MIN_LR = 0.005


######### Load data #########
train_dl, val_dl = prepare_trainset(BATCH_SIZE, NUM_WORKERS, SEED, IMG_SIZE, debug, 
                                    nonempty_only=False, crop=False)#True: Only using nonempty-mask!

######### Run the training process #########
run_check_net(train_dl, val_dl, multi_gpu=multi_gpu, nonempty_only_loss=False)

print('------------------------\nComplete SEED=%d\n------------------------'%SEED)
