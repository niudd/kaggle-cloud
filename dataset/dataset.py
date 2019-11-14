import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import collections
from tqdm import tqdm_notebook, tqdm
import os
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import random
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from .mask_functions import rle2mask, mask2rle
from .augmentation import do_augmentation


class SteelDataset(Dataset):
    def __init__(self, img_id_list, IMG_SIZE, mode='train', augmentation=False, crop=False, output_shape=None):
        self.img_id_list = img_id_list
        self.IMG_SIZE = IMG_SIZE
        self.mode = mode
        self.augmentation = augmentation
        self.crop = crop
        self.output_shape = output_shape
        if self.mode=='train':
            self.path = '../data/raw/train/'
            self.mask_data = build_mask()
        elif self.mode=='test':
            self.path = '../data/raw/test/'
    
    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        if self.mode=='train':
            img_path = self.path + img_id
            #img = plt.imread(img_path)[:,:,0]/255 #checked: 3-channel all the same
            img = cv2.imread(img_path)#defualt read in BGR format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#cv2.COLOR_BGR2RGB
            img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]))
            img = img / 255
            #img = np.rollaxis(img, 2, 0)#1400x2100x3 --> 3x1400x2100
            #width, height = img.shape

            masks_in_rle = self.mask_data[img_id]
            masks_arr = np.array([cv2.resize(rle2mask(_mask), (self.IMG_SIZE[1], self.IMG_SIZE[0])) 
                                  for _mask in masks_in_rle])
            masks_arr  = (masks_arr>0.5).astype(np.float32)####is adding this correct???
            
            ##augmentation
            if self.augmentation:
                img, masks_arr = do_augmentation(img, masks_arr, self.crop)
            img = np.expand_dims(img, 0)
            if self.output_shape is not None:
                masks_arr = np.array([cv2.resize(_mask, (self.output_shape[1], self.output_shape[0])) 
                                      for _mask in masks_arr])
                masks_arr  = (masks_arr>0.5).astype(np.float32)
            return img, masks_arr
        elif self.mode=='test':
            img_path = self.path + img_id
            #img = plt.imread(img_path)[:,:,0]/255
            img = cv2.imread(img_path)#defualt read in BGR format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#cv2.COLOR_BGR2RGB
            img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]))
            img = img / 255
            #img = np.rollaxis(img, 2, 0)#1400x2100x3 --> 3x1400x2100
            #width, height = img.shape
            img = np.expand_dims(img, 0)
            return img
    
    def __len__(self):
        return len(self.img_id_list)


def build_mask():
    """
    load or build&save mask: {img_id: [mask0, mask1, mask2, mask3], ...}
    """
    ## if already built, load
    if os.path.exists('../data/processed/train_mask_in_rle.pkl'):
        with open('../data/processed/train_mask_in_rle.pkl', 'rb') as f:
            mask_data = pickle.load(f)
        return mask_data
    
    ## if not built, build and save
    train_mask = pd.read_csv('../data/raw/train.csv')
    train_mask.sort_values(by=['Image_Label'], inplace=True)
    train_mask['ImageId'] = train_mask['Image_Label'].apply(lambda x: x.split('_')[0])
    img_id_list = train_mask.ImageId.unique()
    train_mask['ClassId'] = train_mask['Image_Label'].apply(lambda x: x.split('_')[1]).astype(str)
    d = {'Fish':0, 'Flower':1, 'Gravel':2, 'Sugar':3}
    train_mask['ClassId'] = [d[c] for c in train_mask.ClassId]
    train_mask['EncodedPixels'] = train_mask.EncodedPixels.fillna('')
    train_mask.set_index(['ImageId'], inplace=True)
    #train_mask.head(5)
    
    mask_data = {}
    for img_id in img_id_list:
        masks = mask_df = train_mask.loc[img_id, 'EncodedPixels'].tolist()
        #mask_df.sort_values(by=['ClassId'], ascending=True, inplace=True)
        mask_data[img_id] = masks
    # save
    with open('../data/processed/train_mask_in_rle.pkl', 'wb') as f:
        pickle.dump(mask_data, f)
    return mask_data


def prepare_trainset(BATCH_SIZE, NUM_WORKERS, SEED, IMG_SIZE=(1400, 2100), debug=False, 
                     nonempty_only=False, crop=False, output_shape=None):
    train_path = '../data/raw/train/*'
    test_path = '../data/raw/test/*'
    train_fname_list = glob.glob(train_path)
    test_fname_list = glob.glob(test_path)
    print('Count images in train/test folder: ', len(train_fname_list), len(test_fname_list))#(10712, 1377)
    
    train_mask = pd.read_csv('../data/raw/train.csv')
    train_mask.sort_values(by=['Image_Label'], inplace=True)
    train_mask['ImageId'] = train_mask['Image_Label'].apply(lambda x: x.split('_')[0])
    train_mask['ClassId'] = train_mask['Image_Label'].apply(lambda x: x.split('_')[1]).astype(str)
    d = {'Fish':0, 'Flower':1, 'Gravel':2, 'Sugar':3}
    train_mask['ClassId'] = [d[c] for c in train_mask.ClassId]
    train_mask['EncodedPixels'] = train_mask.EncodedPixels.fillna('')
    train_mask.set_index(['ImageId'], inplace=True)
    train_mask['has_mask'] = (train_mask.EncodedPixels!='').astype(np.int)
    #train_mask['mask_class'] = train_mask['ClassId']*train_mask['has_mask']
    
    train_mask_merged = train_mask.groupby(['ImageId'])[['has_mask']].agg('sum').rename(columns={'has_mask': 'cnt_mask'})
    
    if nonempty_only:
        train_mask_merged = train_mask_merged.loc[train_mask_merged.cnt_mask>0, ]
    else:
        pass

    #================================================#
    #stratify by cnt_mask
    train_fname_list = train_mask_merged.index.tolist()
    train_fnames, valid_fnames = train_test_split(train_fname_list, test_size=0.15, 
                   stratify=train_mask_merged.cnt_mask.tolist(), random_state=SEED)#test_size=0.1
    #================================================#
    
    #debug mode
    if debug:
        train_fnames = np.random.choice(train_fnames, int(len(train_fnames)/5), replace=True).tolist()
        valid_fnames = np.random.choice(valid_fnames, int(len(valid_fnames)/5), replace=True).tolist()
    print('Count of trainset (for training): ', len(train_fnames))
    print('Count of validset (for training): ', len(valid_fnames))
    
    ## build pytorch dataset and dataloader
    train_ds = SteelDataset(train_fnames, IMG_SIZE, mode='train', augmentation=True, crop=crop, output_shape=output_shape)
    val_ds = SteelDataset(valid_fnames, IMG_SIZE, mode='train', augmentation=False, crop=crop, output_shape=output_shape)
    #print(len(train_ds.fname_list), len(val_ds.fname_list))
    train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            #sampler=sampler,
            num_workers=NUM_WORKERS,
            drop_last=True
        )
    val_dl = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            #sampler=sampler,
            num_workers=NUM_WORKERS,
            drop_last=True
        )
    
    return train_dl, val_dl

def prepare_testset(BATCH_SIZE, NUM_WORKERS, IMG_SIZE=512):
    #sub = pd.read_csv('data/raw/sample_submission.csv')
    #test_fnames = sub.ImageId.tolist()
    test_fnames = [f.split('/')[-1] for f in glob.glob('../data/raw/test/*')]
    test_ds = SteelDataset(test_fnames, IMG_SIZE, mode='test', augmentation=False)
    #print(len(train_ds.fname_list), len(val_ds.fname_list))
    test_dl = DataLoader(
                        test_ds,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        #sampler=sampler,
                        num_workers=NUM_WORKERS,
                        drop_last=False
                    )
    return test_dl

