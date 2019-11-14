import numpy as np
import pandas as pd


##========================================================================================
##========================================================================================
import torch
import os
import shutil
import logging

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return model, optimizer#checkpoint

def set_n_get_device(device_id, data_device_id="cuda:0"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id#"0"#"0, 1, 2, 3, 4, 5"
    device = torch.device(data_device_id if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    #torch.set_num_threads(20)
    return device

##========================================================================================
##========================================================================================
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

##========================================================================================
##========================================================================================
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import glob

class SteelDataset(Dataset):
    def __init__(self, img_id_list, IMG_SIZE, mode='train', augmentation=False):
        self.img_id_list = img_id_list
        self.IMG_SIZE = IMG_SIZE
        self.mode = mode
        self.augmentation = augmentation
        if self.mode=='test':
            self.path = '../data/raw/test/'
    
    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        if self.mode=='test':
            img_path = self.path + img_id
            img = plt.imread(img_path)[:,:,0]/255
            #width, height = img.shape
            #img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = np.expand_dims(img, 0)
            return img
    
    def __len__(self):
        return len(self.img_id_list)

def prepare_testset(test_fnames, BATCH_SIZE, NUM_WORKERS, IMG_SIZE=512):
    #sub = pd.read_csv('data/raw/sample_submission.csv')
    #test_fnames = sub.ImageId.tolist()
    #test_fnames = [f.split('/')[-1] for f in glob.glob('../input/severstal-steel-defect-detection/test_images/*')]
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




