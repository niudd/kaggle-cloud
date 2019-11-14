import numpy as np
import pandas as pd
import cv2
import random
import math
import time, os


def do_resize(image, H, W):
    resized_image = cv2.resize(image,dsize=(W,H))
    return resized_image

def do_augmentation(image, mask=None, crop=False):
    """image: shape=(1400, 2100)
       mask: shape=(4, 1400, 2100)
    """
    if crop:
        image, mask = do_random_crop(image, mask, w=512, h=512)
    
    #if np.random.rand() < 0.6:
    #    image, mask = do_random_black_out(image, mask)
    
    if np.random.rand() < 0.5:
        c = np.random.choice(2)
        if c==0:
            image = do_horizontal_flip(image)
            #image = do_4channels(image, do_horizontal_flip)
            if mask is not None:
                mask = do_4channels(mask, do_horizontal_flip)
        elif c==1:
            image = do_vertical_flip(image)
            #image = do_4channels(image, do_vertical_flip)
            if mask is not None:
                mask = do_4channels(mask, do_vertical_flip)
#         elif c==2:
#             k = np.random.randint(1, 4)
#             image = do_rotation(image, k=k)
#             if mask is not None:
#                 mask = do_4channels(mask, do_rotation, k=k)

    if np.random.rand() < 0.5:
        c = np.random.choice(2)
        if c==0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, limit=0.25)#v10:0.25 #v6:0.125
        elif c==1:
            image, mask = do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 20))#v10: 20, v6:10
        #if c==2:
        #    image, mask = do_elastic_transform2(image, mask, grid=10, distort=np.random.uniform(0, 0.1))

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c==0:
            image = do_brightness_shift(image, alpha=np.random.uniform(-0.15, +0.15))#v10: 0.15
            #image = do_4channels(image, do_brightness_shift, alpha=np.random.uniform(-0.15, +0.15))
        elif c==1:
            image = do_brightness_multiply(image, alpha=np.random.uniform(1-0.25, 1+0.25))
            #image = do_4channels(image, do_brightness_multiply, alpha=np.random.uniform(1-0.25, 1+0.25))
        elif c==2:
            image = do_gamma(image, gamma=np.random.uniform(1-0.25, 1+0.25))
            #image = do_4channels(image, do_gamma, gamma=np.random.uniform(1-0.25, 1+0.25))
    
    if np.random.rand() < 0.5:
        c = np.random.choice(1)
        if c==0:
            image = do_guassian_blur(image, kernal_size=(3, 3))
            #image = do_4channels(image, do_guassian_blur, kernal_size=(3, 3))
#         elif c==1:
#             image = do_perspective_transform(image)
#             if mask is not None:
#                 mask = do_4channels(mask, do_perspective_transform)

    #image = cv2.resize(image, (1024, 512))
    #mask = np.array([cv2.resize(_mask, (1024, 512)) for _mask in mask])
    
    if mask is not None:
        return image, mask
    return image


def get_seed():
    seed = int(time.time()) + int(os.getpid())
    return seed

def do_4channels(image, aug_method, **kwargs):
    output = []
    for _image in image:
        aug_img = aug_method(_image, **kwargs)
        output.append(aug_img)
    return np.array(output)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Define some augmentation methods below
#----------------------------------------------------------------------
#----------------------------------------------------------------------
def do_horizontal_flip(image):
    #flip left-right
    image = cv2.flip(image, 1)
    return image

def do_vertical_flip(image):
    #flip top-down
    image = cv2.flip(image, 0)
    return image

def do_rotation(image, k=1):
    # k: how many times rotate 90 degrees
    image = np.rot90(image, k)
    return image

#----
def do_invert_intensity(image):
    #flip left-right
    image = np.clip(1-image,0,1)
    return image

def do_brightness_shift(image, alpha=0.125):
    image = image + alpha
    image = np.clip(image, 0, 1)
    return image

def do_brightness_multiply(image, alpha=1):
    image = alpha*image
    image = np.clip(image, 0, 1)
    return image

#https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def do_gamma(image, gamma=1.0):
    image = image ** (1.0 / gamma)
    image = np.clip(image, 0, 1)
    return image

#----
def do_guassian_blur(image, kernal_size=(3, 3)):
    image = cv2.GaussianBlur(image, kernal_size, 0)
    return image

def do_perspective_transform(image):
    pts1 = np.float32([[0, 0],[100, 0],[0, 100],[100, 510]])
    pts2 = np.float32([[0, 0],[100, 0],[0, 100],[100, 520]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return image

#----
def do_random_crop(image, mask, w, h):
    height, width = image.shape[:2]
    x,y=0,0
    if width>w:
        x = np.random.choice(width-w)
    if height>h:
        y = np.random.choice(height-h)
    image = image[y:y+h,x:x+w]
    mask  = mask [:,y:y+h,x:x+w]
    return image, mask

def do_shift_scale_crop(image, mask=None, x0=0, y0=0, x1=1, y1=1 ):
    #cv2.BORDER_REFLECT_101
    #cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    image = image[y0:y1,x0:x1]
    if mask is not None:
        mask  = mask [:,y0:y1,x0:x1]

    image = cv2.resize(image,dsize=(width,height))
    if mask is not None:
        mask = np.array([cv2.resize(arr, dsize=(width,height)) for arr in mask])
        mask  = (mask>0.5).astype(np.float32)
    return image, mask

def do_random_shift_scale_crop_pad2(image, mask=None, limit=0.10):

    H, W = image.shape[:2]

    dy = int(H*limit)
    y0 =   np.random.randint(0,dy)
    y1 = H-np.random.randint(0,dy)

    dx = int(W*limit)
    x0 =   np.random.randint(0,dx)
    x1 = W-np.random.randint(0,dx)

    #y0, y1, x0, x1
    image, mask = do_shift_scale_crop(image, mask, x0, y0, x1, y1 )
    return image, mask

def do_shift_scale_rotate2( image, mask, dx=0, dy=0, scale=1, angle=0 ):
    borderMode=cv2.BORDER_REFLECT_101
    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    sx = scale
    sy = scale
    cc = math.cos(angle/180*math.pi)*(sx)
    ss = math.sin(angle/180*math.pi)*(sy)
    rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

    box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ],np.float32)
    box1 = box0 - np.array([width/2,height/2])
    box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat  = cv2.getPerspectiveTransform(box0,box1)

    image = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,
                                borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask_rot = []
    for ch in range(mask.shape[0]):
        _mask = mask[ch,:,:]
        _mask = cv2.warpPerspective(_mask, mat, (width,height),flags=cv2.INTER_NEAREST,#cv2.INTER_LINEAR
                                    borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
        _mask  = (_mask>0.5).astype(np.float32)
        mask_rot.append(_mask)
    mask_rot = np.array(mask_rot)
    return image, mask_rot

#=====================================
def do_random_black_out(image, mask):
    ## generate black box: position and area
    H, W = image.shape

    h, w = int(H * np.random.randint(5, 10) / 10), int(W * np.random.randint(2, 5) / 10)
    #print(w, h)

    x0, y0 = np.random.randint(0, W-w), np.random.randint(0, H-h)
    #print(x0, y0)
    
    ## black out on the image and mask
    image[y0:(y0+h), x0:(x0+w)] = 0
    mask[:, y0:(y0+h), x0:(x0+w)] = 0
    
    return image, mask
    

