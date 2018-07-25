# coding=utf8
#########################################################################
# File Name: data.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: Sat 03 Feb 2018 11:58:12 PM CST
#########################################################################

import os
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import interpolation
from PIL import Image
import sys
sys.path.append('..')
import traceback
from tools import plot
from analysis import add_stain

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class DataBowl(Dataset):
    def __init__(self, filelist, phase='train'):
        super(DataBowl, self).__init__()
        self.phase = phase
        self.filelist = filelist
        self.loader = default_loader
        
    def __getitem__(self, idx, split=None):
        clean_file = self.filelist[idx]
        stain_file = clean_file.replace('trainB','trainA').replace('testB','testA').replace('.jpg','_.jpg')

        # label
        clean = self.loader(clean_file)
        if os.path.exists(stain_file):
            stain = self.loader(stain_file)
            if self.phase == 'prepare':
                clean = np.array(clean)
                stain = np.array(stain)
                mask = np.ones(stain.size, dtype=np.float32) * -1
                clean = (clean.transpose([2,0,1]).astype(np.float32) - 128) / 100
                stain = (stain.transpose([2,0,1]).astype(np.float32) - 128) / 100
                return clean, stain, mask, clean_file
            data = augment(clean, idx, stain, clean_file, phase=self.phase)
        else:
            data = augment(clean, idx, phase=self.phase)
        clean, stain, del_stain, mask, pred_mask = data
        clean = (clean.transpose([2,0,1]).astype(np.float32) - 128) / 100
        stain = (stain.transpose([2,0,1]).astype(np.float32) - 128) / 100
        del_stain = (del_stain.transpose([2,0,1]).astype(np.float32) - 128) / 100
        stain = np.concatenate((stain,del_stain, pred_mask.reshape([1, 250,250])),0).astype(np.float32)

        return clean, stain, mask, clean_file, np.float32(1), np.float32(0)

    def __len__(self):
        return len(self.filelist)

def transform(image, theta, flip, x1,y1, phase):
    if phase != 'train':
        return np.array(image)
    if theta > 0:
        image = image.rotate(theta)
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    image = image.resize((286,286))
    image = image.crop((x1,y1,x1+250,y1+250))
    return np.array(image).astype(np.float32)


def augment(clean,idx, stain=None, clean_file=None, phase='train'):
    if np.random.random() > 0.2:
        theta = int(np.random.random() * 45)
    else:
        theta = 0
    flip = np.random.random() > 0.5
    x1 = int(np.random.random() * 30)
    y1 = int(np.random.random() * 30)
    clean = transform(clean, theta, flip, x1,y1, phase)

    if stain is None:
        stain, mask = add_stain.add_stain(clean, idx)
        del_stain, mask = add_stain.del_stain(copy.deepcopy(stain),None,mask)
        return clean, stain, del_stain, mask.astype(np.float32)
    else:
        last_pred_file = clean_file.replace('../data/','../data/pred_clean/').replace('.jpg','.png')
        last_mask_file = clean_file.replace('../data/','../data/pred_mask/').replace('.jpg','.npy')
        try:
            last_pred = Image.open(last_pred_file)
        except:
            # traceback.print_exc()
            # 没有last_pred, 用stain替代
            last_pred = copy.deepcopy(stain)
        try:
            # last_mask = Image.open(last_mask_file)
            # last_mask = transform(last_mask, theta, flip, x1,y1, phase)
            # last_mask = last_mask[:,:,0] / 255.
            last_mask = np.load(last_mask_file)
        except:
            # traceback.print_exc()
            # 没有last_mask, 用0替代
            last_mask = np.zeros(clean.shape[:2])
        stain = transform(stain, theta, flip, x1,y1, phase)


        diff = np.abs(clean.astype(np.float32) - stain.astype(np.float32)).mean(2)
        mask = np.ones_like(diff, dtype=np.float32) * - 1
        mask[diff>6] = 1
        mask[diff<2] = 0
        if phase == 'del':
            del_stain = transform(last_pred, theta, flip, x1,y1, phase)
            # del_stain, _ = add_stain.del_stain(copy.deepcopy(stain),None,mask)
        else:
            del_stain = transform(last_pred, theta, flip, x1,y1, phase)
        # print mask.max()
        return clean, stain, del_stain, mask, last_mask
