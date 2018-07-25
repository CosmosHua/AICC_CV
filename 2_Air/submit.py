# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:22:15 2018

@author: yun_yang
"""

from glob import glob
from scipy.misc import imread, imsave
from skimage.color import rgb2gray
import shutil
import numpy
import math
import numpy as np
from skimage.measure import compare_psnr

#
avg_psnr1 = 0
avg_psnr2 = 0
count = 0
total = 0
score = 0
avescore = 0

import pandas as pd

lists = []
imglist = glob('a/*.jpg')
for imgP in imglist:
    oimgP = imgP.replace('a', 'b')
    
    iimg = imread(imgP) / 255.
    oimg = imread(oimgP) / 255.
    pimg = imread(imgP.replace('a', 'result')) / 255.
    
    imgName = imgP.split('\\')[-1].split('.')[0]
    
    if len(iimg.shape) == 3 and len(oimg.shape) == 3:
        count += 1
        psnr1 = compare_psnr(oimg, iimg)
        psnr2 = compare_psnr(pimg, iimg)
        avg_psnr1 += psnr1
        avg_psnr2 += psnr2
        score = (psnr2 - psnr1) / psnr1
        print(imgName, psnr1, psnr2, score)
        lists.append([imgName, psnr1, psnr2, score])
        avescore += score
    if len(iimg.shape) == 2 and len(oimg.shape) == 3:
        print('---'*20, 'gray image')
        pimg = rgb2gray(pimg)
        oimg = rgb2gray(oimg)
        avg_psnr1 += psnr1
        avg_psnr2 += psnr2
        score = (psnr2 - psnr1) / psnr1
        print(imgName, psnr1, psnr2, score)
        lists.append([imgName, psnr1, psnr2, score])
        pimg = imread(imgP.replace('a', 'result'))
        imsave('./{}.jpg'.format(total),rgb2gray(pimg))
        avescore += score
    total += 1    
df = pd.DataFrame(lists)
df.columns = ['id', 'psnr1', 'psnr2', 'score']
df.to_csv('submit.csv', index=False, header=False)

print('='*20 + '>')
print(avescore/total, total, total)
