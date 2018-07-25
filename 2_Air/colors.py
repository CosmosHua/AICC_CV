# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:07:01 2018

@author: yun_yang
"""

import colorsys  
import cv2
import numpy as np

def color_moments(filename):
    img = cv2.imread(filename)
    if img is None:
        return
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average 
    h_mean = np.mean(h).astype('uint8')  # np.sum(h)/float(N)
    s_mean = np.mean(s).astype('uint8')  # np.sum(s)/float(N)
    v_mean = np.mean(v).astype('uint8')  # np.sum(v)/float(N)

    # The second central moment - standard deviation
    h_std = np.std(h).astype('uint8')  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s).astype('uint8')  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v).astype('uint8')  # np.sqrt(np.mean(abs(v - v.mean())**2))

    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = int(h_skewness**(1./3))
    s_thirdMoment = int(s_skewness**(1./3))
    v_thirdMoment = int(v_skewness**(1./3))
    color_feature.extend([h_mean, h_std, h_thirdMoment, s_mean, s_std, s_thirdMoment,  v_mean, v_std, v_thirdMoment])
    return color_feature

def isGray(file_path):
    label = False
    mo = color_moments(imgP)
    if mo[3] > 165 and mo[6] > 105:
        if mo[7] > 30 and mo[7] < 70 and mo[8] > 30 and mo[8] < 70:
            if mo[4] > 24 and mo[4] < 45 and mo[5] > 24 and mo[5] < 50:
                if mo[1] > 20 and mo[1] < 50 and mo[2] > 20 and mo[2] < 50:
                    label = True
    return label