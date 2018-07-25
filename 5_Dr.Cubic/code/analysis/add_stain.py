# coding=utf8
#########################################################################
# File Name: add_stain.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2018年03月23日 星期五 10时25分06秒
#########################################################################
import os
import sys
sys.path.append('..')
import numpy as np
from PIL import Image
import compare_stain
import matplotlib.pyplot as plt
import copy
import traceback

def del_stain(stain_img, pred, mask, vis = 0.01):
    if pred is not None:
        return pred, mask
    else:
        return stain_img, mask
    prob = copy.deepcopy(mask)
    prob = sorted(prob.reshape(-1))
    vis = prob[int(len(prob)*0.9)]
    vis = min(0.99, vis)
    index_set = set(range(256))
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x,y] > vis:
                # stain_img[x,y] = [255,255,255]
                # continue
                clean_x = []
                clean_y = []
                stain_num = 0
                for r in range(1,20):
                    for d in range(-r,r):
                        p1 = [x+r,y+d]
                        p2 = [x-r,y+d]
                        p3 = [x+r,y-d]
                        p4 = [x-r,y-d]
                        for xi,yi in [p1,p2,p3,p4]:
                            if xi in index_set and yi in index_set:
                                if mask[xi,yi] > vis:
                                    stain_num += 1
                                else:
                                    clean_x.append(xi)
                                    clean_y.append(yi)
                    if len(clean_x) > stain_num and r > 1:
                        break
                color = stain_img[clean_x,clean_y].mean(0).astype(np.uint8)
                stain_img[x,y] = color
    return stain_img, mask


def add_stain(img='../../data/AI/trainB/0000045_001.jpg', idx=0, mask_folder = '../data/mask/'):
    if isinstance(img, str):
        img = Image.open(img)
        img = img.resize((256,256))
        img = np.array(img)
    
    if not os.path.exists(mask_folder):
        os.mkdir(mask_folder)
    mask_files = [f for f in os.listdir(mask_folder) if '.npy' in f]
    if len(mask_files) > 0:
        if np.random.random() > 0.8:
            mask = get_stain([256,256], idx, mask_folder)
        else:
            n = int(np.random.random() * len(mask_files))
            mask_file = os.path.join(mask_folder, mask_files[n])
            while True:
                try:
                    mask = np.load(mask_file)
                    break
                except:
                    traceback.print_exc()
                    continue
    else:
        mask = get_stain([256,256], idx, mask_folder)
    stain_img = copy.deepcopy(img)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x,y] > 0:
                c = int(np.random.random() * 90)
                stain_img[x,y,:] = c
            else:
                stain_img[x,y,:] = img[x,y,:]
    # stain_img, mask = del_stain(stain_img, mask)
    # compare_stain.compare_stain_img(stain_img, img)
    return stain_img, mask

def get_stain(img_size,idx, tmp_folder):
    hx, hy = [xy/2 for xy in img_size]
    tmp_file = os.path.join(tmp_folder,'tmp_'+ str(idx) + '.png')
    x = np.arange(0,10*np.pi,0.1)
    y = np.sin(x)
    fig = plt.figure(edgecolor='m')
    dy = 1.5 + np.random.random()
    dx = np.random.random() * np.pi / 4
    w = 0.3 + np.random.random() * 3
    n = np.random.random() * 20 + 6
    for i in np.arange(0,n,dy):
        plt.plot(x,y+i,color='b', linewidth=w)
        plt.plot(x+np.pi+dx,y+i,color='b',linewidth=w)
    plt.xticks([])
    plt.yticks([])
    fig.savefig(tmp_file)
    plt.close()

    img = Image.open(tmp_file)
    theta = int(np.random.random() * 180)
    img = img.rotate(theta)
    img = img.crop((320-hx,240-hy,320+hx,240+hy))
    img.save(tmp_file)

    img = img.convert('RGB')
    img = np.array(img.getdata())[:,0]
    vi = 120
    img[img<vi] = 0
    img[img>=vi] = 1
    img = 1 - img
    img = img.reshape([hx*2,hy*2])
    # os.remove(tmp_file)
    tmp_file = os.path.join(tmp_folder,'tmp_'+ str(idx/10) + '.npy')
    np.save(tmp_file,img.astype(np.uint8))
    return img

if __name__ == '__main__':
    # add_stain(mask_folder = '../../data/mask')

    img = '../../data/pred_mask/AI/trainB/0000099_076_stain.png'
    mask = img.replace('jpg','.npy').replace('_stain.png','.npy')
    img = Image.open(img)
    print img.size
    img = np.array(img)
    print img.shape
    stain_img = copy.deepcopy(img)
    mask = np.load(mask)
    print mask.shape, mask.max()
    '''
    xmask = (mask * 256).astype(np.uint8)
    xmask = np.array([xmask,xmask,xmask]).transpose([1,2,0])
    compare_stain.compare_stain_img(stain_img, xmask)
    '''
    stain_del, mask = del_stain(img,mask, 0.001)
    compare_stain.compare_stain_img(stain_img, stain_del)

