# coding=utf8
#########################################################################
# File Name: delete_stain.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2018年04月10日 星期二 23时33分03秒
#########################################################################

import sys
import os
import numpy as np
from PIL import Image
import traceback
reload(sys)
sys.setdefaultencoding('utf8')
from tqdm import tqdm
from tools import plot, py_op, measures

def del_stain(stain_img, mask, vis = 0.01):
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

def delete_stain(clean_dir, pred_dir, stain_dir, mask_dir, del_dir):
    '''
    用周边点补充预测为网纹的点
    '''
    for fi in tqdm(os.listdir(pred_dir)):
        pred_fi = os.path.join(pred_dir, fi)
        clean_fi = os.path.join(clean_dir, fi.replace('png','jpg'))
        stain_fi = os.path.join(stain_dir, fi.replace('.png','_.jpg'))
        try:
            clean_image = np.array(Image.open(clean_fi).resize((250,250))).astype(np.float32)
            stain_image = np.array(Image.open(stain_fi).resize((250,250))).astype(np.float32)
            pred_image = np.array(Image.open(pred_fi).resize((250,250))).astype(np.float32)
        except:
            traceback.print_exc()
            continue
        clean_image = clean_image.mean(2)
        stain_image = stain_image.mean(2)
        pred_image = pred_image.mean(2)
        image_list = [clean_image, stain_image, pred_image]
        name_list = [' ' for _ in image_list]
        plot.plot_multi_graph(image_list, name_list, show=True)
        break

def main():
    clean_dir = '../data/AI/testB/'
    pred_dir = '../data/pred_clean/AI/testB/'
    stain_dir = '../data/AI/testA/'
    del_dir = '../data/del/AI/testB/'
    mask_dir = '../data/pred_mask/AI/testB/'
    py_op.mkdir(del_dir)
    delete_stain(clean_dir, pred_dir, stain_dir, mask_dir, del_dir)


if __name__ == '__main__':
    main()
