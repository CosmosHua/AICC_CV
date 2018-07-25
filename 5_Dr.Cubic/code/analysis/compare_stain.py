# coding=utf8
#########################################################################
# File Name: compare_stain.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2018年03月23日 星期五 09时49分50秒
#########################################################################
import os
import sys
from PIL import Image
import numpy as np
import traceback

sys.path.append('../')
from tools import plot


trainA_dir = '../../data/AI/testA/'
trainB_dir = '../../data/AI/testB/'
pred_mask_dir = '../../data/pred_mask/AI/testB/'
pred_clean_dir = '../../data/pred_clean/AI/testB/'



def compare_stain_dir(trainA_dir, trainB_dir):
    for fi in os.listdir(trainA_dir):
        fi_a = os.path.join(trainA_dir, fi)
        fi_b = os.path.join(trainB_dir, fi.replace('_.','.'))
        # print fi_a
        img_a = np.array(Image.open(fi_a))
        img_b = np.array(Image.open(fi_b))
        compare_stain_img(img_a, img_b)
        return


def compare_stain_img(img_a, img_b):
    # img_a = img_a[:,:,0]
    # img_b = img_b[:,:,0]
    img_stain = img_a
    img_a = img_a.mean(2)
    img_b = img_b.mean(2)

    img_a = img_a.astype(np.int32)
    img_b = img_b.astype(np.int32)

    same = np.array(img_b - img_a, dtype=img_a.dtype)
    sub = np.array(img_b - img_a > 6, dtype=img_a.dtype)
    mean = (img_b + img_a)/2

    stain_r = img_stain[:,:,0][sub>0]
    stain_g = img_stain[:,:,1][sub>0]
    stain_b = img_stain[:,:,2][sub>0]
    stain_m = img_a[sub>0]
    # print stain_r.max(), stain_r.min(), stain_r.mean()
    # print stain_g.max(), stain_g.min(), stain_g.mean()
    # print stain_b.max(), stain_b.min(), stain_b.mean()
    # print stain_m.max(), stain_m.min(), stain_m.mean()

    plot.plot_multi_graph([same , sub,mean,img_a, img_b], ['same', 'sub','mean','a','b'], show=True)
    # plot.plot_multi_graph([stain_r, stain_g, stain_b, stain_m], ['r', 'g','b','m'], show=True)

def merge_stain_pred(trainA_dir, trainB_dir, pred_mask_dir, pred_clean_dir, save_dir):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for fi in os.listdir(trainA_dir):
        fi_a = os.path.join(trainA_dir, fi)
        fi_b = os.path.join(trainB_dir, fi.replace('_.','.'))
        fi_m = os.path.join(pred_mask_dir, fi.replace('_.jpg','.png'))
        fi_p = os.path.join(pred_clean_dir, fi.replace('_.jpg','.png'))

        img_a = np.array(Image.open(fi_a).resize((256,256)))
        img_b = np.array(Image.open(fi_b).resize((256,256)))
        img_m = np.array(Image.open(fi_m).resize((256,256))).astype(np.float32) / 256
        img_p = np.array(Image.open(fi_p).resize((256,256)))

        fi = fi.replace('_.jpg','')
        img_m = (img_a * (1 - img_m) + img_p * img_m).astype(np.uint8)
        diff = np.abs(img_a - img_b)
        vis = 10
        img_a[diff>vis] = img_b[diff>vis]
        Image.fromarray(img_b).save(os.path.join(save_dir,fi + '_1_clean.png'))
        Image.fromarray(img_p).save(os.path.join(save_dir,fi + '_2_pred.png'))
        Image.fromarray(img_m).save(os.path.join(save_dir,fi + '_3_merge_pred.png'))
        Image.fromarray(img_a).save(os.path.join(save_dir,fi + '_4_merge_clean.png'))

        break

def stati_mse_psnr_contribution(trainA_dir, trainB_dir):
    mse_dict = dict()
    psnr_dict = dict()

    gray_num = 10
    diff_list = np.zeros(gray_num + 1)

    for i in range(gray_num):
        mse_dict[i] = 0

    for j,fi in enumerate(os.listdir(trainA_dir)):
        fi_a = os.path.join(trainA_dir, fi)
        fi_b = os.path.join(trainB_dir, fi.replace('_.','.'))

        try:
            img_a = np.array(Image.open(fi_a)).mean(2).reshape(-1).astype(np.float32)
            img_b = np.array(Image.open(fi_b)).mean(2).reshape(-1).astype(np.float32)
            # img_a = np.array(Image.open(fi_a)).reshape(-1).astype(np.float32)
            # img_b = np.array(Image.open(fi_b)).reshape(-1).astype(np.float32)
            diff = np.abs(img_a - img_b).reshape(-1)

        except:
            traceback.print_exc()
            continue

        for i in range(gray_num):
            diff_list[i] += (diff==i).sum()

        diff_list[gray_num] += (diff>=gray_num).sum()

        print j,len(os.listdir(trainA_dir))

    diff_list = diff_list/sum(diff_list)
    mse_list = [i*i*diff_list[i] for i in range(len(diff_list))]
    for i in range(gray_num+1):
        diff_list[i] = diff_list[i:].sum()

    mse_sum = [0]
    for i in range(1,len(diff_list)):
        # mse_sum.append(sum(mse_list[:i]) + diff_list[i-1]*i*i)
        mse_sum.append(sum(mse_list[:i+1]))



    print mse_sum[:6]
    # gray_num += 1
    plot.plot_multi_line([range(gray_num), range(gray_num), range(gray_num)],[mse_list[:gray_num], diff_list[:gray_num], mse_sum[:gray_num]], ['mse', 'diff','mse_sum'], show=True)

    return



if __name__ == '__main__':
    # compare_stain_dir(trainA_dir, trainB_dir)
    # merge_stain_pred(trainA_dir, trainB_dir, pred_mask_dir, pred_clean_dir, '../../data/tmp/')
    stati_mse_psnr_contribution(trainA_dir, trainB_dir)


