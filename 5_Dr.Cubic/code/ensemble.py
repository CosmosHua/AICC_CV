# coding=utf8
#########################################################################
# File Name: ensemble.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2018年03月28日 星期三 01时08分52秒
#########################################################################
import os
import traceback
import numpy as np
from PIL import Image
from tools import py_op, measures
import multiprocessing

def write_ensemble_image(i,file_name, pred_dir_list, obj_dir, rgb_prob, level, pred_dict):
        print 'start preprocessing', i, file_name
        pred_list = []
        psnr_ratio_list = []
        for pred_dir in pred_dir_list:
            pred_file = os.path.join(pred_dir, 'AI/testB/', file_name).replace('jpg','png')
            pred_img = np.array(Image.open(pred_file))
            pred_list.append(pred_img)
            psnr_ratio= pred_dict[pred_dir.strip('/').split('/')[-1]]
            psnr_ratio_list.append(psnr_ratio)
        # print pred_dir
        pred_list = np.array(pred_list)
        img_new = np.zeros(pred_list.shape[1:], dtype=np.uint8)
        for x in range(pred_list.shape[1]):
            for y in range(pred_list.shape[2]):
                m = pred_list[:,x,y,:]
                r_new, g_new, b_new = 0.,0.,0.
                prob_sum = 0
                for mi in range(m.shape[0]):
                    r,g,b = m[mi]
                    prob = max(10e-10, rgb_prob[r/level,g/level,b/level])
                    prob = prob * np.exp(psnr_ratio_list[mi]*1)
                    prob_sum += prob 
                    r_new += r * prob
                    g_new += g * prob
                    b_new += b * prob
                r_new = r_new / prob_sum
                g_new = g_new / prob_sum
                b_new = b_new / prob_sum
                img_new[x,y] = np.array([r_new, g_new, b_new], dtype=np.uint8)
        file_new = os.path.join(obj_dir, 'AI/testB/', file_name.replace('jpg', 'png'))
        print pred_dir, file_new
        py_op.mkdir(os.path.dirname(file_new))
        Image.fromarray(img_new).save(file_new)
        # print 'end preprocessing', i, file_name

def ensemble(level=2):
    test_clean = '../data/test_clean/'
    clean_dir = '../data/AI/testB/'
    rgb_prob = np.load('../data/rgb_stati/rgb_prob_{:d}.npy'.format(level))
    obj_dir = 'ensemble_1'
    pred_dict = py_op.myreadjson('../data/result/result.json')
    file_names = os.listdir(clean_dir)
    # pred_dir_list = [os.path.join(test_clean,d) for d in os.listdir(test_clean) if obj_dir not in d and pred_dict.get(d,0)>0.94 and pred_dict.get(d,0)<0.95]
    pred_dir_list = [os.path.join(test_clean,d) for d in os.listdir(test_clean) if obj_dir not in d and pred_dict.get(d,0)>0.94]
    if len(pred_dir_list) == 0:
        return
    # print pred_dir_list
    # return

    pool = multiprocessing.Pool(processes=15)
    for fi,file_name in enumerate(os.listdir(clean_dir)):
        pool.apply_async(write_ensemble_image, (fi, file_name,pred_dir_list, os.path.join(test_clean, obj_dir), rgb_prob, level, pred_dict))
        # write_ensemble_image(fi, file_name,pred_dir_list, os.path.join(test_clean, obj_dir), rgb_prob, level, pred_dict)
    pool.close()
    pool.join()
    print 'processed all'


def test_all():
    test_clean = '../data/test_clean'
    try:
        pred_dict = py_op.myreadjson('../data/result/result.json')
    except:
        pred_dict = dict()
    for i,pred_clean in enumerate(os.listdir(test_clean)):
        if pred_clean in pred_dict:
            if pred_dict[pred_clean] < 0.85:
                os.system('rm -r {:s}'.format(os.path.join(test_clean, pred_clean)))
            continue
        result = measures.compute_pred_clean_psnr(pred_clean,'../data/AI/testB', '../data/result')
        if result < 0.88:
            os.system('rm -r {:s}'.format(os.path.join(test_clean, pred_clean)))
        pred_dict[pred_clean] = result
    pred_dict = py_op.mysorteddict(pred_dict, key=lambda s:pred_dict[s])
    py_op.mywritejson('../data/result/result.json',pred_dict)


            




if __name__ == '__main__':
    # test_all()
    ensemble()
