# coding=utf8
import sys
import os
import numpy as np
from PIL import Image
import traceback
reload(sys)
sys.setdefaultencoding('utf8')
from tqdm import tqdm


sys.path.append('..')
from tools import plot, py_op, measures


'''
统计函数说明:
    stati_mse_diff_distribution: 统计生成图片中仍然存在的MSE主要由哪些区域产生
    stati_pixel_gray:
    stati_pixel_rgb
'''

def stati_pixel_rgb(level):
    clean_dir = '../../data/AI/trainB/'
    rgb_number = np.zeros((256/level, 256/level, 256/level), dtype=np.float32)
    for i,clean in enumerate(os.listdir(clean_dir)):
        clean = os.path.join(clean_dir, clean)
        clean = np.array(Image.open(clean))
        try:
            for x in range(clean.shape[0]):
                for y in range(clean.shape[1]):
                    r,g,b = clean[x,y,:] / level
                    rgb_number[r,g,b] += 1
        except:
            traceback.print_exc()
            continue
        print i, len(os.listdir(clean_dir))
        if i > 1000:
            break
    rgb_number = rgb_number / rgb_number.sum() * 256 * 256 / level / level
    print rgb_number.max()
    print rgb_number.min()
    print rgb_number.mean()
    np.save('../../data/rgb_stati/rgb_prob_{:d}.npy'.format(level),rgb_number)

def stati_pixel_gray():
    clean_dir = '../../data/AI/testB/'
    clean_num = np.zeros(256)
    stain_num = np.zeros(256)
    diff_num = np.zeros(256)
    for i,clean in enumerate(os.listdir(clean_dir)):
        clean = os.path.join(clean_dir, clean)
        stain = clean.replace('trainB','trainA').replace('testB','testA').replace('.jpg','_.jpg')
        try:
            clean = np.array(Image.open(clean)).mean(2).astype(np.int32).reshape(-1)
            stain = np.array(Image.open(stain)).mean(2).astype(np.int32).reshape(-1)
        except:
            continue
        for x in clean:
            clean_num[x] += 1
        for x in stain:
            stain_num[x] += 1
        print clean.dtype
        print i,len(os.listdir(clean_dir))
        diff = np.abs(clean - stain).astype(np.int32)
        for x in diff:
            diff_num[x] += 1
        # if i> 10:
        #     break
    clean_num = clean_num / clean_num.sum()
    stain_num = stain_num / stain_num.sum()
    diff_num = 1.*diff_num/diff_num.sum()
    wf = open('../../data/stati/distance_percent.csv','w')
    for i in range(256):
        wf.write(str(i))
        wf.write('\t')
        wf.write(str(diff_num[i]))
        wf.write('\n')
    wf.close()
    diff_num = stain_num - clean_num

    # plot.plot_multi_line([range(256), range(256), range(256)], [clean_num, stain_num, diff_num], ['clean', 'stain', 'diff'],show=True)

def stati_pixel_same():
    '''
    找不同灰度距离的点数量
    '''
    clean_dir = '../../data/AI/testB/'
    same_num = np.zeros(256) 
    n_same = 0.
    n_diff = 1.
    dis_num_dict = dict()
    for i,clean in enumerate(os.listdir(clean_dir)):
        clean = os.path.join(clean_dir, clean)
        stain = clean.replace('trainB','trainA').replace('testB','testA').replace('.jpg','_.jpg')
        try:
            clean = np.array(Image.open(clean)).mean(2).astype(np.int32).reshape(-1)
            stain = np.array(Image.open(stain)).mean(2).astype(np.int32).reshape(-1)
        except:
            continue
        diff = np.abs(clean - stain) 
        for j in range(10):
            try:
                dis_num_dict[j] += np.sum(diff==j)
            except:
                dis_num_dict[j] = np.sum(diff==j)
        dis_num_dict[10] = np.sum(diff>9)
        vis = 7
        n_diff += (diff >  vis + 1).sum()
        n_same += (diff <= vis + 1).sum()
        
        for x,y in zip(clean, stain):
            diff = int(x) - y 
            if diff * diff > 5 :
                same_num[x] += 1
                n_diff += 1
            else:
                n_same += 1
        if i > 100:
            break
        print i,len(os.listdir(clean_dir))
    same_num = same_num/ same_num.sum()
    # plot.plot_multi_line([range(256)], [same_num], ['clean', 'stain', 'diff'],show=True)
    print  n_diff/(n_same + n_diff)
    print
    for j,n in dis_num_dict.items():
        print j,float(n)/np.sum(dis_num_dict.values())

def get_mask(diff, threshhold_list, use_max):
    if use_max:
        diff = diff.max(2)
    else:
        diff = diff.mean(2)
    mask = np.zeros_like(diff)
    for i,threshhold in enumerate(threshhold_list):
        mask[diff>threshhold] = i
    return mask


def stati_mse_diff_distribution(clean_dir, stain_dir, pred_dir, stati_bad=False, result_json='../../data/result/7_311.json', use_max=False):
    '''
    统计生成图片中仍然存在的MSE主要由哪些区域产生
    分为多个区域 以threshhold为界
    stati_bad: 仅仅统计生成较差的图片结果
    result_json: 预测结果统计json文件
    '''
    if stati_bad:
        result_dict = py_op.myreadjson(result_json)
        result_list = sorted([(v,k) for k,v in result_dict.items()])[:int(0.1*len(result_dict))]
        result_list = set([k.split('.')[0] for v,k in result_list])
    threshhold_list = [0,2,6,20,60,100,150,250]
    mse_sum = np.zeros(len(threshhold_list))
    mask_sum = np.zeros(len(threshhold_list))
    for fi in tqdm(os.listdir(pred_dir)):
        if stati_bad:
            if fi.split('.')[0] not in result_list:
                continue
        pred_fi = os.path.join(pred_dir, fi)
        clean_fi = os.path.join(clean_dir, fi.replace('png','jpg'))
        stain_fi = os.path.join(stain_dir, fi.replace('.png','_.jpg'))
        try:
            pred_image = np.array(Image.open(pred_fi).resize((250,250))).astype(np.float32)
            clean_image = np.array(Image.open(clean_fi).resize((250,250))).astype(np.float32)
            stain_image = np.array(Image.open(stain_fi).resize((250,250))).astype(np.float32)
            mask = get_mask(np.abs(clean_image.astype(np.float32) - stain_image), threshhold_list, use_max)
        except:
            continue
        for n in range(len(threshhold_list)):
            mask_sum[n] += (mask==n).sum()
            mse_sum[n] += ((clean_image[mask==n] - pred_image[mask==n]) ** 2).sum()
    print '生成的图片主要的mse分布'
    print '灰度差别 \t 区域占比例 \t mse占比例'
    for n in range(len(threshhold_list)-1):
        threshhold_min = threshhold_list[n]
        threshhold_max = threshhold_list[n+1]
        print '[{:d}, {:d}]   \t {:2.2f} \t\t {:2.2f}'.format(threshhold_min, threshhold_max, mask_sum[n]/sum(mask_sum), mse_sum[n]/sum(mse_sum))

def stati_mse_gray_distribution(clean_dir, stain_dir, pred_dir, stati_bad=False, result_json='../../data/result/7_311.json', use_max=False):
    '''
    统计生成图片中仍然存在的MSE主要由哪些区域产生
    '''
    if stati_bad:
        result_dict = py_op.myreadjson(result_json)
        result_list = sorted([(v,k) for k,v in result_dict.items()])[:int(0.1*len(result_dict))]
        result_list = set([k.split('.')[0] for v,k in result_list])
    threshhold_list = [0, 18, 60,100,150,250]
    mse_sum = np.zeros(len(threshhold_list))
    mask_sum = np.zeros(len(threshhold_list))
    for fi in tqdm(os.listdir(pred_dir)):
        if stati_bad:
            if fi.split('.')[0] not in result_list:
                continue
        pred_fi = os.path.join(pred_dir, fi)
        clean_fi = os.path.join(clean_dir, fi.replace('png','jpg'))
        stain_fi = os.path.join(stain_dir, fi.replace('.png','_.jpg'))
        try:
            pred_image = np.array(Image.open(pred_fi).resize((250,250))).astype(np.float32)
            clean_image = np.array(Image.open(clean_fi).resize((250,250))).astype(np.float32)
            stain_image = np.array(Image.open(stain_fi).resize((250,250))).astype(np.float32)
            mask = get_mask(stain_image, threshhold_list, use_max)
        except:
            continue
        for n in range(len(threshhold_list)):
            mask_sum[n] += (mask==n).sum()
            mse_sum[n] += ((clean_image[mask==n] - pred_image[mask==n]) ** 2).sum()
    print '生成的图片主要的mse分布'
    print '原图灰度 \t 区域占比例 \t mse占比例'
    for n in range(len(threshhold_list)-1):
        threshhold_min = threshhold_list[n]
        threshhold_max = threshhold_list[n+1]
        print '[{:d}, {:d}]   \t {:2.2f} \t\t {:2.2f}'.format(threshhold_min, threshhold_max, mask_sum[n]/sum(mask_sum), mse_sum[n]/sum(mse_sum))

def stati_gray_stain(clean_dir, stain_dir, pred_dir, stati_bad=True, result_json='../../data/result/7_311.json', use_max=False):
    '''
    统计不同灰度下的网纹点比例
    '''
    if stati_bad:
        result_dict = py_op.myreadjson(result_json)
        result_list = sorted([(v,k) for k,v in result_dict.items()])[:int(0.1*len(result_dict))]
        result_list = set([k.split('.')[0] for v,k in result_list])
    threshhold_list = [0,80,120,256]
    diff_sum = np.zeros(len(threshhold_list))
    mask_sum = np.zeros(len(threshhold_list))
    for fi in tqdm(os.listdir(pred_dir)):
        if stati_bad:
            if fi.split('.')[0] not in result_list:
                continue
        pred_fi = os.path.join(pred_dir, fi)
        clean_fi = os.path.join(clean_dir, fi.replace('png','jpg'))
        stain_fi = os.path.join(stain_dir, fi.replace('.png','_.jpg'))
        try:
            clean_image = np.array(Image.open(clean_fi).resize((250,250))).astype(np.float32)
            stain_image = np.array(Image.open(stain_fi).resize((250,250))).astype(np.float32)
            mask = get_mask(stain_image, threshhold_list, use_max)
        except:
            traceback.print_exc()
            continue
        for n in range(len(threshhold_list)):
            mask_sum[n] += (mask==n).sum()
            diff_sum[n] += (np.abs(clean_image[mask==n] - stain_image[mask==n]) > 20).sum()
    print '灰度区域 \t 区域占比例 \t 网纹点占比例'
    for n in range(len(threshhold_list)-1):
        threshhold_min = threshhold_list[n]
        threshhold_max = threshhold_list[n+1]
        print '[{:d}, {:d}]   \t {:2.2f} \t\t {:2.2f}'.format(threshhold_min, threshhold_max, mask_sum[n]/sum(mask_sum), diff_sum[n]/sum(diff_sum))
    # print diff_sum

def main():
    clean_dir = '../../data/AI/testB/'
    pred_dir = '../../data/pred_clean/AI/testB/'
    stain_dir = '../../data/AI/testA/'
    # stati_pixel_gray()
    # for level in [2, 4, 8, 16, 32, 64]:
    #     stati_pixel_rgb(level)
    #     break
    # stati_pixel_same()

    # 统计mse在不同灰度差别下的分布, 结果为mse主要分布在网纹区域,这些区域预测错误多
    stati_mse_diff_distribution(clean_dir, stain_dir, pred_dir, use_max=False, stati_bad=True)

    # 统计mse在不同灰度下的分布, 结果为mse主要分布在网纹图的[20,150]灰度区间
    stati_mse_gray_distribution(clean_dir, stain_dir, pred_dir, use_max=False, stati_bad=True)

    # 统计不同灰度下的网纹点比例, 结果为网纹点主要分布在的[0,120]灰度区间
    stati_gray_stain(clean_dir, stain_dir, pred_dir, stati_bad=True)


if __name__ == '__main__':
    main()
