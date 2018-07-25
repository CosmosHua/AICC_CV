# coding=utf8
#########################################################################
# File Name: show_poor_prediction.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2018年04月10日 星期二 18时01分12秒
#########################################################################

import os
import sys
sys.path.append('../')
from tools import py_op
from PIL import Image

def scp_files(json_file):
    result_dict = py_op.myreadjson(json_file)
    score_jpg_list = sorted([(v,k) for k,v in result_dict.items()])
    for score, jpg in score_jpg_list[:20]:
        png = jpg.replace('jpg','png')
        # png =  + png
        image = Image.open('../../data/pred_clean/AI/testB/' + png)
        image = image.resize((250,250))
        tmp_png = 'tmp.png'
        image.save(tmp_png)
        cmd = 'scp {:s} ycclab:tmp/bad/{:s}'.format(tmp_png, png)
        os.system(cmd)
        os.remove(tmp_png)
        cmd = 'scp ../../data/AI/testB/{:s} ycclab:tmp/bad/'.format(jpg)
        os.system(cmd)
        cmd = 'scp ../../data/AI/testA/{:s} ycclab:tmp/bad/{:s}'.format(jpg.replace('.jpg','_.jpg'), jpg.replace('.jpg','.qs.jpg'))
        os.system(cmd)
        cmd = 'scp ../../data/pred_mask/AI/testB/{:s} ycclab:tmp/bad/{:s}'.format(png, png.replace('.png','.rm.png'))
        os.system(cmd)



if __name__ == '__main__':
    scp_files('../../data/result/7_311.json')
