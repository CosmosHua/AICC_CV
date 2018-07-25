#########################################################################
# File Name: train_3.sh
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2018年03月26日 星期一 16时26分26秒
#########################################################################
#!/bin/bash
cd ../code
CUDA_VISIBLE_DEVICES=0 python main.py --input-filter 3 -b 16
