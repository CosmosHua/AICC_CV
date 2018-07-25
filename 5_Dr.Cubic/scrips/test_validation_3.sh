#########################################################################
# File Name: generate_pred_3.sh
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2018年03月26日 星期一 14时13分39秒
#########################################################################
#!/bin/bash
cd ../code
CUDA_VISIBLE_DEVICES=0 python main.py --input-filter 3 --resume ../data/model/default_3/best_psnr/best_psnr_3.ckpt --write-pred 1 --test 1 -b 2
