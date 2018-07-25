# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function
import matplotlib.pyplot as plt

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
import torch
import torch.optim

from torch.autograd import Variable
from utils.inpainting_utils import *

from PIL import Image

import pandas as pd
import cv2
import scipy.misc
from PIL import Image

import matplotlib.pyplot as plt

import logging

def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def allstart(ori_img_path,pol_img_path,mask_path,recover_img_path,pic_num):
	
	
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark =True
	dtype = torch.cuda.FloatTensor

	#PLOT = True
	PLOT = False
	imsize=-1
	#dim_div_by = 64
	dim_div_by =32
	dtype = torch.cuda.FloatTensor


	img_path  = pol_img_path
	NET_TYPE = 'skip_depth6' # one of skip_depth4|skip_depth2|UNET|ResNet



	img_pil, img_np = get_image(img_path, imsize)
	img_mask_pil, img_mask_np = get_image(mask_path, imsize)



	img_mask_pil = crop_image(img_mask_pil, dim_div_by)
	img_pil      = crop_image(img_pil,      dim_div_by)
	img_np      = pil_to_np(img_pil)
	img_mask_np = pil_to_np(img_mask_pil)

	img_mask_var = np_to_var(img_mask_np).type(dtype)
	#plot_image_grid([img_np, img_mask_np, img_mask_np*img_np], 3,11)


	pad = 'reflection' # 'zero'
	OPT_OVER = 'net'
	OPTIMIZER = 'adam'


	if True:
	    INPUT = 'noise'
	    input_depth = 32
	    LR = 0.01 
	    num_iter = 100000	
	    param_noise = False
	    show_every = 20000  
	    figsize = 5
	    
	    net = skip(input_depth, img_np.shape[0], 
		       num_channels_down = [16, 32, 64, 128, 128],
		       num_channels_up =   [16, 32, 64, 128, 128],
		       num_channels_skip =    [0, 0, 0, 0, 4],  
		       filter_size_up = 7, filter_size_down = 7, 
		       upsample_mode='nearest', filter_skip_size=1,
		       need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)


	net = net.type(dtype)
	net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)


	# Compute number of parameters
	s  = sum(np.prod(list(p.size())) for p in net.parameters())
	print ('Number of params: %d' % s)

	# Loss
	mse = torch.nn.MSELoss().type(dtype)

	img_var = np_to_var(img_np).type(dtype)
	mask_var = np_to_var(img_mask_np).type(dtype)


	i = 0
	def closure():
		global i
	    
		if param_noise:
			for n in [x for x in net.parameters() if len(x.size()) == 4]:
				n.data += n.data.clone().normal_()*n.data.std()/50
	    
		out = net(net_input)
	   
		total_loss = mse(out * mask_var, img_var*mask_var)
		total_loss.backward()
		
		print ('Iteration %05d    Loss %f' % (i, total_loss.data[0]), '\r', end='')
		if  PLOT and i % show_every == 0:
			out_np = var_to_np(out)
			plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
		
		i += 1

		return total_loss

	#print('picture number:',pic_num)
	logging.info('picture number:%s'%pic_num)

	p = get_params(OPT_OVER, net, net_input)
	optimize(OPTIMIZER, p, closure, LR, num_iter)


	out_np = var_to_np(net(net_input))
	r=out_np[0,:,:]
	g=out_np[1,:,:]
	b=out_np[2,:,:]
	r=r*255
	g=g*255
	b=b*255
	out=cv2.merge([b,g,r])
	cv2.imwrite(recover_img_path, out)
	

#ori_psnr=[]
#op_psnr=[]

#first_target:70
#reoptimize_target:100


if __name__=='__main__':
	logfile='/home/jjlin/Desktop/image_inpainting/dataset/trainlog.log'
	trainlog(logfile)
	logging.info('now the target loss is 70')
	for i in range(1,1003):
		ori_img_path    ='/home/jjlin/Desktop/image_inpainting/dataset/stack/test/%i.jpg'%i
		pol_img_path    = '/home/jjlin/Desktop/image_inpainting/dataset/stack/test_pol/%i.jpg'%i
		mask_path       = '/home/jjlin/Desktop/image_inpainting/dataset/dip/mask/%i.jpg'%i
		recover_img_path = '/home/jjlin/Desktop/image_inpainting/dataset/dip/recover_images/%i.jpg'%i
		allstart(ori_img_path,pol_img_path,mask_path,recover_img_path,i)

	






