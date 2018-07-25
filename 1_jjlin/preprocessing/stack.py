from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import pdb
from skimage import io
#from libtiff import TIFF

#从250*250到256*256


up_and_down=np.ones((3,250))*255
left_and_right=np.ones((256,3))*255


def stack(path,new_path):
	img=cv2.imread(path)
	a=np.r_[img[:,:,0],up_and_down]
	b=np.r_[up_and_down,a]
	c=np.c_[b,left_and_right]
	d=np.c_[left_and_right,c]

	e=np.r_[img[:,:,1],up_and_down]
	f=np.r_[up_and_down,e]
	g=np.c_[f,left_and_right]
	h=np.c_[left_and_right,g]

	i=np.r_[img[:,:,2],up_and_down]
	j=np.r_[up_and_down,i]
	k=np.c_[j,left_and_right]
	l=np.c_[left_and_right,k]

	#m=np.r_[d,h,l].reshape((256,256,3))
	out=cv2.merge([d,h,l])
	
	cv2.imwrite(new_path,out)
	#pdb.set_trace()

#train_pol彩图
for i in range(1,10014):
	path='/home/jjlin/Desktop/image_inpainting/dataset/train_pol/%i.jpg'%i
	new_path='/home/jjlin/Desktop/image_inpainting/dataset/stack/train_pol/%i.jpg'%i
	stack(path,new_path)


#trace_mask黑白图
for i in range(1,10014):
	path='/home/jjlin/Desktop/image_inpainting/dataset/train_mask/%i.jpg'%i
	new_path='/home/jjlin/Desktop/image_inpainting/dataset/stack/train_mask/%i.jpg'%i
	stack(path,new_path)


#test彩图
for i in range(1,1003):
	path='/home/jjlin/Desktop/image_inpainting/dataset/test/%i.jpg'%i
	new_path='/home/jjlin/Desktop/image_inpainting/dataset/stack/test/%i.jpg'%i
	stack(path,new_path)

#test_pol彩图
for i in range(1,1003):
	path='/home/jjlin/Desktop/image_inpainting/dataset/test_pol/%i.jpg'%i
	new_path='/home/jjlin/Desktop/image_inpainting/dataset/stack/test_pol/%i.jpg'%i
	stack(path,new_path)





















