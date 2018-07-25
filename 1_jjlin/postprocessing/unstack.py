from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import pdb
from skimage import io
#from libtiff import TIFF

#从256*256到250*250

def unstack(path,new_path):
	img=cv2.imread(path)
	a=np.delete(img[:,:,0], [0, 1, 2,253,254,255], axis=0)
	a=np.delete(a, [0, 1, 2,253,254,255], axis=1)

	b=np.delete(img[:,:,1], [0, 1, 2,253,254,255], axis=0)
	b=np.delete(b, [0, 1, 2,253,254,255], axis=1)

	c=np.delete(img[:,:,2], [0, 1, 2,253,254,255], axis=0)
	c=np.delete(c, [0, 1, 2,253,254,255], axis=1)
	
	out=cv2.merge([a,b,c])
	
	cv2.imwrite(new_path,out)
	#pdb.set_trace()


for i in range(1,1003):
	path='/home/jjlin/Desktop/image_inpainting/dataset/dip/recover_images/%i.jpg'%i
	new_path='/home/jjlin/Desktop/image_inpainting/dataset/finally/recover_images/%i.jpg'%i
	unstack(path,new_path)

for i in range(1,1003):
	path='/home/jjlin/Desktop/image_inpainting/dataset/dip/mask/%i.jpg'%i
	new_path='/home/jjlin/Desktop/image_inpainting/dataset/finally/mask/%i.jpg'%i
	unstack(path,new_path)





















