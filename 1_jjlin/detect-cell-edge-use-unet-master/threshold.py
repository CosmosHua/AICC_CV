import cv2
import numpy as np
import pdb

#加深网纹，转为二值
def threshold(path,new_path):
	img=cv2.imread(path)
	ret,bin_img=cv2.threshold(img,215,255,cv2.THRESH_BINARY)
	cv2.imwrite(new_path,bin_img)

for i in range(1,1003):
	path='/home/jjlin/Desktop/image_inpainting/dataset/u-net/test_mask/%i.jpg'%i
	new_path='/home/jjlin/Desktop/image_inpainting/dataset/dip/mask/%i.jpg'%i
	threshold(path,new_path)
	#pdb.set_trace()

