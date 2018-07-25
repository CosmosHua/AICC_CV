import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
import pdb

#将npy文件保存为图片文件

imgs_test_predict = np.load('/home/jjlin/Desktop/image_inpainting/detect-cell-edge-use-unet-master/imgs_mask_test.npy')

save_path='/home/jjlin/Desktop/image_inpainting/dataset/u-net/test_mask'

#print(imgs_test.shape, imgs_test_predict.shape)

for i in range(0,1002):
	img=imgs_test_predict[i].reshape(256, 256)
	#pdb.set_trace()
	i=i+1
	temp='%i.jpg'%i
	complete_save_path=os.path.join(save_path,temp)
	io.imsave(complete_save_path,img)


