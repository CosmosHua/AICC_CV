from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import pdb
from skimage import io
#from libtiff import TIFF
#转为npydata

out_rows,out_cols=256,256

train_path="/home/jjlin/Desktop/image_inpainting/dataset/stack/train_pol"
label_path="/home/jjlin/Desktop/image_inpainting/dataset/stack/train_mask"
test_path = "/home/jjlin/Desktop/image_inpainting/dataset/stack/test_pol"
npy_path="/home/jjlin/Desktop/image_inpainting/dataset/u-net/npydata"


count=len(os.listdir(train_path))
count_test=len(os.listdir(test_path))

imgdatas = np.ndarray((count, out_rows, out_cols, 1), dtype=np.uint8)
imglabels = np.ndarray((count, out_rows, out_cols, 1), dtype=np.uint8)
imgtest = np.ndarray((count_test, out_rows, out_cols, 1), dtype=np.uint8)

'''
i = 0
for indir in os.listdir(train_path):
	path  =os.path.join(train_path, indir)
	img   = load_img(path, grayscale=True)
	img = img_to_array(img)
	imgdatas[i] = img
	i+=1
np.save(npy_path + '/imgs_train.npy', imgdatas)            # 生成npy数据
'''
'''
i=0
for indir in os.listdir(label_path):
	path  =os.path.join(label_path, indir)
	label =load_img(path, grayscale=True)
	label = img_to_array(label)
	imglabels[i]=label
	i+=1
np.save(npy_path + '/imgs_mask_train.npy', imglabels)
'''

list_test=os.listdir(test_path)
list_test.sort(key=lambda x:int(x[:-4]))#排序，以便训练后对应
#pdb.set_trace()
i=0
for indir in list_test:
	path  =os.path.join(test_path, indir)
	test =load_img(path, grayscale=True)
	test = img_to_array(test)
	imgtest[i]=test
	i+=1
np.save(npy_path + '/imgs_test.npy', imgtest)












