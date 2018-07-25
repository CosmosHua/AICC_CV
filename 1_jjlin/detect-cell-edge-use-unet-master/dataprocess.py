from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
#from libtiff import TIFF

class dataProcess(object):
    def __init__(self, out_rows, out_cols, aug_train_path="/home/jjlin/Desktop/image_inpainting/dataset/stack/train_pol",
                 aug_label_path="/home/jjlin/Desktop/image_inpainting/dataset/stack/train_mask", test_path = '/home/jjlin/Desktop/image_inpainting/dataset/stack/test_pol', npy_path="/home/jjlin/Desktop/image_inpainting/dataset/u-net/npydata",
                 img_type="jpg"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        #self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.test_path = test_path
        self.npy_path = npy_path
        self.img_type = img_type

    def load_train_data(self):
        # 读入训练数据包括label_mask(npy格式), 归一化(只减去了均值)
        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        mean = imgs_train.mean(axis=0)
        imgs_train -= mean
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        mean = imgs_test.mean(axis=0)
        imgs_test -= mean
        return imgs_test


if __name__ == "__main__":
    mydata = dataProcess(256, 256)#生成数据对象
    #mydata.create_train_data()
    #mydata.create_test_data()
    imgs_train, imgs_mask_train = mydata.load_train_data()
    print(imgs_train.shape, imgs_mask_train.shape)
