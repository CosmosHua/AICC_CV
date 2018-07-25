# -*- coding: utf-8 -*-

import os
import training as Network
import tensorflow as tf
import matplotlib.image as img
import numpy as np
import cv2
import glob

from utils import gray2rgb, guided_filter, getres

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # select GPU device

test_path = 'data/test/'
res_path = 'result/'
out_path = 'result.xls'

files = os.listdir(test_path)

num_channels = 3
image = tf.placeholder(tf.float32, shape=(1, 250, 250, num_channels))
detail = tf.placeholder(tf.float32, shape=(1, 250, 250, num_channels))

output, _ = Network.inference(image, detail, is_training=False)

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    if tf.train.get_checkpoint_state('./model/'):
        ckpt = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, ckpt)   #'./model/model-epoch-4'
        print("Loading model")
    else:
        print("No model for loading")
        exit()

    for i, file in enumerate(files):

        if file.endswith('_.jpg'):
            ori = img.imread(os.path.join(test_path, file))

            if file in ['0000486003_.jpg', '0000486005_.jpg', '0000486013_.jpg', '0000492105_.jpg']:
                ori = cv2.cvtColor(ori, cv2.COLOR_RGB2GRAY)
                ori = gray2rgb(ori)

            ori = ori / 255.0
            input_tensor = np.expand_dims(ori[:, :, :], axis=0)
            detail_layer = input_tensor - guided_filter(input_tensor, num_patches=1, width=input_tensor.shape[1], height=input_tensor.shape[2], channel=num_channels)

            final_output = sess.run(output, feed_dict={image: input_tensor, detail: detail_layer})

            final_output[np.where(final_output < 0.)] = 0.
            final_output[np.where(final_output > 1.)] = 1.
            derained = final_output[0, :, :, :]
            img.imsave(os.path.join(res_path, file), derained)

getres(test_path, res_path, out_path)

print('testing done.........')