#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import matplotlib.image as img
import numpy as np
import random
import h5py

from utils import gray2rgb

random.seed = 2018

clean_path = "./data/train/label/"
mesh_path = "./data/train/input/"
map_path = "./data/train/map/"


files = os.listdir(mesh_path)
size_input = 64  # size of the training patch
num_channel = 3  # number of the input's channels.
num_files = len(files)  # total ( num_files - 1 ) training h5 files, the last one is used for validation
num_patch = 500  # number of patches in each h5 file.

for j in range(num_files):
    Data = np.zeros((num_patch, size_input, size_input, num_channel))
    Label = np.zeros((num_patch, size_input, size_input, num_channel))
    Map = np.zeros((num_patch, size_input, size_input, 1))

    for i in range(num_patch):

        r_idx = random.randint(0, len(files) - 1)

        mesh = img.imread(mesh_path + files[r_idx])
        mesh = mesh / 255.0

        if len(mesh.shape) < 3:
            mesh = gray2rgb(mesh)

        label = img.imread(clean_path + files[r_idx])
        label = label / 255.0

        if len(label.shape) < 3:
            label = gray2rgb(label)

        map = img.imread(map_path + files[r_idx])
        map = map.reshape([map.shape[0], map.shape[1], 1])
        map = np.round(map / 255.0)

        x = random.randint(0, mesh.shape[0] - size_input)
        y = random.randint(0, mesh.shape[1] - size_input)

        subim_input = mesh[x: x + size_input, y: y + size_input, :]
        subim_label = label[x: x + size_input, y: y + size_input, :]
        subim_map = map[x: x + size_input, y: y + size_input, :]

        Data[i, :, :, :] = subim_input
        Label[i, :, :, :] = subim_label
        Map[i, :, :, :] = subim_map

    if j + 1 < num_files:
        f = h5py.File('./data/train/h5data/train' + str(j + 1) + '.h5', 'w')
        f['data'] = Data
        f['label'] = Label
        f['map'] = Map
        f.close()
        print(str(j + 1) + '/' + str(num_files - 1) + ' training h5 files are generated')

    else:
        f = h5py.File('./data/train/h5data/validation.h5', 'w')
        f['data'] = Data
        f['label'] = Label
        f['map'] = Map
        f.close()
        print('validation h5 file is generated')

print('all h5 files are generated')