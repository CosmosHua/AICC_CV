# Custom dataset
from PIL import Image
import torch.utils.data as data
import os
from os import path as osp
import random


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, subfolder, transform=None, fliplr=False):
        super(DatasetFromFolder, self).__init__()
        self.input_path = os.path.join(image_dir, subfolder)

        self.image_filenames_input = [x for x in sorted(os.listdir(osp.join(self.input_path, 'input')))]
        self.image_filenames_target = [x for x in sorted(os.listdir(osp.join(self.input_path, 'target')))]

        self.transform = transform
        self.fliplr = fliplr

    def __getitem__(self, index):
        # Load Image
        input_name = os.path.join(self.input_path, 'input', self.image_filenames_input[index])
        target_name = os.path.join(self.input_path, 'target', self.image_filenames_target[index])

        filename = self.image_filenames_target[index].split('.')[0]

        input_img = Image.open(input_name)
        target_img = Image.open(target_name)

        if self.fliplr:
            if random.random() < 0.5:
                input_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
                target_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return filename, input_img, target_img

    def __len__(self):
        return len(self.image_filenames_input)
