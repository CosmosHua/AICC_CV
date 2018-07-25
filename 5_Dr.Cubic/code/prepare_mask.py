# coding=utf8
#########################################################################
# File Name: main.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: Sat 03 Feb 2018 03:50:30 PM CST
#########################################################################

import sys
import os
import time
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

from tools import parse, py_op, measures, utils, plot
from model import dataset, unet_models, layers

reload(sys)
sys.setdefaultencoding('utf8')
args = parse.args

args.input_filter = 3

try:
    os.mkdir(os.path.join(args.data_dir, 'model'))
    os.mkdir(os.path.join(args.data_dir, 'model', args.save_dir))
except:
    pass

logfile = os.path.join(args.data_dir, 'model', args.save_dir, 'log')
py_op.mkdir(os.path.join(args.data_dir, 'model', args.save_dir))
sys.stdout = utils.Logger(logfile)


def main():

    filelist = glob.glob(os.path.join(args.data_dir, 'AI/trainB/', '*'))
    filelist += glob.glob(os.path.join(args.data_dir, 'AI/testB/', '*'))

    test_dataset = dataset.DataBowl(filelist, phase='prepare')
    test_loader = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=False)

    net = unet_models.AlbuNet(num_classes=4, input_filter=3)
    net = net.cuda()
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['state_dict'])

    prepare_mask(test_loader, net)

def mkdir(d):
    path = d.split('/')
    for i in range(len(path)):
        d = '/'.join(path[:i+1])
        if not os.path.exists(d):
            os.mkdir(d)


def prepare_mask(data_loader, net):
    net.eval()
    for i, data in enumerate(tqdm(data_loader)):

        stain_cuda = Variable(data[1].cuda(async=True))
        output = net(stain_cuda)
        pred_mask = output[:,:1,:,:].contiguous()

        for j in range(args.batch_size):
            filename = data[3][j]
            new_file = filename.replace('../data/', '../data/pred_mask/')
            file_type = new_file.split('.')[-1]
            new_file = new_file.replace('.'+file_type, '')
            new_dir = '/'.join(new_file.split('/')[:-1])
            pred_mask_j = torch.sigmoid(pred_mask[j][0]).data.cpu().numpy()
            stain_j = (data[1][j].numpy()*100 + 128).astype(np.uint8)
            if not os.path.exists(new_dir):
                mkdir(new_dir)
            np.save(new_file+'.npy', pred_mask_j)
            pred_mask_j = (np.array([pred_mask_j,pred_mask_j, pred_mask_j]) * 256).astype(np.uint8)
            Image.fromarray(pred_mask_j.transpose(1,2,0)).save(new_file+'.png')
            Image.fromarray(stain_j.transpose(1,2,0)).save(new_file+'_stain.png')


if __name__ == '__main__':
    main()
