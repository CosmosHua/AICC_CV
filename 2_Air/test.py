from __future__ import print_function
import argparse
import os
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from util import is_image_file, load_img, save_img
from vae import VAE
from skimage import io

# Testing settings
parser = argparse.ArgumentParser(description='chongqingDRRN')
parser.add_argument('--model', type=str, default='checkpoint/VAE_45_39.6572.pth', help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

pre = torch.load('checkpoint/stage_24.pth')
model = torch.load(opt.model)

image_dir = "b/"
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
image_filenames.sort()

transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)

    # print(img.size())
    input = Variable(img, volatile=True).view(1, -1, 250, 250)

    if opt.cuda:
        pre = pre.cuda()
        model = model.cuda()
        input = input.cuda()
    # print(model)
    out = pre(input)
    out = model(out)
    out = out.cpu()
    out_img = out.data[0]
    # print(torch.mean(out_img))
    if not os.path.exists("result"):
        os.makedirs("result")
    save_img(out_img, "result/{}".format(image_name))
