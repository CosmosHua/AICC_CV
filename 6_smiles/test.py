import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import _NetG
import argparse
import os, math
import numpy as np
from PIL import Image
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='face', help='input dataset')
parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=16, help='test batch size')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
data_dir = '../Data/'
model_dir = params.dataset + '_model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Data pre-processing
test_transform = transforms.Compose([transforms.ToTensor()])

# Test data
test_data = DatasetFromFolder(data_dir, subfolder='test', transform=test_transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=params.batch_size, num_workers=4, shuffle=False)

# Load model
model = _NetG().cuda()
model.load_state_dict(torch.load(model_dir + 'generator_param.pkl'))
model.eval()

name_list = []
mse_list = []
rmse_list = []
psnr_list = []
# Test
for i, (filenames, inputs, target) in enumerate(test_data_loader):
    # input & target image data
    x_ = Variable(inputs.cuda(), volatile=True)

    gen_image = model(x_)
    gen_image = gen_image.cpu().data

    # Show result for test data
    for (filename, img_g, img_t) in zip(filenames, gen_image, target):
        # Scale to 0-255
        img_g = img_g.numpy().astype(np.float32)
        img_t = img_t.numpy().astype(np.float32)

        img_g = img_g*255.
        img_g = img_g.transpose(1,2,0).astype(np.float32)

        img_t = img_t*255.
        img_t = img_t.transpose(1,2,0).astype(np.float32)

    
        mse = np.mean((img_g-img_t)**2)
        rmse = math.sqrt(mse)
        psnr = 20 * math.log10(255.0/rmse)
    
        name_list.append(filename)
        mse_list.append(mse)
        rmse_list.append(rmse)
        psnr_list.append(psnr)

        im_g = Image.fromarray(img_g.astype(np.uint8))
        im_g.save("./../result/"+filename+"_.jpg")
        im_t = Image.fromarray(img_t.astype(np.uint8))
        im_t.save("./../result/"+filename+".jpg")
        
    print('%d images are generated.' % ((i+1) * params.batch_size))

dataframe = pd.DataFrame({'name':name_list, 'mse':mse_list, 'rmse':rmse_list, 'psnr':psnr_list})
dataframe.to_csv("result.csv",index=False,sep=',')
