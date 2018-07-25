import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from data import get_training_set, get_test_set
from torch.utils.data import DataLoader
import os
from math import log10
from logger import Logger

logger = Logger('./logs')

class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*6*6, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*6*6, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*2*6*6)

        self.d2 = nn.ConvTranspose2d(ngf*8*2, ngf*8, 4, 2)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.d3 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.d4 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.d5 = nn.ConvTranspose2d(ngf*2, ngf, 2, 2)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.d6 = nn.ConvTranspose2d(ngf, ngf, 4, 2)

        self.output = nn.Conv2d(ngf, nc, 3, 1, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*6*6)

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*8*2, 6, 6)
        h2 = self.leakyrelu(self.bn6(self.d2(h1)))
        h3 = self.leakyrelu(self.bn7(self.d3(h2)))
        h4 = self.leakyrelu(self.bn8(self.d4(h3)))
        h5 = self.leakyrelu(self.bn9(self.d5(h4)))
        return self.output(self.d6(h5))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, 250, 250))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, 250, 250))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar



def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


def loss_function(recon_x, x, mu, logvar, reconstruction_function):
    MSE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return MSE + KLD


batch_step = 0

def train(model, epoch, dataloader, optimizer, reconstruction_function, inputs, targets):
    model.train()
    train_loss = 0
    for iteration, batch in enumerate(dataloader, 1):
        inputs_cpu, targets_cpu = batch[0], batch[1]
        # print(inputs_cpu.size(), targets_cpu.size())
        inputs.data.resize_(inputs_cpu.size()).copy_(inputs_cpu)
        targets.data.resize_(targets_cpu.size()).copy_(targets_cpu)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        optimizer.zero_grad()
        recon_batch = model(inputs)
        #loss = loss_function(recon_batch, targets, mu, logvar, reconstruction_function)
        
        criterionMSE = nn.MSELoss()
        loss = criterionMSE(recon_batch, targets)
        if torch.cuda.is_available():
            criterionMSE.cuda()
        # print(torch.max(recon_batch))
        mse = criterionMSE(recon_batch, targets)
        psnr = 10 * log10(torch.max(targets)/mse.data[0])

        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if iteration % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t PSNR: {:.6f}'.format(
                epoch, iteration, len(dataloader), 100.*iteration/len(dataloader), loss.data[0]/len(inputs_cpu), psnr))

        info = {'loss': loss.data[0] / len(inputs_cpu), 'psnr': psnr}
        for tag, value in info.items():
            global logger, batch_step
            batch_step += len(inputs_cpu)
            logger.scalar_summary(tag, value, batch_step)

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader) * len(inputs_cpu)))
    return model


def checkpoint(epoch, model):
    if not os.path.exists("checkpoint"): os.mkdir("checkpoint")
    savePath = "checkpoint/DRRN_{}.pth".format(epoch)
    torch.save(model, savePath)
    print("Checkpoint saved")


from drrn import DRRN

if __name__ == '__main__':
    batch_size = 16
    input_nc = 3

    inputs = torch.FloatTensor(batch_size, input_nc, 250, 250)
    targets = torch.FloatTensor(batch_size, input_nc, 250, 250)

    inputs = Variable(inputs)
    targets = Variable(targets)

    root_path = "data/"
    train_set = get_training_set(root_path)
    test_set = get_test_set(root_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=batch_size, shuffle=True)
    # testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

   # model = VAE(nc=input_nc, ngf=64, ndf=64, latent_variable_size=500)
    model = DRRN()  
    reconstruction_function = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    
    if torch.cuda.is_available():  
        model.cuda()
        reconstruction_function.cuda()

    for epoch in range(1, 2000+1):
        model = train(model, epoch, training_data_loader, optimizer, reconstruction_function, inputs, targets)
        checkpoint(epoch, model)
    # l = None
    # for epoch in range(100):
    #     for i, data in enumerate(dataloader, 0):
    #         inputs, classes = data
    #         inputs, classes = Variable(inputs.resize_(batch_size, input_dim)), Variable(classes)
    #         optimizer.zero_grad()
    #         dec = vae(inputs)
    #         ll = latent_loss(vae.z_mean, vae.z_sigma)
    #         loss = criterion(dec, inputs) + ll
    #         loss.backward()
    #         optimizer.step()
    #         l = loss.data[0]
    #     print(epoch, l)

    # plt.imshow(vae(inputs).data[0].numpy().reshape(28, 28), cmap='gray')
    # plt.show(block=True)
