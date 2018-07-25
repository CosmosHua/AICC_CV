import torch
from torch import nn
from torch.autograd import Variable

class UNet(torch.nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(UNet, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        self.inputs = nn.Conv2d(nc, ndf, 3, 1, 1)

        # encoder
        self.e1 = nn.Conv2d(ndf, ndf, 4, 2)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.mid_conv1 = torch.nn.Conv2d(ndf*8, ndf*8, 3, padding=1)
        self.mid_bn1 = torch.nn.BatchNorm2d(ndf*8)
        self.mid_conv2 = torch.nn.Conv2d(ndf*8, ndf*8, 3, padding=1)
        self.mid_bn2 = torch.nn.BatchNorm2d(ndf*8)
        self.mid_conv3 = torch.nn.Conv2d(ndf*8, ndf*8, 3, padding=1)
        self.mid_bn3 = torch.nn.BatchNorm2d(ndf*8)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*2*6*6)

        self.d2 = nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2)  
        self.conv2 = nn.Conv2d(2*ngf*8, ngf*8, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3) 

        self.d3 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2)
        self.conv3 = nn.Conv2d(ngf*8, ngf*4, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.d4 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2)
        self.conv4 = nn.Conv2d(ngf*4, ngf*2, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.d5 = nn.ConvTranspose2d(ngf*2, ngf, 2, 2)
        self.conv5 = nn.Conv2d(ngf*2, ngf, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.d6 = nn.ConvTranspose2d(ngf, ngf, 4, 2)

        self.output = nn.Conv2d(2*ngf, nc, 3, 1, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))   #124
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))  #62
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))  #30
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))  #14
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))  #6
        return h1, h2, h3, h4, h5

    def decode(self, z, encode_outputs):
        z = z.view(-1, self.ngf*8, 6, 6)
        h2 = self.leakyrelu(self.bn6(self.d2(z)))
        cat2 = torch.cat([h2, encode_outputs[3]], 1)
        con2 = self.leakyrelu(self.conv2(cat2))
        h3 = self.leakyrelu(self.bn7(self.d3(con2)))
        cat3 = torch.cat([h3, encode_outputs[2]], 1)
        con3 = self.leakyrelu(self.conv3(cat3))
        h4 = self.leakyrelu(self.bn8(self.d4(con3)))
        cat4 = torch.cat([h4, encode_outputs[1]], 1)
        con4 = self.leakyrelu(self.conv4(cat4))
        h5 = self.leakyrelu(self.bn9(self.d5(con4)))
        cat5 = torch.cat([h5, encode_outputs[0]], 1)
        con5 = self.leakyrelu(self.conv5(cat5))
        return self.leakyrelu(self.d6(con5))

    def forward(self, x):
        inputs = self.inputs(x.view(-1, self.nc, 250, 250))
        h1, h2, h3, h4, h5 = self.encode(inputs)
        mid = self.leakyrelu(self.mid_bn1(self.mid_conv1(h5)))
        mid = self.leakyrelu(self.mid_bn2(self.mid_conv2(mid)))
        result = self.decode(mid, (h1, h2, h3, h4))
        outputs = self.output(torch.cat([inputs, result], 1))
        return torch.clamp(outputs, 0, 255)

if __name__ == '__main__':
    net = UNet()
    print(net)