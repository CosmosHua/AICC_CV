import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.nn import DataParallel
from torch.autograd import Variable


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class DRRN(nn.Module):
	def __init__(self, blocks=9, units=2):
		super(DRRN, self).__init__()
		self.blocks, self.units = blocks, units
		self.input = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.output = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
		self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
		self.residual_blocks = ResnetBlock(blocks, units)

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, sqrt(2. / n))

	def forward(self, x):
		out = self.input(self.LeakyReLU(x))
		out = self.residual_blocks(out)
		out = self.output(self.LeakyReLU(out))
		return out

class ResnetBlock(nn.Module):
	def __init__(self, blocks=9, units=2):
		super(ResnetBlock, self).__init__()
		self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
		self.units = units
		self.blocks = blocks
		self.residual_block = self.makeblocks(self.units)

	def makeblocks(self, units=2, norm_layer=nn.BatchNorm2d):
		layers = []
		for _ in range(units):
			layers += [self.LeakyReLU]
			layers += [self.ConvRes(128, 128)]
		return nn.Sequential(*layers)

	def ConvRes(self, input_nc, output_nc):
		return nn.Conv2d(in_channels=input_nc, out_channels=output_nc, kernel_size=3, stride=1, padding=1, bias=False)

	def forward(self, x):
		output = x
		for _ in range(self.blocks):
			output = self.residual_block(output)
			output += x
		return output
