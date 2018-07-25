# coding=utf8
#########################################################################
# File Name: layers.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: Sun 04 Feb 2018 12:02:50 AM CST
#########################################################################
import numpy as np
import torch
from torch import nn


class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv2d(
            n_in, n_out, kernel_size=3,
            stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out,
                          kernel_size=1,
                          stride=stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        if type(x) is tuple:
            x = x[0]
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.classify_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        # self.d_loss = nn.BCELoss()
        self.d_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_mask, pred_clean, pred_merge, mask, clean, clean_pred_D=None, stain_pred_D=None, clean_label=None, stain_label=None):


        # mask classification loss
        pred_mask = pred_mask.view(-1)
        mask = mask.view(-1)
        pos_index = mask == 1
        neg_index = mask == 0
        mask_loss = 0
        pos_pred_mask = pred_mask[pos_index]
        if len(pos_pred_mask) > 0:
            pos_mask = mask[pos_index]
            mask_loss += 0.5 * self.classify_loss(pos_pred_mask,pos_mask)
        neg_pred_mask = pred_mask[neg_index]
        if len(neg_pred_mask) > 0:
            neg_mask = mask[neg_index]
            mask_loss += 0.5 * self.classify_loss(neg_pred_mask,neg_mask)

        # clean regression loss
        clean_loss = self.mse_loss(pred_clean, clean) * 100 * 0.1
        clean_loss += self.mse_loss(pred_merge, clean) * 100 * 0.9

        # net_D loss
        if clean_pred_D is not None:
            d_loss= 0.5 * self.d_loss(clean_pred_D.view(-1), clean_label) + 0.5 * self.d_loss(stain_pred_D.view(-1), stain_label)
            g_loss = self.d_loss(stain_pred_D.view(-1), clean_label)
        else:
            d_loss= 0
            g_loss = 0

        return [mask_loss, clean_loss, g_loss, d_loss]

def clean_loss():
    pass

