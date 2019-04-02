#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from models.Conv_rnn import ConvLSTMCell
from models.masked_convolution import MaskedConv2d
from models.masked_convolution import MaskedConv1d

class Network(nn.Module):
    def __init__(self,H,W):
        super(Network, self).__init__()
        fm=64
        self.masked_2d = nn.Sequential(
                            MaskedConv2d('A', 1,  fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
                            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
                            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
                            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
                            nn.Conv2d(fm, 2, 1))
        self.rnn = ConvLSTMCell(2,64)
        post_net = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

    def forward(self, x1, h1):  # x1: batch*1*14*30
        x1 = self.masked_2d(x1)  # x1: batch*2*14*30
        for i in range(x[2]):
            h1 = self.rnn(x1[:, :, 0, :], h1)  ##input  batch*2*30   batch*64*30
