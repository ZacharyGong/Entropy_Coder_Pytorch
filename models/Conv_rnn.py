#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from models.masked_convolution import MaskedConv1d, MaskedConv2d

class ConvRNNCellBase(nn.Module):
    def __repr__(self):
        s = (
            '{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}'
            ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ', hidden_kernel_size={hidden_kernel_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ConvLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 hidden_kernel_size=1,
                 bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.hidden_kernel_size = _pair(hidden_kernel_size)

        hidden_padding = _pair(hidden_kernel_size // 2)

        gate_channels = 4 * self.hidden_channels
        self.conv_ih = nn.Sequential(
            MaskedConv2d('B', input_channels, gate_channels, [1, 3], 1, [0, 1], bias=False)
            #MaskedConv1d('B', input_channels, gate_channels, 3, 1, 1, bias=False),
            #, nn.BatchNorm1d(gate_channels),
            #nn.ReLU(True),
            #MaskedConv1d('B', gate_channels, gate_channels, 3, 1, 1, bias=False)
        )

        self.conv_hh = nn.Sequential(
            nn.Conv2d(hidden_channels, gate_channels, [1, 3], 1, [0, 1], bias=False)
            #, nn.BatchNorm1d(gate_channels),
            #nn.ReLU(True),
            #MaskedConv1d('B', gate_channels, gate_channels, 3, 1, 1, bias=False)
            #, nn.BatchNorm1d(gate_channels),
            #nn.ReLU(True),
            #MaskedConv1d('B', gate_channels, gate_channels, 3, 1, 1, bias=False), nn.BatchNorm1d(gate_channels),
            #nn.ReLU(True),
            #MaskedConv1d('B', gate_channels, gate_channels, 3, 1, 1, bias=False), nn.BatchNorm1d(gate_channels),
            #nn.ReLU(True),
            #MaskedConv1d('B', gate_channels, gate_channels, 3, 1, 1, bias=False), nn.BatchNorm1d(gate_channels),
            #nn.ReLU(True)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for e in self.conv_ih:
            if isinstance(e, MaskedConv1d):
                e.reset_parameters()
        for e in self.conv_hh:
            if isinstance(e, MaskedConv1d):
                e.reset_parameters()

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.conv_ih(input) + self.conv_hh(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy




