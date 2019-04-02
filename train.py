#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from models.masked_convolution import MaskedConv2d
import os
import time
import sys
import argparse
import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, utils
from dataloader import Codeloader
from models.Conv_rnn import ConvLSTMCell
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', '-N', type=int, default=32, help='batch size')
parser.add_argument(
    '--train_folder', required=True, type=str, help='folder of training data')
parser.add_argument(
    '--test_folder', required=True, type=str, help='folder of testing data')
parser.add_argument(
    '--max_epochs', '-e', type=int, default=20, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
args = parser.parse_args()


def main():
    batch_size=args.batch_size
    train_set = Codeloader(root=args.train_folder)

    train_loader = data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1)

    test_set = Codeloader(root=args.test_folder)

    test_loader = data.DataLoader(
        dataset=test_set, batch_size=50, shuffle=False, num_workers=1)

    print('total train codes: {}; total batches: {}'.format(
        len(train_set), len(train_loader)))

    print('total test codes: {}; total batches: {}'.format(
        len(test_set), len(test_loader)))

    fm = 64
    adaptation0 = nn.Sequential(
        MaskedConv2d('A', 32, fm, 7, 1, 3, bias=False),
        #, nn.BatchNorm2d(fm), nn.ReLU(True),
        # MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        # MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        # MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        # MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        # MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        # MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        # MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        #nn.Conv2d(fm, 64, 1),
        #nn.Conv2d(64, 2, 1)
        nn.Tanh())
    adaptation0.cuda()

    adaptation1 = nn.Sequential(
        nn.Conv2d(32, fm, 7, 1, 3, bias=False),
        nn.Tanh())
    adaptation1.cuda()

    post_net = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
        nn.Tanh(),
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
        nn.Tanh())
        # nn.BatchNorm1d(64),
        # nn.ReLU(),
        # nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
        # nn.BatchNorm1d(32),
        # nn.ReLU(),
        # nn.Conv1d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0,bias=False),
        # nn.Sigmoid())
    post_net.cuda()

    ConvLSTM = ConvLSTMCell(64, 64).cuda()

    optimizer = optim.Adam([
            {'params': adaptation0.parameters()},
            {'params': adaptation1.parameters()},
            {'params': ConvLSTM.parameters()},
            {'params': post_net.parameters()},
        ],
        lr=args.lr, weight_decay=1e-9)

    #optimizer = optim.Adam(masked_2d.parameters())

    for epoch in range(args.max_epochs):
        train_error = []
        adaptation0.train(True)
        adaptation1.train(True)
        ConvLSTM.train(True)
        post_net.train(True)
        for input in train_loader:
            x = input['content'].squeeze().float().cuda()
            #print(x.shape)
            #print(x.shape)
            #x = x[:, 0, :, :].unsqueeze(1).float().cuda()   # batch*1*14*30
            #target = Variable((input['content'].squeeze()[:, 0, :, :] * 1).long().cuda()) #target: batch*14*30
            target = Variable((input['content'].squeeze() * 1).float().cuda())
            h1 = (Variable(torch.zeros(batch_size, 64, 14, 30).cuda()), Variable(torch.zeros(batch_size, 64, 14, 30).cuda()))
            x = adaptation0(x)   #batch*32*14*30
            losses = []
            #number_iter = x.shape[2]
            #for i in range(number_iter):
                #print(x[:, :, i, :].shape)
            h1 = ConvLSTM(x, h1)   #h1[0] batch*64*30
            #print('h1 shape')
            #print(h1[1].shape)
            y = post_net(h1[0])   #y batch*32*14*30
            #print('y shape')
            #print(y.shape)
            y = torch.clamp(y, min=-1, max=1)
            y = torch.add(y, 1)
            y = torch.div(y, 2)
            #loss = F.cross_entropy(y, target[:, i, :])
            #loss = F.cross_entropy(y, target)
            #losses.append(loss)
            #loss = sum(losses)

            entropy = ((1.0 - target) * torch.log(1.0 - y) + target * torch.log(y)) / (-math.log(2))
            loss = torch.mean(entropy)
            optimizer.zero_grad()
            loss.backward()
            train_error.append(loss.item())
            optimizer.step()
        print('train error: %f' % np.mean(train_error))


        test_error = []
        adaptation0.train(False)
        adaptation1.train(False)
        ConvLSTM.train(False)
        post_net.train(False)
        for input in test_loader:
            x = input['content'].squeeze().float().cuda()
            target = Variable((input['content'].squeeze() * 1).float().cuda())
            h1 = (Variable(torch.zeros(batch_size, 64, 14, 30).cuda()), Variable(torch.zeros(batch_size, 64, 14, 30).cuda()))
            x = adaptation0(x)  # batch*32*14*30
            losses = []
            # number_iter = x.shape[2]
            # for i in range(number_iter):
            # print(x[:, :, i, :].shape)
            h1 = ConvLSTM(x, h1)  # h1[0] batch*64*30
            y = post_net(h1[0])  # y batch*32*14*30

            y = torch.clamp(y, min=-1, max=1)
            y = torch.add(y, 1)
            y = torch.div(y, 2)
            entropy = ((1.0 - target) * torch.log(1.0 - y) + target * torch.log(y)) / (-math.log(2))
            loss = torch.mean(entropy)
            test_error.append(loss.item())
        print('test error: %f' % np.mean(test_error))


if __name__ == '__main__':
    main()
