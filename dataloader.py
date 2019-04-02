#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import os.path

import torch
import torch.utils.data as data
import numpy as np


def is_npz_file(filename):
    return filename.endswith('npz')


def default_loader(path):
    #the encoded file should be .npz file, with {shape, content}
    #TODO
    return np.load(path)


class Codeloader(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, loader=default_loader):
        codes = []
        for filename in os.listdir(root):
            codes.append('{}'.format(filename))

        self.root = root
        self.codes = codes
        self.loader = loader

    def __getitem__(self, index):
        filename = self.codes[index]
        try:
            code = self.loader(os.path.join(self.root, filename))
        except:
            pass

        return code

    def __len__(self):
        return len(self.codes)
