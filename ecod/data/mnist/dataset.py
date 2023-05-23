# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np

from skimage.transform import resize

import torch
from torch.utils.data import Dataset, DataLoader

from ecod.data.mnist.load import load_mnist_data


class MnistDataset(Dataset):
    def __init__(self, shape, tvt="train", pad=True, random_resize=False):
        self.tvt = tvt
        self.shape = shape
        self.pad = pad
        self.padding = self.get_pad()
        self.random_resize = random_resize
        self.split = 1.0 if tvt == "test" else 0.9
        self.imgs, self.labels = None, None
        self.len = len(load_mnist_data(split=self.split, train_val_test=self.tvt, shuffle=False)[0])

    def get_pad(self, img_shape=(28, 28)):
        pad = [0, 0]
        for idx, size in enumerate(self.shape):
            pad_len = size - img_shape[idx]
            if pad_len >= 0:
                pad_size, mod = divmod(pad_len, 2)
                pad[idx] = [int(pad_size), int(pad_size) + int(mod != 0)]
            else:
                raise ValueError(f"Shape has to be at least {img_shape} but is {self.shape}")
        return pad

    def pad_image(self, img):
        if self.pad:
            if self.random_resize:
                shape = [np.random.random_integers(28, self.shape[0])] * 2
                img = resize(img, shape)
                pad = self.get_pad(shape)
            else:
                pad = self.padding
                shape = self.shape
            img = np.pad(img, pad)
        else:
            img = resize(img, self.shape)
            shape = self.shape
        return img, shape

    def __getitem__(self, index):
        # load data here to let it do each worker individually
        if self.imgs is None:
            self.imgs, self.labels = load_mnist_data(split=self.split, train_val_test=self.tvt, shuffle=True)
        img = self.imgs[index]
        img = self.pad_image(img)[0]
        img = torch.from_numpy(img).to(torch.float32)[None]
        label = self.labels[index]
        return img, label

    def __len__(self):
        return self.len


class MnistODDataset(MnistDataset):
    def __getitem__(self, index):
        # load data here to let it do each worker individually
        if self.imgs is None:
            self.imgs, self.labels = load_mnist_data(split=self.split, train_val_test=self.tvt, shuffle=True)
        img = self.imgs[index]
        img, shape = self.pad_image(img)
        img = torch.from_numpy(img).to(torch.float32)[None]
        label = self.labels[index]
        return img, label

    def __len__(self):
        return self.len
