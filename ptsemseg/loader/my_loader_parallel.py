import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data
from ptsemseg.augmentations import *

import cv2 as cv
from torchvision import transforms


class myLoader_parallel(data.Dataset):
    def __init__(
            self,
            root,
            split="train",
            is_transform=False,
            img_size=512,
            augmentations=None,
            img_norm=True,
    ):
        self.root = root
        self.split = split
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.mean = np.array([115.3165639, 83.02458143, 81.95442675])
        self.n_classes = 7 * 7
        self.files_src = collections.defaultdict(list)
        self.files_dst = collections.defaultdict(list)

        for split in ["train", "test", "val"]:
            file_list_src = os.listdir(root + "/" + split + "_src")
            file_list_dst = os.listdir(root + "/" + split + "_dst")

            self.files_src[split] = file_list_src
            self.files_dst[split] = file_list_dst

        self.tf = transforms.ToTensor()
        self.tf_no_train = transforms.ToTensor()

    def __len__(self):
        return len(self.files_src[self.split])

    def __getitem__(self, index):
        img_name_src = self.files_src[self.split][index]
        img_name_dst = self.files_dst[self.split][index]

        img_path_src = self.root + "/" + self.split + "_src" + "/" + img_name_src
        img_path_dst = self.root + "/" + self.split + "_dst" + "/" + img_name_dst
        #img_name_src has same name with corresponding label img
        lbl_path = self.root + "/" + self.split + "_labels/" + img_name_src

        img_src = cv.cvtColor(cv.imread(img_path_src, -1), cv.COLOR_BGR2RGB)
        img_dst = cv.cvtColor(cv.imread(img_path_dst, -1), cv.COLOR_BGR2RGB)

        lbl = cv.imread(lbl_path, -1)
        # im = Image.open(im_path)
        # lbl = Image.open(lbl_path)

        if self.augmentations is not None:
            img_src,img_dst, lbl = self.augmentations(img_src,img_dst, lbl)

        if self.is_transform:
            img_src,img_dst, lbl = self.transform(img_src,img_dst, lbl)
        if self.split=='test':
            return img_path_src,img_src, img_dst
        else:
            return img_src, img_dst, lbl

    def transform(self, img_src,img_dst, lbl):
        if self.img_size == ('same', 'same'):
            pass
        else:
            # opencv resize,(width,heigh)
            img_src = cv.resize(img_src, (self.img_size[1], self.img_size[0]))
            img_dst = cv.resize(img_dst, (self.img_size[1], self.img_size[0]))

            lbl = cv.resize(lbl, (self.img_size[1], self.img_size[0]))

            # img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
            # lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        if self.split == "train":
            img_src = self.tf(img_src)
            img_dst = self.tf(img_dst)

        else:
            img_src = self.tf_no_train(img_src)
            img_dst = self.tf_no_train(img_dst)

        lbl = torch.from_numpy(lbl).long()
        return img_src,img_dst, lbl

    def decode_segmap(self, temp, plot=False):
        # bg=[0,0,0]
        # Farmland=[128,0,0]
        # Bareland = [0,128,0]
        # Industrial = [128, 128, 0]
        # Parking = [0, 0, 128]
        # Residential=[128,0,128]
        # Water=[0,128,128]

        candy_color_list = [(0, 0, 0), (255, 250, 250), (248, 248, 255), (211, 211, 211),
                            (255, 99, 71), (255, 250, 240), (139, 69, 19), (250, 240, 230),
                            (128, 0, 0), (0, 206, 209), (255, 215, 0), (205, 92, 92), (255, 228, 196),
                            (255, 218, 185), (255, 222, 173), (175, 238, 238), (0, 128, 0), (255, 248, 220),
                            (47, 79, 79), (255, 250, 205), (255, 245, 238), (240, 255, 240),
                            (245, 255, 250), (240, 255, 255), (128, 128, 0), (240, 248, 255), (230, 230, 250),
                            (255, 240, 245), (255, 228, 225), (255, 255, 240), (105, 105, 105),
                            (112, 128, 144), (0, 0, 128), (190, 190, 190), (245, 245, 245), (100, 149, 237),
                            (65, 105, 225), (0, 191, 255), (135, 206, 250), (70, 130, 180),
                            (128, 0, 128), (255, 228, 181), (250, 235, 215), (95, 158, 160), (0, 250, 154),
                            (255, 255, 0), (255, 239, 213), (255, 235, 205), (0, 128, 128)]

        label_colours = np.array(candy_color_list)
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]
        # rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
        # rgb[:, :, 0] = r / 255.0
        # rgb[:, :, 1] = g / 255.0
        # rgb[:, :, 2] = b / 255.0
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb
