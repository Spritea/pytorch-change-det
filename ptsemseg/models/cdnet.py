import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class cdnet_contract(nn.Module):
    def __init__(self, in_size, out_size):
        super(cdnet_contract, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 7, 1, 3),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 7, 1, 3),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 7, 1, 3),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        self.conv4 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 7, 1, 3),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=2,return_indices=True)


    def forward(self, inputs):
        outputs = self.conv1(inputs)
        maxpool1,index1=self.maxpool1(outputs)
        conv2 = self.conv2(maxpool1)
        maxpool2,index2=self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3,index3=self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4,index4=self.maxpool4(conv4)

        return index1,index2,index3,index4,maxpool4


class cdnet_expand(nn.Module):
    def __init__(self, in_size, out_size):
        super(cdnet_expand, self).__init__()

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 7, 1, 3),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 7, 1, 3),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 7, 1, 3),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 7, 1, 3),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )

    def forward(self, inputs,index1,index2,index3,index4):
        unpool1 = self.unpool1(inputs,indices=index4)
        conv1=self.conv1(unpool1)
        unpool2 = self.unpool2(conv1, indices=index3)
        conv2 = self.conv2(unpool2)
        unpool3 = self.unpool3(conv2, indices=index2)
        conv3 = self.conv3(unpool3)
        unpool4 = self.unpool4(conv3, indices=index1)
        conv4 = self.conv4(unpool4)

        return conv4

class cdnet(nn.Module):
    def __init__(
        self,
        num_classes=49,
        in_channels=6,
    ):
        super(cdnet, self).__init__()
        self.in_channels = in_channels
        out_channel=64

        # downsampling
        self.contract = cdnet_contract(self.in_channels,out_channel)
        self.expand=cdnet_expand(out_channel,out_channel)

        self.final=nn.Sequential(
            nn.Conv2d(out_channel, num_classes, 7, 1, 3),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
        )

    def forward(self, inputs):
        index1,index2,index3,index4,contract=self.contract(inputs)
        expand=self.expand(contract,index1,index2,index3,index4)
        output=self.final(expand)

        return output
