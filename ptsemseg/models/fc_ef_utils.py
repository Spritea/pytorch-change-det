import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable


class fcUpConv3(nn.Module):
    def __init__(self, in_size, out_size):
        super(fcUpConv3, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_size,in_size//2,3,1,1),nn.BatchNorm2d(in_size//2),nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv2d(in_size//2,in_size//2,3,1,1),nn.BatchNorm2d(in_size//2),nn.ReLU())
        self.conv3=nn.Sequential(nn.Conv2d(in_size//2,out_size,3,1,1),nn.BatchNorm2d(out_size),nn.ReLU())
    def forward(self, input):
        output=self.conv1(input)
        output=self.conv2(output)
        output=self.conv3(output)
        return output

class fcUpConv2(nn.Module):
    def __init__(self, in_size, out_size):
        super(fcUpConv2, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_size,in_size//2,3,1,1),nn.BatchNorm2d(in_size//2),nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv2d(in_size//2,out_size,3,1,1),nn.BatchNorm2d(out_size),nn.ReLU())
    def forward(self, input):
        output=self.conv1(input)
        output=self.conv2(output)
        return output


class fcConv2(nn.Module):
    def __init__(self, in_size, out_size):
        super(fcConv2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, 1, 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 3, 1, 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class fcConv3(nn.Module):
    def __init__(self, in_size, out_size):
        super(fcConv3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, 1, 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 3, 1, 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 3, 1, 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs