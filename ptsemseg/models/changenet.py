import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

IMG_SCALE = 1. / 255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os, sys
    from six.moves import urllib
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)

def batchnorm(in_planes):
    "batch norm 2d"
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)
    # return apex.parallel.SyncBatchNorm(in_planes, affine=True, eps=1e-5, momentum=0.1)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups,
                      bias=False),
            batchnorm(out_planes),
            nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups,
                      bias=False),
            batchnorm(out_planes))


stages_suffixes = {0: '_conv',
                   1: '_conv_relu_varout_dimred'}

data_info = {
    21: 'VOC',
}

models_urls = {
    '101_voc': 'https://cloudstor.aarnet.edu.au/plus/s/Owmttk9bdPROwc6/download',

    '101_imagenet': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    '18_imagenet': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '50_imagenet': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = apex.parallel.SyncBatchNorm(planes)
        # self.bn1=encoding.nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = apex.parallel.SyncBatchNorm(planes)
        # self.bn2=encoding.nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = apex.parallel.SyncBatchNorm(planes)
        # self.bn1=encoding.nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = apex.parallel.SyncBatchNorm(planes)
        # self.bn2=encoding.nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # self.bn3 = apex.parallel.SyncBatchNorm(planes*4)
        # self.bn3=encoding.nn.BatchNorm2d(planes*4)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DeconvNet(nn.Module):
    def __init__(self, in_size, n_class):
        super(DeconvNet, self).__init__()
        self.full_conv1 = conv1x1(in_size, n_class)
        self.full_conv2 = conv1x1(in_size * 2, n_class)
        self.full_conv3 = conv1x1(in_size * 4, n_class)
        self.deconv1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.deconv2 = nn.Upsample(scale_factor=8 * 2, mode='bilinear', align_corners=True)
        self.deconv3 = nn.Upsample(scale_factor=8 * 4, mode='bilinear', align_corners=True)

    def forward(self, l3, l4, l5):
        # l3 of cp3 in the paper=layer2 in resnet
        l3_out = self.full_conv1(l3)
        l3_out = self.deconv1(l3_out)
        l4_out = self.full_conv2(l4)
        l4_out = self.deconv2(l4_out)
        l5_out = self.full_conv3(l5)
        l5_out = self.deconv3(l5_out)
        return l3_out, l4_out, l5_out


class final_full_conv(nn.Module):
    def __init__(self, in_size, n_class):
        super(final_full_conv, self).__init__()
        self.full_conv1 = conv1x1(in_size, n_class)
        self.full_conv2 = conv1x1(in_size, n_class)
        self.full_conv3 = conv1x1(in_size, n_class)

    def forward(self, l3_out_src, l3_out_dst, l4_out_src, l4_out_dst,
                l5_out_src, l5_out_dst):
        l3_cat = torch.cat([l3_out_src, l3_out_dst], dim=1)
        l3_cat_out = self.full_conv1(l3_cat)
        l4_cat = torch.cat([l4_out_src, l4_out_dst], dim=1)
        l4_cat_out = self.full_conv1(l4_cat)
        l5_cat = torch.cat([l5_out_src, l5_out_dst], dim=1)
        l5_cat_out = self.full_conv1(l5_cat)
        sum = l3_cat_out + l4_cat_out + l5_cat_out
        return sum


class ChangeNet(nn.Module):

    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        super(ChangeNet, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn1 = apex.parallel.SyncBatchNorm(64)
        # self.bn1=encoding.nn.SyncBatchNorm(64)
        # self.bn1=encoding.nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.deconv1 = DeconvNet(512, num_classes)
        self.deconv2 = DeconvNet(512, num_classes)
        self.post_process = final_full_conv(num_classes * 2, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                # apex.parallel.SyncBatchNorm(planes * block.expansion),
                # encoding.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def backbone(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return l2, l3, l4

    def forward(self, x_src, x_dst):
        l2_src, l3_src, l4_src = self.backbone(x_src)
        l2_dst, l3_dst, l4_dst = self.backbone(x_src)
        l3_out_src, l4_out_src, l5_out_src = self.deconv1(l2_src, l3_src, l4_src)
        l3_out_dst, l4_out_dst, l5_out_dst = self.deconv2(l2_dst, l3_dst, l4_dst)
        out = self.post_process(l3_out_src, l3_out_dst, l4_out_src, l4_out_dst,
                                l5_out_src, l5_out_dst)
        return out

def changenet101(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = ChangeNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
    if imagenet:
        key = '101_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = '101_' + dataset.lower()
            key = 'rf' + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def changenet50(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = ChangeNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    if imagenet:
        key = '50_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = '101_' + dataset.lower()
            key = 'rf' + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model
