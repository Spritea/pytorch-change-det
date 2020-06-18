
from ptsemseg.models.fc_ef_utils import *


class fc_ef_Up_conv3(nn.Module):
    def __init__(self,in_size,out_size):
        super(fc_ef_Up_conv3,self).__init__()
        self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)
        self.conv=fcUpConv3(in_size*2,out_size)
    def forward(self, input_small,input_big):
        output1=self.up(input_small)
        output_cat=torch.cat([output1, input_big], dim=1)
        output2=self.conv(output_cat)
        return output2

class fc_ef_Up_conv2(nn.Module):
    def __init__(self,in_size,out_size):
        super(fc_ef_Up_conv2,self).__init__()
        self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)
        self.conv=fcUpConv2(in_size*2,out_size)
    def forward(self, input_small,input_big):
        output1=self.up(input_small)
        output_cat=torch.cat([output1, input_big], dim=1)
        output2=self.conv(output_cat)
        return output2

class fc_ef_Up_conv2_classify(nn.Module):
    def __init__(self,in_size,class_number):
        super(fc_ef_Up_conv2_classify,self).__init__()
        self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_size*2, in_size, 3, 1, 1), nn.BatchNorm2d(in_size), nn.ReLU())
        self.conv2 = nn.Conv2d(in_size, class_number, 1)

    def forward(self, input_small,input_big):
        output1=self.up(input_small)
        output_cat=torch.cat([output1, input_big], dim=1)
        output2=self.conv1(output_cat)
        output2=self.conv2(output2)
        return output2


class fc_ef(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        num_classes=49,
        is_deconv=True,
        in_channels=6,
    ):
        super(fc_ef, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.feature_scale = feature_scale

        # filters = [64, 128, 256, 512, 1024]
        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = fcConv2(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = fcConv2(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = fcConv2(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = fcConv3(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = fc_ef_Up_conv3(filters[3], filters[2])
        self.up_concat3 = fc_ef_Up_conv3(filters[2], filters[1])
        self.up_concat2 = fc_ef_Up_conv2(filters[1], filters[0])
        self.up_concat1 = fc_ef_Up_conv2_classify(filters[0],num_classes)

        # final conv (without any concat)
        # self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)


        # center = self.center(maxpool4)
        up4 = self.up_concat4(maxpool4, conv4)
        up3 = self.up_concat3(up4, conv3)
        up2 = self.up_concat2(up3, conv2)
        up1 = self.up_concat1(up2, conv1)

        # final = self.final(up1)

        return up1
