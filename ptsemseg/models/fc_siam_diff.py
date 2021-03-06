
from ptsemseg.models.fc_ef_utils import *
from ptsemseg.models.fc_ef import fc_ef_Up_conv3,\
    fc_ef_Up_conv2,fc_ef_Up_conv2_classify


class fc_siam_diff(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        num_classes=49,
        is_deconv=True,
        in_channels=3,
    ):
        super(fc_siam_diff, self).__init__()
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
    #the same name layer parameter between different class(nn.module) is different
    #the same name layer parameter in __init__() of class(nn.module) is different
    #the same name layer parameter in forward() is same
    def encoder_left(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        return conv1,conv2,conv3,conv4,maxpool4

    def encoder_right(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        # maxpool4 = self.maxpool4(conv4)

        return conv1,conv2,conv3,conv4
    #input_left=input2=image_dst
    #input_right=input1=image_src
    def forward(self, input_right,input_left):
        conv1_l,conv2_l,conv3_l,conv4_l,maxpool4=self.encoder_left(input_left)
        conv1_r,conv2_r,conv3_r,conv4_r=self.encoder_right(input_right)

        up4 = self.up_concat4(maxpool4, conv4_l-conv4_r)
        up3 = self.up_concat3(up4, conv3_l-conv3_r)
        up2 = self.up_concat2(up3, conv2_l-conv2_r)
        up1 = self.up_concat1(up2, conv1_l-conv1_r)

        return up1
