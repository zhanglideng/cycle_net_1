import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, drop_rate):
        super(TransitionBlock, self).__init__()
        self.in1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.in1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)


class DenseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, drop_rate):
        super(DenseLayer, self).__init__()
        self.in1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(in_planes + 32)
        self.conv2 = nn.Conv2d(in_planes + 32, out_planes - in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.in1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=False, training=self.training)
        out = torch.cat([x, out], 1)
        out = self.conv2(self.relu(self.in2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_planes, drop_rate):
        super(DenseBlock, self).__init__()
        self.DenseLayer1 = DenseLayer(in_planes, in_planes + 32 * 1, drop_rate)
        self.DenseLayer2 = DenseLayer(in_planes + 32 * 1, in_planes + 32 * 2, drop_rate)
        self.DenseLayer3 = DenseLayer(in_planes + 32 * 2, in_planes + 32 * 3, drop_rate)
        self.DenseLayer4 = DenseLayer(in_planes + 32 * 3, in_planes + 32 * 4, drop_rate)
        self.DenseLayer5 = DenseLayer(in_planes + 32 * 4, in_planes + 32 * 5, drop_rate)
        self.drop_rate = drop_rate

    def forward(self, x):
        x1 = self.DenseLayer1(x)
        x2 = self.DenseLayer2(x1)
        x3 = self.DenseLayer3(x2)
        x4 = self.DenseLayer4(x3)
        x5 = self.DenseLayer5(x4)
        return x5


class Dense_decoder(nn.Module):
    def __init__(self, out_channel, drop_rate, norm_type):
        super(Dense_decoder, self).__init__()

        self.dense_block1 = DenseBlock(128 + 384, drop_rate)
        self.trans_block1 = TransitionBlock(128 + 384 + 32 * 5, 32 + 128, drop_rate)

        self.dense_block2 = DenseBlock(256 + 32, drop_rate)
        self.trans_block2 = TransitionBlock(256 + 32 + 32 * 5, 64, drop_rate)

        self.dense_block3 = DenseBlock(64, drop_rate)
        self.trans_block3 = TransitionBlock(64 + 32 * 5, 32, drop_rate)

        self.dense_block4 = DenseBlock(32, drop_rate)
        self.trans_block4 = TransitionBlock(32 + 32 * 5, 16, drop_rate)

        self.refine1 = nn.Conv2d(22, 20, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.refine2 = nn.Conv2d(20 + 4, 20, kernel_size=3, stride=1, padding=1)
        self.refine3 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)
        self.refine4 = nn.Conv2d(20, 20, kernel_size=7, stride=1, padding=3)
        self.refine5 = nn.Conv2d(20, out_channel, kernel_size=7, stride=1, padding=3)
        self.upsample = F.upsample
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, x1, x2, x4, activation=None):
        x42 = torch.cat([x4, x2], 1)
        x5 = self.trans_block1(self.dense_block1(x42))
        x52 = torch.cat([x5, x1], 1)
        x6 = self.trans_block2(self.dense_block2(x52))
        x7 = self.trans_block3(self.dense_block3(x6))
        x8 = self.trans_block4(self.dense_block4(x7))
        x8 = torch.cat([x8, x], 1)
        x9 = self.relu(self.refine1(x8))
        shape_out = x9.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out, mode='bilinear', align_corners=True)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out, mode='bilinear', align_corners=True)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out, mode='bilinear', align_corners=True)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out, mode='bilinear', align_corners=True)
        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        dehaze = self.tanh(self.refine2(dehaze))
        dehaze = self.relu(self.refine3(dehaze))
        dehaze = self.relu(self.refine4(dehaze))
        if activation == 'sig':
            dehaze = self.sig(self.refine5(dehaze))
        else:
            dehaze = self.refine5(dehaze)
        return dehaze


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate):
        super().__init__()
        # """In  our experiments, we let each 1×1 convolution
        # produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        # """We find this design especially effective for DenseNet and
        # we refer to our network with such a bottleneck layer, i.e.,
        # to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        # as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.InstanceNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.drop_rate > 0:
            out = F.dropout(self.bottle_neck(x), p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


# """We refer to layers between blocks as transition
# layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # """The transition layers used in our experiments
        # consist of a Instance normalization layer and an 1×1
        # convolutional layer followed by a 2×2 average pooling
        # layer""".
        self.down_sample = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)


# DesneNet-BC
# B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
# C stands for compression factor(0<=theta<=1)
class Encoder(nn.Module):
    def __init__(self, drop_rate, norm_type):
        super(Encoder, self).__init__()
        self.growth_rate = 32
        nblocks = [6, 12, 48]
        block = Bottleneck
        reduction = 0.5
        # """Before entering the first dense block, a convolution
        # with 16 (or twice the growth rate for DenseNet-BC)
        # output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        # For convolutional layers with kernel size 3×3, each
        # side of the inputs is zero-padded by one pixel to keep
        # the feature-map size fixed.
        self.conv1 = nn.Conv2d(6, inner_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.in0 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.feature1 = nn.Sequential()
        self.feature2 = nn.Sequential()
        self.feature3 = nn.Sequential()
        self.feature4 = nn.Sequential()

        # self.feature1.add_module("dense_block_layer_{}".format(0), self._make_dense_layers(block, inner_channels, nblocks[0]))
        self.feature1.add_module("dense_block{}".format(0),
                                 self._make_dense_layers(block, inner_channels, nblocks[0], drop_rate))
        inner_channels += growth_rate * nblocks[0]
        out_channels = int(reduction * inner_channels)  # int() will automatic floor the value
        self.feature1.add_module("transition_layer{}".format(0), Transition(inner_channels, out_channels))
        inner_channels = out_channels

        self.feature2.add_module("dense_block{}".format(1),
                                 self._make_dense_layers(block, inner_channels, nblocks[1], drop_rate))
        inner_channels += growth_rate * nblocks[1]
        out_channels = int(reduction * inner_channels)  # int() will automatic floor the value
        self.feature2.add_module("transition_layer{}".format(1), Transition(inner_channels, out_channels))
        inner_channels = out_channels

        self.feature3.add_module("dense_block{}".format(2),
                                 self._make_dense_layers(block, inner_channels, nblocks[2], drop_rate))
        inner_channels += growth_rate * nblocks[2]
        out_channels = int(reduction * inner_channels)  # int() will automatic floor the value
        self.feature3.add_module("transition_layer{}".format(2), Transition(inner_channels, out_channels))

        self.feature4.add_module("dense_block{}".format(3), DenseBlock(896, drop_rate))
        self.feature4.add_module("transition_layer{}".format(3), TransitionBlock(896 + 32 * 5, 256, drop_rate))

    def forward(self, x):
        x0 = self.pool(self.relu(self.in0(self.conv0(x))))
        x1 = self.feature1(x0)
        x2 = self.feature2(x1)
        x3 = self.feature3(x2)
        x4 = self.feature4(x3)
        return x1, x2, x4

    def _make_dense_layers(self, block, in_channels, nblocks, drop_rate):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index),
                                   block(in_channels, self.growth_rate, drop_rate=drop_rate))
            in_channels += self.growth_rate
        return dense_block


'''
class Encoder(nn.Module):
    def __init__(self, drop_rate, norm_type):
        super(Encoder, self).__init__()
        haze_class = densenet.DenseNet(drop_rate=drop_rate)
        print(haze_class)

        self.conv0 = haze_class.conv1
        self.in0 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense_block1 = haze_class.features.dense_block0
        self.trans_block1 = haze_class.features.transition_layer0

        self.dense_block2 = haze_class.features.dense_block1
        self.trans_block2 = haze_class.features.transition_layer1

        self.dense_block3 = haze_class.features.dense_block2
        self.trans_block3 = haze_class.features.transition_layer2

        self.dense_block4 = DenseBlock(896, drop_rate)  # 896, 256
        self.trans_block4 = TransitionBlock(896 + 32 * 5, 256, drop_rate)  # 1152, 128

    def forward(self, x, activation='sig'):
        x0 = self.pool(self.relu(self.in0(self.conv0(x))))
        x1 = self.trans_block1(self.dense_block1(x0))
        x2 = self.trans_block2(self.dense_block2(x1))
        x3 = self.trans_block3(self.dense_block3(x2))
        x4 = self.trans_block4(self.dense_block4(x3))
        return x1, x2, x4
'''


class cycle(nn.Module):
    def __init__(self, drop_rate, norm_type):
        super(cycle, self).__init__()
        self.encoder = Encoder(drop_rate=drop_rate, norm_type=norm_type)
        self.decoder = Dense_decoder(out_channel=3, drop_rate=drop_rate, norm_type=norm_type)

    def forward(self, x, hazy):
        x = torch.cat([x, hazy], 1)
        x1, x2, x4 = self.encoder(x)
        J = self.decoder(x, x1, x2, x4)
        return J
