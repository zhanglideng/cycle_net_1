import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, drop_rate, norm_type):
        super(TransitionBlock, self).__init__()

        self.transition_block = nn.Sequential()
        if norm_type == 'IN':
            self.transition_block.add_module('in0', nn.InstanceNorm2d(in_planes))
        else:
            self.transition_block.add_module('bn0', nn.BatchNorm2d(in_planes))

        self.transition_block.add_module('relu0', nn.ReLU(inplace=True))
        self.transition_block.add_module('dconv0', nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                                                      padding=0, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.drop_rate > 0:
            out = F.dropout(self.transition_block(x), p=self.drop_rate, inplace=False, training=self.training)
        else:
            out = self.transition_block(x)
        return F.upsample_nearest(out, scale_factor=2)


class DenseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, drop_rate, norm_type):
        super(DenseLayer, self).__init__()

        self.dense_layer0 = nn.Sequential()
        self.dense_layer1 = nn.Sequential()
        if norm_type == 'IN':
            self.dense_layer0.add_module('in0', nn.InstanceNorm2d(in_planes))
        else:
            self.dense_layer0.add_module('bn0', nn.BatchNorm2d(in_planes))
        self.dense_layer0.add_module('relu0', nn.ReLU(inplace=True))
        self.dense_layer0.add_module('conv0', nn.Conv2d(in_planes, 32, kernel_size=3, stride=1, padding=1, bias=False))
        if norm_type == 'IN':
            self.dense_layer1.add_module('in1', nn.InstanceNorm2d(in_planes + 32))
        else:
            self.dense_layer1.add_module('bn1', nn.BatchNorm2d(in_planes + 32))
        self.dense_layer1.add_module('relu1', nn.ReLU(inplace=True))
        self.dense_layer1.add_module('conv1', nn.Conv2d(in_planes + 32, out_planes - in_planes, kernel_size=3, stride=1,
                                                        padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        # 有个担忧：这里的dropout次数是否过多？
        out = self.dense_layer0(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=False, training=self.training)
        out = torch.cat([x, out], 1)
        out = self.dense_layer1(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)
'''
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate, norm_type):
        super().__init__()
        inner_channel = 4 * growth_rate

        self.bottle_neck = nn.Sequential()
        if norm_type == 'IN':
            self.bottle_neck.add_module('in0', nn.InstanceNorm2d(in_channels))
        else:
            self.bottle_neck.add_module('bn0', nn.BatchNorm2d(in_channels))
        self.bottle_neck.add_module('relu0', nn.ReLU(inplace=True))
        self.bottle_neck.add_module('conv0', nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False))
        if norm_type == 'IN':
            self.bottle_neck.add_module('in1', nn.InstanceNorm2d(inner_channel))
        else:
            self.bottle_neck.add_module('bn1', nn.BatchNorm2d(inner_channel))
        self.bottle_neck.add_module('relu1', nn.ReLU(inplace=True))
        self.bottle_neck.add_module('conv1',
                                    nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        if self.drop_rate > 0:
            out = F.dropout(self.bottle_neck(x), p=self.drop_rate, training=self.training)
        else:
            out = self.bottle_neck(x)
        return torch.cat([x, out], 1)
'''


class DenseBlock(nn.Module):
    def __init__(self, in_planes, re_rate, growth_rate, drop_rate, norm_type):
        # re_rate表示使用几个DenseLayer
        super(DenseBlock, self).__init__()

        self.dense_block = nn.Sequential()
        for i in range(re_rate):
            self.dense_block.add_module('dense_layer{}'.format(i),
                                        DenseLayer(in_planes + growth_rate * i, in_planes + growth_rate * (i + 1),
                                                   drop_rate, norm_type))

    def forward(self, x):
        return self.dense_block(x)


class Dense_decoder(nn.Module):
    def __init__(self, out_channel, drop_rate, norm_type):
        super(Dense_decoder, self).__init__()

        self.dense_decoder0 = nn.Sequential(
            DenseBlock(in_planes=128 + 384, re_rate=5, growth_rate=32, drop_rate=drop_rate, norm_type=norm_type),
            TransitionBlock(in_planes=128 + 384 + 32 * 5, out_planes=32 + 128, drop_rate=drop_rate,
                            norm_type=norm_type))

        self.dense_decoder1 = nn.Sequential(
            DenseBlock(in_planes=256 + 32, re_rate=5, growth_rate=32, drop_rate=drop_rate,
                       norm_type=norm_type),
            TransitionBlock(in_planes=256 + 32 + 32 * 5, out_planes=64, drop_rate=drop_rate,
                            norm_type=norm_type))

        self.dense_decoder2 = nn.Sequential(
            DenseBlock(in_planes=64, re_rate=5, growth_rate=32, drop_rate=drop_rate, norm_type=norm_type),
            TransitionBlock(in_planes=64 + 32 * 5, out_planes=32, drop_rate=drop_rate, norm_type=norm_type))

        self.dense_decoder3 = nn.Sequential(
            DenseBlock(in_planes=32, re_rate=5, growth_rate=32, drop_rate=drop_rate,
                       norm_type=norm_type),
            TransitionBlock(in_planes=32 + 32 * 5, out_planes=16, drop_rate=drop_rate,
                            norm_type=norm_type))

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
        x5 = self.dense_decoder0(x42)
        x52 = torch.cat([x5, x1], 1)
        x6 = self.dense_decoder1(x52)
        x7 = self.dense_decoder2(x6)
        x8 = self.dense_decoder3(x7)
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
    def __init__(self, in_channels, growth_rate, drop_rate, norm_type):
        super().__init__()
        inner_channel = 4 * growth_rate

        self.bottle_neck = nn.Sequential()
        if norm_type == 'IN':
            self.bottle_neck.add_module('in0', nn.InstanceNorm2d(in_channels))
        else:
            self.bottle_neck.add_module('bn0', nn.BatchNorm2d(in_channels))
        self.bottle_neck.add_module('relu0', nn.ReLU(inplace=True))
        self.bottle_neck.add_module('conv0', nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False))
        if norm_type == 'IN':
            self.bottle_neck.add_module('in1', nn.InstanceNorm2d(inner_channel))
        else:
            self.bottle_neck.add_module('bn1', nn.BatchNorm2d(inner_channel))
        self.bottle_neck.add_module('relu1', nn.ReLU(inplace=True))
        self.bottle_neck.add_module('conv1',
                                    nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        if self.drop_rate > 0:
            out = F.dropout(self.bottle_neck(x), p=self.drop_rate, training=self.training)
        else:
            out = self.bottle_neck(x)
        return torch.cat([x, out], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type):
        super().__init__()

        self.down_sample = nn.Sequential()
        if norm_type == 'IN':
            self.down_sample.add_module('in0', nn.InstanceNorm2d(in_channels))
        else:
            self.down_sample.add_module('bn0', nn.BatchNorm2d(in_channels))
        self.down_sample.add_module('conv0', nn.Conv2d(in_channels, out_channels, 1, bias=False))
        self.down_sample.add_module('pool0', nn.AvgPool2d(2, stride=2))

    def forward(self, x):
        return self.down_sample(x)


class Encoder(nn.Module):
    def __init__(self, drop_rate, norm_type):
        super(Encoder, self).__init__()
        growth_rate = 32
        self.growth_rate = 32
        nblocks = [6, 12, 48]
        block = Bottleneck
        reduction = 0.5
        inner_channels = 2 * growth_rate

        self.feature0 = nn.Sequential()
        self.feature1 = nn.Sequential()
        self.feature2 = nn.Sequential()
        self.feature3 = nn.Sequential()
        self.feature4 = nn.Sequential()

        self.feature0.add_module("conv0", nn.Conv2d(6, inner_channels, kernel_size=3, stride=2, padding=1, bias=False))
        if norm_type == 'IN':
            self.feature0.add_module("in0", nn.InstanceNorm2d(64))
        else:
            self.feature0.add_module("bn0", nn.BatchNorm2d(64))
        self.feature0.add_module("relu0", nn.ReLU(inplace=True))
        self.feature0.add_module("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.feature1.add_module("dense_block{}".format(0),
                                 self._make_dense_layers(block, inner_channels, nblocks[0], drop_rate, norm_type))
        inner_channels += growth_rate * nblocks[0]
        out_channels = int(reduction * inner_channels)  # int() will automatic floor the value
        self.feature1.add_module("transition_layer{}".format(0), Transition(inner_channels, out_channels, norm_type))
        inner_channels = out_channels

        self.feature2.add_module("dense_block{}".format(1),
                                 self._make_dense_layers(block, inner_channels, nblocks[1], drop_rate, norm_type))
        inner_channels += growth_rate * nblocks[1]
        out_channels = int(reduction * inner_channels)  # int() will automatic floor the value
        self.feature2.add_module("transition_layer{}".format(1), Transition(inner_channels, out_channels, norm_type))
        inner_channels = out_channels

        self.feature3.add_module("dense_block{}".format(2),
                                 self._make_dense_layers(block, inner_channels, nblocks[2], drop_rate, norm_type))
        inner_channels += growth_rate * nblocks[2]
        out_channels = int(reduction * inner_channels)  # int() will automatic floor the value
        self.feature3.add_module("transition_layer{}".format(2), Transition(inner_channels, out_channels, norm_type))

        self.feature4.add_module("dense_block{}".format(3), DenseBlock(896, 5, 32, drop_rate, norm_type))
        self.feature4.add_module("transition_layer{}".format(3),
                                 TransitionBlock(896 + 32 * 5, 256, drop_rate, norm_type))

    def forward(self, x):
        x0 = self.feature0(x)
        x1 = self.feature1(x0)
        x2 = self.feature2(x1)
        x3 = self.feature3(x2)
        x4 = self.feature4(x3)
        return x1, x2, x4

    def _make_dense_layers(self, block, in_channels, nblocks, drop_rate, norm_type):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index),
                                   block(in_channels, self.growth_rate, drop_rate=drop_rate, norm_type=norm_type))
            in_channels += self.growth_rate
        return dense_block


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
