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
    def __init__(self, out_channel, drop_rate):
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


class Encoder(nn.Module):
    def __init__(self, drop_rate):
        super(Encoder, self).__init__()
        ############# 256-256  ##############
        haze_class = models.densenet201(pretrained=True)

        self.conv0 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = haze_class.features.norm0
        self.relu0 = haze_class.features.relu0
        self.pool0 = haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1 = haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2 = haze_class.features.denseblock2
        self.trans_block2 = haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3 = haze_class.features.denseblock3
        self.trans_block3 = haze_class.features.transition3

        # 这里不继续用DenseNet的原因是DenseNet的深层信息是分类信息，无助于去雾
        ############# Block4-up  8-8  ##############
        self.dense_block4 = DenseBlock(896, drop_rate)  # 896, 256
        self.trans_block4 = TransitionBlock(896 + 32 * 5, 256, drop_rate)  # 1152, 128

    def forward(self, x, activation='sig'):
        # 608 X 448
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(x))))
        # 152 X 112
        x1 = self.dense_block1(x0)
        x1 = self.trans_block1(x1)
        # 72 X 56
        x2 = self.trans_block2(self.dense_block2(x1))
        # 36 X 28
        x3 = self.trans_block3(self.dense_block3(x2))
        # 18 X 14
        x4 = self.trans_block4(self.dense_block4(x3))
        # 36 X 28
        return x1, x2, x4


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


'''
# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
'''