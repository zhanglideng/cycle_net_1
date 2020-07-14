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
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)


class DenseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, drop_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes + 32)
        self.conv2 = nn.Conv2d(in_planes + 32, out_planes - in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=False, training=self.training)
        out = torch.cat([x, out], 1)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_planes, drop_rate):
        super(DenseBlock, self).__init__()
        self.DenseLayer1 = DenseLayer(in_planes, in_planes + 32 * 1, dropout=dropout)
        self.DenseLayer2 = DenseLayer(in_planes + 32 * 1, in_planes + 32 * 2, dropout=dropout)
        self.DenseLayer3 = DenseLayer(in_planes + 32 * 2, in_planes + 32 * 3, dropout=dropout)
        self.DenseLayer4 = DenseLayer(in_planes + 32 * 3, in_planes + 32 * 4, dropout=dropout)
        self.DenseLayer5 = DenseLayer(in_planes + 32 * 4, in_planes + 32 * 5, dropout=dropout)
        self.drop_rate = drop_rate

    def forward(self, x):
        x1 = self.DenseLayer1(x)
        x2 = self.DenseLayer2(x1)
        x3 = self.DenseLayer3(x2)
        x4 = self.DenseLayer4(x3)
        x5 = self.DenseLayer5(x4)
        return x5


class Dense_decoder(nn.Module):
    def __init__(self, out_channel, dropout):
        super(Dense_decoder, self).__init__()

        self.dense_block1 = DenseBlock(128 + 384, dropout=dropout)
        self.trans_block1 = TransitionBlock(128 + 384 + 32 * 5, 32 + 128, dropout=dropout)

        self.dense_block2 = DenseBlock(256 + 32, dropout=dropout)
        self.trans_block2 = TransitionBlock(256 + 32 + 32 * 5, 64, dropout=dropout)

        self.dense_block3 = DenseBlock(64, dropout=dropout)
        self.trans_block3 = TransitionBlock(64 + 32 * 5, 32, dropout=dropout)

        self.dense_block4 = DenseBlock(32, dropout=dropout)
        self.trans_block4 = TransitionBlock(32 + 32 * 5, 16, dropout=dropout)

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
    def __init__(self, dropout):
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
        self.dense_block4 = DenseBlock(896, dropout=dropout)  # 896, 256
        self.trans_block4 = TransitionBlock(896 + 32 * 5, 256, dropout=dropout)  # 1152, 128

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
    def __init__(self, dropout):
        super(cycle, self).__init__()
        self.encoder = Encoder(dropout=dropout)
        self.decoder = Dense_decoder(out_channel=3, dropout=dropout)

    def forward(self, x, hazy):
        x = torch.cat([x, hazy], 1)
        x1, x2, x4 = self.encoder(x)
        J = self.decoder(x, x1, x2, x4)
        return J
