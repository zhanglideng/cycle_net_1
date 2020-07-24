"""dense net in pytorch



[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.

    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
"""

import torch
import torch.nn as nn


# """Bottleneck layers. Although each layer only produces k
# output feature-maps, it typically has many more inputs. It
# has been noted in [37, 11] that a 1×1 convolution can be in-
# troduced as bottleneck layer before each 3×3 convolution
# to reduce the number of input feature-maps, and thus to
# improve computational efficiency."""
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout):
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
        self.dropout = dropout

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
class DenseNet(nn.Module):
    def __init__(self, dropout, block=Bottleneck, nblocks=[6, 12, 48], growth_rate=32, reduction=0.5, ):
        super().__init__()
        self.growth_rate = growth_rate

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

        # self.feature1.add_module("dense_block_layer_{}".format(0), self._make_dense_layers(block, inner_channels, nblocks[0]))
        self.feature1.add_module("dense_block{}".format(0),
                                 self._make_dense_layers(block, inner_channels, nblocks[0], dropout))
        inner_channels += growth_rate * nblocks[0]
        out_channels = int(reduction * inner_channels)  # int() will automatic floor the value
        self.feature1.add_module("transition_layer{}".format(0), Transition(inner_channels, out_channels))
        inner_channels = out_channels

        self.feature2.add_module("dense_block{}".format(1),
                                 self._make_dense_layers(block, inner_channels, nblocks[1], dropout))
        inner_channels += growth_rate * nblocks[1]
        out_channels = int(reduction * inner_channels)  # int() will automatic floor the value
        self.feature2.add_module("transition_layer{}".format(1), Transition(inner_channels, out_channels))
        inner_channels = out_channels

        self.feature3.add_module("dense_block{}".format(2),
                                 self._make_dense_layers(block, inner_channels, nblocks[2], dropout))
        inner_channels += growth_rate * nblocks[2]
        out_channels = int(reduction * inner_channels)  # int() will automatic floor the value
        self.feature3.add_module("transition_layer{}".format(2), Transition(inner_channels, out_channels))

    def forward(self, x):

        x0 = self.pool(self.relu(self.in0(self.conv0(x))))
        x1 = self.trans_block1(self.dense_block1(x0))
        x2 = self.trans_block2(self.dense_block2(x1))
        x3 = self.trans_block3(self.dense_block3(x2))
        x4 = self.trans_block4(self.dense_block4(x3))
        return x1, x2, x4

        x0 = self.pool(self.relu(self.in0(self.conv0(x))))
        x1 = self.feature1(output)
        x2 = self.feature2(output)
        x3 = self.feature3(output)
        return x1, x2, x4


    def _make_dense_layers(self, block, in_channels, nblocks, dropout):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate, dropout=dropout))
            in_channels += self.growth_rate
        return dense_block
