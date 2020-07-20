import torch
from utils.ms_ssim import *
import math
from utils.vgg import Vgg16
import time
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

'''
vgg_net = Vgg16().type(torch.cuda.FloatTensor).cuda()
mse_loss = torch.nn.MSELoss().cuda()
l2_loss = torch.nn.MSELoss(reduction='mean').cuda()
ssim_loss = MS_SSIM(max_val=1, channel=3).cuda()
ssim_loss_1 = MS_SSIM(max_val=1, channel=1).cuda()

vgg = Vgg16().type(torch.cuda.FloatTensor).cuda()
loss_mse = torch.nn.MSELoss().cuda()
l2_loss_fn = torch.nn.MSELoss(reduction='mean').cuda()
losser = MS_SSIM(max_val=1, channel=3).cuda()
print('loss初始化成功！')



def l2_loss(output, gth):
    lo = l2_loss_fn(output, gth)
    return lo

def ssim_loss(output, gth, channel=3):
    lo = 1 - losser(output, gth)
    return lo


def vgg_loss(output, gth):
    output_features = vgg_net(output)
    gth_features = vgg_net(gth)
    sum_loss = mse_loss(output_features[0], gth_features[0]) * 0.25 \
               + mse_loss(output_features[1], gth_features[1]) * 0.25 \
               + mse_loss(output_features[2], gth_features[2]) * 0.25 \
               + mse_loss(output_features[3], gth_features[3]) * 0.25
    return sum_loss



def color_loss(input_image, output_image):
    vec1 = input_image.view([-1, 3])
    vec2 = output_image.view([-1, 3])
    clip_value = 0.999999
    norm_vec1 = torch.nn.functional.normalize(vec1)
    norm_vec2 = torch.nn.functional.normalize(vec2)
    dot = norm_vec1 * norm_vec2
    dot = dot.mean(dim=1)
    dot = torch.clamp(dot, -clip_value, clip_value)
    angle = torch.acos(dot) * (180 / math.pi)
    return angle.mean()



def loss_function(image, weight):
    J1, J2, J3, gt_image = image
    l2_1 = l2_loss(J1, gt_image)
    l2_2 = l2_loss(J2, gt_image)
    l2_3 = l2_loss(J3, gt_image)
    l2_2_1 = l2_2 - l2_1
    l2_3_2 = l2_3 - l2_2
    ssim_1 = 1 - ssim_loss(J1, gt_image)
    ssim_2 = 1 - ssim_loss(J2, gt_image)
    ssim_3 = 1 - ssim_loss(J3, gt_image)
    ssim_2_1 = ssim_2 - ssim_1
    ssim_3_2 = ssim_3 - ssim_2
    vgg_1 = vgg_loss(J1, gt_image)
    vgg_2 = vgg_loss(J2, gt_image)
    vgg_3 = vgg_loss(J3, gt_image)
    vgg_2_1 = vgg_2 - vgg_1
    vgg_3_2 = vgg_3 - vgg_2
    loss_train = [l2_1, l2_2, l2_3, l2_2_1, l2_3_2,
                  ssim_1, ssim_2, ssim_3, ssim_2_1, ssim_3_2,
                  vgg_1, vgg_2, vgg_3, vgg_2_1, vgg_3_2]
    loss_sum = 0
    for i in range(len(loss_train)):
        if loss_train[i] < 0:
            loss_sum = loss_sum + loss_train[i] * 0
        else:
            loss_sum = loss_sum + loss_train[i] * weight[i]
        loss_train[i] = loss_train[i].item()
    return loss_sum, loss_train


def loss_test(image):
    J1, J2, J3, J4, J5, gt_image = image
    loss_train = [l2_loss(J1, gt_image),
                  l2_loss(J2, gt_image),
                  l2_loss(J3, gt_image),
                  l2_loss(J4, gt_image),
                  l2_loss(J5, gt_image),
                  1 - ssim_loss(J1, gt_image),
                  1 - ssim_loss(J2, gt_image),
                  1 - ssim_loss(J3, gt_image),
                  1 - ssim_loss(J4, gt_image),
                  1 - ssim_loss(J5, gt_image),
                  vgg_loss(J1, gt_image),
                  vgg_loss(J2, gt_image),
                  vgg_loss(J3, gt_image),
                  vgg_loss(J4, gt_image),
                  vgg_loss(J5, gt_image),
                  l2_loss(J1, J2),
                  l2_loss(J2, J3),
                  l2_loss(J3, J4),
                  l2_loss(J4, J5),
                  1 - ssim_loss(J1, J2),
                  1 - ssim_loss(J2, J3),
                  1 - ssim_loss(J3, J4),
                  1 - ssim_loss(J4, J5),
                  vgg_loss(J1, J2),
                  vgg_loss(J2, J3),
                  vgg_loss(J3, J4),
                  vgg_loss(J4, J5)]
    for i in range(len(loss_train)):
        loss_train[i] = loss_train[i].item()
    return loss_train
'''

weight_base = 1 * torch.ones(1)
weight_base = weight_base.cuda()


class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, x1, x2):
        x1 = x1 - x2
        x1 = abs(x1)
        return x1


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, x1, x2):
        x1 = x1 - x2
        x1 = x1.pow(2)
        return x1


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # features = torch.load(pre_densenet201).features
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = [h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3]
        return out


class MS_SSIM(nn.Module):
    def __init__(self, size_average=False, max_val=255, channel=3):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = channel
        self.max_val = max_val

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    @staticmethod
    def create_window(window_size, sigma, channel):
        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def ssim(self, img1, img2, size_average=True):
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).cuda()
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=self.channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=self.channel) - mu1_mu2
        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()
        else:
            return ssim_map, mcs_map

    def ms_ssim(self, img1, img2, levels=5):
        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())
        msssim = Variable(torch.Tensor(levels, ).cuda())
        mcs = Variable(torch.Tensor(levels, ).cuda())
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2
        value = (torch.prod(mcs[0:levels - 1] ** weight[0:levels - 1]) *
                 (msssim[levels - 1] ** weight[levels - 1]))
        return value

    def forward(self, img1, img2):
        ssim, mcs = self.ssim(img1, img2)
        return ssim


class vgg_loss(nn.Module):
    def __init__(self):
        super(vgg_loss, self).__init__()
        # self.vgg = Vgg16().type(torch.cuda.FloatTensor)
        self.vgg = VGG16()
        self.l2 = MSE()

    def forward(self, hazy, gth):
        output_features = self.vgg(hazy)
        gth_features = self.vgg(gth)
        out_1 = self.l2(output_features[0], gth_features[0])
        out_2 = self.l2(output_features[1], gth_features[1])
        out_3 = self.l2(output_features[2], gth_features[2])
        out_4 = self.l2(output_features[3], gth_features[3])
        return out_1, out_2, out_3, out_4


class train_loss_net(nn.Module):
    def __init__(self, channel=3, pixel_loss='MSE'):
        super(train_loss_net, self).__init__()
        if pixel_loss == 'MSE':
            self.pixel_loss = MSE()
        else:
            self.pixel_loss = MAE()
        self.ssim = MS_SSIM(max_val=1, channel=channel)
        self.vgg_loss = vgg_loss()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, j1, j2, j3, gth, weight):
        l2_1 = self.pixel_loss(j1, gth)
        l2_2 = self.pixel_loss(j2, gth)
        l2_3 = self.pixel_loss(j3, gth)

        ssim_1 = 1 - self.ssim(j1, gth)
        ssim_2 = 1 - self.ssim(j2, gth)
        ssim_3 = 1 - self.ssim(j3, gth)

        vgg_1_1, vgg_1_2, vgg_1_3, vgg_1_4 = self.vgg_loss(j1, gth)
        vgg_2_1, vgg_2_2, vgg_2_3, vgg_2_4 = self.vgg_loss(j2, gth)
        vgg_3_1, vgg_3_2, vgg_3_3, vgg_3_4 = self.vgg_loss(j3, gth)
        # print(vgg_1_1)

        l2_map = [l2_1, l2_2, l2_3, self.relu(l2_2 - l2_1), self.relu(l2_3 - l2_2)]
        ssim_map = [ssim_1, ssim_2, ssim_3, self.relu(ssim_2 - ssim_1), self.relu(ssim_3 - ssim_2)]
        vgg_map = [vgg_1_1, vgg_1_2, vgg_1_3, vgg_1_4, vgg_2_1, vgg_2_2,
                   vgg_2_3, vgg_2_4, vgg_3_1, vgg_3_2, vgg_3_3, vgg_3_4,
                   self.relu(vgg_2_1 - vgg_1_1), self.relu(vgg_2_2 - vgg_1_2),
                   self.relu(vgg_2_3 - vgg_1_3), self.relu(vgg_2_4 - vgg_1_4),
                   self.relu(vgg_3_1 - vgg_2_1), self.relu(vgg_3_2 - vgg_2_2),
                   self.relu(vgg_3_3 - vgg_2_3), self.relu(vgg_3_4 - vgg_2_4)]
        loss_for_train = 0
        loss_for_save = [0] * len(weight)
        for i in range(len(l2_map)):
            loss_for_train = loss_for_train + torch.mean(l2_map[i]) * weight[i]
            loss_for_save[i] = torch.mean(l2_map[i]).item()
            # print(loss_for_save)

        for i in range(len(ssim_map)):
            loss_for_train = loss_for_train + torch.mean(ssim_map[i]) * weight[i + len(l2_map)]
            loss_for_save[i + len(l2_map)] = torch.mean(ssim_map[i]).item()
            # print(ssim_map[i])

        for i in range(len(vgg_map)):
            loss_for_train = loss_for_train + torch.mean(vgg_map[i]) * weight[int(i / 4)] * 0.25
            loss_for_save[int(i / 4) + len(l2_map) + len(ssim_map)] = loss_for_save[int(i / 4)] + torch.mean(
                vgg_map[i]).item()
            # print(loss_for_save)

        return loss_for_train, loss_for_save


class test_loss_net(nn.Module):
    def __init__(self, channel=3):
        super(test_loss_net, self).__init__()
        self.l2 = MSE()
        self.ssim = MS_SSIM(max_val=1, channel=channel)
        self.vgg_loss = vgg_loss()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, j1, j2, j3, j4, j5, gth):
        l2_1 = self.l2(j1, gth)
        l2_2 = self.l2(j2, gth)
        l2_3 = self.l2(j3, gth)
        l2_4 = self.l2(j4, gth)
        l2_5 = self.l2(j5, gth)

        ssim_1 = 1 - self.ssim(j1, gth)
        ssim_2 = 1 - self.ssim(j2, gth)
        ssim_3 = 1 - self.ssim(j3, gth)
        ssim_4 = 1 - self.ssim(j4, gth)
        ssim_5 = 1 - self.ssim(j5, gth)

        vgg_1_1, vgg_1_2, vgg_1_3, vgg_1_4 = self.vgg_loss(j1, gth)
        vgg_2_1, vgg_2_2, vgg_2_3, vgg_2_4 = self.vgg_loss(j2, gth)
        vgg_3_1, vgg_3_2, vgg_3_3, vgg_3_4 = self.vgg_loss(j3, gth)
        vgg_4_1, vgg_4_2, vgg_4_3, vgg_4_4 = self.vgg_loss(j4, gth)
        vgg_5_1, vgg_5_2, vgg_5_3, vgg_5_4 = self.vgg_loss(j5, gth)
        # print(vgg_1_1)

        l2_map = [l2_1, l2_2, l2_3, l2_4, l2_5, self.relu(l2_2 - l2_1), self.relu(l2_3 - l2_2), self.relu(l2_4 - l2_3),
                  self.relu(l2_5 - l2_4)]
        ssim_map = [ssim_1, ssim_2, ssim_3, ssim_4, ssim_5, self.relu(ssim_2 - ssim_1), self.relu(ssim_3 - ssim_2),
                    self.relu(ssim_4 - ssim_3), self.relu(ssim_5 - ssim_4)]
        vgg_map = [vgg_1_1, vgg_1_2, vgg_1_3, vgg_1_4,
                   vgg_2_1, vgg_2_2, vgg_2_3, vgg_2_4,
                   vgg_3_1, vgg_3_2, vgg_3_3, vgg_3_4,
                   vgg_4_1, vgg_4_2, vgg_4_3, vgg_4_4,
                   vgg_5_1, vgg_5_2, vgg_5_3, vgg_5_4,
                   self.relu(vgg_2_1 - vgg_1_1), self.relu(vgg_2_2 - vgg_1_2),
                   self.relu(vgg_2_3 - vgg_1_3), self.relu(vgg_2_4 - vgg_1_4),
                   self.relu(vgg_3_1 - vgg_2_1), self.relu(vgg_3_2 - vgg_2_2),
                   self.relu(vgg_3_3 - vgg_2_3), self.relu(vgg_3_4 - vgg_2_4),
                   self.relu(vgg_4_1 - vgg_3_1), self.relu(vgg_4_2 - vgg_3_2),
                   self.relu(vgg_4_3 - vgg_3_3), self.relu(vgg_4_4 - vgg_3_4),
                   self.relu(vgg_5_1 - vgg_4_1), self.relu(vgg_5_2 - vgg_4_2),
                   self.relu(vgg_5_3 - vgg_4_3), self.relu(vgg_5_4 - vgg_4_4)
                   ]
        loss_for_save = [0] * 27
        for i in range(len(l2_map)):
            loss_for_save[i] = torch.mean(l2_map[i]).item()
            # print(loss_for_save)

        for i in range(len(ssim_map)):
            loss_for_save[i + len(l2_map)] = torch.mean(ssim_map[i]).item()
            # print(ssim_map[i])

        for i in range(len(vgg_map)):
            loss_for_save[int(i / 4) + len(l2_map) + len(ssim_map)] = loss_for_save[int(i / 4)] + torch.mean(
                vgg_map[i]).item()
            # print(loss_for_save)
        return loss_for_save


'''
class test_loss_net(nn.Module):
    def __init__(self, channel=3):
        super(test_loss_net, self).__init__()
        self.l2 = nn.MSELoss()
        self.ssim = MS_SSIM(max_val=1, channel=channel)
        self.vgg = vgg_loss()

    def forward(self, result, gth, weight):
        l2_loss = self.l2(result, gth)
        l2_loss = torch.mean(l2_loss)
        ssim_loss = 1 - self.ssim(result, gth)
        ssim_loss = torch.mean(ssim_loss)
        result_feature, gth_feature = self.vgg(result, gth)
        vgg_loss = 0
        for i in range(len(result_feature)):
            temp_loss = self.l2(result_feature[i], gth_feature[i])
            temp_loss = torch.mean(temp_loss)
            vgg_loss = vgg_loss + temp_loss
        loss_sum = l2_loss * weight[0] + ssim_loss * weight[1] + vgg_loss * weight[2]
        loss_save = [l2_loss.item(), ssim_loss.item(), vgg_loss.item()]
        return loss_sum, loss_save
'''
