import torch
from utils.ms_ssim import *
import math
import torch.nn.functional
from utils.vgg import Vgg16
import time

vgg_net = Vgg16().type(torch.cuda.FloatTensor).cuda()
mse_loss = torch.nn.MSELoss().cuda()
l2_loss = torch.nn.MSELoss(reduction='mean').cuda()
ssim_loss = MS_SSIM(max_val=1, channel=3).cuda()
ssim_loss_1 = MS_SSIM(max_val=1, channel=1).cuda()
'''
vgg = Vgg16().type(torch.cuda.FloatTensor).cuda()
loss_mse = torch.nn.MSELoss().cuda()
l2_loss_fn = torch.nn.MSELoss(reduction='mean').cuda()
losser = MS_SSIM(max_val=1, channel=3).cuda()
print('loss初始化成功！')
'''

'''
def l2_loss(output, gth):
    lo = l2_loss_fn(output, gth)
    return lo

def ssim_loss(output, gth, channel=3):
    lo = 1 - losser(output, gth)
    return lo
'''


def vgg_loss(output, gth):
    output_features = vgg_net(output)
    gth_features = vgg_net(gth)
    sum_loss = mse_loss(output_features[0], gth_features[0]) * 0.25 \
               + mse_loss(output_features[1], gth_features[1]) * 0.25 \
               + mse_loss(output_features[2], gth_features[2]) * 0.25 \
               + mse_loss(output_features[3], gth_features[3]) * 0.25
    return sum_loss


'''
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
'''


def loss_function(image, weight):
    J1, J2, J3, gt_image = image
    loss_train = [l2_loss(J1, gt_image),
                  1 - ssim_loss(J1, gt_image),
                  vgg_loss(J1, gt_image),
                  l2_loss(J2, gt_image),
                  1 - ssim_loss(J2, gt_image),
                  vgg_loss(J2, gt_image),
                  l2_loss(J3, gt_image),
                  1 - ssim_loss(J3, gt_image),
                  vgg_loss(J3, gt_image),
                  l2_loss(J4, gt_image),
                  1 - ssim_loss(J4, gt_image),
                  vgg_loss(J4, gt_image)]
    loss_sum = 0
    for i in range(len(loss_train)):
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
                  vgg_loss(J4, J5),
                  ]
    for i in range(len(loss_train)):
        loss_train[i] = loss_train[i].item()
    return loss_train
