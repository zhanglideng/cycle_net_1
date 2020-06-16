import sys
import argparse
import time
import glob
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
from torchvision import transforms
from dataloader import Cycle_DataSet
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from cycle_model import *
import torch
from utils.loss import *
from utils.save_log_to_excel import *

"""
    测试的具体任务：
    1.测试nyu的测试集，给出可视化结果和指标结果。可以使用excel表格给出指标结果。
    2.测试真实世界的数据集，给出可视化结果。
    3.测试ntire2018数据集，给出可视化结果和指标结果。
"""

# test_visual_path = '/input/data/nyu/test_visual/'

data_path = '/input/data/'
test_hazy_path = data_path + 'nyu/test_hazy/'
test_gth_path = data_path + 'nyu/test_gth/'

BATCH_SIZE = 1
weight = [1, 1, 1, 1, 1, 1, 1, 1, 1]
excel_test_line = 1


def get_image_for_save(img):
    img = img.cpu()
    img = img.numpy()
    img = np.squeeze(img)
    img = img * 255
    img[img < 0] = 0
    img[img > 255] = 255
    img = np.rollaxis(img, 0, 3)
    img = img.astype('uint8')
    return img


save_path = 'test_result_{}'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
if not os.path.exists(save_path):
    os.makedirs(save_path)

excel_save = save_path + '/test_result.xls'

model_path = './mid_model/cycle_model.pt'
net = torch.load(model_path)
net = net.cuda()
transform = transforms.Compose([transforms.ToTensor()])

test_path_list = [test_hazy_path, test_gth_path]
test_data = Cycle_DataSet(transform, test_path_list, flag='test')
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

count = 0
print(">>Start testing...\n")
f, sheet_test = init_excel(kind='test')
for haze_name, haze_image, gt_image in test_data_loader:
    count += 1
    # print('Processing %d...' % count)
    print('name:%s' % haze_name)
    net.eval()
    with torch.no_grad():
        # J = net(haze_image)
        haze_image = haze_image.cuda()
        gt_image = gt_image.cuda()
        J1 = net(haze_image, haze_image)
        J2 = net(J1, haze_image)
        J3 = net(J2, haze_image)
        J4 = net(J3, haze_image)
        J5 = net(J4, haze_image)
        loss_image = [J1, J2, J3, J4, J5, gt_image]
        loss = loss_test(loss_image)

        excel_test_line = write_excel_test(sheet=sheet_test, line=excel_test_line, name=haze_name[0], loss=loss)
        f.save(excel_save)
        im_output_for_save = get_image_for_save(J1)
        filename = haze_name[0] + '_1.bmp'
        cv2.imwrite(os.path.join(save_path, filename), im_output_for_save)

        im_output_for_save = get_image_for_save(J2)
        filename = haze_name[0] + '_2.bmp'
        cv2.imwrite(os.path.join(save_path, filename), im_output_for_save)

        im_output_for_save = get_image_for_save(J3)
        filename = haze_name[0] + '_3.bmp'
        cv2.imwrite(os.path.join(save_path, filename), im_output_for_save)

        im_output_for_save = get_image_for_save(J4)
        filename = haze_name[0] + '_4.bmp'
        cv2.imwrite(os.path.join(save_path, filename), im_output_for_save)

        im_output_for_save = get_image_for_save(J5)
        filename = haze_name[0] + '_5.bmp'
        cv2.imwrite(os.path.join(save_path, filename), im_output_for_save)

print("Finished!")
