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
from PIL import Image


# 转换需要保存的图像
def get_image_for_save(img):
    img = img.cpu()
    img = img.numpy()
    img = np.squeeze(img)
    img = img * 255
    img[img < 0] = 0
    img[img > 255] = 255
    img = np.rollaxis(img, 0, 3)
    img = img.astype('uint8')
    img = Image.fromarray(img).convert('RGB')
    return img


# 查找训练好的模型文件
def find_pretrain(path_name):
    file_list = os.listdir('./')
    length = len(path_name)
    for i in range(len(file_list)):
        if file_list[i][:length] == path_name:
            return file_list[i]
    return 0


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for CycleDehazeNet')
parser.add_argument('-Is_save_image', help='Whether to save the image', default=False, type=bool)
parser.add_argument('-test_round', help='How many rounds of testing?', default=5, type=int)
parser.add_argument('-data_path', help='The data path', default='/home/liu/zhanglideng', type=str)
parser.add_argument('-gth_test', help='Whether to add Gth testing', default=False, type=bool)
parser.add_argument('-batch_size', help='The batch size', default=1, type=int)
parser.add_argument('-excel_row', help='The excel row',
                    default=["num", "A", "beta", "J1_l2", "J2_l2", "J3_l2", "J4_l2", "J5_l2",
                             "J1_ssim", "J2_ssim", "J3_ssim", "J4_ssim", "J5_ssim",
                             "J1_vgg", "J2_vgg", "J3_vgg", "J4_vgg", "J5_vgg"], type=list)

args = parser.parse_args()

Is_save_image = args.Is_save_image  # 是否保存图像测试结果
test_round = args.test_round  # 测试循环次数
data_path = args.data_path  # 数据路径
batch_size = args.batch_size  # 测试批大小
gth_test = args.gth_test  # 是否测试无雾图像
excel_row = args.excel_row  # excel的列属性名

test_hazy_path = data_path + '/data/nyu_cycle/test_hazy/'
test_gth_path = data_path + '/data/nyu_cycle/test_gth/'

# 加载训练好的模型
file_path = find_pretrain('cycle_result')
model_path = file_path + '/cycle_model.pt'
net = torch.load(model_path)
net = net.cuda()
loss_net = test_loss_net().cuda()

# 创建用于保存测试结果的文件夹
local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
save_path = './{}/test_result_{}'.format(file_path, local_time)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 创建用于保存测试指标的表格文件
excel_test_line = 1
excel_save = './{}/test_result_{}.xls'.format(file_path, local_time)
f, sheet_test = init_test_excel(row=excel_row)

# 创建图像数据加载器
transform = transforms.Compose([transforms.ToTensor()])
test_path_list = [test_hazy_path, test_gth_path]
test_data = Cycle_DataSet(transform, is_gth_train=gth_test, path=test_path_list, flag='test')
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

# 开始测试
print("Start testing\n")
J = [0] * len(test_round)
for haze_name, haze_image, gt_image in test_data_loader:
    print('name:%s' % haze_name)
    net.eval()
    with torch.no_grad():
        haze_image = haze_image.cuda()
        gt_image = gt_image.cuda()
        J[0] = net(haze_image, haze_image)
        for i in range(len(test_round) - 1):
            J[i + 1] = net(J[i], haze_image)
        loss = loss_net(J, gt_image)

        # 保存测试指标
        excel_test_line = write_excel_test(sheet=sheet_test, line=excel_test_line, name=haze_name[0], loss=loss)
        f.save(excel_save)

        # 保存图像测试结果
        if Is_save_image:
            for i in range(len(J)):
                im_output_for_save = get_image_for_save(J[i])
                filename = '{}_{}.bmp'.format(haze_name[0], i)
                im_output_for_save.save(os.path.join(save_path, filename))
print("Finished!")
