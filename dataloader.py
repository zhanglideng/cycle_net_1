from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as f
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio
import torch
import random
import math
from PIL import Image


# 一次性读入所有数据
# nyu/test/1318_a=0.55_b=1.21.png

# 实现旋转，反转。一共8种形态。
def data_aug(img1, img2):
    a = random.random()
    b = math.floor(random.random() * 4)
    if a >= 0.5:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    if b == 1:
        img1 = img1.transpose(Image.ROTATE_90)
        img2 = img2.transpose(Image.ROTATE_90)
    elif b == 2:
        img1 = img1.transpose(Image.ROTATE_180)
        img2 = img2.transpose(Image.ROTATE_180)
    elif b == 3:
        img1 = img1.transpose(Image.ROTATE_270)
        img2 = img2.transpose(Image.ROTATE_270)
    return img1, img2


class Cycle_DataSet(Dataset):
    def __init__(self, transform1, is_gth_train, path=None, flag='train', ):
        self.flag = flag
        self.transform1 = transform1
        self.haze_path, self.gt_path = path
        self.haze_data_list = os.listdir(self.haze_path)
        self.haze_data_list.sort(key=lambda x: int(x[:5]))
        self.gt_data_list = os.listdir(self.gt_path)
        self.gt_data_list.sort(key=lambda x: int(x[:5]))
        if is_gth_train:
            self.haze_data_list = self.haze_data_list + self.gt_data_list
        self.length = len(self.haze_data_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        haze_name = self.haze_data_list[idx][:-4]
        gth_name = haze_name.split('_')[0]
        # gth_name = haze_name[:5]
        gt_image = Image.open(self.gt_path + gth_name + '.png')
        if len(haze_name) == 5:
            haze_image = gt_image
        else:
            haze_image = Image.open(self.haze_path + haze_name + '.png')
        # 数据增强
        if self.flag == 'train':
            haze_image, gt_image = data_aug(haze_image, gt_image)
            haze_image = np.asarray(haze_image)
            gt_image = np.asarray(gt_image)

        if self.transform1:
            haze_image = self.transform1(haze_image)
            gt_image = self.transform1(gt_image)

        return haze_name, haze_image, gt_image


class Ntire_DataSet(Dataset):
    def __init__(self, transform1, is_gth_train, path=None, flag='train'):
        self.flag = flag
        self.transform1 = transform1
        self.haze_path, self.gt_path = path
        self.haze_data_list = os.listdir(self.haze_path)
        self.haze_data_list.sort(key=lambda x: int(x[:-4]))
        self.gt_data_list = os.listdir(self.gt_path)
        self.gt_data_list.sort(key=lambda x: int(x[:-4]))
        if is_gth_train:
            self.haze_data_list = self.haze_data_list + self.gt_data_list
        self.length = len(self.haze_data_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        haze_name = self.haze_data_list[idx][:-4]
        gth_name = haze_name
        gt_image = Image.open(self.gt_path + gth_name + '.jpg')
        haze_image = Image.open(self.haze_path + haze_name + '.jpg')
        # 数据增强
        if self.flag == 'train':
            haze_image, gt_image = data_aug(haze_image, gt_image)
            haze_image = np.asarray(haze_image)
            gt_image = np.asarray(gt_image)

        if self.transform1:
            haze_image = self.transform1(haze_image)
            gt_image = self.transform1(gt_image)

        return haze_name, haze_image, gt_image
