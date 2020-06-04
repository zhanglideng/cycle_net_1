from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio
import torch


# 一次性读入所有数据
# nyu/test/1318_a=0.55_b=1.21.png
class AtJDataSet(Dataset):
    def __init__(self, transform1, path=None, flag='train'):
        # print(path)
        self.flag = flag
        self.transform1 = transform1
        self.haze_path, self.gt_path = path

        self.haze_data_list = os.listdir(self.haze_path)
        # print(self.haze_data_list)
        self.haze_data_list.sort(key=lambda x: int(x[:4]))
        self.gt_data_list = os.listdir(self.gt_path)
        self.gt_data_list.sort(key=lambda x: int(x[:4]))

        self.length = len(os.listdir(self.haze_path))
        self.haze_image_dict = {}
        self.gth_image_dict = {}
        # 读入数据
        print('starting read image data...')
        for i in range(len(self.haze_data_list)):
            name = self.haze_data_list[i][:-4]
            # print(self.haze_path + name + '.png')
            self.haze_image_dict[name] = cv2.imread(self.haze_path + name + '.png')
        print('starting read GroundTruth data...')
        for i in range(len(self.gt_data_list)):
            name = self.gt_data_list[i][:4]
            self.haze_image_dict[name] = cv2.imread(self.gt_path + name + '.png')
            self.gth_image_dict[name] = cv2.imread(self.gt_path + name + '.png')
        self.haze_data_list = self.haze_data_list + self.gt_data_list

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name = self.haze_data_list[idx][:-4]
        haze_image = self.haze_image_dict[name]
        gt_image = self.gth_image_dict[name[:4]]

        if self.transform1:
            haze_image = self.transform1(haze_image)
            gt_image = self.transform1(gt_image)

        if self.flag == 'train':
            return haze_image, gt_image
        elif self.flag == 'test':
            return name, haze_image, gt_image

        # if __name__ == '__main__':
