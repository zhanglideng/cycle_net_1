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
        self.haze_path, self.gt_path, self.t_path = path
        self.haze_data_list = os.listdir(self.haze_path)
        self.gt_data_list = os.listdir(self.gt_path)
        self.gt_data_list.sort(key=lambda x: int(x[:-4]))
        self.t_data_list = os.listdir(self.t_path)
        self.haze_data_list.sort(key=lambda x: int(x[:-18]))
        self.length = len(os.listdir(self.haze_path))
        self.haze_image_dict = {}
        self.gth_image_dict = {}
        self.t_dict = {}
        self.A_dict = {}
        # 读入数据
        A_gth = np.ones((608, 448, 3), dtype=np.float32)
        print('starting read image data...')
        for i in range(len(self.haze_data_list)):
            name = self.haze_data_list[i][:-4]
            # print(self.haze_path + name + '.png')
            A = float(name[-11:-7])
            self.haze_image_dict[name] = cv2.imread(self.haze_path + name + '.png')
            # print(self.haze_image_dict[name][0][0][0])
            t_gth = np.load(self.t_path + name + '.npy')
            t_gth = np.expand_dims(t_gth, axis=2)
            t_gth = t_gth.astype(np.float32)
            self.t_dict[name] = t_gth
            self.A_dict[name] = A_gth * A
        print('starting read GroundTruth data...')
        for i in range(len(self.gt_data_list)):
            name = self.gt_data_list[i][:-4]
            self.gth_image_dict[name] = cv2.imread(self.gt_path + name + '.PNG')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name = self.haze_data_list[idx][:-4]
        haze_image = self.haze_image_dict[name]
        gt_image = self.gth_image_dict[name[:-14]]
        t_gth = self.t_dict[name]
        A_gth = self.A_dict[name]
        # print(haze_image[0][0][0])
        # print(gt_image[0][0][0])
        # print(A_gth[0][0])
        # print(t_gth[0][0])

        if self.transform1:
            haze_image = self.transform1(haze_image)
            gt_image = self.transform1(gt_image)
            A_gth = self.transform1(A_gth)
            t_gth = self.transform1(t_gth)

        haze_image = haze_image.cuda()
        gt_image = gt_image.cuda()
        A_gth = A_gth.cuda()
        t_gth = t_gth.cuda()
        if self.flag == 'train':
            return haze_image, gt_image, A_gth, t_gth
        elif self.flag == 'test':
            return name, haze_image, gt_image, A_gth, t_gth

        # if __name__ == '__main__':
