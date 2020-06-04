#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
from PIL import Image
import sys
import math
import random
import cv2
import gc
import time

train_path = '/input/data/nyu/train/'
val_path = '/input/data/nyu/val/'
test_path = '/input/data/nyu/test/'
gth_path = '/input/data/nyu/gth/'

t_path = '/input/data/nyu/transmission/'
mat_path = '/input/data/nyu_depth_v2_labeled.mat'

haze_num = 1  # 无雾图生成几张有雾图
sigma = 1  # 高斯噪声的方差
trim_size = 16
'''
清晰图像
有雾图20张
深度图.npy
对以上所有图像切边
对深度图和有雾图引导滤波
使用png格式压缩
加速运算过程

高最小值为624 （640，8-632）
宽最小值为464 （480，8-472）
'''


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = mean_a * im + mean_b
    return q


if __name__ == '__main__':

    # color_shift = 0  # 合成无偏差的有雾图

    f = h5py.File(mat_path)
    if not os.path.exists(t_path):
        os.makedirs(t_path)
    if not os.path.exists(gth_path):
        os.makedirs(gth_path)
    depths = f['depths']
    images = f['images']
    print(depths.shape)
    print(images.shape)
    depths = np.array(depths)
    images = np.array(images)

    # 裁剪其使其能放入AtJ网络
    depths = depths[:, trim_size:640 - trim_size, trim_size:480 - trim_size]
    images = images[:, :, trim_size:640 - trim_size, trim_size:480 - trim_size]
    print(depths.shape)
    print(images.shape)
    length = depths.shape[0]
    start = time.time()
    for i in range(length):
        if i < length * 0.8 - 1:
            path = train_path
        elif i <= length * 0.9 - 1:
            path = val_path
        else:
            path = test_path
        if not os.path.exists(path):
            os.makedirs(path)
        depth = depths[i]
        m = depth.max()
        depth = depth / m
        image = images[i]
        print('dealing:' + str(i) + '.png')
        image_gray = image[0] * 0.299 + image[1] * 0.587 + image[2] * 0.114
        depth = Guidedfilter(image_gray, depth, 14, 0.0001)
        # depth_index_path = depth_path + str(i) + '.npy'
        # np.save(depth_index_path, depth)

        r = Image.fromarray(images[i][0]).convert('L')
        g = Image.fromarray(images[i][1]).convert('L')
        b = Image.fromarray(images[i][2]).convert('L')
        img = Image.merge("RGB", (r, g, b))
        save_path = gth_path + '0' * (4 - len(str(i))) + str(i) + '.png'
        img.save(save_path, 'PNG', optimize=True)

        for rand in range(haze_num):
            # image_out = np.zeros((3, depth.shape[0], depth.shape[1]))
            noise = np.random.randn(1, depth.shape[0], depth.shape[1]) * sigma
            noise = np.concatenate((noise, noise, noise))

            fog_A = round(random.uniform(0.7, 1), 2)
            map_A = np.ones((3, depth.shape[0], depth.shape[1])) * fog_A

            fog_density = round(random.uniform(0.8, 1.0), 2)

            t = np.exp(-1 * fog_density * depth)
            t_index_path = t_path + '0' * (4 - len(str(i))) + str(i) + '_a=' + '%.02f' % fog_A + '_b=' + '%.02f' % fog_density + '.npy'
            np.save(t_index_path, t)

            t = np.expand_dims(t, axis=0)
            t = np.concatenate((t, t, t))
            # print(image.shape)
            # print(t.shape)
            # print(map_A.shape)
            # print(noise.shape)
            image_out = np.add(np.multiply(image, t), np.add(255 * np.multiply(map_A, (1 - t)), noise))
            image_out[image_out < 0] = 0
            image_out[image_out > 255] = 255
            image_path = path + '0' * (4 - len(str(i))) + str(i) + '_a=' + '%.02f' % fog_A + '_b=' + '%.02f' % fog_density + '.png'
            image_out = np.swapaxes(image_out, 0, 2)
            image_out = np.swapaxes(image_out, 0, 1)
            image_out = Image.fromarray(image_out.astype('uint8')).convert('RGB')
            image_out.save(image_path, 'png', optimize=True)
        end = time.time()
        s = (end - start) / (i + 1) * (length - i - 1)
        print('%d:%02d:%02d' % (s // 3600, s // 60 - s // 3600 * 60, s % 60))
