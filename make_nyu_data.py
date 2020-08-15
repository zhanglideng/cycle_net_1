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

if os.path.exists('/input'):
    data_path = '/input/data/'
else:
    data_path = '/home/liu/zhanglideng/data/'

train_hazy_path = data_path + 'nyu_cycle/train_hazy/'
val_hazy_path = data_path + 'nyu_cycle/val_hazy/'
test_hazy_path = data_path + 'nyu_cycle/test_hazy/'

train_gth_path = data_path + 'nyu_cycle/train_gth/'
val_gth_path = data_path + 'nyu_cycle/val_gth/'
test_gth_path = data_path + 'nyu_cycle/test_gth/'

mat_path = data_path + 'nyu_depth_v2_labeled.mat'

haze_num = 2  # 无雾图生成几张有雾图
sigma = 1  # 高斯噪声的方差
trim_size = 16
size = 256
air_light_range = [0.7, 1.0]
fog_range = [0.8, 1.2]
flag = 'train'
'''
清晰图像
有雾图20张
深度图.npy
对以上所有图像切边
对深度图和有雾图引导滤波
使用png格式压缩
加速运算过程

高最小值为608
宽最小值为448
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
    path = [train_hazy_path, val_hazy_path, test_hazy_path,
            train_gth_path, val_gth_path, test_gth_path]
    for i in path:
        if not os.path.exists(i):
            os.makedirs(i)
    f = h5py.File(mat_path)
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
    count = 0
    for i in range(length):
        print('dealing:' + str(i) + '.bmp')
        if i < length * 0.8 - 1:
            flag = 'train'
            hazy_path = train_hazy_path
            gth_path = train_gth_path
        elif i <= length * 0.9 - 1:
            flag = 'val'
            hazy_path = val_hazy_path
            gth_path = val_gth_path
        else:
            flag = 'test'
            hazy_path = test_hazy_path
            gth_path = test_gth_path
        depth = depths[i]
        m = depth.max()
        depth = depth / m
        image = images[i]
        image_gray = image[0] * 0.299 + image[1] * 0.587 + image[2] * 0.114
        depth = Guidedfilter(image_gray, depth, 14, 0.0001)
        if flag == 'test':
            r = Image.fromarray(image[0]).convert('L')
            g = Image.fromarray(image[1]).convert('L')
            b = Image.fromarray(image[2]).convert('L')
            img = Image.merge("RGB", (r, g, b))
            save_path = gth_path + '0' * (5 - len(str(count))) + str(count) + '.bmp'
            img.save(save_path, 'bmp')
            for rand in range(haze_num):
                # image_out = np.zeros((3, depth.shape[0], depth.shape[1]))
                noise = np.random.randn(1, depth.shape[0], depth.shape[1]) * sigma
                noise = np.concatenate((noise, noise, noise))

                fog_A = round(random.uniform(air_light_range[0], air_light_range[1]), 2)
                map_A = np.ones((3, depth.shape[0], depth.shape[1])) * fog_A
                fog_density = round(random.uniform(fog_range[0], fog_range[1]), 2)

                t = np.exp(-1 * fog_density * depth)
                t = np.expand_dims(t, axis=0)
                t = np.concatenate((t, t, t))
                image_out = np.add(np.multiply(image, t), np.add(255 * np.multiply(map_A, (1 - t)), noise))
                image_out[image_out < 0] = 0
                image_out[image_out > 255] = 255

                image_path = hazy_path + '0' * (5 - len(str(count))) + str(
                    count) + '_a=' + '%.02f' % fog_A + '_b=' + '%.02f' % fog_density + '.bmp'
                image_out = np.swapaxes(image_out, 0, 2)
                image_out = np.swapaxes(image_out, 0, 1)
                image_out = Image.fromarray(image_out.astype('uint8')).convert('RGB')
                image_out.save(image_path, 'bmp')
            print(count)
            count += 1
        else:
            height, width = depth.shape
            height_num = height // size
            width_num = width // size
            height_border = (size * (height_num + 1) - height) // height_num
            width_border = (size * (width_num + 1) - width) // width_num
            for k in range(height_num + 1):
                for m in range(width_num + 1):
                    image_patch = image[:, k * (size - height_border - 1):k * (size - height_border - 1) + size,
                                  m * (size - width_border - 1):m * (size - width_border - 1) + size]
                    depth_patch = depth[k * (size - height_border - 1):k * (size - height_border - 1) + size,
                                  m * (size - width_border - 1):m * (size - width_border - 1) + size]
                    r = Image.fromarray(image_patch[0]).convert('L')
                    g = Image.fromarray(image_patch[1]).convert('L')
                    b = Image.fromarray(image_patch[2]).convert('L')
                    img = Image.merge("RGB", (r, g, b))
                    save_path = gth_path + '0' * (5 - len(str(count))) + str(count) + '.bmp'
                    img.save(save_path, 'bmp')

                    for rand in range(haze_num):
                        # image_out = np.zeros((3, depth.shape[0], depth.shape[1]))
                        noise = np.random.randn(1, depth_patch.shape[0], depth_patch.shape[1]) * sigma
                        noise = np.concatenate((noise, noise, noise))

                        fog_A = round(random.uniform(air_light_range[0], air_light_range[1]), 2)
                        map_A = np.ones((3, depth_patch.shape[0], depth_patch.shape[1])) * fog_A
                        fog_density = round(random.uniform(fog_range[0], fog_range[1]), 2)

                        t = np.exp(-1 * fog_density * depth_patch)
                        t = np.expand_dims(t, axis=0)
                        t = np.concatenate((t, t, t))
                        image_out = np.add(np.multiply(image_patch, t),
                                           np.add(255 * np.multiply(map_A, (1 - t)), noise))
                        image_out[image_out < 0] = 0
                        image_out[image_out > 255] = 255

                        image_path = hazy_path + '0' * (5 - len(str(count))) + str(
                            count) + '_a=' + '%.02f' % fog_A + '_b=' + '%.02f' % fog_density + '.bmp'
                        image_out = np.swapaxes(image_out, 0, 2)
                        image_out = np.swapaxes(image_out, 0, 1)
                        image_out = Image.fromarray(image_out.astype('uint8')).convert('RGB')
                        image_out.save(image_path, 'bmp')
                    print(count)
                    count += 1
        end = time.time()
        s = (end - start) / (i + 1) * (length - i - 1)
        print('%d:%02d:%02d' % (s // 3600, s // 60 - s // 3600 * 60, s % 60))
