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

data_path = '/home/liu/zhanglideng/data/'
get_patch = True

if get_patch:
    name = 'patch'
else:
    name = 'whole'
train_hazy_path = data_path + 'nyu_cycle/train_hazy_' + name + '/'
val_hazy_path = data_path + 'nyu_cycle/val_hazy_' + name + '/'
test_hazy_path = data_path + 'nyu_cycle/test_hazy_' + name + '/'

train_gth_path = data_path + 'nyu_cycle/train_gth_' + name + '/'
val_gth_path = data_path + 'nyu_cycle/val_gth_' + name + '/'
test_gth_path = data_path + 'nyu_cycle/test_gth_' + name + '/'

mat_path = data_path + 'nyu_depth_v2_labeled.mat'

haze_num = 2  # 无雾图生成几张有雾图
sigma = 1  # 高斯噪声的方差
trim_size = 16
size = 256
air_light_range = [0.7, 1.0]
fog_range = [0.8, 1.2]
get_patch = True


def Guidedfilter(im, p, r, eps):
    im = im[0] * 0.299 + im[1] * 0.587 + im[2] * 0.114
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


def read_mat(mat_path, trim_size):
    f = h5py.File(mat_path)
    mat_depths = f['depths']
    mat_images = f['images']
    print(mat_depths.shape)
    print(mat_images.shape)
    mat_depths = np.array(mat_depths)
    mat_images = np.array(mat_images)
    height = mat_depths.shape[1]
    weight = mat_depths.shape[2]
    mat_depths = mat_depths[:, trim_size:height - trim_size, trim_size:weight - trim_size]
    mat_images = mat_images[:, :, trim_size:height - trim_size, trim_size:weight - trim_size]
    print(mat_depths.shape)
    print(mat_images.shape)
    return mat_images, mat_depths


def path_mod(counter, length):
    """
    根据计数器处于数据集中的位置进行训练集、验证集、测试集的划分
    :param counter: 计数器
    :param length: 图像总数
    :return: 标签、有雾图像保存路径、无雾图像保存路径
    """
    if counter < length * 0.8 - 1:
        hazy_path = train_hazy_path
        gth_path = train_gth_path
    elif counter <= length * 0.9 - 1:
        hazy_path = val_hazy_path
        gth_path = val_gth_path
    else:
        hazy_path = test_hazy_path
        gth_path = test_gth_path
    return hazy_path, gth_path


def image_save(im, count, gth_path):
    """
    保存RGB图像，作为Gt
    :param im: RGB图像
    :param count: RGB图像计数
    :param gth_path: RGB图像保存路径
    :return:
    """
    r = Image.fromarray(im[0]).convert('L')
    g = Image.fromarray(im[1]).convert('L')
    b = Image.fromarray(im[2]).convert('L')
    im = Image.merge("RGB", (r, g, b))
    save_path = gth_path + '0' * (5 - len(str(count))) + str(count) + '.bmp'
    im.save(save_path, 'bmp')


def make_hazy_image(depth, image, air_light_range, fog_range, hazy_path, count):
    """
    根据给出的深度图像、RGB图像、大气光范围、散射系数范围合成对应的有雾图像
    :param depth: 深度图像
    :param image: RGB图像
    :param air_light_range: 大气光范围
    :param fog_range: 散射系数范围
    :param hazy_path: 有雾图像保存路径
    :param count: 有雾图像计数
    :return:
    """
    height, width = depth.shape
    noise = np.random.randn(1, height, width) * sigma
    noise = np.concatenate((noise, noise, noise))

    fog_A = round(random.uniform(air_light_range[0], air_light_range[1]), 2)
    map_A = np.ones((3, height, depth.shape[1])) * fog_A
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


def count_border(height, width, size):
    """
    根据给出的图像高和宽计算重叠的边缘的宽
    :param height: 图像的高
    :param width: 图像的宽
    :param size: 设置好的图像块大小
    :return: 垂直方向的重叠边缘的宽度、水平方向的重叠边缘的宽度、垂直方向的图像块数量、水平方向的图像块数量
    """
    height_num = height // size
    width_num = width // size
    height_border = (size * (height_num + 1) - height) // height_num
    width_border = (size * (width_num + 1) - width) // width_num
    return height_border, width_border, height_num, width_num


def make_hazy_patch(depth, image, size):
    """
    :param depth: 深度图
    :param image: RGB图
    :param size: 输出图像块的大小
    :return: 切好的深度图像块和RGB图像块
    """
    height, width = depth.shape
    height_border, width_border, height_num, width_num = count_border(height, width, size)
    patch_num = (height_num + 1) * (width_num + 1)
    image_patch = [0] * patch_num
    depth_patch = [0] * patch_num
    count = 0
    for k in range(height_num + 1):
        for m in range(width_num + 1):
            image_patch[count] = image[:, k * (size - height_border - 1):k * (size - height_border - 1) + size,
                                 m * (size - width_border - 1):m * (size - width_border - 1) + size]
            depth_patch[count] = depth[k * (size - height_border - 1):k * (size - height_border - 1) + size,
                                 m * (size - width_border - 1):m * (size - width_border - 1) + size]
            count += 1
    return image_patch, depth_patch


def image_reshape(image, depth, n):
    image = np.array(image)
    image = image.swapaxes(0, 2).swapaxes(0, 1)
    image = cv2.resize(image, (int(width * n), int(height * n)), interpolation=cv2.INTER_CUBIC)
    depth = cv2.resize(depth, (int(width * n), int(height * n)), interpolation=cv2.INTER_CUBIC)
    image = image.swapaxes(0, 2).swapaxes(1, 2)
    image.tolist()
    return image, depth


if __name__ == '__main__':
    # color_shift = 0  # 合成无偏差的有雾图
    path = [train_hazy_path, val_hazy_path, test_hazy_path,
            train_gth_path, val_gth_path, test_gth_path]
    for i in path:
        if not os.path.exists(i):
            os.makedirs(i)

    images, depths = read_mat(mat_path, trim_size)

    length = depths.shape[0]
    start = time.time()
    count = 0
    for i in range(length):
        print('dealing:' + str(i) + '.bmp')
        hazy_path, gth_path = path_mod(i, length)

        depth = depths[i]
        image = images[i]
        depth = Guidedfilter(image, depth, 14, 0.0001)

        height, width = depth.shape
        for n in [1.0, 0.9, 0.8, 0.7, 0.6]:
            if int(width * n) < size or int(height * n) < size:
                break
            re_image, re_depth = image_reshape(image, depth, n)
            if get_patch:
                re_image, re_depth = make_hazy_patch(re_depth, re_image, size)
            else:
                re_image = [re_image]
                re_depth = [re_depth]
            for j in range(len(re_image)):
                image_save(re_image[j], count, gth_path)
                for rand in range(haze_num):
                    re_depth[j] = re_depth[j] / re_depth[j].max()
                    make_hazy_image(re_depth[j], re_image[j], air_light_range, fog_range, hazy_path, count)
                # print(count)
                count += 1
        print(count)
        end = time.time()
        s = (end - start) / (i + 1) * (length - i - 1)
        print('%d:%02d:%02d' % (s // 3600, s // 60 - s // 3600 * 60, s % 60))
