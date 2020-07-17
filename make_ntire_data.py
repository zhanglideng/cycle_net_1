import os
import cv2
import numpy as np
import shutil


def Scharr_demo(image):
    grad_x = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    # print(np.mean(gradxy))
    return int(np.mean(gradxy))


if os.path.exists('/input'):
    ori_data_path = '/input/data/ntire_2018'
    cut_data_path = '/input/data/cut_ntire_cycle'
else:
    ori_data_path = '/home/ljh/zhanglideng/data/ntire_2018'
    cut_data_path = '/home/ljh/zhanglideng/data/cut_ntire_cycle'
ori_hazy_path = [ori_data_path + '/train_hazy/', ori_data_path + '/val_hazy/', ori_data_path + '/test_hazy/']
ori_gth_path = [ori_data_path + '/train_gth/', ori_data_path + '/val_gth/', ori_data_path + '/test_gth/']
cut_hazy_path = [cut_data_path + '/train_hazy/', cut_data_path + '/flat_train_hazy/',
                 cut_data_path + '/val_hazy/', cut_data_path + '/flat_val_hazy/',
                 cut_data_path + '/test_hazy/', cut_data_path + '/flat_test_hazy/']

cut_gth_path = [cut_data_path + '/train_gth/', cut_data_path + '/flat_train_gth/',
                cut_data_path + '/val_gth/', cut_data_path + '/flat_val_gth/',
                cut_data_path + '/test_gth/', cut_data_path + '/flat_test_gth/']
size = 256
'''
01_indoor_GT.jpg
01_indoor_hazy.jpg

01_outdoor_GT.jpg
01_outdoor_hazy.jpg
624 464 
400*400
'''
for i in range(len(cut_hazy_path)):
    if not os.path.exists(cut_hazy_path[i]):
        os.makedirs(cut_hazy_path[i])

for i in range(len(cut_gth_path)):
    if not os.path.exists(cut_gth_path[i]):
        os.makedirs(cut_gth_path[i])

# hazy_data_list = os.listdir(ori_hazy_path)
# gth_data_list = os.listdir(ori_gth_path)
# length = len(hazy_data_list)
count = 0
for i in range(len(ori_hazy_path)):
    hazy_data_list = os.listdir(ori_hazy_path[i])
    gth_data_list = os.listdir(ori_gth_path[i])
    hazy_path = cut_hazy_path[i * 2]
    gth_path = cut_gth_path[i * 2]
    for j in range(len(hazy_data_list)):
        haze_image = cv2.imread(ori_hazy_path[i] + hazy_data_list[j])
        gth_image = cv2.imread(ori_gth_path[i] + gth_data_list[j])
        height, width, channel = haze_image.shape
        for n in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            if int(width * n) < size or int(height * n) < size:
                continue
            re_haze_image = cv2.resize(haze_image, (int(width * n), int(height * n)), interpolation=cv2.INTER_CUBIC)
            re_gth_image = cv2.resize(gth_image, (int(width * n), int(height * n)), interpolation=cv2.INTER_CUBIC)
            re_height, re_width, channel = re_haze_image.shape
            height_num = re_height // size
            width_num = re_width // size
            height_border = (size * (height_num + 1) - re_height) // height_num
            width_border = (size * (width_num + 1) - re_width) // width_num
            for k in range(height_num + 1):
                for m in range(width_num + 1):
                    # print(count)
                    haze_patch = re_haze_image[k * (size - height_border - 1):k * (size - height_border - 1) + size,
                                 m * (size - width_border - 1):m * (size - width_border - 1) + size]
                    gth_patch = re_gth_image[k * (size - height_border - 1):k * (size - height_border - 1) + size,
                                m * (size - width_border - 1):m * (size - width_border - 1) + size]
                    # print(patch.shape)
                    (haze_mean, haze_stddv) = cv2.meanStdDev(haze_patch)
                    (gth_mean, gth_stddv) = cv2.meanStdDev(gth_patch)
                    haze_mean = int(np.mean(haze_mean))
                    haze_stddv = int(np.mean(haze_stddv))
                    gth_mean = int(np.mean(gth_mean))
                    gth_stddv = int(np.mean(gth_stddv))
                    # haze_grad = Scharr_demo(haze_patch)
                    # gth_grad = Scharr_demo(gth_patch)
                    if gth_stddv >= 10:
                        hazy_path = cut_hazy_path[i * 2]
                        gth_path = cut_gth_path[i * 2]
                        haze_name = hazy_path + str(count) + '.jpg'
                        gth_name = gth_path + str(count) + '.jpg'
                        # print(name)
                        cv2.imwrite(haze_name, haze_patch)
                        cv2.imwrite(gth_name, gth_patch)
                        count += 1
                    # image[:height//8*8, :width//8*8]
                    print(count)
