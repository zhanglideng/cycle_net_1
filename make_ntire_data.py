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


data_path = '/input/data/ntire_2018'
ori_hazy_path = data_path + '/hazy/'
ori_gth_path = data_path + '/GT/'
cut_hazy_path = [data_path + '/train_hazy/', data_path + '/flat_train_hazy/',
                 data_path + '/val_hazy/', data_path + '/flat_val_hazy/',
                 data_path + '/test_hazy/', data_path + '/flat_test_hazy/']

cut_gth_path = [data_path + '/train_gth/', data_path + '/flat_train_gth/',
                data_path + '/val_gth/', data_path + '/flat_val_gth/',
                data_path + '/test_gth/', data_path + '/flat_test_gth/']
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

hazy_data_list = os.listdir(ori_hazy_path)
gth_data_list = os.listdir(ori_gth_path)
hazy_data_list.sort(key=lambda x: int(x[:2]))
hazy_data_list.sort(key=lambda x: (x[4]))
gth_data_list.sort(key=lambda x: int(x[:2]))
gth_data_list.sort(key=lambda x: (x[4]))

for i in range(len(hazy_data_list)):
    name = '0' * (2 - len(str(i))) + str(i) + '.jpg'
    shutil.move(ori_hazy_path + hazy_data_list[i], ori_hazy_path + name)

for i in range(len(gth_data_list)):
    name = '0' * (2 - len(str(i))) + str(i) + '.jpg'
    shutil.move(ori_gth_path + gth_data_list[i], ori_gth_path + name)

hazy_data_list = os.listdir(ori_hazy_path)
gth_data_list = os.listdir(ori_gth_path)
hazy_data_list.sort(key=lambda x: int(x[:2]))
gth_data_list.sort(key=lambda x: int(x[:2]))
length = len(hazy_data_list)
count = 0
for i in range(length):
    if i < length * 0.8 - 1:
        flag = 0
        hazy_path = cut_hazy_path[flag]
        gth_path = cut_gth_path[flag]
    elif i <= length * 0.9 - 1:
        flag = 2
        hazy_path = cut_hazy_path[flag]
        gth_path = cut_gth_path[flag]
    else:
        flag = 4
        hazy_path = cut_hazy_path[flag]
        gth_path = cut_gth_path[flag]
    haze_image = cv2.imread(ori_hazy_path + hazy_data_list[i])
    gth_image = cv2.imread(ori_gth_path + gth_data_list[i])
    height, width, channel = haze_image.shape
    for n in [1, 0.9]:
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
                if gth_mean >= 127 and gth_stddv <= 10:
                    hazy_path = cut_hazy_path[flag + 1]
                    gth_path = cut_gth_path[flag + 1]
                haze_name = hazy_path + str(count) + '_mean=' + str(haze_mean) + '_stddv=' + str(haze_stddv) + '.jpg'
                gth_name = gth_path + str(count) + '_mean=' + str(gth_mean) + '_stddv=' + str(gth_stddv) + '.jpg'
                # print(name)
                cv2.imwrite(haze_name, haze_patch)
                cv2.imwrite(gth_name, gth_patch)
                count += 1
                # image[:height//8*8, :width//8*8]
                print(count)
