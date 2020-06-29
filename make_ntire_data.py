import os
import cv2
import numpy as np
import shutil


def Scharr_demo(image):
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    print(np.mean(gradxy))


data_path = '/input/data/ntire_2018'
ori_hazy_path = data_path + '/hazy/'
ori_gth_path = data_path + '/GT/'
cut_hazy_path = [data_path + '/train_hazy/', data_path + 'val_hazy/', data_path + '/test_hazy/']
cut_gth_path = [data_path + '/train_gth/', data_path + '/val_gth/', data_path + '/test_gth/']
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
for i in range(length):
    count = 0
    if i < length * 0.8 - 1:
        hazy_path = cut_hazy_path[0]
        gth_path = cut_gth_path[0]
    elif i <= length * 0.9 - 1:
        hazy_path = cut_hazy_path[1]
        gth_path = cut_gth_path[1]
    else:
        hazy_path = cut_hazy_path[2]
        gth_path = cut_gth_path[2]
    haze_image = cv2.imread(ori_hazy_path + hazy_data_list[i])
    gth_image = cv2.imread(ori_gth_path + gth_data_list[i])
    height, width, = haze_image.shape
    for n in [1, 0.9, 0.8, 0.7, 0.6, 0.5]:
        re_haze_image = cv2.resize(haze_image, (int(width * n), int(height * n)), interpolation=cv2.INTER_CUBIC)
        re_gth_image = cv2.resize(gth_image, (int(width * n), int(height * n)), interpolation=cv2.INTER_CUBIC)
        re_height, re_width, = re_haze_image.shape
        height_num = re_height // size
        width_num = re_width // size
        height_border = (size * (height_num + 1) - re_height) // height_num
        width_border = (size * (width_num + 1) - re_width) // width_num
        for k in range(height_num + 1):
            for m in range(width_num + 1):
                # print(count)
                patch = image[k * (size - height_border - 1):k * (size - height_border - 1) + size,
                        m * (size - width_border - 1):m * (size - width_border - 1) + size]
                # print(patch.shape)
                Scharr_demo(patch)
                name = cut_path[i] + str(count) + '.jpg'
                # print(name)
                cv2.imwrite(name, patch)
                count += 1
                # image[:height//8*8, :width//8*8]
        print(count)
