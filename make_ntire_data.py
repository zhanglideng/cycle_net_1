import os
import cv2

path = ['/input/data/ntire_2018/hazy/', '/input/data/ntire_2018/GT/']
cut_path = ['/input/data/cut_ntire_2018/hazy/', '/input/data/cut_ntire_2018/GT/']
size = 256
'''
01_indoor_GT.jpg
01_indoor_hazy.jpg

01_outdoor_GT.jpg
01_outdoor_hazy.jpg
624 464 
400*400
'''
for i in range(len(path)):
    count = 0
    if not os.path.exists(cut_path[i]):
        os.makedirs(cut_path[i])
    data_list = os.listdir(path[i])
    data_list.sort(key=lambda x: int(x[:2]))
    data_list.sort(key=lambda x: (x[4]))
    # print(data_list)
    for j in data_list:
        image = cv2.imread(path[i] + j)
        for n in [1, 0.9, 0.8, 0.7, 0.6, 0.5]:
            height, width, = image.shape
            pic = cv2.resize(image, (int(width * n), int(height * n)), interpolation=cv2.INTER_CUBIC)
            height, width, = image.shape
            height_num = height // size
            width_num = width // size
            height_border = (size * (height_num + 1) - height) // height_num
            width_border = (size * (width_num + 1) - width) // width_num
            for k in range(height_num + 1):
                for m in range(width_num + 1):
                    # print(count)
                    patch = image[k * (size - height_border - 1):k * (size - height_border - 1) + 512,
                            m * (size - width_border - 1):m * (size - width_border - 1) + 512]
                    # print(patch.shape)
                    name = cut_path[i] + str(count) + '.jpg'
                    # print(name)
                    cv2.imwrite(name, patch)
                    count += 1
                    # image[:height//8*8, :width//8*8]
            print(count)
