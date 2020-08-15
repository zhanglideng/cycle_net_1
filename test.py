import argparse
import os
from torchvision import transforms
from dataloader import Cycle_DataSet
from torch.utils.data import DataLoader
import torch
from loss import *
from utils import *
from PIL import Image

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for CycleDehazeNet')
parser.add_argument('-Is_save_image', help='Whether to save the image', default=True, type=bool)
parser.add_argument('-test_round', help='How many rounds of testing?', default=10, type=int)
parser.add_argument('-data_path', help='The data path', default='/home/liu/zhanglideng', type=str)
parser.add_argument('-gth_test', help='Whether to add Gth testing', default=False, type=bool)
parser.add_argument('-batch_size', help='The batch size', default=32, type=int)
parser.add_argument('-weight', help='The loss weight', default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], type=list)
'''
parser.add_argument('-excel_row', help='The excel row',
                    default=["num", "A", "beta",
                             "J1_l2", "J2_l2", "J3_l2", "J4_l2", "J5_l2", "J6_l2", "J7_l2", "J8_l2", "J9_l2", "J10_l2",
                             "J1_ssim", "J2_ssim", "J3_ssim", "J4_ssim", "J5_ssim", "J6_ssim", "J7_ssim", "J8_ssim",
                             "J9_ssim", "J10_ssim",
                             "J1_vgg", "J2_vgg", "J3_vgg", "J4_vgg", "J5_vgg", "J6_vgg", "J7_vgg", "J8_vgg", "J9_vgg",
                             "J10_vgg",
                             "l2(J1-J2)", "l2(J2-J3)", "l2(J3-J4)", "l2(J4-J5)", "l2(J5-J6)", "l2(J6-J7)", "l2(J7-J8)",
                             "l2(J8-J9)", "l2(J9-J10)",
                             "ssim(J1-J2)", "ssim(J2-J3)", "ssim(J3-J4)", "ssim(J4-J5)", "ssim(J5-J6)", "ssim(J6-J7)",
                             "ssim(J7-J8)", "ssim(J8-J9)", "ssim(J9-J10)",
                             "vgg(J1-J2)", "vgg(J2-J3)", "vgg(J3-J4)", "vgg(J4-J5)", "vgg(J5-J6)", "vgg(J6-J7)",
                             "vgg(J7-J8)", "vgg(J8-J9)", "vgg(J9-J10)"], type=list)
'''
parser.add_argument('-excel_row', help='The excel row',
                    default=["num", "A", "beta", "轮次", "l2", "ssim", "vgg",
                             "J1_l2", "J2_l2", "J3_l2", "J4_l2", "J5_l2", "J6_l2", "J7_l2", "J8_l2", "J9_l2", "J10_l2",
                             "J1_ssim", "J2_ssim", "J3_ssim", "J4_ssim", "J5_ssim", "J6_ssim", "J7_ssim", "J8_ssim",
                             "J9_ssim", "J10_ssim",
                             "J1_vgg", "J2_vgg", "J3_vgg", "J4_vgg", "J5_vgg", "J6_vgg", "J7_vgg", "J8_vgg", "J9_vgg",
                             "J10_vgg",
                             "l2(J1-J2)", "l2(J2-J3)", "l2(J3-J4)", "l2(J4-J5)", "l2(J5-J6)", "l2(J6-J7)", "l2(J7-J8)",
                             "l2(J8-J9)", "l2(J9-J10)",
                             "ssim(J1-J2)", "ssim(J2-J3)", "ssim(J3-J4)", "ssim(J4-J5)", "ssim(J5-J6)", "ssim(J6-J7)",
                             "ssim(J7-J8)", "ssim(J8-J9)", "ssim(J9-J10)",
                             "vgg(J1-J2)", "vgg(J2-J3)", "vgg(J3-J4)", "vgg(J4-J5)", "vgg(J5-J6)", "vgg(J6-J7)",
                             "vgg(J7-J8)", "vgg(J8-J9)", "vgg(J9-J10)"], type=list)
args = parser.parse_args()
Is_save_image = args.Is_save_image  # 是否保存图像测试结果
test_round = args.test_round  # 测试循环次数
data_path = args.data_path  # 数据路径
batch_size = args.batch_size  # 测试批大小
gth_test = args.gth_test  # 是否测试无雾图像
excel_row = args.excel_row  # excel的列属性名
weight = args.weight  # 损失函数的权重

test_hazy_path = data_path + '/data/nyu_cycle/test_hazy/'
test_gth_path = data_path + '/data/nyu_cycle/test_gth/'
val_hazy_path = data_path + '/data/nyu_cycle/val_hazy/'
val_gth_path = data_path + '/data/nyu_cycle/val_gth/'

# 加载训练好的模型
file_path = find_pretrain('cycle_result')
model_path = file_path + '/cycle_model.pt'
net = torch.load(model_path)
net = net.cuda()
loss_net_1 = test_loss_net_1(weight).cuda()
loss_net_2 = test_loss_net_2(weight).cuda()
gap_net = gap_compute_net(weight).cuda()

# 创建用于保存测试结果的文件夹
local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
save_path = './{}/test_result_{}'.format(file_path, local_time)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 创建用于保存测试指标的表格文件
excel_test_line = 1
excel_save = './{}/test_result_{}.xls'.format(file_path, local_time)
f, sheet_test = init_test_excel(row=excel_row)

# 创建图像数据加载器
transform = transforms.Compose([transforms.ToTensor()])
test_path_list = [test_hazy_path, test_gth_path]
val_path_list = [val_hazy_path, val_gth_path]
test_data = Cycle_DataSet(transform, is_gth_train=gth_test, path=test_path_list, flag='test')
val_data = Cycle_DataSet(transform, is_gth_train=gth_test, path=test_path_list, flag='val')
test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=128)
val_data_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=128)
image_gap = MAE(size_average=True)
# 计算最佳的迭代差距常量

temp_J = [0] * test_round
loss = [0] * (test_round-1)
# b_gap = [0] * len(val_data_loader)
# a_gap = [0] * len(val_data_loader)
# sum_gap = 0
start_time = time.time()
# image_gap = MAE(size_average=True)
count = 1
gap = 0.0001
for haze_name, haze_image, gt_image in test_data_loader:
    loss = [0] * 999
    net.eval()
    with torch.no_grad():
        haze_image = haze_image.cuda()
        gt_image = gt_image.cuda()
        temp_J[0] = net(haze_image, haze_image)
        for i in range(test_round - 1):
            temp_J[i + 1] = net(temp_J[i], haze_image)
        if Is_save_image:
            for k in range(test_round):
                im_output_for_save = get_image_for_save(temp_J[k])
                filename = '{}_{}.bmp'.format(haze_name[0], k)
                im_output_for_save.save(os.path.join(save_path, filename))
        loss_for_save_1 = loss_net_1(temp_J, gt_image)

        loss[0] = image_gap(temp_J[0], temp_J[1]).item()
        for i in range(test_round-2):
            loss[i + 1] = image_gap(temp_J[i + 1], temp_J[i + 2]).item()
            if loss[i] < loss[i + 1] or loss[i + 1] < gap:
                break
        loss_for_save_2 = loss_net_2(temp_J[i], gt_image)
        excel_test_line = write_excel_test(sheet=sheet_test, line=excel_test_line,
                                           content=name_div(haze_name[0]) + [i] + loss_for_save_2 + loss_for_save_1)
        f.save(excel_save)
    print_test_time(start_time, count, len(test_data_loader))
    count += 1

# gap = (np.mean(b_gap) + np.mean(a_gap)) / 2
gap = 0.0001
# 开始测试
'''
print("Start testing\n")
start_time = time.time()
count = 1
J = [0] * 999
for haze_name, haze_image, gt_image in test_data_loader:
    print('name:{}'.format(haze_name))
    loss = [0] * 999
    net.eval()
    with torch.no_grad():
        haze_image = haze_image.cuda()
        gt_image = gt_image.cuda()
        J[0] = net(haze_image, haze_image)
        J[1] = net(J[0], haze_image)
        loss[0] = image_gap(J[0], J[1]).item()
        i = 1
        while 1:
            J[i + 1] = net(J[i], haze_image)
            loss[i] = image_gap(J[i], J[i + 1]).item()
            if loss[i - 1] < loss[i] or loss[i] < gap:
                break
            i += 1
        print('循环了{}次'.format(i))
        loss = loss_net(J[i], gt_image)
        excel_test_line = write_excel_test(sheet=sheet_test, line=excel_test_line, name=haze_name[0], loss=loss)
        f.save(excel_save)

        # 保存图像测试结果
        if Is_save_image:
            im_output_for_save = get_image_for_save(J[i])
            filename = '{}.bmp'.format(haze_name[0])
            im_output_for_save.save(os.path.join(save_path, filename))
    print_test_time(start_time, count, len(test_data_loader))
    count += 1
print("Finished!")
'''
'''
print("Start testing\n")
start_time = time.time()
count = 1
J = [0] * test_round
for haze_name, haze_image, gt_image in test_data_loaderx:
    print('name:{}'.format(haze_name))
    net.eval()
    with torch.no_grad():
        haze_image = haze_image.cuda()
        gt_image = gt_image.cuda()
        J[0] = net(haze_image, haze_image)
        for i in range(test_round - 1):
            J[i + 1] = net(J[i], haze_image)
        for i in range(J[0].shape[0]):
            J_temp = [0] * test_round
            for j in range(test_round):
                J_temp[j] = J[j][i, :, :, :].unsqueeze_(0)
                gt_temp = gt_image[i, :, :, :].unsqueeze_(0)
            loss = loss_net(J_temp, gt_temp)
            excel_test_line = write_excel_test(sheet=sheet_test, line=excel_test_line, name=haze_name[0], loss=loss)
            f.save(excel_save)

            # 保存图像测试结果
            if Is_save_image:   
                for k in range(test_round):
                    im_output_for_save = get_image_for_save(J_temp[k])
                    filename = '{}_{}.bmp'.format(haze_name[i], k)
                    im_output_for_save.save(os.path.join(save_path, filename))
        print_test_time(start_time, count, len(test_data_loader))
        count += 1
print("Finished!")
'''
