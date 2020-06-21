# -*- coding: utf-8 -*-
# git clone https://github.com/zhanglideng/At_J_net.git
import sys

# sys.path.append('/home/aistudio/external-libraries')
import os

# if not os.path.exists('/home/aistudio/.torch/models/vgg16-397923af.pth'):
#    os.system('mkdir /home/aistudio/.torch')
#    os.system('mkdir /home/aistudio/.torch/models')
#    os.system('cp /home/aistudio/work/pre_model/*  /home/aistudio/.torch/models/')
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.loss import *
from utils.print_time import *
from utils.save_log_to_excel import *
from dataloader import Cycle_DataSet
from cycle_model import *
import time
import xlwt
from utils.ms_ssim import *

LR = 0.0005  # 学习率
EPOCH = 200  # 轮次
BATCH_SIZE = 1  # 批大小
excel_train_line = 1  # train_excel写入的行的下标
excel_val_line = 1  # val_excel写入的行的下标
alpha = 1  # 损失函数的权重
accumulation_steps = 8  # 梯度积累的次数，类似于batch-size=64
# itr_to_lr = 10000 // BATCH_SIZE  # 训练10000次后损失下降50%
itr_to_excel = 128 // BATCH_SIZE  # 训练64次后保存相关数据到excel

weight = [1, 1, 1, 1, 1, 1, 1, 1, 1]
loss_num = len(weight)  # 包括参加训练和不参加训练的loss

if os.path.exists('/input'):
    data_path = '/input'
else:
    data_path = '/home/ljh/zhanglideng'
train_hazy_path = data_path + '/data/nyu_cycle/train_hazy/'
val_hazy_path = data_path + '/data/nyu_cycle/val_hazy/'
train_gth_path = data_path + '/data/nyu_cycle/train_gth/'
val_gth_path = data_path + '/data/nyu_cycle/val_gth/'

save_path = './cycle_result_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '/'
save_model_name = save_path + 'cycle_model.pt'  # 保存模型的路径
excel_save = save_path + 'result.xls'  # 保存excel的路径
mid_save_ed_path = './mid_model/cycle_model.pt'  # 保存的中间模型，用于下一步训练。

# 初始化excel
f, sheet_train, sheet_val, sheet_val_every_image = init_excel(kind='train')
if not os.path.exists('./mid_model'):
    os.makedirs('./mid_model')

if os.path.exists(data_path + 'pre_model/J_model/cycle_model.pt'):
    print('加载预训练模型')
    net = torch.load(data_path + 'pre_model/J_model/cycle_model.pt').cuda()
else:
    print('创建新模型')
    net = cycle().cuda()

if not os.path.exists(save_path):
    os.makedirs(save_path)

# 数据转换模式
transform = transforms.Compose([transforms.ToTensor()])
# 读取训练集数据
train_path_list = [train_hazy_path, train_gth_path]
train_data = Cycle_DataSet(transform, train_path_list)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

# 读取验证集数据
val_path_list = [val_hazy_path, val_gth_path]
val_data = Cycle_DataSet(transform, val_path_list)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

min_loss = 999999999
min_epoch = 0
itr = 0
start_time = time.time()

# 开始训练
print("\nstart to train!")
for epoch in range(EPOCH):
    index = 0
    train_loss = 0
    loss_excel = [0] * loss_num
    net.train()
    for name, haze_image, gt_image in train_data_loader:
        index += 1
        itr += 1
        haze_image = haze_image.cuda()
        gt_image = gt_image.cuda()
        # t_gth = t_gth.cuda()
        J1 = net(haze_image, haze_image)
        J2 = net(J1, haze_image)
        J3 = net(J2, haze_image)
        loss_image = [J1, J2, J3, gt_image]
        loss, temp_loss = loss_function(loss_image, weight)
        train_loss += loss.item()
        loss_excel = [loss_excel[i] + temp_loss[i] for i in range(len(loss_excel))]
        loss = loss / accumulation_steps
        loss.backward()
        # 3. update parameters of net
        if ((index + 1) % accumulation_steps) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient
        if np.mod(index, itr_to_excel) == 0:
            loss_excel = [loss_excel[i] / itr_to_excel for i in range(len(loss_excel))]
            print('epoch %d, %03d/%d' % (epoch + 1, index, len(train_data_loader)))
            print('LOSS=')
            print(loss_excel)
            print('\n')
            print_time(start_time, index, EPOCH, len(train_data_loader), epoch)
            excel_train_line = write_excel_train(sheet=sheet_train, line=excel_train_line, epoch=epoch,
                                                 itr=itr, loss=loss_excel, weight=weight)
            f.save(excel_save)
            loss_excel = [0] * loss_num
    optimizer.step()
    optimizer.zero_grad()
    # scheduler.step()
    loss_excel = [0] * loss_num
    val_loss = 0
    with torch.no_grad():
        net.eval()
        for name, haze_image, gt_image in val_data_loader:
            haze_image = haze_image.cuda()
            gt_image = gt_image.cuda()
            # t_gth = t_gth.cuda()
            # J, J_reconstruct, t, haze_reconstruct = net(haze_image, haze_image)

            J1 = net(haze_image, haze_image)
            J2 = net(J1, haze_image)
            J3 = net(J2, haze_image)
            loss_image = [J1, J2, J3, gt_image]
            loss, temp_loss = loss_function(loss_image, weight)
            excel_train_line = write_excel_every_val(sheet=sheet_val_every_image, line=excel_train_line, epoch=epoch,
                                                     name=name[0], loss=temp_loss)
            f.save(excel_save)
            # loss_image = [J, gt_image, J_reconstruct, t, t_gth, haze_reconstruct, haze_image]
            # loss, temp_loss = loss_function(loss_image, weight)
            loss_excel = [loss_excel[i] + temp_loss[i] for i in range(len(loss_excel))]
    train_loss = train_loss / len(train_data_loader)
    loss_excel = [loss_excel[i] / len(val_data_loader) for i in range(len(loss_excel))]
    for i in range(len(loss_excel)):
        val_loss = val_loss + loss_excel[i] * weight[i]
    print('val loss=')
    print(loss_excel)
    print('\n')
    excel_val_line = write_excel_val(sheet=sheet_val, line=excel_val_line, epoch=epoch,
                                     loss=[loss_excel, val_loss, train_loss])
    f.save(excel_save)
    if val_loss < min_loss:
        min_loss = val_loss
        min_epoch = epoch
        torch.save(net, save_model_name)
        torch.save(net, mid_save_ed_path)
        print('saving the epoch %d model with %.5f' % (epoch + 1, min_loss))
print('Train is Done!')
