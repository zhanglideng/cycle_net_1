# -*- coding: utf-8 -*-
# git clone https://github.com/zhanglideng/cycle_net_1.git

import sys
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.loss import *
from utils.print_time import *
from utils.save_log_to_excel import *
from dataloader import *
from new_cycle_model import *
import time
import xlwt
import argparse
from utils.ms_ssim import *

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for CycleDehazeNet')
parser.add_argument('-learning_rate', help='Set the learning rate', default=5e-4, type=float)
parser.add_argument('-batch_size', help='Set the training batch size', default=1, type=int)
parser.add_argument('-accumulation_steps', help='Set the accumulation steps', default=8, type=int)
parser.add_argument('-dropout', help='Set the dropout ratio', default=0.3, type=int)
parser.add_argument('-itr_to_excel', help='Save to excel after every n trainings', default=128, type=int)
parser.add_argument('-epoch', help='Set the epoch', default=50, type=int)
parser.add_argument('-category', help='Set image category (NYU or NTIRE2018?)', default='NYU', type=str)
parser.add_argument('-data_path', help='Set the data_path', default='/input', type=str)
parser.add_argument('-pre_model', help='Whether to use a pre-trained model', default=True, type=bool)
parser.add_argument('-gth_train', help='Whether to add Gth training', default=False, type=bool)
parser.add_argument('-loss_weight', help='Set the loss weight',
                    default=[5, 20, 80, 10, 10, 1, 4, 16, 10, 10, 1, 4, 16, 10, 10], type=list)
args = parser.parse_args()

learning_rate = args.learning_rate  # 学习率
accumulation_steps = args.accumulation_steps  # 梯度累积
batch_size = args.batch_size  # 批大小
epoch = args.epoch  # 轮次
dropout = args.dropout  # dropout的比例
category = args.category  # NYU或NTIRE训练集
itr_to_excel = args.itr_to_excel  # 每训练itr次保存到excel中
weight = args.loss_weight  # 损失函数权重
loss_num = len(weight)  # 损失函数的数量
data_path = args.data_path  # 数据存放的路径
Is_pre_model = args.pre_model  # 是否使用预训练模型
Is_gth_train = args.gth_train  # 是否使用Gth参与训练

if Is_pre_model:
    print('加载预训练模型')
    net = torch.load(data_path + '/pre_model/J_model/best_cycle_model.pt').cuda()
else:
    print('创建新模型')
    net = cycle(dropout=0.3).cuda()

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(total_params))

if category == 'NYU':
    train_hazy_path = data_path + '/data/nyu_cycle/train_hazy/'
    val_hazy_path = data_path + '/data/nyu_cycle/val_hazy/'
    train_gth_path = data_path + '/data/nyu_cycle/train_gth/'
    val_gth_path = data_path + '/data/nyu_cycle/val_gth/'
else:
    train_hazy_path = data_path + '/data/cut_ntire_cycle/train_hazy/'
    val_hazy_path = data_path + '/data/cut_ntire_cycle/val_hazy/'
    train_gth_path = data_path + '/data/cut_ntire_cycle/train_gth/'
    val_gth_path = data_path + '/data/cut_ntire_cycle/val_gth/'

save_path = data_path + '/train_result/cycle_result_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '/'
save_model_name = save_path + 'cycle_model.pt'  # 保存模型的路径
excel_save = save_path + 'result.xls'  # 保存excel的路径
mid_save_ed_path = './mid_model/cycle_model.pt'  # 保存的中间模型，用于意外停止后继续训练。

log = 'learning_rate: {}\nbatch_size: {}\nepoch: {}\ndropout: {}\ncategory: {}\n' \
      'Is_gth_train: {}\nloss_weight: {}\nIs_pre_model: {}\ntotal_params: {}\nsave_file_name: {}'.format(learning_rate,
                                                                                                         batch_size,
                                                                                                         epoch, dropout,
                                                                                                         category,
                                                                                                         Is_gth_train,
                                                                                                         weight,
                                                                                                         Is_pre_model,
                                                                                                         total_params,
                                                                                                         save_path)
print('--- Hyper-parameters for training ---')
print(log)
if not os.path.exists('./mid_model'):
    os.makedirs('./mid_model')
if not os.path.exists(save_path):
    os.makedirs(save_path)
with open(save_path + 'detail.txt', 'w') as f:
    f.write(log)
f, sheet_train, sheet_val, sheet_val_every_image = init_excel(kind='train')

transform = transforms.Compose([transforms.ToTensor()])
train_path_list = [train_hazy_path, train_gth_path]
val_path_list = [val_hazy_path, val_gth_path]
if category == 'NYU':
    train_data = Cycle_DataSet(transform, Is_gth_train, train_path_list)
    val_data = Cycle_DataSet(transform, Is_gth_train, val_path_list)
else:
    train_data = Ntire_DataSet(transform, Is_gth_train, train_path_list)
    val_data = Ntire_DataSet(transform, Is_gth_train, val_path_list)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)

# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

excel_train_line = 1
excel_val_line = 1
excel_val_every_image_line = 1
min_loss = 99999
start_time = time.time()

# 开始训练
print("\nstart to train!")
for epo in range(epoch):
    index = 0
    train_loss = 0
    loss_excel = [0] * loss_num
    net.train()
    for name, haze_image, gt_image in train_data_loader:
        index += 1
        haze_image = haze_image.cuda()
        gt_image = gt_image.cuda()
        J1 = net(haze_image, haze_image)
        J2 = net(J1, haze_image)
        J3 = net(J2, haze_image)
        loss_image = [J1, J2, J3, gt_image]
        loss, temp_loss = loss_function(loss_image, weight)
        train_loss += loss.item()
        loss_excel = [loss_excel[i] + temp_loss[i] for i in range(len(loss_excel))]
        loss = loss / accumulation_steps
        loss.backward()
        if ((index + 1) % accumulation_steps) == 0:
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient
        if np.mod(index, itr_to_excel) == 0:
            loss_excel = [loss_excel[i] / itr_to_excel for i in range(len(loss_excel))]
            print('epoch %d, %03d/%d' % (epo + 1, index, len(train_data_loader)))
            print('train loss = {}\n'.format(loss_excel))
            print_time(start_time, index, epoch, len(train_data_loader), epo)
    optimizer.step()
    optimizer.zero_grad()
    loss_excel = [0] * loss_num
    val_loss = 0
    with torch.no_grad():
        net.eval()
        for name, haze_image, gt_image in val_data_loader:
            haze_image = haze_image.cuda()
            gt_image = gt_image.cuda()
            J1 = net(haze_image, haze_image)
            J2 = net(J1, haze_image)
            J3 = net(J2, haze_image)
            loss_image = [J1, J2, J3, gt_image]
            loss, temp_loss = loss_function(loss_image, weight)
            loss_excel = [loss_excel[i] + temp_loss[i] for i in range(len(loss_excel))]
    train_loss = train_loss / len(train_data_loader)
    loss_excel = [loss_excel[i] / len(val_data_loader) for i in range(len(loss_excel))]
    for i in range(len(loss_excel)):
        val_loss = val_loss + loss_excel[i] * weight[i]
    print('val loss = {}\n'.format(loss_excel))
    excel_val_line = write_excel_val(sheet=sheet_val, line=excel_val_line, epoch=epo,
                                     loss=[loss_excel, val_loss, train_loss])
    f.save(excel_save)
    if val_loss < min_loss:
        min_loss = val_loss
        torch.save(net, save_model_name)
        torch.save(net, mid_save_ed_path)
        print('saving the epoch %d model with %.5f' % (epo + 1, min_loss))
print('Train is Done!')
