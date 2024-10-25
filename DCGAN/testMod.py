# coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com
#
# or create issues
# =============================================================================
import torch
import torch.nn.functional as fun
from model import DCGAN
import numpy as np
import matplotlib.pyplot as plt
from utils import outGrd
from scipy.ndimage import gaussian_filter

# 创建二维数组
x = np.linspace(0, 400, 100)
y = np.linspace(0, 400, 100)
X, Y = np.meshgrid(x, y)

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 载入模型权重
netG = DCGAN.Generator().to(device)
modelpath = 'models/netG400.pth'
# path_check = 'models/checkpoint_G_400_400.pth'
check_point = torch.load(modelpath, map_location=device)
# check_point = torch.load(path_check)
# 将 state_dict 加载到 netG 中
netG.load_state_dict(check_point)
# netG.load_state_dict(check_point['model_state_dict'])
# opt_state = check_point['optimizer_state_dict']
# model_state = check_point['model_state_dict']
# print(opt_state)
# print(model_state)
netG.eval()  # 设置推理模式，使得 dropout 和 batchnorm 等网络层在 train 和 val 模式间切换
torch.no_grad()  # 停止 autograd 模块的工作，以起到加速和节省显存

nz = 200  # 噪声维度
sigma = 4  # 滤波参数

for i in range(0, 9500):
    noise = torch.randn(1, nz, 1, 1, device=device)
    fake = netG(noise)
    # print(fake.shape)
    fake = fun.interpolate(fake, size=[100, 100], mode='bilinear', align_corners=True)
    Z = 16.0 * fake.squeeze(0).squeeze(0).detach().cpu().numpy()
    # print(Z.shape, type(Z))
    # # 绘制等值线图并填充彩虹色标
    # plt.contourf(X, Y, Z, levels=20, cmap='rainbow_r')
    # plt.colorbar(label='Value')
    # plt.xlabel('X/km')
    # plt.ylabel('Y/km')
    # plt.title('Basement depth by DCGAN')
    # plt.show()
    # plt.close()
    Z[Z < 0] = 0
    Z = gaussian_filter(Z, sigma=sigma)
    mean_real = 1.3794324
    std_real = 1.7700067
    min_real = 0.0
    max_real = 17.43
    # Z = ((Z - np.mean(Z)) / np.std(Z)) * std_real + mean_real
    # Z = (Z - np.min(Z)) * (17.43 - 0.0) / (np.max(Z) - np.min(Z)) + 0.0
    filepath = 'data/Generated/SedofBasin_' + "{:04d}".format(i + 1 + 512) + '.grd'
    outGrd.outGrd(filepath, Z, np.min(Z), np.max(Z))
