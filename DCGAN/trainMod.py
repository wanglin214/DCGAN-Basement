# coding: utf-8
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler
import torch.nn.functional as fun
# from torch.cuda.amp import autocast, GradScaler  # 混合精度所需要的模块
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from utils import dataload, outGrd
from model import DCGAN
import os
from visdom import Visdom

# 设置一个随机种子，方便进行可重复性实验
manualSeed = 0
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.benchmark = True

# 基本参数配置
# 数据集所在路径
dataroot = "data/real"
# Batch size 大小
batch_size = 128
# 图片大小
image_size = 64
# 图片的通道数，生成器最后输出图片通道数，也是判别器第一个卷积输入通道
nc = 1
# 噪声向量维度
nz = 200
# 生成器最后一个卷积输入特征图通道数量单位
ngf = 64
# 判别器第一个卷积输出特征图通道数量单位
ndf = 64
# 损失函数
criterion = nn.BCELoss()
# 真假标签
real_label = 1.0
fake_label = 0.0
# 是否使用GPU训练
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# 创建生成器与判别器
netG = DCGAN.Generator().to(device).apply(DCGAN.weights_init)
netD = DCGAN.Discriminator().to(device).apply(DCGAN.weights_init)
# 读取数据加载
dataloader = DataLoader(dataload.Basement(dataroot),
                        batch_size=batch_size,
                        shuffle=True,
                        # drop_last=True,
                        pin_memory=True,
                        prefetch_factor=4,
                        num_workers=8)  # 加载测试数据

# 定义主程序
if __name__ == '__main__':
    lr = 1e-3
    beta1 = 0.5
    # scaler = GradScaler()
    # G和D的优化器，使用Adam
    # Adam学习率与动量参数
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    schedulerD = lr_scheduler.CosineAnnealingLR(optimizerD, T_max=400, eta_min=1e-4, last_epoch=-1)
    schedulerG = lr_scheduler.CosineAnnealingLR(optimizerG, T_max=400, eta_min=1e-8, last_epoch=-1)
    # 损失变量
    G_losses = []
    D_losses = []
    # visdom监控损失对象实例化
    viz_G = Visdom()
    viz_G.line([0.], [0.], win='Generator_loss', opts=dict(title='Generator_loss'))
    viz_D = Visdom()
    viz_D.line([0.], [0.], win='Discriminator_loss', opts=dict(title='Discriminator_loss'))
    # 总epochs
    num_epochs = 800
    # 对于固定的噪声模型测试
    rnoise = torch.randn(1, nz, 1, 1, device=device)
    # 模型缓存接口
    if not os.path.exists('models'):
        os.mkdir('models')
        print("Starting Training Loop...")

    for epoch in range(num_epochs):
        lossG = 0.0
        lossD = 0.0
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            # print(data.shape)
            # os.system('pause')

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # ###########################
            # 训练真实图片
            netD.zero_grad()
            real_data = data.to(device)
            label = torch.full((batch_size,), real_label, device=device)
            output = netD(real_data).view(-1)
            # 计算真实图片损失，梯度反向传播，真实的图片对应于标签1即优化其逐渐判别真图片为真
            errD_real = criterion(output, label)
            errD_real.backward()
            # D_x = output.mean().item()

            # 训练生成图片
            # 产生latent vectors
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # 使用G生成图片
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            # 计算生成图片损失，梯度反向传播，判别器生成的图片对应于标签0即优化其逐渐判别假图片为假
            errD_fake = criterion(output, label)
            errD_fake.backward()
            # D_G_z1 = output.mean().item()
            # 累加误差，参数更新
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z))),即让生成器生成的假图片与真实1误差最小，从而达到让生成器以假乱真
            # ###########################
            netG.zero_grad()
            label.fill_(real_label)  # 给生成图赋标签
            # 对生成图再进行一次判别
            output = netD(fake).view(-1)
            # 计算生成图片损失，梯度反向传播
            errG = criterion(output, label)
            errG.backward()
            # D_G_z2 = output.mean().item()
            optimizerG.step()

            # if (i+1) % 2 ==0:
            #     # 打印每个epoch内部损失
            print(f'{epoch + 1}-{i + 1}-lossG===>>{errG.item()}')
            print(f'{epoch + 1}-{i + 1}-lossD==>>{errD.item()}')

            # 存储损失
            lossG = lossG + errG  # 累加batch损失
            lossD = lossD + errD  # 累加batch损失

        # schedulerD.step()
        schedulerG.step()
        # 每个epoch监控其平均损失绘图
        avg_lossG = lossG / len(dataloader)
        avg_lossD = lossD / len(dataloader)
        G_losses.append(avg_lossG.item())
        D_losses.append(avg_lossD.item())
        viz_G.line([avg_lossG.item()], [epoch + 1], win='Generator_loss', update='append')
        viz_D.line([avg_lossD.item()], [epoch + 1], win='Discriminator_loss', update='append')
        # 每10个epoch保存一次权重
        if (epoch + 1) % 10 == 0:
            checkpointG = {"model_state_dict": netG.state_dict(),
                           "optimizer_state_dict": optimizerG.state_dict(),
                           "epoch": epoch}
            path_checkpointG = "./models/checkpoint_G_{}_800_lr48.pth".format(epoch+1)
            torch.save(checkpointG, path_checkpointG)  # 每隔5个epoch保存一个断点文件
            checkpointD = {"model_state_dict": netD.state_dict(),
                           "optimizer_state_dict": optimizerD.state_dict(),
                           "epoch": epoch}
            path_checkpointD = "./models/checkpoint_D_{}_800_lr48.pth".format(epoch+1)
            torch.save(checkpointD, path_checkpointD)  # 每隔5个epoch保存一个断点文件
            print(' weight save successfully!')

        # 每个epoch输出一次生成结果
        fake_epoch = netG(rnoise)
        # 重采样为100*100
        fake_epoch = fun.interpolate(fake_epoch, size=[100, 100], mode='bilinear', align_corners=True)

        depth = 16.0 * fake_epoch.squeeze(0).squeeze(0).detach().cpu().numpy()
        filepath = 'data/out/fake_' + "{:04d}".format(epoch + 1) + '.grd'
        outGrd.outGrd(filepath, depth, np.min(depth), np.max(depth))
        # plt.close()

    np.savetxt('Generator_loss800_lr48.txt', G_losses)
    np.savetxt('Discriminator_loss800_lr48.txt', D_losses)
    torch.save(netG.state_dict(), 'models/netG800_lr48.pth')
    torch.save(netD.state_dict(), 'models/netD800_lr48.pth')
