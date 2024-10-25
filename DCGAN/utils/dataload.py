import os

import torch
# import numpy as np
import torch.nn.functional as fun
from torch.utils.data import Dataset  # 构造数据集，支持索引，总长度
# from torch.utils.data import DataLoader

from utils.readGrd import readGrdbynp

"""
深度学习训练框架/步骤：
   1. prepare dataset
      tools: Dataset and DataLoader
   2. Design model using Class
      inherit from nn.moudle
   3. Construct loss and optimizer
      using Pytorch API
   4. Training cycle
   forward, backward, update
"""


class Basement(Dataset):

    def __init__(self, root: str):  # root指定根目录,resize自定义数据大小
        super(Basement, self).__init__()
        self.root = root
        assert os.path.join(self.root), f"path '{self.root}' does not exists."  # 判断文件路径是否存在

        dg_names = [i for i in os.listdir(os.path.join(self.root)) if i.endswith(".grd")]
        self.dg_list = [os.path.join(self.root, i) for i in dg_names]  # 获取dg文件夹下的全部grd文件

        # check files
        for i in self.dg_list:  # 注意for i in self.dg_list或者dg_names中i为字符串类型而非整形
            if os.path.exists(i) is False:
                raise FileExistsError(f"file {i} doesn't exissts.")

    def __getitem__(self, index):  # 按照文件列表当前索引下标对应的sample与label
        dg = readGrdbynp(self.dg_list[index])
        dg = torch.from_numpy(dg).unsqueeze(0).unsqueeze(0)
        # print(mod_label.shape,dg.shape)
        # 对数据重采样重构成网络目标输入大小200*200, 200*200，二维数据interpolate函数输入需为[b,c,h,w]
        dg_sample = fun.interpolate(dg, size=[64, 64], mode='bilinear', align_corners=True)
        #  将batch通道数维度去掉
        dg_sample = dg_sample.squeeze(0)

        return dg_sample.float()

    def __len__(self):
        return len(self.dg_list)


# # # 测试数据集读取是否成功
if __name__ == '__main__':
    myroot = r"D:\Project\DCGAN\data\real"
    mydata = Basement(myroot)
    print(mydata.__len__())
    print(mydata.dg_list[0])
    print(mydata[0].shape, torch.max(mydata[0]))
