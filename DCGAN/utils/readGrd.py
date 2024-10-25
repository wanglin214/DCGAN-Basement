##读取异常grd文件与剩余密度模型文件
# @Time: 2023/2/17 10:59
# @Author: WangLin
# @File: ReadGrd.py:读取grd文件
# @Software: PyCharm

import os
import numpy as np


# 创建读取grd文件的函数,返回网格文件数组grdata(Ny,Nx),Ny是列数——对应dim=0,
def readGrdbynp(filepath):
    with open(filepath, "r", encoding="UTF-8") as infile:
        infile.readline()  # 跳过标准surfer 6 text格式文件首行的标注 DSAA
        str = infile.readline().split()  # 读取第二行点、线数
        Nx = int(str[0])
        Ny = int(str[1])
        # print(Nx, Ny)
        str = infile.readline().split()  # 读取第三行X方向最小值、最大值
        Xmin = float(str[0])
        Xmax = float(str[1])
        # print(Xmin, Xmax)
        str = infile.readline().split()  # 读取第四行Y方向最小值、最大值
        Ymin = float(str[0])
        Ymax = float(str[1])
        str = infile.readline().split()  # 读取第五行网格数据最小值、最大值
        gdmin = float(str[0])
        gdmax = float(str[1])
        # print(gdmin, gdmax)
    infile.close()
    gd = np.loadtxt(filepath, skiprows=5)
    # print(gd.shape, gd.min(), gd.max())
    # os.system('pause')
    return gd


# 局部子程序测试
if __name__ == '__main__':
    Fgafile = r"D:\Project\DL_interface_inversion\origin_data\dg\FwGravofBasin_0001_Dens_01.grd"
    grddata = readGrdbynp(Fgafile)
    print(grddata.shape)
    print(grddata.min())
