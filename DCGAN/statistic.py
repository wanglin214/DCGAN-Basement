import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader
from utils import dataload

dataroot = r"D:\Project\DCGAN\data\total"
dataloader = DataLoader(dataload.Basement(dataroot),
                        batch_size=10012,
                        shuffle=False,
                        )  # 加载测试数据
print(len(dataloader))

if __name__ == '__main__':
    for i, data in enumerate(dataloader):
        print(data.shape)
        total = data.view(-1).numpy()
        print(total.shape[0], type(total))
        # 统计均值，标准差
        mean_value = np.mean(total)
        std_dev = np.std(total)
        print(mean_value, std_dev, np.min(total), np.max(total))

        # 设置字体为 'Times New Roman'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

        # 绘图
        plt.hist(total, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')

        # 拟合截断的高斯分布
        # 生成从零开始的高斯分布的数据
        data = np.random.normal(loc=1.54, scale=0.86, size=1000)
        data = np.clip(data, 0, 16.29)  # 将负值截断为零


        # 使用 curve_fit 拟合正态分布的概率密度函数
        def gaussian_pdf(x, mu, sigma):
            return norm.pdf(x, mu, sigma)


        x = np.linspace(0, np.max(total), 10000)
        fit_params, covariance = curve_fit(gaussian_pdf, x, gaussian_pdf(x, np.mean(data), np.std(data)))

        # 生成拟合的概率密度曲线
        fit_curve = gaussian_pdf(x, *fit_params)

        # 绘制拟合曲线
        plt.plot(x, fit_curve, 'k', linewidth=2)
        # 添加图例和标签
        plt.xlabel('Depth/km', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.xlim(0, 17)
        plt.ylim(0, 1)
        plt.text(2.5, 0.9, f'Fit: mean = {fit_params[0]:.2f}, std = {fit_params[1]:.2f}', fontsize=16)
        plt.text(15.0, 0.05, f'(c)', fontsize=20)

        # 设置刻度标签的字体大小
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        # 设置图片大小，使其充满画布
        fig = plt.gcf()
        fig.set_size_inches(4, 4)  # 可以根据需要调整大小

        # 保存图形，设置 dpi 为 600
        plt.savefig("real+DCGANdata_distribution.jpeg", bbox_inches='tight', dpi=600)
        # 显示图形
        plt.show()
