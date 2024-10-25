import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

# 生成从零开始的高斯分布的数据
data = np.random.normal(loc=2, scale=1, size=1000)
data = np.maximum(data, 0)  # 将负值截断为零

# 使用 curve_fit 拟合正态分布的概率密度函数
def gaussian_pdf(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

x = np.linspace(0, max(data), 1000)
fit_params, covariance = curve_fit(gaussian_pdf, x, gaussian_pdf(x, np.mean(data), np.std(data)))

# 生成拟合的概率密度曲线
fit_curve = gaussian_pdf(x, *fit_params)

# 绘制直方图和拟合曲线
plt.hist(data, bins=30, density=True, alpha=0.7, color='b', edgecolor='black', label='Data Histogram')
plt.plot(x, fit_curve, 'k', linewidth=2, label='Fit Curve')

plt.title('Zero-Truncated Gaussian Distribution: Data and Fit')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()
