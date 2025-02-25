# writer: DwlufveX
# date: 25/02/18


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from ACMD.STF import STFT
from acmd import ACMD
from Algorithm import findridges, curvesmooth
import scipy.io
# 读取 .mat 文件
mat_data = scipy.io.loadmat('Sign.mat')


Sign = mat_data['Sign'].T  # 访问 'Sig' 变量


# # 信号生成
SampFreq = 500
t = np.linspace(0, 6, SampFreq * 6+1)
Sig1 = np.exp(-0.03 * t) * np.cos(2 * np.pi * (1.3 + 25 * t + 4 * t**2 - 0.8 * t**3 + 0.07 * t**4))
IF1 = 25 + 8 * t - 2.4 * t**2 + 0.28 * t**3
Sig2 = 0.9 * np.exp(-0.06 * t) * np.cos(2 * np.pi * (2.6 + 40 * t + 8 * t**2 - 1.6 * t**3 + 0.14 * t**4))
IF2 = 40 + 16 * t - 4.8 * t**2 + 0.56 * t**3
Sig3 = 0.8 * np.exp(-0.09 * t) * np.cos(2 * np.pi * (3.9 + 60 * t + 12 * t**2 - 2.4 * t**3 + 0.21 * t**4))
IF3 = 60 + 24 * t - 7.2 * t**2 + 0.84 * t**3

# Sig = Sig1 + Sig2 + Sig3
# noise = np.random.normal(0, 0.5, len(Sig))  # 高斯噪声
# noise = 0
# Sign = Sig + noise

# 绘制加噪信号
plt.figure(figsize=(8, 6))
plt.plot(t, Sign, linewidth=1)
plt.xlabel('Time / Sec', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.show()


# 处理信号并提取成分
delta = 20
alpha0 = 1e-6
beta = 1e-9
tol = 1e-8
N = len(Sign)
# 组件1提取
Spec1, f1 = STFT(Sign.T, SampFreq, 1024, 256)
index1 = findridges(Spec1, delta)
iniIF1 = curvesmooth(f1[index1], beta)
IFest1, IAest1, sest1 = ACMD(Sign, SampFreq, iniIF1, alpha0, beta, tol, max_iter=100)

# 组件2提取
residue1 = (Sign.flatten() - sest1).reshape(N,1)
Spec2, f2 = STFT(residue1.T, SampFreq, 1024, 256)
index2 = findridges(Spec2, delta)
iniIF2 = curvesmooth(f2[index2], beta)
IFest2, IAest2, sest2 = ACMD(residue1, SampFreq, iniIF2, alpha0, beta, tol, max_iter=100)

# 组件3提取
residue2 = (residue1.flatten() - sest2).reshape(N,1)
Spec3, f3 = STFT(residue2.T, SampFreq, 1024, 256)
index3 = findridges(Spec3, delta)
iniIF3 = curvesmooth(f3[index3], beta)
IFest3, IAest3, sest3 = ACMD(residue2, SampFreq, iniIF3, alpha0, beta, tol, max_iter=100)

# 可视化结果
# 估计的瞬时频率
plt.figure(figsize=(8, 6))
# 依次绘制三条曲线
plt.plot(t, IF1, 'k', linewidth=3)
plt.plot(t, IF2, 'k', linewidth=3)
plt.plot(t, IF3, 'k', linewidth=3)
plt.plot(t, iniIF1.flatten(), 'r--', linewidth=3)
plt.plot(t, iniIF2.flatten(), 'r--', linewidth=3)
plt.plot(t, iniIF3.flatten(), 'r--', linewidth=3)
plt.xlabel('Time / Sec', fontsize=12)
plt.ylabel('Frequency / Hz', fontsize=12)
plt.show()

# 重构信号成分
plt.figure(figsize=(8, 6))
plt.subplot(3, 1, 1)
plt.plot(t, Sig1, 'k', linewidth=2)
plt.plot(t, sest1, 'b--', linewidth=2)
plt.subplot(3, 1, 2)
plt.plot(t, Sig2, 'k', linewidth=2)
plt.plot(t, sest2, 'b--', linewidth=2)
plt.subplot(3, 1, 3)
plt.plot(t, Sig3, 'k', linewidth=2)
plt.plot(t, sest3, 'b--', linewidth=2)
plt.show()

