# writer: DwlufveX
# date: 25/02/18

import numpy as np
from scipy.sparse import diags


def curvesmooth(f, beta):
    f = f.reshape(1,len(f))
    K, N = f.shape  # K 是信号成分的数量，N 是采样点数
    e = np.ones(N)
    e2 = -2 * e
    oper = diags([e, e2, e], [0, 1, 2], (N - 2, N))  # 构造二阶差分矩阵
    opedoub = oper.T @ oper
    outf = np.zeros_like(f)

    for i in range(K):
        outf[i, :] = np.linalg.solve((2 / beta) * opedoub + np.eye(N), f[i, :])  # 平滑IF曲线

    return outf


def Differ(y, delta):
    L = len(y)
    ybar = np.zeros(L - 2)

    for i in range(1, L - 1):
        ybar[i - 1] = (y[i + 1] - y[i - 1]) / (2 * delta)

    ybar = np.concatenate(([(y[1] - y[0]) / delta], ybar, [(y[-1] - y[-2]) / delta]))
    return ybar


def TFspec(IFmulti, IAmulti, band):
    frnum = 1024  # 频率区间的频率 bin 数量
    fbin = np.linspace(band[0], band[1], frnum)
    num = IFmulti.shape[0]  # 信号模式数量
    N = IFmulti.shape[1]  # 信号长度
    ASpec = np.zeros((frnum, N))
    delta = int(np.floor(frnum * 0.1e-2))

    for kk in range(num):
        temp = np.zeros((frnum, N))
        for ii in range(N):
            _, index = min(abs(fbin - IFmulti[kk, ii]))
            lindex = max(index - delta, 0)
            rindex = min(index + delta, frnum)
            temp[lindex:rindex, ii] = IAmulti[kk, ii]
        ASpec += temp

    return ASpec, fbin

def findridges(Spec, delta):
    """
    Ridge detection algorithm, Algorithm 1 in the paper:
    "Separation of Overlapped Non-Stationary Signals by Ridge Path Regrouping and Intrinsic Chirp Component Decomposition"
    IEEE Sensors Journal, 2017.
    """
    # Compute the absolute value of the time-frequency distribution
    Spec = np.abs(Spec)
    M, N = Spec.shape
    index = np.zeros(N, dtype=int)

    # Find the maximum value in the spectrogram
    fmax, tmax = np.unravel_index(np.argmax(Spec), Spec.shape)
    index[tmax] = fmax

    f0 = fmax

    # Move forward in time
    for j in range(tmax + 1, N):
        low = max(0, f0 - delta)
        up = min(M - 1, f0 + delta)
        f0 = np.argmax(Spec[low:up, j]) + low
        index[j] = f0

    f1 = fmax
    # Move backward in time
    for j in range(tmax - 1, -1, -1):
        low = max(0, f1 - delta)
        up = min(M - 1, f1 + delta)
        f1 = np.argmax(Spec[low:up, j]) + low
        index[j] = f1

    return index



