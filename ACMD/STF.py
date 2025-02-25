# writer: DwlufveX
# date: 25/02/18

import numpy as np
from scipy.signal import hilbert

def STFT(Sig, SampFreq, N, WinLen):
    if np.isrealobj(Sig):
        Sig = hilbert(Sig)
    Sig = Sig.T
    Sigshape = Sig.shape
    SigLen = Sigshape[0]
    WinLen = int(np.ceil(WinLen / 2) * 2)  # 保证窗函数长度为偶数
    t = np.linspace(-1, 1, WinLen)
    sigma = 0.28
    WinFun = (np.pi * sigma ** 2) ** (-1 / 4) * np.exp(-t ** 2 / (2 * sigma ** 2))

    Lh = (WinLen - 1) / 2  # 窗口的半长度
    Spec = np.zeros((N, SigLen),dtype=complex)

    for i in range(SigLen):
        tau = np.arange(-min([round(N/2) - 1, Lh, i]), min([round(N/2) - 1, Lh, SigLen - i - 1]) + 1)
        temp = np.floor(i + tau).astype(int)

        # 使用 np.clip 来确保 temp 的索引不会超出范围
        # temp = np.clip(temp, 0, SigLen - 1)

        temp1 = np.floor(Lh + tau).astype(int)
        # temp1 = np.clip(temp1, 0, WinLen - 1)  # 确保 temp1 也不越界

        rSig = Sig[temp].T
        rSig = rSig * np.conj(WinFun[temp1])
        Spec[:rSig.shape[1], i] = rSig.flatten()
        # print(i)

    Spec = np.fft.fftshift(np.fft.fft(Spec, axis=0), axes=0)

    nLevel, SigLen = Spec.shape
    f = np.linspace(-SampFreq / 2, SampFreq / 2, nLevel)
    t = np.arange(SigLen) / SampFreq

    return Spec, f
