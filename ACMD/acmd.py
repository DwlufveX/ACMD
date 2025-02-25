# writer: DwlufveX
# date: 25/02/18

import numpy as np
from scipy.sparse import spdiags, diags,bmat
from scipy.sparse.linalg import spsolve
from scipy.integrate import cumtrapz



def Differ(y, delta):
    """
    Compute the derivative of a discrete time series y.

    Parameters:
    y: discrete time series (1D numpy array)
    delta: sampling time interval

    Returns:
    ybar: derivative of the time series (1D numpy array)
    """
    L = len(y)
    ybar = np.zeros(L)

    # Central difference for interior points
    ybar[1:-1] = (y[2:] - y[:-2]) / (2 * delta)

    # Forward difference for the first point
    ybar[0] = (y[1] - y[0]) / delta

    # Backward difference for the last point
    ybar[-1] = (y[-1] - y[-2]) / delta

    return ybar


def ACMD(s, fs, eIF, alpha0, beta, tol, max_iter):
    """
    Adaptive Chirp Mode Decomposition (ACMD)

    Parameters:
    s: measured signal, a 1D numpy array
    fs: sampling frequency
    eIF: initial instantaneous frequency (IF) time series of a certain signal mode, a 1D numpy array
    alpha0: penalty parameter controlling the filtering bandwidth of ACMD
    beta: penalty parameter controlling the smooth degree of the IF increment during iterations
    tol: tolerance of convergence criterion

    Returns:
    IFest: the finally estimated IF
    sest: the finally estimated signal mode
    IAest: the finally estimated instantaneous amplitude (IA) of the signal mode
    """
    N = len(eIF.T)  # Number of samples
    t = np.arange(0, N) / fs  # Time array

    e = np.ones(N)
    e2 = -2 * e
    # 构造第二阶差分矩阵
    oper = spdiags([e, e2, e], [0, 1, 2], N - 2, N, format='csc')  # (N-2) x N
    spzeros = spdiags([np.zeros(N)], [0], N - 2, N, format='csc')  # (N-2) x N

    opedoub = oper.T @ oper  # 确保 opedoub 被初始化c
    # 构造 phim 矩阵
    # MATLAB 中 phim 是 [oper spzeros; spzeros oper]，维度为 (2*(N-2)) x (2*N)
    phim = bmat([[oper, spzeros], [spzeros, oper]], format='csc')  # (2*(N-2)) x (2*N)

    # 构造 phidoubm
    phidoubm = phim.T @ phim  # (2*N) x (2*N)

    iternum = max_iter  # Maximum allowable iterations
    IFsetiter = np.zeros((iternum, N))  # Collection of IF time series at each iteration
    ssetiter = np.zeros((iternum, N))  # Collection of signal modes at each iteration
    ysetiter = np.zeros((iternum, 2 * N))

    iter = 0  # Iteration counter
    sDif = tol + 1  # Initial difference
    alpha = alpha0

    while sDif > tol and iter < iternum:
        cosm = np.cos(2 * np.pi * cumtrapz(eIF, t, initial=0))
        sinm = np.sin(2 * np.pi * cumtrapz(eIF, t, initial=0))
        Cm = np.diag(cosm.flatten())
        Sm = np.diag(sinm.flatten())
        Kerm = np.hstack((Cm, Sm))  # Kernel matrix
        Kerdoubm = Kerm.T @ Kerm

        # Update demodulated signals
        ym = spsolve(1 / alpha * phidoubm + Kerdoubm, Kerm.T @ s)
        si = Kerm @ ym
        ssetiter[iter] = si
        ysetiter[iter] = ym

        # Update the IFs
        ycm = ym[:N]
        ysm = ym[N:]
        ycmbar = Differ(ycm, 1/fs)
        ysmbar = Differ(ysm, 1/fs)
        deltaIF = (ycm * ysmbar - ysm * ycmbar) / (ycm ** 2 + ysm ** 2) / (2 * np.pi)
        deltaIF = spsolve(1 / beta * opedoub + diags(np.ones(N)), deltaIF)
        eIF = eIF - deltaIF
        IFsetiter[iter] = eIF

        # Compute the convergence index
        if iter > 0:
            sDif = np.linalg.norm(ssetiter[iter] - ssetiter[iter - 1]) ** 2 / np.linalg.norm(ssetiter[iter - 1]) ** 2

        iter += 1

    iter -= 1  # Adjust the iteration count
    IFest = IFsetiter[iter]
    sest = ssetiter[iter]
    ycm = ysetiter[iter, :N]
    ysm = ysetiter[iter, N:]
    IAest = np.sqrt(ycm ** 2 + ysm ** 2)  # Estimated IA

    return IFest, IAest, sest