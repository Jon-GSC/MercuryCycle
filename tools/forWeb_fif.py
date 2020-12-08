import numpy as np
import random


def d(x):
    return float(x[-1] - x[0])


def an(x, i):
    return (x[i] - x[i - 1]) / d(x)


def dn(x, i):
    return (x[i - 1] - 0 * x[i]) / d(x)


def cn(x, y, i, sn):
    return (y[i] - y[i - 1]) / d(x) - sn * (y[-1] - y[0]) / d(x)


def en(x, y, i, sn):
    return (x[-1] * y[i - 1] - x[0] * y[i]) / d(x) - sn * (x[-1] + y[0] - x[0] * y[-1]) / d(x)


def Wn(X, U, i, sn):
    R = np.matrix([[an(U[:, 0], i), 0], \
                   [cn(U[:, 0], U[:, 1], i, sn), sn]])
    T = np.matrix([[dn(U[:, 0], i)], \
                   [en(U[:, 0], U[:, 1], i, sn)]])
    tmp = R * np.matrix(X).T + T
    xp, yp = np.array(tmp.T)[0]
    return xp, yp


def T(U, X, deterministic=True):
    M = U.shape[0]
    N = X.shape[0]
    if M % 2 == 0:
        s = M - 1
    else:
        s = M - 2
    u, v = list(), list()
    for i in range(1, M):
        if deterministic:
            x_prime = np.median(X[(i - 1) * s:(i) * s, 0])
            y_prime = np.median(X[(i - 1) * s:(i) * s, 1])
        else:
            x_prime = random.choice(X[(i - 1) * s:(i) * s, 0])
            y_prime = random.choice(X[(i - 1) * s:(i) * s, 1])
        dU = U[i, 0] - U[i - 1, 0]
        dX = X[(i) * s, 0] - X[(i - 1) * s, 0]
        u.append(U[i - 1, 0] + dU * (x_prime - X[(i - 1) * s, 0]) / dX)
        v.append(y_prime)
    X = np.vstack((u, v)).T
    X = np.vstack((U, X))
    X = X[X[:, 0].argsort()]
    return X


def G(U, sn, balance=False):
    X = U.copy()
    x, y = list(X[:, 0]), list(X[:, 1])
    M = U.shape[0]
    N = X.shape[0]
    for i in range(N):
        for j in range(1, M):
            xp, yp = Wn(X[i], U, j, sn)
            x.append(xp)
            y.append(yp)
            if balance:
                xp, yp = Wn(X[i], U, j, -sn)
                x.append(xp)
                y.append(yp)
    x = np.array(x)
    y = np.array(y)
    X = np.vstack((x, y)).T
    X = X[X[:, 0].argsort()]
    null, indices = np.unique(X[:, 0], return_index=True)
    X = X[indices]
    return X


def FIF(U, sn, balance=False, deterministic=True):
    X = G(U, sn, balance)
    X = T(U, X, deterministic)
    return X
