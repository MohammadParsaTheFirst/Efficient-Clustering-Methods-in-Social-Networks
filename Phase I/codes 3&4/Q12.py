import numpy
import numpy as np
from numpy import real
from numpy.linalg import eig


def matrixGeneration(Z0, N):
    p = 0.1
    q = 0.01
    Q = np.eye(N) * (p - q) + np.ones(N) * q
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            a = Z0[i] - 1
            b = Z0[j] - 1
            A[i, j] = int(np.random.binomial(size=1, n=1, p=Q[a, b]))
            A[j, i] = A[i, j]
    return A


def wGeneration(Z0, N):
    p = 0.1
    q = 0.01
    w = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            if z0[i] == Z0[j]:
                w[i][j] = p
            else:
                w[i][j] = q
            w[j][i] = w[i][j]
    return w


def L(f, N):
    Df = np.zeros((N, N))
    S = f.sum(axis=1)
    for i in range(N):
        Df[i][i] = S[i]
    Lf = Df - f
    return Lf


def Grouping(v_f, u_f):
    index = numpy.where(v_f == np.sort(v_f)[1])
    u = u_f[:][index]
    n = len(v_f)
    z = numpy.zeros([1, n])

    for i in range(n):
        # print("{:.3f}".format(100*real(u[0][i])))
        try:
            if real(u_f[i][index]) > 0:
                z[0][i] = 1
            else:
                z[0][i] = 2
        except Exception:
            z[0][i] = 1
    return z


n = 10000
z0 = []

for i in range(n):
    z0.append(int(1 + np.random.binomial(size=1, n=1, p=0.5)))

A = matrixGeneration(z0, n)
W = wGeneration(z0, n)

LW = L(W, n)
LA = L(A, n)

v_W, u_W = eig(LW)
v_A, u_A = eig(LA)

z1 = Grouping(v_W, u_W)
z2 = Grouping(v_A, u_A)

error1 = 0
for i in range(n):
    if z0[i] == z1[0][i]:
        error1 += 1

error2 = 0
for i in range(n):
    if z0[i] == z2[0][i]:
        error2 += 1

print("error of estimate with W = " + str(min(error1, n - error1)))
print("error of estimate with A = " + str(min(error2, n - error2)))
