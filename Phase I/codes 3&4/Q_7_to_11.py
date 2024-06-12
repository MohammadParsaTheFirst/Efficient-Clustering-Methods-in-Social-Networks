import math
from math import log
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt

np.random.seed(140111)


def minimumHamming(z1, z2):
    n = len(z1)
    k = max(z1)

    hamming = []

    Permutations = list(permutations(range(1, k + 1)))

    for i in range(math.factorial(k)):
        output = 0
        for j in range(n):
            index = Permutations[0].index(z1[j])
            if Permutations[i][index] != z2[j]:
                output += 1
        hamming.append(output)

    min_hamming = min(hamming)
    return min_hamming


def calculate(matrix, Z, Q):
    l_z = 0
    n = len(Z)
    for i in range(n):
        for j in range(i + 1, n):
            a = Z[i] - 1
            b = Z[j] - 1
            q = Q[a, b]
            l_z += log(1 + 2 * matrix[i][j] * q - q - matrix[i][j])
    return l_z


def l_tilda(matrix, Z):
    p = 0.6
    q = 0.1
    n = len(Z)
    Q = np.eye(n) * (p - q) + np.ones(n) * q

    l_z = calculate(matrix, Z, Q)

    lTilda = -l_z
    return lTilda


def matrixGeneration(Z0, N):
    p = 0.6
    q = 0.1
    Q = np.eye(N) * (p - q) + np.ones(N) * q
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            a = Z0[i]-1
            b = Z0[j]-1
            A[i, j] = int(np.random.binomial(size=1, n=1, p=Q[a, b]))
            A[j, i] = A[i, j]
    return A


def estimate(A, x0):

    T = 10

    l_min = l_tilda(A, x0)
    x_t = x0.copy()

    X_T = [x0]
    lTilda = [l_min]
    d = [minimumHamming(x0, z0)]

    for t in range(T):
        x_t_1 = x_t.copy()
        for i in range(n):
            for j in range(i + 1, n):

                x = x_t_1.copy()
                x[i] = x_t_1[j]
                x[j] = x_t_1[i]

                l_tilda_ij_t = l_tilda(A, x)

                if l_tilda_ij_t < l_min:
                    x_t = x.copy()
                    l_min = l_tilda_ij_t

        lTilda.append(l_min)
        d.append(minimumHamming(z0, x_t))
        X_T.append(x_t)
    return [lTilda, d, X_T]


z0 = [3, 1, 2, 1, 3, 1, 2, 2, 2, 3, 3, 2, 1, 1, 3]
n = len(z0)

A = matrixGeneration(z0, n)

x0 = [3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]

estimate0 = estimate(A, x0)

d = estimate0[1]
lTilda = estimate0[0]

# Question 7
plt.figure("L Tilda")
plt.axhline(y=l_tilda(A, z0), color='r')
plt.plot(lTilda)
plt.xlabel("L Tilda")

plt.figure("Hamming d")
plt.axhline(y=minimumHamming(z0, x0), color='r')
plt.xlabel("Hamming distance")

plt.plot(d)

plt.show()

# Question 8
x0 = [[3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
        [3, 2, 1, 3, 2, 1, 3, 2, 3, 2, 1, 2, 1, 3, 1],
        [3, 2, 1, 3, 2, 1, 2, 1, 2, 3, 3, 3, 1, 1, 2],
        [1, 3, 2, 3, 2, 1, 3, 1, 2, 1, 3, 2, 3, 2, 1],
        [3, 1, 3, 3, 2, 2, 1, 2, 2, 3, 1, 1, 3, 2, 1],
        [1, 3, 3, 2, 2, 3, 2, 2, 1, 3, 3, 1, 2, 1, 1],
        [1, 2, 3, 1, 3, 2, 3, 2, 2, 3, 1, 3, 1, 2, 1],
        [3, 1, 3, 2, 1, 3, 2, 2, 3, 2, 1, 3, 1, 2, 1],
        [2, 3, 2, 3, 1, 3, 1, 3, 2, 3, 2, 2, 1, 1, 1],
        [1, 1, 3, 2, 3, 3, 1, 3, 2, 1, 2, 3, 2, 1, 2],
        [2, 3, 3, 3, 3, 2, 1, 2, 2, 2, 3, 1, 1, 1, 1]]

estimates = []

for i in range(11):
    estimate_i = estimate(A, x0[i])
    estimates.append(estimate_i)

    d = estimate_i[1]
    lTilda = estimate_i[0]
    plt.figure("Hamming d")
    plt.plot(d)
    plt.figure("L Tilda")
    plt.plot(lTilda)

plt.figure("Hamming d")
plt.axhline(y=minimumHamming(z0, x0[0]), color='r')
plt.xlabel("Hamming distance")

plt.figure("L Tilda")
plt.axhline(y=l_tilda(A, z0), color='r')
plt.xlabel("L Tilda")

plt.show()

# Question 9
print("Question 9:")
for i in range(11):
    if estimates[i][0][10] == l_tilda(A, z0):
        print("z0_hat = " + str(estimates[i][2][0]))
        print("Hamming distance = " + str(estimates[i][1][10])+"\n")

# Question 10
for i in range(11):
    if estimates[i][1][10] == 0:
        print("\nQuestion 10:")
        print(estimates[i][2][10])
        break

# Question 11
print("\n\nQuestion 11:")
A1 = matrixGeneration(z0, n)
A2 = matrixGeneration(z0, n)
A3 = matrixGeneration(z0, n)

z_T_1 = []
d_T_1 = []
z_T_2 = []
d_T_2 = []
z_T_3 = []
d_T_3 = []


for i in range(11):
    z_T_1.append(estimate(A1, x0[i])[2][10])
    d_T_1.append(estimate(A1, x0[i])[1][10])
    z_T_2.append(estimate(A2, x0[i])[2][10])
    d_T_2.append(estimate(A2, x0[i])[1][10])
    z_T_3.append(estimate(A3, x0[i])[2][10])
    d_T_3.append(estimate(A3, x0[i])[1][10])

print("\nbest estimate for first matrix:\n" + "     zHat(T=10) = " + str(z_T_1[d_T_1.index(min(d_T_1))]) +
      "     d(z0, zHat_T) = " + str(min(d_T_1)))

print("\nbest estimate for second matrix:\n" + "    zHat(T=10) = " + str(z_T_2[d_T_2.index(min(d_T_2))]) +
      "     d(z0, zHat_T) = " + str(min(d_T_2)))

print("\nbest estimate for third matrix:\n" + "     zHat(T=10) = " + str(z_T_3[d_T_3.index(min(d_T_3))]) +
      "     d(z0, zHat_T) = " + str(min(d_T_3)))
