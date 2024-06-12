## 1:

def log_likelihood(F, A):
    
    
    B = F.dot(F.T)  #  F . F(Transpose)

    neighbouring_part = A*np.log(1.-np.exp(-1.*B)) # matrix multiplication
    sum_Neighbours = np.sum(neighbouring_part)
    notNeighbouring_part = (1-A)*B # matrix multiplication
    sum_notNeighbours = np.sum(notNeighbouring_part)
    
    
    log_likelihoodEstimation = sum_Neighbours - sum_notNeighbours 
    return log_likelihoodEstimation

def gradient(F, A, i):
    
    rowF, columnF = F.shape

    myNeighbours = np.where(A[i])
    notNeighbours = np.where(1-A[i])
    
    sumNeighbours = np.zeros((columnF,))
    for neighbor in myNeighbours[0]:
        B = F[neighbor].dot(F[i])
        sumNeighbours += F[neighbor]*(np.divide(np.exp(-1.*B),1.-np.exp(-1.*B)))


    sumNotNeighbour = np.zeros((columnF,))
    
    for NotNeighbor in notNeighbours[0]:
        sumNotNeighbour += F[NotNeighbor]

    Gradient = sumNeighbours - sumNotNeighbour
    return Gradient



def train(A, C, iterations = 100):
    # initialize an F
    N = A.shape[0]
    F = np.random.rand(N,C)

    for n in range(iterations):
        for person in range(N):
            grad = gradient(F, A, person)
            F[person] += 0.005*grad                  # updating F   
            F[person] = np.maximum(0.001, F[person]) # F should be nonnegative
        ll = log_likelihood(F, A)
        print('At step %4i logliklihood is %5.4f'%(n,ll))
        
    return F

import numpy as np
#testing in two small groups
A=np.random.rand(40,40)
A[0:15,0:25]=A[0:15,0:25]>1- 0.6 # connection prob people with 1 common group
A[0:15,25:40]=A[0:15,25:40]>1-0.1 # connection prob people with no common group
A[15:40,25:40]=A[15:40,25:40]>1-0.7 # connection prob people with 1 common group
A[15:25,15:25]=A[15:25,15:25]>1-0.8 # connection prob people with 2 common group
for i in range(40):
    A[i,i]=0
    for j in range(i):
        A[i,j]=A[j,i]

import matplotlib.pyplot as plt
import networkx as nx
plt.imshow(A)
delta=np.sqrt(-np.log(1-0.1)) # epsilon=0.1
F=train(A, 2, iterations = 120)
print(F>delta)
G=nx.from_numpy_matrix(A)
#G=nx.from_numpy_array(A)
C=F>delta # groups members
nx.draw(G,node_color=10*(C[:,0])+20*(C[:,1])) 




## 2 and 3:

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(14011111)


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


z0 = [3, 1, 2, 1, 3, 1, 2, 2, 2, 3, 3, 2, 1, 1, 3]

n = len(z0)
k = max(z0)

for k in range(10):
    A = matrixGeneration(z0, n)
    print("A"+str(k+1)+"=")
    print(A)

G = nx.from_numpy_matrix(A)

color_map = []
i = 0

for node in G:
    a = z0[i]
    if a == 1:
        color_map.append('blue')
    else:
        if a == 2:
            color_map.append('red')
        else:
            color_map.append('green')
    i += 1

nx.draw_networkx(G, node_color=color_map, node_size=60)
plt.show()





## 4:


z1 = list(map(int, input().split()))
z2 = list(map(int, input().split()))

n = len(z1)
output = 0

for i in range(n):
    if z1[i] != z2[i]:
        output += 1

print("dH = " + str(output))



## 5:


import math
from itertools import permutations


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


z1 = [2, 2, 1, 2, 1, 1]
z2 = [1, 1, 2, 1, 2, 2]

# z1 = list(map(int, input().split()))
# z2 = list(map(int, input().split()))

min_hamming = minimumHamming(z1, z2)
print(min_hamming)



## 6:


from math import log
import numpy as np


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


z = [3, 1, 2, 1, 3, 1, 2, 2, 2, 3, 3, 2, 1, 1, 3]
A = [[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.],
     [0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 1., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
     [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
     [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
     [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1.],
     [0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 1., 0., 0.],
     [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0.],
     [0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
     [1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1.],
     [0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
     [0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
     [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.]]


# z = list(map(int, input().split()))
# A = list(map(int, input().split()))

l_tilda = l_tilda(A, z)
print("l_tilda(z) = " + str(l_tilda))



## 7-11:

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



## 12:

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


## 13:


import matplotlib.pyplot as plt
import numpy
import numpy as np
from numpy import real
from numpy.linalg import eig

np.random.seed(14011111)


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
            if Z0[i] == Z0[j]:
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
    n = len(v_f)
    z = numpy.zeros([1, n])

    for i in range(n):
        try:
            if real(u_f[i][index]) > 0:
                z[0][i] = 1
            else:
                z[0][i] = 2
        except Exception:
            z[0][i] = 1
    return z


def error(n):
    e1 = 0
    e2 = 0

    if n < 120:
        N = 80
    else:
        N = 20

    for j in range(N):

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

        e1 += min(error1, n - error1)
        e2 += min(error2, n - error2)

    return [e1/N, e2/N]


n = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 250,
     260, 270, 280, 290, 300, 320, 330, 350, 370, 400, 1000]

errors = [[], []]
for i in n:
    errors[0].append(error(i)[0])
    errors[1].append(error(i)[1])

plt.plot(n, errors[0])
plt.plot(n, errors[1])
plt.show()



## 14:


import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.cluster import KMeans

california_housing = fetch_california_housing(as_frame=True)

MedInc = california_housing.frame["MedInc"]
x = california_housing.frame["Latitude"]
y = california_housing.frame["Longitude"]

n = len(MedInc)

V = []
for i in range(n):
    V.append([MedInc[i], 0])

kmeans = KMeans(n_clusters=3, random_state=1, n_init=50).fit(V)

colormap = []
for i in range(n):
    if(kmeans.labels_.astype(float)[i]==0):
        colormap.append('blue')
    elif(kmeans.labels_.astype(float)[i]==1):
        colormap.append('green')
    elif(kmeans.labels_.astype(float)[i]==2):
        colormap.append('red')

plt.scatter(x, y, color=colormap, s=3)
plt.show()



## 15:



import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numpy import real
from numpy.linalg import eig
from sklearn.cluster import KMeans


def L(f, N):
    Df = np.zeros((N, N))
    S = f.sum(axis=1)
    for i in range(N):
        Df[i][i] = S[i]
    Lf = Df - f
    return Lf


G = nx.karate_club_graph()
A = nx.adjacency_matrix(G).todense()

n = 34
K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for k in K:
    L_A = L(A, n)

    v, u = eig(L_A)
    u = real(u)
    v = real(v)

    u = u.T

    index = []
    v_sort = np.sort(v)
    for i in range(k):
        index.append(np.where(v == v_sort[i]))

    u_prim = []
    for i in range(k):
        u_prim.append(u[index[i]])

    u_1tok = np.asarray(u_prim)
    u_1tok = u_1tok.T[:, 0, :]

    kmeans = KMeans(n_clusters=k, random_state=1, n_init=50).fit(u_1tok)

    color_map = []
    for i in range(n):
        if kmeans.labels_.astype(float)[i] == 0:
            color_map.append('blue')
        elif kmeans.labels_.astype(float)[i] == 1:
            color_map.append('green')
        elif kmeans.labels_.astype(float)[i] == 2:
            color_map.append('red')
        else:
            color_map.append('yellow')

    nx.draw_networkx(G, node_color=color_map, node_size=60)
    plt.show()


## 17:

from scipy.stats import bernoulli, binom
import numpy as np

def generate_sample_bernoulli(n,p):
    mlist = a = [[0] * n] * n
    c2n = int(n*(n-1)/2)
    p = 0.0034
    c = 0
    X = bernoulli(p)
    X_samples = X.rvs(c2n)
    for i in range(0, n):
        for j in range(0, i):
            mlist[i][j] = X_samples[c]
            c += 1
            mlist[j][i] = mlist[i][j]
    return mlist
    
def generate_Adjacency_Bernoulli(n,p):
    mlist = np.zeros([n,n])
    for i in range(0,n):
        mlist[i,i] = 0
        for j in range(0,n):
            mlist[j,i] = np.random.binomial(1,p)
            mlist[j,i] = mlist[j,i]
    return mlist


n1 = 1000
m1 = 3000
p1 = 0.0034
list1 = []
h= 0
for N in range(0,10):
    Adj1 = generate_Adjacency_Bernoulli(n1,p1)
    list1.append(sum(sum(Adj1))/2)
    #for row in range(0, n1):
        #if(sum(row)>L)
        #h += Adj1[x][:].count(1)
    
list1


m = 3000
mean= sum(list1)/len(list1)
print("mean of all friendships = "+str(mean))
error = (mean-m)/mean
print("error with respect to m = "+str(error))

import matplotlib.pyplot as plt

ss = 4
list_ = [0,0,0,0]
data = {}
data[0]= 0
data[1]=0
data[2]=0
data[3]=0
for item in sum(Adj1):
    if item<ss:
        data[item] +=1
        #list_[item] +=1

#plt.hist(list_)
plt.bar(data.keys(),data.values())
plt.show()



## 18:


def Hamrang_finder_by_adjmat(adj_mat):
    counter = 0
    n = len(adj_mat[0])
    #L = (adj_mat).count(1)/n
    L = sum(sum(adj_mat))/n
    for row in adj_mat:
        #if adj_mat[x][:].count(1)> L:
        if (sum(row)>L):
            counter+=1
    return counter

list2 = []
n2 =1000
p2 = 0.00016
for N in range(0,10):
    adj_mat2 = generate_Adjacency_Bernoulli(n2,p2)
    list2.append(Hamrang_finder_by_adjmat(adj_mat2))
list2

xx = sum(list2)/len(list2)
print("the mean of Hamrang group among all People is = " + str(xx))


## 19:

import itertools
# from itertools import combination

def meshgrid(n):
    i =0 
    j =0 
    res = [i for j in range(0,n)]
    return res

def find_subsets(set_, n):
    res = list(itertools.combinations(set_,n))
    return res

def find_num_of_triagles(adj_mat):
    res = 0
    n = len(mat)
    indices = meshgrid(n)
    all_3sized_subsets = find_subsets(indices,3)
    
    
    for x,y,z in all_3sized_subsets:
        if ((mat[x,y] + mat[y,z] + mat[z,x] ==2)):
                counter += 1
    
    return res
    

def find_num_of_taragozari(adj_mat):
    res = 0
    n = len(mat)
    indexes = [i for i in range(n)]
    all_3sized_subsets = find_subsets(indexes,3)
    for x,y,z in all_3sized_subsets:
        if (mat[x,y] + mat[y,z] + mat[z,x]) == 3 :
            res += 1
    
    return res


# in order to save our computer we need to take a smaller n with respect to n=3000
n3 = 300
p3 = 0.01
adj_mat3 = generate_Adjacency_Bernoulli(n3,p3)
taragozari_num = find_num_of_taragozari(adj_mat3)
triagle_num = find_num_of_triagles(adj_mat3)
print("number of all trargozari : " + str(taragozari_num))
print("number of all triangles(chains) : "+str(triagle_num))

n3 = 100
p3 = 0.01
N = 5
taragozari = []
zangiri = []
for temp in range(0,N):
    adj_mat3 = generate_Adjacency_Bernoulli(n3,p3)
    zangiri.append(find_num_of_triagles(adj_mat3))
    taragozari.append(find_num_of_taragozari(adj_mat3))
taragozari

tara = sum(taragozari)/len(taragozari)
mosalas = sum(zangiri)/len(zangiri)
print("the mean of taragozari is : "+ str(tara))
print("the mean of chains is : "+ str(mosalas))




## 20:


def find_num_of_edges(adj_mat, vertics):
    s = 0
    for i in vertics:
        for j in vertics:
            if adj_mat[i][j] ==1 :
                s += 1
    return int(s/2)

n4 = 1000
p4 = 0.003
adj_mat4 = generate_Adjacency_Bernoulli(n4,p4) 
total_relations = 0
for i in range(n4):
    temp_neighbors = []
    for j in range(n4):
        if adj_mat4[i][j] == 1:
            temp_neighbors.append(j)
    total_relations += find_num_of_edges(adj_mat4,temp_neighbors)
yyy = total_relations/n4
print("Average friendships among the set of each person's friends = "+ str(yyy))


## 21:

import networkx as nx

n5 = 1000   # /////////////////////////////////////////////////////////// n5 = 1000
p5 = 0.0033
adj_mat5 = generate_Adjacency_Bernoulli(n5,p5)

my_graph = nx.path_graph(n5)
i = 0
total_path = 0
j =0 
# making the graph
while(i<n5):
    j = i
    if adj_mat5[i,j] ==1 :
        my_graph.add_edge(i,j)
        j +=1
    i+=1

# finding distances
for i in range(0,n5):
    for j in range(0,n5):
        total_path += nx.shortest_path_length(my_graph, i,j)
    
c2n = n5*(n5 -1 )/2 
result = total_path*2/c2n 
print(result)


## 22:

N = 100
n6 = 50
p6 = 0.34
longest_paths = []
for i in range(n6):
   
    adj_mat6 = generate_Adjacency_Bernoulli(n6,p6)
    my_gr = nx.path_graph(n6)
    i = 0
    while(i<n6):
        j = i 
        while(j<n6):
            if adj_mat6[i][j] ==1 :
                my_gr.add_edge(i,j)
            j +=1
        i+=1
temp_max =0 
for i in range(0,n6):
    for j in range(0,n6):
        if nx.shortest_path_length(my_gr,i,j)> temp_max:
            temp_max =  nx.shortest_path_length(my_gr,i,j)
    longest_paths.append(temp_max)
zzzz = sum(longest_paths)/len(longest_paths)
print("The mean of longest path between two arbitrary nodes is = "+ str(zzzz))


## 23:

chains_num = []
data_input = [i*10 + 10 for i in range(19)]
steps = (200 - 10)//10
N = 100   # ///////////////////////////////////////////////   N =10
p7 = p6
LongestPaths = []

for z in range(steps):
    for _ in range(N):
        n7 = z*10 + 10
        adj_mat = generate_Adjacency_Bernoulli(n7,p7)
        graph = nx.path_graph(n7)

        i = 0
        while (i < n):
            j = i
            while (j < n):
                if (adj_mat[i,j] == 1):
                    graph.add_edge(i, j)
                j = j+1
            i = i+1

        temp_max=0
        for i in range(n):
            for j in range(n):
                if( nx.shortest_path_length(graph, i, j)>temp_max):
                    temp_max = nx.shortest_path_length(graph, i, j)
        LongestPaths.append(temp_max)

    chains_num.append(sum(LongestPaths)/len(LongestPaths))
    
plt.figure()
plt.bar(data_input,chains_num)




## 24:


def Triangle(n,p):
    res = 0
    for N in range(100):
        adj_mat = generate_Adjacency_Bernoulli(n,p)
        res += cal_number_of_taragozari(adj_mat)
    return res/100


n8= 100
p8 = 0.34
y = 0
x = Triangle(n8,p8)
print("number of triangles = "+str(z))


## 25:

def Triangle(n,p):
    res = 0
    for N in range(100):
        adj_mat = generate_Adjacency_Bernoulli(n,p)
        res += cal_number_of_taragozari(adj_mat)
    return res/100

def get_number_of_taragozari(adj_mat):
    n = len(adj_mat)
    res = 0
    indices = [i for i in range(n)]
    all_3subset = find_subsets(indices,3)
    for x,y,z in all_3subset:
        if mat[x,y]==1 and mat[y,z]==1 and  mat[z,x]==1 :
            res += 1
    return res


y = []
x = [i*10 + 10 for i in range(10)]
steps = 10

for t in range(steps):
    n = t*10 + 10
    p = 60/(n*n)
    y.append(Triangle(n,p))
    
plt.scatter(x,y)
plt.xlabel("n")
plt.ylabel("Triangles with respect to n");
plt.show()



## 26:


import matplotlib.pyplot as plt

y2 = []
x2 = [i*10 + 10 for i in range(10)]
steps = 10
p = 0.34
for t in range(steps):
    n = t*10 + 10
    y2.append(Triangle(n,p))
    
plt.scatter(x2,y2)
plt.xlabel("n",)
plt.ylabel("number of triangles");



## 27:

y3 = []
x3 = [i*50 + 50 for I in range(20)]
steps = 20
for t in range(steps):
    n = 50*t
y3 = []
x3 = [i*50 + 50 for I in range(20)]
steps = 20
for t in range(steps):
    n = 50*t +50
    p = 1/n
    y.append(Triangle(n,p))
plt.scatter(x,y)
plt.xlabel("n")
plt.ylabel("number of triangles with respect to n")
Xcumulate = np.cumsum(x, axis =0)
Ycumulate = np.cumsum(y, axis =0)
Nn = len( Ycumulate)
for i in range(Nn):
    Ycumulate[i] = Ycumulate[i]/Xcumulate[i]

plt.scatter(Xcumulate,Ycumulate)
plt.xlabel("n")
plt.ylabel("cumulative mean with respect to n")


