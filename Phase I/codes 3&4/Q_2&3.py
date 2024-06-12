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
