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



