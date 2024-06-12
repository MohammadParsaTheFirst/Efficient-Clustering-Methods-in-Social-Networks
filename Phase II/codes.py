
#1:

import networkx as nx
from matplotlib import pyplot as plt

G = nx.newman_watts_strogatz_graph(100, 4, 1)

nx.draw_networkx(G, node_size=1, with_labels=0)
plt.show()

#2:

import networkx as nx

sequence = [100]*20000
G = nx.configuration_model(sequence)
g = nx.Graph(G)
g.remove_edges_from(nx.selfloop_edges(g))

print("probability :")
print(nx.number_of_edges(g)/nx.number_of_edges(G))






#5:

import numpy as np

d = 400
n = 20000
p = d/n
N = 100

m = 0
for i in range(N):
    x = np.random.binomial(1, p, size=[n, n])
    m += (np.sum(x)+np.trace(x))/2

print("Expected value:  m = " + str(d*n/2) + "   |d|= " + str(d*n))
print("Avrage number:  m = " + str(m/N) + "   |d|= " + str(2*m/N))





#9:
import networkx as nx
import community
#from communities.algorithms import louvain_method
import matplotlib.cm as cm
import matplotlib.pyplot as plt

Graph = nx.karate_club_graph() #  karate club graph
partition = community.best_partition(G)
pos = nx.spring_layout(Graph)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(Graph, pos, partition.keys(), node_size=60, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(Graph, pos, alpha=0.6)
plt.show()