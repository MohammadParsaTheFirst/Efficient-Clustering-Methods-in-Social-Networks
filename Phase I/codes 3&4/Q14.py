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
