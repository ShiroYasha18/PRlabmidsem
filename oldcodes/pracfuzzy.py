import numpy as np
import skfuzzy as fuzz
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
wine= load_wine()
data =wine.data[:,:2]
n_clusters=3
cntr,u,_,_,_,_,_ = fuzz.cluster.cmeans(data.T ,c=3,m=2, error=0.01,maxiter=1000)
cluster_labels= np.argmax(u,axis=0)
plt.figure(figsize=(8,8))
for i in range(n_clusters):
    plt.scatter(data[cluster_labels==i,0],data[cluster_labels==i,1],label=f"Cluster{i+1}")
plt.scatter(cntr[:,0],cntr[:,1],label="centroids",marker="x",s=200)
plt.title("new graph")
plt.legend()
plt.show()

