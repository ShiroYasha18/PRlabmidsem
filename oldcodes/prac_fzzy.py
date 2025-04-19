import numpy as np
from sklearn.datasets import load_iris
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from pracfuzzy import cluster_labels

iris=load_iris()

data= iris.data[:,:2]
n=3
cntr,u,_,_,_,_,_= fuzz.cluster.cmeans(data.T,c=3,m=2,error=0.01,maxiter=1000)
cluster_labels= np.argmax(u,axis=0)
plt.figure(figsize=(8,6))
for i in range(n):
    plt.scatter(data[cluster_labels==i,0],data[cluster_labels==i,1],label=f"Clusters{i}")
plt.scatter(cntr[:,0],cntr[:,1],s=200,marker="X",label="centroids")

plt.title("practiscing")
plt.legend()
plt.show()