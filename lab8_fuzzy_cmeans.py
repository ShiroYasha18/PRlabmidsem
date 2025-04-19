import numpy as np, skfuzzy as fuzz, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
X = X.T
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X, 3, 2, 0.005, 1000)
labels = np.argmax(u, axis=0)

for i in range(3):
    plt.scatter(X[0, labels==i], X[1, labels==i], label=f'Cluster {i}')
plt.scatter(cntr[:,0], cntr[:,1], c='black', marker='x', s=100, label='Centers')
plt.legend(), plt.title('Fuzzy C-Means'), plt.show()
