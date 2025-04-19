import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn import datasets

# 1️⃣ Load dataset (Iris dataset from sklearn)
iris = datasets.load_wine()
data = iris.data[:, :2]  # Taking only 2 features for easy visualization


# 3️⃣ Define number of clusters
n_clusters = 3

# 4️⃣ Apply Fuzzy C-Means clustering
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    data.T, c=n_clusters, m=2, error=0.005, maxiter=1000
)

# 5️⃣ Assign clusters (each point is assigned to the cluster with the highest membership value)
cluster_labels = np.argmax(u, axis=0)

# 6️⃣ Plot the clustered data
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1], label=f"Cluster {i+1}")

# 7️⃣ Plot centroids
plt.scatter(cntr[:, 0], cntr[:, 1], marker="x", color="red", s=200, label="Centroids")

plt.title("Fuzzy C-Means Clustering on Iris Dataset")
plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.legend()
plt.show()
