import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Load image data
digits = load_digits()
X, y = digits.data, digits.target

# Apply KMeans clustering
kmeans = KMeans(n_clusters=10, n_init=10)
labels = kmeans.fit_predict(X)

# Map cluster labels to true labels (using mode)
mapped_labels = np.zeros_like(labels)
for i in range(10):
    mask = labels == i
    mapped_labels[mask] = mode(y[mask])[0]

# Evaluate clustering accuracy
acc = accuracy_score(y, mapped_labels)
print("K-Means Clustering Accuracy:", acc)
