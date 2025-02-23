# Import necessary libraries
from fcmeans import FCM
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Feature matrix

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and fit Fuzzy C-Means
n_clusters = 3  # Setting the number of clusters
fcm = FCM(n_clusters=n_clusters, m=2.0, max_iter=150, error=1e-5, random_state=42)
fcm.fit(X_scaled)

# Get cluster centers and labels
centers = fcm.centers
labels = fcm.predict(X_scaled)

# Visualizing the first two dimensions
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(X_scaled[labels == i, 0], X_scaled[labels == i, 1], label=f'Cluster {i+1}')

# Plot cluster centers
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, color='black', label='Centers')

# Labels and title
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Fuzzy C-Means Clustering on Iris Dataset")
plt.show()
