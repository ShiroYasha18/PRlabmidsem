import numpy as np
from scipy.spatial import distance

A = np.array([[1,2],[2,3],[3,4]])
B = np.array([[5,6],[6,7],[7,8]])
meanA = np.mean(A, axis=0)
meanB = np.mean(B, axis=0)
cov = np.cov(np.vstack([A,B]).T)
epsilon = 1e-6
cov += np.eye(cov.shape[0]) * epsilon  # Regularization
inv_cov = np.linalg.inv(cov)
d = distance.mahalanobis(meanA, meanB, inv_cov)
print("Mahalanobis Distance:", d)