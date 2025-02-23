import numpy as np
from scipy.spatial.distance import mahalanobis

def mahalanobis_distance(poly1, poly2):
    # Convert to NumPy arrays
    poly1, poly2 = np.array(poly1), np.array(poly2)

    # Compute centroids
    centroid1, centroid2 = np.mean(poly1, axis=0), np.mean(poly2, axis=0)

    # Stack all points and compute covariance matrix
    cov_matrix = np.cov(np.vstack((poly1, poly2)).T)

    # Compute inverse covariance matrix (with regularization)
    cov_inv = np.linalg.pinv(cov_matrix)  # Use pseudo-inverse to avoid singularity

    # Compute Mahalanobis distance
    return mahalanobis(centroid1, centroid2, cov_inv)

# **Test with two distinct polygons**
polygon1 = [(1, 2), (2, 4), (3, 3)]  # Small triangle
polygon2 = [(10, 15), (15, 40), (20, 10)]  # Large quadrilateral

print("Mahalanobis Distance:", mahalanobis_distance(polygon1, polygon2))
