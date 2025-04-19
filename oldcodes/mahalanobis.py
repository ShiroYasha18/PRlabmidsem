import numpy as np
import scipy.spatial.distance as dist


def mahalanobis_distance(poly1, poly2, epsilon=1e-6):
    """
    Compute Mahalanobis Distance between two polygons.

    :param poly1: List of points (x, y) for first polygon
    :param poly2: List of points (x, y) for second polygon
    :param epsilon: Small value to stabilize matrix inversion
    :return: Mahalanobis distance
    """
    poly1, poly2 = np.array(poly1), np.array(poly2)

    # Compute centroids of both polygons
    centroid1 = np.mean(poly1, axis=0)
    centroid2 = np.mean(poly2, axis=0)

    # Stack all points together to compute covariance matrix
    combined_points = np.vstack([poly1, poly2])
    cov_matrix = np.cov(combined_points.T)  # Covariance matrix

    # Regularization: Add small value to diagonal to avoid singular matrix
    cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon

    # Compute inverse of covariance matrix
    cov_inv = np.linalg.inv(cov_matrix)

    # Compute Mahalanobis distance
    return dist.mahalanobis(centroid1, centroid2, cov_inv)


# Example polygons (list of (x, y) coordinates)
polygon1 = [(1, 2), (3, 4), (5, 6)]
polygon2 = [(2, 3), (4, 5), (6, 7)]

# Compute distance
distance = mahalanobis_distance(polygon1, polygon2)
print("Mahalanobis Distance:", distance)
