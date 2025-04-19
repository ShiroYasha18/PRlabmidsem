import numpy as np
from scipy.cluster.hierarchy import centroid
from scipy.spatial.distance import mahalanobis

from mahalanobis import polygon1, polygon2


def find_maha(polygon1,polygon2,epsilion=1e-6):
    p1=np.array(polygon1)
    p2=np.array(polygon2)

    c1 =np.mean(p1,axis=0)
    c2 =np.mean(p2,axis=0)

    combined_points= np.vstack([p1,p2])
    conv_matrix= np.cov(combined_points.T)

    conv_matrix+= np.eye(conv_matrix.shape[0]) *(epsilion)

    cov_inv=np.linalg.inv(conv_matrix)


    return mahalanobis(c1,c2,cov_inv)

polygon1 = [(1,2),(2,3),(3,2),(4,5)]
polygon2 = [ (4,3), (3,4), (5,6), (7,8)]
result = find_maha(polygon1, polygon2)
print("Mahalanobis Distance:", result)

