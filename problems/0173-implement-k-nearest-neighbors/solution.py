import numpy as np
from numpy.typing import NDArray

def k_points_lowest_distance(X: NDArray, query_point: NDArray, k: int) -> NDArray:
    """
    Tính kc euclid của query_point với từng point trước đó
    """
    X2 = np.sum(X*X, axis = 1)
    query_point2 = np.sum(query_point*query_point)
    distance = X2 + query_point2 - 2*(X @ query_point)
    res = np.argsort(distance)[:k]
    return res

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    query_point_arr = np.array(query_point)
    X = np.array(points).reshape(-1, len(points[0]))
    indx = k_points_lowest_distance(X, query_point_arr, k)
    return [points[i] for i in indx]
    