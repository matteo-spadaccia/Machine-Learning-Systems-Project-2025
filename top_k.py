import cupy as cp
from distance_functions import l2_distance

def top_k_nearest(A, X, K=5, metric="l2"):
    if metric == "l2":
        distances = cp.array([l2_distance(X, A[i]) for i in range(A.shape[0])])
    elif metric == "cosine":
        distances = cp.array([cosine_distance(X, A[i]) for i in range(A.shape[0])])
    elif metric == "dot":
        distances = cp.array([dot_product(X, A[i]) for i in range(A.shape[0])])
    elif metric == "manhattan":
        distances = cp.array([manhattan_distance(X, A[i]) for i in range(A.shape[0])])
    else:
        raise ValueError("Invalid distance metric")

    return cp.argsort(distances)[:K]  # Get indices of K smallest distances
