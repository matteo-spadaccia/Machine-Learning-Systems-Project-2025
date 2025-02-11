import cupy as cp
from top_k import top_k_nearest

# Create dataset
N, D = 10000, 128
A = cp.random.randn(N, D)
X = cp.random.randn(D)

K = 5
nearest_indices = top_k_nearest(A, X, K=K, metric="l2")

print(f"Top {K} Nearest Indices (L2 Distance):", nearest_indices)
