import time
import torch
import numpy as np
from task import our_kmeans
from test import testdata_kmeans

# ------------------------------------------------------------------------------------------------
# Benchmarking KMeans with different distance functions
# ------------------------------------------------------------------------------------------------

print("\n\nBENCHMARKING KMEANS WITH DIFFERENT DISTANCES...")

# Initializing data
N, D, A_np, K = testdata_kmeans("")
maxITERS = 100
distance_types = ['cos', 'L2', 'dot', 'L1']
results = {}
print(f" (clusters# = {K:>4})")
print(f" (max iters = {maxITERS:>4})")

# Preparing to benchmark each distance method for KMeans
def benchmark_kmeans(N, D, A, K, distance_type, num_trials=1):
    A_torch = torch.tensor(A, dtype=torch.float32).to('cuda')
    times = []
    for _ in range(num_trials):
        start = time.time()
        R = our_kmeans(N, D, A_torch, K, distance_type=distance_type)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / len(times)
    return avg_time, R

# Computing results
for dist_type in distance_types:
    print(f"\nWITH {dist_type} DISTANCE:")
    avg_time, R = benchmark_kmeans(N, D, A_np, K, dist_type)
    results[dist_type] = {'time': avg_time, 'clusters': R}
    print(f"       Time = {avg_time:.4f}s")
print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n\n")