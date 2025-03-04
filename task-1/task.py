import torch
import cupy as cp
import triton
import triton.language as tl
import numpy as np
import time
import json
import random
from test import testdata_kmeans, testdata_knn, testdata_ann


# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# DISTANCE-COMPUTING FUNCTIONS
# Each distance formula is implemented through torch (default), cupy and torch for multi-dimentional vectors.

# Cosine distance: d(X, Y) = 1 - (X ⋅ Y) / (|X| |Y|)
def cos_dist(X, Y):
    return 1 - (torch.dot(X, Y) / (torch.norm(X) * torch.norm(Y)))
def cos_dist_cupy(X, Y):
    return 1 - (cp.dot(X, Y) / (cp.linalg.norm(X) * cp.linalg.norm(Y)))
def cos_dist_multidim(X, Y):
    return 1 - (torch.sum(X * Y, dim=-1) / (torch.norm(X, dim=-1) * torch.norm(Y, dim=-1)))

# L2 (Euclidean) distance: d(X, Y) = sqrt(sum((X_i - Y_i)^2))
def L2_dist(X, Y):
    return torch.sqrt(torch.sum((X - Y) ** 2))
def L2_dist_cupy(X, Y):
    return cp.sqrt(cp.sum((X - Y) ** 2))
def L2_dist_multidim(X, Y):
    return torch.sqrt(torch.sum((X - Y) ** 2, dim=-1))

# Dot Product distance: d(X, Y) = X ⋅ Y
def dot_dist(X, Y):
    return torch.dot(X, Y)
def dot_dist_cupy(X, Y):
    return cp.dot(X, Y)
def dot_dist_multidim(X, Y):
    return torch.sum(X * Y, dim=-1)

# L1 (Manhattan) distance: d(X, Y) = sum(|X_i - Y_i|)
def L1_dist(X, Y):
    return torch.sum(torch.abs(X - Y))
def L1_dist_cupy(X, Y):
    return cp.sum(cp.abs(X - Y))
def L1_dist_multidim(X, Y):
    return torch.sum(torch.abs(X - Y), dim=-1)


# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

def our_knn(N, D, A, X, K, distance_metric="L2", batch_size=100000):
    """
    Compute Top-K nearest vectors in A for query X using GPU.

    Parameters:
    - N (int): Number of vectors
    - D (int): Dimension of vectors
    - A (torch.Tensor): Dataset of vectors (stored in GPU)
    - X (torch.Tensor): Query vector (stored in GPU)
    - K (int): Number of nearest neighbors to find
    - distance_metric (str): Distance metric ("L2", "cosine", "dot", "L1")
    - batch_size (int): Batch size for processing large datasets

    Returns:
    - top_k_indices (torch.Tensor): Indices of the K nearest vectors
    - top_k_distances (torch.Tensor): Corresponding distances
    """
    torch.cuda.synchronize()
    start_time = time.time()

    A = A.to(torch.float16)  # Reduce precision for memory savings
    X = X.to(torch.float16)

    num_batches = (N + batch_size - 1) // batch_size  # Compute number of batches
    all_indices = []
    all_distances = []

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, N)
        batch = A[batch_start:batch_end]  # Get batch
        
        # Compute distances
        if distance_metric == "L2":
            dists = torch.sqrt(torch.sum((batch - X) ** 2, dim=-1))
        elif distance_metric == "cosine":
            dists = 1 - torch.sum(batch * X, dim=-1) / (torch.norm(batch, dim=-1) * torch.norm(X))
        elif distance_metric == "dot":
            dists = torch.sum(batch * X, dim=-1)
        elif distance_metric == "L1":
            dists = torch.sum(torch.abs(batch - X), dim=-1)
        else:
            raise ValueError("Unsupported distance metric!")

        # Get Top-K indices for current batch
        batch_top_k_distances, batch_top_k_indices = torch.topk(dists, K, largest=False)
        batch_top_k_indices += batch_start  # Adjust index offset

        all_indices.append(batch_top_k_indices)
        all_distances.append(batch_top_k_distances)

    # Merge results from all batches
    top_k_distances = torch.cat(all_distances)
    top_k_indices = torch.cat(all_indices)

    # Get final Top-K from merged results
    final_top_k_distances, final_top_k_indices = torch.topk(top_k_distances, K, largest=False)
    final_top_k_indices = top_k_indices[final_top_k_indices]

    torch.cuda.synchronize()
    end_time = time.time()

    print(f"⏱️ Time taken: {end_time - start_time:.4f} seconds")
    
    return final_top_k_indices, final_top_k_distances
        

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

def our_kmeans(N, D, A, K, distance_type='cos', max_iters=100, tol=1e-4):
    distance_funcs = {'cos': cos_dist,'L2': L2_dist,'dot': dot_dist,'L1': L1_dist}
    if distance_type not in distance_funcs: raise ValueError(f"Unsupported distance type: {distance_type}")

    dist_func = distance_funcs[distance_type]

    # Initializing centroids and cluster assignments
    indices = random.sample(range(N), K)
    centroids = A[indices]
    R = torch.zeros(N, dtype=torch.long, device='cuda')

    for iteration in range(max_iters):
        prev_centroids = centroids.clone()
        for i in range(N): # Assignment
            distances = torch.tensor([dist_func(A[i], centroid) for centroid in centroids], device='cuda')
            R[i] = torch.argmin(distances)
        for k in range(K): # Update
            assigned_points = A[R == k]
            if assigned_points.shape[0] > 0:
                centroids[k] = assigned_points.mean(dim=0)
        centroid_shifts = torch.norm(centroids - prev_centroids, dim=1)
        if torch.max(centroid_shifts) < tol: # Convergence check
            break
    print(f" Iterations = {iteration+1}")

    return R.cpu()


# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

def our_ann(N, D, A, X, K):
    pass


# ------------------------------------------------------------------------------------------------
# Testing and benchmarking the functions
# ------------------------------------------------------------------------------------------------

def test_distances(D1 = 10000, D2 = 128, dimensions = [2, 32768, 1000000], num_trials = 1000):
    """
    Testing and benchmarking distances functions
    """

    # ------------------------------------------------------------------------------------------------
    # Testing distances computation functions
    # ------------------------------------------------------------------------------------------------
    print("\n\nTESTING DISTANCE FUNCTIONS...")
    if torch.cuda.is_available():

        print("\nWITH PyTorch:")
        X_torch = torch.randn(D1, device="cuda")
        Y_torch = torch.randn(D1, device="cuda")
        print("        Cosine Distance =", cos_dist(X_torch, Y_torch).item())
        print("L2 (Euclidean) Distance =", L2_dist(X_torch, Y_torch).item())
        print("   Dot Product Distance =", dot_dist(X_torch, Y_torch).item())
        print("L1 (Manhattan) Distance =", L1_dist(X_torch, Y_torch).item())

        print("\nWITH Cupy:")
        X_cupy = cp.random.randn(D2)
        Y_cupy = cp.random.randn(D2)
        print("        Cosine Distance =", cos_dist_cupy(X_cupy, Y_cupy).item())
        print("L2 (Euclidean) Distance =", L2_dist_cupy(X_cupy, Y_cupy).item())
        print("   Dot Product Distance =", dot_dist_cupy(X_cupy, Y_cupy).item())
        print("L1 (Manhattan) Distance =", L1_dist_cupy(X_cupy, Y_cupy).item())

        print("\nWITH PyTorch AND multi-dimensional vectors:")
        X_multidim = torch.randn((D1,D2), device="cuda")
        Y_multidim = torch.randn((D1,D2), device="cuda")
        print("        Cosine Distance =", cos_dist_multidim(X_multidim, Y_multidim).cpu().numpy())
        print("L2 (Euclidean) Distance =", L2_dist_multidim(X_multidim, Y_multidim).cpu().numpy())
        print("   Dot Product Distance =", dot_dist_multidim(X_multidim, Y_multidim).cpu().numpy())
        print("L1 (Manhattan) Distance =", L1_dist_multidim(X_multidim, Y_multidim).cpu().numpy())

    else:
        print("Error: cuda unavailable!")
    
    print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n")

    # ------------------------------------------------------------------------------------------------
    # Benchmarking distances computation (excluding data transfer times)
    # ------------------------------------------------------------------------------------------------
    print("\nBENCHMARKING DISTANCE FUNCTIONS...")

    def benchmark_distance(func_cpu, func_gpu1, func_gpu2, D, num_trials=num_trials):

        # Generating random vectors
        X_np = np.random.randn(D)
        Y_np = np.random.randn(D)

        # Measuring CPU time
        X_cpu_torch = torch.tensor(X_np, dtype=torch.float32) # PyTorch tensors (for CPU computation)
        Y_cpu_torch = torch.tensor(Y_np, dtype=torch.float32)
        start = time.time()
        for _ in range(num_trials):
            func_cpu(X_cpu_torch, Y_cpu_torch)
        end = time.time()
        cpu_time = (end - start) / num_trials # average time per run

        # Measuring GPU time with method 1
        X_gpu_cupy = cp.array(X_np) # CuPy arrays (for GPU computation)
        Y_gpu_cupy = cp.array(Y_np)
        for _ in range(10): # warming-up GPU
            func_gpu1(X_gpu_cupy, Y_gpu_cupy)
        cp.cuda.Device(0).synchronize()
        start = time.time()
        for _ in range(num_trials):
            func_gpu1(X_gpu_cupy, Y_gpu_cupy)
        cp.cuda.Device(0).synchronize()
        end = time.time()
        gpu1_time = (end - start) / num_trials # average time per run

        # Measuring GPU time with method 2
        X_gpu_torch = torch.tensor(X_np, dtype=torch.float32, device="cuda") # PyTorch tensors (for GPU computation)
        Y_gpu_torch = torch.tensor(Y_np, dtype=torch.float32, device="cuda")
        for _ in range(10): # warming-up GPU
            func_gpu2(X_gpu_torch, Y_gpu_torch)
        cp.cuda.Device(0).synchronize()
        start = time.time()
        for _ in range(num_trials):
            func_gpu2(X_gpu_torch, Y_gpu_torch)
        cp.cuda.Device(0).synchronize()
        end = time.time()
        gpu2_time = (end - start) / num_trials # average time per run

        return cpu_time, gpu1_time, gpu2_time

    for D in dimensions:
        print(f"\nWITH D = {D}:")
        
        methods = ["CPU", "CuPy-GPU", "PyTorch-GPU"]
        for name, func_cpu, func_gpu1, func_gpu2 in [
            ("        Cosine Distance", cos_dist, cos_dist_cupy, cos_dist),
            ("L2 (Euclidean) Distance", L2_dist, L2_dist_cupy, L2_dist),
            ("   Dot Product Distance",  dot_dist, dot_dist_cupy, dot_dist),
            ("L1 (Manhattan) Distance", L1_dist, L1_dist_cupy, L1_dist),
        ]:
            cpu_time, gpu1_time, gpu2_time = benchmark_distance(func_cpu, func_gpu1, func_gpu2, D, num_trials)
            speedup1 = cpu_time / gpu1_time if gpu1_time > 0 else float("inf")
            speedup2 = cpu_time / gpu2_time if gpu2_time > 0 else float("inf")

            print(f"{name} -> {methods[0]} {cpu_time*1e6:>8.2f} µs  |  {methods[1]} {gpu1_time*1e6:>8.2f} µs ({speedup1:>5.2f}x)  |  {methods[2]} {gpu2_time*1e6:>8.2f} µs ({speedup2:>5.2f}x)")

def test_kmeans(maxITERS = 100):
    """
    Benchmarking KMeans with different distance functions
    """
    print("\n\nBENCHMARKING KMEANS WITH DIFFERENT DISTANCES...")

    N, D, A_np, K = testdata_kmeans("")
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

def test_knn():
    """
    Benchmarking KNN with different distance functions
    """
    print("\n\nBENCHMARKING KNN WITH DIFFERENT DISTANCES...")

    N, D, A, X, K = testdata_knn("")
    distance_types = ['cos', 'L2', 'dot', 'L1']
    #knn_result = our_knn(N, D, A, X, K)
    #print(knn_result)

    def benchmark_distance(distance_metric, num_trials=5, N=N, D=D, A=A, X=X, K=K):
        """
        Benchmark the execution time of the Top-K function for a given distance metric.

        Parameters:
        N : int - Number of vectors
        D : int - Dimension of vectors
        A : torch.Tensor (N, D) - Dataset of vectors (on GPU)
        X : torch.Tensor (D) - Query vector (on GPU)
        K : int - Number of neighbors
        distance_metric : str - Distance function ("L2", "cosine", "dot", "L1")
        num_trials : int - Number of runs to average the timing

        Returns:
        avg_time : float - Average execution time in milliseconds
        """
        torch.cuda.synchronize()
        total_time = 0.0

        for _ in range(num_trials):
            start_time = time.time()
            indices, distances = our_knn(N, D, A, X, K, distance_metric)
            torch.cuda.synchronize()  # Ensure all CUDA operations are finished
            end_time = time.time()
            total_time += (end_time - start_time)

        avg_time = (total_time / num_trials) * 1000  # Convert to milliseconds
        return avg_time
    
    # CODE TO BENCHMARK WITH DIFFERENT DISTANCES
    
def test_ann():
    N, D, A, X, K = testdata_ann("") # or test_file.json
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

# Example usage
if __name__ == "__main__":
    N, D = 4000000, 128  # 4000 vectors, each of 128 dimensions
    K = 5  # Top-K neighbors

    # Generate random dataset and query vector on GPU
    A = torch.randn(N, D, device="cuda")
    X = torch.randn(D, device="cuda")

    # Test different distance metrics
    distance_types = ['cos', 'L2', 'dot', 'L1']

    # for metric in metrics:
    #     time_taken = benchmark_distance(N, D, A, X, K, metric)
    #     results[metric] = time_taken
    #     print(f"{metric} distance took {time_taken:.2f} ms")

    # # Print best-performing metric
    # fastest_metric = min(results, key=results.get)
    # print(f"\n🚀 Fastest distance function: {fastest_metric} ({results[fastest_metric]:.2f} ms)")

    test_distances()

    print("_______________________________________\n\n")
    
    test_kmeans()

    print("_______________________________________\n\n")
    
    test_knn()

    print("_______________________________________\n\n")
    
    test_ann()

    print("_______________________________________\n\n")
    
    #recall_rate()