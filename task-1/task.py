import torch
import cupy as cp
import triton
import triton.language as tl
import numpy as np
import time
import json
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

# You can create any kernel here

import torch
import time

def top_k_gpu(N, D, A, X, K, distance_metric="L2", batch_size=100000):
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
        

# Example usage:
    

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
def test_kmeans():
    N, D, A, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

def benchmark_distance(N, D, A, X, K, distance_metric, num_trials=5):
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
        indices, distances = top_k_gpu(N, D, A, X, K, distance_metric)
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished
        end_time = time.time()
        total_time += (end_time - start_time)

    avg_time = (total_time / num_trials) * 1000  # Convert to milliseconds
    return avg_time

# Example usage
if __name__ == "__main__":
    N, D = 4000000, 128  # 4000 vectors, each of 128 dimensions
    K = 5  # Top-K neighbors

    # Generate random dataset and query vector on GPU
    A = torch.randn(N, D, device="cuda")
    X = torch.randn(D, device="cuda")

    # Test different distance metrics
    metrics = ["L2", "cosine", "dot", "L1"]
    results = {}

    for metric in metrics:
        time_taken = benchmark_distance(N, D, A, X, K, metric)
        results[metric] = time_taken
        print(f"{metric} distance took {time_taken:.2f} ms")

    # Print best-performing metric
    fastest_metric = min(results, key=results.get)
    print(f"\n🚀 Fastest distance function: {fastest_metric} ({results[fastest_metric]:.2f} ms)")
