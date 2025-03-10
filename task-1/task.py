# Importing useful libraries
import torch
import cupy as cp
import numpy as np
import time
import json
import random
import warnings
from scipy.spatial.distance import cosine, euclidean, cityblock # (for testing)
from sklearn.neighbors import NearestNeighbors # (for testing)
from test import testdata_kmeans, testdata_knn, testdata_ann

# Defining float type
# (32-bit for fast computations and better precision than float16, which already made fail KNNs' proper recognition)
DTYPE = torch.float32


# ------------------------------------------------------------------------------------------------
# 1.1 - Distance-computing functions
# ------------------------------------------------------------------------------------------------
# Each distance formula is implemented through torch (default), cupy and torch for multi-dimentional vectors.

# Cosine distance: d(X, Y) = 1 - (X ⋅ Y) / (|X| |Y|)
# (an eps of 1e-8 is added to the denominator to avoid division by zero)
def cos_dist(X:torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    """
    INPUT
    - X[D],Y[D]: two vectors of the same dimension D
    OUTPUT
    - distance: the cosine distance between X and Y
    """
    return 1 - (torch.dot(X, Y) / (torch.norm(X) * torch.norm(Y) + 1e-8))
def cos_dist_cupy(X:cp.ndarray, Y:cp.ndarray) -> cp.ndarray:
    """
    INPUT
    - X[D],Y[D]: two vectors of the same dimension D
    OUTPUT
    - distance: the cosine distance between X and Y
    """
    return 1 - (cp.dot(X, Y) / (cp.linalg.norm(X) * cp.linalg.norm(Y) + 1e-8))
def cos_dist_multidim(X:torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    """
    INPUT
    - X,Y: two multidimensional vectors in the same space
    OUTPUT
    - distance: the cosine distance between X and Y
    """
    return 1 - (torch.sum(X * Y, dim=-1) / (torch.norm(X, dim=-1) * torch.norm(Y, dim=-1) + 1e-8))

# L2 (Euclidean) distance: d(X, Y) = sqrt(sum((X_i - Y_i)^2))
def L2_dist(X:torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    """
    INPUT
    - X[D],Y[D]: two vectors of the same dimension D
    OUTPUT
    - distance: the Euclidean (L2) distance between X and Y
    """
    return torch.sqrt(torch.sum((X - Y) ** 2))
def L2_dist_cupy(X:cp.ndarray, Y:cp.ndarray) -> cp.ndarray:
    """
    INPUT
    - X[D],Y[D]: two vectors of the same dimension D
    OUTPUT
    - distance: the Euclidean (L2) distance between X and Y
    """
    return cp.sqrt(cp.sum((X - Y) ** 2))
def L2_dist_multidim(X:torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    """
    INPUT
    - X,Y: two multidimensional vectors in the same space
    OUTPUT
    - distance: the Euclidean (L2) distance between X and Y
    """
    return torch.norm(X - Y, dim=-1)

# Dot-product distance: d(X, Y) = -(X ⋅ Y)
# (negative dot product is used, since high X ⋅ Y values mean vectors' proximity)
def dot_dist(X:torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    """
    INPUT
    - X[D],Y[D]: two vectors of the same dimension D
    OUTPUT
    - distance: the dot-product distance between X and Y
    """
    return - torch.dot(X, Y)
def dot_dist_cupy(X:cp.ndarray, Y:cp.ndarray) -> cp.ndarray:
    """
    INPUT
    - X[D],Y[D]: two vectors of the same dimension D
    OUTPUT
    - distance: the dot-product distance between X and Y
    """
    return - cp.dot(X, Y)
def dot_dist_multidim(X:torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    """
    INPUT
    - X,Y: two multidimensional vectors in the same space
    OUTPUT
    - distance: the dot-product distance between X and Y
    """
    return - torch.sum(X * Y, dim=-1)

# L1 (Manhattan) distance: d(X, Y) = sum(|X_i - Y_i|)
def L1_dist(X:torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    """
    INPUT
    - X[D],Y[D]: two vectors of the same dimension D
    OUTPUT
    - distance: the Manhattan (L1) distance between X and Y
    """
    return torch.sum(torch.abs(X - Y))
def L1_dist_cupy(X:cp.ndarray, Y:cp.ndarray) -> cp.ndarray:
    """
    INPUT
    - X[D],Y[D]: two vectors of the same dimension D
    OUTPUT
    - distance: the Manhattan (L1) distance between X and Y
    """
    return cp.sum(cp.abs(X - Y))
def L1_dist_multidim(X:torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    """
    INPUT
    - X,Y: two multidimensional vectors in the same space
    OUTPUT
    - distance: the Manhattan (L1) distance between X and Y
    """
    return torch.sum(torch.abs(X - Y), dim=-1)

# Mapping functions for later usage
distance_types = ['cos', 'L2', 'dot', 'L1']
dist_functions = {'cos':cos_dist, 'L2':L2_dist, 'dot':dot_dist, 'L1':L1_dist}
dist_multidim_functions = {'cos':cos_dist_multidim, 'L2':L2_dist_multidim, 'dot':dot_dist_multidim, 'L1':L1_dist_multidim}


# ------------------------------------------------------------------------------------------------
# 1.2 - KNN implementation
# ------------------------------------------------------------------------------------------------

def our_knn(N:int, D:int, A:torch.Tensor, X:torch.Tensor, K:int, distance_metric:str='dot', batch_size:int=100000) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes top-K nearest vectors in A for the query vector X.

    INPUTS
    - N: number of vectors
    - D: dimension of vectors
    - A[N, D]: dataset of vectors
    - X[D]: query vector
    - K: number of nearest neighbors to find
    - distance_metric: function used to compute the distances ('cos', 'L2', 'dot', 'L1')
    - batch_size: size of each batch to process large datasets

    OUTPUTS
    - top_k_indices[K]: indices of the K nearest vectors
    - top_k_distances[K]: corresponding distances
    """
    if distance_metric not in distance_types: raise ValueError(f"Invalid distance metric: {distance_metric}. Choose from {distance_types}.")
    dist_multidim_funct = dist_multidim_functions[distance_metric]

    # Ensuring inputs' proper format
    A = torch.as_tensor(A, dtype=DTYPE, device='cuda')
    device = A.device  # (same device)
    X = torch.as_tensor(X, dtype=DTYPE, device=device)

    # Preallocating memory
    num_batches = (N + batch_size - 1) // batch_size
    top_k_distances = torch.empty(K, dtype=DTYPE, device=device)
    top_k_indices = torch.empty(K, dtype=torch.long, device=device)

    # Computing top-k nearest vectors and their distances
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, N)
        batch = A[batch_start:batch_end]

        distances = dist_multidim_funct(batch, X) # Applying relevant distance function
        batch_top_k_distances, batch_top_k_indices = torch.topk(distances, K, largest=False)

        batch_top_k_indices += batch_start
        if i == 0:
            top_k_distances = batch_top_k_distances
            top_k_indices = batch_top_k_indices
        else:
            temp_distances = torch.cat((top_k_distances, batch_top_k_distances))
            temp_indices = torch.cat((top_k_indices, batch_top_k_indices))
            top_k_distances, indices = torch.topk(temp_distances, K, largest=False)
            top_k_indices = temp_indices[indices]

    return top_k_indices, top_k_distances


# ------------------------------------------------------------------------------------------------
# 2.1 - KMeans implementation
# ------------------------------------------------------------------------------------------------

def our_kmeans(N:int, D:int, A:torch.Tensor, K:int, distance_metric:str='dot', max_iters:int=1000, tol:float=1e-4, device:str='cuda') -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Clusters the vectors in A with the KMeans method.

    INPUTS
    - N: number of vectors
    - D: dimension of vectors
    - A[N, D]: dataset of vectors
    - K: number of clusters to devide A into
    - distance_metric: function used to compute the distances ('cos', 'L2', 'dot', 'L1')
    - max_iters: maximum number of iterations
    - tol: tolerance for convergence

    OUTPUTS
    - clusters[N]: cluster assignments respective to the vectors in A
    - centroids[K, D]: clusters' centroids
    - iterations: number of iterations taken to converge
    """
    if distance_metric not in distance_types: raise ValueError(f"Invalid distance metric: {distance_metric}. Choose from {distance_types}.")
    dist_multidim_funct = dist_multidim_functions[distance_metric]

    # Ensuring inputs' proper format
    A = torch.as_tensor(A, dtype=DTYPE, device=device)

    # Initializing centroids
    centroids = A[torch.randperm(N)[:K]]

    # Iterating KMeans steps
    for iteration in range(max_iters):
        prev_centroids = centroids.clone()
        # Assignment step
        distances = dist_multidim_funct(A[:, None, :], centroids[None, :, :])
        clusters = torch.argmin(distances, dim=1)
        # Update step
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(K, device=device, dtype=DTYPE)
        new_centroids.scatter_add_(0, clusters[:, None].expand(-1, D), A)
        counts.scatter_add_(0, clusters, torch.ones_like(clusters, dtype=DTYPE))
        mask = counts > 0
        new_centroids[mask] /= counts[mask, None]
        empty_clusters = (counts == 0).nonzero(as_tuple=True)[0]
        if empty_clusters.numel() > 0: # Reinitializing in case of empty clusters
            new_centroids[empty_clusters] = A[torch.randint(0, N, (empty_clusters.numel(),))]
        centroids = new_centroids
        # Convergence check
        if torch.max(torch.norm(centroids - prev_centroids, dim=1)) < tol:
            break
    
    return clusters, centroids, iteration+1


# ------------------------------------------------------------------------------------------------
# 2.2 - ANN implementation
# ------------------------------------------------------------------------------------------------

def our_ann(N:int, D:int, A:torch.Tensor, X:torch.Tensor, K:int, K_kmeans:int=20, K_knn:int=10, distance_metric:str='dot', batch_size:int=100000, max_iters:int=1000, tol:float=1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the top-K nearest vectors in A for the query vector X through the Approximate Nearest Neighbors (ANN) algorithm.

    INPUTS
    - N: number of vectors
    - D: dimension of vectors
    - A[N, D]: dataset of vectors
    - X[D]: query vector
    - K: number of nearest neighbors to find
    - K_kmeans: number of clusters to divide the dataset into
    - K_knn: number of centroid-nearest clusters to consider for retrieving the nearest vectors
    - distance_metric: function used to compute the distances ('cos', 'L2', 'dot', 'L1')
    - batch_size: size of each batch to process large datasets for KNN
    - max_iters: maximum number of iterations for KMeans
    - tol: tolerance for convergence in KMeans

    OUTPUTS
    - top_k_indices[K]: indices of the K nearest vectors
    - top_k_distances[K]: corresponding distances
    """
    if distance_metric not in distance_types: raise ValueError(f"Invalid distance metric: {distance_metric}. Choose from {distance_types}.")

    # Ensuring inputs' proper format
    A = torch.as_tensor(A, dtype=DTYPE, device='cuda')
    device = A.device  # (same device)
    X = torch.as_tensor(X, dtype=DTYPE, device=device)

    # Clustering dataset via KMeans
    clusters, centroids, _ = our_kmeans(N, D, A, K_kmeans, distance_metric, max_iters, tol)
    
    # KNN-finding nearest centroids to query vector
    top_indices, _ = our_knn(K_kmeans, D, centroids, X, K_knn, distance_metric, batch_size)
    
    # Retrieving candidate vectors from selected clusters
    candidate_indices = torch.cat([torch.nonzero(clusters == idx, as_tuple=True)[0] for idx in top_indices])

    # Warning if not enough candidates are found with the input parameters
    if len(candidate_indices) < K:
        warnings.warn(f"Only {len(candidate_indices)} candidate vectors were found, consider increasing K_knn or reducing K_kmeans to obtain {K} nearest neighbors instead", UserWarning)
        K = len(candidate_indices)
    candidates = A[candidate_indices]

    # KNN-finding nearest neighbors from selected clusters
    top_k_indices, top_k_distances = our_knn(len(candidate_indices), D, candidates, X, K, distance_metric, batch_size)
    top_k_indices = candidate_indices[top_k_indices]
    
    return top_k_indices, top_k_distances


# ------------------------------------------------------------------------------------------------
# Testing and benchmarking all functions
# ------------------------------------------------------------------------------------------------

def test_distances(D1 = 64, D2 = 256, dimensions = [2, 2**5, 2**10, 2**15, 2**20], num_trials = 1000, tolerance=1e-2):
    """
    Testing and benchmarking distance functions.
    """

    # Testing distances computation functions
    # (the test of the dot-product distance as defined is trivial, no specific already implemented version is used)
    print("\n\nTESTING DISTANCE FUNCTIONS...")

    def test_distance_function(func, std_func, X, Y, tolerance=tolerance):
        result = func(X, Y)

        # Ensuring inputs' proper format
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
            Y_np = Y.detach().cpu().numpy()
        elif isinstance(X, cp.ndarray):
            X_np = cp.asnumpy(X)
            Y_np = cp.asnumpy(Y)
        else:
            X_np = X
            Y_np = Y

        standard_result = std_func(X_np, Y_np)

        # Ensuring outputs' proper format
        if isinstance(result, torch.Tensor):
            result_np = result.detach().cpu().numpy()
        elif isinstance(result, cp.ndarray):
            result_np = cp.asnumpy(result)
        else:
            result_np = result

        # Compatring results with expected ones
        if np.allclose(result_np, standard_result, rtol=tolerance):
            return 'OK'
        else:
            return f'INCORRECT -> Error: {standard_result-result_np}'

    if torch.cuda.is_available():

        print("\nWITH PyTorch:")
        X_torch = torch.randn(D1, device='cuda', dtype=DTYPE)
        Y_torch = torch.randn(D1, device=X_torch.device, dtype=DTYPE)
        print("        Cosine Distance =", test_distance_function(cos_dist, cosine, X_torch, Y_torch))
        print("L2 (Euclidean) Distance =", test_distance_function(L2_dist, euclidean, X_torch, Y_torch))
        print("   Dot Product Distance =", test_distance_function(dot_dist, lambda X,Y: - np.dot(X, Y), X_torch, Y_torch))
        print("L1 (Manhattan) Distance =", test_distance_function(L1_dist, cityblock, X_torch, Y_torch))

        print("\nWITH Cupy:")
        X_cupy = cp.random.randn(D1).astype(np.float16 if DTYPE == torch.float16 else np.float32)
        Y_cupy = cp.random.randn(D1).astype(np.float16 if DTYPE == torch.float16 else np.float32)
        print("        Cosine Distance =", test_distance_function(cos_dist_cupy, cosine, X_cupy, Y_cupy))
        print("L2 (Euclidean) Distance =", test_distance_function(L2_dist_cupy, euclidean, X_cupy, Y_cupy))
        print("   Dot Product Distance =",  test_distance_function(dot_dist_cupy, lambda X,Y: - np.dot(X, Y), X_cupy, Y_cupy))
        print("L1 (Manhattan) Distance =", test_distance_function(L1_dist_cupy, cityblock, X_cupy, Y_cupy))

        print("\nWITH PyTorch AND multi-dimensional vectors:")
        X_multidim = torch.randn((D1,D2), device='cuda', dtype=DTYPE)
        Y_multidim = torch.randn((D1,D2), device=X_multidim.device, dtype=DTYPE)
        print("        Cosine Distance =", test_distance_function(cos_dist_multidim, lambda X, Y: 1 - np.sum(X * Y, axis=-1) / (np.linalg.norm(X, axis=-1) * np.linalg.norm(Y, axis=-1)), X_multidim, Y_multidim))
        print("L2 (Euclidean) Distance =", test_distance_function(L2_dist_multidim, lambda X, Y: np.linalg.norm(X - Y, axis=-1), X_multidim, Y_multidim))
        print("   Dot Product Distance =", test_distance_function(dot_dist_multidim, lambda X, Y: -np.sum(X * Y, axis=-1), X_multidim, Y_multidim))
        print("L1 (Manhattan) Distance =", test_distance_function(L1_dist_multidim, lambda X, Y: np.sum(np.abs(X - Y), axis=-1), X_multidim, Y_multidim))

    else:
        raise RuntimeError("CUDA unavailable!")
    
    print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n")

    # Benchmarking distances computation functions with different dimensions
    print("\nBENCHMARKING DISTANCE FUNCTIONS...")
    print(f"(trials# = {num_trials})")

    def benchmark_distance_functions(func_cpu, func_gpu1, func_gpu2, D, num_trials=num_trials):

        # Generating random vectors
        X_np = np.random.randn(D)
        Y_np = np.random.randn(D)

        # Measuring CPU time
        X_cpu_torch = torch.tensor(X_np, dtype=DTYPE) # PyTorch tensors (for CPU computation)
        Y_cpu_torch = torch.tensor(Y_np, dtype=DTYPE)
        start = time.time()
        for _ in range(num_trials):
            func_cpu(X_cpu_torch, Y_cpu_torch)
        end = time.time()
        cpu_time = (end - start) / num_trials # Average time per run

        # Measuring GPU time with method 1
        X_gpu_cupy = cp.array(X_np) # CuPy arrays (for GPU computation)
        Y_gpu_cupy = cp.array(Y_np)
        for _ in range(10): # Warming-up GPU
            func_gpu1(X_gpu_cupy, Y_gpu_cupy)
        cp.cuda.Device(0).synchronize()
        start = time.time()
        for _ in range(num_trials):
            func_gpu1(X_gpu_cupy, Y_gpu_cupy)
        cp.cuda.Device(0).synchronize()
        end = time.time()
        gpu1_time = (end - start) / num_trials # Average time per run

        # Measuring GPU time with method 2
        X_gpu_torch = torch.tensor(X_np, dtype=DTYPE, device='cuda') # PyTorch tensors (for GPU computation)
        Y_gpu_torch = torch.tensor(Y_np, dtype=DTYPE, device='cuda')
        for _ in range(10): # Warming-up GPU
            func_gpu2(X_gpu_torch, Y_gpu_torch)
        cp.cuda.Device(0).synchronize()
        start = time.time()
        for _ in range(num_trials):
            func_gpu2(X_gpu_torch, Y_gpu_torch)
        cp.cuda.Device(0).synchronize()
        end = time.time()
        gpu2_time = (end - start) / num_trials # Average time per run

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
            cpu_time, gpu1_time, gpu2_time = benchmark_distance_functions(func_cpu, func_gpu1, func_gpu2, D, num_trials)
            speedup1 = cpu_time / gpu1_time if gpu1_time > 0 else float("inf")
            speedup2 = cpu_time / gpu2_time if gpu2_time > 0 else float("inf")

            print(f"{name} -> {methods[0]} {cpu_time*1e6:>8.2f} µs  |  {methods[1]} {gpu1_time*1e6:>8.2f} µs ({speedup1:>5.2f}x)  |  {methods[2]} {gpu2_time*1e6:>8.2f} µs ({speedup2:>5.2f}x)")

metric_map = {'L2':'euclidean', 'L1':'manhattan', 'cos':'cosine', 'dot':'cosine'}    
def standard_knn(X_ref, X_query, K, metric='euclidean'):
    """
    Comparative standard KNN function.
    """
    X_ref = np.atleast_2d(X_ref)
    X_query = np.atleast_2d(X_query)
    if X_ref.ndim == 2 and X_ref.shape[1] == 1:
        X_ref = X_ref.reshape(-1, 1)
    if X_query.ndim == 2 and X_query.shape[1] == 1:
        X_query = X_query.reshape(-1, 1)
    knn = NearestNeighbors(n_neighbors=K, metric=metric)
    knn.fit(X_ref)
    distances, indices = knn.kneighbors(X_query)
    return indices, distances

def test_knn(N:int=1000, batch_size=100000, num_trials = 10, test_file=""):
    """
    Testing and benchmarking KNN with different distances functions.
    """
    print("\nTESTING AND BENCHMARKING KNN WITH DIFFERENT DISTANCES...")

    N, D, A, X, K = testdata_knn(test_file,N)

    print(f"N = {N} (D = {D}, K = {K}, trials# = {num_trials})")
    if N > batch_size: print(f"Data split in batches (batch_size = {batch_size})!")

    results = {}

    def benchmark_knn(distance_metric, num_trials=num_trials):
        torch.cuda.synchronize()
        total_time = 0.0
        last_indices, last_distances = None, None
        for _ in range(num_trials):
            start_time = time.time()
            indices, distances = our_knn(N, D, A, X, K, distance_metric, batch_size=batch_size)
            torch.cuda.synchronize()
            total_time += (time.time() - start_time)
            last_indices, last_distances = indices, distances
        avg_time = (total_time / num_trials) * 1000
        return avg_time, last_indices, last_distances
    
    # Benchmarking for each distance metric
    for distance_type in distance_types:
        avg_time, indices_custom, distances_custom = benchmark_knn(distance_type)
        indices_custom_np = indices_custom.cpu().numpy() if isinstance(indices_custom, torch.Tensor) else indices_custom

        A_np = A.cpu().numpy() if isinstance(A, torch.Tensor) else A
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        A_np = A_np.reshape(-1, D)
        X_np = X_np.reshape(-1, D)

        if distance_type == "dot":
            sim_matrix = -np.dot(X_np, A_np.T)
            indices_std = np.argsort(sim_matrix, axis=1)[:, :K]
        else:
            metric_std = metric_map[distance_type]
            indices_std, _ = standard_knn(A_np, X_np, K, metric=metric_std)

        # Comparing results
        correct_indices = np.array_equal(indices_custom_np, indices_std)
        if np.sum(indices_custom_np != indices_std)==0:
            print(f"WITH {distance_type:>3} distance: OK -> Avg.time = {avg_time:.2f}ms")
        else:
            mismatches = np.sum(indices_custom_np != indices_std)
            print(f"WITH {distance_type:>3} distance: INCORRECT ({mismatches} mismatches) -> Avg.time = {avg_time:.2f}ms")

        results[distance_type] = {'time_ms': avg_time, 'indices_match': correct_indices}

    return results

def test_kmeans(D:int=100, device:str='cuda' , max_iters:int=1000, tol:float=1e-4, num_trials=10, test_file=""):
    """
    Benchmarking KMeans with different distance functions.
    """
    print("\nBENCHMARKING KMEANS WITH DIFFERENT DISTANCES...")

    N, D, A_np, K = testdata_kmeans(test_file,D)
    
    print(f"D = {D} (N = {N}, K = {K}, trials# = {num_trials})")
    if device != 'cuda': print(f"Device: {device}!)")

    results = {}

    # Preparing to benchmark each distance method for KMeans
    def benchmark_kmeans(N, D, A, K, distance_metric, num_trials=num_trials):
        A_torch = torch.tensor(A, dtype=DTYPE).to(device)
        times, iterations = [], []
        for _ in range(num_trials):
            start = time.time()
            Kmeans_labels, Kmeans_centroids, Kmeans_iterations = our_kmeans(N, D, A_torch, K, distance_metric, max_iters, tol, device)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
            iterations.append(Kmeans_iterations)
        avg_time = sum(times) / len(times)
        avg_iters = sum(iterations) / len(iterations)
        return avg_time, avg_iters

    # Computing results
    for dist_type in distance_types:
        avg_time, avg_iters = benchmark_kmeans(N, D, A_np, K, dist_type)
        avg_time = avg_time*1000
        results[dist_type] = {'time_ms': avg_time, 'iterations': avg_iters}
        print(f"WITH {dist_type:>3} DISTANCE: Avg.time = {avg_time:.2f}ms, Avg.iters = {avg_iters:.2f}")
    
    return results

def recall_rate(list1, list2):
    """
    Calculates average recall rate between two index arrays (2D).
    """
    list1 = np.atleast_2d(list1)
    list2 = np.atleast_2d(list2)
    
    total_recall = 0.0
    for row1, row2 in zip(list1, list2):
        matches = len(set(row1) & set(row2))
        recall = matches / len(row2)
        total_recall += recall
    return total_recall / len(list1)

def test_ann(K_kmeans:int=20, K_knn:int=10, batch_size:int=100000, max_iters:int=1000, tol:float=1e-4, num_trials=10, test_file=""):
    """
    Testing and benchmarking ANN with different distance functions.
    """
    print("\nBENCHMARKING ANN WITH DIFFERENT DISTANCES...")
    
    N, D, A, X, K = testdata_ann(test_file)

    print(f"K_kmeans = {K_kmeans}, K_knn = {K_knn} (N = {N}, D = {D}, K = {K}, trials# = {num_trials})")
    if N > batch_size: print(f" Data split in batches (batch_size = {batch_size})")

    results = {}

    def benchmark_ann(distance_metric, num_trials=num_trials):
        torch.cuda.synchronize()
        total_time = 0.0
        last_indices, last_distances = None, None
        for _ in range(num_trials):
            start_time = time.time()
            indices, distances = our_ann(N, D, A, X, K, K_kmeans, K_knn, distance_metric, batch_size, max_iters, tol)
            torch.cuda.synchronize()
            total_time += (time.time() - start_time)
            last_indices, last_distances = indices, distances
        avg_time = (total_time / num_trials) * 1000
        return avg_time, last_indices, last_distances
    
    # Benchmarking for each distance metric
    for distance_type in distance_types:
        avg_time, indices_custom, distances_custom = benchmark_ann(distance_type)
        indices_custom_np = indices_custom.cpu().numpy() if isinstance(indices_custom, torch.Tensor) else indices_custom

        A_np = A.cpu().numpy() if isinstance(A, torch.Tensor) else A
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        A_np = A_np.reshape(-1, D)
        X_np = X_np.reshape(-1, D)

        if distance_type == "dot":
            sim_matrix = -np.dot(X_np, A_np.T)
            indices_std = np.argsort(sim_matrix, axis=1)[:, :K]
        else:
            metric_std = metric_map[distance_type]
            indices_std, _ = standard_knn(A_np, X_np, K, metric=metric_std)

        # Comparing results
        recRate = recall_rate(indices_custom_np, indices_std)
        print(f"WITH {distance_type:>3} distance -> recall_rate = {recRate:.2f}, Avg.time = {avg_time:.2f}ms")
        
        results[distance_type] = {'time_ms': avg_time, 'recall_rate': recRate}

    return results


if __name__ == "__main__":

    test_distances()

    print("_______________________________________\n\n")
    
    test_knn()
    print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n")
    test_knn(N=4000)
    print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n")
    test_knn(N=4000000)
    
    print("_______________________________________\n\n")
    
    test_kmeans()
    for D in [2, 2**10]:
        print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n")
        resCPU = test_kmeans(D=D, device='cpu')
        resGPU = test_kmeans(D=D, device='cuda')
        speedups = []
        for distance_type in distance_types:
            speedups.append(resCPU[distance_type]['time_ms']/resGPU[distance_type]['time_ms'])
        print(f" -> Avg. GPU speedup = {np.average(speedups):.2f}x")

    print("_______________________________________\n")
    
    for K_kmeans, K_knn in [(20, 10), (10, 5), (5, 3), (3, 1)]:
        print()
        test_ann(K_kmeans=K_kmeans, K_knn=K_knn)
        print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")

    print("_______________________________________\n\n")