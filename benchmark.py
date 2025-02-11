import time
import numpy as np
import cupy as cp
import torch
from distance_functions import *

def benchmark_distance(func_gpu, func_cpu, D, num_trials=1000):
    """
    Measures execution time of a function for GPU (CuPy) and CPU (NumPy/Torch),
    excluding data transfer times.
    """
    # Generate random vectors
    X_cpu_np = np.random.randn(D)
    Y_cpu_np = np.random.randn(D)

    # Convert NumPy to Torch tensor (for CPU functions using PyTorch)
    X_cpu_torch = torch.tensor(X_cpu_np, dtype=torch.float32)
    Y_cpu_torch = torch.tensor(Y_cpu_np, dtype=torch.float32)

    # Convert NumPy to CuPy array (for GPU functions)
    X_gpu = cp.array(X_cpu_np)
    Y_gpu = cp.array(Y_cpu_np)

    # Measure CPU time
    start_cpu = time.time()
    for _ in range(num_trials):
        func_cpu(X_cpu_torch, Y_cpu_torch)  # Use Torch tensors
    end_cpu = time.time()
    cpu_time = (end_cpu - start_cpu) / num_trials  # Average time per run

    # Warm-up GPU
    for _ in range(10):
        func_gpu(X_gpu, Y_gpu)
    cp.cuda.Device(0).synchronize()

    # Measure GPU time
    start_gpu = time.time()
    for _ in range(num_trials):
        func_gpu(X_gpu, Y_gpu)
    cp.cuda.Device(0).synchronize()
    end_gpu = time.time()
    gpu_time = (end_gpu - start_gpu) / num_trials  # Average time per run

    return cpu_time, gpu_time

if __name__ == "__main__":
    dimensions = [2, 32768, 1000000]  # Add large case (1M)
    num_trials = 1000

    for D in dimensions:
        print(f"\nBenchmarking for Dimension D={D}...")

        for name, func_gpu, func_cpu in [
            ("Cosine Distance", cosine_distance, cosine_distance_torch),
            ("L2 Distance", l2_distance, l2_distance_torch),
            ("Dot Product", dot_product, dot_product_torch),
            ("Manhattan Distance", manhattan_distance, manhattan_distance_torch),
        ]:
            cpu_time, gpu_time = benchmark_distance(func_gpu, func_cpu, D, num_trials)
            speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

            print(f"{name}: CPU {cpu_time*1e6:.2f} µs | GPU {gpu_time*1e6:.2f} µs | Speedup: {speedup:.2f}x")
