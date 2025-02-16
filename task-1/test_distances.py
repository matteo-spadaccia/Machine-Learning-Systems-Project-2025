import time
import numpy as np
import cupy as cp
import torch
from task import *


# ------------------------------------------------------------------------------------------------
# Testing distances computation functions
# ------------------------------------------------------------------------------------------------

D1 = 10000
D2 = 128

if torch.cuda.is_available():
    print("\n\nTESTING DISTANCE FUNCTIONS...")

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


# ------------------------------------------------------------------------------------------------
# Benchmarking distances computation (excluding data transfer times)
# ------------------------------------------------------------------------------------------------

dimensions = [2, 32768, 1000000]
num_trials = 1000

print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n")
print("\nBENCHMARKING DISTANCE FUNCTIONS...")

def benchmark_distance(func_cpu, func_gpu1, func_gpu2, D, num_trials=1000):

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

print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n\n")