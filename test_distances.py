import cupy as cp
import torch
from distance_functions import *

# Test vectors
D = 128
X = cp.random.randn(D)
Y = cp.random.randn(D)

print("Testing Distance Functions with CuPy:")
print("Cosine Distance:", cosine_distance(X, Y).item())
print("L2 Distance:", l2_distance(X, Y).item())
print("Dot Product:", dot_product(X, Y).item())
print("Manhattan Distance:", manhattan_distance(X, Y).item())

X_torch = torch.randn(D, device="cuda")
Y_torch = torch.randn(D, device="cuda")

print("\nTesting Distance Functions with PyTorch:")
print("Cosine Distance:", cosine_distance_torch(X_torch, Y_torch).item())
print("L2 Distance:", l2_distance_torch(X_torch, Y_torch).item())
print("Dot Product:", dot_product_torch(X_torch, Y_torch).item())
print("Manhattan Distance:", manhattan_distance_torch(X_torch, Y_torch).item())
