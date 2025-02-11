import cupy as cp
import torch

# Cosine Distance
def cosine_distance(X, Y):
    return 1 - (cp.dot(X, Y) / (cp.linalg.norm(X) * cp.linalg.norm(Y)))

def cosine_distance_torch(X, Y):
    return 1 - (torch.dot(X, Y) / (torch.norm(X) * torch.norm(Y)))

# L2 Distance
def l2_distance(X, Y):
    return cp.sqrt(cp.sum((X - Y) ** 2))

def l2_distance_torch(X, Y):
    return torch.sqrt(torch.sum((X - Y) ** 2))

# Dot Product
def dot_product(X, Y):
    return cp.dot(X, Y)

def dot_product_torch(X, Y):
    return torch.dot(X, Y)

# Manhattan (L1) Distance
def manhattan_distance(X, Y):
    return cp.sum(cp.abs(X - Y))

def manhattan_distance_torch(X, Y):
    return torch.sum(torch.abs(X - Y))
