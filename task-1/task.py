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

def our_knn(N, D, A, X, K):
    pass

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

if __name__ == "__main__":
    test_kmeans()
