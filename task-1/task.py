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

def distance_cosine(X, Y):
    """
    Compute the cosine distance: d(X, Y) = 1 - (X ⋅ Y) / (|X| |Y|)
    """
    dot_product = torch.sum(X * Y, dim=-1)
    norm_X = torch.norm(X, dim=-1)
    norm_Y = torch.norm(Y, dim=-1)
    return 1 - (dot_product / (norm_X * norm_Y))


def distance_l2(X, Y):
    """
    Compute the L2 (Euclidean) distance: d(X, Y) = sqrt(sum((X_i - Y_i)^2))
    """
    return torch.sqrt(torch.sum((X - Y) ** 2, dim=-1))


def distance_dot(X, Y):
    """
    Compute the dot product: d(X, Y) = X ⋅ Y
    """
    return torch.sum(X * Y, dim=-1)


def distance_manhattan(X, Y):
    """
    Compute the Manhattan (L1) distance: d(X, Y) = sum(|X_i - Y_i|)
    """
    return torch.sum(torch.abs(X - Y), dim=-1)


# Example usage (on GPU if available)
def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.randn((10000, 128), device=device)  # Large batch of vectors
    Y = torch.randn((10000, 128), device=device)
    
    print("Cosine Distance:", distance_cosine(X, Y))
    print("L2 Distance:", distance_l2(X, Y))
    print("Dot Product:", distance_dot(X, Y))
    print("Manhattan Distance:", distance_manhattan(X, Y))

test()


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
