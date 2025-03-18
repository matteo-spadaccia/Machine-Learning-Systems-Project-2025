# Task 1

> [!TIP]
> To run the test and benchmaking of Kmeans, KNN and ANN with the different distances fucntions:
> ```shell
> python task.py
> ```
> To save the output in a text file too:
> ```shell
> python -u task.py | tee output.txt
> ```

## Part 1: KNN

### Part 1.1: Distance Functions

Implement distance functions for two vectors in cupy/torch/triton.  

**Input:**  
- D: dimension of the vector.  
- X[D], Y[D]: two vectors  

**Output:**  
- the distance between X and Y 

There are four distinct types of distance, so the implementation of four separate functions:

**Cosine distance:**:

$d(X, Y) = 1 - \frac{X \cdot Y}{\|X\| \|Y\|}$

**L2 distance:**:

$d(X, Y) = \sqrt{\sum_{i=1}^D (X_i - Y_i)^2}$

**Dot product:**:

$d(X, Y) = X \cdot Y$

**Manhattan(L1) distance:**:

$d(X, Y) = \sum_{i=1}^D |X_i - Y_i|$

### Part 1.2: Top-K with GPU

Identify the K nearest vectors within a set of vectors.

**Input:**
-  N: Number of vectors
-  D: Dimension of vectors
-  A[N, D]: A collection of vectors
-  X: A specified vector
-  K: Top K

**Output:**
-  Result[K]: The top K nearest vectors ID (index of the vector in A)

### Report

In the Task 1 report, you are required to answer the following questions:

1. How did you implement four distinct distance functions on the GPU?  
2. What is the speed advantage of the GPU over the CPU version when the dimension is 2? Additionally, what is the speed advantage when the dimension is 2^15?  
3. Please provide a detailed description of your Top K algorithm.  
4. What steps did you undertake to implement the Top K on the GPU? How do you manage data within GPU memory?  
5. When processing 4,000 vectors, how many seconds does the operation take? Furthermore, when handling 4,000,000 vectors, what modifications did you implement to ensure the effective functioning of your code?


## Part 2: KMeans and ANN

### Part 2.1: KMeans

**What is Kmeans algorithm:**
https://en.wikipedia.org/wiki/K-means_clustering

K-means clustering is an unsupervised learning algorithm that groups similar data points into K clusters. It works by iteratively assigning points to the nearest cluster center (centroid) and updating centroids based on the mean of assigned points until convergence.
In this task we only use L2 distance and cosine similarity.

**Pesudo code:**
```
1. Initialize:
   - Randomly select K points from dataset as initial centroids

2. REPEAT:
   a. Assignment step:
      - For each data point:
         - Calculate distance to each centroid
         - Assign point to closest centroid's cluster
   
   b. Update step:
      - For each cluster:
         - Calculate mean of all points in cluster
         - Set new centroid position to cluster mean

3. UNTIL:
   - Centroids no longer move significantly OR
   - Maximum iterations reached
```

**Input:**
-  N: Number of vectors
-  D: Dimension of vectors
-  A[N, D]: A collection of vectors
-  K: number of clusters

**Output:**
-  Result[N]: cluster ID for each vector

### Part 2.2: ANN

**What is ANN algorithm:**
https://en.wikipedia.org/wiki/Nearest_neighbor_search


**Pesudo code:**
```
1. Use KMeans to cluster the data into K clusters
2. In each query, find the nearest K1 cluster center as the approximate nearest neighbor
3. Use KNN to find the nearest K2 neighbor from the K1 cluster centers
4. Merge K1 * K2 vectors and find top K neighbors
```

Keep in mind that the ANN algorithm is merely an approximation algorithm. We determine the recall rate by comparing the results with your KNN results.

Recall rate = (#Same vectors in KNN result and ANN result) / K

If the recall rate exceeds 70% across all data, we consider your result to be correct.

(You can also implement other ANN algorithms such as HNSW or IVFPQ. However, you cannot use libraries other than cupy/triton/pytorch. In other words, you cannot use libraries like faiss/milvus/cuvs to complete the task in just one line of code.)

**Input:**
-  N: Number of vectors
-  D: Dimension of vectors
-  A[N, D]: A collection of vectors
-  X: A specified vector
-  K: Top K

**Output:**
-  Result[K]: The top K nearest vectors ID (index of the vector in A)


### Report

In the Task 2 report, you are required to answer the following questions:  

1. How did you implement your K-means algorithm on the GPU?  
2. What is the speed advantage of the GPU over the CPU version when the dimension is 2? Additionally, what is the speed advantage when the dimension is 1024?  
3. Please provide a detailed description of your ANN algorithm.  
4. If you implemented another clustering algorithm/ANN algorithm, which algorithm did you use?

## Other aspects for the report

You can write a report by discussing from the following aspects:

1. Implementation Analysis
   - Describe your implementation approach on GPU
   - Include code snippets and explain key optimization techniques used
   - Discuss any challenges encountered and how they were resolved

2. Performance Comparison
   - Conduct benchmarking tests comparing GPU vs CPU(numpy) implementations
   - Present results in a table/figure showing execution times and speedup ratios
   - Analyze the factors contributing to performance differences

3. Scalability Analysis
   - Benchmark performance with small number of vectors
     * Report execution time and resource utilization (e.g. GPU memory usage/GPU utilization if you can measure it)
   - Scale testing to large number of vectors
     * Document optimization techniques used
     * Compare performance before and after optimizations
     * Analyze memory usage and bottlenecks

Your report should include relevant charts, tables and metrics to support your analysis. Focus on quantitative results and technical insights gained from the implementation.
