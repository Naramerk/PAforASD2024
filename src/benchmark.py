import numpy as np
import time
from sklearn.datasets import make_blobs
from kmeans_sequential import KMeansSequential
from kmeans_parallel import KMeansParallel
import matplotlib.pyplot as plt
import os
from multiprocessing import cpu_count

def run_benchmark(X, n_clusters, max_iter=300):
    # Sequential version
    start = time.time()
    kmeans_seq = KMeansSequential(n_clusters=n_clusters, max_iter=max_iter)
    kmeans_seq.fit(X)
    seq_time = time.time() - start

    # Parallel version with different number of processes
    times_parallel = []
    max_cores = min(cpu_count(), 16)
    n_jobs_range = [2**i for i in range(1, int(np.log2(max_cores))+1)]

    for n_jobs in n_jobs_range:
        start = time.time()
        kmeans_par = KMeansParallel(n_clusters=n_clusters,
                                    max_iter=max_iter,
                                    n_jobs=n_jobs)
        kmeans_par.fit(X)
        times_parallel.append(time.time() - start)

    return seq_time, times_parallel, n_jobs_range

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    n_samples = 100000
    n_features = 10
    n_clusters = 8
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    print("Dataset generated successfully")

    os.makedirs('../benchmarks', exist_ok=True)

    print("Starting benchmark...")
    seq_time, parallel_times, n_jobs_range = run_benchmark(X, n_clusters)

    speedups = [seq_time / p_time for p_time in parallel_times]

    plt.figure(figsize=(10, 6))
    plt.plot(n_jobs_range, speedups, marker='o')
    plt.title('K-means Clustering Speedup')
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.grid(True)
    plt.savefig('../benchmarks/speedup.png')
    plt.close()

    print(f"Sequential time: {seq_time:.2f} seconds")
    for jobs, time in zip(n_jobs_range, parallel_times):
        print(f"Parallel time ({jobs} processes): {time:.2f} seconds")
    print(f"Speedups: {speedups}")
