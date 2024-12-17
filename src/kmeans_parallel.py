import numpy as np
from multiprocessing import Pool


class KMeansParallel:
    def __init__(self, n_clusters=8, max_iter=300, n_jobs=4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.centroids = None

    def _assign_labels(self, chunk):
        distances = np.sqrt(((chunk - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def fit(self, X):
        # Initialize centroids
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]

        # Split data into chunks
        chunks = np.array_split(X, self.n_jobs)

        for _ in range(self.max_iter):
            # Parallel label assignment
            with Pool(self.n_jobs) as pool:
                labels_chunks = pool.map(self._assign_labels, chunks)

            labels = np.concatenate(labels_chunks)

            # Update centroids
            new_centroids = np.array([X[labels == k].mean(axis=0)
                                      for k in range(self.n_clusters)])

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids
