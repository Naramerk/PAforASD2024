# K-means Clustering with Parallel Processing

## Algorithm and Parallelization Method
- Classic k-means clustering algorithm implementation
- Parallelization using Python's multiprocessing module
- Data parallelization strategy for distance calculations and cluster assignments
- Dynamic process allocation based on available CPU cores

## Instructions to Run
1. Clone the repository and set up environment:
```bash
git clone https://github.com/[username]/[repository]
cd [repository]
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install numpy matplotlib scikit-learn
```

3. Run the benchmark:
```bash
python3 src/benchmark.py
```

## Data Generation
- Synthetic dataset generated using sklearn.datasets.make_blobs
- Parameters:
  - 100,000 samples
  - 10 features per sample
  - 8 clusters
  - Random state: 42 for reproducibility
- Data is generated automatically when running benchmark.py

## Parallelized Components
The following parts of the k-means algorithm are parallelized:
- Distance calculations between points and centroids
- Assignment of points to nearest clusters
- Data is split into chunks and processed by multiple processes

## Project Structure
```
.
├── src/
│   ├── benchmark.py          # Performance testing
│   ├── kmeans_sequential.py  # Sequential implementation
│   └── kmeans_parallel.py    # Parallel implementation
└── benchmarks/              # Contains speedup graph
```

## Performance Analysis
- Speedup measured as: sequential_time / parallel_time
- Number of processes: powers of 2 up to available CPU cores.
- Results:
  - Execution times printed to console:
    Sequential time: 4.30 seconds
    Parallel time (2 processes): 3.56 seconds
    Parallel time (4 processes): 43.15 seconds
    Parallel time (8 processes): 2.55 seconds
    Speedups: [1.2077684089421359, 0.09956750927130907, 1.6870601610674947]
    ![image](https://github.com/user-attachments/assets/4e4edfb4-60de-482c-bb8a-e07016672577)

  - Process count vs speedup relationship visualized
  - - Speedup graph saved as speedup.png
![image](https://github.com/user-attachments/assets/344ab990-afa2-416f-b7e0-8e03d35b030a)

