# Distributed Log Anomaly Detection System (MPI, OpenMP, Hybrid)

## Overview
This project implements a high-performance **log anomaly detection system** in C using three parallelization strategies:

- **MPI (Message Passing Interface)** → distributed memory parallelism
- **OpenMP** → shared memory parallelism
- **Hybrid (MPI + OpenMP)** → combines both for scalability and speed

The system processes HDFS log data, extracts features, applies a lightweight classification model, and evaluates performance using standard metrics.

---

## Core Processing Pipeline
All three implementations follow the same logical pipeline:

1. Load model parameters (weights and bias)
2. Read dataset
3. Parse log sequences
4. Extract features
5. Compute linear score
6. Apply sigmoid function
7. Apply threshold for classification
8. Compute TP, FP, TN, FN
9. Aggregate results (thread-level or process-level)
10. Compute final metrics

---

## Implementations

### 1. MPI Version (Distributed)

**Key Idea:** Work is divided across multiple processes (possibly across machines).

**Flow:**
- Initialize MPI and assign ranks
- Split dataset into chunks using rank
- Each process handles its portion
- Compute local metrics (TP, FP, TN, FN)
- Combine results using `MPI_Reduce`
- Rank 0 computes final metrics and prints results

**Parallelism Type:** Process-level

---

### 2. OpenMP Version (Shared Memory)

**Key Idea:** Parallelize loops using threads on a single machine.

**Flow:**
- Load dataset into memory
- Use `#pragma omp parallel for` to distribute loop iterations
- Each thread processes part of dataset
- Use reduction to combine metrics
- Compute final metrics

**Parallelism Type:** Thread-level

---

### 3. Hybrid (MPI + OpenMP)

**Key Idea:** Combine distributed and shared memory parallelism.

**Flow:**
- MPI distributes data across processes (nodes)
- Each process uses OpenMP threads internally
- Threads process data in parallel
- Local results reduced using OpenMP
- Global results reduced using MPI

**Parallelism Type:** Process + Thread level

---

## Key Differences

| Approach | Parallelism | Scope |
|--------|------------|------|
| MPI | Process-level | Multi-node |
| OpenMP | Thread-level | Single node |
| Hybrid | Both | Multi-node + multi-core |

---

## Output Definition

### Primary Output
```
Total Logs Processed: <N>
Total Anomalies Detected: <A>
Normal Logs: <N - A>
```

### Timing Output
```
Execution Time: <T> seconds
```

### Metrics
- True Positive (TP)
- False Positive (FP)
- True Negative (TN)
- False Negative (FN)
- Accuracy
- Precision
- Recall
- F1 Score

---

## Datasets

### Small Dataset (~59MB)
- Located in `dataset/` and `dataset_split/`
- Used for testing and debugging

### Large Dataset (~1.5GB)
- File: `HDFS.log`
- Used for scalability and stress testing

---

## System Architecture

### MPI Layer
- Distributes workload across nodes
- Uses rank-based partitioning
- Aggregates results via `MPI_Reduce`

### OpenMP Layer
- Parallel processing inside each node
- Uses loop-level parallelism

### Hybrid Model
- MPI handles distribution
- OpenMP accelerates local computation

---

## Project Structure
```
log_monit/
├── dataset/
├── dataset_split/
├── model/
├── only_openmp/
├── only_mpi/
├── hybrid/
├── main.c
├── Makefile
└── README.md
```

---

## Requirements

- GCC with OpenMP support
- MPI (OpenMPI or MPICH)
- Linux (Ubuntu recommended)
- VirtualBox (for multi-node setup)

---

## Compilation

### OpenMP
```
gcc -fopenmp main.c -lm -o openmp_run
```

### MPI
```
mpicc main.c -lm -o mpi_run
```

### Hybrid
```
mpicc -fopenmp main.c -lm -o hybrid_run
```

---

## Execution

### OpenMP
```
export OMP_NUM_THREADS=4
./openmp_run
```

### MPI
```
mpirun -np 4 --hostfile hosts ./mpi_run
```

### Hybrid
```
export OMP_NUM_THREADS=2
mpirun -np 4 --hostfile hosts ./hybrid_run
```

---

## Execution Flow (Hybrid)

1. Initialize MPI
2. Load dataset
3. Distribute data using rank
4. Process data using OpenMP threads
5. Perform classification
6. Reduce results using MPI
7. Print metrics (rank 0)
8. Finalize MPI

---

## Sample Output
```
TP: 2, FP: 7, TN: 110521, FN: 3366
Total Samples: 113896
Time: 0.17 sec
Accuracy: 0.9704
Precision: 0.2222
Recall: 0.0006
F1 Score: 0.0012
```

---

## Observations

- Dataset is highly imbalanced
- Accuracy alone is misleading
- F1-score is a better performance metric

### Performance Comparison
- OpenMP → Fast on single machine
- MPI → Scales across machines
- Hybrid → Best overall performance

---

## Conclusion

This project demonstrates how **parallel computing techniques** can significantly improve performance in large-scale log processing systems. The hybrid model provides the best balance between scalability and efficiency.

---

## Future Work

- Deep learning models (LSTM, Transformers)
- Real-time log streaming
- GPU acceleration
- Improved handling of class imbalance

---

## Authors

- Meraj Alam Siddique
- Kotha Anshul Reddy
- Mihir Hajare
- Pranav Sai
