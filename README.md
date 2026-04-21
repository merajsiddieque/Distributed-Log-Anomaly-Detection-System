# Distributed Log Anomaly Detection System (MPI + OpenMP Hybrid)

## 📌 Overview
This project implements a **Distributed Log Anomaly Detection System** using High Performance Computing (HPC). It analyzes HDFS logs and classifies them as **Normal** or **Anomalous** using parallel processing.

We implemented and compared three execution models:
- **OpenMP** → Shared-memory parallelism (single VM)
- **MPI** → Distributed parallelism (multi-VM)
- **Hybrid (MPI + OpenMP)** → Combined model for best scalability

---

## 🧠 What We Did

- Set up **two VMs (node1 & node2)** on VirtualBox
- Configured **SSH + hostfile-based MPI cluster**
- Implemented:
  - Pure **OpenMP** version
  - Pure **MPI** version
  - **Hybrid MPI + OpenMP** version
- Built full pipeline in **C**:
  - Log parsing
  - Feature extraction
  - Lightweight classification model
- Used **rank-based data distribution** (`i % size == rank`)
- Used **OpenMP reduction** for parallel metrics
- Evaluated using:
  - Accuracy, Precision, Recall, F1-score, Execution time

---

## 📊 Datasets Used

We worked with **two different datasets**:

### 1. Small HDFS Dataset (LogHub split)
- Located in `dataset/` and `dataset_split/`
- Includes:
  - Training data
  - Normal test logs
  - Abnormal test logs
- Size: ~59MB total

Used for:
- Model testing
- Debugging
- Performance comparison (fast runs)

### 2. Large Dataset (HDFS.log)
- Located in `hybrid-openmp-mpi/`
- File: `HDFS.log`
- Size: **~1.5 GB**

Used for:
- Real scalability testing
- Stress testing MPI + OpenMP
- Evaluating distributed performance

---

## 🏗️ System Architecture

### 🔹 MPI Layer
- Runs across **node1 and node2**
- Splits workload using rank
- Uses `MPI_Reduce` to combine results

### 🔹 OpenMP Layer
- Runs inside each node
- Parallel loop:
```c
#pragma omp parallel for reduction(+:TP,FP,TN,FN)
```

### 🔹 Hybrid Model
- MPI distributes logs across machines
- OpenMP accelerates processing inside each machine

---

## 📁 Updated Project Structure

```
log_monit/
│
├── Makefile
├── main.c
├── run.sh
├── hosts
├── log_monitor
├── hybrid_run
├── mpi
│
├── dataset/                     # Small dataset (~59MB)
│   ├── anomaly_label.csv
│   ├── hdfs_train.txt
│   ├── hdfs_test_normal.txt
│   └── hdfs_test_abnormal.txt
│
├── dataset_split/
│   ├── train/
│   │   ├── normal.txt
│   │   └── abnormal.txt
│   └── test/
│       ├── normal.txt
│       └── abnormal.txt
│
├── model/
│   ├── fc_weight.txt
│   └── fc_bias.txt
│
├── only_openmp/
│   ├── main.c
│   ├── openmp_run
│   └── run_openmp.sh
│
├── only_mpi/
│   ├── main.c
│   ├── mpi_run
│   └── run_mpi.sh
│
├── hybrid/
│   ├── main.c
│   ├── hybrid_run
│   └── run_hybrid.sh
│
├── hybrid-openmp-mpi/          # Large dataset experiments (~1.5GB)
│   ├── HDFS.log
│   ├── openmp.c
│   ├── mpi.c
│   ├── hybrid.c
│   ├── openmp_run
│   ├── mpi_run
│   ├── hybrid_run
│   ├── run-openmp.sh
│   ├── run-mpi.sh
│   └── run-hybrid.sh
│
└── README.md
```

---

## 🌐 Cluster Setup

Nodes used:
- `vm1@node1`
- `vm1@node2`

### hosts file:
```
node1 slots=2
node2 slots=2
```

---

## ⚙️ Requirements

- GCC with OpenMP
- OpenMPI / MPICH
- Linux (Ubuntu)
- VirtualBox (multi-node setup)

---

## 🛠️ Compilation

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

## 🚀 Execution

### OpenMP
```
export OMP_NUM_THREADS=4
./openmp_run
```

### MPI
```
mpirun -np 4 \
--hostfile hosts \
--map-by ppr:2:node \
--bind-to core \
./mpi_run
```

### Hybrid
```
export OMP_NUM_THREADS=2

mpirun -np 4 \
--hostfile hosts \
--map-by ppr:2:node \
--bind-to core \
./hybrid_run
```

---

## 🔄 Execution Flow

1. Initialize MPI
2. Load dataset (small or large)
3. Distribute logs using rank
4. Process logs in parallel (OpenMP)
5. Perform anomaly detection
6. Reduce results (MPI)
7. Print final metrics (rank 0)
8. Finalize MPI

---

## 📊 Metrics

- TP, FP, TN, FN
- Accuracy
- Precision
- Recall
- F1 Score
- Execution Time

---

## 🧪 Sample Output

```
=== HYBRID CLASSIFICATION RESULTS ===
TP: 2, FP: 7, TN: 110521, FN: 3366
Total Samples: 113896
Time: 0.17 sec
Accuracy: 0.9704
Precision: 0.2222
Recall: 0.0006
F1 Score: 0.0012
```

---

## ⚠️ Observations

- Dataset is **highly imbalanced**
- Accuracy can be misleading
- **F1-score is more reliable**

### Performance:
- OpenMP → Fast (single VM)
- MPI → Scales across nodes
- Hybrid → Best performance overall

---

## 🎯 Conclusion

We successfully built a **distributed anomaly detection system in C** and validated it on both:
- Small structured dataset
- Large real-world log dataset (~1.5GB)

The **Hybrid (MPI + OpenMP)** approach provides the best balance of speed and scalability.

---

## 🚀 Future Work

- Deep learning (LSTM / Transformer)
- Real-time log streaming
- GPU acceleration
- Better handling of class imbalance

---

## 👤 Author

Meraj Alam Siddique
Kotha Anshul Reddy
Mihir Hajare
