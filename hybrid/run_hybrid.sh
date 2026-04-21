#!/bin/bash

echo "=== COMPILING ==="
mpicc -fopenmp main.c -lm -o hybrid_run

if [ 0 -ne 0 ]; then
    echo "Compilation failed ❌"
    exit 1
fi

echo "=== COPYING TO node2 ==="
scp hybrid_run vm1@node2:~/log_monit/hybrid/

if [ 0 -ne 0 ]; then
    echo "SCP failed ❌"
    exit 1
fi

echo "=== SETTING THREADS ==="
export OMP_NUM_THREADS=2

echo "=== RUNNING HYBRID (MPI + OpenMP) ==="
mpirun -np 4 \
--hostfile ../hosts \
--map-by ppr:2:node \
--bind-to core \
./hybrid_run

echo "=== DONE ==="
