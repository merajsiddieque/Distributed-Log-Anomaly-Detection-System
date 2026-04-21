#!/bin/bash

echo "=== COMPILING MPI CODE ==="
mpicc main.c -lm -o mpi_run

if [ 0 -ne 0 ]; then
    echo "Compilation failed ❌"
    exit 1
fi

echo "=== COPYING TO node2 ==="
scp mpi_run vm1@node2:~/log_monit/only_mpi/

if [ 0 -ne 0 ]; then
    echo "SCP failed ❌"
    exit 1
fi

echo "=== RUNNING MPI ==="
mpirun -np 4 \
--hostfile ../hosts \
--map-by ppr:2:node \
--bind-to core \
./mpi_run

echo "=== DONE ==="
