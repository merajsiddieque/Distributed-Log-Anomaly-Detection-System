#!/bin/bash

echo "=== COMPILING OPENMP CODE ==="
gcc -fopenmp main.c -lm -o openmp_run

if [ 0 -ne 0 ]; then
    echo "Compilation failed ❌"
    exit 1
fi

echo "=== SETTING THREADS ==="
export OMP_NUM_THREADS=4

echo "=== RUNNING OPENMP ==="
./openmp_run

echo "=== DONE ==="
