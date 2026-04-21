# Compile
gcc -O3 -fopenmp openmp.c -o run_omp

# Set threads (e.g., 4) and run
export OMP_NUM_THREADS=4
./run_omp
