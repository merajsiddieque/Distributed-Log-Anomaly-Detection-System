# A. Compile the hybrid binary
mpicc -O3 -fopenmp hybrid.c -o run_hybrid

# B. Send the binary to node2
scp run_hybrid node2:~/log_monit/hybrid-openmp-mpi/

# C. Run across both nodes
export OMP_NUM_THREADS=2
mpirun -np 4 --hostfile hosts --map-by ppr:2:node \
--mca io ompio \
--mca sharedfp ^lockedfile \
./run_hybrid
