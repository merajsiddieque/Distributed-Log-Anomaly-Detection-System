mpicc -O3 mpi.c -o run_mpi && \
scp run_mpi node2:~/log_monit/hybrid-openmp-mpi/ && \
mpirun -np 4 --hostfile hosts --map-by ppr:2:node \
--mca btl_tcp_if_include enp0s3 \
--mca oob_tcp_if_include enp0s3 \
--mca io ompio \
--mca sharedfp ^lockedfile \
./run_mpi
