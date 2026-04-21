export OMP_NUM_THREADS=2

mpirun -np 4 \
--hostfile hosts \
--map-by ppr:2:node \
--bind-to core \
./log_monitor
