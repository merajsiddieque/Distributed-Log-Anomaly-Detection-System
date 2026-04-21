#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    int provided;
    // Request thread support for MPI
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_File fh;
    // Use MPI_COMM_SELF to bypass global locking issues on local filesystems
    int err = MPI_File_open(MPI_COMM_SELF, "HDFS.log", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) {
        printf("Rank %d: Error opening HDFS.log. Check if file exists on this node!\n", rank);
        MPI_Finalize();
        return 1;
    }

    MPI_Offset filesize;
    MPI_File_get_size(fh, &filesize);

    // Calculate chunks based on MPI rank
    MPI_Offset chunk = filesize / size;
    int overlap = 1024; // Buffer overlap to ensure words aren't cut between ranks
    if (rank == size - 1) overlap = 0;

    long read_size = (long)chunk + overlap;
    char *buffer = malloc(read_size + 1);
    if (!buffer) { 
        MPI_File_close(&fh);
        MPI_Finalize(); 
        return 1; 
    }

    double start = MPI_Wtime();

    // Perform the read at the specific offset for this rank
    MPI_File_read_at(fh, rank * chunk, buffer, read_size, MPI_CHAR, MPI_STATUS_IGNORE);
    buffer[read_size] = '\0';

    // Local counters for thread reduction
    long l_info=0, l_warn=0, l_error=0, l_fatal=0, l_repl=0;

    #pragma omp parallel reduction(+:l_info, l_warn, l_error, l_fatal, l_repl)
    {
        int t_id = omp_get_thread_num();
        int t_count = omp_get_num_threads();
        long t_chunk = read_size / t_count;
        long t_start = t_id * t_chunk;
        
        // Threads overlap slightly (10 bytes) to ensure words aren't missed at thread boundaries
        long t_end = (t_id == t_count - 1) ? read_size : (t_id + 1) * t_chunk + 10;

        for (long i = t_start; i < t_end - 5 && i < read_size - 5; i++) {
            // Efficient single-character check before full match to save cycles
            if (buffer[i] == 'I') {
                if (buffer[i+1] == 'N' && buffer[i+2] == 'F' && buffer[i+3] == 'O') l_info++;
            }
            else if (buffer[i] == 'W') {
                if (buffer[i+1] == 'A' && buffer[i+2] == 'R' && buffer[i+3] == 'N') l_warn++;
            }
            else if (buffer[i] == 'E') {
                if (buffer[i+1] == 'R' && buffer[i+2] == 'R' && buffer[i+3] == 'O') l_error++;
            }
            else if (buffer[i] == 'F') {
                if (buffer[i+1] == 'A' && buffer[i+2] == 'T' && buffer[i+3] == 'A') l_fatal++;
            }
            else if (buffer[i] == 'R') {
                if (buffer[i+1] == 'E' && buffer[i+2] == 'P' && buffer[i+3] == 'L') l_repl++;
            }
        }
    }

    // Aggregate all local counts to Rank 0
    long g_info, g_warn, g_error, g_fatal, g_repl;
    MPI_Reduce(&l_info, &g_info, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&l_warn, &g_warn, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&l_error, &g_error, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&l_fatal, &g_fatal, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&l_repl, &g_repl, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("\n=== HYBRID LOG ANALYSIS RESULTS ===\n");
        printf("Time taken:    %f sec\n", end - start);
        printf("-----------------------------------\n");
        printf("INFO:     %ld\n", g_info);
        printf("WARN:     %ld\n", g_warn);
        printf("ERROR:    %ld\n", g_error);
        printf("FATAL:    %ld\n", g_fatal);
        printf("REPLICA:  %ld\n", g_repl);
        printf("-----------------------------------\n");
    }

    free(buffer);
    MPI_File_close(&fh);
    MPI_Finalize();
    return 0;
}
