#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void count_words(const char* filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Could not open %s\n", filename);
        return;
    }

    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);
    fclose(fp);

    // Read the entire file into memory for faster processing
    char *buffer = malloc(filesize + 1);
    fp = fopen(filename, "rb");
    fread(buffer, 1, filesize, fp);
    buffer[filesize] = '\0';
    fclose(fp);

    double start = omp_get_wtime();

    // Counters for common log words
    long l_info=0, l_warn=0, l_error=0, l_fatal=0, l_repl=0;

    #pragma omp parallel reduction(+:l_info, l_warn, l_error, l_fatal, l_repl)
    {
        int t_id = omp_get_thread_num();
        int t_count = omp_get_num_threads();
        long t_chunk = filesize / t_count;
        long t_start = t_id * t_chunk;
        
        // Overlap of 10 bytes to ensure words aren't cut at thread boundaries
        long t_end = (t_id == t_count - 1) ? filesize : (t_id + 1) * t_chunk + 10;

        for (long i = t_start; i < t_end - 5 && i < filesize - 5; i++) {
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

    double end = omp_get_wtime();

    printf("\n=== OPENMP LOG ANALYSIS RESULTS ===\n");
    printf("Time taken:    %f sec\n", end - start);
    printf("-----------------------------------\n");
    printf("INFO:     %ld\n", l_info);
    printf("WARN:     %ld\n", l_warn);
    printf("ERROR:    %ld\n", l_error);
    printf("FATAL:    %ld\n", l_fatal);
    printf("REPLICA:  %ld\n", l_repl);
    printf("-----------------------------------\n");

    free(buffer);
}

int main() {
    count_words("HDFS.log");
    return 0;
}
