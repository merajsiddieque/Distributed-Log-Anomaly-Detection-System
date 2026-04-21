#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#define MAX_LINE 1024
#define MAX_LEN 100
#define DIM 16
#define MAX_LINES 200000

// ================= MODEL =================
double weights[DIM];
double bias;

// ================= LOAD MODEL =================
void load_model() {
    FILE *fw = fopen("../model/fc_weight.txt", "r");
    if (!fw) {
        printf("Error loading fc_weight.txt\n");
        exit(1);
    }
    for (int i = 0; i < DIM; i++) {
        if (fscanf(fw, "%lf", &weights[i]) == EOF) break;
    }
    fclose(fw);

    FILE *fb = fopen("../model/fc_bias.txt", "r");
    if (!fb) {
        printf("Error loading fc_bias.txt\n");
        exit(1);
    }
    if (fscanf(fb, "%lf", &bias) == EOF) bias = 0;
    fclose(fb);
}

// ================= PARSER (Thread Safe) =================
void parse_line(const char* line, int* seq) {
    char buffer[MAX_LINE];
    strncpy(buffer, line, MAX_LINE);

    char *saveptr1, *saveptr2;
    // strtok_r is essential for OpenMP thread safety
    char *token = strtok_r(buffer, ",", &saveptr1); // Skip first column (e.g., ID)
    token = strtok_r(NULL, ",", &saveptr1);         // Get sequence column

    int i = 0;
    if (token) {
        char *num = strtok_r(token, " ", &saveptr2);
        while (num && i < MAX_LEN) {
            seq[i++] = atoi(num);
            num = strtok_r(NULL, " ", &saveptr2);
        }
    }
    while (i < MAX_LEN) seq[i++] = 0;
}

// ================= FEATURE =================
void extract_features(int *seq, double *feat) {
    for (int i = 0; i < DIM; i++) feat[i] = 0.0;
    for (int i = 0; i < MAX_LEN; i++) {
        feat[i % DIM] += (double)seq[i];
    }
    for (int i = 0; i < DIM; i++) {
        feat[i] /= MAX_LEN;
    }
}

// ================= PREDICT =================
int predict(int *seq) {
    double feat[DIM];
    extract_features(seq, feat);

    double score = bias;
    for (int i = 0; i < DIM; i++) {
        score += feat[i] * weights[i];
    }

    double prob = 1.0 / (1.0 + exp(-score));
    
    // Balanced threshold
    return (prob > 0.6) ? 1 : 0;
}

// ================= PROCESS FILE =================
int process_file(const char* filename, int label,
                 int rank, int size,
                 int *TP, int *FP, int *TN, int *FN) {

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        if (rank == 0) printf("Error opening %s\n", filename);
        return 0;
    }

    // Static allocation to avoid stack overflow, though heap is safer for scale
    static char lines[MAX_LINES][MAX_LINE];
    int total_lines = 0;

    while (total_lines < MAX_LINES && fgets(lines[total_lines], MAX_LINE, fp)) {
        total_lines++;
    }
    fclose(fp);

    // MPI Distribution logic
    int chunk = total_lines / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? total_lines : start + chunk;

    int local_TP=0, local_FP=0, local_TN=0, local_FN=0;

    #pragma omp parallel for reduction(+:local_TP,local_FP,local_TN,local_FN)
    for (int i = start; i < end; i++) {
        int seq[MAX_LEN];
        parse_line(lines[i], seq);
        int pred = predict(seq);

        if (pred == 1 && label == 1) local_TP++;
        else if (pred == 1 && label == 0) local_FP++;
        else if (pred == 0 && label == 0) local_TN++;
        else if (pred == 0 && label == 1) local_FN++;
    }

    *TP += local_TP;
    *FP += local_FP;
    *TN += local_TN;
    *FN += local_FN;

    return total_lines;
}

// ================= MAIN =================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    load_model();

    int TP=0, FP=0, TN=0, FN=0;
    double start_t = MPI_Wtime();

    // Process both classes
    process_file("../dataset_split/test/normal.txt", 0, rank, size, &TP, &FP, &TN, &FN);
    process_file("../dataset_split/test/abnormal.txt", 1, rank, size, &TP, &FP, &TN, &FN);

    int gTP, gFP, gTN, gFN;
    // Gather results from all ranks to Rank 0
    MPI_Reduce(&TP, &gTP, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&FP, &gFP, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&TN, &gTN, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&FN, &gFN, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_t = MPI_Wtime();

    if (rank == 0) {
        int total_samples = gTP + gFP + gTN + gFN;
        double acc = (total_samples > 0) ? (double)(gTP + gTN) / total_samples : 0;
        double prec = (gTP + gFP > 0) ? (double)gTP / (gTP + gFP) : 0;
        double rec = (gTP + gFN > 0) ? (double)gTP / (gTP + gFN) : 0;
        double f1 = (prec + rec > 0) ? 2 * (prec * rec) / (prec + rec) : 0;

        printf("\n=== HYBRID CLASSIFICATION RESULTS ===\n");
        printf("TP: %d, FP: %d, TN: %d, FN: %d\n", gTP, gFP, gTN, gFN);
        printf("Total Samples: %d\n", total_samples);
        printf("Time: %f sec\n", end_t - start_t);
        printf("Accuracy:  %.4f\n", acc);
        printf("Precision: %.4f\n", prec);
        printf("Recall:    %.4f\n", rec);
        printf("F1 Score:  %.4f\n", f1);
    }

    MPI_Finalize();
    return 0;
}
