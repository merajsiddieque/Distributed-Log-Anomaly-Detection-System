#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
        fscanf(fw, "%lf", &weights[i]);
    }
    fclose(fw);

    FILE *fb = fopen("../model/fc_bias.txt", "r");
    if (!fb) {
        printf("Error loading fc_bias.txt\n");
        exit(1);
    }

    fscanf(fb, "%lf", &bias);
    fclose(fb);
}

// ================= PARSER =================
void parse_line(const char* line, int* seq) {
    char buffer[MAX_LINE];
    strcpy(buffer, line);

    char *token = strtok(buffer, ",");
    token = strtok(NULL, ",");

    int i = 0;

    if (token) {
        char *num = strtok(token, " ");
        while (num && i < MAX_LEN) {
            seq[i++] = atoi(num);
            num = strtok(NULL, " ");
        }
    }

    while (i < MAX_LEN) seq[i++] = 0;
}

// ================= FEATURE =================
void extract_features(int *seq, double *feat) {
    for (int i = 0; i < DIM; i++) feat[i] = 0.0;

    for (int i = 0; i < MAX_LEN; i++) {
        feat[i % DIM] += seq[i];
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

    return (score > 0) ? 1 : 0;
}

// ================= PROCESS FILE =================
int process_file(const char* filename, int label,
                 int rank, int size,
                 int *TP, int *FP, int *TN, int *FN) {

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Error opening %s\n", filename);
        return 0;
    }

    static char lines[MAX_LINES][MAX_LINE];
    int total_lines = 0;

    while (fgets(lines[total_lines], MAX_LINE, fp)) {
        total_lines++;
        if (total_lines >= MAX_LINES) break;
    }

    fclose(fp);

    // MPI split
    int chunk = total_lines / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? total_lines : start + chunk;

    int local_TP=0, local_FP=0, local_TN=0, local_FN=0;

    // OpenMP parallel
    #pragma omp parallel for reduction(+:local_TP,local_FP,local_TN,local_FN)
    for (int i = start; i < end; i++) {

        int seq[MAX_LEN];
        parse_line(lines[i], seq);

        int pred = predict(seq);

        if (pred == 1 && label == 1) local_TP++;
        else if (pred == 1 && label == 0) local_FP++;
        else if (pred == 0 && label == 0) local_TN++;
        else local_FN++;
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
    char hostname[256];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(hostname, sizeof(hostname));

    printf("Rank %d running on %s\n", rank, hostname);

    load_model();

    int TP=0, FP=0, TN=0, FN=0;

    double start = MPI_Wtime();

    int total = 0;

    // USE TEST SPLIT
    total += process_file("../dataset_split/test/normal.txt", 0,
                          rank, size, &TP, &FP, &TN, &FN);

    total += process_file("../dataset_split/test/abnormal.txt", 1,
                          rank, size, &TP, &FP, &TN, &FN);

    int gTP, gFP, gTN, gFN;

    MPI_Reduce(&TP, &gTP, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&FP, &gFP, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&TN, &gTN, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&FN, &gFN, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {

        int total_samples = gTP + gFP + gTN + gFN;

        double acc = (double)(gTP + gTN) / total_samples;
        double prec = (gTP + gFP) ? (double)gTP / (gTP + gFP) : 0;
        double rec = (gTP + gFN) ? (double)gTP / (gTP + gFN) : 0;
        double f1 = (prec + rec) ? 2 * prec * rec / (prec + rec) : 0;

        printf("\n=== HYBRID TRANSFORMER RESULTS ===\n");
        printf("Total Samples: %d\n", total_samples);
        printf("Time: %f sec\n", end - start);
        printf("Accuracy: %f\n", acc);
        printf("Precision: %f\n", prec);
        printf("Recall: %f\n", rec);
        printf("F1 Score: %f\n", f1);
    }

    MPI_Finalize();
    return 0;
}
