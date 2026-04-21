#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

// ================= PARSER =================
void parse_line(const char* line, int* seq) {
    char buffer[MAX_LINE];
    strncpy(buffer, line, MAX_LINE);

    char *saveptr1, *saveptr2;
    char *token = strtok_r(buffer, ",", &saveptr1);
    token = strtok_r(NULL, ",", &saveptr1);

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

    return (prob > 0.6) ? 1 : 0;
}

// ================= PROCESS FILE =================
int process_file(const char* filename, int label,
                 int *TP, int *FP, int *TN, int *FN) {

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Error opening %s\n", filename);
        return 0;
    }

    static char lines[MAX_LINES][MAX_LINE];
    int total_lines = 0;

    while (total_lines < MAX_LINES && fgets(lines[total_lines], MAX_LINE, fp)) {
        total_lines++;
    }
    fclose(fp);

    int local_TP=0, local_FP=0, local_TN=0, local_FN=0;

    #pragma omp parallel for reduction(+:local_TP,local_FP,local_TN,local_FN)
    for (int i = 0; i < total_lines; i++) {

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
int main() {

    load_model();

    int TP=0, FP=0, TN=0, FN=0;

    double start = omp_get_wtime();

    process_file("../dataset_split/test/normal.txt", 0, &TP, &FP, &TN, &FN);
    process_file("../dataset_split/test/abnormal.txt", 1, &TP, &FP, &TN, &FN);

    double end = omp_get_wtime();

    int total = TP + FP + TN + FN;

    double acc = (double)(TP + TN) / total;
    double prec = (TP + FP) ? (double)TP / (TP + FP) : 0;
    double rec = (TP + FN) ? (double)TP / (TP + FN) : 0;
    double f1 = (prec + rec) ? 2 * prec * rec / (prec + rec) : 0;

    printf("\n=== ONLY OPENMP RESULTS ===\n");
    printf("Total Samples: %d\n", total);
    printf("Time: %f sec\n", end - start);
    printf("Accuracy:  %.4f\n", acc);
    printf("Precision: %.4f\n", prec);
    printf("Recall:    %.4f\n", rec);
    printf("F1 Score:  %.4f\n", f1);

    return 0;
}
