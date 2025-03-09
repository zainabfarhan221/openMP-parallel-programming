#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_SIZE 2000  

void initialize_matrix(double A[MAX_SIZE][MAX_SIZE], int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (rand() % 10) + 1;
}

double determinant_parallel_dynamic(double A[MAX_SIZE][MAX_SIZE], int N) {
    double det = 1.0;
    for (int i = 0; i < N; i++) {
        if (A[i][i] == 0) {
            for (int j = i + 1; j < N; j++) {
                if (A[j][i] != 0) {
                    #pragma omp parallel for
                    for (int k = 0; k < N; k++) {
                        double temp = A[i][k];
                        A[i][k] = A[j][k];
                        A[j][k] = temp;
                    }
                    #pragma omp atomic
                    det *= -1;
                    break;
                }
            }
        }
        if (A[i][i] == 0)
            return 0;
        #pragma omp parallel for schedule(dynamic) shared(A)
        for (int j = i + 1; j < N; j++) {
            double factor = A[j][i] / A[i][i];
            for (int k = i; k < N; k++) {
                A[j][k] -= factor * A[i][k];
            }
        }
        #pragma omp critical
        det *= A[i][i];
    }
    return det;
}

int main() {
    int sizes[] = {512, 1024, 2000};
    int num_threads[] = {4, 8};
    
    for (int t = 0; t < 2; t++) {
        omp_set_num_threads(num_threads[t]);
        printf("\n--- Dynamic Scheduling with %d Threads ---\n", num_threads[t]);
        for (int s = 0; s < 3; s++) {
            int N = sizes[s];
            printf("\nMatrix Size: %d x %d\n", N, N);
            static double A[MAX_SIZE][MAX_SIZE];
            initialize_matrix(A, N);
            double start = omp_get_wtime();
            double det = determinant_parallel_dynamic(A, N);
            double end = omp_get_wtime();
            printf("Execution Time: %.6f sec\n", end - start);
            printf("Determinant: %.6f\n", det);
        }
    }
    return 0;
}