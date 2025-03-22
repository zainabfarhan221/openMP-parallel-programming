#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define MAX_SIZE 1024
#define NUM_RUNS 10
#define CHUNK_SIZE 200
 // Define the chunk size for static scheduling

void initialize_matrix(double A[MAX_SIZE][MAX_SIZE], int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (rand() % 10) + 1;
}

double determinant_parallel_static(double A[MAX_SIZE][MAX_SIZE], int N) {
    double det = 1.0;
    for (int i = 0; i < N; i++) {
        if (A[i][i] == 0) {
            for (int j = i + 1; j < N; j++) {
                if (A[j][i] != 0) {
                    #pragma omp parallel for schedule(static, CHUNK_SIZE)
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

        #pragma omp parallel for schedule(static, CHUNK_SIZE) shared(A)
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
    int N = 2000;
    int num_threads[] = {4, 8};
    
    for (int t = 0; t < 2; t++) {
        omp_set_num_threads(num_threads[t]);
        printf("\n--- Static Scheduling with %d Threads (Chunk Size: %d) ---\n", num_threads[t], CHUNK_SIZE);
        printf("\nMatrix Size: %d x %d\n", N, N);
        double total_time = 0.0;
        double determinant = 0.0;
        
        for (int run = 0; run < NUM_RUNS; run++) {
            static double A[MAX_SIZE][MAX_SIZE];
            initialize_matrix(A, N);
            double start = omp_get_wtime();
            determinant = determinant_parallel_static(A, N);
            double end = omp_get_wtime();
            double exec_time = end - start;
            total_time += exec_time;
            printf("Run %d Execution Time: %.6f sec\n", run + 1, exec_time);
        }
        
        double avg_time = total_time / NUM_RUNS;
        printf("Average Execution Time: %.6f sec\n", avg_time);
        printf("Determinant (last run): %.6f\n", determinant);
    }
    return 0;
}
