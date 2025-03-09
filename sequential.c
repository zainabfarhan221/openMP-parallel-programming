#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_SIZE 2000  // Adjust based on memory constraints

// Function to initialize matrix with random values
void initialize_matrix(double A[MAX_SIZE][MAX_SIZE], int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (rand() % 10) + 1;
}

// Function to calculate determinant using Gaussian Elimination (Sequential)
double determinant_sequential(double A[MAX_SIZE][MAX_SIZE], int N) {
    double det = 1.0;

    for (int i = 0; i < N; i++) {
        // Pivoting to avoid division by zero
        if (A[i][i] == 0) {
            for (int j = i + 1; j < N; j++) {
                if (A[j][i] != 0) {
                    // Swap rows
                    for (int k = 0; k < N; k++) {
                        double temp = A[i][k];
                        A[i][k] = A[j][k];
                        A[j][k] = temp;
                    }
                    det *= -1; // Row swap changes determinant sign
                    break;
                }
            }
        }

        if (A[i][i] == 0)
            return 0; // Singular matrix

        // Row elimination (Sequential)
        for (int j = i + 1; j < N; j++) {
            double factor = A[j][i] / A[i][i];
            for (int k = i; k < N; k++) {
                A[j][k] -= factor * A[i][k];
            }
        }

        det *= A[i][i];
    }

    return det;
}

int main() {
    int sizes[] = {512, 1024, 2000};  // Test different matrix sizes

    for (int s = 0; s < 3; s++) {
        int N = sizes[s];
        printf("\nMatrix Size: %d x %d\n", N, N);

        static double A[MAX_SIZE][MAX_SIZE];
        initialize_matrix(A, N); // Reset matrix

        clock_t start = clock(); // Start time
        double determinant = determinant_sequential(A, N);
        clock_t end = clock(); // End time

        double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC; // Convert to seconds

        printf("Determinant: %.6f\n", determinant);
        printf("Execution Time: %.6f sec\n", elapsed_time);
    }

    return 0;
}