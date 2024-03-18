#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <arm_neon.h>

#define MATRIX_SIZE 1000

double matrix[MATRIX_SIZE][MATRIX_SIZE];
double vector[MATRIX_SIZE];
double result[MATRIX_SIZE];

void initialize()
{
    // Initialize matrix and vector with random values
    srand(time(NULL));
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
        vector[i] = (double)rand() / RAND_MAX;
    }
}

#include <arm_neon.h>

void matrix_vector_multiply()
{
    for (int i = 0; i < MATRIX_SIZE; i += 2)
    {
        // row i and i+1
        float64x2_t result_vector1 = vdupq_n_f64(0.0); 
        float64x2_t result_vector2 = vdupq_n_f64(0.0); 
        
        for (int j = 0; j < MATRIX_SIZE; j += 2)
        {
            float64x2x2_t matrix_values = vld2q_f64(&matrix[i][j]); // load 2 matrix elements
            
            float64x2_t vector_values = vld1q_f64(&vector[j]); // load 2 vector elements

            // multiply and accumulate
            result_vector1 = vmlaq_f64(result_vector1, matrix_values.val[0], vector_values);
            result_vector2 = vmlaq_f64(result_vector2, matrix_values.val[1], vector_values);
        }

        float64x2_t sum1 = vpaddq_f64(result_vector1, result_vector1);
        float64x2_t sum2 = vpaddq_f64(result_vector2, result_vector2);

        result[i] = vgetq_lane_f64(sum1, 0);
        result[i+1] = vgetq_lane_f64(sum2, 0);
    }
}

long long current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000LL + (long long)tv.tv_usec;
}

int main()
{
    initialize();

    long long start_time = current_time();
    matrix_vector_multiply();
    long long end_time = current_time();
    double elapsed_time = (end_time - start_time) / 1000000.0; // Convert to seconds

    // Print time taken
    printf("Time taken: %f seconds\n", elapsed_time);

    // Calculate total floating point operations
    long long flops = 2 * MATRIX_SIZE * MATRIX_SIZE;

    // Calculate GFLOPS
    double gflops = flops / (elapsed_time * 1e9);
    printf("GFLOPS: %f\n", gflops);

    return 0;
}