#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <arm_neon.h>

#define MATRIX_SIZE 1000

double matrix[MATRIX_SIZE][MATRIX_SIZE];
double vector[MATRIX_SIZE];
double result[MATRIX_SIZE];
double readtime = 0.0;
double optime = 0.0;
double writetime = 0.0;

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

void matrix_vector_multiply()
{
#pragma omp parallel for schedule(static, 16)
    for (int i = 0; i < MATRIX_SIZE; i += 2)
    {
        float64x2_t result_vector1 = vdupq_n_f64(0.0); // 2w + 0r
        float64x2_t result_vector2 = vdupq_n_f64(0.0); // 2w + 0r

        for (int j = 0; j < MATRIX_SIZE; j += 2)
        {
            float64x2x2_t matrix_values = vld2q_f64(&matrix[i][j]); // 4r + 0w
            float64x2_t vector_values = vld1q_f64(&vector[j]); // 2r + 0w

            result_vector1 = vmlaq_f64(result_vector1, matrix_values.val[0], vector_values); // 4r + 2w
            result_vector2 = vmlaq_f64(result_vector2, matrix_values.val[1], vector_values); // 4r + 2w
        }

        float64x2_t sum1 = vpaddq_f64(result_vector1, result_vector1);
        float64x2_t sum2 = vpaddq_f64(result_vector2, result_vector2);

        result[i] = vgetq_lane_f64(sum1, 0); // 1r + 1w
        result[i + 1] = vgetq_lane_f64(sum2, 0); // 1r + 1w
    }
}

long long current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000LL + (long long)tv.tv_usec;
}

void calc_read_bw()
{
    long long start_time = current_time();

    double sum = 0.0;
    // #pragma omp parallel for schedule(static, 16)
    for (int i = 0; i < MATRIX_SIZE; i += 2)
    {
        for (int j = 0; j < MATRIX_SIZE; j += 2)
        {
            float64x2x2_t matrix_values = vld2q_f64(&matrix[i][j]); // load 2 matrix elements

            float64x2_t vector_values = vld1q_f64(&vector[j]); // load 2 vector elements
            // sum+=(i+j);

            sum += vgetq_lane_f64(matrix_values.val[0], 0) + vgetq_lane_f64(matrix_values.val[0], 1) + vgetq_lane_f64(matrix_values.val[1], 0) + vgetq_lane_f64(matrix_values.val[1], 1) + vgetq_lane_f64(vector_values, 0) + vgetq_lane_f64(vector_values, 1);
        }
    }

    long long end_time = current_time();
    printf("sum = %f\n", sum); // dummy
    double read_time = (end_time - start_time) / 1000000.0;
    printf("Read Time: %f seconds\n", read_time);
    double read_bw = (MATRIX_SIZE * MATRIX_SIZE * 6 / 4) * sizeof(double) / (read_time * 1024 * 1024 * 1024); // GB/s
    printf("Estimated Read Bandwidth: %.2f GB/s\n", read_bw);
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

    // bandwidth calculation
    double read_bw = (MATRIX_SIZE / 2) * (2 + 7 * MATRIX_SIZE) * sizeof(double) / (elapsed_time * 1024 * 1024 * 1024);

    printf("Read Bandwidth: %.2f GB/s\n", read_bw);
    double write_bw = (MATRIX_SIZE / 2) * (6 + 2 * MATRIX_SIZE) * sizeof(double) / (elapsed_time * 1024 * 1024 * 1024); 

    printf("Write Bandwidth: %.2f GB/s\n", write_bw);

    return 0;
}
