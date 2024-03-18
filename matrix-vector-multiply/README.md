## Device Specifications
```
Chip: Apple M1 Pro
Total Number of Cores:	10 (8 performance + 2 efficiency)
Clock Speed: 3.2 GHz (performance cores)
Target: aarch64-apple-darwin23
Device architecture: ARM
Vectorization Intrinsic: NEON
Cache line size: 128 Bytes
```

## File Structure
```
.
├── Makefile
├── README.md
├── mvm_bw.c (Bandwidth calculations)
├── mvm_omp_v1.c (OpemMP parallelization)
├── mvm_omp_v2.c (OmenMP parallelization + SIMD)
├── mvm_omp_vec.c (OmenMP parallelization + NEON intrinsics)
├── mvm_original.c (Original code)
├── mvm_vec_1.c (NEON instrinsics on inner loop only)
├── mvm_vec_2.c (NEON instrinsics on both loops)
└── stream.c (origina stream test)
```

## Original Code
Running the original code as given in the repo `mvm_original.c` without any changes and modifying only compiler flags
| Flags    | Description                           | Avg GFLOPS | Max GFLOPS | Avg Time   |
| -------- | ------------------------------------- | ---------- | ---------- | ---------- |
| `-O0`    | Default - No optimization             | 0.568001   | 0.584966   | 0.00353274 |
| `-O1`    | Optimize Compilation                  | 1.714098   | 1.745201   | 0.00120984 |
| `-O2`    | More Optimization                     | 2.334087   | 2.412545   | 0.00090219 |
| `-O3`    | Even More Optimization                | 2.355227   | 2.424242   | 0.00088684 |
| `-Os`    | Optimize for space                    | 2.341965   | 2.415459   | 0.00089244 |
| `-Ofast` | Disregard strict standards compliance | 3.676732   | 3.694222   | 0.00054234 |
- `-O2` and `-O3` give very similar results.
- `-Ofast` enables optimizations that are not valid for all standard compliant programs like `-ffast-math` giving even significant performance
- Speed up compared to `-O0` is `6.47x`

### Additional Flags
- `-O3` enables flags most other optimization flags like `-ffast-math` (faster math operations) and `-ftree-vectorize` (auto vectorization) by default. Manually adding similar flags including `-funroll-loops` (unroll loops to increase performance) gives the following results:
  ```
    Average time: 0.0004994 seconds
    Average GFLOPS: 4.02303
    Maximum GFLOPS: 4.184100
  ```
- `-funsafe-math-optimizations` is another interesting flag that is equivalent to `-fassociative-math`, `-freciprocal-math`, `-fno-signed-zeros`, and `-fno-trapping-math` whcih allows arbitrary reassociations and transformations with no accuracy guarantees. The `-Ofast` flag enables this by default and we notice a slight boost in performance compared to the previous result:
  ```
    Average time: 0.00049748 seconds
    Average GFLOPS: 4.0301
    Maximum GFLOPS: 4.192872
  ```
- Adding CPU specific flags like `-march=native -mtune=native` allows the compiler to generate code that uses all features of the native CPU. As I'm running on an ARM Processor, the compiler will automatically use flags such as `-mfpu=neon` for auto vectorization and such optimizations. This gives us very similar performance as the previous example because in clang builds for the native architecture by default.
  ```
    Average time: 0.00049096 seconds
    Average GFLOPS: 4.0893
    Maximum GFLOPS: 4.210526
  ```
- In all these cases, speed up compared to `-O0` is `7.2x`


## OpenMP

Modified original code using OpenMP directives only. Running flags: `-O3 -fopenmp`

### Parallelization
C Programs stores data in a matrix in row major form, therefore accessing a row in a matrix is faster than accessing a column. We utilize this fact by having each thread work on one row of the matrix along with the vector to compute one element of the result by multiplication and addition stored in local variable and then written to the final result. Total operations are `2*size*size`. However, result becomes a shared variable between multiple threads running on different cores. This causes updation to be a bottleneck as there is a notion of false sharing (when one thread modifies, the entire cache line gets invalidated, forcing other threads to reload data they weren't even modifying) between of the result variable between the cores when contiguous memory locations have to be updated. 

The `CACHE_LINE_SIZE` in the system is `128 Bytes`. So each thread should update `128/8 = 16` contiguous locations in the result. Thus the `schedule(static, 16)` clause has been used.

Code:
```
#pragma omp parallel for schedule(static, 16)
for (int i = 0; i < MATRIX_SIZE; i++)
{
    double local_result = 0.0;
    for (int j = 0; j < MATRIX_SIZE; j++)
    {
        local_result += matrix[i][j] * vector[j];
    }
    result[i] = local_result;
}
```
Results:
| Threads | Avg GFLOPS | Max GFLOPS |
| ------- | ---------- | ---------- |
| 1       | 1.692690   | 1.742160   |
| 2       | 2.982940   | 3.149606   |
| 3       | 3.999210   | 4.184100   |
| 4       | 4.635907   | 5.347594   |
| 5       | 4.627897   | 5.305040   |
| 6       | 4.767428   | 5.420054   |
| 7       | 5.167703   | 6.153846   |
| 8       | 5.232895   | 6.254346   |
| 9       | 4.495553   | 5.102041   |
| 10      | 4.416029   | 5.194805   |

The peak performance is at `8 threads`, giving a maximum speed up of about `9.2x` at `6.254346 GFLOPS`

### SIMD

OpenMP supports SIMD vectorization as well using the `#pragma omp simd` clause which can be applied in the calculation of `local_result` in the previous code.
```
#pragma omp parallel for schedule(static, 16)
for (int i = 0; i < MATRIX_SIZE; i++)
{
    double local_result = 0.0;
#pragma omp simd reduction(+ : local_result)
    for (int j = 0; j < MATRIX_SIZE; j++)
    {
        local_result += matrix[i][j] * vector[j];
    }
    result[i] = local_result;
}
```

Results:
| Threads | Avg GFLOPS | Max GFLOPS |
| ------- | ---------- | ---------- |
| 1       | 3.612008   | 3.731343   |
| 2       | 5.541030   | 5.847953   |
| 3       | 7.029361   | 7.604563   |
| 4       | 7.647644   | 9.090909   |
| 5       | 7.107571   | 9.049774   |
| 6       | 6.117851   | 7.434944   |
| 7       | 6.317678   | 7.751938   |
| 8       | 6.482675   | 7.662835   |
| 9       | 5.804964   | 6.756757   |
| 10      | 5.825912   | 6.644518   |

The performance peaks at 4 threads with `9.090909 GFLOPS` which is a `13.46x` speed up than the original code with no optimization. The performance then slightly reduces till 6-7 threads. At 8-10 threads, the performance somewhat decreases as there are 2 efficiency cores on the system instead of performance. These efficiency cores become be bottleneck, slowing down the performance. The reason for peak performance at 4 threads is possibly because of writing to the result array which reamains cached in once core and the other cores have to fetch again giving the notion of `false sharing`.

### Compiler Flag Optimizations
Running the same codes with `-Ofast -march=native -mtune=native -Ofast -fopenmp -ftree-vectorize  -funroll-loops -ffast-math -funsafe-math-optimizations` the results with 4 threads are:
- Only parallelization:
  ```
    Average time: 0.00027032 seconds
    Average GFLOPS: 7.6752
    Maximum GFLOPS: 9.216590
  ```
- Parallelization + SIMD:
  ```
    Average time: 0.00025374 seconds
    Average GFLOPS: 8.00529
    Maximum GFLOPS: 9.389671
  ```
The best performance boost achieved by OpenMP is about `14.3x` speedup than non parallelized code with `-O0` flag.

OpenMP parallelization with `-O0` flag gives the result:
```
Average time: 0.0034323 seconds
Average GFLOPS: 0.583231
Maximum GFLOPS: 0.592242
```
Which is not very significant compared to non parallelized code.

## NEON Vectorization
NEON Vectorization Instrinsics have been implemented as the underlying processer is ARM based. 

### Inner Loop Vectorization
The calculation of an element of the result vector consists of repeaded multiplicationa and addition. To optimize this, we can multiply and accumulate two elements simultaneously using a vector register thereby doubling the speed. At max, we can store only 2 double precision floating point numbers in a vector register of type `float64x2_t`. 
```
for (int i = 0; i < MATRIX_SIZE; i++)
{
    float64x2_t result_vector = vdupq_n_f64(0.0); // result vector
    for (int j = 0; j < MATRIX_SIZE; j += 2)
    {
        float64x2_t matrix_values = vld1q_f64(&matrix[i][j]); // load 2 matrix elements
        float64x2_t vector_values = vld1q_f64(&vector[j]);   // load 2 vector elements
        result_vector = vmlaq_f64(result_vector, matrix_values, vector_values); // multiply and accumulate
    }
    float64x2_t sum = vpaddq_f64(result_vector, result_vector);
    result[i] = vgetq_lane_f64(sum, 0); // Store the result
}
```
On running with the compilation flags `-O3 -march=native -mtune=native -funroll-loops -ffast-math -funsafe-math-optimizations`, the results are:
```
Average time: 0.00042336 seconds
Average GFLOPS: 4.78688
Maximum GFLOPS: 5.115090
```
This vectorization is computing the same elements in half the instructions giving almost 2x speedup than without any vectorization. Speedup is `8.4x` the original code with no optimization.

### Both Loop Vectorization
We can also vectorize the outer loop computing two elements of the result array simultaneously. The inner loop remains the same as above but updates both the sum values in the register.
```
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
```
The results obtained are:
```
Average time: 0.0002435 seconds
Average GFLOPS: 8.32993
Maximum GFLOPS: 8.810573
```
The doubling of GFLOPS is as expected as we compute the same number of values in half the instructions where each instruction manipulates 2 data values (SIMD). Speedup is `15.51x` the original code with no optimization.

### Base vectorization
Running the optimized vectorization code with `-O0` flag gives the following results:
```
Average time: 0.00172954 seconds
Average GFLOPS: 1.16681
Maximum GFLOPS: 1.253918
```
THe vectorized code with optimized flags `-O3 -march=native -mtune=native -funroll-loops -ffast-math -funsafe-math-optimizations` gives `8.32993 GFLOPS` on average which is about `7.17x` speedup comared to base vectorization code.


## Vectorization + Multithreading
Combining the idea of multithreading and vectorization using intrinsics, we can use OpenMP to spawn threads which compute two elements of the result vector simultaneously giving us a performance boost.
```
#pragma omp parallel for schedule(static, 16)
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
    result[i + 1] = vgetq_lane_f64(sum2, 0);
}
```
The performance results based on different thread counts are:
| Threads | Avg GFLOPS | Max GFLOPS |
| ------- | ---------- | ---------- |
| 1       | 8.140103   | 8.658009   |
| 2       | 9.636027   | 10.471204  |
| 3       | 11.567496  | 12.987013  |
| 4       | 11.748506  | 15.151515  |
| 5       | 9.283410   | 12.121212  |
| 6       | 7.610653   | 9.174312   |
| 7       | 7.597328   | 9.090909   |
| 8       | 7.496056   | 9.803922   |
| 9       | 7.298581   | 8.771930   |
| 10      | 6.942242   | 8.547009   |

Best result with 4 threads:
```
Average time: 0.00014492 seconds
Average GFLOPS: 14.0893
Maximum GFLOPS: 15.503876
```
We again notice the peak performance with 4 threads (`25.15x` speedup from orignal code with no optimizations) after which performance reduces. The reasons are cache effects (the same reason for only parallelization) and also instruction level parallelism (ILP). The use of vectorization intrinsics exploits ILP by performing multiple operations simultaneously on vector data. However, as the number of threads increases, the amount of ILP available per thread may decrease, as each thread is processing a smaller portion of the data.

With only `-O0 -fopenmp` flags, the results are:
```
Average time: 0.00057414 seconds
Average GFLOPS: 3.51326
Maximum GFLOPS: 3.853565
```

## Bandwidth Calculations
The code with read and write operations for each line is given as follows (r - read, w - write)
```
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
```
The operations that involve the memory are:
- Read operations: `(MATRIX_SIZE / 2) * (2 + 7 * MATRIX_SIZE)` 
- Write operations: `(MATRIX_SIZE / 2) * (6 + 2 * MATRIX_SIZE)`

The CPU bound floating point operations are: `2 * MATRIX_SIZE * MATRIX_SIZE;` (ignoring the redundant calculation of sum in a vector for both elements which does not affect the value significantly anyways)

On running the above program with 4 threads, we get the following results:
```
Average time: 0.00015742 seconds
Average GFLOPS: 13.3086
Maximum GFLOPS: 15.873016
Avg Read Bandwidth: 173.574 GB/s
Avg Write Bandwidth: 49.7272 GB/s
```
- The average GFLOPS as `13.3` implies that each second `13.3` floating point computations are done by the CPU.
- Read bandwidth of `173.574 GB/s` implies that `21.7` double values can be read from the memory in a second.
- Write bandwidth of `49.72 GB/s` implies that `6.2` double values can be written to the memory in a second.
  
When `MATRIX_SIZE = 1000`:
- Number of read operations: `3501000`
- Number of write operations: `1003000`
- Number of CPU operations: `2000000`

Therefore, in the same timeframe, for `every 2 CPU operations`, there are `3.5 read operations` and `1 write operation` that takes place. We notice that the max GFLOPS (where operation is reading/writing) will be around `21.7*1.5=32.55` for read and `6.2*2=12.4` for write.

Since our calculated results are around `13 GLFOPS`, we can conclude that writing to memory (more appropriately - cache) operation serves as the bottleneck in the entire process. The program is thus `memory bound by write operations`

This also explains why we did not get a significant performance boost on increasing threads from 4 to 8. Even through calculations were done faster, writing back the result to memory served as the bottleneck.

## Summary
| Optimization             | GFLOPS   | Time (sec) | Speedup |
| ------------------------ | -------- | ---------- | ------- |
| -O0                      | 0.568001 | 0.00353274 | 1x      |
| -O3                      | 2.355227 | 0.00088684 | 4.2x    |
| Fancy GCC Compiler flags | 4.0893   | 0.00049096 | 7.2x    |
| OpenMP (4 threads)       | 8.00529  | 0.00025374 | 14.3x   |
| NEON Vectorization       | 8.32993  | 0.0002435  | 15.51x  |
| OpenMP + NEON            | 14.0893  | 0.00014492 | 25.15x  |

```
Read Bandwidth: 173.574 GB/s
Write Bandwidth: 49.72 GB/s
```
```
Bottleneck: Write to memory
Program is memory bound
```
## Resources Used
- https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
- https://developer.arm.com/documentation/den0018/a/NEON-Intrinsics/Using-NEON-intrinsics
  
