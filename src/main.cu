#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

namespace solution
{
#define CUDA_ERROR_CHECK(ans)                          \
        {                                              \
                cudaAssert((ans), __FILE__, __LINE__); \
        }
        inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
        {
                if (code != cudaSuccess)
                {
                        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
                        if (abort)
                                exit(code);
                }
        }

        __global__ void convolution2D(float *img_d, float *kernel_d, float* result_d, int n, int start_row)
        {
                int col = blockIdx.x * blockDim.x + threadIdx.x;
                int row = blockIdx.y * blockDim.y + threadIdx.y + start_row;
                if (row < n && col < n)
                {
                        float sum = 0.0;
                        for(int di = -1; di <= 1; di++)
                        {
                                for(int dj = -1; dj <= 1; dj++) 
                                {
                                        int ni = row + di, nj = col + dj;
                                        if(ni >= 0 and ni < n and nj >= 0 and nj < n) 
                                        {
                                                sum += kernel_d[(di+1)*3 + dj+1] * img_d[ni * n + nj];
                                        }
                                }
                        }
                        result_d[(row-start_row)*n+col] = sum;
                }
        }

        std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols)
        {
                std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";

                int size = num_rows * num_cols;

                int bitmap_fd = open(bitmap_path.c_str(), O_RDONLY);
                float *img = static_cast<float *>(mmap(NULL, sizeof(float) * size, PROT_READ, MAP_PRIVATE, bitmap_fd, 0));

                int result_fd = open(sol_path.c_str(), O_CREAT | O_RDWR, 0644);
                ftruncate(result_fd, sizeof(float) * size);
                float *result = reinterpret_cast<float *>(mmap(NULL, sizeof(float) * size, PROT_WRITE | PROT_READ, MAP_SHARED, result_fd, 0));

                float kernel_flat[9];
                for (int i = 0; i < 9; i++)
                {
                        kernel_flat[i] = kernel[i/3][i%3];
                }

                const int num_gpus = 1;
                const int rows_per_gpu = num_rows / num_gpus;

                #pragma omp parallel for num_threads(num_gpus)
                for (int i = 0; i < num_gpus; ++i)
                {
                        int gpu_id = i;

                        cudaSetDevice(gpu_id);

                        int start_row = gpu_id * rows_per_gpu;

                        float *img_d, *kernel_d, *result_d;
                        CUDA_ERROR_CHECK(cudaMalloc((void**)&img_d, size * sizeof(float)));
                        CUDA_ERROR_CHECK(cudaMemcpy(img_d, img, size * sizeof(float), cudaMemcpyHostToDevice));

                        CUDA_ERROR_CHECK(cudaMalloc((void**)&kernel_d, 9 * sizeof(float)));
                        CUDA_ERROR_CHECK(cudaMemcpy(kernel_d, kernel_flat, 9 * sizeof(float), cudaMemcpyHostToDevice));

                        CUDA_ERROR_CHECK(cudaMalloc((void **)&result_d, num_cols*rows_per_gpu * sizeof(float)));

                        dim3 DimGrid(rows_per_gpu / 8, num_cols / 8, 1);
                        dim3 DimBlock(8, 8, 1);
                        convolution2D<<<DimGrid, DimBlock>>>(img_d, kernel_d, result_d, num_cols, start_row);
                        
                        cudaDeviceSynchronize();
                        
                        CUDA_ERROR_CHECK(cudaMemcpy(result + start_row * num_cols, result_d, rows_per_gpu * num_cols * sizeof(float), cudaMemcpyDeviceToHost));

                        CUDA_ERROR_CHECK(cudaFree(img_d));
                        CUDA_ERROR_CHECK(cudaFree(kernel_d));
                        CUDA_ERROR_CHECK(cudaFree(result_d));
                }

                return sol_path;
        }
};