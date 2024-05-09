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

        #define TILE_WIDTH 32
        __global__ void convolution2D(float *img_d, float *kernel_d, float* result_d, int n)
        {
                __shared__ float img_s[TILE_WIDTH][TILE_WIDTH];
                __shared__ float kernel_s[3][3];

                int tx = threadIdx.x, ty = threadIdx.y;
                int col = blockIdx.x * blockDim.x + tx;
                int row = blockIdx.y * blockDim.y + ty;
                if (row < n && col < n)
                {
                        if (tx < 3 && ty < 3)
                                kernel_s[tx][ty] = kernel_d[tx][ty];
                        img_s[tx][ty] = img_d[row][col];
                        __syncthreads();
                        float sum = 0.0;
                        for(int di = -1; di <= 1; di++)
                        {
                                for(int dj = -1; dj <= 1; dj++) 
                                {
                                        int ni = ty + di, nj = tx + dj;
                                        if(ni >= 0 and ni < TILE_WIDTH and nj >= 0 and nj < TILE_WIDTH) 
                                        {
                                                sum += kernel_s[di+1][dj+1] * img_s[ni][nj];
                                        }
                                        else if(ni >= 0 and ni < n and nj >= 0 and nj < n) 
                                        {
                                                sum += kernel_s[di+1][dj+1] * img_d[(row+di) * n + (col+dj)];
                                        }
                                }
                        }
                        result_d[row*n+col] = sum;
                        // result_d[row*n+col] = img_d[row*n+col];
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
                cudaSetDevice(0);

                float *img_d, *kernel_d, *result_d;
                CUDA_ERROR_CHECK(cudaMalloc((void**)&img_d, size * sizeof(float)));
                CUDA_ERROR_CHECK(cudaMemcpy(img_d, img, size * sizeof(float), cudaMemcpyHostToDevice));

                CUDA_ERROR_CHECK(cudaMalloc((void**)&kernel_d, 9 * sizeof(float)));
                CUDA_ERROR_CHECK(cudaMemcpy(kernel_d, kernel_flat, 9 * sizeof(float), cudaMemcpyHostToDevice));

                CUDA_ERROR_CHECK(cudaMalloc((void **)&result_d, size * sizeof(float)));

                dim3 DimGrid(num_rows / TILE_WIDTH, num_cols / TILE_WIDTH, 1);
                dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
                convolution2D<<<DimGrid, DimBlock>>>(img_d, kernel_d, result_d, num_cols);
                
                cudaDeviceSynchronize();
                
                CUDA_ERROR_CHECK(cudaMemcpy(result, result_d, num_rows * num_cols * sizeof(float), cudaMemcpyDeviceToHost));

                return sol_path;
        }
};