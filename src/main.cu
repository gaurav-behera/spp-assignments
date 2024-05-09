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

        __global__ void convolution2D(float *img_d, float *kernel_d, float* result_d, int n)
        {
                int col = blockIdx.x * blockDim.x + threadIdx.x;
                int row = blockIdx.y * blockDim.y + threadIdx.y;
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
                        result_d[row*n+col] = sum;
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

                float *img_d, *kernel_d, *result_d;
                cudaMalloc((void**)&img_d, size);
                cudaMemcpy(img_d, img, size, cudaMemcpyHostToDevice);

                cudaMalloc((void**)&kernel_d, 9 * sizeof(float));
                cudaMemcpy(kernel_d, kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

                cudaMalloc((void **) &result_d, size);

                dim3 DimGrid(num_rows/8, num_cols/8, 1);
                dim3 DimBlock(8,8,1);
                convolution2D<<<DimGrid, DimBlock>>>(img_d, kernel_d, result_d, num_cols);

                cudaDeviceSynchronize();

                cudaMemcpy(result, result_d, size, cudaMemcpyDeviceToHost);
                return sol_path;
        }
};
