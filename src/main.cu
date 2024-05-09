#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>

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
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                // printf("Hello from the GPU\n");
                if (i < n)
                {
                        result_d[i] = img_d[i];
                }
        }

        std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols)
        {
                std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
                std::ofstream sol_fs(sol_path, std::ios::binary);
                std::ifstream bitmap_fs(bitmap_path, std::ios::binary);
                const auto img = std::make_unique<float[]>(num_rows * num_cols);
                bitmap_fs.read(reinterpret_cast<char *>(img.get()), sizeof(float) * num_rows * num_cols);
                size_t size = num_cols * num_rows * sizeof(float);
                auto result = std::make_unique<float[]>(num_cols * num_rows);

                float *img_d, *kernel_d, *result_d;
                cudaMalloc((void**)&img_d, size);
                cudaMemcpy(img_d, img, size, cudaMemcpyHostToDevice);

                cudaMalloc((void**)&kernel_d, 9 * sizeof(float));
                cudaMemcpy(kernel_d, kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

                cudaMalloc((void **) &result_d, size);

                convolution2D<<<1, 1>>>(img_d, kernel_d, result_d, size);

                cudaDeviceSynchronize();

                cudaMemcpy(result, result_d, size, cudaMemcpyDeviceToHost);

                sol_fs.write(reinterpret_cast<const char*>(result.get()), size);

                bitmap_fs.close();
                return sol_path;
        }
};
