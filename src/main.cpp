#pragma GCC optimize("O3,unroll-loops")
// #pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <immintrin.h>
#include <string>
#include <vector>
#include <omp.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>

namespace solution
{
	std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols)
	{
		std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";

		int bitmap_fd = open(bitmap_path.c_str(), O_RDONLY);
		if (bitmap_fd == -1)
		{
			std::cerr << "Failed to open bitmap file: " << std::endl;
			return "";
		}

		void *mapped_img = mmap(NULL, sizeof(float) * num_rows * num_cols, PROT_READ, MAP_PRIVATE, bitmap_fd, 0);
		if (mapped_img == MAP_FAILED)
		{
			std::cerr << "Failed to mmap bitmap file: " << std::endl;
			close(bitmap_fd);
			return "";
		}
		float *img = static_cast<float *>(mapped_img);

		int result_fd = open(sol_path.c_str(), O_CREAT | O_RDWR);
		if (result_fd == -1)
		{
			std::cerr << "Failed to open result file: " << std::endl;
			munmap(mapped_img, sizeof(float) * num_rows * num_cols);
			close(bitmap_fd);
			return "";
		}
		if (ftruncate(result_fd, sizeof(float) * num_rows * num_cols) == -1)
		{
			std::cerr << "Failed to set file size: " << std::endl;
			close(result_fd);
			return "";
		}

		float *result = reinterpret_cast<float *>(mmap(NULL, sizeof(float) * num_rows * num_cols, PROT_WRITE | PROT_READ, MAP_SHARED, result_fd, 0));
		if (result == MAP_FAILED)
		{
			// std::cerr << "Failed to mmap result file: " << std::endl;
			std::cerr << "Failed to mmap result file: " << strerror(errno) << std::endl;
			munmap(mapped_img, sizeof(float) * num_rows * num_cols);
			close(bitmap_fd);
			close(result_fd);
			return "";
		}

		float paddedImage[(num_rows + 2) * (num_cols + 2)];
		std::fill(paddedImage, paddedImage + (num_cols + 2), 0.0f);
		for (int i = 0; i < num_rows; i++)
		{
			paddedImage[i * (num_cols + 2)] = 0;
			memcpy(paddedImage + i * (num_cols + 2) + 1, img + (i*num_cols), num_cols);
			paddedImage[(i + 1) * (num_cols + 2) - 1] = 0;
		}
		std::fill(paddedImage + ((num_cols + 2) * (num_rows + 1)), paddedImage + ((num_cols + 2) * (num_rows + 1)) + (num_cols + 2), 0.0f);
		// munmap(img, sizeof(float) * num_rows * num_cols);
		// munmap(result, sizeof(float) * num_rows * num_cols);
		// close(bitmap_fd);
		// close(result_fd);

#pragma omp parallel for schedule(dynamic, 16)
		for (int i = 1; i < num_rows - 1; ++i)
		{
			for (int j = 1; j < num_cols - 1; j += 16)
			{
				__m512 sum = _mm512_setzero_ps();
				for (int di = -1; di <= 1; di++)
				{
					for (int dj = -1; dj <= 1; dj++)
					{
						int ni = i + di;
						int nj = j + dj;
						// _mm_prefetch((const char *)&img[(ni + 1) * num_cols + nj], _MM_HINT_T2);
						__m512 pixels = _mm512_loadu_ps(&paddedImage[ni * (num_cols+2) + nj]);
						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}
				}
				_mm512_storeu_ps(&result[(i-1) * num_cols + (j-1)], sum);
			}
		}

// 		// std::cout << "done" << std::endl;
// 		// sol_fs.write(reinterpret_cast<const char *>(result), sizeof(float) * num_rows * num_cols);
		munmap(mapped_img, sizeof(float) * num_rows * num_cols);
		munmap(result, sizeof(float) * num_rows * num_cols);
		close(bitmap_fd);
		close(result_fd);

		return sol_path;
	}
}