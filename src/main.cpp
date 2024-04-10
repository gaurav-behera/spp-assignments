#pragma GCC optimize("O3,unroll-loops")
// #pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt,avx512f")
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
		std::filesystem::path path_to_tmp = "/tmp";
		if (std::filesystem::exists(path_to_tmp) && std::filesystem::is_directory(path_to_tmp))
		{
			std::cout << "Contents of /tmp directory:" << std::endl;

			for (const auto &entry : std::filesystem::directory_iterator(path_to_tmp))
			{
				std::cout << entry.path() << std::endl;
			}
		}
		else
		{
			std::cout << "Error: /tmp directory does not exist or is not a directory." << std::endl;
		}

		std::filesystem::path path_to_tmp2 = "./";
		if (std::filesystem::exists(path_to_tmp2) && std::filesystem::is_directory(path_to_tmp2))
		{
			std::cout << "Contents of /tmp directory:" << std::endl;

			for (const auto &entry : std::filesystem::directory_iterator(path_to_tmp2))
			{
				std::cout << entry.path() << std::endl;
			}
		}
		else
		{
			std::cout << "Error: /tmp directory does not exist or is not a directory." << std::endl;
		}

		std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
		return sol_path;
		int size = num_cols * num_rows;

		int bitmap_fd = open(bitmap_path.c_str(), O_RDONLY);
		float *img = static_cast<float *>(mmap(NULL, sizeof(float) * size, PROT_READ, MAP_PRIVATE, bitmap_fd, 0));

		int result_fd = open(sol_path.c_str(), O_CREAT | O_RDWR);
		ftruncate(result_fd, sizeof(float) * size);
		float *result = reinterpret_cast<float *>(mmap(NULL, sizeof(float) * size, PROT_WRITE | PROT_READ, MAP_SHARED, result_fd, 0));

		__m512 filterVals[3][3];
		for (int i = 0; i < 9; i++)
		{
			filterVals[i / 3][i % 3] = _mm512_set1_ps(kernel[i / 3][i % 3]);
		}

#pragma omp parallel proc_bind(spread)
#pragma omp single
		{
#pragma omp taskloop collapse(2)
			for (int i = 0; i < num_rows; i++)
			{
				for (int j = 16; j < num_cols - 16; j += 16)
				{
					__m512 sum = _mm512_setzero_ps();
					for (int di = -1; di <= 1; di++)
					{
						for (int dj = -1; dj <= 1; dj++)
						{
							int ni = i + di, nj = j + dj;
							if (ni >= 0 && ni < num_rows)
							{
								__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0xFFFF, &img[ni * num_cols + nj]);
								sum = _mm512_fmadd_ps(pixels, filterVals[di + 1][dj + 1], sum);
							}
						}
					}
					_mm512_storeu_ps(&result[i * num_cols + j], sum);
				}
			}
#pragma omp taskloop
			for (int i = 0; i < num_rows; i++)
			{
				int j = 0;
				__m512 sum = _mm512_setzero_ps();
				for (int di = -1; di <= 1; di++)
				{
					for (int dj = -1; dj <= 1; dj++)
					{
						int ni = i + di, nj = j + dj;
						if (ni >= 0 && ni < num_rows)
						{
							__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0xFFFE, &img[ni * num_cols + nj]);
							sum = _mm512_fmadd_ps(pixels, filterVals[di + 1][dj + 1], sum);
						}
					}
				}
				_mm512_storeu_ps(&result[i * num_cols], sum);
			}
#pragma omp taskloop
			for (int i = 0; i < num_rows; i++)
			{
				int j = num_cols - 16;
				__m512 sum = _mm512_setzero_ps();
				for (int di = -1; di <= 1; di++)
				{
					for (int dj = -1; dj <= 1; dj++)
					{
						int ni = i + di, nj = j + dj;
						if (ni >= 0 && ni < num_rows)
						{
							__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0x7FFF, &img[ni * num_cols + nj]);
							sum = _mm512_fmadd_ps(pixels, filterVals[di + 1][dj + 1], sum);
						}
					}
				}
				_mm512_storeu_ps(&result[i * num_cols + j], sum);
			}
		}
		return sol_path;
	}
}