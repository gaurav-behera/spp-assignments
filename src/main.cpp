#pragma GCC optimize("Ofast,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
// #pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt,avx512f")
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
#include <pthread.h>
#include <sched.h>

namespace solution
{
	std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols)
	{
		std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";

		int size = num_cols * num_rows;

		int bitmap_fd = open(bitmap_path.c_str(), O_RDONLY);
		float *img = static_cast<float *>(mmap(NULL, sizeof(float) * size, PROT_READ, MAP_PRIVATE, bitmap_fd, 0));

		int result_fd = open(sol_path.c_str(), O_CREAT | O_RDWR);
		ftruncate(result_fd, sizeof(float) * size);
		float *result = reinterpret_cast<float *>(mmap(NULL, sizeof(float) * size, PROT_WRITE | PROT_READ, MAP_SHARED, result_fd, 0));

		__m256 filterVals[3][3];
		for (int i = 0; i < 9; i++)
		{
			filterVals[i / 3][i % 3] = _mm256_set1_ps(kernel[i / 3][i % 3]);
		}

#pragma omp parallel proc_bind(spread) num_threads(48)
		{
			int tid = omp_get_thread_num();
			cpu_set_t cpuset;
			CPU_SET(tid, &cpuset);
			pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
			if (tid % 2)
			{
#pragma omp single
				{
#pragma omp taskloop collapse(2)
					for (int i = 0; i < num_rows / 2; i++)
					{
						for (int j = 0; j < num_cols; j += 8)
						{
							__m256 sum = _mm256_setzero_ps();
							for (int di = -1; di <= 1; di++)
							{
								if (i + di >= 0 && i + di < num_rows)
								{
									for (int dj = -1; dj <= 1; dj++)
									{
										__mmask8 mask = 0xFF;
										if (j + dj < 0)
											mask &= 0xFE;
										if (j + dj + 7 >= num_cols)
											mask &= 0x7F;

										__m256 pixels = _mm256_mask_loadu_ps(_mm256_setzero_ps(), mask, &img[(i + di) * num_cols + j + dj]);
										sum = _mm256_fmadd_ps(pixels, filterVals[di + 1][dj + 1], sum);
									}
								}
							}
							_mm256_storeu_ps(&result[i * num_cols + j], sum);
						}
					}
				}
			}
			else
			{
#pragma omp single
				{
#pragma omp taskloop collapse(2)
					for (int i = num_rows / 2; i < num_rows; i++)
					{
						for (int j = 0; j < num_cols; j += 8)
						{
							__m256 sum = _mm256_setzero_ps();
							for (int di = -1; di <= 1; di++)
							{
								if (i + di >= 0 && i + di < num_rows)
								{
									for (int dj = -1; dj <= 1; dj++)
									{
										__mmask8 mask = 0xFF;
										if (j + dj < 0)
											mask &= 0xFE;
										if (j + dj + 7 >= num_cols)
											mask &= 0x7F;

										__m256 pixels = _mm256_mask_loadu_ps(_mm256_setzero_ps(), mask, &img[(i + di) * num_cols + j + dj]);
										sum = _mm256_fmadd_ps(pixels, filterVals[di + 1][dj + 1], sum);
									}
								}
							}
							_mm256_storeu_ps(&result[i * num_cols + j], sum);
						}
					}
				}
			}
		}
		return sol_path;
	}
}