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

		__m512 filterVals[3][3];
		for (int i = 0; i < 9; i++)
		{
			filterVals[i / 3][i % 3] = _mm512_set1_ps(kernel[i / 3][i % 3]);
		}
		// omp_set_affinity_format("0");

#pragma omp parallel proc_bind(close) num_threads(24)
		{
			int tid = omp_get_thread_num();
			int cpu_id;

			if (tid % 2)
			{
				cpu_id = tid - 1;
			}
			else
			{
				cpu_id = tid;
			}

			cpu_set_t cpuset;
			CPU_ZERO(&cpuset);
			CPU_SET(cpu_id, &cpuset);
			pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

#pragma omp single
			{
#pragma omp taskloop collapse(2)
				for (int i = 0; i < num_rows; i++)
				{
					for (int j = 0; j < num_cols; j += 16)
					{
						__m512 sum = _mm512_setzero_ps();
						for (int di = -1; di <= 1; di++)
						{
							if (i + di >= 0 && i + di < num_rows)
							{
								for (int dj = -1; dj <= 1; dj++)
								{
									__mmask16 mask = 0xFFFF;
									if (j + dj < 0)
										mask &= 0xFFFE;
									if (j + dj + 15 >= num_cols)
										mask &= 0x7FFF;

									__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, &img[(i + di) * num_cols + j + dj]);
									sum = _mm512_fmadd_ps(pixels, filterVals[di + 1][dj + 1], sum);
								}
							}
						}
						_mm512_storeu_ps(&result[i * num_cols + j], sum);
					}
				}
			}
		}

		return sol_path;
	}
}