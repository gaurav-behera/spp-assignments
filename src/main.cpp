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

namespace solution
{
	std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols)
	{
		// int chunk_size = 32768;

		std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
		std::ofstream sol_fs(sol_path, std::ios::binary);
		std::ifstream bitmap_fs(bitmap_path, std::ios::binary);
		// const auto img = std::make_unique<float[]>(num_rows * num_cols);
		void *aligned_img_ptr;
		if (posix_memalign(&aligned_img_ptr, 64, sizeof(float) * num_rows * num_cols) != 0)
		{
			throw std::bad_alloc();
		}
		float *img = static_cast<float *>(aligned_img_ptr);

		void *aligned_result;
		if (posix_memalign(&aligned_result, 64, sizeof(float) * num_rows * num_cols) != 0)
		{
			throw std::bad_alloc();
		}
		float *result = static_cast<float *>(aligned_result);

		bitmap_fs.read(reinterpret_cast<char *>(img), sizeof(float) * num_rows * num_cols);
		bitmap_fs.close();

		omp_set_nested(1);
#pragma omp parallel num_threads(48)
		{
			int thread_id = omp_get_thread_num();
			int node_id = thread_id / 28;
			int cpu_id = thread_id % 28;

			cpu_set_t cpu_set;
			CPU_ZERO(&cpu_set);
			CPU_SET(cpu_id, &cpu_set);
			sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);

#pragma omp for collapse(2)
			for (int i = 0; i < num_rows; ++i)
			{
				for (int j = 0; j < num_cols; j += 16)
				{
					__m512 sum = _mm512_setzero_ps();
					__m512 pixels, filterVal;
					for (int di = -1; di <= 1; di++)
					{
						int ni = i + di;
						if (ni >= 0 && ni < num_rows)
						{
							__mmask16 mask[3] = {0xFFFF, 0xFFFF, 0xFFFF};
							if (j - 1 < 0)
								mask[0] &= 0xFFFE;
							if (j + 16 >= num_cols)
								mask[2] &= 0x7FFF;

							for (int k = 0; k < 3; ++k)
							{
								pixels[k] = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask[k], &img[ni * num_cols + j + k - 1]);
								filterVal[k] = _mm512_set1_ps(kernel[di + 1][k]);
								sum = _mm512_fmadd_ps(pixels[k], filterVal[k], sum);
							}
						}
					}
					_mm512_storeu_ps(&result[i * num_cols + j], sum);
				}
			}
		}

		sol_fs.write(reinterpret_cast<const char *>(result), sizeof(float) * num_rows * num_cols);
		sol_fs.close();
		free(result);
		free(img);

		return sol_path;
	}
}