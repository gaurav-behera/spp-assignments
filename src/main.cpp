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
		std::string sol_path = "student_sol.bmp";

		int bitmap_fd = open(bitmap_path.c_str(), O_RDONLY);

		float *img = static_cast<float *>(mmap(NULL, sizeof(float) * num_rows * num_cols, PROT_READ, MAP_PRIVATE, bitmap_fd, 0));

		int result_fd = open(sol_path.c_str(), O_CREAT | O_RDWR);
		ftruncate(result_fd, sizeof(float) * num_rows * num_cols);
		float *result = reinterpret_cast<float *>(mmap(NULL, sizeof(float) * num_rows * num_cols, PROT_WRITE | PROT_READ, MAP_SHARED, result_fd, 0));

#pragma omp parallel proc_bind(spread) 
		{
#pragma omp for 
			for (int i = 0; i < num_rows; ++i)
			{
				for (int j = 0; j < num_cols; j += 16)
				{
					__m512 sum = _mm512_setzero_ps();
					for (int di = -1; di <= 1; di++)
					{
						for (int dj = -1; dj <= 1; dj++)
						{
							int ni = i + di, nj = j + dj;

							if (ni >= 0 && ni < num_rows)
							{
								__mmask16 mask = 0xFFFF;
								if (nj < 0)
									mask &= 0xFFFE;
								if (nj + 15 >= num_cols)
									mask &= 0x7FFF;

								__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, &img[ni * num_cols + nj]);
								__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
								sum = _mm512_fmadd_ps(pixels, filterVal, sum);
							}
						}
					}
					_mm512_storeu_ps(&result[i * (num_cols) + j], sum);
				}
			}
		}
		// sol_fs.write(reinterpret_cast<const char *>(result), sizeof(float) * num_rows * num_cols);
		// munmap(mapped_img, sizeof(float) * num_rows * num_cols);
		// munmap(result, sizeof(float) * num_rows * num_cols);
		// close(bitmap_fd);
		// close(result_fd);

		return sol_path;
	}
}