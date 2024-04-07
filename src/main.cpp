#pragma GCC optimize("O3,unroll-loops")
// #pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <immintrin.h>
#include <string>

namespace solution
{
	std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols)
	{
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
		if (posix_memalign(&aligned_result, 64, sizeof(float) * 16) != 0)
		{
			throw std::bad_alloc();
		}
		float *result = static_cast<float *>(aligned_result);

		bitmap_fs.read(reinterpret_cast<char *>(img), sizeof(float) * num_rows * num_cols);
		bitmap_fs.close();

		std::int32_t k = 0;
		for (k = 0; k < num_rows * num_cols; k += 16)
		{
			int i = k / num_cols, j = k % num_cols;
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
						{
							mask &= 0xFFFE;
						}
						if (nj + 15 >= num_cols)
						{
							mask &= 0x7FFF;
						}
						__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, &img[ni * num_cols + nj]);

						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}
				}
			}
			_mm512_store_ps(result, sum);
			sol_fs.write(reinterpret_cast<const char *>(result), sizeof(float) * 16);
		}
		sol_fs.close();
		free(result);
		free(img);
		return sol_path;
	}
};