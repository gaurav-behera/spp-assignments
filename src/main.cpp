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
		const auto img = std::make_unique<float[]>(num_rows * num_cols);
		bitmap_fs.read(reinterpret_cast<char *>(img.get()), sizeof(float) * num_rows * num_cols);
		bitmap_fs.close();

		// for (int k = 0; k < num_rows * num_cols; k += 16)
		// {
		// 	int i = k / num_cols, j = k % num_cols;
		// 	if (j == 0)
		// 	{
		// 		std::cout << std::endl;
		// 	}
		// 	std::cout << img[i * num_cols + j] << " ";
		// }

		// for (int i = 0; i < 3; i++)
		// {
		// 	for (int j = 0; j < 3; j++)
		// 	{
		// 		std::cout << kernel[i][j] << " ";
		// 	}
		// 	std::cout << std::endl;
		// }

		// std::cout << std::endl;
		// std::cout << "computed" << std::endl;

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
						// __m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
						float tempPixels[16];
						std::copy(&img[ni * num_cols + nj], &img[ni * num_cols + nj] + 16, tempPixels);
						if (nj < 0)
						{
							// std::cout << "l" << std::endl;
							tempPixels[0] = 0.0f;
						}
						if (nj + 15 >= num_cols)
						{
							// std::cout << "r" << std::endl;
							tempPixels[15] = 0.0f;
						}
						__m512 pixels = _mm512_load_ps(tempPixels);
						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}
					// sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
				}
			}
			float result[16];
			_mm512_store_ps(result, sum);
			sol_fs.write(reinterpret_cast<const char *>(result), sizeof(result));
		}

		// std::cout << std::endl;
		// std::cout << "correct" << std::endl;
		// for (k = 0; k < num_rows * num_cols; k++)
		// {
		// 	// std::cout << "here" << std::endl;
		// 	float sum = 0.0;
		// 	int i = k / num_cols, j = k % num_cols;
		// 	for (int di = -1; di <= 1; di++)
		// 	{
		// 		for (int dj = -1; dj <= 1; dj++)
		// 		{
		// 			int ni = i + di, nj = j + dj;
		// 			if (ni >= 0 and ni < num_rows and nj >= 0 and nj < num_cols)
		// 				sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
		// 		}
		// 	}
		// 	std::cout << sum << " ";
		// }
		sol_fs.close();
		return sol_path;
	}
};