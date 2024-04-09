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
		// std::string sol_path = std::filesystem::temp_directory_path() / "sol-32768x32768.bmp";
		// return sol_path;
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
			std::cerr << "Failed to open result file: " << strerror(errno) << std::endl;
			munmap(mapped_img, sizeof(float) * num_rows * num_cols);
			close(bitmap_fd);
			return "";
		}
		if (ftruncate(result_fd, sizeof(float) * num_rows * num_cols) == -1)
		{
			std::cerr << "Failed to set file size: " << strerror(errno) << std::endl;
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
							pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask[k], &img[ni * num_cols + j + k - 1]);
							filterVal = _mm512_set1_ps(kernel[di + 1][k]);
							sum = _mm512_fmadd_ps(pixels, filterVal, sum);
						}
					}
				}
				_mm512_storeu_ps(&result[i * num_cols + j], sum);
			}
		}
		// #pragma omp parallel
		// 		{

		// 			{
		// 				int i = 0;
		// #pragma omp for
		// 				for (int j = 16; j < num_cols - 16; j++)
		// 				{
		// 					__m512 sum = _mm512_setzero_ps();
		// 					for (int di = 0; di <= 1; di++)
		// 					{
		// 						for (int dj = -1; dj <= 1; dj++)
		// 						{
		// 							int ni = i + di, nj = j + dj;
		// 							__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
		// 							__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 							sum = _mm512_fmadd_ps(pixels, filterVal, sum);
		// 						}
		// 					}
		// 					_mm512_storeu_ps(&result[j], sum);
		// 				}
		// 			}
		// 			// #pragma omp task
		// 			{
		// 				int i = num_rows - 1;
		// #pragma omp for
		// 				for (int j = 16; j < num_cols - 16; j++)
		// 				{
		// 					__m512 sum = _mm512_setzero_ps();
		// 					for (int di = -1; di <= 0; di++)
		// 					{
		// 						for (int dj = -1; dj <= 1; dj++)
		// 						{
		// 							int ni = i + di, nj = j + dj;
		// 							__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
		// 							__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 							sum = _mm512_fmadd_ps(pixels, filterVal, sum);
		// 						}
		// 					}
		// 					_mm512_storeu_ps(&result[i * num_cols + j], sum);
		// 				}
		// 			}
		// 			// #pragma omp task
		// 			{
		// 				int j = 0;
		// #pragma omp for
		// 				for (int i = 1; i < num_rows - 1; i++)
		// 				{
		// 					__m512 sum = _mm512_setzero_ps();
		// 					for (int di = -1; di <= 1; di++)
		// 					{
		// 						int dj = -1;
		// 						__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0xFFFE, &img[(i + di) * num_cols + (j - 1)]);
		// 						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 						sum = _mm512_fmadd_ps(pixels, filterVal, sum);

		// 						for (int dj = 0; dj <= 1; dj++)
		// 						{
		// 							int ni = i + di, nj = j + dj;
		// 							__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
		// 							__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 							sum = _mm512_fmadd_ps(pixels, filterVal, sum);
		// 						}
		// 					}
		// 					_mm512_storeu_ps(&result[i * num_cols], sum);
		// 				}
		// 			}
		// 			// #pragma omp task
		// 			{
		// 				int j = num_cols - 16;
		// #pragma omp for
		// 				for (int i = 1; i < num_rows - 1; i++)
		// 				{
		// 					__m512 sum = _mm512_setzero_ps();
		// 					for (int di = -1; di <= 1; di++)
		// 					{
		// 						for (int dj = -1; dj <= 0; dj++)
		// 						{
		// 							int ni = i + di, nj = j + dj;
		// 							__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
		// 							__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 							sum = _mm512_fmadd_ps(pixels, filterVal, sum);
		// 						}
		// 						int dj = 1;
		// 						__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0x7FFF, &img[(i + di) * num_cols + (j + 1)]);
		// 						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
		// 					}
		// 					_mm512_storeu_ps(&result[i * num_cols + j], sum);
		// 				}
		// 			}
		// #pragma omp single
		// 			{
		// 				int i = 0, j = 0;
		// 				__m512 sum = _mm512_setzero_ps();
		// 				for (int di = 0; di <= 1; di++)
		// 				{
		// 					int dj = -1;
		// 					__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0xFFFE, &img[(i + di) * num_cols + (j - 1)]);
		// 					__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 					sum = _mm512_fmadd_ps(pixels, filterVal, sum);

		// 					for (int dj = 0; dj <= 1; dj++)
		// 					{
		// 						int ni = i + di, nj = j + dj;
		// 						__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
		// 						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
		// 					}
		// 				}
		// 				_mm512_storeu_ps(&result[i * num_cols], sum);
		// 			}
		// // #pragma omp single
		// 			{
		// 				int i = num_rows - 1, j = 0;
		// 				__m512 sum = _mm512_setzero_ps();
		// 				for (int di = -1; di <= 0; di++)
		// 				{
		// 					int dj = -1;
		// 					__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0xFFFE, &img[(i + di) * num_cols + (j - 1)]);
		// 					__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 					sum = _mm512_fmadd_ps(pixels, filterVal, sum);

		// 					for (int dj = 0; dj <= 1; dj++)
		// 					{
		// 						int ni = i + di, nj = j + dj;
		// 						__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
		// 						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
		// 					}
		// 				}
		// 				_mm512_storeu_ps(&result[i * num_cols], sum);
		// 			}
		// // #pragma omp single
		// 			{
		// 				int i = 0, j = num_cols - 16;
		// 				__m512 sum = _mm512_setzero_ps();
		// 				for (int di = 0; di <= 1; di++)
		// 				{
		// 					for (int dj = -1; dj <= 0; dj++)
		// 					{
		// 						int ni = i + di, nj = j + dj;
		// 						__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
		// 						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
		// 					}
		// 					int dj = 1;
		// 					__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0x7FFF, &img[(i + di) * num_cols + (j + 1)]);
		// 					__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
		// 				}
		// 				_mm512_storeu_ps(&result[i * num_cols + j], sum);
		// 			}
		// // #pragma omp single
		// 			{
		// 				// std::cout << "4"<< omp_get_thread_num() << std::endl;
		// 				int i = num_rows - 1, j = num_cols - 16;
		// 				__m512 sum = _mm512_setzero_ps();
		// 				for (int di = -1; di <= 0; di++)
		// 				{
		// 					for (int dj = -1; dj <= 0; dj++)
		// 					{
		// 						int ni = i + di, nj = j + dj;
		// 						__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
		// 						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
		// 					}
		// 					int dj = 1;
		// 					__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0x7FFF, &img[(i + di) * num_cols + (j + 1)]);
		// 					__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
		// 				}
		// 				_mm512_storeu_ps(&result[i * num_cols + j], sum);
		// 			}
		// 		}
		// // #pragma omp for schedule(dynamic)
		// 		for (int i = 1; i < num_rows - 1; ++i)
		// 		{
		// 			for (int j = 16; j < num_cols - 16; j += 16)
		// 			{
		// 				__m512 sum = _mm512_setzero_ps();
		// 				for (int di = -1; di <= 1; di++)
		// 				{
		// 					for (int dj = -1; dj <= 1; dj++)
		// 					{
		// 						int ni = i + di;
		// 						int nj = j + dj;
		// 						// _mm_prefetch((const char *)&img[(ni + 1) * num_cols + nj], _MM_HINT_T2);
		// 						__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
		// 						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
		// 						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
		// 					}
		// 				}
		// 				_mm512_storeu_ps(&result[i * num_cols + j], sum);
		// 			}
		// 		}
		// std::cout << "done" << std::endl;

		// sol_fs.write(reinterpret_cast<const char *>(result), sizeof(float) * num_rows * num_cols);
		munmap(mapped_img, sizeof(float) * num_rows * num_cols);
		munmap(result, sizeof(float) * num_rows * num_cols);
		close(bitmap_fd);
		close(result_fd);

		return sol_path;
	}
}