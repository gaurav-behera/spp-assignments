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
#include <fcntl.h>
#include <sys/mman.h>

namespace solution
{
	std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols)
	{
		// int chunk_size = 32768;

		std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
		std::ofstream sol_fs(sol_path, std::ios::binary);
		std::ifstream bitmap_fs(bitmap_path, std::ios::binary);
		// const auto img = std::make_unique<float[]>(num_rows * num_cols);
		// void *aligned_img_ptr;
		// if (posix_memalign(&aligned_img_ptr, 64, sizeof(float) * num_rows * num_cols) != 0)
		// {
		// 	throw std::bad_alloc();
		// }
		// float *img = static_cast<float *>(aligned_img_ptr);

		// void *aligned_result;
		// if (posix_memalign(&aligned_result, 64, sizeof(float) * num_rows * num_cols) != 0)
		// {
		// 	throw std::bad_alloc();
		// }
		// float *result = static_cast<float *>(aligned_result);

		// mmap
		int bitmap_fd = open(bitmap_path.c_str(), O_RDONLY);

		void *mapped_img = mmap(NULL, sizeof(float) * num_rows * num_cols, PROT_READ, MAP_PRIVATE, bitmap_fd, 0);
		float *img = static_cast<float *>(mapped_img);

		void *mapped_result = mmap(NULL, sizeof(float) * num_rows * num_cols, PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		float *result = static_cast<float *>(mapped_result);

		bitmap_fs.read(reinterpret_cast<char *>(img), sizeof(float) * num_rows * num_cols);
		bitmap_fs.close();

// omp_set_num_threads(4);
#pragma omp parallel
		{
			// int tid = omp_get_thread_num();
			// int cpu_id = (tid % 24) * 2;
			// cpu_set_t cpuset;
			// CPU_ZERO(&cpuset);
			// CPU_SET(cpu_id, &cpuset);
			// pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
			// #pragma omp single nowait
			// 			{
			// #pragma omp task
			{
				int i = 0;
#pragma omp parallel for num_threads(4) schedule(dynamic)
				for (int j = 16; j < num_cols - 16; j++)
				{
					__m512 sum = _mm512_setzero_ps();
					for (int di = 0; di <= 1; di++)
					{
						for (int dj = -1; dj <= 1; dj++)
						{
							int ni = i + di, nj = j + dj;
							__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
							__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
							sum = _mm512_fmadd_ps(pixels, filterVal, sum);
						}
					}
					_mm512_storeu_ps(&result[j], sum);
				}
			}
			// #pragma omp task
			{
				int i = num_rows - 1;
#pragma omp parallel for num_threads(4) schedule(dynamic)
				for (int j = 16; j < num_cols - 16; j++)
				{
					__m512 sum = _mm512_setzero_ps();
					for (int di = -1; di <= 0; di++)
					{
						for (int dj = -1; dj <= 1; dj++)
						{
							int ni = i + di, nj = j + dj;
							__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
							__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
							sum = _mm512_fmadd_ps(pixels, filterVal, sum);
						}
					}
					_mm512_storeu_ps(&result[i * num_cols + j], sum);
				}
			}
			// #pragma omp task
			{
				int j = 0;
#pragma omp parallel for num_threads(4) schedule(dynamic)
				for (int i = 1; i < num_rows - 1; i++)
				{
					__m512 sum = _mm512_setzero_ps();
					for (int di = -1; di <= 1; di++)
					{
						int dj = -1;
						__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0xFFFE, &img[(i + di) * num_cols + (j - 1)]);
						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
						sum = _mm512_fmadd_ps(pixels, filterVal, sum);

						for (int dj = 0; dj <= 1; dj++)
						{
							int ni = i + di, nj = j + dj;
							__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
							__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
							sum = _mm512_fmadd_ps(pixels, filterVal, sum);
						}
					}
					_mm512_storeu_ps(&result[i * num_cols], sum);
				}
			}
			// #pragma omp task
			{
				int j = num_cols - 16;
#pragma omp parallel for num_threads(4) schedule(dynamic)
				for (int i = 1; i < num_rows - 1; i++)
				{
					__m512 sum = _mm512_setzero_ps();
					for (int di = -1; di <= 1; di++)
					{
						for (int dj = -1; dj <= 0; dj++)
						{
							int ni = i + di, nj = j + dj;
							__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
							__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
							sum = _mm512_fmadd_ps(pixels, filterVal, sum);
						}
						int dj = 1;
						__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0x7FFF, &img[(i + di) * num_cols + (j + 1)]);
						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}
					_mm512_storeu_ps(&result[i * num_cols + j], sum);
				}
			}
// #pragma omp task
#pragma omp parallel num_threads(1)
			{
				int i = 0, j = 0;
				__m512 sum = _mm512_setzero_ps();
				for (int di = 0; di <= 1; di++)
				{
					int dj = -1;
					__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0xFFFE, &img[(i + di) * num_cols + (j - 1)]);
					__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);

					for (int dj = 0; dj <= 1; dj++)
					{
						int ni = i + di, nj = j + dj;
						__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}
				}
				_mm512_storeu_ps(&result[i * num_cols], sum);
			}
// #pragma omp task
#pragma omp parallel num_threads(1)
			{
				int i = num_rows - 1, j = 0;
				__m512 sum = _mm512_setzero_ps();
				for (int di = -1; di <= 0; di++)
				{
					int dj = -1;
					__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0xFFFE, &img[(i + di) * num_cols + (j - 1)]);
					__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);

					for (int dj = 0; dj <= 1; dj++)
					{
						int ni = i + di, nj = j + dj;
						__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}
				}
				_mm512_storeu_ps(&result[i * num_cols], sum);
			}
// #pragma omp task
#pragma omp parallel num_threads(1)
			{
				int i = 0, j = num_cols - 16;
				__m512 sum = _mm512_setzero_ps();
				for (int di = 0; di <= 1; di++)
				{
					for (int dj = -1; dj <= 0; dj++)
					{
						int ni = i + di, nj = j + dj;
						__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}
					int dj = 1;
					__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0x7FFF, &img[(i + di) * num_cols + (j + 1)]);
					__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
				}
				_mm512_storeu_ps(&result[i * num_cols + j], sum);
			}
// #pragma omp task
#pragma omp parallel num_threads(1)
			{
				// std::cout << "4"<< omp_get_thread_num() << std::endl;
				int i = num_rows - 1, j = num_cols - 16;
				__m512 sum = _mm512_setzero_ps();
				for (int di = -1; di <= 0; di++)
				{
					for (int dj = -1; dj <= 0; dj++)
					{
						int ni = i + di, nj = j + dj;
						__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
						__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
						sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}
					int dj = 1;
					__m512 pixels = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0x7FFF, &img[(i + di) * num_cols + (j + 1)]);
					__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
				}
				_mm512_storeu_ps(&result[i * num_cols + j], sum);
			}
			// }
#pragma omp for schedule(dynamic)
			for (int i = 1; i < num_rows - 1; ++i)
			{
				for (int j = 16; j < num_cols - 16; j += 16)
				{
					__m512 sum = _mm512_setzero_ps();
					for (int di = -1; di <= 1; di++)
					{
						for (int dj = -1; dj <= 1; dj++)
						{
							int ni = i + di, nj = j + dj;
							_mm_prefetch((const char *)&img[(ni + 1) * num_cols + nj], _MM_HINT_T2);
							__m512 pixels = _mm512_loadu_ps(&img[ni * num_cols + nj]);
							__m512 filterVal = _mm512_set1_ps(kernel[di + 1][dj + 1]);
							sum = _mm512_fmadd_ps(pixels, filterVal, sum);
						}
					}
					_mm512_storeu_ps(&result[i * num_cols + j], sum);
				}
			}
			// std::cout << "done" << std::endl;
		}
		sol_fs.write(reinterpret_cast<const char *>(result), sizeof(float) * num_rows * num_cols);
		sol_fs.close();
		munmap(mapped_img, sizeof(float) * num_rows * num_cols);
		munmap(mapped_result, sizeof(float) * num_rows * num_cols);
		return sol_path;
	}
}