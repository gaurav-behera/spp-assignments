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

		int result_fd = open(sol_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
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

		void *mapped_result = mmap(NULL, sizeof(float) * num_rows * num_cols, PROT_WRITE | PROT_READ, MAP_SHARED, result_fd, 0);
		if (mapped_result == MAP_FAILED)
		{
			// std::cerr << "Failed to mmap result file: " << std::endl;
			std::cerr << "Failed to mmap result file: " << strerror(errno) << std::endl;
			munmap(mapped_img, sizeof(float) * num_rows * num_cols);
			close(bitmap_fd);
			close(result_fd);
			return "";
		}
		float *result = static_cast<float *>(mapped_result);

		// for (int i = 0; i < num_cols * num_rows; i++)
		// {
		// 	result[i] = img[i];
		// }


#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			int num_threads = omp_get_num_threads();
			int node_cpus[] = {0, 1};	// CPU cores per NUMA node
			int num_cpus_per_node = 24; // Number of CPU cores per NUMA node
			int nodes = 2;				// Number of NUMA nodes

			// Calculate loop iteration range for each thread based on NUMA node
			int chunk_size = (num_rows - 2) / num_threads;
			int start_i = 1 + tid * chunk_size;
			int end_i = (tid == num_threads - 1) ? num_rows - 1 : start_i + chunk_size;

			int node_id = tid % nodes;

#pragma omp critical
			{
				cpu_set_t cpuset;
				CPU_ZERO(&cpuset);
				for (int i = 0; i < num_cpus_per_node; i++)
				{
					CPU_SET(node_cpus[node_id] + i, &cpuset);
				}
				sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
			}
#pragma omp single nowait
			{
#pragma omp task
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
#pragma omp task
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
#pragma omp task
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
#pragma omp task
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
#pragma omp task
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
#pragma omp task
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
#pragma omp task
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
#pragma omp task
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
			}
// #pragma omp for schedule(dynamic)
			for (int i = start_i; i < end_i; ++i)
			{
				for (int j = 16; j < num_cols - 16; j += 16)
				{
					__m512 sum = _mm512_setzero_ps();
					for (int di = -1; di <= 1; di++)
					{
						for (int dj = -1; dj <= 1; dj++)
						{
							int ni = i + di;
							int nj = j + dj;
							// _mm_prefetch((const char *)&img[(ni + 1) * num_cols + nj], _MM_HINT_T2);
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
		// sol_fs.write(reinterpret_cast<const char *>(result), sizeof(float) * num_rows * num_cols);
		munmap(mapped_img, sizeof(float) * num_rows * num_cols);
		munmap(mapped_result, sizeof(float) * num_rows * num_cols);
		close(bitmap_fd);
		close(result_fd);

		return sol_path;
	}
}