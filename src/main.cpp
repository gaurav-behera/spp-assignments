#pragma GCC optimize("O3,unroll-loops")
// #pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt,avx512f")
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <omp.h>
#include <immintrin.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>
#include <pthread.h>
#include <sched.h>

namespace solution
{
	std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m)
	{
		std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";

		// std::ofstream sol_fs(sol_path, std::ios::binary);
		// std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);
		// auto m1 = std::make_unique<float[]>(n * k), m2 = std::make_unique<float[]>(k * m);
		// m1_fs.read(reinterpret_cast<char *>(m1.get()), sizeof(float) * n * k);
		// m2_fs.read(reinterpret_cast<char *>(m2.get()), sizeof(float) * k * m);
		// m1_fs.close();
		// m2_fs.close();
		// auto result = std::make_unique<float[]>(n * m);

		int m1_fd = open(m1_path.c_str(), O_RDONLY);
		float *m1 = static_cast<float *>(mmap(NULL, sizeof(float) * n * k, PROT_READ, MAP_PRIVATE, m1_fd, 0));
		int m2_fd = open(m2_path.c_str(), O_RDONLY);
		float *m2 = static_cast<float *>(mmap(NULL, sizeof(float) * k * m, PROT_READ, MAP_PRIVATE, m2_fd, 0));

		int result_fd = open(sol_path.c_str(), O_CREAT | O_RDWR, 0644);
		ftruncate(result_fd, sizeof(float) * n * m);
		float *result = reinterpret_cast<float *>(mmap(NULL, sizeof(float) * n * m, PROT_WRITE | PROT_READ, MAP_SHARED, result_fd, 0));

		// for (int i = 0; i < n * m; i++)
		// {
		// 	result[i] = 0;
		// }

		int block_size = 128;

#pragma omp parallel num_threads(24)
		{
			int tid = omp_get_thread_num() * 2;
			cpu_set_t cpuset;
			CPU_SET(tid, &cpuset);
			pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

#pragma omp single
			{
#pragma omp taskloop collapse(2)
				for (int block_i = 0; block_i < n / block_size; block_i++)
				{
					for (int block_j = 0; block_j < n / block_size; block_j++)
					{
						for (int sub_block_k = n / block_size - 1; sub_block_k >= 0; sub_block_k--)
						{
							for (int idx = 0; idx < block_size; idx++)
							{
								for (int i = 0; i < block_size; i++)
								{
									for (int j = 0; j < block_size; j += 16)
									{
										int base1 = (i + block_i * block_size) * n + (sub_block_k * block_size + idx);
										int base2 = (sub_block_k * block_size + idx) * n + (block_j * block_size + j);
										int final_base = (i + block_i * block_size) * n + (j + block_j * block_size);
										_mm512_storeu_ps(&result[final_base], _mm512_fmadd_ps(_mm512_set1_ps(m1[base1]), _mm512_loadu_ps(&m2[base2]), _mm512_loadu_ps(&result[final_base])));
										// result[(i + block_i * block_size) * n + (j + block_j * block_size)] += m1[(i + block_i * block_size) * n + (sub_block_k * block_size + idx)] * m2[(sub_block_k * block_size + idx) * n + (block_j * block_size + j)];
									}
								}
							}
						}
					}
				}
			}
		}

		// works - 800ms
		// #pragma omp parallel for collapse(2)
		// for (int block_i = 0; block_i < n / block_size; block_i++)
		// {
		// 	for (int block_j = 0; block_j < n / block_size; block_j++)
		// 	{
		// 		for (int sub_block_k = 0; sub_block_k < n / block_size; sub_block_k++)
		// 		{
		// 			for (int idx = 0; idx < block_size; idx++)
		// 			{
		// 				for (int i = 0; i < block_size; i++)
		// 				{
		// 					for (int j = 0; j < block_size; j++)
		// 					{
		// 						result[(i + block_i * block_size) * n + (j + block_j * block_size)] += m1[(i + block_i * block_size) * n + (sub_block_k * block_size + idx)] * m2[(sub_block_k * block_size + idx) * n + (block_j * block_size + j)];
		// 					}
		// 				}
		// 			}
		// 		}
		// 	}
		// }

		// for (int i = 0; i < n * m; i++)
		// {
		// 	std::cout << result[i] << " ";
		// }
		// std::cout << std::endl;

		// for (int i = 0; i < n * m; i++)
		// {
		// 	std::cout << m1[i] << " ";
		// }
		// std::cout << std::endl;

		// for (int i = 0; i < n * m; i++)
		// {
		// 	std::cout << m2[i] << " ";
		// }
		// std::cout << std::endl;

		// sol_fs.write(reinterpret_cast<const char *>(result.get()), sizeof(float) * n * m);
		// sol_fs.close();
		return sol_path;
	}
};
