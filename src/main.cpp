#pragma GCC optimize("O3,unroll-loops")
// #pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt,avx512f")
#include <filesystem>
#include <omp.h>
#include <cstring>
#include <immintrin.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstdlib>

namespace solution
{
	std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m)
	{
		std::string sol_path = "student_sol.dat";

		int size = 16777216;

		int m1_fd = open(m1_path.c_str(), O_RDONLY);
		void *m1_ptr;
		posix_memalign(&m1_ptr, 64, size);
		float *m1 = static_cast<float *>(mmap(m1_ptr, size, PROT_READ, MAP_PRIVATE, m1_fd, 0));
		int m2_fd = open(m2_path.c_str(), O_RDONLY);
		void *m2_ptr;
		posix_memalign(&m2_ptr, 64, size);
		float *m2 = static_cast<float *>(mmap(m2_ptr, size, PROT_READ, MAP_PRIVATE, m2_fd, 0));


		int result_fd = open(sol_path.c_str(), O_CREAT | O_RDWR, 0644);
		ftruncate(result_fd, size);
		float *result = static_cast<float *>(mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_SHARED, result_fd, 0));

		// int block_size = 128;
		// int block_count = n / block_size;

#pragma omp parallel num_threads(48)
		{
			int tid = omp_get_thread_num();
			cpu_set_t cpuset;
			CPU_SET(tid, &cpuset);
			pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

#pragma omp for collapse(2)
			for (int block_i = 0; block_i < 16; block_i++)
			{
				for (int block_j = 0; block_j < 16; block_j++)
				{
					// float temp[16384] __attribute__((aligned(64))) = {0};
					alignas(64) float temp[16384] = {0};
					for (int sub_block_k = 0; sub_block_k < 16; sub_block_k++)
					{
						for (int idx = 0; idx < 128; idx++)
						{
							for (int i = 0; i < 128; i++)
							{
								for (int j = 0; j < 128; j += 16)
								{
									int base1 = (i + block_i * 128) * n + (sub_block_k * 128 + idx);
									int base2 = (sub_block_k * 128 + idx) * n + (block_j * 128 + j);
									int temp_base = i * 128 + j;
									_mm512_store_ps(&temp[temp_base], _mm512_fmadd_ps(_mm512_set1_ps(m1[base1]), _mm512_load_ps(&m2[base2]), _mm512_load_ps(&temp[temp_base])));
								}
							}
						}
					}
					for (int i = 0; i < 128; i++)
					{
						memcpy(&result[(block_i * 128 + i) * 2048 + (block_j * 128)], &temp[i * 128], 512);
					}
				}
			}
		}

		return sol_path;
	}
};