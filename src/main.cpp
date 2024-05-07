#pragma GCC optimize("Ofast,unroll-loops")
// #pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#pragma GCC target("avx512f")
#include <filesystem>
#include <omp.h>
#include <immintrin.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

namespace solution
{
	std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m)
	{
		std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";

		int m1_fd = open(m1_path.c_str(), O_RDONLY);
		float *m1 = static_cast<float *>(mmap(NULL, sizeof(float) * n * k, PROT_READ, MAP_PRIVATE, m1_fd, 0));
		int m2_fd = open(m2_path.c_str(), O_RDONLY);
		float *m2 = static_cast<float *>(mmap(NULL, sizeof(float) * k * m, PROT_READ, MAP_PRIVATE, m2_fd, 0));

		int result_fd = open(sol_path.c_str(), O_CREAT | O_RDWR, 0644);
		ftruncate(result_fd, sizeof(float) * n * m);
		float *result = static_cast<float *>(mmap(NULL, sizeof(float) * n * m, PROT_WRITE | PROT_READ, MAP_SHARED, result_fd, 0));

		int block_size = 128;
		int block_count = n / block_size;

#pragma omp parallel num_threads(24)
		{
			int tid = omp_get_thread_num() * 2;
			cpu_set_t cpuset;
			CPU_SET(tid, &cpuset);
			pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

#pragma omp for collapse(2)
			for (int m1_i = 0; m1_i < block_count; m1_i++)
			{
				for (int m2_j = 0; m2_j < block_count; m2_j++)
				{
					for (int m1_j = 0; m1_j < block_count; m1_j++)
					{
						for (int idx = 0; idx < block_size; idx++)
						{
							for (int i = 0; i < block_size; i++)
							{
								for (int j = 0; j < block_size; j += 16)
								{
									int base1 = (i + m1_i * block_size) * n + (m1_j * block_size + idx);
									int base2 = (m1_j * block_size + idx) * n + (m2_j * block_size + j);
									int final_base = (i + m1_i * block_size) * n + (j + m2_j * block_size);
									_mm512_storeu_ps(&result[final_base], _mm512_fmadd_ps(_mm512_set1_ps(m1[base1]), _mm512_loadu_ps(&m2[base2]), _mm512_loadu_ps(&result[final_base])));
								}
							}
						}
					}
				}
			}
		}

		return sol_path;
	}
};