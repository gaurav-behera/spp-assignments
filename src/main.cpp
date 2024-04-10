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
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";

        int size = num_cols * num_rows;

        int bitmap_fd = open(bitmap_path.c_str(), O_RDONLY);
        float *img = static_cast<float *>(mmap(NULL, sizeof(float) * size, PROT_READ, MAP_PRIVATE, bitmap_fd, 0));

        int result_fd = open(sol_path.c_str(), O_CREAT | O_RDWR);
        ftruncate(result_fd, sizeof(float) * size);
        float *result = reinterpret_cast<float *>(mmap(NULL, sizeof(float) * size, PROT_WRITE | PROT_READ, MAP_SHARED, result_fd, 0));

        // Convert kernel to half-precision floats
        __m512h filterVals[3][3];
        for (int i = 0; i < 9; i++)
        {
            filterVals[i / 3][i % 3] = _mm512_set1_ph(kernel[i / 3][i % 3]);
        }

        omp_set_affinity_format("0");

#pragma omp parallel proc_bind(close)
#pragma omp single
        {
#pragma omp taskloop collapse(2)
            for (int i = 0; i < num_rows; i++)
            {
                for (int j = 0; j < num_cols; j += 16)
                {
                    __m512h sum = _mm512_setzero_ph();
                    for (int di = -1; di <= 1; di++)
                    {
                        if (i + di >= 0 && i + di < num_rows)
                        {
                            for (int dj = -1; dj <= 1; dj++)
                            {
                                __mmask16 mask = 0xFFFF;
                                if (j + dj < 0)
                                    mask &= 0xFFFE;
                                if (j + dj + 15 >= num_cols)
                                    mask &= 0x7FFF;

                                // Load half-precision floats
                                __m512h pixels = _mm512_mask_loadu_ps(_mm512_setzero_ph(), mask, &img[(i + di) * num_cols + j + dj]);
                                sum = _mm512_fmadd_ps(pixels, filterVals[di + 1][+1], sum);
                            }
                        }
                    }
                    // Store half-precision floats
                    _mm512_storeu_ps(&result[i * num_cols + j], sum);
                }
            }
        }
        return sol_path;
    }
}
