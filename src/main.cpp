#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>

namespace solution{
	std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols){
		std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
		std::ofstream sol_fs(sol_path, std::ios::binary);
		std::ifstream bitmap_fs(bitmap_path, std::ios::binary);
		const auto img = std::make_unique<float[]>(num_rows * num_cols);
		bitmap_fs.read(reinterpret_cast<char*>(img.get()), sizeof(float) * num_rows * num_cols);
		bitmap_fs.close();
		for(std::int32_t k = 0; k < num_rows * num_cols; k++) {
		    float sum = 0.0;
		    int i = k / num_cols, j = k % num_cols;
		    for(int di = -1; di <= 1; di++)
		        for(int dj = -1; dj <= 1; dj++) {
		        	int ni = i + di, nj = j + dj;
		            if(ni >= 0 and ni < num_rows and nj >= 0 and nj < num_cols) 
		            	sum += kernel[di+1][dj+1] * img[ni * num_cols + nj];
	        	}
		    sol_fs.write(reinterpret_cast<char*>(&sum), sizeof(sum));
		}
		sol_fs.close();
		return sol_path;
	}
};