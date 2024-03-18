# CC = /opt/homebrew/Cellar/llvm/17.0.6_1/bin/clang
# CC = /opt/homebrew/Cellar/gcc/13.2.0/bin/gcc-13
CC = gcc-13
OMP_FLAGS = -O3 -fopenmp
OMP_FLAGS_MAX =  -Ofast -march=native -mtune=native -Ofast -fopenmp -ftree-vectorize  -funroll-loops -ffast-math -funsafe-math-optimizations
VEC_FLAGS =  -O3 -march=native -mtune=native -funroll-loops -ffast-math -funsafe-math-optimizations

define run_multiple_times
	@echo "Running $(1) 50 times..."
	@for i in $$(seq 1 50); do \
		output=$$($(1)); \
		time=$$(echo "$$output" | awk '/Time taken:/ {print $$3}'); \
		gflops=$$(echo "$$output" | awk '/GFLOPS:/ {print $$2}'); \
		echo "$$i | $$time seconds | $$gflops GFLOPS"; \
	done > output.txt; \
	avg_time=$$(awk '{sum += $$3} END {print sum/NR}' output.txt); \
	max_gflops=$$(awk 'BEGIN {max = 0} {if ($$6 > max) max = $$6} END {print max}' output.txt); \
	avg_gflops=$$(awk '{sum += $$6} END {print sum/NR}' output.txt); \
	echo "Average time: $$avg_time seconds"; \
	echo "Average GFLOPS: $$avg_gflops";\
	echo "Maximum GFLOPS: $$max_gflops";
endef

define run_bandwidth_multiple_times
	@echo "Running $(1) 50 times..."
	@for i in $$(seq 1 50); do \
		output=$$($(1)); \
		time=$$(echo "$$output" | awk '/Time taken:/ {print $$3}'); \
		gflops=$$(echo "$$output" | awk '/GFLOPS:/ {print $$2}'); \
		read_bw=$$(echo "$$output" | awk '/Read Bandwidth:/ {print $$3}'); \
		write_bw=$$(echo "$$output" | awk '/Write Bandwidth:/ {print $$3}'); \
		echo "$$i | $$time seconds | $$gflops GFLOPS | $$read_bw GB/s | $$write_bw GB/s"; \
	done > output.txt; \
	avg_time=$$(awk '{sum += $$3} END {print sum/NR}' output.txt); \
	max_gflops=$$(awk 'BEGIN {max = 0} {if ($$6 > max) max = $$6} END {print max}' output.txt); \
	avg_gflops=$$(awk '{sum += $$6} END {print sum/NR}' output.txt); \
	avg_read_bw=$$(awk '{sum += $$9} END {print sum/NR}' output.txt); \
	avg_write_bw=$$(awk '{sum += $$12} END {print sum/NR}' output.txt); \
	echo "Average time: $$avg_time seconds"; \
	echo "Average GFLOPS: $$avg_gflops";\
	echo "Maximum GFLOPS: $$max_gflops";\
	echo "Avg Read Bandwidth: $$avg_read_bw GB/s";\
	echo "Avg Write Bandwidth: $$avg_write_bw GB/s";
endef

original: mvm_original.c
	$(CC) -O0 mvm_original.c -o mvm_original_O0
	$(CC) -O1 mvm_original.c -o mvm_original_O1 
	$(CC) -O2 mvm_original.c -o mvm_original_O2 
	$(CC) -O3 mvm_original.c -o mvm_original_O3
	$(CC) -Os mvm_original.c -o mvm_original_Os
	$(CC) -Ofast mvm_original.c -o mvm_original_Ofast
	
	
	$(call run_multiple_times, ./mvm_original_O0)
	$(call run_multiple_times, ./mvm_original_O1)
	$(call run_multiple_times, ./mvm_original_O2)
	$(call run_multiple_times, ./mvm_original_O3)
	$(call run_multiple_times, ./mvm_original_Os)
	$(call run_multiple_times, ./mvm_original_Ofast)

original_flags: mvm_original.c
	$(CC) -O3 mvm_original.c -ftree-vectorize  -funroll-loops -ffast-math -o mvm_original_opt1
	$(CC) -O3 mvm_original.c -ftree-vectorize  -funroll-loops -ffast-math -funsafe-math-optimizations -o mvm_original_opt2
	$(CC) -O3 -march=native -mtune=native mvm_original.c -ftree-vectorize  -funroll-loops -ffast-math -funsafe-math-optimizations -o mvm_original_opt3

	$(call run_multiple_times, ./mvm_original_opt1)
	$(call run_multiple_times, ./mvm_original_opt2)
	$(call run_multiple_times, ./mvm_original_opt3)

omp: 
	$(CC) $(OMP_FLAGS) mvm_omp_v1.c -o mvm_omp_v1
	$(CC) $(OMP_FLAGS) mvm_omp_v2.c -o mvm_omp_v2
	$(CC) $(OMP_FLAGS_MAX) mvm_omp_v1.c -o mvm_omp_v1_max
	$(CC) $(OMP_FLAGS_MAX) mvm_omp_v2.c -o mvm_omp_v2_max
	$(CC) -O0 mvm_omp_v2.c -o mvm_omp_base

	$(call run_multiple_times, ./mvm_omp_v1)
	$(call run_multiple_times, ./mvm_omp_v2)
	$(call run_multiple_times, ./mvm_omp_v1_max)
	$(call run_multiple_times, ./mvm_omp_v2_max)
	$(call run_multiple_times, ./mvm_omp_base)


vec:
	$(CC) $(VEC_FLAGS) -o mvm_vec_1 mvm_vec_1.c
	$(CC) $(VEC_FLAGS) -o mvm_vec_2 mvm_vec_2.c
	$(CC) -O0 mvm_vec_2.c -o mvm_vec_base

	$(call run_multiple_times, ./mvm_vec_1)
	$(call run_multiple_times, ./mvm_vec_2)
	$(call run_multiple_times, ./mvm_vec_base)
	
omp_vec:
	$(CC) $(VEC_FLAGS) $(OMP_FLAGS) mvm_omp_vec.c -o mvm_omp_vec
	$(CC) -O0 -fopenmp mvm_omp_vec.c -o mvm_omp_vec_base
	$(call run_multiple_times, ./mvm_omp_vec)
	$(call run_multiple_times, ./mvm_omp_vec_base)

bandwidth: mvm_bw.c
	$(CC) $(VEC_FLAGS) $(OMP_FLAGS) mvm_bw.c -o mvm_bw
	$(call run_bandwidth_multiple_times, ./mvm_bw)

stream: stream.c
	$(CC) $(OMP_FLAGS) stream.c -o stream_c
	./stream_c

clean:
	rm mvm_original_O0 mvm_original_O1 mvm_original_O2 mvm_original_O3 mvm_original_Ofast mvm_original_Os mvm_original_opt1 mvm_original_opt2 mvm_original_opt3 mvm_omp_v1 mvm_omp_v2 mvm_omp_v1_max mvm_omp_v2_max mvm_omp_base mvm_vec_1 mvm_vec_2 mvm_vec_base mvm_omp_vec mvm_omp_vec_base mvm_bw stream_c

