add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_compile_options(studentlib PRIVATE -fopenmp)
endif()

target_compile_options(studentlib PRIVATE -Ofast -march=native -fopenmp) 
