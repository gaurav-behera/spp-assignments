add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_compile_options(studentlib PRIVATE -fopenmp)
    target_link_libraries(studentlib PRIVATE OpenMP::OpenMP_CXX)
endif()

find_library(NUMA_LIBRARY numa)
if(NUMA_LIBRARY)
    target_link_libraries(studentlib PRIVATE ${NUMA_LIBRARY})
endif()

target_compile_options(studentlib PRIVATE -Ofast -march=native -fopenmp) 
