cmake_minimum_required(VERSION 3.16)

project(magnus LANGUAGES CXX)

find_package(BLAS REQUIRED)

add_executable(magnus
    src/main.cpp
)

target_compile_features(magnus PRIVATE cxx_std_23)
target_compile_options(magnus PRIVATE -std=c++23 -O3 -march=native)
target_include_directories(magnus PRIVATE src)
target_link_libraries(magnus PRIVATE ${BLAS_LIBRARIES})
