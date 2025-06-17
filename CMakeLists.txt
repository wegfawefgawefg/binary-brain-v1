cmake_minimum_required(VERSION 3.25)
project(BinaryBrain LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(cuda)
add_subdirectory(python)
add_subdirectory(tests)
