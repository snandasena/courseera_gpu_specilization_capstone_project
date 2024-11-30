cmake_minimum_required(VERSION 3.30)

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-10)

# Suppress deprecated GPU target warnings
add_compile_options(-Wno-deprecated-gpu-targets)

# Enable CMP0104 to properly manage CUDA architectures
if (POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

# Project details
project(CapstoneProject LANGUAGES C CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)  # Set C++ standard to C++17
set(CMAKE_CUDA_STANDARD 17)  # Set CUDA standard to C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA root directory (if it's installed in a custom location)
set(CUDA_ROOT /usr/local/cuda)

# Find OpenCV and include its directories
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS}
        ${CUDA_ROOT}/include)


# Add executable target
add_executable(${PROJECT_NAME} src/main.cu)

# Link CUDA libraries, including OpenCV libraries
target_link_libraries(${PROJECT_NAME}
        ${CUDA_ROOT}/lib64/libcudart.so
        ${OpenCV_LIBS}
        -lstdc++
)

# Display summary of project configuration
message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "CUDA Host Compiler: ${CUDA_HOST_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
