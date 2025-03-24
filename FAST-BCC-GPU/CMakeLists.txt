cmake_minimum_required(VERSION 3.17)
project(External_BCC LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Debug build flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "-G")

# Release build flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --expt-relaxed-constexpr")

include_directories(include)

file(GLOB_RECURSE CUDA_SRC
    "src/extern_bcc/*.cu"
    "src/fast_bcc/*.cu"
    "src/main.cu"
    "src/graph/*.cu"
    "src/utility/*.cu"
)

add_executable(${PROJECT_NAME}
    ${CUDA_SRC}
)

target_link_libraries(${PROJECT_NAME}
    OpenMP::OpenMP_CXX
)

# Enable separable compilation for CUDA files. This allows CUDA device code
# to be compiled into separate objects and then linked together, similar to
# C++ code. It's useful for larger CUDA applications where device code is 
# spread across multiple source files.
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify include directories for the target
target_include_directories(${PROJECT_NAME} PRIVATE 
# The PRIVATE keyword means these directories are only used by ${PROJECT_NAME}.
# 'include' is the directory where header files are stored. This command tells 
# the compiler to look for header files in the 'include' directory.
    include
)
