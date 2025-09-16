cmake_minimum_required(VERSION 3.6)  # Minimum version for CUDA support

# Set the project name and languages
project(BlogStockWalk LANGUAGES CXX CUDA)

include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake OPTIONAL)
find_package(Catch2 REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

# Set CUDA architecture (adjust based on your GPU)
# Using widely supported architectures for compatibility
#set(CMAKE_CUDA_ARCHITECTURES "75;80;86")  # Covers RTX 20xx, RTX 30xx, RTX 40xx, and L40s

# Configure data directory (can be overridden at build time)
set(DATA_DIR "${CMAKE_SOURCE_DIR}/data" CACHE PATH "Path to data directory containing CSV files")
message(STATUS "Data directory: ${DATA_DIR}")

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)


enable_testing()


# Print configuration summary
message(STATUS "=== Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
if(Catch2_FOUND)
    message(STATUS "Catch2 Testing: Available")
else()
    message(STATUS "Catch2 Testing: Not Available")
endif()
