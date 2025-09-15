cmake_minimum_required(VERSION 3.18)  # Minimum version for CUDA support

# Set the project name and languages
project(BlogStockWalk LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find Conan packages
find_package(Catch2 QUIET)

# Set CUDA architecture (adjust based on your GPU)
# Using widely supported architectures for compatibility
#set(CMAKE_CUDA_ARCHITECTURES "75;80;86")  # Covers RTX 20xx, RTX 30xx, RTX 40xx, and L40s

# Configure data directory (can be overridden at build time)
set(DATA_DIR "${CMAKE_SOURCE_DIR}/data" CACHE PATH "Path to data directory containing CSV files")
message(STATUS "Data directory: ${DATA_DIR}")

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)

# Optional: Add testing if Catch2 is available
if(Catch2_FOUND)
    enable_testing()
    # add_subdirectory(test)  # Uncomment when you add tests
endif()

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
