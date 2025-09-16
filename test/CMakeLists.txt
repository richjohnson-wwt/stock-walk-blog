# Test CMakeLists.txt for stock signal analysis unit tests

# Find Catch2 (from Conan)
find_package(Catch2 REQUIRED)

# Create test executable with CUDA support
add_executable(stock_walk_tests 
    test_stock_walk.cu
    ../src/stockWalk.cu  # Include the source file to test
)

# Link with Catch2
target_link_libraries(stock_walk_tests 
    PRIVATE 
    Catch2::Catch2WithMain
)

# Include directories
target_include_directories(stock_walk_tests 
    PRIVATE 
    ../src  # To access stockWalk.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Pass test data directory as compile definition
target_compile_definitions(stock_walk_tests PRIVATE 
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data"  # For compatibility with existing code
)

# Register test with CTest
include(CTest)
include(Catch)
catch_discover_tests(stock_walk_tests)