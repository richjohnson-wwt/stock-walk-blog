# Define the CUDA executable
add_executable(stock_signal main.cu stockWalk.cu)

# Set target properties for CUDA
set_target_properties(stock_signal PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Pass data directory as compile definition

target_compile_definitions(stock_signal PRIVATE 
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data")

# Set output directory
set_target_properties(stock_signal PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)


# Print target information
message(STATUS "CUDA executable 'stock_signal' configured")
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}/bin")
