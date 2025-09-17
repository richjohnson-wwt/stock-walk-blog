#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include "stockWalk.cuh"
#include "calc.cuh"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)


int main() {
    // Configuration - use CMake-defined DATA_DIR
#ifdef DATA_DIR
    std::string csv_filename = std::string(DATA_DIR) + "/stock_prices.csv";
#else
    std::string csv_filename = "../../../data/stock_prices.csv";  // Fallback path
#endif
    const int long_window_size = 50; // 50-day moving average
    const int short_window_size = 20; // 20-day moving average
    
    printf("=== CUDA Stock Signal Analysis ===\n");
    printf("Reading TSLA data from: %s\n", csv_filename.c_str());
    
    // Read CSV data
    StockData data = read_csv_data(csv_filename.c_str());
    printf("Loaded %d price points\n", data.size);
    
    if (data.size == 0) {
        printf("Error: No data loaded from CSV file\n");
        return 1;
    }
    
    // Allocate device memory
    float *d_prices, *d_long_moving_avg, *d_short_moving_avg;
    int *d_signals;
    
    CUDA_CHECK(cudaMalloc(&d_prices, data.size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_long_moving_avg, data.size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_short_moving_avg, data.size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_signals, data.size * sizeof(int)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_prices, data.prices, data.size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (data.size + blockSize - 1) / blockSize;
    
    printf("\nLaunching CUDA kernels...\n");
    printf("Grid size: %d, Block size: %d\n", gridSize, blockSize);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch long moving average kernel
    cudaEventRecord(start);
    dual_long_moving_average_kernel<<<gridSize, blockSize>>>(d_prices, d_long_moving_avg, data.size, long_window_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);    
    
    float long_ma_time;
    cudaEventElapsedTime(&long_ma_time, start, stop);
    printf("Long Moving Average kernel completed in %.3f ms\n", long_ma_time);

    // Launch short moving average kernel
    cudaEventRecord(start);
    dual_short_moving_average_kernel<<<gridSize, blockSize>>>(d_prices, d_short_moving_avg, data.size, short_window_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);    
    
    float short_ma_time;
    cudaEventElapsedTime(&short_ma_time, start, stop);
    printf("Short Moving Average kernel completed in %.3f ms\n", short_ma_time);
    
    // Launch signal detection kernel
    cudaEventRecord(start);
    detect_moving_average_crossover<<<gridSize, blockSize>>>(d_prices, d_long_moving_avg, d_short_moving_avg, d_signals, data.size, long_window_size, short_window_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float signal_time;
    cudaEventElapsedTime(&signal_time, start, stop);
    printf("Signal Detection kernel completed in %.3f ms\n", signal_time);
    printf("Total GPU computation time: %.3f ms\n", long_ma_time + short_ma_time + signal_time);
    
    // Allocate host memory for results
    float* h_long_moving_avg = (float*)malloc(data.size * sizeof(float));
    float* h_short_moving_avg = (float*)malloc(data.size * sizeof(float));
    int* h_signals = (int*)malloc(data.size * sizeof(int));
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_long_moving_avg, d_long_moving_avg, data.size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_short_moving_avg, d_short_moving_avg, data.size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_signals, d_signals, data.size * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Print results
    print_signals(data.prices, h_long_moving_avg, h_short_moving_avg, h_signals, data.size, long_window_size);
    
    // Cleanup
    cudaFree(d_prices);
    cudaFree(d_long_moving_avg);
    cudaFree(d_short_moving_avg);
    cudaFree(d_signals);
    free_stock_data(data);
    free(h_long_moving_avg);
    free(h_short_moving_avg);
    free(h_signals);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\nAnalysis complete!\n");
    return 0;
}