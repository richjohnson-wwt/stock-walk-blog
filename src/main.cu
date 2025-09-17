#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include "stockWalk.cuh"

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

// Moving Average CUDA Kernel
__global__ void moving_average_kernel(float* prices, float* moving_avg, int size, int window_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    // Calculate moving average for current position
    if (idx >= window_size - 1) {
        float sum = 0.0f;
        for (int i = 0; i < window_size; i++) {
            sum += prices[idx - i];
        }
        moving_avg[idx] = sum / window_size;
    } else {
        // Not enough data for full window
        moving_avg[idx] = 0.0f;
    }
}

// Signal Detection CUDA Kernel
__global__ void detect_signals(float* prices, float* moving_avg, int* signals, int size, int window_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size || idx < window_size) return;
    
    // Get current and previous values
    float current_price = prices[idx];
    float previous_price = prices[idx - 1];
    float current_ma = moving_avg[idx];
    float previous_ma = moving_avg[idx - 1];
    
    // Signal detection logic:
    // BUY (1): Price rising and above moving average
    // SELL (-1): Price falling and below moving average  
    // HOLD (0): Otherwise
    
    int signal = 0; // Default HOLD
    
    if (current_ma > 0.0f && previous_ma > 0.0f) { // Ensure valid moving averages
        if (current_price > previous_price && current_price > current_ma) {
            signal = 1; // BUY signal
        } else if (current_price < previous_price && current_price < current_ma) {
            signal = -1; // SELL signal
        }
    }
    
    signals[idx] = signal;
}

int main() {
    // Configuration - use CMake-defined DATA_DIR
#ifdef DATA_DIR
    std::string csv_filename = std::string(DATA_DIR) + "/stock_prices.csv";
#else
    std::string csv_filename = "../../../data/stock_prices.csv";  // Fallback path
#endif
    const int window_size = 20; // 20-day moving average
    
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
    float *d_prices, *d_moving_avg;
    int *d_signals;
    
    CUDA_CHECK(cudaMalloc(&d_prices, data.size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_moving_avg, data.size * sizeof(float)));
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
    
    // Launch moving average kernel
    cudaEventRecord(start);
    moving_average_kernel<<<gridSize, blockSize>>>(d_prices, d_moving_avg, data.size, window_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ma_time;
    cudaEventElapsedTime(&ma_time, start, stop);
    printf("Moving Average kernel completed in %.3f ms\n", ma_time);
    
    // Launch signal detection kernel
    cudaEventRecord(start);
    detect_signals<<<gridSize, blockSize>>>(d_prices, d_moving_avg, d_signals, data.size, window_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float signal_time;
    cudaEventElapsedTime(&signal_time, start, stop);
    printf("Signal Detection kernel completed in %.3f ms\n", signal_time);
    printf("Total GPU computation time: %.3f ms\n", ma_time + signal_time);
    
    // Allocate host memory for results
    float* h_moving_avg = (float*)malloc(data.size * sizeof(float));
    int* h_signals = (int*)malloc(data.size * sizeof(int));
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_moving_avg, d_moving_avg, data.size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_signals, d_signals, data.size * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Print results
    print_signals(data.prices, h_moving_avg, h_signals, data.size, window_size);
    
    // Cleanup
    cudaFree(d_prices);
    cudaFree(d_moving_avg);
    cudaFree(d_signals);
    free_stock_data(data);
    free(h_moving_avg);
    free(h_signals);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\nAnalysis complete!\n");
    return 0;
}