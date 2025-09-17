#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ void whoami(void) {
    int block_id =
        blockIdx.x +    // apartment number on this floor (points across)
        blockIdx.y * gridDim.x +    // floor number in this building (rows high)
        blockIdx.z * gridDim.x * gridDim.y;   // building number in this city (panes deep)

    int block_offset =
        block_id * // times our apartment number
        blockDim.x * blockDim.y * blockDim.z; // total threads per block (people per apartment)

    int thread_offset =
        threadIdx.x +  
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset; // global person id in the entire apartment complex

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

__device__ __host__ inline void do_dual_long_moving_average(int idx, float* prices, float* long_moving_avg, int long_window_size) {
    
    // Calculate moving averages for current position
    if (idx >= long_window_size - 1) {
        float sum = 0.0f;
        for (int i = 0; i < long_window_size; i++) {
            sum += prices[idx - i];
        }
        long_moving_avg[idx] = sum / long_window_size;
    } else {
        // Not enough data for full window
        long_moving_avg[idx] = 0.0f;
    }  
}

__device__ __host__ inline void do_dual_short_moving_average(int idx, float* prices, float* short_moving_avg, int short_window_size) {
    
    
    // Calculate moving averages for current position
    if (idx >= short_window_size - 1) {
        float sum = 0.0f;
        for (int i = 0; i < short_window_size; i++) {
            sum += prices[idx - i];
        }
        short_moving_avg[idx] = sum / short_window_size;
    } else {
        // Not enough data for full window
        short_moving_avg[idx] = 0.0f;
    }    
}

__global__ void dual_long_moving_average_kernel(float* prices, float* long_moving_avg, int size, int long_window_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;
    
    do_dual_long_moving_average(idx, prices, long_moving_avg, long_window_size);
}

__global__ void dual_short_moving_average_kernel(float* prices, float* short_moving_avg, int size, int short_window_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;
    
    do_dual_short_moving_average(idx, prices, short_moving_avg, short_window_size);
} 

__global__ void detect_moving_average_crossover(float* prices, float* long_moving_avg, float* short_moving_avg, int* signals, int size, int long_window_size, int short_window_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size || idx == 0) return; // Need previous values for crossover detection

    // Determin the GOLDEN CROSS when short moving average crosses above long moving average
    // Determin the DEAD CROSS when short moving average crosses below long moving average
    
    // Get current and previous values
    float current_long_ma = long_moving_avg[idx];
    float previous_long_ma = long_moving_avg[idx - 1];
    float current_short_ma = short_moving_avg[idx];
    float previous_short_ma = short_moving_avg[idx - 1];
    
    // Signal detection logic:
    // BUY (1): Short moving average crosses above long moving average
    // SELL (-1): Short moving average crosses below long moving average  
    // HOLD (0): Otherwise
    
    int signal = 0; // Default HOLD
    
    if (current_short_ma > current_long_ma && previous_short_ma < previous_long_ma) { // GOLDEN CROSS
        signal = 1; // BUY signal
    } else if (current_short_ma < current_long_ma && previous_short_ma > previous_long_ma) { // DEAD CROSS
        signal = -1; // SELL signal
    }
    
    signals[idx] = signal;
}