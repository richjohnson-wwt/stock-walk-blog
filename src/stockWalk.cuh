#ifndef STOCK_WALK_CUH
#define STOCK_WALK_CUH

// Structure to hold stock data
struct StockData {
    float* prices;
    int size;
};

// Function declarations
StockData read_csv_data(const char* filename);
void free_stock_data(StockData& data);
void print_signals(float* prices, float* long_moving_avg, float* short_moving_avg, int* signals, int size, int window_size);

// __global__ void dual_long_moving_average_kernel(float* prices, float* long_moving_avg, int size, int long_window_size);
// __device__ __host__ void do_dual_long_moving_average(int idx, float* prices, float* long_moving_avg, int long_window_size);

// __global__ void dual_short_moving_average_kernel(float* prices, float* short_moving_avg, int size, int short_window_size);
// __device__ __host__ void do_dual_short_moving_average(int idx, float* prices, float* short_moving_avg, int short_window_size);

#endif // STOCK_WALK_CUH
