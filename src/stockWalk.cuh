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

#endif // STOCK_WALK_CUH
