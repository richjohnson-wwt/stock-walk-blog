#include "stockWalk.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cstdlib>

// Function to read CSV data
StockData read_csv_data(const char* filename) {
    std::ifstream file(filename);
    std::vector<float> prices;
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        int col = 0;
        
        while (std::getline(ss, item, ',')) {
            if (col == 4) { // Close price column
                prices.push_back(std::stof(item));
                break;
            }
            col++;
        }
    }
    
    file.close();
    
    StockData data;
    data.size = prices.size();
    data.prices = (float*)malloc(data.size * sizeof(float));
    
    for (int i = 0; i < data.size; i++) {
        data.prices[i] = prices[i];
    }
    
    return data;
}

// Function to free StockData memory
void free_stock_data(StockData& data) {
    if (data.prices != nullptr) {
        free(data.prices);
        data.prices = nullptr;
        data.size = 0;
    }
}


// Function to print signals with dates (sample)
void print_signals(float* prices, float* moving_avg, int* signals, int size, int window_size) {
    printf("\n=== STOCK SIGNAL ANALYSIS ===\n");
    printf("Window Size: %d\n", window_size);
    printf("Total Data Points: %d\n", size);
    printf("\nSignal Legend: BUY(1), SELL(-1), HOLD(0)\n");
    printf("%-8s %-10s %-10s %-8s\n", "Index", "Price", "MovAvg", "Signal");
    printf("----------------------------------------\n");
    
    int buy_count = 0, sell_count = 0, hold_count = 0;
    
    // Print first few and last few signals
    for (int i = window_size; i < std::min(size, window_size + 10); i++) {
        printf("%-8d %-10.2f %-10.2f %-8d\n", i, prices[i], moving_avg[i], signals[i]);
        if (signals[i] == 1) buy_count++;
        else if (signals[i] == -1) sell_count++;
        else hold_count++;
    }
    
    if (size > window_size + 20) {
        printf("...\n");
        for (int i = size - 10; i < size; i++) {
            printf("%-8d %-10.2f %-10.2f %-8d\n", i, prices[i], moving_avg[i], signals[i]);
            if (signals[i] == 1) buy_count++;
            else if (signals[i] == -1) sell_count++;
            else hold_count++;
        }
    }
    
    // Count all signals for summary
    buy_count = sell_count = hold_count = 0;
    for (int i = window_size; i < size; i++) {
        if (signals[i] == 1) buy_count++;
        else if (signals[i] == -1) sell_count++;
        else hold_count++;
    }
    
    printf("\n=== SIGNAL SUMMARY ===\n");
    printf("BUY signals: %d\n", buy_count);
    printf("SELL signals: %d\n", sell_count);
    printf("HOLD signals: %d\n", hold_count);
    printf("Total analyzed: %d\n", buy_count + sell_count + hold_count);
}