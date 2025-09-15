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