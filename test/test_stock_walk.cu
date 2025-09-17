#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "stockWalk.cuh"
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>

using namespace Catch;

// Helper function to create test CSV data
void create_test_csv(const std::string& filename, const std::vector<std::vector<std::string>>& data) {
    std::ofstream file(filename);
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

TEST_CASE("StockData CSV Reading Tests", "[csv][stockwalk]") {
    
    SECTION("Read valid CSV with multiple stock prices") {
        // Create test data with realistic stock prices
        std::vector<std::vector<std::string>> test_data = {
            {"Date", "Open", "High", "Low", "Close", "Volume"},
            {"2023-01-01", "100.0", "105.0", "99.0", "102.5", "1000000"},
            {"2023-01-02", "102.0", "108.0", "101.0", "107.2", "1200000"},
            {"2023-01-03", "107.0", "110.0", "106.0", "108.8", "900000"},
            {"2023-01-04", "108.5", "112.0", "107.5", "111.25", "850000"}
        };
        
        std::string test_filename = "test_stock_data.csv";
        create_test_csv(test_filename, test_data);
        
        // Test the function
        StockData result = read_csv_data(test_filename.c_str());
        
        // Verify results
        REQUIRE(result.size == 4);  // Should have 4 data rows (header excluded)
        REQUIRE(result.prices != nullptr);
        
        // Check price values (Close column is index 4)
        REQUIRE(result.prices[0] == Approx(111.25f));
        REQUIRE(result.prices[1] == Approx(108.8f));
        REQUIRE(result.prices[2] == Approx(107.2f));
        REQUIRE(result.prices[3] == Approx(102.5f));
        
        // Cleanup
        free_stock_data(result);
        std::filesystem::remove(test_filename);
    }
    
    SECTION("Read CSV with single data point") {
        std::vector<std::vector<std::string>> test_data = {
            {"Date", "Open", "High", "Low", "Close", "Volume"},
            {"2023-01-01", "100.0", "105.0", "99.0", "150.25", "1000000"}
        };
        
        std::string test_filename = "test_single_stock.csv";
        create_test_csv(test_filename, test_data);
        
        StockData result = read_csv_data(test_filename.c_str());
        
        REQUIRE(result.size == 1);
        REQUIRE(result.prices != nullptr);
        REQUIRE(result.prices[0] == Approx(150.25f));
        
        free_stock_data(result);
        std::filesystem::remove(test_filename);
    }
    
    SECTION("Handle empty CSV (header only)") {
        std::vector<std::vector<std::string>> test_data = {
            {"Date", "Open", "High", "Low", "Close", "Volume"}
        };
        
        std::string test_filename = "test_empty_stock.csv";
        create_test_csv(test_filename, test_data);
        
        StockData result = read_csv_data(test_filename.c_str());
        
        REQUIRE(result.size == 0);
        // Note: The current implementation still allocates memory for size 0
        // This is actually the expected behavior based on the stockWalk.cu code
        
        free_stock_data(result);
        std::filesystem::remove(test_filename);
    }
    
    SECTION("Test with decimal prices and edge cases") {
        std::vector<std::vector<std::string>> test_data = {
            {"Date", "Open", "High", "Low", "Close", "Volume"},
            {"2023-01-01", "0.01", "0.02", "0.005", "0.015", "5000000"},
            {"2023-01-02", "1000.50", "1001.75", "999.25", "1000.00", "100000"},
            {"2023-01-03", "50.333", "50.999", "49.111", "50.567", "750000"}
        };
        
        std::string test_filename = "test_decimal_prices.csv";
        create_test_csv(test_filename, test_data);
        
        StockData result = read_csv_data(test_filename.c_str());
        
        REQUIRE(result.size == 3);
        REQUIRE(result.prices != nullptr);
        
        // Test precision with small and large numbers
        REQUIRE(result.prices[0] == Approx(50.567f).epsilon(0.001f));
        REQUIRE(result.prices[1] == Approx(1000.00f));
        REQUIRE(result.prices[2] == Approx(0.015f).epsilon(0.001f));
        
        free_stock_data(result);
        std::filesystem::remove(test_filename);
    }
}

TEST_CASE("StockData Memory Management Tests", "[memory][stockwalk]") {
    
    SECTION("free_stock_data properly cleans up allocated memory") {
        StockData data;
        data.size = 5;
        data.prices = (float*)malloc(data.size * sizeof(float));
        
        // Initialize some test data
        for (int i = 0; i < data.size; i++) {
            data.prices[i] = (float)(i + 1) * 10.0f;
        }
        
        // Verify data is set up correctly
        REQUIRE(data.prices != nullptr);
        REQUIRE(data.size == 5);
        REQUIRE(data.prices[0] == Approx(10.0f));
        REQUIRE(data.prices[4] == Approx(50.0f));
        
        // Test cleanup
        free_stock_data(data);
        
        // Verify cleanup
        REQUIRE(data.prices == nullptr);
        REQUIRE(data.size == 0);
    }
    
    SECTION("free_stock_data handles null pointer safely") {
        StockData data;
        data.prices = nullptr;
        data.size = 0;
        
        // Should not crash
        REQUIRE_NOTHROW(free_stock_data(data));
        
        // Should remain null
        REQUIRE(data.prices == nullptr);
        REQUIRE(data.size == 0);
    }
    
    SECTION("StockData structure initialization") {
        StockData data;
        data.size = 3;
        data.prices = (float*)malloc(data.size * sizeof(float));
        data.prices[0] = 100.5f;
        data.prices[1] = 200.25f;
        data.prices[2] = 300.75f;
        
        // Verify structure works correctly
        REQUIRE(data.size == 3);
        REQUIRE(data.prices[0] == Approx(100.5f));
        REQUIRE(data.prices[1] == Approx(200.25f));
        REQUIRE(data.prices[2] == Approx(300.75f));
        
        free_stock_data(data);
    }
}
