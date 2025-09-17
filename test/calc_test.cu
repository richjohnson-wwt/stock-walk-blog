#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "calc.cuh"

using namespace Catch;

TEST_CASE("do_dual_long_moving_average", "[calc]") {
    // Create test data with 100 realistic stock prices
    std::vector<float> prices;
        
    // Generate 100 realistic stock prices
    for (int i = 0; i < 100; i++) {
        prices.push_back(100.0f);
    }

    int idx = 99;
    int long_moving_size = 50;
    float long_moving_avg[50]; // first 50 values are 0
   
    do_dual_long_moving_average(idx, prices.data(), long_moving_avg, long_moving_size);
    
    // Verify results
    REQUIRE(long_moving_avg[99] == Approx(100.0f));
    // REQUIRE(long_moving_avg[98] == Approx(100.0f));
}

TEST_CASE("do_dual_short_moving_average", "[calc]") {
    // Create test data with 100 realistic stock prices
    std::vector<float> prices;
        
    // Generate 100 realistic stock prices
    for (int i = 0; i < 100; i++) {
        prices.push_back(100.0f);
    }

    int idx = 99;
    int short_moving_size = 10;
    float short_moving_avg[10];
   
    do_dual_short_moving_average(idx, prices.data(), short_moving_avg, short_moving_size);
    
    // Verify results
    REQUIRE(short_moving_avg[99] == Approx(100.0f));
    // REQUIRE(short_moving_avg[98] == Approx(100.0f));
}