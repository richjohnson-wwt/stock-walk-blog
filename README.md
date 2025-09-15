As a proof of concept, I developed a stock price signal predictor with the help of Windsurf and the Windsurf Cascade chat/write feature. This application analyzes historical stock data using moving averages to generate simple but effective buy, sell, or hold recommendations — a practical way to experiment with parallel GPU computing for financial analytics.

## Intial Setup

Do every time a new VM is created

    uv venv
    source .venv/bin/activate
    uv pip install conan
    conan profile detect
    Install C++ and CMake Extensions in VSCode



### Debug Config

    conan install .\
        --output-folder=build/debug \
        --build=missing \
        --settings=build_type=Debug

    cd build/debug 
    
    # All commands in build/debug
    cmake ../.. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug
    cmake --build .
    cmake --build . --parallel

    # Run tests
    ./test/stock_walk_tests
    ctest
    ctest --verbose

    # Run app
    ./bin/stock_signal
    