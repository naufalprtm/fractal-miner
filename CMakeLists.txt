cmake_minimum_required(VERSION 3.10)
project(fractal-miner)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Specify CUDA as a language
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Find required packages
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

# Specify the source files
set(SOURCES main.cu sha256.cu monitor_gpu_temperature.cu)

# Create the executable
add_executable(fractal-miner ${SOURCES})

# Link libraries
target_link_libraries(fractal-miner PRIVATE CURL::libcurl nlohmann_json::nlohmann_json nvidia-ml)
