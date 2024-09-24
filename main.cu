#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include "sha256.h"
#include "monitor_gpu_temperature.h"
#include "mining_pool.h"

using json = nlohmann::json;

// Error handling macros
#define CUDA_CHECK(call) do {                         \
    cudaError_t error = call;                         \
    if (error != cudaSuccess) {                       \
        fprintf(stderr, "CUDA Error: %s\n",           \
                cudaGetErrorString(error));           \
        exit(1);                                      \
    }                                                 \
} while (0)

#define NVML_CHECK(call) do {                        \
    nvmlReturn_t result = call;                       \
    if (result != NVML_SUCCESS) {                     \
        fprintf(stderr, "NVML Error: %s\n",           \
                nvmlErrorString(result));             \
        exit(1);                                      \
    }                                                 \
} while (0)

size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    // Append the received data to a string
    std::string* str = static_cast<std::string*>(userp);
    str->append(static_cast<char*>(contents), total_size);
    return total_size; // Return the number of bytes received
}

int main() {
    // Start GPU temperature monitoring in a separate thread
    std::thread gpu_monitor_thread(monitor_gpu_temperature);
    gpu_monitor_thread.detach();

    // Load configuration from JSON file
    std::ifstream config_file("config.json");
    if (!config_file.is_open()) {
        std::cerr << "Error: Could not open config.json" << std::endl;
        return 1;
    }

    json config;
    config_file >> config;
    config_file.close();

    std::string pool_url = config["pool_url"];
    std::string user_name = config["user_name"];
    std::string password = config["password"];

    std::cout << "Pool URL: " << pool_url << std::endl;
    std::cout << "User Name: " << user_name << std::endl;
    std::cout << "Password: " << password << std::endl;

    // Setup for SHA-256 hashing
    uint8_t h_input[64] = {0}; // Sample input
    uint8_t h_output[32] = {0};
    uint8_t *d_input, *d_output;

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(h_input)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(h_output)));

    // Copy input data to the device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    // Measure execution time of the kernel
    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch the kernel (make sure sha256d_kernel is properly defined in sha256.cu)
    sha256d_kernel<<<1, 1>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Copy the hash result back to the host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    // Output the hash result
    std::cout << "Hash Result: ";
    for (int i = 0; i < 32; ++i) {
        printf("%02x", h_output[i]);
    }
    std::cout << std::endl;

    // Calculate and output hashrate
    double hashrate = 1.0 / elapsed.count();
    std::cout << "Hashrate: " << hashrate << " H/s" << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    // Connect to the mining pool
    connectToPool(pool_url, user_name, password);

    return 0;
}