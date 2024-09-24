#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include "sha256.h"
#include "monitor_gpu_temperature.h"
#include "mining_rpc.h" // Updated to reflect RPC instead of pool
#include "benchmark.h" 

// Include the benchmark function prototype
void benchmark();

using json = nlohmann::json;

// Error handling macros
#define CUDA_CHECK(call) do {                         \
    cudaError_t error = call;                         \
    if (error != cudaSuccess) {                       \
        fprintf(stderr, "[ERROR] CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(1);                                      \
    }                                                 \
} while (0)

#define NVML_CHECK(call) do {                        \
    nvmlReturn_t result = call;                       \
    if (result != NVML_SUCCESS) {                     \
        fprintf(stderr, "[ERROR] NVML Error: %s at %s:%d\n", \
                nvmlErrorString(result), __FILE__, __LINE__); \
        exit(1);                                      \
    }                                                 \
} while (0)

// Function to log messages with a timestamp
void logWithTimestamp(const std::string& message) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::cout << "[" << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") << "] " << message << std::endl;
}

int main(int argc, char *argv[]) {
    const int num_hashes = 1000000; // Define num_hashes here

    // Check command-line arguments for benchmark option
    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        logWithTimestamp("Starting GPU benchmark...");
        benchmark();
        logWithTimestamp("Benchmark completed.");
        return 0; // Exit after benchmark
    }

    logWithTimestamp("Starting the mining process...");

    // Start GPU temperature monitoring in a separate thread
    std::thread gpu_monitor_thread(monitor_gpu_temperature);
    gpu_monitor_thread.detach();
    logWithTimestamp("GPU temperature monitoring started in a separate thread.");

    // Load configuration from JSON file
    std::ifstream config_file("config.json");
    if (!config_file.is_open()) {
        logWithTimestamp("[ERROR] Could not open config.json");
        return 1;
    }

    json config;
    config_file >> config;
    config_file.close();

    // Extract configuration values
    std::string rpc_url = config["rpc_url"];
    std::string user_name = config["user_name"];
    std::string password = config["password"];
    std::string address = config["address"];

    // Log configuration
    logWithTimestamp("[INFO] Loaded configuration:");
    logWithTimestamp("    RPC URL: " + rpc_url);
    logWithTimestamp("    User Name: " + user_name);
    logWithTimestamp("    Password: " + password);
    logWithTimestamp("    Address: " + address);

    // Setup for SHA-256 hashing
    uint8_t h_input[64] = {0}; // Sample input
    uint8_t h_output[32] = {0};
    uint8_t *d_input, *d_output;

    // Allocate memory on the device
    logWithTimestamp("[INFO] Allocating memory on the device...");
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(h_input)));
    logWithTimestamp("[INFO] Device memory allocated for input data.");

    CUDA_CHECK(cudaMalloc(&d_output, sizeof(h_output)));
    logWithTimestamp("[INFO] Device memory allocated for output data.");

    // Copy input data to the device
    logWithTimestamp("[INFO] Copying input data to the device...");
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));
    logWithTimestamp("[INFO] Input data copied to device successfully.");

    // Measure execution time of the kernel
    auto start_time = std::chrono::high_resolution_clock::now();
    logWithTimestamp("[INFO] Launching SHA-256 kernel...");

    // Launch the kernel using the defined num_hashes
    sha256d_kernel<<<(num_hashes + 255) / 256, 256>>>(d_input, d_output, num_hashes);
    CUDA_CHECK(cudaDeviceSynchronize());
    logWithTimestamp("[INFO] SHA-256 kernel executed successfully.");

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Copy the hash result back to the host
    logWithTimestamp("[INFO] Copying hash result back to the host...");
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));
    logWithTimestamp("[INFO] Hash result copied back to host successfully.");

    // Output the hash result
    logWithTimestamp("[INFO] Hash Result: ");
    std::cout << "    ";
    for (int i = 0; i < 32; ++i) {
        printf("%02x", h_output[i]);
    }
    std::cout << std::endl;

    // Calculate and output hashrate
    double hashrate = num_hashes / elapsed.count(); // Calculate hashrate based on number of hashes and elapsed time
    logWithTimestamp("[INFO] Hashrate: " + std::to_string(hashrate) + " H/s");

    // Free device memory
    logWithTimestamp("[INFO] Freeing device memory...");
    CUDA_CHECK(cudaFree(d_input));
    logWithTimestamp("[INFO] Device memory for input data freed.");
    CUDA_CHECK(cudaFree(d_output));
    logWithTimestamp("[INFO] Device memory for output data freed.");

    // Connect to the mining RPC
    logWithTimestamp("[INFO] Connecting to the mining RPC...");
    connectToRPC(rpc_url, user_name, password);

    logWithTimestamp("[INFO] Mining process completed.");
    return 0;
}
