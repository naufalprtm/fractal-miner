#include <iostream>
#include <chrono>
#include <thread>
#include <csignal>
#include <iomanip>
#include <cuda_runtime.h>
#include <nvml.h>  // Include NVML for GPU metrics
#include "sha256.h"

// Define CUDA_CHECK macro
#define CUDA_CHECK(call)                                          \
    {                                                            \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__  \
                      << std::endl;                             \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    }

// Global flag to control the benchmark loop
volatile sig_atomic_t keep_running = 1;

// Global variable for the last hashrate
double last_hashrate = 0.0;

// Function prototypes
void runBenchmark();
void benchmark();
float getGPUTemperature(); // Declare the function
double getLastHashrate() { return last_hashrate; } // Get last hashrate

// Signal handler to catch Ctrl+C
void signal_handler(int signum) {
    keep_running = 0;
}

// Function to log a separator line
void logSeparator() {
    std::cout << "----------------------------------------" << std::endl;
}

// Function to initialize NVML
void initNVML() {
    nvmlInit();
}

// Function to clean up NVML
void cleanupNVML() {
    nvmlShutdown();
}

// Function to get GPU memory usage
void getGPUMemoryUsage(size_t &free_memory, size_t &total_memory) {
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result == NVML_SUCCESS) {
        nvmlMemory_t memory_info;
        nvmlDeviceGetMemoryInfo(device, &memory_info);
        free_memory = memory_info.free;
        total_memory = memory_info.total;
    } else {
        std::cerr << "[ERROR] Unable to get GPU memory information." << std::endl;
        free_memory = total_memory = 0;
    }
}

// Function to get GPU utilization
float getGPUUtilization() {
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);
    nvmlUtilization_t utilization; // Correct type
    if (result == NVML_SUCCESS) {
        nvmlDeviceGetUtilizationRates(device, &utilization);
        return utilization.gpu; // Return GPU utilization
    } else {
        std::cerr << "[ERROR] Unable to get GPU utilization." << std::endl;
        return 0;
    }
}

// Mock function to simulate GPU temperature (update with real implementation)
float getGPUTemperature() {
    return 70.0f; // Example temperature in Celsius
}

// Function to convert SM version to cores
int _ConvertSMVer2Cores(int major, int minor) {
    // Map of compute capability to number of CUDA cores
    if (major == 2) {
        return (minor == 0) ? 48 : 32;  // Fermi
    } else if (major == 3) {
        return 192; // Kepler
    } else if (major == 5) {
        return 128; // Maxwell
    } else if (major == 6) {
        return (minor == 0) ? 64 : 128; // Pascal
    } else if (major == 7) {
        return (minor == 0) ? 64 : 128; // Volta
    } else if (major == 8) {
        return (minor == 0) ? 64 : 128; // Ampere
    } else {
        return 0; // Unknown architecture
    }
}

// Function to get the number of CUDA cores based on GPU model
int getCUDACoreCount() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        int device;
        cudaDeviceProp prop;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        return prop.multiProcessorCount * _ConvertSMVer2Cores(prop.major, prop.minor);
    }
    return 0;
}

// Function to benchmark and log GPU metrics
void benchmark() {
    // Initialize NVML for GPU metrics
    initNVML();
    
    // Set up signal handling for graceful shutdown
    signal(SIGINT, signal_handler);
    
    std::cout << "[INFO] Starting GPU benchmark..." << std::endl;

    while (keep_running) {
        size_t free_memory, total_memory;
        getGPUMemoryUsage(free_memory, total_memory);
        float gpu_utilization = getGPUUtilization();

        // Log a separator every 3 seconds
        logSeparator();
        std::cout << "[INFO] Benchmark running..." << std::endl;
        std::cout << "[INFO] Press Ctrl+C to stop the benchmark." << std::endl;
        std::cout << "[INFO] Current GPU Temperature: " << getGPUTemperature() << " °C" << std::endl;
        std::cout << "[INFO] Current GPU Memory Usage: " << 
            (static_cast<float>(total_memory - free_memory) / total_memory) * 100 << " % used" << std::endl;
        std::cout << "[INFO] Free GPU Memory: " << free_memory / (1024 * 1024) << " MB" << std::endl;
        std::cout << "[INFO] Total GPU Memory: " << total_memory / (1024 * 1024) << " MB" << std::endl;
        std::cout << "[INFO] Current GPU Utilization: " << gpu_utilization << " %" << std::endl;
        std::cout << "[INFO] Last Hashrate: " << std::fixed << std::setprecision(2) << getLastHashrate() << " H/s" << std::endl;

        // Sleep for a short duration to avoid flooding the output
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    std::cout << "[INFO] Benchmark stopped gracefully." << std::endl;

    // Clean up NVML
    cleanupNVML();
}

// Function to run the benchmark with increased workload
void runBenchmark() {
    const int num_hashes = 10000000; // Increase number of hashes
    uint8_t *d_input, *d_output;
    size_t input_size = sizeof(uint8_t) * 64; // SHA256 input size
    size_t output_size = sizeof(uint8_t) * 32; // SHA256 output size

    // Allocate pinned memory for input
    uint8_t *h_input;
    CUDA_CHECK(cudaHostAlloc((void**)&h_input, input_size * num_hashes, cudaHostAllocDefault));

    // Initialize input data with dynamic values
    for (int i = 0; i < num_hashes; ++i) {
        for (int j = 0; j < 64; ++j) {
            h_input[i * 64 + j] = (i + j) % 256; // Example input
        }
    }

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc(&d_input, input_size * num_hashes));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * num_hashes));

    // Copy input data to the device using asynchronous transfer
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, input_size * num_hashes, cudaMemcpyHostToDevice));

    // Define number of blocks and threads for maximum utilization
    int cuda_cores = getCUDACoreCount();
    int threads_per_block = cuda_cores; // Use all available CUDA cores
    int num_blocks = (num_hashes + threads_per_block - 1) / threads_per_block; // Calculate number of blocks

    // Measure execution time of the kernel
    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch multiple kernels with increased workload
    for (int i = 0; i < 4; ++i) { // Launching 4 kernels to increase load
        sha256d_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, num_hashes);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();

    // Copy the hash result back to the host
    uint8_t *h_output = new uint8_t[output_size * num_hashes];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size * num_hashes, cudaMemcpyDeviceToHost));

    // Calculate elapsed time and output result
    std::chrono::duration<double> elapsed = end_time - start_time;
    last_hashrate = (num_hashes * 4 / elapsed.count()); // Update last hashrate (4 kernels)

    // Output the hash result of the last hash
    std::cout << "[INFO] Hash Result: ";
    for (int i = 0; i < 32; ++i) {
        printf("%02x", h_output[(num_hashes - 1) * 32 + i]);
    }
    std::cout << std::endl;

    std::cout << "[INFO] Hashrate: " << std::fixed << std::setprecision(2) << last_hashrate << " H/s" << std::endl;
    std::cout << "[INFO] Elapsed Time: " << elapsed.count() * 1000 << " ms" << std::endl;

    // Free device and host memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFreeHost(h_input)); // Free pinned memory
    delete[] h_output; // Free output buffer
}