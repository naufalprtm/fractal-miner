#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <iostream>
#include <chrono>
#include <thread>
#include <csignal>
#include <iomanip>
#include <cuda_runtime.h>
#include "sha256.h"
#include "monitor_gpu_temperature.h"
#include "benchmark.h"
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
extern volatile sig_atomic_t keep_running;

// Signal handler to catch Ctrl+C
void signal_handler(int signum);

// Function prototypes
void runBenchmark();
void benchmark();
void logSeparator();
float getGPUTemperature();
float getGPUMemoryUsage();
float getCUDAUsage();
double getLastHashrate();

#endif // BENCHMARK_H
