#include "sha256.h"
#include <nvml.h>

__global__ void sha256d_kernel(const uint8_t *input, uint8_t *output, int num_hashes) {
    __shared__ uint8_t shared_input[64];
    __shared__ uint8_t shared_output[32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't exceed the number of hashes
    if (idx >= num_hashes) return;

    // Copy input to shared memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < 64; ++i) {
            shared_input[i] = input[i];  // Adjust if accessing multiple inputs
        }
    }
    __syncthreads();

    uint8_t temp[32]; // Temporary storage for SHA256 output

    // Perform SHA256d hashing
    sha256(shared_input, temp);   // First SHA256
    sha256(temp, shared_output);   // Second SHA256 (SHA256d)

    // Write result to output
    if (threadIdx.x == 0) {
        for (int i = 0; i < 32; ++i) {
            output[idx * 32 + i] = shared_output[i]; // Each thread writes its output
        }
    }
}

// Error handling function
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
    return result;
}

// Function to monitor GPU temperature using NVML
void monitor_gpu_temperature() {
    nvmlDevice_t device;
    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &device);

    unsigned int temp;
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    printf("GPU Temperature: %d°C\n", temp);

    if (temp > 80) {
        printf("Warning: GPU temperature too high! Throttling...\n");
        // Implement throttling logic or stop the mining process here
    }

    nvmlShutdown();
}
