#include "sha256.h"
#include <nvml.h>

__global__ void sha256d_kernel(const uint8_t *input, uint8_t *output) {
    __shared__ uint8_t shared_input[64];
    __shared__ uint8_t shared_output[32];

    // Copy input to shared memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < 64; ++i)
            shared_input[i] = input[i];
    }
    __syncthreads();

    uint8_t temp[32]; // Temporary storage for SHA256 output

    sha256(shared_input, temp);   // First SHA256
    sha256(temp, shared_output);  // Second SHA256 (SHA256d)

    // Copy result to output
    if (threadIdx.x == 0) {
        for (int i = 0; i < 32; ++i)
            output[i] = shared_output[i];
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
        // Add throttling code or stop mining process
    }

    nvmlShutdown();
}
