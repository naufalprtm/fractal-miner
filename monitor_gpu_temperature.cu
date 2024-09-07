#include <nvml.h>
#include <cstdio>
#include <thread>
#include <chrono>

void monitor_gpu_temperature() {
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "NVML Error: %s\n", nvmlErrorString(result));
        return;
    }
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "NVML Error: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        return;
    }

    while (true) {
        unsigned int temp;
        result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
        if (result != NVML_SUCCESS) {
            fprintf(stderr, "NVML Error: %s\n", nvmlErrorString(result));
            break;
        }
        printf("GPU Temperature: %d°C\n", temp);

        if (temp > 80) {
            printf("Warning: GPU temperature too high! Throttling...\n");
        }

        std::this_thread::sleep_for(std::chrono::seconds(5)); // Check every 5 seconds
    }

    nvmlShutdown();
}
