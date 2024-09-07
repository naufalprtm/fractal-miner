#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include "sha256.cu"
#include "monitor_gpu_temperature.cu"

using json = nlohmann::json;

// Error handling and logging function
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
    return size * nmemb;
}

void connectToPool(const std::string& url, const std::string& user, const std::string& pass) {
    CURL *curl;
    CURLcode res;
    long response_code;

    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L); // Aktifkan output verbose untuk debugging
        
        // Atur header jika diperlukan
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // Nonaktifkan verifikasi SSL (hanya untuk debugging)
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

        // Persiapkan permintaan Stratum
        std::string stratum_request = "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[]}\n";
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, stratum_request.c_str());

        // Lakukan permintaan
        res = curl_easy_perform(curl);
        if(res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
            std::cout << "HTTP Response Code: " << response_code << std::endl;
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
}

int main() {
    // Start GPU temperature monitoring in a separate thread
    std::thread gpu_monitor_thread(monitor_gpu_temperature);
    gpu_monitor_thread.detach();

    // Load configuration from JSON
    std::ifstream config_file("config.json");
    if (!config_file.is_open()) {
        std::cerr << "Error: Could not open config.json" << std::endl;
        return 1;
    }

    json config;
    config_file >> config;
    config_file.close();

    // Extract pool URL, user name, and password
    std::string pool_url = config["pool_url"];
    std::string user_name = config["user_name"];
    std::string password = config["password"];

    // Print configurations for debugging
    std::cout << "Pool URL: " << pool_url << std::endl;
    std::cout << "User Name: " << user_name << std::endl;
    std::cout << "Password: " << password << std::endl;

    uint8_t h_input[64] = {0}; // Sample input
    uint8_t h_output[32];
    uint8_t *d_input, *d_output;

    // Allocate device memory and check for errors
    CUDA_CHECK(cudaMalloc(&d_input, 64));
    CUDA_CHECK(cudaMalloc(&d_output, 32));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, 64, cudaMemcpyHostToDevice));

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch kernel
    sha256d_kernel<<<1, 1>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 32, cudaMemcpyDeviceToHost));

    // Print the result
    std::cout << "Hash Result: ";
    for (int i = 0; i < 32; ++i)
        printf("%02x", h_output[i]);
    std::cout << std::endl;

    // Print hashrate
    double hashrate = 1.0 / elapsed.count(); // Hashes per second
    std::cout << "Hashrate: " << hashrate << " H/s" << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    // Connect to the mining pool
    connectToPool(pool_url, user_name, password);

    return 0;
}
