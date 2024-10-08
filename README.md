# fractal-miner

Fractal Miner is a CUDA-based mining application that performs hashing operations and connects to a mining rpc for cryptocurrency mining. This guide will help you set up and run Fractal Miner on your system.

## Requirements

- **CUDA Toolkit**: Ensure you have the CUDA Toolkit installed on your system. This application is compatible with CUDA version 12.6.
- **CURL**: Required for making HTTP requests.
- **nlohmann/json**: For JSON parsing.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/naufalprtm/fractal-miner.git
    cd fractal-miner
    ```

2. **Install dependencies**:

    - **CURL**: Install CURL on your system. For example, on Ubuntu you can use:
    
      ```bash
      sudo apt-get install libcurl4-openssl-dev
      ```
    
    - **nlohmann/json**: Download and install the nlohmann/json library. You can use a package manager like `vcpkg`:
    
      ```bash
      vcpkg install nlohmann-json
      ```

3. **Build the project**:

    ```bash
    chmod +x install.sh && chmod +x build.sh
    ./install
    ./build
    ```

## Configuration
start node https://github.com/fractal-bitcoin/fractald-release
Create a `config.json` file in the same directory as the executable with the following format:

```json
{
    "rpc_url": "http://IP:PORT", 
    "user_name": "test", 
    "password": "test",
    "address": "bc1pxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
}
```

Replace "rpc_url", "user_name", and "password" with the appropriate values for your mining rpc.

## Running the Miner
Start the miner:

```
./fractal_miner
```

## TEST Running the Miner

```
./fractal-miner --benchmark
```


Monitor the output:

The miner will display the GPU temperature, hash result, and hashrate. It will also attempt to connect to the mining rpc using the provided configuration.

Details
Hashing: The miner performs hashing operations using a CUDA kernel defined in sha256.cu. The hashing result is displayed as a hexadecimal string.
GPU Monitoring: GPU temperature is monitored using a separate thread defined in monitor_gpu_temperature.cu.
HTTP Requests: The miner uses the CURL library to communicate with the mining rpc. Ensure that CURL is correctly installed and configured on your system.
