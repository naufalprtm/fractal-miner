#!/bin/bash

# Function to check if a command exists
check_command() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print debug logs
debug_log() {
    echo "[DEBUG] $1"
}

# Check NVIDIA GPU status
debug_log "Checking NVIDIA GPU status..."
if ! nvidia-smi; then
    echo "Error: nvidia-smi command failed. Ensure that NVIDIA drivers are installed and functioning."
    exit 1
fi

# Define required commands/packages
REQUIRED_COMMANDS=("sudo" "apt-get" "nvidia-smi" "gcc-10" "g++-10")
MISSING_COMMANDS=()

# Check for required commands/packages
for cmd in "${REQUIRED_COMMANDS[@]}"; do
    if ! check_command "$cmd"; then
        MISSING_COMMANDS+=("$cmd")
    else
        debug_log "$cmd is installed."
    fi
done

# Handle missing commands/packages
if [ ${#MISSING_COMMANDS[@]} -ne 0 ]; then
    echo "The following required commands/packages are missing: ${MISSING_COMMANDS[*]}"
    echo "Choose an option:"
    echo "1) Install the missing packages"
    echo "2) Exit"
    read -p "Enter your choice [1-2]: " choice

    if [ "$choice" -eq 1 ]; then
        # Update package list
        debug_log "Updating package list..."
        if ! sudo apt-get update; then
            echo "Error: Failed to update package list."
            exit 1
        fi

        # Install additional dependencies
        debug_log "Installing required packages..."
        if ! sudo apt-get install -y build-essential cuda-toolkit libcurl4-openssl-dev libssl-dev libnghttp2-dev; then
            echo "Error: Failed to install required packages."
            exit 1
        fi
        
        # Install GCC and G++ version 10
        debug_log "Installing gcc-10 and g++-10..."
        if ! sudo apt install -y gcc-10 g++-10; then
            echo "Error: Failed to install gcc-10 and g++-10."
            exit 1
        fi

        # Set gcc and g++ alternatives
        debug_log "Setting gcc and g++ alternatives..."
        if ! sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 60 --slave /usr/bin/g++ g++ /usr/bin/g++-10; then
            echo "Error: Failed to set gcc alternatives."
            exit 1
        fi
        
        # Configure gcc alternatives
        debug_log "Configuring gcc alternatives..."
        if ! sudo update-alternatives --config gcc; then
            echo "Error: Failed to configure gcc alternatives."
            exit 1
        fi

    else
        echo "Exiting..."
        exit 0
    fi
else
    debug_log "All required commands are installed."
fi

# Check for required headers
debug_log "Checking for required headers..."
REQUIRED_HEADERS=(
    "/usr/include/cuda_runtime.h"
    "/usr/include/nlohmann/json.hpp"
    "/usr/include/sha256.h"
    "/usr/include/monitor_gpu_temperature.h"
    "/usr/include/mining_pool.h"
)

MISSING_HEADERS=()

for header in "${REQUIRED_HEADERS[@]}"; do
    if [ ! -f "$header" ]; then
        MISSING_HEADERS+=("$header")
        debug_log "Header not found: $header"
    else
        debug_log "Header found: $header"
    fi
done

# Handle missing headers
if [ ${#MISSING_HEADERS[@]} -ne 0 ]; then
    echo "The following required headers are missing: ${MISSING_HEADERS[*]}"
    echo "Do you want to install nlohmann-json3-dev? (This might include other dependencies)"
    echo "Choose an option:"
    echo "1) Install the missing packages"
    echo "2) Exit"
    read -p "Enter your choice [1-2]: " header_choice

    if [ "$header_choice" -eq 1 ]; then
        debug_log "Installing nlohmann-json3-dev..."
        if ! sudo apt install -y nlohmann-json3-dev; then
            echo "Error: Failed to install nlohmann-json3-dev."
            exit 1
        fi
    else
        echo "Exiting..."
        exit 0
    fi
fi

# Check for nlohmann/json.hpp explicitly
debug_log "Checking if nlohmann/json.hpp is present..."
if [ -f "/usr/include/nlohmann/json.hpp" ]; then
    debug_log "nlohmann/json.hpp is present."
else
    echo "Error: nlohmann/json.hpp is not found. Please install nlohmann-json3-dev."
    exit 1
fi

# Check for library presence (optional)
debug_log "Checking if required libraries are installed..."
REQUIRED_LIBRARIES=("libcurl" "libssl" "libnghttp2")
MISSING_LIBRARIES=()

for lib in "${REQUIRED_LIBRARIES[@]}"; do
    if ! ldconfig -p | grep -q "$lib"; then
        echo "Error: Library $lib is not installed."
        MISSING_LIBRARIES+=("$lib")
    else
        debug_log "Library $lib is installed."
    fi
done

# Handle missing libraries
if [ ${#MISSING_LIBRARIES[@]} -ne 0 ]; then
    echo "The following required libraries are missing: ${MISSING_LIBRARIES[*]}"
    echo "Choose an option:"
    echo "1) Install the missing libraries"
    echo "2) Exit"
    read -p "Enter your choice [1-2]: " lib_choice

    if [ "$lib_choice" -eq 1 ]; then
        for lib in "${MISSING_LIBRARIES[@]}"; do
            case $lib in
                "libcurl")
                    debug_log "Installing libcurl..."
                    sudo apt install -y libcurl4-openssl-dev
                    ;;
                "libssl")
                    debug_log "Installing libssl..."
                    sudo apt install -y libssl-dev
                    ;;
                "libnghttp2")
                    debug_log "Installing libnghttp2..."
                    sudo apt install -y libnghttp2-dev
                    ;;
                *)
                    echo "Error: Unknown library $lib."
                    ;;
            esac
        done
    else
        echo "Exiting..."
        exit 0
    fi
fi

# Create symbolic links for gcc and g++ in CUDA directory
debug_log "Creating symbolic links for gcc and g++ in CUDA directory..."
export CUDA_ROOT=/usr/local/cuda
if ! sudo ln -sf /usr/bin/gcc-10 "$CUDA_ROOT/bin/gcc"; then
    echo "Error: Failed to create symbolic link for gcc."
    exit 1
fi
if ! sudo ln -sf /usr/bin/g++-10 "$CUDA_ROOT/bin/g++"; then
    echo "Error: Failed to create symbolic link for g++."
    exit 1
fi

# Update .bashrc for PATH and LD_LIBRARY_PATH
debug_log "Updating .bashrc for CUDA environment variables..."
{
    echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH'
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH'
} >> ~/.bashrc

debug_log "Sourcing .bashrc to apply changes..."
source ~/.bashrc

debug_log "Script completed successfully."
