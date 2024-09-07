#!/bin/bash

# Exit on error
set -e

# Print each command before executing
set -x

# Source .bashrc to ensure all environment variables are set
echo "Sourcing ~/.bashrc..."
source ~/.bashrc

# Print NVIDIA GPU status
echo "Checking NVIDIA GPU status..."
if ! nvidia-smi; then
    echo "Error: nvidia-smi command failed. Ensure that NVIDIA drivers are installed and functioning."
    exit 1
fi

# Update and install necessary packages
echo "Updating package list and installing dependencies..."
if ! sudo apt-get update; then
    echo "Error: Failed to update package list."
    exit 1
fi

# Install additional dependencies
if ! sudo apt-get install -y build-essential cuda-toolkit libcurl4-openssl-dev libssl-dev libnghttp2-dev; then
    echo "Error: Failed to install required packages."
    exit 1
fi

# Ensure the CUDA libraries are properly linked
echo "Setting up LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Define the paths to include and library directories
INCLUDE_DIR="$HOME/fractal/vcpkg/installed/x64-linux/include"
LIB_DIR="$HOME/fractal/vcpkg/installed/x64-linux/lib"

# Print paths for debugging
echo "Include directory: $INCLUDE_DIR"
echo "Library directory: $LIB_DIR"

# Check if cURL version is correct
REQUIRED_CURL_VERSION="7.81.0"
echo "Checking cURL version..."
if command -v curl &> /dev/null; then
    INSTALLED_CURL_VERSION=$(curl --version | head -n 1 | awk '{print $2}')
    if [ "$INSTALLED_CURL_VERSION" == "$REQUIRED_CURL_VERSION" ]; then
        echo "cURL version $REQUIRED_CURL_VERSION is already installed. Skipping download and installation."
        exit 0
    else
        echo "cURL version $INSTALLED_CURL_VERSION found, but not the required version $REQUIRED_CURL_VERSION. Proceeding with installation."
    fi
else
    echo "cURL is not installed. Proceeding with installation."
fi

# Download, build, and install cURL
echo "Downloading and installing cURL..."
if ! wget https://curl.se/download/curl-7.81.0.tar.gz; then
    echo "Error: Failed to download cURL."
    exit 1
fi

if ! tar -xzvf curl-7.81.0.tar.gz; then
    echo "Error: Failed to extract cURL source."
    exit 1
fi

if ! rm -rf curl-7.81.0.tar.gz; then
    echo "Error: Remove curl-7.81.0.tar.gz."
    exit 1
fi

cd curl-7.81.0

if ! ./configure --with-ssl; then
    echo "Error: cURL configuration failed."
    exit 1
fi

if ! make; then
    echo "Error: cURL build failed."
    exit 1
fi

if ! sudo make install; then
    echo "Error: cURL installation failed."
    exit 1
fi

cd ..

# Compile the code
echo "Compiling the code..."
if ! nvcc -arch=sm_86 -o fractal_miner main.cu -lcudart -lcurl -lssl -lcrypto -lz \
-I$INCLUDE_DIR \
-L$LIB_DIR \
-lnvidia-ml; then
    echo "Error: Compilation failed."
    exit 1
fi

# Print a success message
echo "Compilation complete. The 'fractal_miner' executable is ready."
