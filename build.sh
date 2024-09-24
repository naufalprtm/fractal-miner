#!/bin/bash

# Remove the build directory if it exists
rm -rf build

# Create the build directory
mkdir build

# Navigate into the build directory
cd build

# Run cmake with the desired settings
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc -DCMAKE_CXX_STANDARD=17

# Run make to build the project
make

# Move the built binary to the parent directory
mv fractal-miner ../

# Navigate back to the parent directory
cd ..

# Run the fractal-miner application
./fractal-miner
