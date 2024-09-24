// sha256.h
#ifndef SHA256_H
#define SHA256_H
#include <cuda_runtime.h>
#include <stdint.h>

void sha256_transform(uint32_t state[8], const uint32_t block[16]);
void sha256(const uint8_t *input, uint8_t *output);
__global__ void sha256d_kernel(const unsigned char* input, unsigned char* output, int num_hashes);
void monitor_gpu_temperature();

#endif // SHA256_H
