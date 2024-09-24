#include "sha256.h"
#include <cuda_runtime.h>
#include <string.h> 

__device__ const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ void sha256_transform(uint32_t state[8], const uint32_t block[16]) {
    uint32_t a, b, c, d, e, f, g, h, t1, t2;
    uint32_t W[64];

    for (int i = 0; i < 16; ++i)
        W[i] = block[i];
    for (int i = 16; i < 64; ++i)
        W[i] = W[i-16] + W[i-7] + (W[i-15] >> 3) + (W[i-2] >> 10);

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    for (int i = 0; i < 64; ++i) {
        t1 = h + ((e >> 6) ^ (e >> 11) ^ (e >> 25)) + ((e & f) ^ ((~e) & g)) + K[i] + W[i];
        t2 = ((a >> 2) ^ (a >> 13) ^ (a >> 22)) + ((a & b) ^ (a & c) ^ (b & c));

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ void sha256(const uint8_t *input, uint8_t *output) {
    uint32_t state[8] = { /* Initial hash values */ };
    uint32_t block[16];
    
    // Padding and length encoding
    uint64_t bit_len = 8 * 64; // example length
    uint8_t padded_input[64] = {0};
    memcpy(padded_input, input, 64);

    // Padding
    padded_input[0] = 0x80;
    if (64 - (bit_len % 64) < 8) {
        // Additional padding
    }
    // Append length
    for (int i = 0; i < 8; ++i) {
        padded_input[64 - 8 + i] = (bit_len >> (56 - 8 * i)) & 0xFF;
    }

    for (int i = 0; i < 16; ++i)
        block[i] = (padded_input[i*4+0] << 24) | (padded_input[i*4+1] << 16) | (padded_input[i*4+2] << 8) | padded_input[i*4+3];

    sha256_transform(state, block);

    // Write output
    for (int i = 0; i < 8; ++i) {
        output[i*4+0] = (state[i] >> 24) & 0xFF;
        output[i*4+1] = (state[i] >> 16) & 0xFF;
        output[i*4+2] = (state[i] >> 8) & 0xFF;
        output[i*4+3] = state[i] & 0xFF;
    }
}

__global__ void sha256d_kernel(const unsigned char* input, unsigned char* output, int num_hashes) {
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
