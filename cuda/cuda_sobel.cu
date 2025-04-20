#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__device__ int clamp(int value, int min, int max) {
    return value < min ? min : (value > max ? max : value);
}

__global__ void sobel_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    int Gx[3][3] = {{-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1},
                    { 0,  0,  0},
                    { 1,  2,  1}};

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int sumX = 0, sumY = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int pixel = input[(y + dy) * width + (x + dx)];
                sumX += Gx[dy + 1][dx + 1] * pixel;
                sumY += Gy[dy + 1][dx + 1] * pixel;
            }
        }
        int mag = (int)sqrtf((float)(sumX * sumX + sumY * sumY));
        output[y * width + x] = clamp(mag, 0, 255);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_image_path>\n", argv[0]);
        return 1;
    }

    cudaEvent_t full_start, full_stop;
    cudaEventCreate(&full_start);
    cudaEventCreate(&full_stop);
    cudaEventRecord(full_start);

    int width, height, channels;
    unsigned char *img = stbi_load(argv[1], &width, &height, &channels, 1);
    if (!img) {
        printf("Failed to load image.\n");
        return 1;
    }

    size_t img_size = width * height * sizeof(unsigned char);
    unsigned char *d_input, *d_output, *h_output;

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);
    h_output = (unsigned char *)malloc(img_size);

    cudaMemcpy(d_input, img, img_size, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float total_time = 0.0f;
    for (int i = 0; i < 3; ++i) {
        cudaEventRecord(start);
        sobel_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        total_time += elapsed;
    }
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    printf("CUDA Execution Time (avg of 3 runs): %f ms\n", total_time / 3.0f);
    stbi_write_jpg("../output/output_cuda.jpg", width, height, 1, h_output, 100);

    stbi_image_free(img);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventRecord(full_stop);
    cudaEventSynchronize(full_stop);
    float full_elapsed;
    cudaEventElapsedTime(&full_elapsed, full_start, full_stop);
    printf("CUDA Total Program Time: %f ms\n", (full_elapsed) / 3.0f);

    return 0;
}
