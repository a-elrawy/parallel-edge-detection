#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE + 2)

__device__ int clamp(int value, int min, int max) {
    return value < min ? min : (value > max ? max : value);
}

__global__ void sobel_shared_memory_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char tile[TILE_SIZE][TILE_SIZE];
    
    int Gx[3][3] = {{-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1},
                    { 0,  0,  0},
                    { 1,  2,  1}};

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    if (x < width && y < height) {
        tile[ty][tx] = input[y * width + x];
    }
    
    if (threadIdx.y == 0) {
        int y_pos = y - 1;
        if (y_pos >= 0 && x < width) {
            tile[0][tx] = input[y_pos * width + x];
        } else {
            tile[0][tx] = 0;
        }
        
        if (threadIdx.x < BLOCK_SIZE) {
            int y_pos = y + BLOCK_SIZE;
            if (y_pos < height && x < width) {
                tile[TILE_SIZE-1][tx] = input[y_pos * width + x];
            } else {
                tile[TILE_SIZE-1][tx] = 0;
            }
        }
    }
    
    if (threadIdx.x == 0) {
        int x_pos = x - 1;
        if (x_pos >= 0 && y < height) {
            tile[ty][0] = input[y * width + x_pos];
        } else {
            tile[ty][0] = 0;
        }
        
        if (threadIdx.y < BLOCK_SIZE) {
            int x_pos = x + BLOCK_SIZE;
            if (x_pos < width && y < height) {
                tile[ty][TILE_SIZE-1] = input[y * width + x_pos];
            } else {
                tile[ty][TILE_SIZE-1] = 0;
            }
        }
    }
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (x > 0 && y > 0) {
            tile[0][0] = input[(y-1) * width + (x-1)];
        } else {
            tile[0][0] = 0;
        }
        
        if (x + BLOCK_SIZE < width && y > 0) {
            tile[0][TILE_SIZE-1] = input[(y-1) * width + (x+BLOCK_SIZE)];
        } else {
            tile[0][TILE_SIZE-1] = 0;
        }
        
        if (x > 0 && y + BLOCK_SIZE < height) {
            tile[TILE_SIZE-1][0] = input[(y+BLOCK_SIZE) * width + (x-1)];
        } else {
            tile[TILE_SIZE-1][0] = 0;
        }
        
        if (x + BLOCK_SIZE < width && y + BLOCK_SIZE < height) {
            tile[TILE_SIZE-1][TILE_SIZE-1] = input[(y+BLOCK_SIZE) * width + (x+BLOCK_SIZE)];
        } else {
            tile[TILE_SIZE-1][TILE_SIZE-1] = 0;
        }
    }
    
    __syncthreads();
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int sumX = 0, sumY = 0;
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int pixel = tile[ty + dy][tx + dx];
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
        sobel_shared_memory_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        total_time += elapsed;
    }
    
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    printf("CUDA Shared Memory Execution Time (avg of 3 runs): %f ms\n", total_time / 3.0f);
    
    stbi_write_jpg("../output/output_cuda.jpg", width, height, 1, h_output, 100);

    stbi_image_free(img);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}