#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int clamp(int value, int min, int max) {
    return value < min ? min : (value > max ? max : value);
}

void apply_sobel(unsigned char* input, unsigned char* output, int width, int height) {
    const int Gx_flat[9] = {-1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1};
    const int Gy_flat[9] = {-1, -2, -1,
                             0,  0,  0,
                             1,  2,  1};

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int sumX = 0, sumY = 0;
            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    int pixel = input[(i + di) * width + (j + dj)];
                    int k = (di + 1) * 3 + (dj + 1);
                    sumX += Gx_flat[k] * pixel;
                    sumY += Gy_flat[k] * pixel;
                }
            }
            int magnitude = (int)sqrt(sumX * sumX + sumY * sumY);
            output[i * width + j] = clamp(magnitude, 0, 255);
        }
    }
}

int main(int argc, char *argv[]) {
    double program_start = omp_get_wtime();
    
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

    size_t img_size = width * height;
    unsigned char *output = (unsigned char *)calloc(img_size, 1);
    if (!output) {
        printf("Failed to allocate memory.\n");
        stbi_image_free(img);
        return 1;
    }

    double total = 0.0;
    for (int run = 0; run < 3; ++run) {
        double start = omp_get_wtime();
        apply_sobel(img, output, width, height);
        double end = omp_get_wtime();
        total += (end - start);
    }
    double average = total / 3.0;
    printf("OpenMP Execution Time with (avg of 3 runs): %f seconds\n", average);


    stbi_write_jpg("../output/output_omp.jpg", width, height, 1, output, 100);

    stbi_image_free(img);
    free(output);
    
    double program_end = omp_get_wtime();
    printf("OpenMP Total Program Time: %f seconds\n", (program_end - program_start) / 3.0);
    
    return 0;
}