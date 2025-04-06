#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>

int clamp(int val, int min, int max) {
    return val < min ? min : (val > max ? max : val);
}

void apply_sobel(unsigned char *input, unsigned char *output, int width, int height) {
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int sumX = 0, sumY = 0;
            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    int pixel = input[(i + di) * width + (j + dj)];
                    sumX += Gx[di + 1][dj + 1] * pixel;
                    sumY += Gy[di + 1][dj + 1] * pixel;
                }
            }
            int magnitude = (int)sqrt(sumX * sumX + sumY * sumY);
            output[i * width + j] = clamp(magnitude, 0, 255);
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <input_image>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int width, height, channels;
    unsigned char *img = NULL;

    if (rank == 0) {
        img = stbi_load(argv[1], &width, &height, &channels, 1);
        if (!img) {
            printf("Failed to load image.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_height = height / size;
    int remainder = height % size;
    int local_height = chunk_height + (rank < remainder ? 1 : 0);
    int start_row = chunk_height * rank + (rank < remainder ? rank : remainder);
    int padded_height = local_height + 2;

    unsigned char *local_input = malloc(width * padded_height);
    unsigned char *local_output = calloc(width * padded_height, 1);

    if (rank == 0) {
        for (int r = 0; r < size; r++) {
            int h = chunk_height + (r < remainder ? 1 : 0);
            int sr = chunk_height * r + (r < remainder ? r : remainder);
            for (int i = 0; i < h + 2; i++) {
                int src_row = clamp(sr + i - 1, 0, height - 1);
                if (r == 0) {
                    memcpy(local_input + i * width, img + src_row * width, width);
                } else {
                    MPI_Send(img + src_row * width, width, MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        for (int i = 0; i < padded_height; i++) {
            MPI_Recv(local_input + i * width, width, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    double total_time = 0.0;
    for (int run = 0; run < 3; run++) {
        double start = MPI_Wtime();
        apply_sobel(local_input, local_output, width, padded_height);
        double end = MPI_Wtime();
        total_time += (end - start);
    }

    unsigned char *gathered = NULL;
    int *recvcounts = NULL, *displs = NULL;
    int local_size = local_height * width;

    if (rank == 0) {
        gathered = malloc(width * height);
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));

        for (int r = 0, offset = 0; r < size; r++) {
            int h = chunk_height + (r < remainder ? 1 : 0);
            recvcounts[r] = h * width;
            displs[r] = offset;
            offset += recvcounts[r];
        }
    }

    MPI_Gatherv(local_output + width, local_size, MPI_UNSIGNED_CHAR,
                gathered, recvcounts, displs, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        stbi_write_jpg("../output/output_hybrid.jpg", width, height, 1, gathered, 100);
        printf("Hybrid MPI+OpenMP Time: %f seconds\n", total_time / 3.0);
        stbi_image_free(img);
        free(gathered);
        free(recvcounts);
        free(displs);
    }

    free(local_input);
    free(local_output);

    MPI_Finalize();
    return 0;
}
