#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MASTER 0

int clamp(int value, int min, int max) {
    return value < min ? min : (value > max ? max : value);
}

void apply_sobel(unsigned char* input, unsigned char* output, int width, int height) {
    int Gx[3][3] = {{-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1},
                    { 0,  0,  0},
                    { 1,  2,  1}};
    
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
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double overall_start_time = MPI_Wtime();
    
    int width, height, channels;
    unsigned char *img = NULL;
    unsigned char *local_chunk = NULL;
    unsigned char *local_output = NULL;
    unsigned char *gathered = NULL;
    int *sendcounts = NULL, *displs = NULL;
    int chunk_rows;

    if (rank == MASTER) {
        if (argc < 2) {
            printf("Usage: %s <input_image_path>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        img = stbi_load(argv[1], &width, &height, &channels, 1);
        if (!img) {
            printf("Failed to load image.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
     
    double total_time = 0.0;
    for (int run = 0; run < 3; run++) {
        double start_time = MPI_Wtime();

        MPI_Bcast(&width, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
        MPI_Bcast(&height, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
 
        chunk_rows = height / size;
        int extra = height % size;
        int start_row = rank * chunk_rows + (rank < extra ? rank : extra);
        int local_height = chunk_rows + (rank < extra);
 
        int padded_height = local_height + 2;
        local_chunk = (unsigned char *)malloc(width * padded_height);
        local_output = (unsigned char *)calloc(width * padded_height, 1);
 
        if (rank == MASTER) {
            sendcounts = malloc(size * sizeof(int));
            displs = malloc(size * sizeof(int));
            int offset = 0;
            for (int i = 0; i < size; i++) {
                int rows = chunk_rows + (i < extra);
                sendcounts[i] = rows * width;
                displs[i] = offset;
                offset += sendcounts[i];
            }
        }
 
        MPI_Scatterv(img, sendcounts, displs, MPI_UNSIGNED_CHAR,
                     &local_chunk[width], local_height * width,
                     MPI_UNSIGNED_CHAR, MASTER, MPI_COMM_WORLD);
 
        if (rank != 0) {
            MPI_Send(&local_chunk[width], width, MPI_UNSIGNED_CHAR, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(local_chunk, width, MPI_UNSIGNED_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            for (int i = 0; i < width; i++) local_chunk[i] = 0;
        }
 
        if (rank != size - 1) {
            MPI_Recv(&local_chunk[(local_height + 1) * width], width, MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&local_chunk[local_height * width], width, MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD);
        } else {
            for (int i = 0; i < width; i++) local_chunk[(local_height + 1) * width + i] = 0;
        }
 
        apply_sobel(local_chunk, local_output, width, padded_height);
 
        if (rank == MASTER) {
            gathered = malloc(width * height);
        }
 
        MPI_Gatherv(&local_output[width], local_height * width, MPI_UNSIGNED_CHAR,
                    gathered, sendcounts, displs, MPI_UNSIGNED_CHAR,
                    MASTER, MPI_COMM_WORLD);
    
        double end_time = MPI_Wtime();
        total_time += (end_time - start_time);
    }

    if (rank == MASTER) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "../output/output_mpi_%dx%d.jpg", width, height);
        stbi_write_jpg(output_filename, width, height, 1, gathered, 100);
        stbi_image_free(img);
        free(gathered);
        free(sendcounts);
        free(displs);
    }
 
    free(local_chunk);
    free(local_output);
     
    if (rank == MASTER) {
        printf("MPI Execution Time (avg of 3 runs): %f seconds\n", total_time / 3.0);
    }
     
    double overall_end_time = MPI_Wtime();
    if (rank == MASTER) {
        printf("MPI Total Program Time: %f seconds\n", (overall_end_time - overall_start_time) / 3.0 );
    }
    MPI_Finalize();
    return 0;
}