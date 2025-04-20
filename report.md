## 1. Introduction

Edge detection is a fundamental task in image processing and computer vision. In this project, we implement the Sobel edge detection algorithm using various parallel programming paradigms—MPI, OpenMP, CUDA, and a hybrid MPI+OpenMP approach. We evaluate their performance on input images of increasing size and compare their efficiency and scalability.

## 2. Objectives

- Implement the Sobel filter using different parallel computing models.
- Benchmark each implementation over multiple image sizes.
- Analyze performance in terms of execution time and scalability.
- Demonstrate how hybrid models can leverage both inter-node and intra-node parallelism.

## 3. Methodology

### 3.1 Algorithm Overview

The Sobel operator is a widely used edge detection method in image processing and computer vision. It works by computing an approximation of the gradient of image intensity, highlighting regions with high spatial frequency that typically correspond to edges. The operator uses a pair of 3×3 convolution kernels to detect changes in horizontal (Gx) and vertical (Gy) directions. The gradient magnitude is then computed from these two components, providing a measure of edge strength at each pixel.

The Sobel operator applies two 3x3 convolution kernels (Gx and Gy) to estimate the gradient magnitude at each pixel. The result is an edge map highlighting sharp changes in intensity.

$$
G_x =
\begin{bmatrix}
-1 & 0 & +1 \\
-2 & 0 & +2 \\
-1 & 0 & +1 \\
\end{bmatrix},
\quad
G_y =
\begin{bmatrix}
-1 & -2 & -1 \\
0 &  0 &  0 \\
+1 & +2 & +1 \\
\end{bmatrix}
$$

For each pixel, the horizontal and vertical gradients \( G_x \) and \( G_y \) are computed via convolution. The final gradient magnitude is calculated as:

$$
G = \sqrt{G_x^2 + G_y^2}
$$

This value is used to determine edge intensity at each pixel.

### 3.2 Parallel Strategies

- **MPI**: The image is partitioned row-wise among processes. Overlapping halo rows are exchanged between neighboring processes. Sobel computation is done independently, and results are gathered.
  
- **OpenMP**: The convolution loop is parallelized across image rows using OpenMP `#pragma` directives.

- **CUDA**: GPU threads are launched in a 2D grid to parallelize pixel-wise operations. Shared memory is optionally used for improved data access.

- **Hybrid**: Each MPI process handles a chunk of the image and applies OpenMP threads within that chunk.

## 4. Implementation

Each implementation was written in C/C++ with the following structure:

- MPI: `mpi/mpi_sobel.c`
- OpenMP: `openmp/omp_sobel.c`
- CUDA: `cuda/cuda_sobel.cu`
- Hybrid: `hybrid/hybrid_sobel.c`

We used `stb_image` for I/O, and benchmark scripts run each test 3 times and report the average execution time.

## 5. Experimental Setup

- System: mcs1.wlu.ca (multicore CPU + CUDA-capable GPU)
- Compilers: `mpicc`, `gcc`, `nvcc`
- Datasets: JPEG images resized to 512x512, 1024x1024, 2048x2048, 4000x4000

## 6. Results

### 6.1 Timing Summary

#### MPI

| Image Size | Config     | Execution Time (s) | Total Time (s) |
|------------|------------|--------------------|----------------|
| 512x512    | 1 Process  | 0.012174           | 0.026008       |
|            | 2 Processes| 0.006471           | 0.020507       |
|            | 4 Processes| 0.005402           | 0.020067       |
|            | 8 Processes| 0.003211           | 0.018324       |
| 1024x1024  | 1 Process  | 0.048680           | 0.101924       |
|            | 2 Processes| 0.025262           | 0.078910       |
|            | 4 Processes| 0.021143           | 0.077750       |
|            | 8 Processes| 0.012020           | 0.067273       |
| 2048x2048  | 1 Process  | 0.194833           | 0.396318       |
|            | 2 Processes| 0.099562           | 0.300289       |
|            | 4 Processes| 0.053257           | 0.260993       |
|            | 8 Processes| 0.030817           | 0.243945       |
| 4000x4000  | 1 Process  | 0.734651           | 1.442408       |
|            | 2 Processes| 0.379577           | 1.092674       |
|            | 4 Processes| 0.203375           | 0.921975       |
|            | 8 Processes| 0.111395           | 0.845566       |

#### OpenMP

| Image Size | Config      | Execution Time (s) | Total Time (s) |
|------------|-------------|--------------------|----------------|
| 512x512    | 1 Thread    | 0.014701           | 0.029211       |
|            | 2 Threads   | 0.007606           | 0.021766       |
|            | 4 Threads   | 0.003905           | 0.018122       |
|            | 8 Threads   | 0.002103           | 0.015754       |
| 1024x1024  | 1 Thread    | 0.055895           | 0.105850       |
|            | 2 Threads   | 0.028571           | 0.079856       |
|            | 4 Threads   | 0.015199           | 0.066502       |
|            | 8 Threads   | 0.007755           | 0.061705       |
| 2048x2048  | 1 Thread    | 0.224288           | 0.413010       |
|            | 2 Threads   | 0.121853           | 0.327364       |
|            | 4 Threads   | 0.061820           | 0.268336       |
|            | 8 Threads   | 0.031104           | 0.236352       |
| 4000x4000  | 1 Thread    | 0.868364           | 1.573971       |
|            | 2 Threads   | 0.451760           | 1.142580       |
|            | 4 Threads   | 0.229483           | 0.940972       |
|            | 8 Threads   | 0.115872           | 0.833919       |

#### CUDA

| Image Size | Execution Time (ms) | Total Program Time (ms) |
|------------|---------------------|--------------------------|
| 512x512    | 0.040555            | 4.816682                |
| 1024x1024  | 0.115424            | 17.480306               |
| 2048x2048  | 0.418869            | 64.778061               |
| 4000x4000  | 0.392021            | 230.974508              |

#### Hybrid (MPI + OpenMP)

| Image Size | Config       | Execution Time (s) | Total Time (s) |
|------------|--------------|--------------------|----------------|
| 512x512    | MPI=1,OMP=1  | 0.003295           | 0.010760       |
|            | MPI=2,OMP=2  | 0.001700           | 0.009184       |
|            | MPI=4,OMP=4  | 0.001033           | 0.008260       |
| 1024x1024  | MPI=1,OMP=1  | 0.013194           | 0.038703       |
|            | MPI=2,OMP=2  | 0.006813           | 0.033307       |
|            | MPI=4,OMP=4  | 0.004452           | 0.031632       |
| 2048x2048  | MPI=1,OMP=1  | 0.052099           | 0.145904       |
|            | MPI=2,OMP=2  | 0.025421           | 0.122941       |
|            | MPI=4,OMP=4  | 0.011220           | 0.110006       |
| 4000x4000  | MPI=1,OMP=1  | 0.201632           | 0.526048       |
|            | MPI=2,OMP=2  | 0.094653           | 0.428769       |
|            | MPI=4,OMP=4  | 0.024128           | 0.359698       |

### 6.2 Observations
 
 - **MPI** shows linear improvement with process count, particularly effective on large images.
 - **OpenMP** benefits well from multithreading, with best gains on 8 threads.
 - **CUDA** achieves the lowest absolute times, especially for large image sizes.
 - **Hybrid** combines both models effectively, outperforming pure MPI or OpenMP for all sizes tested.
 - Overhead becomes less dominant as image size increases, highlighting better scalability in hybrid and CUDA models.
 
 This behavior is visualized in **Figure 1**, which illustrates the hybrid model’s execution time for each image resolution and configuration. Furthermore, **Figure 2** compares the speedup of all implementations on the 4000×4000 image relative to the OpenMP single-thread baseline.
 
 **Figure 1:** Hybrid MPI+OpenMP Execution Time (Kernel Only) Across Image Sizes  
 ![Figure 1](figures/hybrid_exec_time_only.png)
 
 **Figure 2:** Speedup for 4000×4000 Image Based on Total Program Time
 ![Figure 2](figures/speedup_total_program_time.png)

## 7. Parallel Efficiency

### Speedup and Efficiency (4000×4000 image)

#### Based on Execution Time

| Method        | Config        | Time (s)   | Speedup | Efficiency |
|---------------|---------------|------------|---------|------------|
| OpenMP        | 1 Thread      | 0.868364   | 1.00    | 1.00       |
| OpenMP        | 2 Threads     | 0.451760   | 1.92    | 0.96       |
| OpenMP        | 4 Threads     | 0.229483   | 3.78    | 0.95       |
| OpenMP        | 8 Threads     | 0.115872   | 7.49    | 0.94       |
| MPI           | 2 Procs       | 0.379577   | 2.29    | 1.14       |
| MPI           | 4 Procs       | 0.203375   | 4.27    | 1.07       |
| MPI           | 8 Procs       | 0.111395   | 7.79    | 0.97       |
| Hybrid        | MPI=2, OMP=2  | 0.094653   | 9.17    | 2.29*      |
| Hybrid        | MPI=4, OMP=4  | 0.024128   | 35.98   | 2.25       |
| CUDA          | GPU           | 0.000392   | 2214.40 | -          |

> \* Efficiency appears >1 because execution time measurements exclude some initialization overhead. See total program time for full efficiency metrics.

#### Based on Total Program Time (divided by the 3 runs)

| Method        | Config        | Time (s)   | Speedup | Efficiency |
|---------------|---------------|------------|---------|------------|
| OpenMP        | 1 Thread      | 1.573971   | 1.00    | 1.00       |
| OpenMP        | 2 Threads     | 1.142580   | 1.38    | 0.69       |
| OpenMP        | 4 Threads     | 0.940972   | 1.67    | 0.42       |
| OpenMP        | 8 Threads     | 0.833919   | 1.89    | 0.24       |
| MPI           | 2 Procs       | 1.092674   | 1.44    | 0.72       |
| MPI           | 4 Procs       | 0.921975   | 1.71    | 0.43       |
| MPI           | 8 Procs       | 0.845566   | 1.86    | 0.23       |
| Hybrid        | MPI=2, OMP=2  | 0.428769   | 3.67    | 0.92       |
| Hybrid        | MPI=4, OMP=4  | 0.359698   | 4.38    | 0.27       |
| CUDA          | GPU           | 0.230975   | 6.81    | -          |


## 8. Conclusion

We successfully implemented and evaluated multiple parallel strategies for Sobel edge detection. Hybrid MPI+OpenMP offers an excellent balance of performance and scalability on CPU-only nodes. CUDA excels for GPU-bound workloads. The choice of paradigm should be based on hardware availability and data size.

## 9. Future Work

- Extend support for colored (RGB) images.
- Use overlapping tiles in CUDA to improve memory coalescing.

## 10. Algorithm Analysis

The Sobel operator computes gradients using two 3×3 convolutions. This process is highly parallelizable due to the independence of each pixel's calculation.

- **Complexity**: O(n²) for an n×n image
- **Parallel Suitability**: Each output pixel depends on a 3×3 neighborhood; no data dependency across pixels

## 11. Group Information

- **Group Members**: Abderlhman Elrawy, Trisha Reddy Kilaru
- **Submitted File**: `Elrawy_Kilaru_project.zip`

## 12. Appendix A: Compilation & Execution

Use the following commands to build and run each implementation.

### MPI

```bash
cd mpi
mpicc -std=c99 -O2 mpi_sobel.c -o mpi_sobel -lm
mpirun --mca btl self,vader -np 4 ./mpi_sobel ../data/input_2048.jpg
```

### OpenMP

```bash
cd openmp
gcc -fopenmp -std=c99 -O2 omp_sobel.c -o omp_sobel -lm
export OMP_NUM_THREADS=4
./omp_sobel ../data/input_2048.jpg
```

### Hybrid

```bash
cd hybrid
mpicc -fopenmp -std=c99 -O2 hybrid_sobel.c -o hybrid_sobel -lm
export OMP_NUM_THREADS=4
mpirun --mca btl self,vader -np 2 ./hybrid_sobel ../data/input_2048.jpg
```

### CUDA

```bash
cd cuda
nvcc cuda_sobel.cu -o cuda_sobel
./cuda_sobel ../data/input_2048.jpg
```

### Benchmark Scripts

```bash
bash run_mpi_exp.sh
bash run_omp_exp.sh
bash run_cuda_exp.sh
bash run_hybrid_exp.sh
```

## 13. References

- OpenMP Specification: https://www.openmp.org
- MPI Standard: https://www.mpi-forum.org
- CUDA Toolkit Docs: https://docs.nvidia.com/cuda/
- stb_image.h by Sean Barrett