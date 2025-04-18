# Parallel Sobel Edge Detection (MPI, OpenMP, CUDA, Hybrid)

This project explores parallel implementations of the Sobel edge detection algorithm using multiple paradigms:
- **MPI** for distributed-memory CPU systems
- **OpenMP** for shared-memory CPU systems
- **CUDA** for GPU acceleration
- **Hybrid MPI + OpenMP** for node-level and core-level parallelism

---

## Objective

The goal of this project is to analyze and compare parallel programming models for 2D image processing using Sobel edge detection.

We aim to:
- Apply and evaluate parallel strategies (MPI, OpenMP, CUDA)
- Compare execution performance across different architectures (multicore CPU vs GPU)
- Demonstrate hybrid CPU-based parallelism with MPI + OpenMP

---

## Methodology

### Sobel Edge Detection

We compute the image gradient magnitude using the Sobel operator, a common method for edge detection.

### Parallelization Strategies

- **MPI**: Distribute image rows with boundary overlap, using `MPI_Scatterv` and `MPI_Gatherv` for data transfer.
- **OpenMP**: Use `#pragma omp parallel for` to parallelize row-wise Sobel computation.
- **CUDA**: Launch a 2D grid of GPU threads for pixel-wise Sobel filtering using coalesced memory access and shared memory.
- **Hybrid**: Each MPI process handles a chunk of the image and applies OpenMP for multi-threaded processing within its chunk.

---

## Implementation Details

### Directory Structure

```
parallel-edge-detection/
├── cuda/                # CUDA implementation
├── data/                # Input images
├── hybrid/              # Hybrid MPI + OpenMP
├── include/             # Header files (e.g., stb_image)
├── mpi/                 # MPI implementation
├── openmp/              # OpenMP implementation
├── output/              # Output images
├── README.md            # Documentation
```

### Source Files

- `mpi/mpi_sobel.c`
- `openmp/omp_sobel.c`
- `cuda/cuda_sobel.cu`
- `hybrid/hybrid_sobel.c`

---

## Build Instructions

Use the following commands to build each variant.

### MPI

```bash
cd mpi
mpicc -std=c99 -O2 mpi_sobel.c -o mpi_sobel -lm
```

### OpenMP

```bash
cd openmp
gcc -fopenmp -std=c99 -O2 omp_sobel.c -o omp_sobel -lm
```

### Hybrid (MPI + OpenMP)

```bash
cd hybrid
mpicc -fopenmp -std=c99 -O2 hybrid_sobel.c -o hybrid_sobel -lm
```

### CUDA

```bash
cd cuda
nvcc cuda_sobel.cu -o cuda_sobel
```

---

## Execution Examples

### MPI

```bash
mpirun --mca btl self,vader -np 4 ./mpi_sobel ../data/input_2048.jpg
```

### OpenMP

```bash
export OMP_NUM_THREADS=4
./omp_sobel ../data/input_2048.jpg
```

### Hybrid

```bash
export OMP_NUM_THREADS=4
mpirun --mca btl self,vader -np 2 ./hybrid_sobel ../data/input_2048.jpg
```

### CUDA

```bash
./cuda_sobel ../data/input_2048.jpg
```

---

## Benchmarking

Each implementation includes a benchmarking script:
- `run_mpi_exp.sh`
- `run_omp_exp.sh`
- `run_cuda_exp.sh`
- `run_hybrid_exp.sh`

To execute all:
```bash
bash run_mpi_exp.sh
bash run_omp_exp.sh
bash run_cuda_exp.sh
bash run_hybrid_exp.sh
```

---

## Results and Analysis

- `report.md`: Summary
- Output images for verification:
  - `output_mpi.jpg`
  - `output_omp.jpg`
  - `output_cuda.jpg`
  - `output_hybrid.jpg`

---

## Dependencies

- OpenMPI
- GCC with OpenMP
- CUDA Toolkit
- `stb_image.h`, `stb_image_write.h`

---

## Clean Up

To clean the workspace:

```bash
rm mpi/mpi_sobel openmp/omp_sobel cuda/cuda_sobel hybrid/hybrid_sobel\
    mpi/*.log openmp/*.log cuda/*.log hybrid/*.log\
   output/output_*.jpg
```

