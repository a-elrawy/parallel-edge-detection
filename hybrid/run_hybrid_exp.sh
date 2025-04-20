#!/bin/bash


mpicc -fopenmp -std=c99 -O2 hybrid_sobel.c -o hybrid_sobel -lm

OUTPUT_LOG="hybrid_benchmark.log"
echo "Hybrid MPI+OpenMP Benchmark" > "$OUTPUT_LOG"
echo "==============================" >> "$OUTPUT_LOG"
echo "" >> "$OUTPUT_LOG"

# Input images to test
images=(../data/input_512.jpg ../data/input_1024.jpg ../data/input_2048.jpg ../data/input_4000.jpg)

# MPI and OpenMP combinations
mpi_procs=(1 2 4)
omp_threads=(1 2 4)

for IMG in "${images[@]}"; do
  echo "Testing Image: $IMG" >> "$OUTPUT_LOG"
  echo "----------------------------------------" >> "$OUTPUT_LOG"

  for np in "${mpi_procs[@]}"; do
    for nt in "${omp_threads[@]}"; do
      export OMP_NUM_THREADS=$nt
      echo "MPI=$np, OMP=$nt" >> "$OUTPUT_LOG"
      mpirun --mca btl self,vader -np "$np" ./hybrid_sobel "$IMG" >> "$OUTPUT_LOG"
      echo "----------------------------------------" >> "$OUTPUT_LOG"
    done
  done

  echo "" >> "$OUTPUT_LOG"
done
