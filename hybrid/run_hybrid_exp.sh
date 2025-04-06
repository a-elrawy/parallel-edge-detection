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
      total=0
      count=0

      for i in {1..3}; do
        output=$(mpirun --mca btl self,vader -np "$np" ./hybrid_sobel "$IMG")
        result=$(echo "$output" | grep "Hybrid MPI+OpenMP Time" | awk '{print $(NF-1)}')

        if [[ $result =~ ^[0-9.]+$ ]]; then
          echo "Run $i: $result s" >> "$OUTPUT_LOG"
          total=$(echo "$total + $result" | bc -l)
          ((count++))
        else
          echo "Run $i: FAILED (no valid timing output)" >> "$OUTPUT_LOG"
        fi
      done

      if [[ $count -gt 0 ]]; then
        avg=$(echo "$total / $count" | bc -l)
        echo "Average Time: $avg seconds" >> "$OUTPUT_LOG"
      else
        echo "Average Time: N/A (all runs failed)" >> "$OUTPUT_LOG"
      fi

      echo "----------------------------------------" >> "$OUTPUT_LOG"
    done
  done

  echo "" >> "$OUTPUT_LOG"
done
