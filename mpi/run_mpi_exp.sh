#!/bin/bash

# Output log file
LOG_FILE="mpi_results.log"
echo "MPI Performance Test Results" > $LOG_FILE
echo "============================" >> $LOG_FILE
echo "" >> $LOG_FILE

# Compile the program
mpicc -std=c99 mpi_sobel.c -o mpi_sobel -lm
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

# Test parameters
IMAGE_SIZES=("512" "1024" "2048" "4000")
PROCESS_COUNTS=(1 2 4 8)

for size in "${IMAGE_SIZES[@]}"; do
    INPUT="../data/input_${size}.jpg"
    echo "Testing on ${size}x${size} image" >> $LOG_FILE
    echo "Input: $INPUT" >> $LOG_FILE

    if [ ! -f "$INPUT" ]; then
        echo "File $INPUT not found, skipping..." >> $LOG_FILE
        echo "" >> $LOG_FILE
        continue
    fi

    for np in "${PROCESS_COUNTS[@]}"; do
        echo "Running with $np processes..." >> $LOG_FILE
        mpirun --mca btl ^openib -np $np ./mpi_sobel $INPUT >> $LOG_FILE
        echo "" >> $LOG_FILE
    done
    echo "-----------------------------------" >> $LOG_FILE
done

echo "Tests completed. Results saved in $LOG_FILE"