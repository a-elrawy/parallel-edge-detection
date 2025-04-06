#!/bin/bash

# Output log file
LOG_FILE="omp_results.log"
echo "OpenMP Performance Test Results" > $LOG_FILE
echo "===============================" >> $LOG_FILE
echo "" >> $LOG_FILE

# Compile the program
gcc -fopenmp -std=c99 omp_sobel.c -o omp_sobel -lm
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

# Test parameters
IMAGE_SIZES=("512" "1024" "2048" "4000")

for size in "${IMAGE_SIZES[@]}"; do
    INPUT="../data/input_${size}.jpg"
    echo "Testing on ${size}x${size} image" >> $LOG_FILE
    echo "Input: $INPUT" >> $LOG_FILE

    if [ ! -f "$INPUT" ]; then
        echo "File $INPUT not found, skipping..." >> $LOG_FILE
        echo "" >> $LOG_FILE
        continue
    fi

    ./omp_sobel $INPUT >> $LOG_FILE
    echo "" >> $LOG_FILE
    echo "-----------------------------------" >> $LOG_FILE
done

echo "Tests completed. Results saved in $LOG_FILE"
