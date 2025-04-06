#!/bin/bash

# Output log file
LOG_FILE="cuda_results.log"
echo "CUDA Performance Test Results" > $LOG_FILE
echo "=============================" >> $LOG_FILE
echo "" >> $LOG_FILE

# Compile the CUDA program
nvcc -o cuda_sobel cuda_sobel.cu -lm
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

    ./cuda_sobel $INPUT >> $LOG_FILE
    echo "" >> $LOG_FILE
    echo "-----------------------------------" >> $LOG_FILE
done

echo "Tests completed. Results saved in $LOG_FILE"
