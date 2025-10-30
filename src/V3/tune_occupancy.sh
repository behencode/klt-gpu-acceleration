#!/bin/bash

echo "Testing different block sizes for occupancy..."

for BLOCK in 8 16 32
do
    echo "Building with BLOCK_DIM=$BLOCK"
    make clean
    make gpu NVCC=nvcc CUDA_ARCH="-arch=sm_75 -DBLOCK_DIM=$BLOCK"
    echo "Running with BLOCK_DIM=$BLOCK"
    /usr/bin/time -v ./example3_gpu 2>&1 | grep -E "(elapsed|Maximum resident)"
    echo "---"
done
!git clone https://github.com/behencode/KLT-GPU-ACCELERATION.git