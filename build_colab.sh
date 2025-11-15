#!/bin/bash
# Build script for Google Colab with T4 GPU
# Run this in your Colab notebook

set -e

echo "=========================================="
echo "KLT V4 - OpenACC Build for T4 GPU (Colab)"
echo "=========================================="

# Check for GPU
echo ""
echo "Checking for T4 GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: No GPU detected!"

# Navigate to V4 directory
cd /content/klt-gpu-acceleration/src/V4 2>/dev/null || cd V4

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
make clean

# Set environment for GPU
export ACC_DEVICE_TYPE=nvidia
export CUDA_VISIBLE_DEVICES=0

# Try to use PGI compiler if available
if command -v pgcc &> /dev/null; then
    echo "Using PGI/NVHPC compiler for better T4 support..."
    export CC=pgcc
    make OPENACC_FLAGS="-acc=gpu -gpu=cc75 -O3" -B acc
elif command -v gcc &> /dev/null; then
    echo "Using GCC with OpenACC..."
    make -B acc
else
    echo "ERROR: No C compiler found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "To run benchmarks, use:"
echo "  cd /content/klt-gpu-acceleration/src/V4"
echo "  time ./example3_acc"
echo ""
