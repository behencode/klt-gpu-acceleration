#!/bin/bash
# Build script for Google Colab with T4 GPU - PGI Version (RECOMMENDED)

set -e

echo "=========================================="
echo "KLT V4 - Build for T4 GPU (Colab)"
echo "=========================================="

# Check for GPU
echo ""
echo "Checking for T4 GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: No GPU detected!"

# Try to install PGI/NVHPC if not present
echo ""
echo "Checking for PGI/NVHPC compiler..."
if ! command -v pgcc &> /dev/null; then
    echo "Installing NVIDIA HPC SDK (PGI compiler)..."
    apt-get update > /dev/null 2>&1
    apt-get install -y nvhpc > /dev/null 2>&1 || echo "WARNING: Could not install nvhpc"
fi

# Navigate to V4 directory
cd /content/klt-gpu-acceleration/src/V4 2>/dev/null || cd V4

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
make clean

# Try PGI first, fall back to GCC
if command -v pgcc &> /dev/null; then
    echo "Using PGI/NVHPC compiler (better T4 support)..."
    export CC=pgcc
    export CFLAGS="-acc=gpu -gpu=cc75 -O3 -DNDEBUG"
    make lib
    # Manual build of examples
    pgcc -O3 -acc=gpu -gpu=cc75 -DNDEBUG -o example3_acc example3.c -L. -lklt -lm
    
elif command -v gcc &> /dev/null; then
    echo "Using GCC with OpenACC (CPU fallback)..."
    export CC=gcc
    export CFLAGS="-fopenacc -O3 -DNDEBUG"
    make lib
    gcc -fopenacc -O3 -DNDEBUG -o example3_acc example3.c -L. -lklt -lm
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
echo "To compare with V3 CUDA:"
echo "  time ../V3/example3_gpu"
echo ""
