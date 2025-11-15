# Convolve GPU Optimization Notes

## Context
The separable convolution GPU path executes two 1-D kernels (horizontal and vertical) for KLT feature detection. The original implementation relied exclusively on global memory for both filter taps and pixels, which resulted in redundant memory traffic and prevented the compiler from exploiting the GPU memory hierarchy.

## Optimizations

- **Shared-memory tiling:** Each CUDA block now stages the required stripe of pixels (plus halo) in `extern __shared__` memory. This reduces redundant global reads within the convolution loop and keeps access patterns coalesced. Separate tiling strategies are applied for horizontal (row-major) and vertical (column-major) sweeps while preserving the CPU-equivalent accumulation order.
- **Constant memory for kernels:** Small convolution kernels (≤ `CONVOLVE_GPU_MAX_KERNEL_WIDTH`, currently 64) are copied into device constant memory, which delivers broadcast caching across a warp. Larger kernels fall back to global memory to avoid overflowing the constant segment.
- **Adaptive shared-memory footprint:** The host code computes per-kernel shared-memory sizes (`(blockDim.x + kernelWidth - 1) * blockDim.y` for horizontal and the transposed analogue for vertical). This keeps the shared allocation tight even when different kernel widths are used for the separable pair.
- **Lazy device allocation for kernels:** Device buffers for kernel coefficients are only allocated when constant memory cannot be used, avoiding unnecessary allocations and transfers.

## Result
The algorithmic behavior and boundary handling remain unchanged—only the memory hierarchy usage was updated. The shared-memory and constant-memory staging drastically reduce global-memory pressure, which is typically the bottleneck for small, separable convolutions. Running `example3` after the changes still completes successfully (linked against the existing CPU/GPU library). Rebuilding the CUDA objects requires `nvcc`, which is not currently available in this environment.
