# GPU Optimization Notes (V3)

## Context
- **Target:** `convolve_gpu.cu` separable convolution implementation in `src/V3`.
- **Goal:** Improve runtime on repeated calls while keeping the code simple and avoiding link/time regressions.

## Changes Applied
- Added reusable device image buffers (`gDevImgIn`, `gDevTmp`, `gDevImgOut`) with an `ensureImageBuffers` helper and `atexit` cleanup to avoid repeated `cudaMalloc`/`cudaFree` overhead for each call.
- Cached convolution coefficients in device memory (`gDevKernelA`, `gDevKernelB`) so we only copy what we need and skip reallocations.
- Tuned the launch shape to a 32×8 block configuration, providing better warp utilization for typical image widths while leaving the code easy to read.
- Removed the extra `cudaDeviceSynchronize()` between kernels; we now rely on implicit stream ordering and keep the host/device copies guarded with `CUDA_CHECK`.
- Added lightweight helper functions (`makeGrid2d`, `numImageElements`) and inline usage to keep the code tidy without introducing complex abstractions.
- Tweaked the GPU Makefile defaults (`CUDA_ARCH=sm_75`, `CUDA_HOME=/usr/local/cuda`) so Google Colab’s Tesla T4 setup builds cleanly without manual edits.
- Added a `safeCudaFree` helper so end-of-process cleanup ignores the expected `cudaErrorCudartUnloading` when the driver is already shutting down.

## Verification
- Attempted to run `make gpu`; the build fails in this environment because `nvcc` is not installed. No additional errors observed in the host compile stage.
- Recommend rebuilding once CUDA tooling is available and running `./example3_gpu` to ensure runtime behaviour is intact.

## Future Opportunities
- Evaluate introducing shared-memory tiling per axis if additional performance is required.
- Consider small-loop unrolling for common Gaussian widths (e.g., 3, 5, 7) if profiling shows loop overhead dominates.
- Update profiling documentation (`profile.txt`) after running on a CUDA-enabled system to capture new baseline timings.
