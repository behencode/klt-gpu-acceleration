# V2 Performance Report

## What Changed
- GPU kernels now handle the separable convolutions (horizontal and vertical) in `convolve_gpu.cu`.
- Both CPU and GPU paths use the same Gaussian weights via `_KLTGetKernels`.
- The public API still calls `_KLTComputeGradients` and `_KLTComputeSmoothedImage`; they pick CPU or GPU at compile time.
- The Makefile can build the regular CPU library (`libklt.a`) and a GPU library (`libklt_gpu.a`) plus `example3` / `example3_gpu` binaries.

## Test Setup
- Example: `example3` (default 320×240 image sequence from V1).
- CPU build command: `make && ./example3`.
- GPU build command: `make gpu NVCC=/usr/local/cuda/bin/nvcc && ./example3_gpu`.
- Timing tool: `/usr/bin/time -f '%E real, %M KB maxrss'`.
- Note: this system does not have `nvcc`, so only CPU numbers are recorded here.

## Results
| Version | Command | Real Time | Max RSS | Comment |
|---------|---------|-----------|---------|---------|
| CPU     | `./example3`      | 0.49 s | ~5.4 MB | Works end-to-end. |
| GPU     | `./example3_gpu`  | pending | pending | Build/run once CUDA tools are installed. |

To fill in the GPU row, run the build on a CUDA machine and time it with the same `/usr/bin/time` command.

## Current Bottlenecks
1. Every GPU call copies the full image to and from the device. This will hurt for bigger frames.
2. Device memory is allocated and freed on each call. Reusing buffers will help later.
3. Kernels use a simple 16×16 block layout. Tuning block size and using shared memory should cut runtime.
4. We synchronize after each vertical pass. Using streams/events could hide some of that cost.

## Next Steps
- Gather the GPU timing on the target server and update the table.
- Add buffer reuse and tune launch configs.
- Explore shared-memory tiling and kernel fusion for higher speed.
- Profile other hot spots (feature selection, tracking loops) once convolution speedups are confirmed.
