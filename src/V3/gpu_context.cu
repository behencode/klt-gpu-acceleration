#include <cuda_runtime.h>
#include "gpu_context.h"
#include "error.h"

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
      KLTError("CUDA error %s at %s:%d", cudaGetErrorString(err__), __FILE__, __LINE__); \
    } \
  } while (0)

static GPUContext global_ctx = {NULL, NULL, NULL, NULL, NULL, 0, 0, 0};

void initGPUContext(GPUContext *ctx, int ncols, int nrows, int max_kernel_width) {
    size_t imageBytes = (size_t)ncols * nrows * sizeof(float);
    size_t kernelBytes = max_kernel_width * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&ctx->d_buffer1, imageBytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_buffer2, imageBytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_buffer3, imageBytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_hkernel, kernelBytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_vkernel, kernelBytes));
    
    ctx->allocated_ncols = ncols;
    ctx->allocated_nrows = nrows;
    ctx->allocated_kernel_size = max_kernel_width;
}

void freeGPUContext(GPUContext *ctx) {
    if (ctx->d_buffer1) cudaFree(ctx->d_buffer1);
    if (ctx->d_buffer2) cudaFree(ctx->d_buffer2);
    if (ctx->d_buffer3) cudaFree(ctx->d_buffer3);
    if (ctx->d_hkernel) cudaFree(ctx->d_hkernel);
    if (ctx->d_vkernel) cudaFree(ctx->d_vkernel);
    ctx->d_buffer1 = ctx->d_buffer2 = ctx->d_buffer3 = NULL;
    ctx->d_hkernel = ctx->d_vkernel = NULL;
    ctx->allocated_ncols = ctx->allocated_nrows = ctx->allocated_kernel_size = 0;
}

GPUContext* getGlobalGPUContext(void) {
    return &global_ctx;
}