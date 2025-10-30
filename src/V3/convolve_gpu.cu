#include <assert.h>
#include <cuda_runtime.h>

extern "C" {
#include "convolve.h"
#include "error.h"
}

#define CUDA_CHECK(call)                                                             \
  do {                                                                               \
    cudaError_t err__ = (call);                                                      \
    if (err__ != cudaSuccess) {                                                      \
      KLTError("CUDA error %s (%d) at %s:%d", cudaGetErrorString(err__),             \
               static_cast<int>(err__), __FILE__, __LINE__);                          \
    }                                                                                \
  } while (0)

static dim3 makeGrid2d(int ncols, int nrows, dim3 block)
{
  return dim3((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);
}
#define BLOCK_DIM 16
#define TILE_WIDTH 16

// Optimized horizontal convolution with shared memory
__global__ void convolveHorizontalKernelOptimized(
    const float *imgin,
    float *imgtmp,
    const float *kernel,
    int kernelWidth,
    int ncols,
    int nrows)
{
    __shared__ float s_data[BLOCK_DIM][BLOCK_DIM + 70]; // 70 = max kernel radius * 2
    
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int radius = kernelWidth / 2;
    
    // Load data into shared memory with halo
    if (row < nrows) {
        int sharedCol = threadIdx.x + radius;
        
        // Load main data
        if (col < ncols) {
            s_data[threadIdx.y][sharedCol] = imgin[row * ncols + col];
        }
        
        // Load left halo
        if (threadIdx.x < radius && col >= radius) {
            s_data[threadIdx.y][threadIdx.x] = imgin[row * ncols + col - radius];
        }
        
        // Load right halo
        if (threadIdx.x < radius && col + BLOCK_DIM < ncols) {
            s_data[threadIdx.y][sharedCol + BLOCK_DIM] = imgin[row * ncols + col + BLOCK_DIM];
        }
    }
    
    __syncthreads();
    
    // Compute convolution
    if (col >= ncols || row >= nrows) return;
    
    const int idx = row * ncols + col;
    
    if (col < radius || col >= ncols - radius) {
        imgtmp[idx] = 0.0f;
        return;
    }
    
    float sum = 0.0f;
    #pragma unroll
    for (int k = -radius; k <= radius; ++k) {
        sum += s_data[threadIdx.y][threadIdx.x + radius + k] * kernel[radius + k];
    }
    
    imgtmp[idx] = sum;
}

// Optimized vertical convolution with shared memory
__global__ void convolveVerticalKernelOptimized(
    const float *imgtmp,
    float *imgout,
    const float *kernel,
    int kernelWidth,
    int ncols,
    int nrows)
{
    __shared__ float s_data[BLOCK_DIM + 70][BLOCK_DIM]; // 70 = max kernel radius * 2
    
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int radius = kernelWidth / 2;
    
    // Load data into shared memory with halo
    if (col < ncols) {
        int sharedRow = threadIdx.y + radius;
        
        // Load main data
        if (row < nrows) {
            s_data[sharedRow][threadIdx.x] = imgtmp[row * ncols + col];
        }
        
        // Load top halo
        if (threadIdx.y < radius && row >= radius) {
            s_data[threadIdx.y][threadIdx.x] = imgtmp[(row - radius) * ncols + col];
        }
        
        // Load bottom halo
        if (threadIdx.y < radius && row + BLOCK_DIM < nrows) {
            s_data[sharedRow + BLOCK_DIM][threadIdx.x] = imgtmp[(row + BLOCK_DIM) * ncols + col];
        }
    }
    
    __syncthreads();
    
    // Compute convolution
    if (col >= ncols || row >= nrows) return;
    
    const int idx = row * ncols + col;
    
    if (row < radius || row >= nrows - radius) {
        imgout[idx] = 0.0f;
        return;
    }
    
    float sum = 0.0f;
    #pragma unroll
    for (int k = -radius; k <= radius; ++k) {
        sum += s_data[threadIdx.y + radius + k][threadIdx.x] * kernel[radius + k];
    }
    
    imgout[idx] = sum;
}

static void convolveSeparateGPU(
    _KLT_FloatImage imgin,
    const KLT_ConvolutionKernel *horiz,
    const KLT_ConvolutionKernel *vert,
    _KLT_FloatImage imgout)
{
    GPUContext *ctx = getGlobalGPUContext();
    const int ncols = imgin->ncols;
    const int nrows = imgin->nrows;
    
    // Initialize context if needed
    if (ctx->allocated_ncols < ncols || ctx->allocated_nrows < nrows) {
        if (ctx->d_buffer1) freeGPUContext(ctx);
        initGPUContext(ctx, ncols, nrows, KLT_MAX_KERNEL_WIDTH);
    }
    
    const size_t imageBytes = (size_t)ncols * nrows * sizeof(float);
    const size_t horizBytes = horiz->width * sizeof(float);
    const size_t vertBytes = vert->width * sizeof(float);
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(ctx->d_buffer1, imgin->data, imageBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_hkernel, horiz->data, horizBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_vkernel, vert->data, vertBytes, cudaMemcpyHostToDevice));
    
    // Launch optimized kernels
    const dim3 block(BLOCK_DIM, BLOCK_DIM);
    const dim3 grid((ncols + BLOCK_DIM - 1) / BLOCK_DIM, 
                    (nrows + BLOCK_DIM - 1) / BLOCK_DIM);
    
    convolveHorizontalKernelOptimized<<<grid, block>>>(
        ctx->d_buffer1, ctx->d_buffer2, ctx->d_hkernel, horiz->width, ncols, nrows);
    CUDA_CHECK(cudaGetLastError());
    
    convolveVerticalKernelOptimized<<<grid, block>>>(
        ctx->d_buffer2, ctx->d_buffer3, ctx->d_vkernel, vert->width, ncols, nrows);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(imgout->data, ctx->d_buffer3, imageBytes, cudaMemcpyDeviceToHost));
    imgout->ncols = ncols;
    imgout->nrows = nrows;
}

void _KLTComputeGradientsGPU(_KLT_FloatImage img,
                             float sigma,
                             _KLT_FloatImage gradx,
                             _KLT_FloatImage grady)
{
  KLT_ConvolutionKernel gauss;
  KLT_ConvolutionKernel gaussDeriv;
  _KLTGetKernels(sigma, &gauss, &gaussDeriv);

  convolveSeparateGPU(img, &gaussDeriv, &gauss, gradx);
  convolveSeparateGPU(img, &gauss, &gaussDeriv, grady);
}

void _KLTComputeSmoothedImageGPU(_KLT_FloatImage img,
                                 float sigma,
                                 _KLT_FloatImage smooth)
{
  KLT_ConvolutionKernel gauss;
  KLT_ConvolutionKernel gaussDeriv;
  _KLTGetKernels(sigma, &gauss, &gaussDeriv);
  (void)gaussDeriv;

  convolveSeparateGPU(img, &gauss, &gauss, smooth);
}

