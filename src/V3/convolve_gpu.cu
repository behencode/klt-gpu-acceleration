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
               static_cast<int>(err__), __FILE__, __LINE__);                         \
    }                                                                                \
  } while (0)

#ifndef CONVOLVE_GPU_MAX_KERNEL_WIDTH
#define CONVOLVE_GPU_MAX_KERNEL_WIDTH 64
#endif

__constant__ float c_horizontalKernel[CONVOLVE_GPU_MAX_KERNEL_WIDTH];
__constant__ float c_verticalKernel[CONVOLVE_GPU_MAX_KERNEL_WIDTH];

static dim3 makeGrid2d(int ncols, int nrows, dim3 block)
{
  return dim3((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);
}

__device__ inline float loadKernelValue(const float *kernel,
                                        int k,
                                        bool useConstant,
                                        bool horizontal)
{
  if (!useConstant) return kernel[k];
  return horizontal ? c_horizontalKernel[k] : c_verticalKernel[k];
}

__global__ void convolveHorizontalKernel(const float *imgin,
                                         float *imgtmp,
                                         const float *kernel,
                                         int kernelWidth,
                                         int ncols,
                                         int nrows,
                                         bool useConstant)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  const int radius = kernelWidth / 2;
  const int idx = row * ncols + col;

  const int tileWidth = blockDim.x + kernelWidth - 1;
  extern __shared__ float sharedTile[];
  float *tileRow = sharedTile + threadIdx.y * tileWidth;

  if (row < nrows) {
    const int baseCol = blockIdx.x * blockDim.x;
    for (int i = threadIdx.x; i < tileWidth; i += blockDim.x) {
      const int globalCol = baseCol + i - radius;
      float val = 0.0f;
      if (globalCol >= 0 && globalCol < ncols) {
        val = imgin[row * ncols + globalCol];
      }
      tileRow[i] = val;
    }
  }

  __syncthreads();

  if (row >= nrows || col >= ncols) return;

  if (col < radius || col >= ncols - radius) {
    imgtmp[idx] = 0.0f;
    return;
  }

  float sum = 0.0f;
  const int sharedOffset = threadIdx.x;
  for (int k = kernelWidth - 1; k >= 0; --k) {
    const float pixel = tileRow[sharedOffset + (kernelWidth - 1 - k)];
    const float kval = loadKernelValue(kernel, k, useConstant, true);
    sum += pixel * kval;
  }
  imgtmp[idx] = sum;
}

__global__ void convolveVerticalKernel(const float *imgtmp,
                                       float *imgout,
                                       const float *kernel,
                                        int kernelWidth,
                                        int ncols,
                                       int nrows,
                                       bool useConstant)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  const int radius = kernelWidth / 2;
  const int idx = row * ncols + col;

  const int tileHeight = blockDim.y + kernelWidth - 1;
  extern __shared__ float sharedTile[];
  float *tileCol = sharedTile + threadIdx.x;

  if (col < ncols) {
    const int baseRow = blockIdx.y * blockDim.y;
    for (int j = threadIdx.y; j < tileHeight; j += blockDim.y) {
      const int globalRow = baseRow + j - radius;
      float val = 0.0f;
      if (globalRow >= 0 && globalRow < nrows) {
        val = imgtmp[globalRow * ncols + col];
      }
      tileCol[j * blockDim.x] = val;
    }
  }

  __syncthreads();

  if (row >= nrows || col >= ncols) return;

  if (row < radius || row >= nrows - radius) {
    imgout[idx] = 0.0f;
    return;
  }

  float sum = 0.0f;
  const int sharedOffset = threadIdx.y;
  for (int k = kernelWidth - 1; k >= 0; --k) {
    const float pixel = tileCol[(sharedOffset + (kernelWidth - 1 - k)) * blockDim.x];
    const float kval = loadKernelValue(kernel, k, useConstant, false);
    sum += pixel * kval;
  }
  imgout[idx] = sum;
}

static void convolveSeparateGPU(_KLT_FloatImage imgin,
                                const KLT_ConvolutionKernel *horiz,
                                const KLT_ConvolutionKernel *vert,
                                _KLT_FloatImage imgout)
{
  assert(imgin != NULL);
  assert(imgout != NULL);
  assert(horiz != NULL);
  assert(vert != NULL);
  assert(horiz->width % 2 == 1);
  assert(vert->width % 2 == 1);
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  const int ncols = imgin->ncols;
  const int nrows = imgin->nrows;
  const size_t numel = static_cast<size_t>(ncols) * static_cast<size_t>(nrows);
  const size_t imageBytes = numel * sizeof(float);
  const size_t horizBytes = static_cast<size_t>(horiz->width) * sizeof(float);
  const size_t vertBytes = static_cast<size_t>(vert->width) * sizeof(float);

  float *d_imgin = NULL;
  float *d_tmp = NULL;
  float *d_imgout = NULL;
  float *d_hkernel = NULL;
  float *d_vkernel = NULL;

  CUDA_CHECK(cudaMalloc(&d_imgin, imageBytes));
  CUDA_CHECK(cudaMalloc(&d_tmp, imageBytes));
  CUDA_CHECK(cudaMalloc(&d_imgout, imageBytes));

  const bool useHorizontalConst = horiz->width <= CONVOLVE_GPU_MAX_KERNEL_WIDTH;
  const bool useVerticalConst = vert->width <= CONVOLVE_GPU_MAX_KERNEL_WIDTH;

  if (!useHorizontalConst) {
    CUDA_CHECK(cudaMalloc(&d_hkernel, horizBytes));
  }
  if (!useVerticalConst) {
    CUDA_CHECK(cudaMalloc(&d_vkernel, vertBytes));
  }

  CUDA_CHECK(cudaMemcpy(d_imgin, imgin->data, imageBytes, cudaMemcpyHostToDevice));
  if (useHorizontalConst) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_horizontalKernel, horiz->data, horizBytes));
  } else {
    CUDA_CHECK(cudaMemcpy(d_hkernel, horiz->data, horizBytes, cudaMemcpyHostToDevice));
  }
  if (useVerticalConst) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_verticalKernel, vert->data, vertBytes));
  } else {
    CUDA_CHECK(cudaMemcpy(d_vkernel, vert->data, vertBytes, cudaMemcpyHostToDevice));
  }

  const dim3 block(16, 16);
  const dim3 grid = makeGrid2d(ncols, nrows, block);
  const size_t sharedHorizontal =
      static_cast<size_t>(block.x + horiz->width - 1) * block.y * sizeof(float);
  const size_t sharedVertical =
      static_cast<size_t>(block.y + vert->width - 1) * block.x * sizeof(float);

  const float *horizontalKernelPtr = useHorizontalConst ? NULL : d_hkernel;
  const float *verticalKernelPtr = useVerticalConst ? NULL : d_vkernel;

  convolveHorizontalKernel<<<grid, block, sharedHorizontal>>>(
      d_imgin, d_tmp, horizontalKernelPtr, horiz->width, ncols, nrows, useHorizontalConst);
  CUDA_CHECK(cudaGetLastError());

  convolveVerticalKernel<<<grid, block, sharedVertical>>>(
      d_tmp, d_imgout, verticalKernelPtr, vert->width, ncols, nrows, useVerticalConst);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(imgout->data, d_imgout, imageBytes, cudaMemcpyDeviceToHost));
  imgout->ncols = ncols;
  imgout->nrows = nrows;

  if (d_hkernel) cudaFree(d_hkernel);
  if (d_vkernel) cudaFree(d_vkernel);
  cudaFree(d_imgin);
  cudaFree(d_tmp);
  cudaFree(d_imgout);
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
