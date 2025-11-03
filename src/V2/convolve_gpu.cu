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

__global__ void convolveHorizontalKernel(const float *imgin,
                                         float *imgtmp,
                                         const float *kernel,
                                         int kernelWidth,
                                         int ncols,
                                         int nrows)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= ncols || row >= nrows) return;

  const int radius = kernelWidth / 2;
  const int idx = row * ncols + col;

  if (col < radius || col >= ncols - radius) {
    imgtmp[idx] = 0.0f;
    return;
  }

  const int base = row * ncols;
  float sum = 0.0f;
  // Iterate in reverse to mirror CPU implementation that reads kernel backwards.
  for (int k = kernelWidth - 1; k >= 0; --k) {
    const int offset = col + (k - radius);
    sum += imgin[base + offset] * kernel[k];
  }
  imgtmp[idx] = sum;
}

__global__ void convolveVerticalKernel(const float *imgtmp,
                                       float *imgout,
                                       const float *kernel,
                                       int kernelWidth,
                                       int ncols,
                                       int nrows)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= ncols || row >= nrows) return;

  const int radius = kernelWidth / 2;
  const int idx = row * ncols + col;

  if (row < radius || row >= nrows - radius) {
    imgout[idx] = 0.0f;
    return;
  }

  float sum = 0.0f;
  // Iterate in reverse to mirror CPU implementation that reads kernel backwards.
  for (int k = kernelWidth - 1; k >= 0; --k) {
    const int offsetRow = row + (k - radius);
    const int offset = offsetRow * ncols + col;
    sum += imgtmp[offset] * kernel[k];
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
  CUDA_CHECK(cudaMalloc(&d_hkernel, horizBytes));
  CUDA_CHECK(cudaMalloc(&d_vkernel, vertBytes));

  CUDA_CHECK(cudaMemcpy(d_imgin, imgin->data, imageBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_hkernel, horiz->data, horizBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vkernel, vert->data, vertBytes, cudaMemcpyHostToDevice));

  const dim3 block(16, 16);
  const dim3 grid = makeGrid2d(ncols, nrows, block);

  convolveHorizontalKernel<<<grid, block>>>(d_imgin, d_tmp, d_hkernel,
                                            horiz->width, ncols, nrows);
  CUDA_CHECK(cudaGetLastError());

  convolveVerticalKernel<<<grid, block>>>(d_tmp, d_imgout, d_vkernel,
                                          vert->width, ncols, nrows);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(imgout->data, d_imgout, imageBytes, cudaMemcpyDeviceToHost));
  imgout->ncols = ncols;
  imgout->nrows = nrows;

  cudaFree(d_imgin);
  cudaFree(d_tmp);
  cudaFree(d_imgout);
  cudaFree(d_hkernel);
  cudaFree(d_vkernel);
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
