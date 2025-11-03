#include <assert.h>
#include <cstdlib>
#include <cuda_runtime.h>

extern "C" {
#include "convolve.h"
#include "error.h"
}

// Keep device buffers around so repeated calls avoid repeated cudaMalloc/cudaFree.
static float *gDevImgIn = NULL;
static float *gDevTmp = NULL;
static float *gDevImgOut = NULL;
static float *gDevKernelA = NULL;
static float *gDevKernelB = NULL;
static size_t gDeviceCapacity = 0;
static int gKernelCapacity = 0;

#define CUDA_CHECK(call)                                                             \
  do {                                                                               \
    cudaError_t err__ = (call);                                                      \
    if (err__ != cudaSuccess) {                                                      \
      KLTError("CUDA error %s (%d) at %s:%d", cudaGetErrorString(err__),             \
               static_cast<int>(err__), __FILE__, __LINE__);                          \
    }                                                                                \
  } while (0)

static inline void safeCudaFree(float **ptr)
{
  if (ptr == NULL || *ptr == NULL) return;
  cudaError_t err = cudaFree(*ptr);
  if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
    KLTError("CUDA free error %s (%d) at %s:%d", cudaGetErrorString(err),
             static_cast<int>(err), __FILE__, __LINE__);
  }
  *ptr = NULL;
}

static inline dim3 makeGrid2d(int ncols, int nrows, dim3 block)
{
  return dim3((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);
}

static inline size_t numImageElements(int ncols, int nrows)
{
  return static_cast<size_t>(ncols) * static_cast<size_t>(nrows);
}

static void releaseBuffers(void)
{
  safeCudaFree(&gDevImgIn);
  safeCudaFree(&gDevTmp);
  safeCudaFree(&gDevImgOut);
  safeCudaFree(&gDevKernelA);
  safeCudaFree(&gDevKernelB);

  gDeviceCapacity = 0;
  gKernelCapacity = 0;
}

static void ensureImageBuffers(size_t numel)
{
  if (numel <= gDeviceCapacity) return;

  static bool cleanupRegistered = false;
  if (!cleanupRegistered) {
    atexit(releaseBuffers);
    cleanupRegistered = true;
  }

  if (gDevImgIn != NULL) CUDA_CHECK(cudaFree(gDevImgIn));
  if (gDevTmp != NULL) CUDA_CHECK(cudaFree(gDevTmp));
  if (gDevImgOut != NULL) CUDA_CHECK(cudaFree(gDevImgOut));

  const size_t bytes = numel * sizeof(float);
  CUDA_CHECK(cudaMalloc(&gDevImgIn, bytes));
  CUDA_CHECK(cudaMalloc(&gDevTmp, bytes));
  CUDA_CHECK(cudaMalloc(&gDevImgOut, bytes));

  gDeviceCapacity = numel;
}

static void ensureKernelBuffers(int widthA, int widthB)
{
  const int needed = widthA > widthB ? widthA : widthB;
  if (needed <= gKernelCapacity) return;

  if (gDevKernelA != NULL) CUDA_CHECK(cudaFree(gDevKernelA));
  if (gDevKernelB != NULL) CUDA_CHECK(cudaFree(gDevKernelB));

  CUDA_CHECK(cudaMalloc(&gDevKernelA, static_cast<size_t>(needed) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&gDevKernelB, static_cast<size_t>(needed) * sizeof(float)));

  gKernelCapacity = needed;
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
  for (int k = -radius; k <= radius; ++k) {
    sum += imgin[base + col + k] * kernel[radius + k];
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
  for (int k = -radius; k <= radius; ++k) {
    const int offset = (row + k) * ncols + col;
    sum += imgtmp[offset] * kernel[radius + k];
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
  const size_t numel = numImageElements(ncols, nrows);
  const size_t imageBytes = numel * sizeof(float);

  assert(horiz->width <= KLT_MAX_KERNEL_WIDTH);
  assert(vert->width <= KLT_MAX_KERNEL_WIDTH);

  ensureImageBuffers(numel);
  ensureKernelBuffers(horiz->width, vert->width);

  CUDA_CHECK(cudaMemcpy(gDevImgIn, imgin->data, imageBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gDevKernelA, horiz->data,
                        static_cast<size_t>(horiz->width) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gDevKernelB, vert->data,
                        static_cast<size_t>(vert->width) * sizeof(float),
                        cudaMemcpyHostToDevice));

  const dim3 block(32, 8);
  const dim3 grid = makeGrid2d(ncols, nrows, block);

  convolveHorizontalKernel<<<grid, block>>>(gDevImgIn, gDevTmp,
                                            gDevKernelA,
                                            horiz->width, ncols, nrows);
  CUDA_CHECK(cudaGetLastError());

  convolveVerticalKernel<<<grid, block>>>(gDevTmp, gDevImgOut,
                                          gDevKernelB,
                                          vert->width, ncols, nrows);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(imgout->data, gDevImgOut, imageBytes, cudaMemcpyDeviceToHost));
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
