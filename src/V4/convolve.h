/*********************************************************************
 * convolve.h
 *********************************************************************/

#ifndef _CONVOLVE_H_
#define _CONVOLVE_H_

#include "klt.h"
#include "klt_util.h"


#ifndef KLT_MAX_KERNEL_WIDTH
#define KLT_MAX_KERNEL_WIDTH 71
#endif

typedef struct  {
  int width;
  float data[KLT_MAX_KERNEL_WIDTH];
} KLT_ConvolutionKernel;

void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg);

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady);

void _KLTComputeGradientsCPU(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady);

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width);

void _KLTGetKernels(
  float sigma,
  KLT_ConvolutionKernel *gauss,
  KLT_ConvolutionKernel *gaussderiv);

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth);

void _KLTComputeSmoothedImageCPU(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth);

#ifdef KLT_USE_OPENACC
void _KLTComputeGradientsACC(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady);

void _KLTComputeSmoothedImageACC(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth);
#endif

#ifdef KLT_USE_CUDA
void _KLTComputeGradientsGPU(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady);

void _KLTComputeSmoothedImageGPU(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth);
#endif

#endif
