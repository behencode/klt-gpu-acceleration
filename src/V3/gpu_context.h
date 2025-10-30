#ifndef GPU_CONTEXT_H
#define GPU_CONTEXT_H

#ifdef KLT_USE_CUDA
typedef struct {
    float *d_buffer1;
    float *d_buffer2;
    float *d_buffer3;
    float *d_hkernel;
    float *d_vkernel;
    int allocated_ncols;
    int allocated_nrows;
    int allocated_kernel_size;
} GPUContext;

void initGPUContext(GPUContext *ctx, int ncols, int nrows, int max_kernel_width);
void freeGPUContext(GPUContext *ctx);
GPUContext* getGlobalGPUContext(void);
#endif

#endif