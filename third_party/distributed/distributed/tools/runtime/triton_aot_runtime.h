/*
 * Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include <cuda.h>
// CUDA 12.0+ has CUDA context independent module loading. but what about
// CUDA 11.8
// https://developer.nvidia.com/blog/cuda-context-independent-module-loading/
#ifdef __cplusplus
extern "C" {
#endif

// CUDA driver stubs to avoid direct dependency on libcuda.so
CUresult cuGetErrorString_stub(CUresult error, const char **pStr);
CUresult cuDeviceGetAttribute_stub(int *pi, CUdevice_attribute attrib,
                                   CUdevice dev);

// CUDA patch for Multiple CUDA context support: using any CUDA context
typedef struct CUDAModule *CUDAModuleHandle;
typedef struct CUDAFunction *CUDAFunctionHandle;

CUresult CUDAModuleLoadData(CUDAModuleHandle *module, const void *image);

CUresult CUDAModuleUnload(CUDAModuleHandle module);

CUresult CUDAModuleGetFunction(CUDAFunctionHandle *hfunc, CUDAModuleHandle hmod,
                               const char *name);

CUresult CUDALaunchKernel(CUDAFunctionHandle f, unsigned int gridDimX,
                          unsigned int gridDimY, unsigned int gridDimZ,
                          unsigned int blockDimX, unsigned int blockDimY,
                          unsigned int blockDimZ, unsigned int sharedMemBytes,
                          CUstream hStream, void **kernelParams, void **extra);

CUresult CUDAFuncSetAttribute(CUDAFunctionHandle func,
                              CUfunction_attribute attrib, int value);

CUresult CUDAFuncSetCacheConfig(CUDAFunctionHandle func, CUfunc_cache config);

#ifdef __cplusplus
}
#endif
