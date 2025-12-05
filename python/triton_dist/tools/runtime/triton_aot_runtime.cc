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
#include <cstring>
#include <cuda.h>
#include <dlfcn.h>
#include <map>
#include <mutex>

#include <nvshmem.h>
#include <nvshmemx.h>

#ifdef __cplusplus
#define TRITON_DIST_EXTERN extern "C" __attribute__((visibility("hidden")))
#else
#define TRITON_DIST_EXTERN
#endif

#define CHECK(rtn)                                                             \
  do {                                                                         \
    bool rtn_ = (rtn);                                                         \
    if (!rtn_) {                                                               \
      fprintf(stderr, "CHECK error at %s:%d, errno: %d (%s)\n", __FILE__,      \
              __LINE__, errno, strerror(errno));                               \
      exit(rtn);                                                               \
    }                                                                          \
  } while (0)

#define CHECK_RTN_RETURN(rtn)                                                  \
  do {                                                                         \
    auto rtn_ = (rtn);                                                         \
    if (rtn_ != CUDA_SUCCESS) {                                                \
      fprintf(stderr, "CUDA error %d at %s:%d\n", rtn_, __FILE__, __LINE__);   \
      return rtn;                                                              \
    }                                                                          \
  } while (0)

class CUDAModule {
public:
  CUDAModule(const void *image) : image_(image) {}
  CUresult Load(CUcontext context) {
    std::lock_guard<std::mutex> _(mutex_);
    return LoadImpl(context);
  }

  CUresult Unload(CUcontext context) {
    std::lock_guard<std::mutex> _(mutex_);
    if (module_.find(context) != module_.end()) {
      return CUDA_ERROR_UNKNOWN;
    }
    auto rtn = cuModuleUnload(module_[context]);
    module_.erase(context);
    return rtn;
  }

  CUresult GetOrLoad(CUcontext context, CUmodule &module) {
    std::lock_guard<std::mutex> _(mutex_);
    auto iter = module_.find(context);
    if (iter == module_.end()) { // Get from another CUcontext
      CHECK_RTN_RETURN(LoadImpl(context));
      module = module_[context];
    } else {
      module = iter->second;
    }
    return CUDA_SUCCESS;
  }

private:
  CUresult LoadImpl(CUcontext context) {
    auto iter = module_.find(context);
    if (iter != module_.end()) {
      return CUDA_ERROR_UNKNOWN;
    }
    return cuModuleLoadData(&module_[context], image_);
  }

private:
  const void *image_;
  std::mutex mutex_;
  std::map<const CUcontext, CUmodule> module_;
};

class CUDAFunction {
public:
  CUDAFunction(CUDAModule *module, const char *name)
      : module_(module), name_(name) {}
  CUresult Load(CUcontext context) {
    std::lock_guard<std::mutex> _(mutex_);
    return LoadImpl(context);
  }

  CUresult GetOrLoad(CUcontext context, CUfunction *function) {
    auto iter = func_.find(context);
    if (iter == func_.end()) { // Get from another CUcontext
      CHECK_RTN_RETURN(LoadImpl(context));
      *function = func_[context];
    } else {
      *function = iter->second;
    }
    return CUDA_SUCCESS;
  }

private:
  CUresult LoadImpl(CUcontext context) {
    CUmodule mod;
    CHECK_RTN_RETURN(module_->GetOrLoad(context, mod));
    auto iter = func_.find(context);
    if (iter != func_.end()) { // duplicate load
      return CUDA_ERROR_UNKNOWN;
    }
    return cuModuleGetFunction(&func_[context], mod, name_);
  }

private:
  CUDAModule *module_ = nullptr;
  const char *name_ = nullptr;
  std::mutex mutex_;
  std::map<CUcontext, CUfunction> func_;
};

using CUDAModuleHandle = CUDAModule *;
using CUDAFunctionHandle = CUDAFunction *;

TRITON_DIST_EXTERN CUresult CUDAModuleLoadData(CUDAModuleHandle *module,
                                               const void *image) {
  CUDAModule *mod = new CUDAModule(image);
  CUcontext context;
  CHECK_RTN_RETURN(cuCtxGetCurrent(&context));
  CHECK_RTN_RETURN(mod->Load(context));
  *module = mod;
  return CUDA_SUCCESS;
}

TRITON_DIST_EXTERN CUresult CUDAModuleGetFunction(CUDAFunctionHandle *hfunc,
                                                  CUDAModuleHandle hmod,
                                                  const char *name) {
  CUDAFunction *func = new CUDAFunction(hmod, name);
  CUcontext context;
  CHECK_RTN_RETURN(cuCtxGetCurrent(&context));
  CHECK_RTN_RETURN(func->Load(context));
  *hfunc = func;
  return CUDA_SUCCESS;
}

TRITON_DIST_EXTERN CUresult CUDAModuleUnload(CUDAModuleHandle hmod) {
  CUcontext context;
  CHECK_RTN_RETURN(cuCtxGetCurrent(&context));
  CHECK_RTN_RETURN(hmod->Unload(context));
  return CUDA_SUCCESS;
}

TRITON_DIST_EXTERN CUresult CUDAFuncSetAttribute(CUDAFunctionHandle func,
                                                 CUfunction_attribute attrib,
                                                 int value) {
  CUcontext context;
  CHECK_RTN_RETURN(cuCtxGetCurrent(&context));
  CUfunction f;
  CHECK_RTN_RETURN(func->GetOrLoad(context, &f));
  return cuFuncSetAttribute(f, attrib, value);
}

TRITON_DIST_EXTERN CUresult CUDAFuncSetCacheConfig(CUDAFunctionHandle func,
                                                   CUfunc_cache config) {
  CUcontext context;
  CHECK_RTN_RETURN(cuCtxGetCurrent(&context));
  CUfunction f;
  CHECK_RTN_RETURN(func->GetOrLoad(context, &f));
  return cuFuncSetCacheConfig(f, config);
}

TRITON_DIST_EXTERN CUresult CUDALaunchKernel(
    CUDAFunctionHandle f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra) {
  CUcontext context;
  CHECK_RTN_RETURN(cuCtxGetCurrent(&context));
  CUfunction func;
  CHECK_RTN_RETURN(f->GetOrLoad(context, &func));
  CHECK_RTN_RETURN(cuLaunchKernel(func, gridDimX, gridDimY, gridDimZ, blockDimX,
                                  blockDimY, blockDimZ, sharedMemBytes, hStream,
                                  kernelParams, extra));
  return CUDA_SUCCESS;
}

TRITON_DIST_EXTERN CUresult NVSHMEMModuleInit(CUDAModuleHandle hmod) {
  CUmodule mod;
  CUcontext context;
  CHECK_RTN_RETURN(cuCtxGetCurrent(&context));
  CHECK_RTN_RETURN(hmod->GetOrLoad(context, mod));
  int init_err = nvshmemx_cumodule_init(mod);
  if (init_err == 0) {
    return CUDA_SUCCESS;
  } else {
    fprintf(stderr, "nvshmem cumodule init failed: (error code: %d)\n",
            init_err);
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}
