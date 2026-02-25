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

#include "flash_comm/buffer/shareable_block.h"
#include "flash_comm/common.h"
#include <cstring>

namespace flash_comm {
namespace buffer {

ShareableBlock::ShareableBlock(size_t size, BlockBackend backend)
    : size_(size), backend_(backend) {

  if (backend_ == BlockBackend::CUDA_IPC) {
    allocate_cuda_ipc(size_);
  } else {
    try {
      allocate_vmm(size_);
    } catch (...) {
      if (has_vmm_handle_) {
        cuMemRelease(vmm_handle_);
        has_vmm_handle_ = false;
      }
      throw;
    }
  }
}

ShareableBlock::~ShareableBlock() {
  if (ptr_) {
    if (backend_ == BlockBackend::CUDA_IPC) {
      cudaFree(ptr_);
    } else {
      CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr_);
      cuMemUnmap(dptr, size_);
      cuMemAddressFree(dptr, size_);
    }
  }
  if (has_vmm_handle_) {
    cuMemRelease(vmm_handle_);
  }
}

void ShareableBlock::allocate_cuda_ipc(size_t size) {
  CUDA_CHECK(cudaMalloc(&ptr_, size));
}

void ShareableBlock::allocate_vmm(size_t size) {
  cuInit(0);

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

  CUdevice device_id;
  DRIVER_CHECK(cuCtxGetDevice(&device_id));
  prop.location.id = device_id;

  if (!is_fabric_supported((int)device_id)) {
    throw std::runtime_error("Fabric handle not supported on this device.");
  }

  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t granularity;
  DRIVER_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                             CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  size_ = ((size + granularity - 1) / granularity) * granularity;

  DRIVER_CHECK(cuMemCreate(&vmm_handle_, size_, &prop, 0));
  has_vmm_handle_ = true;

  CUdeviceptr dptr;
  DRIVER_CHECK(cuMemAddressReserve(&dptr, size_, 0, 0, 0));
  DRIVER_CHECK(cuMemMap(dptr, size_, 0, vmm_handle_, 0));

  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device_id;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  DRIVER_CHECK(cuMemSetAccess(dptr, size_, &access, 1));

  ptr_ = reinterpret_cast<void *>(dptr);
}

bool ShareableBlock::is_fabric_supported(int device) {
  int supported = 0;
  cuInit(0);
  cuDeviceGetAttribute(
      &supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
  return supported != 0;
}

bool ShareableBlock::is_vmm_supported(int device) {
  int supported = 0;
  cuInit(0);
  cuDeviceGetAttribute(&supported,
                       CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                       device);
  return supported != 0;
}

std::vector<char> ShareableBlock::get_handle_data() const {
  if (backend_ == BlockBackend::CUDA_IPC) {
    cudaIpcMemHandle_t handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, ptr_));
    std::vector<char> bytes(sizeof(handle));
    std::memcpy(bytes.data(), &handle, sizeof(handle));
    return bytes;
  } else {
    CUmemFabricHandle handle;
    DRIVER_CHECK(cuMemExportToShareableHandle(&handle, vmm_handle_,
                                              CU_MEM_HANDLE_TYPE_FABRIC, 0));
    std::vector<char> bytes(sizeof(handle));
    std::memcpy(bytes.data(), &handle, sizeof(handle));
    return bytes;
  }
}

void *ShareableBlock::map_remote(const std::vector<char> &handle, size_t size) {
  if (backend_ == BlockBackend::CUDA_IPC) {
    if (handle.size() != sizeof(cudaIpcMemHandle_t)) {
      throw std::runtime_error("Invalid handle size for CUDA_IPC backend");
    }
    cudaIpcMemHandle_t ipc_handle;
    std::memcpy(&ipc_handle, handle.data(), sizeof(ipc_handle));

    void *ptr;
    CUDA_CHECK(
        cudaIpcOpenMemHandle(&ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess));
    return ptr;
  } else {
    CUmemGenericAllocationHandle alloc_h;
    CUmemFabricHandle fabric_h;
    std::memcpy(&fabric_h, handle.data(), sizeof(fabric_h));
    DRIVER_CHECK(cuMemImportFromShareableHandle(&alloc_h, &fabric_h,
                                                CU_MEM_HANDLE_TYPE_FABRIC));

    CUdeviceptr dptr;
    DRIVER_CHECK(cuMemAddressReserve(&dptr, size, 0, 0, 0));
    DRIVER_CHECK(cuMemMap(dptr, size, 0, alloc_h, 0));

    cuMemRelease(alloc_h);

    CUdevice ctx_device;
    DRIVER_CHECK(cuCtxGetDevice(&ctx_device));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = ctx_device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUresult res = cuMemSetAccess(dptr, size, &access, 1);
    if (res != CUDA_SUCCESS) {
      cuMemUnmap(dptr, size);
      cuMemAddressFree(dptr, size);
      DRIVER_CHECK(res);
    }

    return reinterpret_cast<void *>(dptr);
  }
}

void ShareableBlock::unmap_remote(void *ptr, size_t size) {
  if (backend_ == BlockBackend::CUDA_IPC) {
    CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
  } else {
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
    cuMemUnmap(dptr, size);
    cuMemAddressFree(dptr, size);
  }
}

} // namespace buffer
} // namespace flash_comm
