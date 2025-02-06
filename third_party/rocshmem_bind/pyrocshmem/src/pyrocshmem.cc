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
// TODO: include rocshmem headers
#include "c10/hip/HIPFunctions.h"
#include "c10/hip/HIPGuard.h"
#include "c10/hip/HIPStream.h"
#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rocshmem/rocshmem.hpp>
#include <torch/all.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>

using namespace rocshmem;
// TODO: add pyrocshmem pybinding

class LazyLogger {
public:
  LazyLogger(bool no_error = false) {
    _no_print = no_error;
    _no_error = no_error;
  };

  ~LazyLogger() {
    if (!_no_print) {
      std::cerr << _message.str() << std::endl;
    }
    if (!_no_error) {
      throw std::runtime_error(_message.str());
    }
  }

  template <typename T> LazyLogger &operator<<(const T &value) {
    _message << value;
    return *this;
  }

private:
  bool _no_print = false;
  bool _no_error = false;
  std::ostringstream _message;
};

#define HIP_CHECK(hip_error)                                                   \
  {                                                                            \
    if (hip_error != hipSuccess) {                                             \
      printf("hipError %s in %s:%d\n", hipGetErrorString(hip_error), __func__, \
             __LINE__);                                                        \
      throw std::runtime_error("hip error.");                                  \
    }                                                                          \
  }

#define PYROCSHMEM_CHECK(cond)                                                 \
  LazyLogger(cond) << __FILE__ << ":" << __LINE__                              \
                   << " Check failed: " #cond ". "
#define PYROCSHMEM_CHECK_NE(a, b) PYROCSHMEM_CHECK(((a) != (b)))

#define CHECK_ROCSHMEMX(expr)                                                  \
  do {                                                                         \
    int x = expr;                                                              \
    if (x != ROCSHMEMX_SUCCESS) {                                              \
      throw std::runtime_error(__FILE__ ":" + std::to_string(__LINE__) +       \
                               " " #expr " failed with status code " +         \
                               std::to_string(x));                             \
    }                                                                          \
  } while (0)

// TODO: found rocshmem init state related API or returns.

#define ENABLE_ROCSHMEM 1

#if ENABLE_ROCSHMEM
inline torch::Tensor create_tensor(const std::vector<int64_t> &shape,
                                   c10::ScalarType dtype) {
  // TODO: check rocshmem init state.
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kHIP).device_index(
          c10::hip::current_device());
  auto size =
      torch::elementSize(dtype) *
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  return at::from_blob(
      rocshmem_malloc(size), shape, [](void *ptr) { rocshmem_free(ptr); },
      option_gpu);
}
#endif

std::vector<torch::Tensor> rocshmem_get_tensors_from_ipchandle(
    int64_t rank, int64_t world_size, const std::vector<torch::Tensor> &handles,
    const std::vector<int64_t> &offsets, const std::vector<int64_t> &shape,
    c10::ScalarType dtype) {
  auto current_device = c10::hip::current_device();
  auto option_gpu =
      at::TensorOptions(at::kHIP).dtype(dtype).device_index(current_device);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(world_size);
  hipDeviceptr_t basePtr;
  hipDeviceptr_t memPtr;
  for (size_t i = 0; i < world_size; ++i) {
    auto handle_ts = handles[i].tensor_data();
    TORCH_CHECK(handle_ts.dtype() == at::ScalarType::Byte,
                "the " + std::to_string(i) +
                    "-th cuda shm handle is not a byte tensor");
    hipIpcMemHandle_t handle =
        *reinterpret_cast<hipIpcMemHandle_t *>(handle_ts.data_ptr());
    if (i != rank) {
      HIP_CHECK(
          hipIpcOpenMemHandle(&basePtr, handle, hipIpcMemLazyEnablePeerAccess));
      memPtr = static_cast<size_t *>(basePtr) + offsets[i];
      tensors.emplace_back(at::from_blob(
          memPtr, shape,
          [=](void *ptr) {
            at::hip::HIPGuard guard(current_device);
            at::hip::device_synchronize();
            hipIpcCloseMemHandle(ptr);
            at::hip::device_synchronize();
          },
          option_gpu));
    } else {
      // skip local rank
    }
  }
  return tensors;
}

torch::Tensor hipcreate_tensor_and_handle(const std::vector<int64_t> &shape,
                                          c10::ScalarType dtype) {
  auto current_device = c10::hip::current_device();
  auto option_gpu =
      at::TensorOptions(at::kHIP).dtype(dtype).device_index(current_device);
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), (size_t)1,
                              std::multiplies<>());
  void *ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, size));
  HIP_CHECK(hipMemset(ptr, 0, size)); // memset the allocated buffer
  hipIpcMemHandle_t handle;
  HIP_CHECK(hipIpcGetMemHandle(&handle, ptr));
  auto tensor = at::from_blob(
      ptr, shape,
      [=](void *ptr) {
        std::cerr << "enter hipmem_free " << ptr << "\n";
        at::hip::HIPGuard guard(current_device);
        at::hip::device_synchronize();
        std::cerr << "do hipmem_free " << ptr << "\n";
        hipFree(ptr);
        at::hip::device_synchronize();
        std::cerr << "exit hipmem_free " << ptr << "\n";
      },
      option_gpu);
  return std::move(tensor);
}

static void all_gather_helper(c10d::ProcessGroup *pg, const void *src,
                              void *dst, int64_t nbytes) {
  auto option_cpu = at::TensorOptions(torch::kUInt8).device(at::kCPU);
  auto option_gpu = at::TensorOptions(torch::kUInt8)
                        .device(at::kHIP)
                        .device_index(c10::hip::current_device());
  auto dst_tensor = at::from_blob(dst, {nbytes * pg->getSize()}, option_cpu);
  auto src_tensor =
      at::from_blob(const_cast<void *>(src), {nbytes}, option_cpu);
  auto dst_tensor_gpu = dst_tensor.to(option_gpu);
  auto src_tensor_gpu = src_tensor.to(option_gpu);
  pg->_allgather_base(dst_tensor_gpu, src_tensor_gpu)->wait();
  // dst_tensor = dst_tensor_gpu.to(option_cpu);
  dst_tensor.copy_(dst_tensor_gpu.to(option_cpu));
}

static std::vector<torch::Tensor>
hipipc_create_tensor_list(c10d::ProcessGroup *group,
                          const std::vector<int64_t> &shape,
                          c10::ScalarType dtype) {
  assert(group != nullptr);
  int cur_rank = group->getRank();
  int world_size = group->getSize();

  // This ring mode behavior is typically used in scenarios where the p2p
  // protocol is not worked any more such as the number of peers exceeds 8.
  auto option_gpu = at::TensorOptions(dtype).device(at::kHIP).device_index(
      c10::hip::current_device());

  assert(world_size <= torch::hip::device_count() &&
         "create_ipc_tensors should only be used intra node");
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), (size_t)1,
                              std::multiplies<>());
  assert(size != 0);
  void *ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, size));
  // void *ptr = rocshmem_malloc(size);
  HIP_CHECK(hipMemset(ptr, 0, size)); // memset the allocated buffer
  hipIpcMemHandle_t handle;
  HIP_CHECK(hipIpcGetMemHandle(&handle, ptr));
  std::vector<hipIpcMemHandle_t> handles(world_size);
  // group->all_gather_cpu(&handle, handles.data(), sizeof(hipIpcMemHandle_t));
  all_gather_helper(group, &handle, handles.data(), sizeof(hipIpcMemHandle_t));

  int prev_peer = (cur_rank - 1 + world_size) % world_size;
  int next_peer = (cur_rank + 1) % world_size;
  std::vector<torch::Tensor> tensors;
  std::vector<void *> ptrs(world_size);
  for (int i = 0; i < world_size; ++i) {
    if (i != cur_rank) {
      HIP_CHECK(hipIpcOpenMemHandle(&ptrs[i], handles[i],
                                    hipIpcMemLazyEnablePeerAccess));
    } else {
      ptrs[i] = ptr;
    }
  }

  for (int i = 0; i < world_size; ++i) {
    torch::Tensor tensor;
    if (i == cur_rank) {
      tensor = at::from_blob(
          ptr, shape, [](void *ptr) { hipFree(ptr); }, option_gpu);
      // tensor = at::from_blob(ptr, shape, [](void *ptr) { rocshmem_free(ptr);
      // }, option_gpu);
    } else {
      tensor = at::from_blob(
          ptrs[i], shape, [](void *ptr) { hipIpcCloseMemHandle(ptr); },
          option_gpu);
    }
    tensors.emplace_back(tensor);
  }

  return tensors;
}

void test_ipc_handle_impl(c10d::ProcessGroup *group,
                          const std::vector<int64_t> &shape,
                          c10::ScalarType dtype) {
  int cur_rank = group->getRank();
  int world_size = group->getSize();
  auto option_gpu = at::TensorOptions(dtype).device(at::kHIP).device_index(
      c10::hip::current_device());

  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), (size_t)1,
                              std::multiplies<>());
  void *ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, size));
  HIP_CHECK(hipMemset(ptr, cur_rank, size)); // memset the allocated buffer
  hipIpcMemHandle_t handle;
  HIP_CHECK(hipIpcGetMemHandle(&handle, ptr));
  std::vector<hipIpcMemHandle_t> handles(world_size);
  // group.all_gather_cpu(&handle, handles.data(), sizeof(hipIpcMemHandle_t));
  std::cerr << "sizeof(hipIpcMemHandle_t)==" << sizeof(hipIpcMemHandle_t)
            << "\n";
  all_gather_helper(group, &handle, handles.data(), sizeof(hipIpcMemHandle_t));
  std::cerr << "test pg. cur_rank=" << cur_rank
            << ", after all_gather_helper\n";

  int target_rank = (cur_rank + 1) % world_size;
  void *target_ptr;
  // get remote device memory ptr
  HIP_CHECK(hipIpcOpenMemHandle(&target_ptr, handles[target_rank],
                                hipIpcMemLazyEnablePeerAccess));
  // create remote tensor
  torch::Tensor target_tensor = at::from_blob(
      target_ptr, shape, [](void *ptr) { hipIpcCloseMemHandle(ptr); },
      option_gpu);

  std::cerr << "rank[" << cur_rank << "] get remote_rank[" << target_rank
            << "]\n";
}

#if ENABLE_ROCSHMEM
std::vector<torch::Tensor>
rocshmem_create_tensor_list(const std::vector<int64_t> &shape,
                            c10::ScalarType dtype) {
  // TODO: check rocshmem init state.
  auto current_device = c10::hip::current_device();
  auto option_gpu =
      at::TensorOptions(at::kHIP).dtype(dtype).device_index(current_device);
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), (size_t)1,
                              std::multiplies<>());
  PYROCSHMEM_CHECK_NE(size, 0);
  int local_world_size = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);
  int rank = rocshmem_my_pe();
  int local_rank = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(local_world_size);
  std::cerr << "enter rocshmem_malloc\n";
  at::hip::device_synchronize();
  std::cerr << "do rocshmem_malloc\n";
  void *ptr = rocshmem_malloc(size);
  std::cerr << "exit rocshmem_malloc " << ptr << "\n";

  HIP_CHECK(hipMemset(ptr, 0, size)); // memset the allocated buffer
  PYROCSHMEM_CHECK(ptr != nullptr);
  int rank_offset = rank - local_rank;
  for (int i = 0; i < local_world_size; i++) {
    int rank_global = i + rank_offset;
    if (rank == rank_global) {
      tensors.emplace_back(at::from_blob(
          ptr, shape,
          [=](void *ptr) {
            std::cerr << "enter rocshmem_free " << ptr << "\n";
            at::hip::HIPGuard guard(current_device);
            at::hip::device_synchronize();
            std::cerr << "do rocshmem_free " << ptr << "\n";
            rocshmem_free(ptr);
            at::hip::device_synchronize();
            std::cerr << "exit rocshmem_free " << ptr << "\n";
          },
          option_gpu));
    } else {
      // FIXME: `rocshmem_ptr` is a devce side API.
      // void *rptr = rocshmem_ptr(ptr, rank_global);
      // PYROCSHMEM_CHECK(rptr != nullptr) << "rank " << rank;
      // tensors.emplace_back(at::from_blob(rptr, shape, option_gpu));
    }
  }

  return tensors;
}
#endif

PYBIND11_MODULE(_pyrocshmem, m) {
#if ENABLE_ROCSHMEM
  m.def("rocshmem_init", []() { rocshmem_init(); });
  m.def("rocshmem_finalize", []() { rocshmem_finalize(); });
  m.def("rocshmem_malloc", [](size_t size) {
    void *ptr = rocshmem_malloc(size);
    if (ptr == nullptr) {
      throw std::runtime_error("rocshmem_malloc failed");
    }
    return (intptr_t)ptr;
  });
#endif
  // TODO: find the related rocshmem Host side API.
  /*m.def("rocshmemx_get_uniqueid", []() {
    rocshmemx_uniqueid_t id;
    CHECK_ROCSHMEMX(rocshmemx_get_uniqueid(&id));
    std::string bytes((char *)&id, sizeof(id));
    return pybind11::bytes(bytes);
  });*/
  /*m.def("nvshmemx_init_attr_with_uniqueid", [](int rank, int nranks,
                                               pybind11::bytes bytes) {
    nvshmemx_uniqueid_t id;
    std::string id_str = bytes;
    if (id_str.size() != sizeof(id)) {
      throw std::runtime_error(
          "nvshmemx_init_attr_with_uniqueid: invalid size");
    }
    nvshmemx_init_attr_t init_attr;
    CHECK_ROCSHMEMX(
        nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &init_attr));
    memcpy(&id, id_str.data(), sizeof(id));
    CHECK_ROCSHMEMX(nvshmemx_init_attr(ROCSHMEMX_INIT_WITH_UNIQUEID,
  &init_attr));
  });*/
#if ENABLE_ROCSHMEM
  m.def("rocshmem_create_tensor",
        [](const std::vector<int64_t> shape, py::object dtype) {
          auto cast_dtype = torch::python::detail::py_object_to_dtype(dtype);
          return create_tensor(shape, cast_dtype);
        });
  m.def("rocshmem_barrier_all", []() { rocshmem_barrier_all(); });
#endif

#if ENABLE_ROCSHMEM
  m.def(
      "rocshmem_create_tensor_list_intra_node",
      [](const std::vector<int64_t> &shape, py::object dtype) {
        return rocshmem_create_tensor_list(
            shape, torch::python::detail::py_object_to_dtype(std::move(dtype)));
      },
      py::arg("shape"), py::arg("dtype"));
#endif
  m.def(
      "rocshmem_get_tensors_from_ipchandle",
      [](int64_t rank, int64_t world_size,
         const std::vector<torch::Tensor> &handles,
         const std::vector<int64_t> &offsets, const std::vector<int64_t> &shape,
         c10::ScalarType dtype) {
        return rocshmem_get_tensors_from_ipchandle(rank, world_size, handles,
                                                   offsets, shape, dtype);
      },
      py::arg("rank"), py::arg("world_size"), py::arg("handles"),
      py::arg("offsets"), py::arg("shape"), py::arg("dtype"));
  m.def("ipc_create_tensor_and_handle",
        [](const std::vector<int64_t> &shape, c10::ScalarType dtype) {
          return hipcreate_tensor_and_handle(shape, dtype);
        });
  m.def(
      "hipipc_create_tensor_list",
      [](c10::intrusive_ptr<c10d::ProcessGroup> group,
         const std::vector<int64_t> &shape, c10::ScalarType dtype) {
        return hipipc_create_tensor_list(group.get(), shape, dtype);
      },
      py::arg("group"), py::arg("shape"), py::arg("dtype"));
  m.def(
      "test_ipc_handle",
      [](c10::intrusive_ptr<c10d::ProcessGroup> group,
         const std::vector<int64_t> &shape, c10::ScalarType dtype) {
        return test_ipc_handle_impl(group.get(), shape, dtype);
      },
      py::arg("group"), py::arg("shape"), py::arg("dtype"));
}
