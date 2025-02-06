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
#include "c10/cuda/CUDAGuard.h"
#include <ATen/ops/from_blob.h>
#include <bootstrap_device_host/nvshmem_uniqueid.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <nvshmemx.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/all.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>

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

#define CUDA_CHECK(cuda_error)                                                 \
  {                                                                            \
    if (cuda_error != cudaSuccess) {                                           \
      printf("cudaError %s in %s:%d\n", cudaGetErrorString(cuda_error),        \
             __func__, __LINE__);                                              \
      throw std::runtime_error("cuda error.");                                 \
    }                                                                          \
  }

#define PYNVSHMEM_CHECK(cond)                                                  \
  LazyLogger(cond) << __FILE__ << ":" << __LINE__                              \
                   << " Check failed: " #cond ". "
#define PYNVSHMEM_CHECK_NE(a, b) PYNVSHMEM_CHECK(((a) != (b)))

#define CHECK_NVSHMEMX(expr)                                                   \
  do {                                                                         \
    int x = expr;                                                              \
    if (x != NVSHMEMX_SUCCESS) {                                               \
      throw std::runtime_error(__FILE__ ":" + std::to_string(__LINE__) +       \
                               " " #expr " failed with status code " +         \
                               std::to_string(x));                             \
    }                                                                          \
  } while (0)

namespace {
std::array<const char *, 5> kNvshmemInitStatus = {
    "NVSHMEM_STATUS_NOT_INITIALIZED", "NVSHMEM_STATUS_IS_BOOTSTRAPPED",
    "NVSHMEM_STATUS_IS_INITIALIZED", "NVSHMEM_STATUS_LIMITED_MPG",
    "NVSHMEM_STATUS_FULL_MPG"};
void check_nvshmem_init() {
  CHECK(nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED);
}
} // namespace

#define NVSHMEMI_TYPENAME_P_IMPL_PYBIND(TYPENAME, TYPE)                        \
  void TYPENAME##_p(ptrdiff_t ptr, TYPE value, int peer) {                     \
    check_nvshmem_init();                                                      \
    nvshmem_##TYPENAME##_p((TYPE *)ptr, value, peer);                          \
  }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_P_IMPL_PYBIND)
#undef NVSHMEMI_TYPENAME_P_IMPL_PYBIND

inline torch::Tensor create_tensor(const std::vector<int64_t> &shape,
                                   c10::ScalarType dtype) {
  check_nvshmem_init();
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(
          c10::cuda::current_device());
  auto size =
      torch::elementSize(dtype) *
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  void *ptr = nvshmem_malloc(size);
  CHECK(ptr != nullptr) << " nvshmem_malloc failed for malloc " << size;
  return at::from_blob(
      ptr, shape, [](void *ptr) { nvshmem_free(ptr); }, option_gpu);
}

std::vector<torch::Tensor>
nvshmem_create_tensor_list(const std::vector<int64_t> &shape,
                           c10::ScalarType dtype) {
  check_nvshmem_init();
  auto current_device = c10::cuda::current_device();
  auto option_gpu =
      at::TensorOptions(at::kCUDA).dtype(dtype).device_index(current_device);
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), (size_t)1,
                              std::multiplies<>());
  PYNVSHMEM_CHECK_NE(size, 0);
  int local_world_size = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  int rank = nvshmem_my_pe();
  int local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(local_world_size);
  // std::cerr << "enter nvshmem_malloc\n";
  at::cuda::device_synchronize();
  // std::cerr << "do nvshmem_malloc\n";
  void *ptr = nvshmem_malloc(size);
  // std::cerr << "exit nvshmem_malloc " << ptr << "\n";

  CUDA_CHECK(cudaMemset(ptr, 0, size)); // memset the allocated buffer
  PYNVSHMEM_CHECK(ptr != nullptr);
  int rank_offset = rank - local_rank;
  for (int i = 0; i < local_world_size; i++) {
    // runs this call nvshmem failure, don't know why
    //  nvshmem_team_translate_pe(NVSHMEMX_TEAM_NODE, local_rank,
    //  NVSHMEM_TEAM_WORLD)
    int rank_global = i + rank_offset;
    if (rank == rank_global) {
      tensors.emplace_back(at::from_blob(
          ptr, shape,
          [=](void *ptr) {
            // std::cerr << "enter nvshmem_free "
            // << ptr << "\n";
            at::cuda::CUDAGuard guard(current_device);
            at::cuda::device_synchronize();
            // std::cerr << "do nvshmem_free " <<
            // ptr << "\n";
            nvshmem_free(ptr);
            at::cuda::device_synchronize();
            // std::cerr << "exit nvshmem_free "
            // << ptr << "\n";
          },
          option_gpu));
    } else {
      void *rptr = nvshmem_ptr(ptr, rank_global);
      PYNVSHMEM_CHECK(rptr != nullptr) << "rank " << rank;
      tensors.emplace_back(at::from_blob(rptr, shape, option_gpu));
    }
  }

  return tensors;
}

PYBIND11_MODULE(_pynvshmem, m) {
  m.def("nvshmemx_cumodule_init", [](intptr_t module) {
    CHECK_NVSHMEMX(nvshmemx_cumodule_init((CUmodule)module));
  });
  m.def("nvshmemx_cumodule_finalize", [](intptr_t module) {
    CHECK_NVSHMEMX(nvshmemx_cumodule_finalize((CUmodule)module));
  });
  m.def("nvshmem_malloc", [](size_t size) {
    void *ptr = nvshmem_malloc(size);
    if (ptr == nullptr) {
      throw std::runtime_error("nvshmem_malloc failed");
    }
    return (intptr_t)ptr;
  });
  m.def("nvshmem_ptr", [](intptr_t ptr, int peer) {
    return (intptr_t)nvshmem_ptr((void *)ptr, peer);
  });
  m.def("nvshmemx_mc_ptr", [](nvshmemx_team_t team, intptr_t ptr) {
    return (intptr_t)nvshmemx_mc_ptr(team, (void *)ptr);
  });
  m.def("nvshmemx_get_uniqueid", []() {
    nvshmemx_uniqueid_t id;
    CHECK_NVSHMEMX(nvshmemx_get_uniqueid(&id));
    std::string bytes((char *)&id, sizeof(id));
    return pybind11::bytes(bytes);
  });
  m.def("nvshmemx_init_attr_with_uniqueid", [](int rank, int nranks,
                                               pybind11::bytes bytes) {
    nvshmemx_uniqueid_t id;
    std::string id_str = bytes;
    if (id_str.size() != sizeof(id)) {
      throw std::runtime_error(
          "nvshmemx_init_attr_with_uniqueid: invalid size");
    }
    nvshmemx_init_attr_t init_attr;
    CHECK_NVSHMEMX(
        nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &init_attr));
    memcpy(&id, id_str.data(), sizeof(id));
    CHECK_NVSHMEMX(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &init_attr));
  });
#define NVSHMEMI_TYPENAME_P_PYBIND(TYPENAME, TYPE)                             \
  m.def("nvshmem_" #TYPENAME "_p", &TYPENAME##_p);
  NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_P_PYBIND)
#undef NVSHMEMI_TYPENAME_P_PYBIND
  m.def("nvshmem_create_tensor",
        [](const std::vector<int64_t> shape, py::object dtype) {
          auto cast_dtype = torch::python::detail::py_object_to_dtype(dtype);
          return create_tensor(shape, cast_dtype);
        });
  m.def("nvshmem_barrier_all", []() {
    check_nvshmem_init();
    nvshmem_barrier_all();
  });
  m.def("nvshmem_barrier_all_on_stream", [](intptr_t stream) {
    nvshmemx_barrier_all_on_stream((cudaStream_t)stream);
  });
  m.def(
      "nvshmem_create_tensor_list_intra_node",
      [](const std::vector<int64_t> &shape, py::object dtype) {
        return nvshmem_create_tensor_list(
            shape, torch::python::detail::py_object_to_dtype(std::move(dtype)));
      },
      py::arg("shape"), py::arg("dtype"));
}
