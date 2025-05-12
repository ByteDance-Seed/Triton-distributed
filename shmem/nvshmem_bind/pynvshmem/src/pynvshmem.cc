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
#include <bootstrap_device_host/nvshmem_uniqueid.h>
#include <cstdint>
#include <cstdio>
#include <host/nvshmem_api.h>
#include <iostream>
#include <mutex>
#include <nvshmemx.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <sstream>

namespace py = pybind11;

class LazyLogger {
public:
  LazyLogger(bool no_error = false) {
    _no_print = no_error;
    _no_error = no_error;
  };

  ~LazyLogger() noexcept(false) {
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
    cudaError x = (cuda_error);                                                \
    if (x != cudaSuccess) {                                                    \
      fprintf(stderr, "cudaError %s in %s:%d\n", cudaGetErrorString(x),        \
              __func__, __LINE__);                                             \
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
  PYNVSHMEM_CHECK(nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED);
}
} // namespace

#define NVSHMEMI_TYPENAME_P_IMPL_PYBIND(TYPENAME, TYPE)                        \
  void TYPENAME##_p(ptrdiff_t ptr, TYPE value, int peer) {                     \
    check_nvshmem_init();                                                      \
    nvshmem_##TYPENAME##_p((TYPE *)ptr, value, peer);                          \
  }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_P_IMPL_PYBIND)
#undef NVSHMEMI_TYPENAME_P_IMPL_PYBIND

// https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
// free pynvshmem from torch dependency
class symm_cuda_buffer {
public:
  symm_cuda_buffer(size_t size) : size_(size), ptr_(nullptr), own_data_(true) {
    CUDA_CHECK(cudaGetDevice(&device_index_));
    check_nvshmem_init();
    ptr_ = nvshmem_malloc(size);
    rank_ = nvshmem_my_pe();
    PYNVSHMEM_CHECK(ptr_ != nullptr);
  }

  symm_cuda_buffer(size_t size, int rank, void *ptr)
      : size_(size), rank_(rank), ptr_(ptr), own_data_(false) {}

  symm_cuda_buffer(const symm_cuda_buffer &) = delete;
  symm_cuda_buffer &operator=(const symm_cuda_buffer &) = delete;
  symm_cuda_buffer(symm_cuda_buffer &&other) noexcept
      : size_(other.size_), rank_(other.rank_), ptr_(other.ptr_),
        own_data_(other.own_data_), device_index_(other.device_index_) {
    other.ptr_ = nullptr;
    other.own_data_ = false;
  }

  ~symm_cuda_buffer() noexcept(false) {
    if (ptr_ && own_data_) {
      int current_device = -1;
      CUDA_CHECK(cudaGetDevice(&current_device));

      if (device_index_ != current_device) {
        static std::once_flag flag;
        std::call_once(flag, [&]() {
          fprintf(stderr,
                  "Warning: nvshmem_free is called from a different device "
                  "than the one that allocated the memory. This may lead "
                  "to undefined behavior. temporarily switch to device %d from "
                  "%d\n",
                  device_index_, current_device);
        });
        CUDA_CHECK(cudaSetDevice(device_index_));
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      nvshmem_free(ptr_);
      CUDA_CHECK(cudaDeviceSynchronize());
      ptr_ = nullptr;
      if (device_index_ != current_device) {
        CUDA_CHECK(cudaSetDevice(current_device));
      }
    }
  }

  symm_cuda_buffer symm_at(int rank) {
    PYNVSHMEM_CHECK(rank != rank_);
    void *ptr = nvshmem_ptr(ptr_, rank);
    PYNVSHMEM_CHECK(ptr != nullptr);
    return symm_cuda_buffer(size_, rank, ptr); // don't own data.
  }

  void *data_ptr() const { return ptr_; }

  size_t nbytes() const { return size_; }

private:
  size_t size_;
  int rank_;
  void *ptr_;
  bool own_data_ = false;
  int device_index_ = -1;
};

py::dict get_cuda_array_interface(const symm_cuda_buffer &buf) {
  py::dict interface;
  interface["data"] =
      py::make_tuple(reinterpret_cast<uintptr_t>(buf.data_ptr()), false);
  interface["shape"] = py::make_tuple(buf.nbytes());
  interface["typestr"] = "<i1";      // uint8 data type
  interface["strides"] = py::none(); // Contiguous memory
  interface["version"] = 3;
  return interface;
}

PYBIND11_MODULE(_pynvshmem, m) {

  py::class_<symm_cuda_buffer>(m, "symm_cuda_buffer")
      .def(py::init<size_t>())
      .def("data_ptr", &symm_cuda_buffer::data_ptr)
      .def("nbytes", &symm_cuda_buffer::nbytes)
      .def("symm_at", &symm_cuda_buffer::symm_at)
      .def_property_readonly("__cuda_array_interface__",
                             &get_cuda_array_interface);

  m.def("nvshmem_my_pe", []() -> int {
    check_nvshmem_init();
    return nvshmem_my_pe();
  });
  m.def("nvshmem_n_pes", []() -> int {
    check_nvshmem_init();
    return nvshmem_n_pes();
  });
  m.def("nvshmemx_cumodule_init", [](intptr_t module) {
    CHECK_NVSHMEMX(nvshmemx_cumodule_init((CUmodule)module));
  });
  m.def("nvshmemx_cumodule_finalize", [](intptr_t module) {
    CHECK_NVSHMEMX(nvshmemx_cumodule_finalize((CUmodule)module));
  });
  m.def("nvshmem_team_my_pe", [](int team) {
    check_nvshmem_init();
    return nvshmem_team_my_pe(team);
  });
  m.def("nvshmem_team_n_pes", [](int team) {
    check_nvshmem_init();
    return nvshmem_team_n_pes(team);
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

  m.def("nvshmem_barrier_all", []() {
    check_nvshmem_init();
    nvshmem_barrier_all();
  });
  m.def("nvshmemx_barrier_all_on_stream", [](intptr_t stream) {
    nvshmemx_barrier_all_on_stream((cudaStream_t)stream);
  });

  m.def("nvshmem_putmem",
        [](intptr_t dest, const intptr_t source, size_t nelems, int pe) {
          check_nvshmem_init();
          nvshmem_putmem((void *)dest, (const void *)source, nelems, pe);
        });
  m.def("nvshmem_getmem",
        [](intptr_t dest, const intptr_t source, size_t nelems, int pe) {
          check_nvshmem_init();
          nvshmem_getmem((void *)dest, (const void *)source, nelems, pe);
        });

  m.def("nvshmemx_putmem_on_stream",
        [](intptr_t dest, const intptr_t source, size_t nelems, int pe,
           intptr_t stream) {
          check_nvshmem_init();
          nvshmemx_putmem_on_stream((void *)dest, (const void *)source, nelems,
                                    pe, (cudaStream_t)stream);
        });
  m.def("nvshmemx_getmem_on_stream",
        [](intptr_t dest, const intptr_t source, size_t nelems, int pe,
           intptr_t stream) {
          check_nvshmem_init();
          nvshmemx_getmem_on_stream((void *)dest, (const void *)source, nelems,
                                    pe, (cudaStream_t)stream);
        });
  m.def("nvshmemx_putmem_signal_on_stream",
        [](intptr_t dest, const intptr_t source, size_t nelems,
           intptr_t sig_addr, uint64_t signal, int sig_op, int pe,
           intptr_t stream) {
          check_nvshmem_init();
          nvshmemx_putmem_signal_on_stream((void *)dest, (const void *)source,
                                           nelems, (uint64_t *)sig_addr, signal,
                                           sig_op, pe, (cudaStream_t)stream);
        });
}
