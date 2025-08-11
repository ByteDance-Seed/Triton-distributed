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

namespace py = pybind11;

using namespace rocshmem;

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

#define CHECK_ROCSHMEM(expr)                                                   \
  do {                                                                         \
    int x = expr;                                                              \
    if (x != ROCSHMEM_SUCCESS) {                                               \
      throw std::runtime_error(__FILE__ ":" + std::to_string(__LINE__) +       \
                               " " #expr " failed with status code " +         \
                               std::to_string(x));                             \
    }                                                                          \
  } while (0)

PYBIND11_MODULE(_pyrocshmem, m) {
  m.def("rocshmem_init", []() { rocshmem_init(); });
  m.def("rocshmem_my_pe", []() -> int { return rocshmem_my_pe(); });
  m.def("rocshmem_n_pes", []() -> int { return rocshmem_n_pes(); });
  m.def("rocshmem_team_my_pe", [](uintptr_t team) -> int {
    return rocshmem_team_my_pe((rocshmem_team_t)team);
  });
  m.def("rocshmem_team_n_pes", [](uintptr_t team) -> int {
    return rocshmem_team_n_pes((rocshmem_team_t)team);
  });
  m.def("rocshmem_malloc", [](size_t size) {
    void *ptr = rocshmem_malloc(size);
    if (ptr == nullptr) {
      throw std::runtime_error("rocshmem_malloc failed");
    }
    return (intptr_t)ptr;
  });
  m.def("rocshmem_free", [](intptr_t ptr) { rocshmem_free((void *)ptr); });
  m.def("rocshmem_ptr", [](intptr_t dest, int pe) -> intptr_t {
    return (intptr_t)rocshmem_ptr((void *)dest, pe);
  });
  m.def("rocshmem_finalize", []() { rocshmem_finalize(); });
  m.def("rocshmem_barrier_all", []() { rocshmem_barrier_all(); });
  m.def("rocshmem_get_device_ctx",
        []() -> int64_t { return (int64_t)rocshmem_get_device_ctx(); });
  m.def("rocshmem_get_uniqueid", []() {
    rocshmem_uniqueid_t uid;
    CHECK_ROCSHMEM(rocshmem_get_uniqueid(&uid));
    std::string bytes((char *)&uid, sizeof(uid));
    return pybind11::bytes(bytes);
  });
  m.def("rocshmem_init_attr", [](int rank, int nranks, pybind11::bytes bytes) {
    rocshmem_uniqueid_t uid;
    std::string uid_str = bytes;
    if (uid_str.size() != sizeof(uid)) {
      throw std::runtime_error("rocshmem_init_attr: invalid size");
    }
    rocshmem_init_attr_t init_attr;
    memcpy(&uid, uid_str.data(), uid_str.size());
    CHECK_ROCSHMEM(
        rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &init_attr));
    CHECK_ROCSHMEM(rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &init_attr));
  });
  m.def("rocshmem_putmem",
        [](intptr_t dest, const intptr_t source, size_t nelems, int pe) {
          rocshmem_putmem((void *)dest, (const void *)source, nelems, pe);
        });
  m.def("rocshmem_getmem",
        [](intptr_t dest, const intptr_t source, size_t nelems, int pe) {
          rocshmem_getmem((void *)dest, (const void *)source, nelems, pe);
        });
}