#include "c10/cuda/CUDAGuard.h"
#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <device/mxshmem_defines.h>
#include <host_device_common/mxshmem_error.h>
#include <host_device_common/mxshmem_types.h>
#include <mxshmem.h>
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

#define PYMXSHMEM_CHECK(cond)                                                  \
  LazyLogger(cond) << __FILE__ << ":" << __LINE__                              \
                   << " Check failed: " #cond ". "
#define PYMXSHMEM_CHECK_NE(a, b) PYMXSHMEM_CHECK(((a) != (b)))

#define CHECK_MXSHMEMX(expr)                                                   \
  do {                                                                         \
    int x = expr;                                                              \
    if (x != MXSHMEM_SUCCESS) {                                                \
      throw std::runtime_error(__FILE__ ":" + std::to_string(__LINE__) +       \
                               " " #expr " failed with status code " +         \
                               std::to_string(x));                             \
    }                                                                          \
  } while (0)

namespace {
std::array<const char *, 5> kMxshmemInitStatus = {
    "MXSHMEM_STATUS_NOT_INITIALIZED", "MXSHMEM_STATUS_IS_BOOTSTRAPPED",
    "MXSHMEM_STATUS_IS_INITIALIZED", "MXSHMEM_STATUS_LIMITED_MPG",
    "MXSHMEM_STATUS_FULL_MPG"};
void check_mxshmem_init() {
  CHECK(mxshmemx_init_status() >= MXSHMEM_STATUS_IS_INITIALIZED);
}
} // namespace

extern "C" void flush_l2c(cudaStream_t stream);

#define MXSHMEMI_TYPENAME_P_IMPL_PYBIND(TYPENAME, TYPE)                        \
  void TYPENAME##_p(ptrdiff_t ptr, TYPE value, int peer) {                     \
    check_mxshmem_init();                                                      \
    mxshmem_##TYPENAME##_p((TYPE *)ptr, value, peer);                          \
  }

MXSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(MXSHMEMI_TYPENAME_P_IMPL_PYBIND)
#undef MXSHMEMI_TYPENAME_P_IMPL_PYBIND

inline torch::Tensor create_tensor(const std::vector<int64_t> &shape,
                                   c10::ScalarType dtype) {
  check_mxshmem_init();
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(
          c10::cuda::current_device());
  auto size =
      torch::elementSize(dtype) *
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  void *ptr = mxshmem_malloc(size);
  CHECK(ptr != nullptr) << " mxshmem_malloc failed for malloc " << size;
  return at::from_blob(
      ptr, shape, [](void *ptr) { mxshmem_free(ptr); }, option_gpu);
}

std::vector<torch::Tensor>
mxshmem_create_tensor_list(const std::vector<int64_t> &shape,
                           c10::ScalarType dtype) {
  check_mxshmem_init();
  auto current_device = c10::cuda::current_device();
  auto option_gpu =
      at::TensorOptions(at::kCUDA).dtype(dtype).device_index(current_device);
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), (size_t)1,
                              std::multiplies<>());
  PYMXSHMEM_CHECK_NE(size, 0);
  int local_world_size = mxshmem_team_n_pes(MXSHMEMX_TEAM_NODE);
  int rank = mxshmem_my_pe();
  int local_rank = mxshmem_team_my_pe(MXSHMEMX_TEAM_NODE);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(local_world_size);
  at::cuda::device_synchronize();
  void *ptr = mxshmem_malloc(size);

  CUDA_CHECK(cudaMemset(ptr, 0, size)); // memset the allocated buffer
  PYMXSHMEM_CHECK(ptr != nullptr);
  int rank_offset = rank - local_rank;
  for (int i = 0; i < local_world_size; i++) {
    int rank_global = i + rank_offset;
    if (rank == rank_global) {
      tensors.emplace_back(at::from_blob(
          ptr, shape,
          [=](void *ptr) {
            at::cuda::CUDAGuard guard(current_device);
            at::cuda::device_synchronize();
            mxshmem_free(ptr);
            at::cuda::device_synchronize();
          },
          option_gpu));
    } else {
      void *rptr = mxshmem_ptr(ptr, rank_global);
      PYMXSHMEM_CHECK(rptr != nullptr) << "rank " << rank;
      tensors.emplace_back(at::from_blob(rptr, shape, option_gpu));
    }
  }

  return tensors;
}

PYBIND11_MODULE(_pymxshmem, m) {
  m.def("mxshmemx_mcmodule_init", [](intptr_t module) {
    CHECK_MXSHMEMX(mxshmemx_mcmodule_init((CUmodule)module));
  });
  m.def("mxshmemx_mcmodule_finalize", [](intptr_t module) {
    CHECK_MXSHMEMX(mxshmemx_mcmodule_finalize((CUmodule)module));
  });
  m.def("mxshmem_malloc", [](size_t size) {
    void *ptr = mxshmem_malloc(size);
    if (ptr == nullptr) {
      throw std::runtime_error("mxshmem_malloc failed");
    }
    return (intptr_t)ptr;
  });
  m.def("mxshmem_ptr", [](intptr_t ptr, int peer) {
    return (intptr_t)mxshmem_ptr((void *)ptr, peer);
  });
  m.def("mxshmemx_get_uniqueid", []() {
    mxshmemx_uniqueid_t id;
    CHECK_MXSHMEMX(mxshmemx_get_uniqueid(&id));
    std::string bytes((char *)&id, sizeof(id));
    return pybind11::bytes(bytes);
  });
  m.def("mxshmemx_init_attr_with_uniqueid", [](int rank, int nranks,
                                               pybind11::bytes bytes) {
    mxshmemx_uniqueid_t id;
    std::string id_str = bytes;
    if (id_str.size() != sizeof(id)) {
      throw std::runtime_error(
          "mxshmemx_init_attr_with_uniqueid: invalid size");
    }
    mxshmemx_init_attr_t init_attr;
    CHECK_MXSHMEMX(
        mxshmemx_set_attr_uniqueid_args(rank, nranks, &id, &init_attr));
    memcpy(&id, id_str.data(), sizeof(id));
    CHECK_MXSHMEMX(mxshmemx_init_attr(MXSHMEMX_INIT_WITH_UNIQUEID, &init_attr));
  });
#define MXSHMEMI_TYPENAME_P_PYBIND(TYPENAME, TYPE)                             \
  m.def("mxshmem_" #TYPENAME "_p", &TYPENAME##_p);
  MXSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(MXSHMEMI_TYPENAME_P_PYBIND)
#undef MXSHMEMI_TYPENAME_P_PYBIND
  m.def("mxshmem_create_tensor",
        [](const std::vector<int64_t> shape, py::object dtype) {
          auto cast_dtype = torch::python::detail::py_object_to_dtype(dtype);
          return create_tensor(shape, cast_dtype);
        });
  m.def("mxshmem_barrier_all", []() {
    check_mxshmem_init();
    mxshmem_barrier_all();
  });
  m.def("mxshmem_barrier_all_on_stream", [](intptr_t stream) {
    mxshmemx_barrier_all_on_stream((cudaStream_t)stream);
  });
  m.def(
      "mxshmem_create_tensor_list_intra_node",
      [](const std::vector<int64_t> &shape, py::object dtype) {
        return mxshmem_create_tensor_list(
            shape, torch::python::detail::py_object_to_dtype(std::move(dtype)));
      },
      py::arg("shape"), py::arg("dtype"));
  m.def("flush_l2c", [](intptr_t stream) { flush_l2c((cudaStream_t)stream); });
  m.def("mxshmem_finalize", []() { mxshmem_finalize(); });
}
