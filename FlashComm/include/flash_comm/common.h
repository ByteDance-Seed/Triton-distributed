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

#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace flash_comm {
enum class FlashCommDType { Float32, Float16, BFloat16, Int8, Int32, Int64 };

constexpr int32_t kMaxWorldSize = 72;

namespace internal {

class CheckError {
public:
  CheckError(const char *file, int line, const char *condition) {
    ss_ << "FLASH_CHECK failed at " << file << ":" << line
        << " -> Check failed: " << condition << ". ";
  }

  ~CheckError() {
    std::cerr << ss_.str() << std::endl;
    std::abort();
  }

  std::ostream &stream() { return ss_; }

private:
  std::ostringstream ss_;
};

} // namespace internal
} // namespace flash_comm

#if defined(__GNUC__) || defined(__clang__)
#define FLASH_LIKELY(x) __builtin_expect(!!(x), 1)
#else
#define FLASH_LIKELY(x) (x)
#endif

#define FLASH_CHECK(cond)                                                      \
  if (FLASH_LIKELY(cond))                                                      \
    ;                                                                          \
  else                                                                         \
    flash_comm::internal::CheckError(__FILE__, __LINE__, #cond).stream()

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err = x;                                                       \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA Error: ") +                   \
                               cudaGetErrorString(err) + " at " + __FILE__ +   \
                               ":" + std::to_string(__LINE__));                \
    }                                                                          \
  } while (0)

#define DRIVER_CHECK(x)                                                        \
  do {                                                                         \
    CUresult _driver_check_res = x;                                            \
    if (_driver_check_res != CUDA_SUCCESS) {                                   \
      const char *err_str;                                                     \
      cuGetErrorString(_driver_check_res, &err_str);                           \
      throw std::runtime_error(std::string("CUDA Driver Error: ") +            \
                               (err_str ? err_str : "Unknown") + " at " +      \
                               __FILE__ + ":" + std::to_string(__LINE__));     \
    }                                                                          \
  } while (0)

// kernel dispatch macro
#define DISPATCH_BOOL(VAL, ARG_NAME, ...)                                      \
  if ((VAL)) {                                                                 \
    constexpr bool ARG_NAME = true;                                            \
    __VA_ARGS__;                                                               \
  } else {                                                                     \
    constexpr bool ARG_NAME = false;                                           \
    __VA_ARGS__;                                                               \
  }

#define _DISPATCH_INT_CASE_IMPL(VAL, ARG_NAME, ...)                            \
  case VAL: {                                                                  \
    constexpr int ARG_NAME = VAL;                                              \
    __VA_ARGS__;                                                               \
    break;                                                                     \
  }

#define _DISPATCH_TYPE_CASE_IMPL(ENUM_VAL, CPP_TYPE, TYPE_NAME, ...)           \
  case ENUM_VAL: {                                                             \
    using TYPE_NAME = CPP_TYPE;                                                \
    __VA_ARGS__;                                                               \
    break;                                                                     \
  }

#define DISPATCH_INT_BY_LIST(LIST_MACRO, VAL, ARG_NAME, ...)                   \
  switch ((VAL)) {                                                             \
    LIST_MACRO(_DISPATCH_INT_CASE_IMPL, ARG_NAME, __VA_ARGS__)                 \
  default:                                                                     \
    throw std::runtime_error("Unsupported value for " #LIST_MACRO ": " +       \
                             std::to_string(VAL));                             \
  }

#define DISPATCH_TYPE_BY_LIST(LIST_MACRO, VAL, TYPE_NAME, ...)                 \
  switch ((VAL)) {                                                             \
    LIST_MACRO(_DISPATCH_TYPE_CASE_IMPL, TYPE_NAME, __VA_ARGS__)               \
  default:                                                                     \
    throw std::runtime_error("Unsupported DType");                             \
  }

#define SUPPORTED_HIDDEN_SIZES(OP, ...)                                        \
  OP(1536, __VA_ARGS__)                                                        \
  OP(2048, __VA_ARGS__)                                                        \
  OP(3584, __VA_ARGS__)                                                        \
  OP(5120, __VA_ARGS__)                                                        \
  OP(6144, __VA_ARGS__)                                                        \
  OP(7168, __VA_ARGS__)

#define SUPPORTED_TYPES(OP, ...)                                               \
  OP(flash_comm::FlashCommDType::Float32, float, __VA_ARGS__)                  \
  OP(flash_comm::FlashCommDType::Float16, __half, __VA_ARGS__)                 \
  OP(flash_comm::FlashCommDType::BFloat16, nv_bfloat16, __VA_ARGS__)

// Token types: only BFloat16 for now
#define SUPPORTED_TOKEN_TYPES(OP, ...)                                         \
  OP(flash_comm::FlashCommDType::BFloat16, nv_bfloat16, __VA_ARGS__)

// Weight types: only Float32 for now
#define SUPPORTED_WEIGHT_TYPES(OP, ...)                                        \
  OP(flash_comm::FlashCommDType::Float32, float, __VA_ARGS__)

#define SUPPORTED_TOPK(OP, ...)                                                \
  OP(6, __VA_ARGS__)                                                           \
  OP(8, __VA_ARGS__)

// Offset types: only Int32 for now
#define SUPPORTED_OFFSET_TYPES(OP, ...)                                        \
  OP(flash_comm::FlashCommDType::Int32, int32_t, __VA_ARGS__)

#define DISPATCH_HIDDEN_SIZE(VAL, TYPE_NAME, ...)                              \
  DISPATCH_INT_BY_LIST(SUPPORTED_HIDDEN_SIZES, VAL, TYPE_NAME, __VA_ARGS__)

#define DISPATCH_TOPK(VAL, TYPE_NAME, ...)                                     \
  DISPATCH_INT_BY_LIST(SUPPORTED_TOPK, VAL, TYPE_NAME, __VA_ARGS__)

#define DISPATCH_DTYPE(VAL, TYPE_NAME, ...)                                    \
  DISPATCH_TYPE_BY_LIST(SUPPORTED_TYPES, VAL, TYPE_NAME, __VA_ARGS__)

#define DISPATCH_TOKEN_DTYPE(VAL, TYPE_NAME, ...)                              \
  DISPATCH_TYPE_BY_LIST(SUPPORTED_TOKEN_TYPES, VAL, TYPE_NAME, __VA_ARGS__)

#define DISPATCH_WEIGHT_DTYPE(VAL, TYPE_NAME, ...)                             \
  DISPATCH_TYPE_BY_LIST(SUPPORTED_WEIGHT_TYPES, VAL, TYPE_NAME, __VA_ARGS__)

#define DISPATCH_OFFSET_TYPE(VAL, TYPE_NAME, ...)                              \
  DISPATCH_TYPE_BY_LIST(SUPPORTED_OFFSET_TYPES, VAL, TYPE_NAME, __VA_ARGS__)
