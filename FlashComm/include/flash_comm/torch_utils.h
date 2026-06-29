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

#include <initializer_list>
#include <torch/extension.h>

#include "flash_comm/common.h"

namespace flash_comm {

// convert torch::ScalarType to flash_comm::FlashCommDType
inline FlashCommDType get_flash_comm_dtype(torch::ScalarType type) {
  switch (type) {
  case torch::kFloat32:
    return FlashCommDType::Float32;
  case torch::kFloat16:
    return FlashCommDType::Float16;
  case torch::kBFloat16:
    return FlashCommDType::BFloat16;
  case torch::kInt32:
    return FlashCommDType::Int32;
  case torch::kInt64:
    return FlashCommDType::Int64;
  default:
    throw std::runtime_error("Unsupported dtype: " +
                             std::string(torch::toString(type)));
  }
}

inline void check_tensor_defined(const torch::Tensor &t, const char *name) {
  FLASH_CHECK(t.defined()) << name << " must be defined";
}

inline void check_tensor_device(const torch::Tensor &t, const char *name,
                                bool expect_cuda) {
  if (expect_cuda) {
    FLASH_CHECK(t.is_cuda()) << name << " must be a CUDA tensor";
  } else {
    FLASH_CHECK(!t.is_cuda()) << name << " must be a CPU tensor";
  }
}

inline void check_tensor_contiguous(const torch::Tensor &t, const char *name) {
  FLASH_CHECK(t.is_contiguous()) << name << " must be contiguous";
}

inline void check_tensor_dtype(const torch::Tensor &t, const char *name,
                               torch::ScalarType expect_dtype) {
  FLASH_CHECK(t.scalar_type() == expect_dtype)
      << name << " dtype mismatch, expected ScalarType="
      << static_cast<int64_t>(expect_dtype)
      << ", got ScalarType=" << static_cast<int64_t>(t.scalar_type());
}

inline void check_tensor_dim(const torch::Tensor &t, const char *name,
                             int64_t expect_dim) {
  FLASH_CHECK(t.dim() == expect_dim)
      << name << " must be " << expect_dim << "D";
}

inline void check_tensor_shape(const torch::Tensor &t, const char *name,
                               std::initializer_list<int64_t> expected_sizes) {
  check_tensor_dim(t, name, static_cast<int64_t>(expected_sizes.size()));
  auto sz = t.sizes();
  int64_t i = 0;
  for (int64_t e : expected_sizes) {
    if (e >= 0) {
      FLASH_CHECK(sz[i] == e) << name << " shape mismatch at dim " << i
                              << ", expected " << e << ", got " << sz[i];
    }
    ++i;
  }
}

inline void check_tensor_common(const torch::Tensor &t, const char *name,
                                bool expect_cuda,
                                torch::ScalarType expect_dtype,
                                int64_t expect_dim,
                                bool expect_contiguous = true) {
  check_tensor_defined(t, name);
  check_tensor_device(t, name, expect_cuda);
  if (expect_contiguous)
    check_tensor_contiguous(t, name);
  check_tensor_dtype(t, name, expect_dtype);
  check_tensor_dim(t, name, expect_dim);
}

inline void check_ptrs_tensor_i64(const torch::Tensor &t, int32_t n,
                                  const char *name) {
  check_tensor_common(t, name, /*expect_cuda=*/true, torch::kInt64,
                      /*expect_dim=*/1);
  check_tensor_shape(t, name, {n});
}

inline void check_pinned_cpu_i32_vector(const torch::Tensor &t, int32_t n,
                                        const char *name) {
  check_tensor_common(t, name, /*expect_cuda=*/false, torch::kInt32,
                      /*expect_dim=*/1);
  FLASH_CHECK(t.is_pinned())
      << name << " must be pinned CPU memory (pin_memory=True)";
  check_tensor_shape(t, name, {n});
}

} // namespace flash_comm
