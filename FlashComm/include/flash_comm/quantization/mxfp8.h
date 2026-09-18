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

#include <cstdint>
#include <cuda_runtime.h>

#include "flash_comm/common.h"

namespace flash_comm {
namespace quantization {

constexpr int32_t kMXFP8BlockSize = 32;
constexpr int32_t kMXFP8BlocksPerChunk = 32;
constexpr int32_t kMXFP8PackedBlockBytes = 33;
// The fused intranode kernel reserves one consumer warp for the local rank and
// one for each remote rank, so its fixed 32-warp CTA supports at most eight
// ranks.
constexpr int32_t kMXFP8MaxIntranodeRanks = 8;

// cp.async.bulk requires 16-byte aligned addresses and transfer sizes. Shared
// staging-buffer bases may retain a stronger alignment independently.
constexpr int32_t kMXFP8RowAlignment = 128;

static_assert(kMXFP8RowAlignment % 16 == 0,
              "cp.async.bulk requires 16-byte aligned addresses and sizes");
static_assert(
    kMXFP8RowAlignment % 2 == 0,
    "packed rows travel through BF16-typed buffers, so the padded row "
    "size must stay even");

// E4M3 data plus one E8M0 scale byte per block, before row padding.
constexpr int32_t mxfp8_packed_payload_bytes(int32_t hidden_size) {
  return hidden_size / kMXFP8BlockSize * kMXFP8PackedBlockBytes;
}

// The padded wire row.  This is the single definition of the packed geometry:
// it is constexpr so kernels can derive their row size from the hidden size
// they are already templated on, which keeps retuning kMXFP8RowAlignment from
// silently invalidating a separately maintained list of packed sizes.
constexpr int32_t mxfp8_packed_row_bytes_of(int32_t hidden_size) {
  return (mxfp8_packed_payload_bytes(hidden_size) + kMXFP8RowAlignment - 1) /
         kMXFP8RowAlignment * kMXFP8RowAlignment;
}

// Packed rows are carried by the BF16 communication buffers, so kernels that
// address them elementwise want the row length in BF16 elements.
constexpr int32_t mxfp8_packed_row_bf16_elems(int32_t hidden_size) {
  return mxfp8_packed_row_bytes_of(hidden_size) / 2;
}

inline int32_t mxfp8_packed_row_bytes(int32_t hidden_size) {
  FLASH_CHECK(hidden_size > 0 && hidden_size % kMXFP8BlockSize == 0)
      << "hidden_size must be positive and divisible by " << kMXFP8BlockSize;
  return mxfp8_packed_row_bytes_of(hidden_size);
}

void mxfp8_quantize_cuda(void *x, void *packed, int32_t num_token,
                         int32_t hidden_size, int32_t packed_row_bytes,
                         int32_t num_sm, cudaStream_t stream);

} // namespace quantization
} // namespace flash_comm
