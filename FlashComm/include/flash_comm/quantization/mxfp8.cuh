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

#include "flash_comm/quantization/mxfp8.h"

namespace flash_comm {
namespace quantization {

__device__ __forceinline__ uint8_t float_to_e8m0(float value) {
  constexpr uint32_t kFP32MantissaBits = 23;
  if (isnan(value))
    return 0xFF;
  if (isinf(value))
    return 0xFE;
  if (value == 0.0f)
    return 0x00;

  uint32_t bits = __float_as_uint(value);
  uint8_t exponent = static_cast<uint8_t>(bits >> kFP32MantissaBits);
  uint32_t mantissa = bits & 0x7FFFFF;
  if ((mantissa > 0 && exponent != 0xFE) &&
      !(exponent == 0 && mantissa <= 0x400000)) {
    ++exponent;
  }
  return exponent;
}

__device__ __forceinline__ float e8m0_reciprocal(uint8_t biased_exp) {
  constexpr uint32_t kFP32MantissaBits = 23;
  if (biased_exp == 255)
    return __uint_as_float(0x7fffffff);
  if (biased_exp == 254)
    return __uint_as_float(0x00400000);
  return __uint_as_float(static_cast<uint32_t>(254 - biased_exp)
                         << kFP32MantissaBits);
}

constexpr int32_t kMXFP8FullChunkBytes =
    kMXFP8BlocksPerChunk * kMXFP8PackedBlockBytes;

__device__ __forceinline__ int32_t mxfp8_chunk_blocks(int32_t chunk,
                                                      int32_t num_blocks) {
  int32_t remaining = num_blocks - chunk * kMXFP8BlocksPerChunk;
  return remaining < kMXFP8BlocksPerChunk ? remaining : kMXFP8BlocksPerChunk;
}

__device__ __forceinline__ int32_t mxfp8_data_offset(int32_t block) {
  int32_t chunk = block / kMXFP8BlocksPerChunk;
  int32_t block_in_chunk = block % kMXFP8BlocksPerChunk;
  return chunk * kMXFP8FullChunkBytes + block_in_chunk * kMXFP8BlockSize;
}

__device__ __forceinline__ int32_t mxfp8_scale_offset(int32_t block,
                                                      int32_t num_blocks) {
  int32_t chunk = block / kMXFP8BlocksPerChunk;
  int32_t block_in_chunk = block % kMXFP8BlocksPerChunk;
  int32_t blocks_in_chunk = mxfp8_chunk_blocks(chunk, num_blocks);
  return chunk * kMXFP8FullChunkBytes + blocks_in_chunk * kMXFP8BlockSize +
         block_in_chunk;
}

} // namespace quantization
} // namespace flash_comm
