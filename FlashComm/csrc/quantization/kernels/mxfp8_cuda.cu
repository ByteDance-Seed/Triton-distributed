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

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "flash_comm/common.h"
#include "flash_comm/quantization/mxfp8.cuh"
#include "flash_comm/utils.cuh"

namespace flash_comm {
namespace quantization {
namespace kernels {

template <typename T>
__device__ __forceinline__ T *offset_ptr(T *base, int32_t index,
                                         int32_t stride) {
  return base + static_cast<int64_t>(index) * stride;
}

template <int32_t kHiddenSize, int32_t kPackedRowBytes, int32_t kNumWarps>
void __global__ __launch_bounds__(kNumWarps *WARP_SIZE,
                                  2048 / (kNumWarps * WARP_SIZE))
    kernel_mxfp8_quantize(const nv_bfloat16 *x, uint8_t *packed,
                          int32_t num_token) {
  constexpr int32_t kNumBlocks = kHiddenSize / kMXFP8BlockSize;
  constexpr int32_t kLanesPerBlock = 2;
  constexpr int32_t kBlocksPerWarp = WARP_SIZE / kLanesPerBlock;
  constexpr int32_t kNumBlockGroups =
      (kNumBlocks + kBlocksPerWarp - 1) / kBlocksPerWarp;
  static_assert(kNumBlockGroups % kNumWarps == 0,
                "quant warps must evenly divide the block groups");
  constexpr int32_t kGroupsPerWarp = kNumBlockGroups / kNumWarps;
  const int32_t lane = threadIdx.x % WARP_SIZE;
  const int32_t warp = threadIdx.x / WARP_SIZE;
  const int32_t block_in_warp = lane / kLanesPerBlock;
  const int32_t lane_in_block = lane % kLanesPerBlock;
  for (int32_t token = blockIdx.x; token < num_token; token += gridDim.x) {
    const nv_bfloat16 *src = offset_ptr(x, token, kHiddenSize);
    uint8_t *dst = offset_ptr(packed, token, kPackedRowBytes);
#pragma unroll
    for (int32_t group_iter = 0; group_iter < kGroupsPerWarp; ++group_iter) {
      int32_t block_group = group_iter * kNumWarps + warp;
      int32_t mx_block = block_group * kBlocksPerWarp + block_in_warp;
      bool valid_block = mx_block < kNumBlocks;
      union {
        uint4 raw[2];
        __nv_bfloat162 value[8];
      } input_vec;
      input_vec.raw[0] = make_uint4(0, 0, 0, 0);
      input_vec.raw[1] = make_uint4(0, 0, 0, 0);
      if (valid_block) {
        const uint4 *block_src =
            reinterpret_cast<const uint4 *>(src + mx_block * kMXFP8BlockSize);
        input_vec.raw[0] = block_src[lane_in_block];
        input_vec.raw[1] = block_src[lane_in_block + kLanesPerBlock];
      }
      float2 values[8];
      float amax = 0.0f;
#pragma unroll
      for (int32_t i = 0; i < 8; ++i) {
        values[i] = __bfloat1622float2(input_vec.value[i]);
        amax = fmaxf(amax, fmaxf(fabsf(values[i].x), fabsf(values[i].y)));
      }
#pragma unroll
      for (int32_t delta = kLanesPerBlock / 2; delta > 0; delta >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta, WARP_SIZE));
      }
      uint32_t scale_word = 0;
      if (lane_in_block == 0) {
        scale_word = float_to_e8m0(amax * (1.0f / 448.0f));
      }
      scale_word =
          __shfl_sync(0xffffffff, scale_word, block_in_warp * kLanesPerBlock);
      uint8_t scale = static_cast<uint8_t>(scale_word);
      float scale_reciprocal = e8m0_reciprocal(scale);
      __nv_fp8x2_storage_t quantized[8];
#pragma unroll
      for (int32_t i = 0; i < 8; ++i) {
        quantized[i] = __nv_cvt_float2_to_fp8x2(
            make_float2(values[i].x * scale_reciprocal,
                        values[i].y * scale_reciprocal),
            __NV_SATFINITE, __NV_E4M3);
      }
      if (valid_block) {
        uint32_t first_lo = static_cast<uint32_t>(quantized[0]) |
                            (static_cast<uint32_t>(quantized[1]) << 16);
        uint32_t first_hi = static_cast<uint32_t>(quantized[2]) |
                            (static_cast<uint32_t>(quantized[3]) << 16);
        uint32_t second_lo = static_cast<uint32_t>(quantized[4]) |
                             (static_cast<uint32_t>(quantized[5]) << 16);
        uint32_t second_hi = static_cast<uint32_t>(quantized[6]) |
                             (static_cast<uint32_t>(quantized[7]) << 16);
        uint32_t exchange_lo = lane_in_block == 0 ? second_lo : first_lo;
        uint32_t exchange_hi = lane_in_block == 0 ? second_hi : first_hi;
        uint32_t peer_lo =
            __shfl_xor_sync(0xffffffff, exchange_lo, 1, WARP_SIZE);
        uint32_t peer_hi =
            __shfl_xor_sync(0xffffffff, exchange_hi, 1, WARP_SIZE);
        uint4 packed_fp8 =
            lane_in_block == 0
                ? make_uint4(first_lo, first_hi, peer_lo, peer_hi)
                : make_uint4(peer_lo, peer_hi, second_lo, second_hi);
        reinterpret_cast<uint4 *>(
            dst + mxfp8_data_offset(mx_block))[lane_in_block] = packed_fp8;
        if (lane_in_block == 0) {
          dst[mxfp8_scale_offset(mx_block, kNumBlocks)] = scale;
        }
      }
    }
    constexpr int32_t kPayloadBytes = mxfp8_packed_payload_bytes(kHiddenSize);
    for (int32_t i = kPayloadBytes + threadIdx.x; i < kPackedRowBytes;
         i += blockDim.x) {
      dst[i] = 0;
    }
  }
}

} // namespace kernels

void mxfp8_quantize_cuda(void *x, void *packed, int32_t num_token,
                         int32_t hidden_size, int32_t packed_row_bytes,
                         int32_t num_sm, cudaStream_t stream) {
  bool use_token_grid = num_sm <= 0;
  if (num_sm <= 0) {
    int32_t device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount,
                                      device));
  }
  DISPATCH_HIDDEN_SIZE(hidden_size, kHiddenSize, {
    constexpr int32_t kNumMXBlocks = kHiddenSize / kMXFP8BlockSize;
    constexpr int32_t kNumBlockGroups =
        (kNumMXBlocks + (WARP_SIZE / 2) - 1) / (WARP_SIZE / 2);
    constexpr int32_t kNumWarps = kNumBlockGroups == 14      ? 14
                                  : kNumBlockGroups % 8 == 0 ? 8
                                  : kNumBlockGroups % 7 == 0 ? 7
                                  : kNumBlockGroups % 6 == 0 ? 6
                                  : kNumBlockGroups % 5 == 0 ? 5
                                  : kNumBlockGroups % 4 == 0 ? 4
                                  : kNumBlockGroups % 3 == 0 ? 3
                                                             : 2;
    constexpr int32_t kResidentBlocks = 2048 / (kNumWarps * WARP_SIZE);
    int32_t grid_target = num_sm * kResidentBlocks;
    int32_t num_blocks =
        use_token_grid ? num_token
                       : (num_token < grid_target ? num_token : grid_target);
    num_blocks = num_blocks > 0 ? num_blocks : 1;
    constexpr int32_t kPackedRowBytes = mxfp8_packed_row_bytes_of(kHiddenSize);
    FLASH_CHECK(packed_row_bytes == kPackedRowBytes)
        << "packed_row_bytes mismatch, expected " << kPackedRowBytes << ", got "
        << packed_row_bytes;
    kernels::kernel_mxfp8_quantize<kHiddenSize, kPackedRowBytes, kNumWarps>
        <<<num_blocks, kNumWarps * WARP_SIZE, 0, stream>>>(
            reinterpret_cast<nv_bfloat16 *>(x),
            reinterpret_cast<uint8_t *>(packed), num_token);
  });
  CUDA_CHECK(cudaGetLastError());
}

} // namespace quantization
} // namespace flash_comm
