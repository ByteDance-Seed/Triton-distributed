################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import triton
import triton.language as tl
from triton_dist.language.extra.language_extra import st
from .task_context import TaskBaseInfo, Scoreboard, TensorDesc


@triton.jit
def tile_wise_matmul_compute(tile_id, a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                             NUM_STAGES):
    # linear: a (M, K) x b (N, K) -> c (M, N)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ki in tl.range(0, k_tiles, num_stages=NUM_STAGES):
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
        b_ptrs = b_ptr + (offs_bn[:, None] * K + offs_k[None, :])

        a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b.T, accumulator)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def tile_range_matmul_compute_and_notify(tile_start, sb_base_ptr, a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M,
                                         BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_STAGES, TILE_READY_SIGNAL, NUM_SMS):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    for tile_id in tl.range(tile_start, num_tiles, NUM_SMS, flatten=True, warp_specialize=True):
        tile_wise_matmul_compute(tile_id, a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                 NUM_STAGES)
        st(sb_base_ptr + tile_id, TILE_READY_SIGNAL, "gpu", "release")


@triton.jit
def linear_task_compute(task_base_info: TaskBaseInfo, scoreboard: Scoreboard, BLOCK_SIZE_M: tl.constexpr,
                        BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, NUM_STAGES: tl.constexpr,
                        ALIGNMENT_K: tl.constexpr):

    input: TensorDesc = task_base_info.get_tensor(0)
    weight: TensorDesc = task_base_info.get_tensor(1)
    output: TensorDesc = task_base_info.get_tensor(2)

    M = input.size(0)
    K = input.size(1, ALIGNMENT_K)
    N = weight.size(0)
    a_ptr = input.data_ptr(tl.bfloat16)
    b_ptr = weight.data_ptr(tl.bfloat16)
    c_ptr = output.data_ptr(tl.bfloat16)

    tile_id = task_base_info.tile_id_or_start
    tile_wise_matmul_compute(tile_id, a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                             NUM_STAGES)
    scoreboard.release_tile(task_base_info, tile_id)
