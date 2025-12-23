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
# bash ./scripts/launch_amd.sh --nproc_per_node=1 python/triton_dist/kernels/amd/gemm.py --autotune --impl matmul_persistent --trans_b

import argparse
import datetime

import torch

import triton
import triton.language as tl
import triton_dist.tune
from triton_dist.kernels.amd.perf_model import get_max_shared_memory_size
from triton_dist.language.extra.language_extra import (__syncthreads, ld, st, tid)
from triton_dist.profiler_utils import AutoExportProfiler, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.tools.profiler import Profiler, alloc_profiler_buffer, export_to_perfetto_trace
from triton_dist.utils import sleep_async

SPLIT_K_ALGO_FAST = 0
SPLIT_K_ALGO_DETERMINISTIC = 1
SPLIT_K_ALGO_DETERMINISTIC_FP32_ACC = 2


@triton_dist.jit
def _compute_pid(pid, NUM_SMS, NUM_XCDS, GROUP_SIZE_M):
    GROUP_SIZE_M_XCDS = NUM_XCDS * GROUP_SIZE_M
    if pid > NUM_SMS // GROUP_SIZE_M_XCDS * GROUP_SIZE_M_XCDS:
        return pid
    xcd_id = pid % NUM_XCDS
    load_pid = pid // NUM_XCDS
    local_chunk_id = load_pid // GROUP_SIZE_M
    pos_in_chunk = load_pid % GROUP_SIZE_M
    new_pid = (pos_in_chunk + xcd_id * GROUP_SIZE_M + local_chunk_id * GROUP_SIZE_M * NUM_XCDS)
    return new_pid


@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0})
@triton_dist.jit
def kernel_gemm(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                GROUP_SIZE_M: tl.constexpr, NUM_XCDS: tl.constexpr, EVEN_K: tl.constexpr, profiler_buf=None,
                PROFILE: tl.constexpr = False):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if PROFILE:
        is_leader = (tid(0) == 0)
        profiler = Profiler.create(profiler_buf, 0, is_leader=is_leader, ENABLE_PROFILING=True)
        profiler = profiler.record(is_start=True, task_type=0)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rk = tl.arange(0, BLOCK_SIZE_K)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)

    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    if not EVEN_K:
        loop_k -= 1

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in range(0, loop_k):
        a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
        b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
        acc += tl.dot(a, b)
        A_BASE += BLOCK_SIZE_K * stride_ak
        B_BASE += BLOCK_SIZE_K * stride_bk

    if not EVEN_K:
        k = loop_k
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
        b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
        acc += tl.dot(a, b)

    c = acc.to(C.type.element_ty)
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C_, c, c_mask)
    if PROFILE:
        profiler = profiler.record(is_start=False, task_type=0)


@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0})
@triton_dist.jit
def kernel_gemm_persistent(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                           BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                           GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr,
                           EVEN_K: tl.constexpr, profiler_buf=None, PROFILE: tl.constexpr = False):
    if PROFILE:
        is_leader = (tid(0) == 0)
        profiler = Profiler.create(profiler_buf, 0, is_leader=is_leader, ENABLE_PROFILING=True)
        profiler = profiler.record(is_start=True, task_type=0)

    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _compute_pid(pid, NUM_SMS, NUM_XCDS, GROUP_SIZE_M)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    for tile_id in range(pid, total_tiles, NUM_SMS):

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        tl.assume(pid_m > 0)
        tl.assume(pid_n > 0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)
    if PROFILE:
        profiler = profiler.record(is_start=False, task_type=0)


# too easy to generate code that raise register spills. why?
@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0})
@triton_dist.jit
def kernel_gemm_split_k_persistent(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                                   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                                   GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr,
                                   EVEN_K: tl.constexpr, profiler_buf=None, PROFILE: tl.constexpr = False,
                                   SPLIT_K: tl.constexpr = 1, SPLITK_ALGO: tl.constexpr = 0, C_fp32_ptr=None):
    if PROFILE:
        is_leader = (tid(0) == 0)
        profiler = Profiler.create(profiler_buf, 0, is_leader=is_leader, ENABLE_PROFILING=True)
        profiler = profiler.record(is_start=True, task_type=0)

    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _compute_pid(pid, NUM_SMS, NUM_XCDS, GROUP_SIZE_M)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    K_per_split = tl.cdiv(K // SPLIT_K, BLOCK_SIZE_K) * BLOCK_SIZE_K

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    for pid_k in range(SPLIT_K):
        for tile_id in tl.range(pid, total_tiles, NUM_SMS, num_stages=2):
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rk = pid_k * K_per_split + tl.arange(0, BLOCK_SIZE_K)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            tl.assume(pid_m > 0)
            tl.assume(pid_n > 0)

            loop_k = tl.cdiv(min(K_per_split, K - K_per_split * pid_k), BLOCK_SIZE_K)
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for k in range(0, loop_k):
                k_remaining = K - k * BLOCK_SIZE_K
                a = tl.load(A_BASE, mask=rk[None, :] < k_remaining, other=0.0)
                b = tl.load(B_BASE, mask=rk[:, None] < k_remaining, other=0.0)
                acc += tl.dot(a, b)
                A_BASE += BLOCK_SIZE_K * stride_ak
                B_BASE += BLOCK_SIZE_K * stride_bk

            rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            c_mask = (rm[:, None] < M) & (rn[None, :] < N)
            c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            # already keept the order. no atomic_add and memory release
            if SPLITK_ALGO == 0 or SPLITK_ALGO == 1:
                if pid_k == 0:
                    tl.store(c_ptrs, acc.to(C.dtype.element_ty), c_mask)
                else:
                    c_val = tl.load(c_ptrs, c_mask)
                    tl.store(c_ptrs, (c_val + acc).to(C.dtype.element_ty), c_mask)
            elif SPLITK_ALGO == 2:
                c_fp32_ptrs = C_fp32_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
                if pid_k == 0:
                    tl.store(c_fp32_ptrs, acc, c_mask)
                else:
                    c_val = tl.load(c_fp32_ptrs, c_mask)
                    if pid_k != SPLIT_K - 1:
                        tl.store(c_fp32_ptrs, (c_val + acc), c_mask)
                    else:
                        tl.store(c_ptrs, (c_val + acc).to(C_fp32_ptr.type.element_ty), c_mask)
        __syncthreads()

    if PROFILE:
        profiler = profiler.record(is_start=False, task_type=0)


@triton.heuristics({'EVEN_K': lambda args: args['K'] % (args['BLOCK_SIZE_K'] * args['SPLIT_K']) == 0})
@triton_dist.jit
def kernel_gemm_splitk(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                       BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                       GROUP_SIZE_M: tl.constexpr, NUM_XCDS: tl.constexpr, EVEN_K: tl.constexpr, profiler_buf=None,
                       PROFILE: tl.constexpr = False, SPLIT_K: tl.constexpr = 1, SPLITK_ALGO: tl.constexpr = 0,
                       splitk_workspace=None):
    pid_mn = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if PROFILE:
        is_leader = (tid(0) == 0)
        profiler = Profiler.create(profiler_buf, 0, is_leader=is_leader, ENABLE_PROFILING=True)
        profiler = profiler.record(is_start=True, task_type=0)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    pid_k = tl.program_id(1)

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

    A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)

    loop_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in range(0, loop_k):
        k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
        a = tl.load(A_BASE, mask=rk[None, :] < k_remaining, other=0.0)
        b = tl.load(B_BASE, mask=rk[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        A_BASE += BLOCK_SIZE_K * stride_ak * SPLIT_K
        B_BASE += BLOCK_SIZE_K * stride_bk * SPLIT_K

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    if SPLITK_ALGO == 0:
        tl.atomic_add(c_ptrs, acc.to(C.type.element_ty), c_mask, "relaxed")
    elif SPLITK_ALGO == 1:
        while ld(splitk_workspace + pid_mn, semantic="acquire", scope="agent") != pid_k:
            pass

        if pid_k == 0:
            tl.store(c_ptrs, acc.to(C.type.element_ty), c_mask)
        else:
            c_val = tl.load(c_ptrs, c_mask)
            tl.store(c_ptrs, (acc + c_val).to(C.type.element_ty), c_mask)

        __syncthreads()
        if tid(0) == 0:
            st(splitk_workspace + pid_mn, (pid_k + 1) % SPLIT_K, semantic="release", scope="agent")
    elif SPLITK_ALGO == 2:  # the very deterministic implementation, accumulate with FP32
        thread_idx = tid(0)
        while ld(splitk_workspace + pid_mn, semantic="acquire", scope="agent") != pid_k:
            pass

        c_buffer = tl.cast(splitk_workspace + 1024 * 64, tl.pointer_type(tl.float32))  # pid_mn < 64K
        c_buffer_ptrs = c_buffer + rm[:, None] * stride_cm + rn[None, :] * stride_cn

        if pid_k == 0:
            tl.store(c_buffer_ptrs, acc, c_mask)
        elif pid_k != SPLIT_K - 1:
            c_val = tl.load(c_buffer_ptrs, c_mask)
            tl.store(c_buffer_ptrs, acc + c_val, c_mask)
        else:
            c_val = tl.load(c_buffer_ptrs, c_mask)
            tl.store(c_ptrs, (acc + c_val).to(C.dtype.element_ty), c_mask)

        __syncthreads()
        if thread_idx == 0:
            st(splitk_workspace + pid_mn, (pid_k + 1) % SPLIT_K, semantic="release", scope="agent")
    else:
        tl.static_assert(False)

    if PROFILE:
        profiler = profiler.record(is_start=False, task_type=0)


@triton_dist.jit
def splitk_epilogue(C_ptr, acc, pid_k, pid_m, pid_n, pid_mn, M, N, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr, SPLIT_K: tl.constexpr, SPLITK_ALGO: tl.constexpr, splitk_workspace_ptr,
                    C_fp32_ptr):
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn

    if SPLITK_ALGO == 0:
        tl.atomic_add(c_ptrs, acc.to(C_ptr.type.element_ty), c_mask, "relaxed")
    elif SPLITK_ALGO == 1:
        thread_idx = tid(0)
        while ld(splitk_workspace_ptr + pid_mn, semantic="acquire", scope="agent") != pid_k:
            pass

        if pid_k == 0:
            tl.store(c_ptrs, acc.to(C_ptr.type.element_ty), c_mask)
        else:
            c_val = tl.load(c_ptrs, c_mask)
            tl.store(c_ptrs, (acc + c_val).to(C_ptr.type.element_ty), c_mask)

        __syncthreads()
        if thread_idx == 0:
            st(splitk_workspace_ptr + pid_mn, (pid_k + 1) % SPLIT_K, semantic="release", scope="agent")
    elif SPLITK_ALGO == 2:  # the very deterministic implementation, accumulate with FP32
        thread_idx = tid(0)
        while ld(splitk_workspace_ptr + pid_mn, semantic="acquire", scope="agent") != pid_k:
            pass

        c_fp32_ptrs = C_fp32_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn

        if pid_k == 0:
            tl.store(c_fp32_ptrs, acc, c_mask)
        elif pid_k != SPLIT_K - 1:
            c_val = tl.load(c_fp32_ptrs, c_mask)
            tl.store(c_fp32_ptrs, acc + c_val, c_mask)
        else:
            c_val = tl.load(c_fp32_ptrs, c_mask)
            tl.store(c_ptrs, (acc + c_val).to(C_ptr.dtype.element_ty), c_mask)

        __syncthreads()
        if thread_idx == 0:
            st(splitk_workspace_ptr + pid_mn, (pid_k + 1) % SPLIT_K, semantic="release", scope="agent")
    elif SPLITK_ALGO == 3:
        c_fp32_ptrs = C_fp32_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn + pid_k * M * stride_cm
        tl.store(c_fp32_ptrs, acc, c_mask)  # leave it to epilogue
    else:
        tl.static_assert(False)


@triton.heuristics({'EVEN_K': lambda args: args['K'] % (args['BLOCK_SIZE_K'] * args['SPLIT_K']) == 0})
@triton_dist.jit
def kernel_gemm_splitk_chunked(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                               GROUP_SIZE_M: tl.constexpr, NUM_XCDS: tl.constexpr, EVEN_K: tl.constexpr,
                               profiler_buf=None, PROFILE: tl.constexpr = False, SPLIT_K: tl.constexpr = 1,
                               SPLITK_ALGO: tl.constexpr = 0, splitk_workspace_ptr=None, C_fp32_ptr=None):
    pid_mn = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if PROFILE:
        is_leader = (tid(0) == 0)
        profiler = Profiler.create(profiler_buf, 0, is_leader=is_leader, ENABLE_PROFILING=True)
        profiler = profiler.record(is_start=True, task_type=0)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    K_per_split = tl.cdiv(K // SPLIT_K, BLOCK_SIZE_K) * BLOCK_SIZE_K

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rk = pid_k * K_per_split + tl.arange(0, BLOCK_SIZE_K)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

    A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)

    loop_k = tl.cdiv(min(K_per_split, K - K_per_split * pid_k), BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in range(0, loop_k):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(A_BASE, mask=rk[None, :] < k_remaining, other=0.0)
        b = tl.load(B_BASE, mask=rk[:, None] < k_remaining, other=0.0, cache_modifier=".ca")
        acc += tl.dot(a, b)
        A_BASE += BLOCK_SIZE_K * stride_ak
        B_BASE += BLOCK_SIZE_K * stride_bk

    splitk_epilogue(C, acc, pid_k, pid_m, pid_n, pid_mn, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N,
                    SPLIT_K, SPLITK_ALGO, splitk_workspace_ptr, C_fp32_ptr)
    if PROFILE:
        profiler = profiler.record(is_start=False, task_type=0)


NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
DEFAULT_CONFIG = triton.Config(
    {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "NUM_SMS": NUM_SMS,
        "NUM_XCDS": 4,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
    }, num_warps=8, num_stages=2)

DEFAULT_CONFIG_PERSISTENT = triton.Config(
    {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 4,
        "NUM_SMS": NUM_SMS,
        "NUM_XCDS": 4,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
    }, num_warps=8, num_stages=2)


def get_config_space():
    NUM_XCDS = 4
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                "GROUP_SIZE_M": GROUP_SIZE_M,
                "NUM_XCDS": NUM_XCDS,
                "waves_per_eu": waves_per_eu,
                "matrix_instr_nonkdim": 16,
            }, num_stages=num_stages, num_warps=num_warps)
        for BLOCK_SIZE_M in [32, 64, 128, 256]
        for BLOCK_SIZE_N in [32, 64, 128, 256]
        for BLOCK_SIZE_K in [32, 64, 128]
        for GROUP_SIZE_M in [1]
        for waves_per_eu in [2]
        for num_warps in [4, 8]
        for num_stages in [2]
    ]


def get_splitk_config_space():
    return get_config_space()


def get_config_space_persistent():
    # persistent has more limit
    NUM_XCDS = 4
    return [DEFAULT_CONFIG_PERSISTENT] + [
        triton.Config(
            {
                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                "GROUP_SIZE_M": GROUP_SIZE_M,
                "NUM_SMS": NUM_SMS,
                "NUM_XCDS": NUM_XCDS,
                "waves_per_eu": waves_per_eu,
                "matrix_instr_nonkdim": 16,
            }, num_stages=num_stages, num_warps=num_warps)
        for BLOCK_SIZE_M in [32, 64, 128, 256]
        for BLOCK_SIZE_N in [32, 64, 128, 256]
        for BLOCK_SIZE_K in [32, 64, 128]
        for GROUP_SIZE_M in [1]
        for waves_per_eu in [2]
        for num_warps in [8]
        for num_stages in [2]
    ]


def key_fn(A: torch.Tensor, B: torch.Tensor, *args, **kwargs):
    return triton_dist.tune.to_hashable(A), triton_dist.tune.to_hashable(B)


def key_fn_splitk(A: torch.Tensor, B: torch.Tensor, *args, **kwargs):
    return triton_dist.tune.to_hashable(A), triton_dist.tune.to_hashable(B), kwargs.get("split_k"), kwargs.get(
        "split_k_algo")


def prune_fn_by_shared_mem(config, A: torch.Tensor, B: torch.Tensor, *args, **kwargs):
    gemm_config: triton.Config = config["config"]
    BLOCK_SIZE_M = gemm_config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = gemm_config.kwargs["BLOCK_SIZE_K"]
    num_stages = gemm_config.num_stages
    itemsize = A.dtype.itemsize
    shared_memory_size = (BLOCK_SIZE_M * BLOCK_SIZE_K + BLOCK_SIZE_K * BLOCK_SIZE_N) * itemsize * (num_stages - 1)
    max_occupancy = int(get_max_shared_memory_size(0)) // shared_memory_size
    if max_occupancy < 1:
        return False
    return True


def prune_fn_by_quatization(config, A: torch.Tensor, B: torch.Tensor, *args, MAX_BLOCK_SIZE_M=256, MAX_BLOCK_SIZE_N=256,
                            **kwargs):
    gemm_config: triton.Config = config["config"]
    BLOCK_SIZE_M = gemm_config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.kwargs["BLOCK_SIZE_N"]
    M, K = A.shape
    K, N = B.shape

    def _tile_ratio(m, block_size_m):
        tiled_m = triton.cdiv(m, block_size_m)
        return m / (tiled_m * block_size_m)

    def _wave_ratio(block_size_m, block_size_n):
        tiled_m = triton.cdiv(M, block_size_m)
        tiled_n = triton.cdiv(N, block_size_n)
        num_tiles = tiled_m * tiled_n
        wavefronts = triton.cdiv(num_tiles, NUM_SMS)
        return wavefronts * NUM_SMS / num_tiles

    # should we use larget block_size: 1. tile quantization 2. wave quantization
    # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
    # use larger block size does not make tile quantization better
    if BLOCK_SIZE_M < MAX_BLOCK_SIZE_M:
        if _tile_ratio(M, BLOCK_SIZE_M) / _tile_ratio(M, BLOCK_SIZE_M * 2) < 1.05 and _wave_ratio(
                BLOCK_SIZE_M, BLOCK_SIZE_N) / _wave_ratio(BLOCK_SIZE_M * 2, BLOCK_SIZE_N) < 1.05:
            return False
    if BLOCK_SIZE_N < MAX_BLOCK_SIZE_N:
        if _tile_ratio(N, BLOCK_SIZE_N) / _tile_ratio(N, BLOCK_SIZE_N * 2) < 1.05 and _wave_ratio(
                BLOCK_SIZE_M, BLOCK_SIZE_N) / _wave_ratio(BLOCK_SIZE_M, BLOCK_SIZE_N * 2) < 1.05:
            return False

    return True


def prune_fn_persistent(config, A: torch.Tensor, B: torch.Tensor, *args, **kwargs):
    if not prune_fn_by_shared_mem(config, A, B, *args, **kwargs):
        return False

    if not prune_fn_by_quatization(config, A, B, *args, **kwargs, MAX_BLOCK_SIZE_M=256, MAX_BLOCK_SIZE_N=256):
        return False

    return True


def prune_fn(config, A: torch.Tensor, B: torch.Tensor, *args, **kwargs):
    if not prune_fn_by_shared_mem(config, A, B, *args, **kwargs):
        return False

    if not prune_fn_by_quatization(config, A, B, *args, **kwargs, MAX_BLOCK_SIZE_M=128, MAX_BLOCK_SIZE_N=128):
        return False

    return True


@triton_dist.tune.autotune(config_space=[{"config": c} for c in get_config_space_persistent()], key_fn=key_fn,
                           prune_fn=prune_fn_persistent)
def matmul_persistent_triton(A: torch.Tensor, B: torch.Tensor, config: triton.Config = DEFAULT_CONFIG_PERSISTENT,
                             profiler_buf: torch.Tensor | None = None):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    kernel_gemm_persistent[(NUM_SMS, )](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                                        C.stride(0), C.stride(1), **config.all_kwargs(), profiler_buf=profiler_buf,
                                        PROFILE=profiler_buf is not None)
    return C


@triton_dist.tune.autotune(config_space=[{"config": c} for c in get_config_space_persistent()], key_fn=key_fn_splitk,
                           prune_fn=prune_fn_persistent)
def matmul_split_k_persistent_triton(A: torch.Tensor, B: torch.Tensor, split_k=4, split_k_algo=2,
                                     config: triton.Config = DEFAULT_CONFIG_PERSISTENT,
                                     profiler_buf: torch.Tensor | None = None):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    C_fp32 = None
    if split_k_algo == 2:
        C_fp32 = torch.empty((M, N), dtype=torch.float32, device=A.device)
    config.num_stages = 1
    kernel_gemm_split_k_persistent[(NUM_SMS, )](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                                                C.stride(0), C.stride(1), **config.all_kwargs(),
                                                profiler_buf=profiler_buf, PROFILE=profiler_buf is not None,
                                                SPLITK_ALGO=split_k_algo, SPLIT_K=split_k, C_fp32_ptr=C_fp32)
    return C


@triton_dist.tune.autotune(config_space=[{"config": c} for c in get_config_space()], key_fn=key_fn, prune_fn=prune_fn)
def matmul_triton(A: torch.Tensor, B: torch.Tensor, config: triton.Config = DEFAULT_CONFIG,
                  profiler_buf: torch.Tensor | None = None):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    kernel_gemm[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                      **config.all_kwargs(), profiler_buf=profiler_buf, PROFILE=profiler_buf is not None)
    return C


splitk_workspace = torch.zeros((1024 * 1024 * 16, ), device="cuda", dtype=torch.int32)


@triton_dist.tune.autotune(config_space=[{"config": c} for c in get_splitk_config_space()], key_fn=key_fn_splitk,
                           prune_fn=prune_fn)
def matmul_splitk_triton(A: torch.Tensor, B: torch.Tensor, split_k=4, split_k_algo=2,
                         config: triton.Config = DEFAULT_CONFIG, profiler_buf: torch.Tensor | None = None):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), META["SPLIT_K"])
    kernel_gemm_splitk[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0),
                             C.stride(1), **config.all_kwargs(), profiler_buf=profiler_buf, PROFILE=profiler_buf
                             is not None, SPLIT_K=split_k, SPLITK_ALGO=split_k_algo, splitk_workspace=splitk_workspace)
    return C


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": block_size}, num_warps=num_warps)
        for block_size in [16 * 1024, 32 * 1024, 64 * 1024]
        for num_warps in [4, 8, 16]
    ], key=["N_per_rank"])
@triton_dist.jit
def splitk_reduce_kernel(dst_ptr, src_ptr, N_per_rank, num_ranks: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    num_blocks = tl.cdiv(N_per_rank, BLOCK_SIZE)
    for bid in range(pid, num_blocks, npid):
        offs = tl.arange(0, BLOCK_SIZE) + bid * BLOCK_SIZE
        acc = tl.zeros((BLOCK_SIZE, ), src_ptr.dtype.element_ty)
        mask = offs < N_per_rank
        for n in range(num_ranks):
            src_ptrs = src_ptr + offs + n * N_per_rank
            acc += tl.load(src_ptrs, mask=mask)
        tl.store(dst_ptr + offs, acc.to(dst_ptr.dtype.element_ty), mask=mask, cache_modifier=".cs")


@triton_dist.tune.autotune(config_space=[{"config": c} for c in get_splitk_config_space()], key_fn=key_fn_splitk,
                           prune_fn=prune_fn)
def matmul_splitk_chunked_triton(A: torch.Tensor, B: torch.Tensor, split_k=4, split_k_algo=2,
                                 config: triton.Config = DEFAULT_CONFIG, profiler_buf: torch.Tensor | None = None):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    C_fp32 = None
    if split_k_algo == 2:
        C_fp32 = torch.empty((M, N), dtype=torch.float32, device=A.device)
    elif split_k_algo == 3:
        C_fp32 = torch.empty((split_k, M, N), dtype=torch.float32, device=A.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), META["SPLIT_K"])
    kernel_gemm_splitk_chunked[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0),
                                     C.stride(1), **config.all_kwargs(), profiler_buf=profiler_buf, PROFILE=profiler_buf
                                     is not None, SPLIT_K=split_k, SPLITK_ALGO=split_k_algo,
                                     splitk_workspace_ptr=splitk_workspace, C_fp32_ptr=C_fp32)
    if split_k_algo == 3:
        # C = C_fp32.sum(dim=0).to(C.dtype) # this is too slow. use custom reduce kernel
        splitk_reduce_kernel[(64, )](C, C_fp32, C.numel(), split_k)
    return C


def _pretty_duration(duration_ms):
    if duration_ms < 100 * 1e-3:
        return f"{duration_ms * 1e3:0.2f} us"
    if duration_ms < 10:
        return f"{duration_ms:0.3f} ms"
    return f"{duration_ms:0.2f} ms"


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("-M", default=8192, type=int)
        parser.add_argument("-N", default=1376, type=int)
        parser.add_argument("-K", default=4096, type=int)
        parser.add_argument("--trans_a", "--trans_A", default=False, action="store_true")
        parser.add_argument("--trans_b", "--trans_B", default=False, action="store_true")
        parser.add_argument("--iters", type=int, default=20)
        parser.add_argument("--warmup_iters", type=int, default=10)
        parser.add_argument("--autotune", default=False, action="store_true")
        parser.add_argument("--verbose", "-v", default=False, action="store_true")
        parser.add_argument(
            "--impl", choices=[
                "matmul", "matmul_persistent", "matmul_splitk", "matmul_splitk_chunked", "matmul_splitk_persistent"
            ], default="matmul")
        parser.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="bfloat16")
        parser.add_argument("--profile", default=False, action="store_true")
        parser.add_argument("--split_k", default=2, type=int)
        parser.add_argument(
            "--split_k_algo", default=2, help=
            "0/1/2, 0 for non-deterministic, 1 for deterministic with low-precision accumulation, 2 for FP32 accumulation",
            type=int)
        parser.add_argument
        return parser.parse_args()

    args = parse_args()
    M, N, K = args.M, args.N, args.K
    trans_a, trans_b = args.trans_a, args.trans_b
    dtype = DTYPE_MAP[args.dtype]
    if trans_a:
        A = torch.randn((K, M), dtype=dtype, device="cuda").T
    else:
        A = torch.randn((M, K), dtype=dtype, device="cuda")

    if trans_b:
        B = torch.randn((N, K), dtype=dtype, device="cuda").T
    else:
        B = torch.randn((K, N), dtype=dtype, device="cuda")

    IMPL_MAP = {
        "matmul": matmul_triton, "matmul_persistent": matmul_persistent_triton, "matmul_splitk": matmul_splitk_triton,
        "matmul_splitk_chunked": matmul_splitk_chunked_triton, "matmul_splitk_persistent":
        matmul_split_k_persistent_triton
    }

    def get_triton_runner(profiler_buffer):
        kwargs = {
            "autotune": args.autotune,
            "autotune_verbose": args.verbose,
        }
        if "split" in args.impl:
            kwargs["split_k"] = args.split_k
            kwargs["split_k_algo"] = args.split_k_algo
        return lambda: IMPL_MAP[args.impl](A, B, profiler_buf=profiler_buffer, **kwargs)

    fn_triton = get_triton_runner(profiler_buffer=None)
    fn_torch = lambda: torch.matmul(A, B)

    C_torch = fn_torch()
    if torch.any(C_torch.isnan()):
        print("C has nan")
    if torch.any(C_torch.isinf()):
        print("C has inf")

    C_triton = fn_triton()

    if args.impl.find("splitk") > 0:
        assert_allclose(C_triton, C_torch, atol=2e-1, rtol=5e-2)
    else:
        assert_allclose(C_triton, C_torch, atol=1e-2, rtol=1e-2)

    gflops = 2 * M * N * K / 1e9
    mem_read_in_mb = (dtype.itemsize * M * K + dtype.itemsize * N * K) / 2**20
    mem_write_in_mb = dtype.itemsize * M * N / 2**20
    for n in range(3):  # in case AMD GPU frequency throttle
        if args.profile:
            suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
            trace_file = f"prof/gemm_{suffix}.json.tgz"
        else:
            trace_file = None
        with AutoExportProfiler(trace_file):
            sleep_async(100)
            _, duration_ms_triton = perf_func(fn_triton, iters=args.iters, warmup_iters=args.warmup_iters)
            sleep_async(100)
            _, duration_ms_torch = perf_func(fn_torch, iters=args.iters, warmup_iters=args.warmup_iters)

        tflops_torch = gflops / duration_ms_torch
        mem_read_gbps_torch = mem_read_in_mb / duration_ms_torch
        mem_write_gbps_torch = mem_write_in_mb / duration_ms_torch

        tflops_triton = gflops / duration_ms_triton
        mem_read_gbps_triton = mem_read_in_mb / duration_ms_triton
        mem_write_gbps_triton = mem_write_in_mb / duration_ms_triton
        print(f"iter {n:02}: torch {_pretty_duration(duration_ms_torch)}/iter {tflops_torch:0.1f} TFLOPS  mem read {mem_read_gbps_torch:0.1f} GB/s write {mem_write_gbps_torch:0.1f} GB/s" \
          f"  triton {args.impl} {_pretty_duration(duration_ms_triton)}/iter {tflops_triton:0.1f} TFLOPS mem read {mem_read_gbps_triton:0.1f} GB/s write {mem_write_gbps_triton:0.1f} GB/s")

    # intra kernel profile
    profile_buf = alloc_profiler_buffer(max_num_profile_slots=1000000)
    fn_triton = get_triton_runner(profiler_buffer=profile_buf)
    fn_triton()
    export_to_perfetto_trace(
        profiler_buffer=profile_buf,
        task_names=["gemm"],
        file_name='prof/intra-kernel-amd-gemm.perfetto-trace',
    )
