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

import os
import torch
import triton
import triton.language as tl

from typing import Optional

import itertools

import triton_dist.language as tdl
from triton_dist.utils import nvshmem_create_tensor, nvshmem_free_tensor_sync
from triton_dist.kernels.nvidia.common_ops import barrier_all_intra_node_atomic_cas_block, _wait_eq_cuda
from triton.language.extra.cuda.language_extra import tid, __syncthreads, st


def _kernel_producer_gemm_persistent_repr(proxy):
    constexprs = proxy.constants
    cap_major, cap_minor = torch.cuda.get_device_capability()
    a_dtype = proxy.signature["a_ptr"].lstrip("*")
    b_dtype = proxy.signature["b_ptr"].lstrip("*")
    c_dtype = proxy.signature["c_ptr"].lstrip("*")
    BM, BN, BK = constexprs["BLOCK_SIZE_M"], constexprs["BLOCK_SIZE_N"], constexprs["BLOCK_SIZE_K"]

    return f"triton3x_sm{cap_major}{cap_minor}_gemm_notify_persistent_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_ntn"


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_GEMM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(repr=_kernel_producer_gemm_persistent_repr)
def kernel_persistent_tma_gemm_notify(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    gemm_barrier_ptr,  #
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bn,
    stride_bk,  #
    stride_cm,
    stride_cn,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_GEMM_SMS: tl.constexpr,  #
    HAS_BIAS: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    tiles_per_SM = num_tiles // NUM_GEMM_SMS
    if start_pid < num_tiles % NUM_GEMM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_GEMM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dtype = c_ptr.dtype.element_ty
    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_GEMM_SMS
            pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_GEMM_SMS)
            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)
            if HAS_BIAS:
                offs_bias_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c = c + tl.load(bias_ptr + offs_bias_n, mask=(offs_bias_n < N))[None, :]
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)
            __syncthreads()
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            thread_idx = tid(0)
            gemm_barrier_idx = pid_m * num_pid_n + pid_n
            if thread_idx == 0:
                st(gemm_barrier_ptr + gemm_barrier_idx, 1, scope="gpu", semantic="release")


def persistent_gemm_notify(a, b, bias, out, barrier, gemm_config: triton.Config):

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    if bias is not None:
        assert bias.shape[0] == a.shape[1], "Bias shape does not match input dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    grid = lambda META: (min(META["NUM_GEMM_SMS"],
                             triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )

    kernel_persistent_tma_gemm_notify[grid](
        a, b, bias, out, barrier,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        N, 1,  #
        **gemm_config.all_kwargs(),  #
        HAS_BIAS=1 if bias is not None else 0)
    return out


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_descriptor_persistent(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,  #
    gemm_barrier_ptr,
    M,
    N,
    K,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_GEMM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
    HAS_BIAS: tl.constexpr,
):
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2],
    )

    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue
    tile_id_c = start_pid - NUM_GEMM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for tile_id in tl.range(start_pid, num_tiles, NUM_GEMM_SMS, flatten=False, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_GEMM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_GEMM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_GEMM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if HAS_BIAS:
            offs_bias_n = tl.arange(0, BLOCK_SIZE_N)
            bias_data = tl.load(bias_ptr + offs_cn + offs_bias_n, mask=(offs_cn + offs_bias_n < N)).to(tl.float32)
            accumulator = accumulator + bias_data[None, :]

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            # Optionally, TMA can be used
            # c_desc.store([offs_cm, offs_cn], c)
            c_ptrs = c_ptr + (offs_cm + tl.arange(0, BLOCK_SIZE_M))[:, None] * N + (offs_cn +
                                                                                    tl.arange(0, BLOCK_SIZE_N))[None, :]
            tl.store(c_ptrs, c)

        __syncthreads()
        # Optionally, this can be used for TMA sync
        # tma_sync()
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        thread_idx = tid(0)
        gemm_barrier_idx = pid_m * num_pid_n + pid_n
        if thread_idx == 0:
            st(gemm_barrier_ptr + gemm_barrier_idx, 1, scope="gpu", semantic="release")


def matmul_descriptor_persistent(a, b, bias, c, gemm_barrier, gemm_config: triton.Config,
                                 warp_specialize: bool = False):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape

    # c = torch.empty((M, N), device=a.device, dtype=dtype)
    # NUM_GEMM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(META["NUM_GEMM_SMS"],
                             triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_descriptor_persistent[grid](
        a,
        b,
        bias,
        c,
        gemm_barrier,  #
        M,
        N,
        K,  #
        EPILOGUE_SUBTILE=False,  # have to use False because signal store can't be synced with tma
        WARP_SPECIALIZE=warp_specialize,  #
        **gemm_config.all_kwargs(),  #
        HAS_BIAS=1 if bias is not None else 0,
    )
    return c


@triton.jit(do_not_specialize=["rank", "sp_rank"])
def kernel_all2all_pull_intra_node_nvl(
    gemm_out_ptr,
    gemm_barrier_ptr,
    cum_seqlen_cpu_tuple,  # sp_size + 1 elems
    cum_seqlen_gpu_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    sp_size: tl.constexpr,
    rank,
    sp_rank,
    qkv_out_features: tl.constexpr,
    head_dim: tl.constexpr,
    gqa: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
    HAS_KV: tl.constexpr = 1,
    NEED_BARRIER: tl.constexpr = 1,
):
    tl.static_assert(BLOCK_SIZE_N % head_dim == 0, "BLOCK_SIZE_N must be divisible by head_dim")
    num_heads_per_gemm_tile: tl.constexpr = BLOCK_SIZE_N // head_dim

    for i in tl.static_range(sp_size + 1):
        tl.store(cum_seqlen_gpu_ptr + i, cum_seqlen_cpu_tuple[i])
    __syncthreads()

    if NUM_COMM_SMS >= sp_size:
        tl.static_assert(NUM_COMM_SMS % sp_size == 0, "NUM_COMM_SMS must be divisible by sp_size")
        SMS_per_sp_rank: tl.constexpr = NUM_COMM_SMS // sp_size
        SP_rank_per_sm: tl.constexpr = 1
    else:
        tl.static_assert(sp_size % NUM_COMM_SMS == 0, "sp_size must be divisible by NUM_COMM_SMS")
        SMS_per_sp_rank: tl.constexpr = 1
        SP_rank_per_sm = sp_size // NUM_COMM_SMS

    pid = tl.program_id(axis=0)
    rank_offset = rank - sp_rank

    for tile in range(SP_rank_per_sm):
        remote_sp_rank = pid * SP_rank_per_sm // SMS_per_sp_rank + tile
        remote_rank = rank_offset + remote_sp_rank

        remote_gemm_out_ptr = tdl.symm_at(gemm_out_ptr, remote_rank)
        if NEED_BARRIER:
            remote_gemm_barrier_ptr = tdl.symm_at(gemm_barrier_ptr, remote_rank)

        seq_len_beg = tl.load(cum_seqlen_gpu_ptr + remote_sp_rank)
        seq_len_end = tl.load(cum_seqlen_gpu_ptr + remote_sp_rank + 1)
        seq_len = seq_len_end - seq_len_beg
        num_tiles_m = tl.cdiv(seq_len, BLOCK_SIZE_M)
        tl.static_assert(qkv_out_features % head_dim == 0, "qkv_out_features must be divisible by head_dim")
        num_tiles_n: tl.constexpr = qkv_out_features // head_dim  # number of heads in total
        tl.static_assert(num_tiles_n % sp_size == 0, "num_tiles_n must be divisible by sp_size")
        num_tiles_n_per_sp_rank: tl.constexpr = num_tiles_n // sp_size
        num_tiles_n_for_q: tl.constexpr = num_tiles_n // (gqa + 2 * HAS_KV) * gqa
        num_tiles_n_for_k: tl.constexpr = num_tiles_n // (gqa + 2 * HAS_KV) * HAS_KV
        num_tiles_n_per_sp_rank_for_q: tl.constexpr = num_tiles_n_per_sp_rank // (gqa + 2 * HAS_KV) * gqa
        num_tiles_n_per_sp_rank_for_k: tl.constexpr = num_tiles_n_per_sp_rank // (gqa + 2 * HAS_KV) * HAS_KV

        tl.static_assert(qkv_out_features % BLOCK_SIZE_N == 0, "qkv_out_features must be divisible by BLOCK_SIZE_N")
        num_tiles_gemm_n: tl.constexpr = qkv_out_features // BLOCK_SIZE_N
        num_tiles = num_tiles_m * num_tiles_n_per_sp_rank

        offs_m = tl.arange(0, BLOCK_SIZE_M)
        offs_n = tl.arange(0, head_dim)

        for tile_id in range(pid % SMS_per_sp_rank, num_tiles, SMS_per_sp_rank):
            tile_id_m = tile_id // num_tiles_n_per_sp_rank
            tile_id_n = tile_id % num_tiles_n_per_sp_rank

            a2a_out_ptr = q_out_ptr
            a2a_head_id = 0
            gemm_head_id = 0
            a2a_num_head_per_sp_rank = 0

            if tile_id_n < num_tiles_n_per_sp_rank_for_q:
                a2a_out_ptr = q_out_ptr
                a2a_head_id = tile_id_n
                gemm_head_id = a2a_head_id + sp_rank * num_tiles_n_per_sp_rank_for_q
                a2a_num_head_per_sp_rank = num_tiles_n_per_sp_rank_for_q
            elif tile_id_n < num_tiles_n_per_sp_rank_for_q + num_tiles_n_per_sp_rank_for_k:
                if HAS_KV:
                    a2a_out_ptr = k_out_ptr
                    a2a_head_id = tile_id_n - num_tiles_n_per_sp_rank_for_q
                    gemm_head_id = a2a_head_id + num_tiles_n_for_q + sp_rank * num_tiles_n_per_sp_rank_for_k
                    a2a_num_head_per_sp_rank = num_tiles_n_per_sp_rank_for_k
            else:
                if HAS_KV:
                    a2a_out_ptr = v_out_ptr
                    a2a_head_id = tile_id_n - num_tiles_n_per_sp_rank_for_q - num_tiles_n_per_sp_rank_for_k
                    gemm_head_id = a2a_head_id + num_tiles_n_for_q + num_tiles_n_for_k + sp_rank * num_tiles_n_per_sp_rank_for_k
                    a2a_num_head_per_sp_rank = num_tiles_n_per_sp_rank_for_k

            start_seq = tile_id_m * BLOCK_SIZE_M
            end_seq = min(start_seq + BLOCK_SIZE_M, seq_len)

            gemm_out_ptrs = remote_gemm_out_ptr + (tile_id_m * BLOCK_SIZE_M + offs_m[:, None]
                                                   ) * qkv_out_features + gemm_head_id * head_dim + offs_n[None, :]
            gemm_out_mask = (tile_id_m * BLOCK_SIZE_M + offs_m < end_seq)[:, None] & (offs_n < head_dim)[None, :]
            if NEED_BARRIER:
                tile_id_gemm_n = gemm_head_id // num_heads_per_gemm_tile
                barrier_ptr = remote_gemm_barrier_ptr + tile_id_m * num_tiles_gemm_n + tile_id_gemm_n
                token = tdl.wait(barrier_ptr, 1, "sys", "acquire", 1)
                gemm_out_ptrs = tdl.consume_token(gemm_out_ptrs, token)

            data = tl.load(gemm_out_ptrs, mask=gemm_out_mask)

            a2a_offs_m = seq_len_beg + start_seq + offs_m
            a2a_offs_n = a2a_head_id * head_dim + offs_n
            a2a_out_ptrs = a2a_out_ptr + a2a_offs_m[:, None] * a2a_num_head_per_sp_rank * head_dim + a2a_offs_n[None, :]
            a2a_mask = (a2a_offs_m < seq_len_beg + end_seq)[:, None] & (offs_n < head_dim)[None, :]
            tl.store(a2a_out_ptrs, data, mask=a2a_mask)


class SpUlysessQKVGemmAll2AllKernel:

    def __init__(self, world_group: torch.distributed.ProcessGroup, nnodes: int, sp_size: int, max_batch_size: int,
                 max_seq_len: int, hidden_size: int, head_dim: int,
                 qkv_out_features: int,  # qkv_out_features can be different from 3 * hidden_size
                 input_dtype=torch.bfloat16, output_dtype=torch.bfloat16, gqa: int = 1, max_num_comm_buf: int = 1):
        self.world_group = world_group
        self.nnodes = nnodes
        self.sp_size = sp_size
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.qkv_out_features = qkv_out_features
        assert self.qkv_out_features % self.head_dim == 0, f"qkv_out_features={self.qkv_out_features} must be divisible by head_dim={self.head_dim}"
        self.total_num_heads = self.qkv_out_features // self.head_dim
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.gqa = gqa
        self.max_num_comm_buf = max_num_comm_buf

        assert self.total_num_heads % self.sp_size == 0, f"total_num_heads {self.total_num_heads} must be divisible by sp_size {self.sp_size}"
        self.num_heads_per_sp_rank = self.total_num_heads // self.sp_size
        assert self.num_heads_per_sp_rank % (
            self.gqa +
            2) == 0, f"num_heads_per_sp_rank {self.num_heads_per_sp_rank} must be divisible by (gqa + 2) {self.gqa + 2}"
        self.num_heads_per_sp_rank_for_q = self.num_heads_per_sp_rank // (self.gqa + 2) * self.gqa
        self.num_heads_per_sp_rank_for_k = self.num_heads_per_sp_rank // (self.gqa + 2)
        self.num_heads_per_sp_rank_for_v = self.num_heads_per_sp_rank // (self.gqa + 2)

        self.rank = self.world_group.rank()
        self.world_size = self.world_group.size()
        self.local_world_size = self.world_size // nnodes
        self.local_rank = self.rank % self.local_world_size

        assert self.sp_size <= self.local_world_size, f"sp_size={self.sp_size} exceeds limit {self.local_world_size}"
        assert self.local_world_size % self.sp_size == 0, f"local_world_size={self.local_world_size} is not divisible by sp_size={self.sp_size}"

        # GEMM config
        self.BLOCK_SIZE_M = 128
        self.BLOCK_SIZE_N = 256
        self.BLOCK_SIZE_K = 64
        self.GROUP_SIZE_M = 8
        self.max_gemm_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        self.num_warps = 8
        self.num_stages = 3
        self.warp_specialize = False

        self.init_output_buffer()
        self.init_group_sync_barrier()
        self.init_gemm_barriers()
        self.init_local_tensors()

        self._ready_event = torch.cuda.Event(enable_timing=False)
        self._comm_event = torch.cuda.Event(enable_timing=False)
        self._a2a_stream = torch.cuda.Stream()

    def finalize(self):
        self.deinit_output_buffer()
        self.deinit_group_sync_barrier()
        self.deinit_gemm_barriers()

    def __del__(self):
        self.finalize()

    def init_output_buffer(self):
        max_local_seq = self.max_seq_len // self.sp_size
        max_gemm_m = self.max_batch_size * max_local_seq
        gemm_n = self.qkv_out_features
        self._gemm_output_buffer = nvshmem_create_tensor([self.max_num_comm_buf, max_gemm_m, gemm_n], self.output_dtype)

    def deinit_output_buffer(self):
        if hasattr(self, "_gemm_output_buffer"):
            nvshmem_free_tensor_sync(self._gemm_output_buffer)
            del self._gemm_output_buffer

    def init_group_sync_barrier(self):
        self._sp_group_sync_buffer = nvshmem_create_tensor([self.world_size], torch.int32)
        self._sp_group_sync_buffer.zero_()

    def deinit_group_sync_barrier(self):
        if hasattr(self, "_sp_group_sync_buffer"):
            nvshmem_free_tensor_sync(self._sp_group_sync_buffer)
            del self._sp_group_sync_buffer

    def init_gemm_barriers(self):
        max_m = self.max_batch_size * (self.max_seq_len // self.sp_size)
        max_n = self.qkv_out_features
        max_num_tile_m = triton.cdiv(max_m, self.BLOCK_SIZE_M)
        max_num_tile_n = triton.cdiv(max_n, self.BLOCK_SIZE_N)
        self._gemm_barrier_buffer = nvshmem_create_tensor([max_num_tile_m * max_num_tile_n], dtype=torch.int32)
        self._gemm_barrier_buffer.zero_()

    def deinit_gemm_barriers(self):
        if hasattr(self, "_gemm_barrier_buffer"):
            nvshmem_free_tensor_sync(self._gemm_barrier_buffer)
            del self._gemm_barrier_buffer

    def init_local_tensors(self):
        self._cum_seq_len_gpu = torch.empty([self.sp_size + 1], dtype=torch.int32, device="cuda")
        self._cum_seq_len_cpu_tuple = None

    def get_comm_buf(self, comm_buf_idx: int):
        assert comm_buf_idx < self.max_num_comm_buf, f"comm_buf_idx={comm_buf_idx} out of range ({self.max_num_comm_buf})"
        max_local_seq_len = self.max_seq_len // self.sp_size
        buf_size = max_local_seq_len * self.max_batch_size * self.qkv_out_features
        return self._gemm_output_buffer.view([self.max_num_comm_buf, buf_size])[comm_buf_idx]

    def check_input(self, tensor):
        assert len(
            tensor.shape) == 4, f"input tensor shape should be [bs, local_seq_len, nh, hd], but got {tensor.shape}"
        assert tensor.is_contiguous(), f"input tensor should be contiguous, but got {tensor.is_contiguous()}"
        assert tensor.shape[
            0] <= self.max_batch_size, f"batch size {tensor.shape[0]} exceeds limit {self.max_batch_size}"
        assert tensor.shape[
            1] <= self.max_seq_len // self.sp_size, f"local seq len {tensor.shape[1]} exceeds limit {self.max_seq_len // self.sp_size}"
        assert tensor.shape[
            2] <= self.qkv_out_features // self.head_dim, f"hidden size {tensor.shape[2]} exceeds limit {self.qkv_out_features // self.head_dim}"
        assert tensor.shape[3] == self.head_dim, f"head dim {tensor.shape[3]} != {self.head_dim}"
        assert tensor.dtype == self.input_dtype

    def get_input_comm_buf(self, tensor, comm_buf_idx):
        nelems = tensor.numel()
        full_comm_buf = self.get_comm_buf(comm_buf_idx=comm_buf_idx)
        return full_comm_buf.view(tensor.dtype).view(-1)[:nelems].view(tensor.shape)

    def sp_group_barrier_all_intra_node(self, stream=None):
        stream = torch.cuda.current_stream() if stream is None else stream
        sp_local_rank = self.local_rank % self.sp_size
        with torch.cuda.stream(stream):
            barrier_all_intra_node_atomic_cas_block[(1, )](sp_local_rank, self.rank, self.sp_size,
                                                           self._sp_group_sync_buffer)

    def reset_cusum_seq_lens(self, local_seqlen, seq_lens_cpu=None):
        if seq_lens_cpu is None:
            seq_lens_cpu = [local_seqlen] * self.sp_size
        else:
            seq_lens_cpu = seq_lens_cpu.tolist()
        assert local_seqlen == seq_lens_cpu[
            self.local_rank % self.
            sp_size], f"local_seqlen {local_seqlen} != seq_lens_cpu[{self.local_rank % self.sp_size}]={seq_lens_cpu[self.local_rank % self.sp_size]}"
        cum_seqlen_cpu = [0] + list(itertools.accumulate(seq_lens_cpu))
        self._cum_seq_len_cpu_tuple = tuple(cum_seqlen_cpu)

    def forward(
        self,
        attention_input,
        weight,
        seq_lens_cpu=None,
        bias=None,
        outputs=None,
        num_comm_sms=-1,
        sm_margin=0,
    ):
        self.sp_group_barrier_all_intra_node()
        assert len(
            attention_input.shape
        ) == 3, f"attention_input shape should be [bs, local_seq_len, hidden], but got {attention_input.shape}"
        assert len(weight.shape) == 2, f"weight shape should be [qkv_out_features, hidden], but got {weight.shape}"
        assert attention_input.is_contiguous(
        ), f"attention_input should be contiguous, but got {attention_input.is_contiguous()}"
        assert weight.is_contiguous(), f"weight should be contiguous, but got {weight.is_contiguous()}"
        assert attention_input.shape[
            0] <= self.max_batch_size, f"batch size {attention_input.shape[0]} exceeds limit {self.max_batch_size}"
        assert attention_input.shape[
            2] == self.hidden_size, f"hidden size {attention_input.shape[2]} != {self.hidden_size}"
        assert attention_input.shape[
            1] <= self.max_seq_len // self.sp_size, f"local seq len {attention_input.shape[1]} exceeds limit {self.max_seq_len // self.sp_size}"
        assert weight.shape[
            0] == self.qkv_out_features, f"qkv_out_features {weight.shape[0]} != {self.qkv_out_features}"
        assert weight.shape[1] == self.hidden_size, f"hidden size {weight.shape[1]} != {self.hidden_size}"

        local_seq_len = attention_input.shape[1]
        self.reset_cusum_seq_lens(local_seqlen=local_seq_len, seq_lens_cpu=seq_lens_cpu)

        if num_comm_sms < 0:
            num_comm_sms = self.world_size
        assert num_comm_sms < self.max_gemm_sms, f"num_comm_sms {num_comm_sms} exceeds max_gemm_sms {self.max_gemm_sms}"

        gemm_config = triton.Config(
            {
                'BLOCK_SIZE_M': self.BLOCK_SIZE_M, 'BLOCK_SIZE_N': self.BLOCK_SIZE_N, 'BLOCK_SIZE_K': self.BLOCK_SIZE_K,
                'GROUP_SIZE_M': self.GROUP_SIZE_M, 'NUM_GEMM_SMS': self.max_gemm_sms - num_comm_sms - sm_margin
            }, num_stages=self.num_stages, num_warps=self.num_warps)

        self._gemm_barrier_buffer.zero_()

        cur_stream = torch.cuda.current_stream()
        self.sp_group_barrier_all_intra_node(cur_stream)
        self._ready_event.record(cur_stream)
        bs, local_seq_len, hidden = attention_input.shape
        # persistent_gemm_notify(
        #   attention_input.view(bs * local_seq_len, hidden),
        #   weight,
        #   bias,
        #   self._gemm_output_buffer,
        #   self._gemm_barrier_buffer,
        #   gemm_config
        # )
        matmul_descriptor_persistent(attention_input.view(bs * local_seq_len,
                                                          hidden), weight, bias, self._gemm_output_buffer,
                                     self._gemm_barrier_buffer, gemm_config, warp_specialize=self.warp_specialize)
        self._a2a_stream.wait_event(self._ready_event)

        if os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", -1) != 1:
            _wait_eq_cuda(self._gemm_barrier_buffer, 1, self._a2a_stream)

        total_seq_len = self._cum_seq_len_cpu_tuple[self.sp_size]

        if outputs is not None:
            assert isinstance(outputs, (list, tuple)), f"outputs should be a list or tuple, but got {type(outputs)}"
            output_tensors = outputs
        else:
            output_tensors = [
                torch.empty([bs, total_seq_len, self.num_heads_per_sp_rank_for_q, self.head_dim],
                            dtype=self.output_dtype, device="cuda"),
                torch.empty([bs, total_seq_len, self.num_heads_per_sp_rank_for_k, self.head_dim],
                            dtype=self.output_dtype, device="cuda"),
                torch.empty([bs, total_seq_len, self.num_heads_per_sp_rank_for_v, self.head_dim],
                            dtype=self.output_dtype, device="cuda"),
            ]
        assert len(output_tensors) == 3, f"outputs should have 3 tensors, but got {len(output_tensors)}"

        with torch.cuda.stream(self._a2a_stream):
            grid = (num_comm_sms, )
            kernel_all2all_pull_intra_node_nvl[grid](
                self._gemm_output_buffer,
                self._gemm_barrier_buffer,
                self._cum_seq_len_cpu_tuple,
                self._cum_seq_len_gpu,
                output_tensors[0],
                output_tensors[1],
                output_tensors[2],
                self.sp_size,
                self.rank,
                self.local_rank % self.sp_size,
                self.qkv_out_features,
                self.head_dim,
                self.gqa,
                self.BLOCK_SIZE_M,
                self.BLOCK_SIZE_N,
                num_comm_sms,
                HAS_KV=1,
                NEED_BARRIER=1,
                num_warps=32,
            )

        self._comm_event.record(self._a2a_stream)
        cur_stream.wait_event(self._comm_event)

        return output_tensors

    def pre_attn_a2a(
        self,
        inputs: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        num_comm_sms: int = -1,
        is_input_in_comm_buf: bool = False,
        comm_buf_idx: int = 0,
    ):
        stream = torch.cuda.current_stream()
        self.check_input(inputs)
        self.reset_cusum_seq_lens(inputs.shape[1], seq_lens_cpu)
        if num_comm_sms == -1:
            num_comm_sms = self.world_size
        assert num_comm_sms > 0, f"num_comm_sms {num_comm_sms} must be greater than 0"

        # copy input to symm buf
        if not is_input_in_comm_buf:
            assert comm_buf_idx == 0, f"comm_buf_idx {comm_buf_idx} must be 0 when input is not in comm buf"
            self.sp_group_barrier_all_intra_node(stream)
            self._gemm_output_buffer.view(-1)[:inputs.numel()].copy_(inputs.view(-1))
            self.sp_group_barrier_all_intra_node(stream)

        bs, _, heads, head_dim = inputs.shape
        n = heads * head_dim

        seq_len = self._cum_seq_len_cpu_tuple[self.sp_size]
        local_heads = heads // self.sp_size

        assert comm_buf_idx < self.max_num_comm_buf, f"comm_buf_idx {comm_buf_idx} exceeds max_num_comm_buf {self.max_num_comm_buf}"
        # a2a
        out_q = torch.empty([bs, seq_len, local_heads, head_dim], dtype=inputs.dtype, device=inputs.device)
        grid = (num_comm_sms, )
        kernel_all2all_pull_intra_node_nvl[grid](
            self._gemm_output_buffer[comm_buf_idx],
            None,  # barrier
            self._cum_seq_len_cpu_tuple,
            self._cum_seq_len_gpu,
            out_q,
            None,  # k_out
            None,  # v_out
            self.sp_size,
            self.rank,
            self.local_rank % self.sp_size,
            n,  # qkv_out_features
            self.head_dim,
            1,  # gqa
            self.BLOCK_SIZE_M,
            self.BLOCK_SIZE_N,
            num_comm_sms,
            HAS_KV=0,
            NEED_BARRIER=0,
            num_warps=32,
        )

        return out_q

    def pre_attn_a2a_no_cpy(
        self,
        inputs: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        num_comm_sms: int = -1,
        comm_buf_idx: int = 0,
    ):
        return self.pre_attn_a2a(
            inputs,
            seq_lens_cpu,
            num_comm_sms,
            is_input_in_comm_buf=True,
            comm_buf_idx=comm_buf_idx,
        )

    def qkv_pack_a2a(
        self,
        qkv,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        num_comm_sms: int = -1,
        is_input_in_comm_buf: bool = False,
        comm_buf_idx: int = 0,
    ):
        stream = torch.cuda.current_stream()
        self.check_input(qkv)
        self.reset_cusum_seq_lens(qkv.shape[1], seq_lens_cpu)
        if num_comm_sms == -1:
            num_comm_sms = self.world_size
        assert num_comm_sms > 0, f"num_comm_sms {num_comm_sms} must be greater than 0"

        # copy input to symm buf
        if not is_input_in_comm_buf:
            assert comm_buf_idx == 0, f"comm_buf_idx {comm_buf_idx} must be 0 when input is not in comm buf"
            self.sp_group_barrier_all_intra_node(stream)
            self._gemm_output_buffer.view(-1)[:qkv.numel()].copy_(qkv.view(-1))
            self.sp_group_barrier_all_intra_node(stream)
        bs, _, heads, head_dim = qkv.shape
        n = heads * head_dim

        seq_len = self._cum_seq_len_cpu_tuple[self.sp_size]
        local_nheads = heads // self.sp_size
        assert local_nheads % (self.gqa +
                               2) == 0, f"local_nheads {local_nheads} must be divisible by (gqa + 2) {self.gqa + 2}"
        local_q_nheads = local_nheads // (self.gqa + 2) * self.gqa
        local_k_nheads = local_nheads // (self.gqa + 2)
        local_v_nheads = local_k_nheads
        assert comm_buf_idx < self.max_num_comm_buf, f"comm_buf_idx {comm_buf_idx} exceeds max_num_comm_buf {self.max_num_comm_buf}"
        # a2a
        out_q = torch.empty([bs, seq_len, local_q_nheads, head_dim], dtype=qkv.dtype, device=qkv.device)
        out_k = torch.empty([bs, seq_len, local_k_nheads, head_dim], dtype=qkv.dtype, device=qkv.device)
        out_v = torch.empty([bs, seq_len, local_v_nheads, head_dim], dtype=qkv.dtype, device=qkv.device)
        grid = (num_comm_sms, )
        kernel_all2all_pull_intra_node_nvl[grid](
            self._gemm_output_buffer[comm_buf_idx],
            None,  # barrier
            self._cum_seq_len_cpu_tuple,
            self._cum_seq_len_gpu,
            out_q,
            out_k,  # k_out
            out_v,  # v_out
            self.sp_size,
            self.rank,
            self.local_rank % self.sp_size,
            n,  # qkv_out_features
            self.head_dim,
            self.gqa,
            self.BLOCK_SIZE_M,
            self.BLOCK_SIZE_N,
            num_comm_sms,
            HAS_KV=1,
            NEED_BARRIER=0,
            num_warps=32,
        )

        return [out_q, out_k, out_v]

    def qkv_pack_a2a_no_cpy(
        self,
        qkv,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        num_comm_sms: int = -1,
        comm_buf_idx: int = 0,
    ):
        return self.qkv_pack_a2a(
            qkv,
            seq_lens_cpu,
            num_comm_sms,
            is_input_in_comm_buf=True,
            comm_buf_idx=comm_buf_idx,
        )
