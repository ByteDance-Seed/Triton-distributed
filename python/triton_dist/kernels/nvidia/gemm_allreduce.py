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
import torch
import dataclasses
from typing import List
import triton
import triton.language as tl
import triton_dist
from triton_dist.kernels.allreduce import OverlappingAllReduceMethod, get_auto_all_reduce_method
import triton_dist.language as dl
from triton_dist.utils import (nvshmem_barrier_all_on_stream, nvshmem_create_tensor, nvshmem_free_tensor_sync,
                               launch_cooperative_grid_options)
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cuda.utils import num_warps
from triton_dist.language.extra.cuda.language_extra import (__syncthreads, ld, st, tid, multimem_ld_reduce_v4,
                                                            multimem_st_v4, st_v4_b32, atomic_add)
from triton_dist.kernels.nvidia.common_ops import barrier_on_this_grid
from triton_dist.utils import is_nvshmem_multimem_supported


@dataclasses.dataclass
class GemmARContext:
    rank: int
    num_ranks: int

    symm_gemm_out_buf: torch.Tensor
    symm_ar_out_buf: torch.Tensor

    gemm_barrier_buf: torch.Tensor
    multi_st_barrier_buf: torch.Tensor
    grid_barrier_buf: torch.Tensor
    tile_barrier_buf: torch.Tensor

    NUM_COMM_SMS: int
    ar_stream: torch.cuda.Stream
    TILE_MAP_LEVEL: int = 0
    all_reduce_method: OverlappingAllReduceMethod = OverlappingAllReduceMethod.Auto

    def finalize(self):
        nvshmem_free_tensor_sync(self.symm_gemm_out_buf)
        nvshmem_free_tensor_sync(self.symm_ar_out_buf)
        nvshmem_free_tensor_sync(self.gemm_barrier_buf)
        nvshmem_free_tensor_sync(self.multi_st_barrier_buf)

    def get_gemm_out_buf(self, input, weight):
        M, N = input.shape[0], weight.shape[0]
        assert self.symm_gemm_out_buf.numel() >= M * N
        return self.symm_gemm_out_buf.reshape(-1)[:M * N].reshape(M, N)

    def reset_all_barrier_buf(self):
        self.gemm_barrier_buf.zero_()
        self.tile_barrier_buf.zero_()
        self.grid_barrier_buf.zero_()
        self.multi_st_barrier_buf.zero_()


@dataclasses.dataclass
class LLGemmARContext:

    ctxs: List[GemmARContext]
    num_phases: int
    phase: int

    def __post_init__(self):
        assert len(self.ctxs) == self.num_phases

    def update_phase(self):
        self.phase = (self.phase + 1) % self.num_phases

    def __getattr__(self, name):
        return getattr(self.ctxs[self.phase], name)

    def finalize(self):
        for ctx in self.ctxs:
            ctx.finalize()

    def reset_all_barrier_buf(self):
        self.ctxs[self.phase].reset_all_barrier_buf()


def create_gemm_ar_context(ar_stream: torch.cuda.Stream, rank, world_size, local_world_size, max_M, N, dtype,
                           MIN_BLOCK_SIZE_M=16, MIN_BLOCK_SIZE_N=16, NUM_COMM_SMS=16, TILE_MAP_LEVEL=0):
    assert local_world_size == world_size
    gemm_out_buf = nvshmem_create_tensor((world_size, max_M, N), dtype)
    symm_ar_out_buf = nvshmem_create_tensor((max_M, N), dtype)
    gemm_barrier_buf = nvshmem_create_tensor(
        (world_size, triton.cdiv(max_M, MIN_BLOCK_SIZE_M), triton.cdiv(N, MIN_BLOCK_SIZE_N)), torch.int32)
    multi_st_barrier_buf = nvshmem_create_tensor((world_size * NUM_COMM_SMS, ), torch.int32)
    grid_barrier_buf = torch.zeros((1, ), dtype=torch.int32, device=torch.cuda.current_device())
    tile_barrier_buf = torch.zeros((world_size, triton.cdiv(max_M, MIN_BLOCK_SIZE_M), triton.cdiv(N, MIN_BLOCK_SIZE_N)),
                                   dtype=torch.int32, device=torch.cuda.current_device())
    gemm_out_buf.zero_()
    gemm_barrier_buf.zero_()
    multi_st_barrier_buf.zero_()

    all_reduce_method = get_auto_all_reduce_method(world_size, local_world_size)
    nvshmem_barrier_all_on_stream()
    return GemmARContext(rank=rank, num_ranks=world_size, symm_gemm_out_buf=gemm_out_buf,
                         symm_ar_out_buf=symm_ar_out_buf, gemm_barrier_buf=gemm_barrier_buf,
                         multi_st_barrier_buf=multi_st_barrier_buf, grid_barrier_buf=grid_barrier_buf,
                         tile_barrier_buf=tile_barrier_buf, NUM_COMM_SMS=NUM_COMM_SMS, ar_stream=ar_stream,
                         TILE_MAP_LEVEL=TILE_MAP_LEVEL, all_reduce_method=all_reduce_method)


def create_ll_gemm_ar_context(rank, world_size, local_world_size, max_M, N, dtype, MIN_BLOCK_SIZE_M=16,
                              MIN_BLOCK_SIZE_N=16, NUM_COMM_SMS=132, num_phases=2):
    ar_stream = torch.cuda.Stream(priority=-1)
    ctxs = []
    for i in range(num_phases):
        ctxs.append(
            create_gemm_ar_context(ar_stream, rank, world_size, local_world_size, max_M, N, dtype,
                                   NUM_COMM_SMS=NUM_COMM_SMS))
    nvshmem_barrier_all_on_stream()
    return LLGemmARContext(ctxs=ctxs, num_phases=num_phases, phase=0)


@triton_dist.jit(do_not_specialize=[])
def consumer_all_reduce_kernel(
    symm_input_ptr,
    symm_ar_out_ptr,
    ar_out_ptr,  #
    gemm_barrier_ptr,
    multi_st_barrier_ptr,  #
    M,
    N,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    NUM_COMM_SMS: tl.constexpr,
    USE_MULTIMEM_ST: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    thread_idx = tid(0)
    block_dim = num_warps() * 32
    VEC_SIZE: tl.constexpr = 128 // tl.constexpr(symm_input_ptr.dtype.element_ty.primitive_bitwidth)
    # TODO(zhengxuegui.0): non perfect N support
    tl.static_assert(BLOCK_SIZE_N % VEC_SIZE == 0)
    VEC_PER_ROW = BLOCK_SIZE_N // VEC_SIZE
    src_data_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_input_ptr)
    if not USE_MULTIMEM_ST:
        for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            if thread_idx < world_size:
                peer_gemm_barrier_ptr = dl.symm_at(gemm_barrier_ptr, thread_idx)
                while ld(peer_gemm_barrier_ptr + tile_id, scope="sys", semantic="acquire") != 1:
                    pass
            __syncthreads()
            tile_m = min(M - pid_m * BLOCK_SIZE_M, BLOCK_SIZE_M)
            cur_tile_nelem = tile_m * BLOCK_SIZE_N
            for idx in range(thread_idx, cur_tile_nelem // VEC_SIZE, block_dim):
                row_id = idx // VEC_PER_ROW
                col_id = idx % VEC_PER_ROW
                offset = (row_id + pid_m * BLOCK_SIZE_M) * N + col_id * VEC_SIZE + pid_n * BLOCK_SIZE_N
                val0, val1, val2, val3 = multimem_ld_reduce_v4(src_data_mc_ptr + offset)
                st_v4_b32(ar_out_ptr + offset, val0, val1, val2, val3)
    else:
        symm_out_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_ar_out_ptr)
        for tile_id in range(pid + rank * NUM_COMM_SMS, num_tiles, NUM_COMM_SMS * world_size):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            if thread_idx < world_size:
                peer_gemm_barrier_ptr = dl.symm_at(gemm_barrier_ptr, thread_idx)
                while ld(peer_gemm_barrier_ptr + tile_id, scope="sys", semantic="acquire") != 1:
                    pass
            __syncthreads()

            tile_m = min(M - pid_m * BLOCK_SIZE_M, BLOCK_SIZE_M)
            cur_tile_nelem = tile_m * BLOCK_SIZE_N
            for idx in range(thread_idx, cur_tile_nelem // VEC_SIZE, block_dim):
                row_id = idx // VEC_PER_ROW
                col_id = idx % VEC_PER_ROW
                offset = (row_id + pid_m * BLOCK_SIZE_M) * N + col_id * VEC_SIZE + pid_n * BLOCK_SIZE_N
                val0, val1, val2, val3 = multimem_ld_reduce_v4(src_data_mc_ptr + offset)
                multimem_st_v4(symm_out_mc_ptr + offset, val0, val1, val2, val3)
        __syncthreads()

        # barrier on all blocks with same pid
        # 0. set barrier to all blocks with same pid on all peer ranks
        if thread_idx < world_size:
            peer_ptr = dl.symm_at(multi_st_barrier_ptr, thread_idx)
            st(peer_ptr + rank * NUM_COMM_SMS + pid, 1, scope="sys", semantic="release")

        # 1. wait barrier
        if thread_idx < world_size:
            multi_st_barrier_idx = thread_idx * NUM_COMM_SMS + pid
            while ld(multi_st_barrier_ptr + multi_st_barrier_idx, scope="sys", semantic="acquire") != 1:
                pass
            st(multi_st_barrier_ptr + multi_st_barrier_idx, 0)

    # Each block can safely reset the part of the barriers it is waiting for.
    # In low latency kernel, we ensure that the gemm barriers used for two consecutive iteration are different,
    # it can be reset without any sync.
    for tile_id in range(pid + rank * NUM_COMM_SMS, num_tiles, NUM_COMM_SMS * world_size):
        peer_gemm_barrier_ptr = dl.symm_at(gemm_barrier_ptr, thread_idx)
        if thread_idx < world_size:
            st(peer_gemm_barrier_ptr + tile_id, 0, scope="sys", semantic="relaxed")


@triton_dist.jit(do_not_specialize=[])
def consumer_all_reduce_load_store_kernel(symm_input_ptr, symm_ar_out_ptr, ar_out_ptr,  #
                                          gemm_barrier_ptr, M, N: tl.constexpr,  #
                                          BLOCK_SIZE_M: tl.constexpr,  #
                                          BLOCK_SIZE_N: tl.constexpr,  #
                                          BLOCK_SIZE_COMM: tl.constexpr, NUM_COMM_SMS: tl.constexpr,
                                          TILE_MAP_LEVEL: tl.constexpr = 0):
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    thread_idx = tid(0)
    elem_per_pid_m = BLOCK_SIZE_M * N
    if TILE_MAP_LEVEL == 0:
        num_tiles = num_pid_m * num_pid_n
    elif TILE_MAP_LEVEL == 1:
        num_tiles = tl.cdiv(M * N, BLOCK_SIZE_COMM)

    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        if thread_idx < world_size:
            peer_gemm_barrier_ptr = dl.symm_at(gemm_barrier_ptr, thread_idx)
            if TILE_MAP_LEVEL == 0:
                barrier_offset = tile_id
            else:
                barrier_offset = tile_id * BLOCK_SIZE_COMM // elem_per_pid_m
            while ld(peer_gemm_barrier_ptr + barrier_offset, scope="gpu", semantic="acquire"
                     ) != 1:  # WARN(fangjin.018): sys is slow in L20, use gpu scope here, pass the acc test in sglang
                pass
        __syncthreads()

        if TILE_MAP_LEVEL == 0:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            column_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask = column_offset < N
            tile_m = min(M - pid_m * BLOCK_SIZE_M, BLOCK_SIZE_M)
            for row in range(tile_m):
                row_offset = (pid_m * BLOCK_SIZE_M + row) * N
                accum = tl.zeros((BLOCK_SIZE_N, ), dtype=tl.float32)
                for i in range(0, world_size):
                    segment = (rank + world_size - i) % world_size
                    peer_ptr = dl.symm_at(symm_input_ptr, segment)
                    data = tl.load(peer_ptr + row_offset + column_offset, mask=mask, other=0)
                    accum += data
                c = accum.to(tl.bfloat16)
                tl.store(ar_out_ptr + row_offset + column_offset, c, mask=mask)
        else:
            pid_m = tile_id * BLOCK_SIZE_COMM // elem_per_pid_m
            pid_n = tile_id * BLOCK_SIZE_COMM % elem_per_pid_m // BLOCK_SIZE_COMM
            tile_m = min(M - pid_m * BLOCK_SIZE_M, BLOCK_SIZE_M)
            row_offset = pid_m * BLOCK_SIZE_M * N
            column_offset = pid_n * BLOCK_SIZE_COMM + tl.arange(0, BLOCK_SIZE_COMM)
            mask = column_offset < tile_m * N
            accum = tl.zeros((BLOCK_SIZE_COMM, ), dtype=tl.float32)
            for i in range(0, world_size):
                segment = (rank + world_size - i) % world_size
                peer_ptr = dl.symm_at(symm_input_ptr, segment)
                data = tl.load(peer_ptr + row_offset + column_offset, mask=mask, other=0)
                accum += data
            c = accum.to(tl.bfloat16)
            tl.store(ar_out_ptr + row_offset + column_offset, c, mask=mask)


@triton_dist.jit(do_not_specialize=[])
def consumer_ring_all_reduce_load_store_kernel(symm_input_ptr, symm_ar_out_ptr, ar_out_ptr,  #
                                               gemm_barrier_ptr, M, N: tl.constexpr,  #
                                               BLOCK_SIZE_M: tl.constexpr,  #
                                               BLOCK_SIZE_N: tl.constexpr,  #
                                               BLOCK_SIZE_COMM: tl.constexpr, NUM_COMM_SMS: tl.constexpr,
                                               TILE_MAP_LEVEL: tl.constexpr = 0):
    rank = dl.rank()
    world_size = dl.num_ranks()
    numel = M * N
    num_tiles = tl.cdiv(numel, BLOCK_SIZE_COMM)
    thread_idx = tid(0)
    segment = rank
    elem_per_rank = numel // world_size
    pid = tl.program_id(0)

    if thread_idx == 0:
        while ld(gemm_barrier_ptr + segment, scope="sys", semantic="acquire") != 1:
            pass
    __syncthreads()

    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        offs = tile_id * BLOCK_SIZE_COMM + tl.arange(0, BLOCK_SIZE_COMM)
        mask = offs < elem_per_rank
        data = tl.load(symm_input_ptr + segment * elem_per_rank + offs, mask=mask)
        tl.store(symm_ar_out_ptr + segment * elem_per_rank + offs, data, mask=mask)
    __syncthreads()

    for i in range(1, world_size):
        to_rank = (rank + world_size - i) % world_size
        if thread_idx == 0:
            peer_gemm_barrier_ptr = dl.symm_at(gemm_barrier_ptr, to_rank)
            while ld(peer_gemm_barrier_ptr + segment, scope="sys", semantic="acquire") != 1:
                pass

        for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
            offs = tile_id * BLOCK_SIZE_COMM + tl.arange(0, BLOCK_SIZE_COMM)
            mask = offs < elem_per_rank
            peer_ptr = dl.symm_at(symm_input_ptr, to_rank)
            src_ptr = peer_ptr + segment * elem_per_rank + offs
            dst_ptr = symm_ar_out_ptr + segment * elem_per_rank + offs
            accum = tl.zeros((BLOCK_SIZE_COMM, ), dtype=tl.float32)
            accum += tl.load(src_ptr, mask=mask)
            accum += tl.load(dst_ptr, mask=mask)
            c = accum.to(tl.bfloat16)
            tl.store(dst_ptr, c, mask=mask)
        __syncthreads()

    for i in range(1, world_size):
        to_rank = (rank + world_size - i) % world_size
        peer_ptr = dl.symm_at(symm_ar_out_ptr, to_rank)
        for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
            offs = tile_id * BLOCK_SIZE_COMM + tl.arange(0, BLOCK_SIZE_COMM)
            mask = offs < elem_per_rank
            src_ptr = symm_ar_out_ptr + segment * elem_per_rank + offs
            dst_ptr = peer_ptr + segment * elem_per_rank + offs
            data = tl.load(src_ptr, mask=mask)
            tl.store(dst_ptr, data, mask=mask)


@triton_dist.jit(do_not_specialize=["nelems"])
def copy_1d_tilewise_kernel(src_ptr, dst_ptr,  #
                            nelems,  #
                            BLOCK_SIZE: tl.constexpr,  #
                            ):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles = nelems // BLOCK_SIZE

    for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        data = tl.load(src_ptr + offs)
        tl.store(dst_ptr + offs, data)

    if nelems % BLOCK_SIZE:
        if pid == NUM_COPY_SMS - 1:
            offs = num_tiles * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < nelems
            data = tl.load(src_ptr + offs, mask=mask)
            tl.store(dst_ptr + offs, data, mask=mask)


@triton_dist.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton_dist.jit(do_not_specialize=[])
def kernel_persistent_gemm_notify(a_ptr, b_ptr, c_ptr, gemm_barrier_ptr, tile_barrier_ptr,  #
                                  M, N, K,  #
                                  stride_am, stride_ak,  #
                                  stride_bn, stride_bk,  #
                                  stride_cm, stride_cn,  #
                                  BLOCK_SIZE_M: tl.constexpr,  #
                                  BLOCK_SIZE_N: tl.constexpr,  #
                                  BLOCK_SIZE_K: tl.constexpr,  #
                                  GROUP_SIZE_M: tl.constexpr,  #
                                  NUM_GEMM_SMS: tl.constexpr,  #
                                  As_ptr=None, Bs_ptr=None, USE_INT8: tl.constexpr = False,
                                  TILE_MAP_LEVEL: tl.constexpr = 0, FUSE_START_OFFSET: tl.constexpr = 0):
    if USE_INT8:
        tl.device_assert(As_ptr is not None)
        tl.device_assert(Bs_ptr is not None)

    start_pid = tl.program_id(axis=0) - FUSE_START_OFFSET
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_GEMM_SMS):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_GEMM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        if USE_INT8:
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
        else:
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
            if USE_INT8:
                a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0).to(tl.int8)
                b = tl.load(b_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0).to(tl.int8)
            else:
                a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator += tl.dot(a, b.T)

        if USE_INT8:
            # apply scale
            offs_scale_am = start_m + tl.arange(0, BLOCK_SIZE_M)
            offs_scale_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
            a_scale = tl.load(As_ptr + offs_scale_am, mask=offs_scale_am < M, other=1.0).to(tl.float32)
            b_scale = tl.load(Bs_ptr + offs_scale_bn, mask=offs_scale_bn < N, other=1.0).to(tl.float32)
            accumulator_fp = accumulator.to(tl.float32)
            c = (accumulator_fp * a_scale[:, None] * b_scale[None, :]).to(tl.bfloat16)
        else:
            c = accumulator.to(c_ptr.dtype.element_ty)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

        thread_idx = tid(0)
        if thread_idx == 0:
            if TILE_MAP_LEVEL == 0:  # tile_wise_map_to_comm
                gemm_barrier_idx = pid_m * num_pid_n + pid_n
                st(gemm_barrier_ptr + gemm_barrier_idx, 1, scope="gpu", semantic="release")
            elif TILE_MAP_LEVEL == 1:  # row_wise_map_to_comm
                count = atomic_add(tile_barrier_ptr + pid_m, 1, scope="gpu", semantic="release")
                if count == num_pid_n - 1:
                    st(gemm_barrier_ptr + pid_m, 1, scope="gpu", semantic="release")
            elif TILE_MAP_LEVEL == 2:  # rank_wise_map_to_comm
                world_size = dl.num_ranks()
                M_per_rank = M // world_size
                tile_m = pid_m * BLOCK_SIZE_M
                barrier_start = tile_m // M_per_rank
                barrier_end = (tile_m + BLOCK_SIZE_M - 1) // M_per_rank
                barrier_end = min(barrier_end, world_size - 1)
                for barrier_id in range(barrier_start, barrier_end + 1):
                    m_start = M_per_rank * barrier_id
                    m_end = M_per_rank * (barrier_id + 1) - 1
                    tiled_m_start = m_start // BLOCK_SIZE_M
                    tiled_m_end = m_end // BLOCK_SIZE_M
                    tiled_m_size = tiled_m_end - tiled_m_start + 1
                    val = atomic_add(tile_barrier_ptr + barrier_id, 1, scope="gpu", semantic="release")
                    if val == tiled_m_size * num_pid_n - 1:
                        st(gemm_barrier_ptr + barrier_id, 1, scope="gpu", semantic="release")


@triton_dist.jit
def kernel_persistent_tma_gemm_notify(a_ptr, b_ptr, c_ptr, gemm_barrier_ptr,  #
                                      M, N, K,  #
                                      stride_am, stride_ak,  #
                                      stride_bn, stride_bk,  #
                                      stride_cm, stride_cn,  #
                                      BLOCK_SIZE_M: tl.constexpr,  #
                                      BLOCK_SIZE_N: tl.constexpr,  #
                                      BLOCK_SIZE_K: tl.constexpr,  #
                                      GROUP_SIZE_M: tl.constexpr,  #
                                      NUM_GEMM_SMS: tl.constexpr,  #
                                      EPILOGUE_SUBTILE: tl.constexpr = False,  #
                                      FUSE_START_OFFSET: tl.constexpr = 0):
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )

    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    start_pid = tl.program_id(axis=0) - FUSE_START_OFFSET
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    tiles_per_SM = num_tiles // NUM_GEMM_SMS
    if start_pid < num_tiles % NUM_GEMM_SMS:
        tiles_per_SM += 1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    dtype = c_ptr.dtype.element_ty

    for tile_id in tl.range(start_pid, num_tiles, NUM_GEMM_SMS, flatten=False):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_GEMM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for ki in tl.range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am, offs_bn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_am, offs_bn], c)
        __syncthreads()
        thread_idx = tid(0)
        gemm_barrier_idx = pid_m * num_pid_n + pid_n
        if thread_idx == 0:
            st(gemm_barrier_ptr + gemm_barrier_idx, 1, scope="gpu", semantic="release")


@triton_dist.jit(do_not_specialize=[])
def kernel_fused_gemm_allreduce(a_ptr, b_ptr, c_ptr,  #
                                symm_ar_out_ptr, ar_out_ptr,  #
                                gemm_barrier_ptr, multi_st_barrier_ptr, grid_barrier_ptr, tile_barrier_ptr,  #
                                M, N, K,  #
                                stride_am, stride_ak,  #
                                stride_bn, stride_bk,  #
                                stride_cm, stride_cn,  #
                                BLOCK_SIZE_M: tl.constexpr,  #
                                BLOCK_SIZE_N: tl.constexpr,  #
                                BLOCK_SIZE_K: tl.constexpr,  #
                                GROUP_SIZE_M: tl.constexpr,  #
                                NUM_GEMM_SMS: tl.constexpr,  #
                                NUM_COMM_SMS: tl.constexpr,  #
                                USE_MULTIMEM_ST: tl.constexpr,  #
                                FUSE_OUTPUT_CP: tl.constexpr, use_cooperative: tl.constexpr,
                                USE_LD_REDUCE: tl.constexpr, BLOCK_SIZE_COMM: tl.constexpr = 8192,
                                TILE_MAP_LEVEL: tl.constexpr = 0, USE_INT8: tl.constexpr = True, As_ptr=None,
                                Bs_ptr=None):
    global_pid = tl.program_id(axis=0)
    if global_pid < NUM_COMM_SMS:
        if USE_LD_REDUCE:
            consumer_all_reduce_kernel(c_ptr, symm_ar_out_ptr, ar_out_ptr, gemm_barrier_ptr, multi_st_barrier_ptr, M, N,
                                       BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, NUM_COMM_SMS=NUM_COMM_SMS,
                                       USE_MULTIMEM_ST=USE_MULTIMEM_ST)
        else:
            consumer_all_reduce_load_store_kernel(c_ptr, symm_ar_out_ptr, ar_out_ptr, gemm_barrier_ptr, M, N,
                                                  BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                                  BLOCK_SIZE_COMM=BLOCK_SIZE_COMM, NUM_COMM_SMS=NUM_COMM_SMS,
                                                  TILE_MAP_LEVEL=TILE_MAP_LEVEL)
    else:
        kernel_persistent_gemm_notify(a_ptr, b_ptr, c_ptr, gemm_barrier_ptr, tile_barrier_ptr, M, N, K, stride_am,
                                      stride_ak, stride_bn, stride_bk, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N,
                                      BLOCK_SIZE_K, GROUP_SIZE_M, NUM_GEMM_SMS, As_ptr=As_ptr, Bs_ptr=Bs_ptr,
                                      USE_INT8=USE_INT8, TILE_MAP_LEVEL=TILE_MAP_LEVEL, FUSE_START_OFFSET=NUM_COMM_SMS)

    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    # if USE_MULTIMEM_ST == false, the result
    if USE_LD_REDUCE and FUSE_OUTPUT_CP and USE_MULTIMEM_ST:
        copy_1d_tilewise_kernel(symm_ar_out_ptr, ar_out_ptr, M * N, BLOCK_SIZE=2048)


def consumer_all_reduce(symm_input, symm_ar_out, ar_out, gemm_barrier, multi_st_barrier, BLOCK_SIZE_M=16,
                        BLOCK_SIZE_N=64, NUM_COMM_SMS=16, BLOCK_SIZE_COMM=4096, USE_MULTIMEM_ST=False, TILE_MAP_LEVEL=0,
                        all_reduce_method=OverlappingAllReduceMethod.Consumer_Load):
    M, N = symm_input.shape
    assert N % BLOCK_SIZE_N == 0
    if all_reduce_method == OverlappingAllReduceMethod.Consumer_Multimem:
        assert TILE_MAP_LEVEL == 0, "Only legal for per-tile allreduce"
        consumer_all_reduce_kernel[(NUM_COMM_SMS, )](symm_input, symm_ar_out, ar_out, gemm_barrier, multi_st_barrier, M,
                                                     N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                                     NUM_COMM_SMS=NUM_COMM_SMS, USE_MULTIMEM_ST=USE_MULTIMEM_ST,
                                                     num_warps=32)
    else:
        consumer_all_reduce_load_store_kernel[(NUM_COMM_SMS, )](symm_input, symm_ar_out, ar_out, gemm_barrier, M, N,
                                                                BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                                                BLOCK_SIZE_COMM=BLOCK_SIZE_COMM,
                                                                NUM_COMM_SMS=NUM_COMM_SMS, num_warps=32,
                                                                TILE_MAP_LEVEL=TILE_MAP_LEVEL)


def persistent_gemm_notify(a, b, out, gemm_barrier, tile_barrier, gemm_config: triton.Config, use_tma=False,
                           use_int8=False, As=None, Bs=None, TILE_MAP_LEVEL=0):

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    if use_int8:
        assert (As is not None and Bs is not None)
        assert As.shape[0] == a.shape[
            0], f"Incompatible scale dimensions, scale shape = {As.shape}, input shape = {a.shape}"
        assert Bs.shape[0] == b.shape[
            0], f"Incompatible scale dimensions, scale shape = {Bs.shape}, input shape = {b.shape}"
    M, K = a.shape
    N, K = b.shape
    grid = lambda META: (min(META["NUM_GEMM_SMS"],
                             triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )

    if not use_tma:
        kernel_persistent_gemm_notify[grid](
            a, b, out, gemm_barrier, tile_barrier,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            out.stride(0), out.stride(1),  #
            **gemm_config.all_kwargs(),  #
            USE_INT8=use_int8, As_ptr=As, Bs_ptr=Bs, TILE_MAP_LEVEL=TILE_MAP_LEVEL)
    else:
        kernel_persistent_tma_gemm_notify[grid](
            a, b, out, gemm_barrier,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            out.stride(0), out.stride(1),  #
            **gemm_config.all_kwargs(),  #
        )
    return out


def low_latency_gemm_allreduce_op(
    ctx: LLGemmARContext,
    A: torch.Tensor,
    B: torch.Tensor,
    gemm_config: triton.Config,
    copy_to_local=True,
    USE_MULTIMEM_ST=True,
    TILE_MAP_LEVEL=0,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
):
    ctx.update_phase()
    M, N, K = A.shape[0], B.shape[0], B.shape[1]
    # Check constraints.
    assert A.shape[1] == B.shape[1], "Incompatible dimensions"
    assert A.dtype == B.dtype, "Incompatible dtypes"

    symm_c = ctx.get_gemm_out_buf(A, B)
    symm_ar_out = ctx.symm_ar_out_buf
    gemm_barrier = ctx.gemm_barrier_buf
    multi_st_barrier = ctx.multi_st_barrier_buf
    grid_barrier = ctx.grid_barrier_buf
    tile_barrier = ctx.tile_barrier_buf

    NUM_COMM_SMS = ctx.NUM_COMM_SMS
    USE_LD_REDUCE = ctx.all_reduce_method == OverlappingAllReduceMethod.Consumer_Multimem
    if USE_MULTIMEM_ST or USE_LD_REDUCE:
        assert is_nvshmem_multimem_supported(), "multimem is unsupported"
    with_scale = (A_scale is not None and B_scale is not None)

    if with_scale:
        assert A.dtype == torch.int8
        ar_out = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    else:
        ar_out = torch.empty((M, N), dtype=A.dtype, device=A.device)

    current_stream = torch.cuda.current_stream()
    if not USE_LD_REDUCE:  # Multimem kernel will reset barrier inside the ar kernel
        nvshmem_barrier_all_on_stream(current_stream)
        ctx.reset_all_barrier_buf()
        nvshmem_barrier_all_on_stream(current_stream)

    grid = lambda META: (NUM_COMM_SMS + min(META["NUM_GEMM_SMS"],
                                            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])
                                            ), )
    kernel_fused_gemm_allreduce[grid](
        A, B, symm_c, symm_ar_out, ar_out,  #
        gemm_barrier, multi_st_barrier, grid_barrier, tile_barrier,  #
        M, N, K,  #
        A.stride(0), A.stride(1),  #
        B.stride(0), B.stride(1),  #
        symm_c.stride(0), symm_c.stride(1),  #
        **gemm_config.all_kwargs(),  #
        NUM_COMM_SMS=NUM_COMM_SMS, BLOCK_SIZE_COMM=8192, TILE_MAP_LEVEL=TILE_MAP_LEVEL, USE_MULTIMEM_ST=USE_MULTIMEM_ST,
        FUSE_OUTPUT_CP=copy_to_local, use_cooperative=True, USE_LD_REDUCE=USE_LD_REDUCE,
        **launch_cooperative_grid_options(), USE_INT8=with_scale, As_ptr=A_scale, Bs_ptr=B_scale)

    if USE_MULTIMEM_ST and not copy_to_local:
        return symm_ar_out.reshape(-1)[:M * N].reshape(M, N)
    return ar_out


def gemm_allreduce_op(ctx: GemmARContext, a, b, gemm_config: triton.Config, copy_to_local=True, USE_MULTIMEM_ST=False,
                      As=None, Bs=None, pg=None):
    M, N = a.shape[0], b.shape[0]
    NUM_COMM_SMS = ctx.NUM_COMM_SMS
    BLOCK_SIZE_M = gemm_config.all_kwargs()["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.all_kwargs()["BLOCK_SIZE_N"]
    # add mask in `consumer_all_reduce` can remove this constraint
    assert N % BLOCK_SIZE_N == 0
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    symm_c = ctx.get_gemm_out_buf(a, b)
    symm_ar_out = ctx.symm_ar_out_buf
    gemm_barrier = ctx.gemm_barrier_buf
    multi_st_barrier = ctx.multi_st_barrier_buf
    tile_barrier = ctx.tile_barrier_buf

    with_scale = (As is not None and Bs is not None)
    USE_LD_REDUCE = (ctx.all_reduce_method == OverlappingAllReduceMethod.Consumer_Multimem)
    USE_MULTIMEM_ST = (ctx.all_reduce_method == OverlappingAllReduceMethod.Consumer_Multimem) and USE_MULTIMEM_ST

    if with_scale:
        assert a.dtype == torch.int8
        ar_out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    else:
        ar_out = torch.empty((M, N), dtype=a.dtype, device=a.device)

    current_stream = torch.cuda.current_stream()
    ar_stream = ctx.ar_stream
    ar_stream.wait_stream(current_stream)

    if not USE_LD_REDUCE:  # Multimem kernel will reset barrier inside the ar kernel
        nvshmem_barrier_all_on_stream(current_stream)
        ctx.reset_all_barrier_buf()
        nvshmem_barrier_all_on_stream(current_stream)

    persistent_gemm_notify(a, b, symm_c, gemm_barrier, tile_barrier, gemm_config, use_int8=with_scale, As=As, Bs=Bs,
                           TILE_MAP_LEVEL=ctx.TILE_MAP_LEVEL)

    with torch.cuda.stream(ar_stream):
        consumer_all_reduce(symm_c, symm_ar_out, ar_out, gemm_barrier, multi_st_barrier, BLOCK_SIZE_M=BLOCK_SIZE_M,
                            BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_COMM=8192, NUM_COMM_SMS=NUM_COMM_SMS,
                            USE_MULTIMEM_ST=USE_MULTIMEM_ST, TILE_MAP_LEVEL=ctx.TILE_MAP_LEVEL,
                            all_reduce_method=ctx.all_reduce_method)
    current_stream.wait_stream(ar_stream)

    # out still in comm buffer, copy to user buffer
    if USE_MULTIMEM_ST and copy_to_local:
        ar_out.copy_(symm_ar_out.reshape(-1)[:M * N].reshape(M, N))
    if USE_MULTIMEM_ST and not copy_to_local:
        return symm_ar_out.reshape(-1)[:M * N].reshape(M, N)
    return ar_out


def gemm_op(ctx: GemmARContext, a, b, gemm_config: triton.Config, copy_to_local=True, USE_MULTIMEM_ST=False, As=None,
            Bs=None):
    M, N = a.shape[0], b.shape[0]
    BLOCK_SIZE_N = gemm_config.all_kwargs()["BLOCK_SIZE_N"]
    # add mask in `consumer_all_reduce` can remove this constraint
    assert N % BLOCK_SIZE_N == 0
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    symm_c = ctx.get_gemm_out_buf(a, b)
    gemm_barrier = ctx.gemm_barrier_buf
    tile_barrier = ctx.tile_barrier_buf

    with_scale = (As is not None and Bs is not None)

    if with_scale:
        assert a.dtype == torch.int8
        ar_out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    else:
        ar_out = torch.empty((M, N), dtype=a.dtype, device=a.device)

    persistent_gemm_notify(a, b, symm_c, gemm_barrier, tile_barrier, gemm_config, use_int8=with_scale, As=As, Bs=Bs,
                           TILE_MAP_LEVEL=ctx.TILE_MAP_LEVEL)
    ar_out.copy_(symm_c.reshape(-1)[:M * N].reshape(M, N))
    return ar_out


def allreduce_op(ctx: GemmARContext, c, gemm_config: triton.Config, TILE_MAP_LEVEL=0, copy_to_local=True,
                 USE_MULTIMEM_ST=False):
    M, N = c.shape
    NUM_COMM_SMS = ctx.NUM_COMM_SMS
    BLOCK_SIZE_M = gemm_config.all_kwargs()["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.all_kwargs()["BLOCK_SIZE_N"]
    # add mask in `consumer_all_reduce` can remove this constraint
    assert N % BLOCK_SIZE_N == 0

    symm_c = ctx.symm_gemm_out_buf.reshape(-1)[:M * N].reshape(M, N)
    symm_c.copy_(c)
    symm_ar_out = ctx.symm_ar_out_buf
    gemm_barrier = ctx.gemm_barrier_buf
    multi_st_barrier = ctx.multi_st_barrier_buf
    ar_out = torch.empty((M, N), dtype=c.dtype, device=c.device)

    USE_MULTIMEM_ST = (ctx.all_reduce_method == OverlappingAllReduceMethod.Consumer_Multimem) and USE_MULTIMEM_ST

    gemm_barrier.fill_(1)
    consumer_all_reduce(symm_c, symm_ar_out, ar_out, gemm_barrier, multi_st_barrier, BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N, NUM_COMM_SMS=NUM_COMM_SMS, USE_MULTIMEM_ST=USE_MULTIMEM_ST,
                        BLOCK_SIZE_COMM=8192, TILE_MAP_LEVEL=TILE_MAP_LEVEL, all_reduce_method=ctx.all_reduce_method)

    # out still in comm buffer, copy to user buffer
    if USE_MULTIMEM_ST and copy_to_local:
        ar_out.copy_(symm_ar_out.reshape(-1)[:M * N].reshape(M, N))
    if USE_MULTIMEM_ST and not copy_to_local:
        return symm_ar_out.reshape(-1)[:M * N].reshape(M, N)
    return ar_out
