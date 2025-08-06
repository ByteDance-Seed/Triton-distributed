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
import triton_dist.language as dl
from triton_dist.utils import (nvshmem_barrier_all_on_stream, nvshmem_create_tensor, nvshmem_free_tensor_sync,
                               launch_cooperative_grid_options)
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cuda.utils import num_warps
from triton.language.extra.cuda.language_extra import (
    __syncthreads,
    ld,
    st,
    tid,
    multimem_ld_reduce_v4,
    multimem_st_v4,
    st_v4_b32,
)
from triton_dist.kernels.nvidia.common_ops import barrier_on_this_grid


@dataclasses.dataclass
class GemmARContext:

    symm_gemm_out_buf: torch.Tensor
    symm_ar_out_buf: torch.Tensor

    gemm_barrier_buf: torch.Tensor
    multi_st_barrier_buf: torch.Tensor
    grid_barrier_buf: torch.Tensor

    NUM_COMM_SMS: int

    ar_stream: torch.cuda.Stream

    def finalize(self):
        nvshmem_free_tensor_sync(self.symm_gemm_out_buf)
        nvshmem_free_tensor_sync(self.symm_ar_out_buf)
        nvshmem_free_tensor_sync(self.gemm_barrier_buf)
        nvshmem_free_tensor_sync(self.multi_st_barrier_buf)

    def get_gemm_out_buf(self, input, weight):
        M, N = input.shape[0], weight.shape[0]
        assert self.symm_gemm_out_buf.numel() >= M * N
        return self.symm_gemm_out_buf.reshape(-1)[:M * N].reshape(M, N)


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


def create_gemm_ar_context(ar_stream: torch.cuda.Stream, rank, world_size, local_world_size, max_M, N, dtype,
                           MIN_BLOCK_SIZE_M=16, MIN_BLOCK_SIZE_N=16, NUM_COMM_SMS=132):
    assert local_world_size == world_size
    gemm_out_buf = nvshmem_create_tensor((world_size, max_M, N), dtype)
    symm_ar_out_buf = nvshmem_create_tensor((max_M, N), dtype)
    gemm_barrier_buf = nvshmem_create_tensor(
        (world_size, triton.cdiv(max_M, MIN_BLOCK_SIZE_M), triton.cdiv(N, MIN_BLOCK_SIZE_N)), torch.int32)
    multi_st_barrier_buf = nvshmem_create_tensor((world_size * NUM_COMM_SMS, ), torch.int32)
    grid_barrier_buf = torch.zeros((1, ), dtype=torch.int32, device=torch.cuda.current_device())
    gemm_barrier_buf.zero_()
    multi_st_barrier_buf.zero_()
    nvshmem_barrier_all_on_stream()
    return GemmARContext(symm_gemm_out_buf=gemm_out_buf, symm_ar_out_buf=symm_ar_out_buf,
                         gemm_barrier_buf=gemm_barrier_buf, multi_st_barrier_buf=multi_st_barrier_buf,
                         grid_barrier_buf=grid_barrier_buf, NUM_COMM_SMS=NUM_COMM_SMS, ar_stream=ar_stream)


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


@triton.jit(do_not_specialize=[])
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


@triton.jit(do_not_specialize=["nelems"])
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


@triton.jit(do_not_specialize=[])
def kernel_fused_gemm_allreduce(
    a_ptr,
    b_ptr,
    c_ptr,  #
    symm_ar_out_ptr,
    ar_out_ptr,  #
    gemm_barrier_ptr,
    multi_st_barrier_ptr,
    grid_barrier_ptr,  #
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
    NUM_COMM_SMS: tl.constexpr,  #
    USE_MULTIMEM_ST: tl.constexpr,  #
    FUSE_OUTPUT_CP: tl.constexpr,
    use_cooperative: tl.constexpr,
):
    global_pid = tl.program_id(axis=0)
    if global_pid < NUM_COMM_SMS:
        consumer_all_reduce_kernel(c_ptr, symm_ar_out_ptr, ar_out_ptr, gemm_barrier_ptr, multi_st_barrier_ptr, M, N,
                                   BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, NUM_COMM_SMS=NUM_COMM_SMS,
                                   USE_MULTIMEM_ST=USE_MULTIMEM_ST)
    else:
        gemm_start_pid = global_pid - NUM_COMM_SMS
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
        num_tiles = num_pid_m * num_pid_n
        offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

        for tile_id in tl.range(gemm_start_pid, num_tiles, NUM_GEMM_SMS):
            # pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
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
            for ki in range(k_tiles):
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)

                a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
                accumulator = tl.dot(a, b.T, accumulator)

            # tile_id_c += NUM_GEMM_SMS
            # pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_GEMM_SMS)
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            c = accumulator.to(c_ptr.dtype.element_ty)
            tl.store(c_ptrs, c, mask=c_mask)
            __syncthreads()
            thread_idx = tid(0)
            gemm_barrier_idx = pid_m * num_pid_n + pid_n
            if thread_idx == 0:
                st(gemm_barrier_ptr + gemm_barrier_idx, 1, scope="gpu", semantic="release")

    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    # if USE_MULTIMEM_ST == false, the result
    if FUSE_OUTPUT_CP and USE_MULTIMEM_ST:
        copy_1d_tilewise_kernel(symm_ar_out_ptr, ar_out_ptr, M * N, BLOCK_SIZE=2048)


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(do_not_specialize=[])
def kernel_persistent_gemm_notify(a_ptr, b_ptr, c_ptr, gemm_barrier_ptr,  #
                                  M, N, K,  #
                                  stride_am, stride_ak,  #
                                  stride_bn, stride_bk,  #
                                  stride_cm, stride_cn,  #
                                  BLOCK_SIZE_M: tl.constexpr,  #
                                  BLOCK_SIZE_N: tl.constexpr,  #
                                  BLOCK_SIZE_K: tl.constexpr,  #
                                  GROUP_SIZE_M: tl.constexpr,  #
                                  NUM_GEMM_SMS: tl.constexpr,  #
                                  ):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_GEMM_SMS

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

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_GEMM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_GEMM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)
        __syncthreads()

        thread_idx = tid(0)
        gemm_barrier_idx = pid_m * num_pid_n + pid_n
        if thread_idx == 0:
            st(gemm_barrier_ptr + gemm_barrier_idx, 1, scope="gpu", semantic="release")


@triton.jit
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


def persistent_gemm_notify(a, b, out, barrier, gemm_config: triton.Config, use_tma=False):

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    grid = lambda META: (min(META["NUM_GEMM_SMS"],
                             triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    if not use_tma:
        kernel_persistent_gemm_notify[grid](
            a, b, out, barrier,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            out.stride(0), out.stride(1),  #
            **gemm_config.all_kwargs(),  #
        )
    else:
        kernel_persistent_tma_gemm_notify[grid](
            a, b, out, barrier,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            out.stride(0), out.stride(1),  #
            **gemm_config.all_kwargs(),  #
        )
    return out


def consumer_all_reduce(symm_input, symm_ar_out, ar_out, gemm_barrier, multi_st_barrier, BLOCK_SIZE_M=16,
                        BLOCK_SIZE_N=64, NUM_COMM_SMS=16, USE_MULTIMEM_ST=False):
    M, N = symm_input.shape
    assert N % BLOCK_SIZE_N == 0
    consumer_all_reduce_kernel[(NUM_COMM_SMS, )](symm_input, symm_ar_out, ar_out, gemm_barrier, multi_st_barrier, M, N,
                                                 BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                                 NUM_COMM_SMS=NUM_COMM_SMS, USE_MULTIMEM_ST=USE_MULTIMEM_ST,
                                                 num_warps=32)


def low_latency_gemm_allreduce_op(ctx: LLGemmARContext, a, b, gemm_config: triton.Config, copy_to_local=True,
                                  USE_MULTIMEM_ST=True):
    ctx.update_phase()
    M, N = a.shape[0], b.shape[0]
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    symm_c = ctx.get_gemm_out_buf(a, b)
    symm_ar_out = ctx.symm_ar_out_buf
    gemm_barrier = ctx.gemm_barrier_buf
    multi_st_barrier = ctx.multi_st_barrier_buf
    grid_barrier = ctx.grid_barrier_buf

    NUM_COMM_SMS = ctx.NUM_COMM_SMS

    ar_out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    # print(f"M, N, K = {M, N, K}")
    grid = lambda META: (NUM_COMM_SMS + min(META["NUM_GEMM_SMS"],
                                            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])
                                            ), )
    kernel_fused_gemm_allreduce[grid](
        a, b, symm_c, symm_ar_out, ar_out,  #
        gemm_barrier, multi_st_barrier, grid_barrier,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        symm_c.stride(0), symm_c.stride(1),  #
        **gemm_config.all_kwargs(),  #
        NUM_COMM_SMS=NUM_COMM_SMS, USE_MULTIMEM_ST=USE_MULTIMEM_ST, FUSE_OUTPUT_CP=copy_to_local, use_cooperative=True,
        **launch_cooperative_grid_options())
    if USE_MULTIMEM_ST and not copy_to_local:
        return symm_ar_out.reshape(-1)[:M * N].reshape(M, N)
    return ar_out


def gemm_allreduce_op(ctx: GemmARContext, a, b, gemm_config: triton.Config, copy_to_local=True, USE_MULTIMEM_ST=True):

    current_stream = torch.cuda.current_stream()
    ar_stream = ctx.ar_stream
    ar_stream.wait_stream(current_stream)

    M, N = a.shape[0], b.shape[0]
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    symm_c = ctx.get_gemm_out_buf(a, b)
    symm_ar_out = ctx.symm_ar_out_buf
    gemm_barrier = ctx.gemm_barrier_buf
    multi_st_barrier = ctx.multi_st_barrier_buf
    NUM_COMM_SMS = ctx.NUM_COMM_SMS
    ar_out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    BLOCK_SIZE_M = gemm_config.all_kwargs()["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.all_kwargs()["BLOCK_SIZE_N"]
    # add mask in `consumer_all_reduce` can remove this constraint
    assert N % BLOCK_SIZE_N == 0
    persistent_gemm_notify(a, b, symm_c, gemm_barrier, gemm_config)
    with torch.cuda.stream(ar_stream):
        consumer_all_reduce(symm_c, symm_ar_out, ar_out, gemm_barrier, multi_st_barrier, BLOCK_SIZE_M=BLOCK_SIZE_M,
                            BLOCK_SIZE_N=BLOCK_SIZE_N, NUM_COMM_SMS=NUM_COMM_SMS, USE_MULTIMEM_ST=USE_MULTIMEM_ST)
    current_stream.wait_stream(ar_stream)
    # out still in comm buffer, copy to user buffer
    if USE_MULTIMEM_ST and copy_to_local:
        ar_out.copy_(symm_ar_out.reshape(-1)[:M * N].reshape(M, N))
    # some ranks may not reset the barrier, other ranks will read dirty data during allreduce in the next iter.
    nvshmem_barrier_all_on_stream(current_stream)
    if USE_MULTIMEM_ST and not copy_to_local:
        return symm_ar_out.reshape(-1)[:M * N].reshape(M, N)
    return ar_out
