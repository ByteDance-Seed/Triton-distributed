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
import dataclasses
from typing import List, Optional

import torch

import triton
import triton.language as tl
import triton_dist.language as dl
from triton.language.extra.cuda.language_extra import (__syncthreads, atomic_add, tid, st)
from triton_dist.kernels.nvidia.reduce_scatter import (ReduceScatter2DContext, create_reduce_scater_2d_ctx,
                                                       reduce_scatter_2d_op, ring_reduce)
from triton_dist.kernels.nvidia.gemm_rs_threadblock_swizzle import threadblock_swizzle_gemm_reduce_scatter_kernel
from triton_dist.utils import has_tma, nvshmem_barrier_all_on_stream, nvshmem_create_tensors, nvshmem_free_tensor_sync, requires


################### context ###################
@dataclasses.dataclass
class GEMMReduceScatterTensorParallelContext:
    rs_ctx: ReduceScatter2DContext
    output_dtype: torch.dtype

    # gemm bufs (symm address)
    gemm_out_bufs: List[torch.Tensor]

    # stream
    rs_stream: torch.cuda.Stream

    # gemm kernel config
    num_gemm_sms: int
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    GROUP_M: int = 8
    stages: int = 3

    def finalize(self):
        self.rs_ctx.finalize()
        nvshmem_free_tensor_sync(self.gemm_out_bufs[self.rs_ctx.local_rank])

    def get_gemm_out_buf(self, input):
        M, _ = input.shape
        local_rank = self.rs_ctx.local_rank
        return self.gemm_out_bufs[local_rank][:M]


def create_gemm_rs_context(max_M, N, rank, world_size, local_world_size, output_dtype: torch.dtype,
                           rs_stream: torch.cuda.Stream, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, GROUP_M=8,
                           stages=3) -> GEMMReduceScatterTensorParallelContext:
    rs_ctx = create_reduce_scater_2d_ctx(max_M, N, rank, world_size, local_world_size, output_dtype,
                                         overlap_with_gemm=True)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_gemm_sms = NUM_SMS - rs_ctx.num_rs_sms
    gemm_out_bufs = nvshmem_create_tensors((max_M, N), output_dtype, rank, local_world_size)
    ctx = GEMMReduceScatterTensorParallelContext(rs_ctx=rs_ctx, output_dtype=output_dtype, gemm_out_bufs=gemm_out_bufs,
                                                 rs_stream=rs_stream, num_gemm_sms=num_gemm_sms, BLOCK_M=BLOCK_M,
                                                 BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M, stages=stages)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    return ctx


@triton.jit
def swizzle_2d(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# TMA related test
def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = args["a_ptr"].element_size()
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


def _gemm_rs_persistent_repr(proxy):
    constexprs = proxy.constants
    cap_major, cap_minor = torch.cuda.get_device_capability()
    a_dtype = proxy.signature["a_ptr"].lstrip("*")
    b_dtype = proxy.signature["b_ptr"].lstrip("*")
    c_dtype = proxy.signature["c_ptr"].lstrip("*")
    BM, BN, BK = constexprs["BLOCK_SIZE_M"], constexprs["BLOCK_SIZE_N"], constexprs["BLOCK_SIZE_K"]

    return f"triton3x_sm{cap_major}{cap_minor}_gemm_rs_persistent_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_ntn"


@triton.jit(launch_metadata=_matmul_launch_metadata, repr=_gemm_rs_persistent_repr)
def kernel_gemm_rs_producer_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    barrier_ptr,
    counter_ptr,
    FUSE_SCATTER: tl.constexpr,
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):  #
    # Matmul using TMA and device-side descriptor creation
    rank = dl.rank()
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

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
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    M_per_rank = M // WORLD_SIZE

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    pid_m_offset = (rank + 1) * M_per_rank // BLOCK_SIZE_M

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            pid_m, pid_n = swizzle_2d(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)
            if NNODES != 1:  # with complex threadblock swizzle logic
                pid_m = threadblock_swizzle_gemm_reduce_scatter_kernel(pid_m, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M)
            else:
                pid_m = (pid_m + pid_m_offset) % num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if not FUSE_SCATTER:
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

                counter_start = offs_am // M_per_rank
                counter_end = (offs_am + BLOCK_SIZE_M - 1) // M_per_rank
                counter_end = min(counter_end, WORLD_SIZE - 1)
                for counter_id in range(counter_start, counter_end + 1):
                    m_start = M_per_rank * counter_id
                    m_end = M_per_rank * (counter_id + 1) - 1
                    tiled_m_start = m_start // BLOCK_SIZE_M
                    tiled_m_end = m_end // BLOCK_SIZE_M
                    tiled_m_size = tiled_m_end - tiled_m_start + 1
                    val = tl.atomic_add(counter_ptr + counter_id, 1, sem="release", scope="gpu")
                    if val == tiled_m_size * num_pid_n - 1:
                        dl.notify(barrier_ptr + counter_id, rank, signal=1, comm_scope="gpu")
            else:
                rank_start = pid_m * BLOCK_SIZE_M // M_per_rank
                rank_end = (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) // M_per_rank
                offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                for cur_rank in range(rank_start, rank_end + 1):
                    m_start = max(M_per_rank * cur_rank, pid_m * BLOCK_SIZE_M)
                    m_end = min(M_per_rank * (cur_rank + 1) - 1, (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1))
                    remote_c_ptr = dl.symm_at(c_ptr, cur_rank)
                    mask_offset = m_start - pid_m * BLOCK_SIZE_M
                    remote_offs_cm = m_start % M_per_rank + rank * M_per_rank + tl.arange(0, BLOCK_SIZE_M) - mask_offset
                    remote_c_ptrs = remote_c_ptr + N * remote_offs_cm[:, None] + offs_cn[None, :]
                    remote_mask = (offs_cm[:, None] <= m_end) & (offs_cm[:, None] >= m_start) & (offs_cn[None, :] < N)
                    tl.store(remote_c_ptrs, accumulator, mask=remote_mask)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def _gemm_rs_non_persistent_repr(proxy):
    constexprs = proxy.constants
    cap_major, cap_minor = torch.cuda.get_device_capability()
    a_dtype = proxy.signature["a_ptr"].lstrip("*")
    b_dtype = proxy.signature["b_ptr"].lstrip("*")
    c_dtype = proxy.signature["c_ptr"].lstrip("*")
    BM, BN, BK = constexprs["BLOCK_SIZE_M"], constexprs["BLOCK_SIZE_N"], constexprs["BLOCK_SIZE_K"]
    if constexprs.get("stride_am", None) == 1:  # column major => n
        a_trans = "n"
    elif constexprs.get("stride_ak", None) == 1:  # row-major => t
        a_trans = "t"
    else:
        raise Exception("both stride_am/stride_ak != 1")

    if constexprs.get("stride_bk", None) == 1:
        b_trans = "n"
    elif constexprs.get("stride_bn", None) == 1:
        b_trans = "t"
    else:
        raise Exception("both stride_am/stride_ak != 1")

    if constexprs.get("stride_cm", None) == 1:
        c_trans = "n"
    elif constexprs.get("stride_cn", None) == 1:
        c_trans = "t"
    else:
        raise Exception("both stride_am/stride_ak != 1")

    return f"triton3x_sm{cap_major}{cap_minor}_gemm_rs_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_{a_trans}{b_trans}{c_trans}"


@triton.jit(launch_metadata=_matmul_launch_metadata, repr=_gemm_rs_non_persistent_repr)
def kernel_gemm_rs_producer_non_persistent(
    # Pointers to matrices
    a_ptr,  # [M, K]_Ti
    b_ptr,  # [K, N]_Ti
    c_ptr,  # [M, N]_To
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    barrier_ptr,
    counter_ptr,
    FUSE_SCATTER: tl.constexpr,
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    tl.static_assert(a_ptr.dtype.is_ptr(), "A should be a pointer")
    tl.static_assert(b_ptr.dtype.is_ptr(), "B should be a pointer")
    tl.static_assert(c_ptr.dtype.is_ptr(), "C should be a pointer")
    a_dtype = a_ptr.dtype.element_ty
    b_dtype = b_ptr.dtype.element_ty
    tl.static_assert(a_dtype == b_dtype, "A and B should have the same dtype")
    # IS_FP8 = tl.constexpr(a_dtype == tl.float8e4nv) or tl.constexpr(a_dtype == tl.float8e5)

    rank = dl.rank()
    NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    M_per_rank = M // WORLD_SIZE
    pid_m, pid_n = swizzle_2d(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    if NNODES != 1:  # with complex threadblock swizzle logic
        pid_m = threadblock_swizzle_gemm_reduce_scatter_kernel(pid_m, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M)
    else:
        pid_m_offset = (rank + 1) * M_per_rank // BLOCK_SIZE_M
        pid_m = (pid_m + pid_m_offset) % num_pid_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if a_ptr.dtype.element_ty == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if not FUSE_SCATTER:
        tl.store(c_ptrs, accumulator, mask=out_mask)

        # inc barrier
        segment_start = pid_m * BLOCK_SIZE_M // M_per_rank
        segment_end = (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) // M_per_rank
        __syncthreads()
        segment = segment_start + tid(axis=0)
        if segment <= segment_end:
            m_start = M_per_rank * segment
            m_end = M_per_rank * (segment + 1) - 1
            tiled_m_start = m_start // BLOCK_SIZE_M
            tiled_m_end = m_end // BLOCK_SIZE_M
            tiled_m_size = tiled_m_end - tiled_m_start + 1
            val = atomic_add(counter_ptr + segment, 1, semantic="release", scope="gpu")
            if val == num_pid_n * tiled_m_size - 1:
                # or use other signal op semantic
                st(barrier_ptr + segment, 1, scope="gpu", semantic="release")
    else:
        rank_start = pid_m * BLOCK_SIZE_M // M_per_rank
        rank_end = (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) // M_per_rank
        for cur_rank in range(rank_start, rank_end + 1):
            m_start = max(M_per_rank * cur_rank, pid_m * BLOCK_SIZE_M)
            m_end = min(M_per_rank * (cur_rank + 1) - 1, (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1))
            remote_c_ptr = dl.symm_at(c_ptr, cur_rank)
            mask_offset = m_start - pid_m * BLOCK_SIZE_M
            remote_offs_cm = m_start % M_per_rank + rank * M_per_rank + tl.arange(0, BLOCK_SIZE_M) - mask_offset
            remote_c_ptrs = remote_c_ptr + stride_cm * remote_offs_cm[:, None] + stride_cn * offs_cn[None, :]
            remote_mask = (offs_cm[:, None] <= m_end) & (offs_cm[:, None] >= m_start) & (offs_cn[None, :] < N)
            tl.store(remote_c_ptrs, accumulator, mask=remote_mask)


@requires(has_tma)
def gemm_rs_producer_persistent(a, b, c, barrier, workspace, world_size, local_world_size, fuse_scatter, num_gemm_sms,
                                triton_config: triton.Config):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, local_K = a.shape
    N, local_K = b.shape

    grid = lambda META: (min(
        num_gemm_sms,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    ), )

    compiled = kernel_gemm_rs_producer_persistent[grid](a, b, c, M, N, local_K, barrier, workspace, fuse_scatter,
                                                        local_world_size, world_size, **triton_config.all_kwargs())

    return compiled


def gemm_rs_producer_non_persistent(a, b, c, barrier, workspace, world_size, local_world_size, fuse_scatter,
                                    triton_config: triton.Config):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, local_K = a.shape
    N, local_K = b.shape

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    compiled = kernel_gemm_rs_producer_non_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        local_K,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        c.stride(0),
        c.stride(1),
        barrier,
        workspace,
        fuse_scatter,
        local_world_size,
        world_size,
        **triton_config.all_kwargs(),
    )
    return compiled


def padded_to_BLOCK_M(input, world_size, BLOCK_SIZE_M):
    M, local_K = input.shape

    M_per_rank = M // world_size
    pad_size = (M_per_rank + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * BLOCK_SIZE_M
    if pad_size == M_per_rank:
        return input
    input = input.reshape(world_size, M_per_rank, local_K)
    pad_input = torch.empty((world_size, pad_size, local_K), dtype=input.dtype, device=input.device)
    pad_input[:, :M_per_rank].copy_(input)
    pad_input = pad_input.reshape(-1, local_K)
    return pad_input


def update_triton_config(M, N, K, dtype: torch.dtype, world_size, local_world_size, config: triton.Config):
    """
        It's hard to autotune all parameters and record them all, especially when there are so many shapes and devices and dtypes.
        So we just use a simple heuristic rule to update the config.
    """
    from triton_dist.kernels.nvidia.comm_perf_model import (estimate_reduce_scatter_time_ms, get_nic_gbps_per_gpu)
    from triton_dist.kernels.nvidia.gemm_perf_model import \
        estimate_gemm_sol_time_ms
    from triton_dist.utils import get_intranode_max_speed
    gemm_time_ms = estimate_gemm_sol_time_ms(M, N, K, dtype)
    rs_time_ms = estimate_reduce_scatter_time_ms(M * N * dtype.itemsize, world_size, local_world_size,
                                                 get_intranode_max_speed(), get_nic_gbps_per_gpu())
    BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
    GROUP_SIZE_M = config.kwargs["GROUP_SIZE_M"]
    if gemm_time_ms < rs_time_ms:
        # comm efficiency first
        M_per_rank = M // world_size
        tiled_m_per_rank = (M_per_rank + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        if tiled_m_per_rank < GROUP_SIZE_M:
            # don't use too large GROUP_SIZE_M
            # TODO(houqi.1993) maybe we should take care into a wave
            config.kwargs["GROUP_SIZE_M"] = tiled_m_per_rank

    return config


def gemm_rs_op(input, weight, ctx: GEMMReduceScatterTensorParallelContext, persistent: bool = True,
               fuse_scatter: bool = False):
    if fuse_scatter:
        assert ctx.rs_ctx.nnodes == 1, "`fuse_scatter` does not support multi node`"
    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    rs_stream = ctx.rs_stream
    output_dtype = ctx.output_dtype
    num_gemm_sms = ctx.num_gemm_sms

    M, local_K = input.shape
    N = weight.shape[0]
    assert N == ctx.rs_ctx.N

    assert M % world_size == 0
    assert weight.shape[1] == local_K
    M_per_rank = M // world_size
    current_stream = torch.cuda.current_stream()
    rs_stream.wait_stream(current_stream)

    output = torch.empty((M_per_rank, N), dtype=output_dtype, device=input.device)
    workspace = torch.zeros((world_size, ), dtype=torch.int32, device=input.device)
    gemm_out = ctx.get_gemm_out_buf(input)
    scatter_signal = ctx.rs_ctx.scatter_signal_buf

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    if persistent:
        triton_config = triton.Config(
            {
                "BLOCK_SIZE_M": ctx.BLOCK_M, "BLOCK_SIZE_N": ctx.BLOCK_N, "BLOCK_SIZE_K": ctx.BLOCK_K, "GROUP_SIZE_M":
                ctx.GROUP_M, "NUM_SMS": num_gemm_sms, "EPILOGUE_SUBTILE": False
            }, num_stages=ctx.stages, num_warps=8)
        gemm_rs_producer_persistent(input, weight, gemm_out, scatter_signal, workspace, world_size, local_world_size,
                                    fuse_scatter, num_gemm_sms, triton_config)
    else:
        triton_config = triton.Config(
            {
                "BLOCK_SIZE_M": ctx.BLOCK_M, "BLOCK_SIZE_N": ctx.BLOCK_N, "BLOCK_SIZE_K": ctx.BLOCK_K, "GROUP_SIZE_M":
                ctx.GROUP_M
            }, num_stages=ctx.stages, num_warps=8)
        triton_config = update_triton_config(M, N, local_K, input.dtype, world_size, local_world_size, triton_config)
        gemm_rs_producer_non_persistent(input, weight, gemm_out, scatter_signal, workspace, world_size,
                                        local_world_size, fuse_scatter, triton_config)

    if not fuse_scatter:
        with torch.cuda.stream(rs_stream):
            # don't allocate memory on other stream: error-prune
            reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output)
        current_stream.wait_stream(rs_stream)
    else:
        nvshmem_barrier_all_on_stream(current_stream)
        ring_reduce(gemm_out, output, ctx.rs_ctx.local_rank, local_world_size)
        nvshmem_barrier_all_on_stream(current_stream)
    return output


def gemm_rs(a, b, ctx, persistent=True, fuse_scatter=False):
    """GEMM Reduce-Scatter for Multi-Node

    computes local GEMM (a x b) to generate partial results, followed by `reduce_scatter` to produce c

    Args:
        a (torch.Tensor<bfloat16/float16>): local matmul A matrix. shape: [M, local_K]
        b (torch.Tensor<bfloat16/float16>): local matmul B matrix. shape: [N, local_K]
        ctx(GEMMReduceScatterTensorParallelContext): context

    Returns:
        c (torch.Tensor<bfloat16/float16>): local matmul C matrix. shape: [M // world_size, N]
    """
    c = gemm_rs_op(a, b, ctx, persistent, fuse_scatter)
    return c
