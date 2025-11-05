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
import triton
import triton.language as tl
from typing import List

import pyrocshmem
from triton_dist.kernels.amd.common_ops import barrier_all_on_stream
import triton_dist.tune
from triton.runtime.driver import driver

SIGNAL_DTYPE = torch.int32

################# triton kernel ###################


def get_hip_autotune_config():
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'M_PER_COPY_CHUNK':
                128, 'waves_per_eu': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'M_PER_COPY_CHUNK':
                128, 'waves_per_eu': 2
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'M_PER_COPY_CHUNK': 64,
                'waves_per_eu': 3
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'M_PER_COPY_CHUNK': 128,
                'waves_per_eu': 3
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'M_PER_COPY_CHUNK': 256,
                'waves_per_eu': 3
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'M_PER_COPY_CHUNK': 512,
                'waves_per_eu': 3
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'M_PER_COPY_CHUNK': 64,
                'waves_per_eu': 0, 'matrix_instr_nonkdim': 16, 'kpack': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'M_PER_COPY_CHUNK': 128,
                'waves_per_eu': 0, 'matrix_instr_nonkdim': 16, 'kpack': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'M_PER_COPY_CHUNK': 256,
                'waves_per_eu': 0, 'matrix_instr_nonkdim': 16, 'kpack': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'M_PER_COPY_CHUNK': 512,
                'waves_per_eu': 0, 'matrix_instr_nonkdim': 16, 'kpack': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'M_PER_COPY_CHUNK':
                1024, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'M_PER_COPY_CHUNK':
                512, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'M_PER_COPY_CHUNK':
                256, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'M_PER_COPY_CHUNK':
                128, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'M_PER_COPY_CHUNK':
                256, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'M_PER_COPY_CHUNK':
                512, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=4, num_stages=2),
        ##
    ]


@triton.jit
def kernel_gemm_rs_producer_fuse_scatter(
        # Pointers to matrices
        a_ptr, b_ptr, scatter_bufs_ptr, rank, num_ranks,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr, M_PER_COPY_CHUNK: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # rank swizzle
    M_per_rank = M // num_ranks
    num_pid_m_per_copy_chunk = M_PER_COPY_CHUNK // BLOCK_SIZE_M
    chunk_offset = pid_m // (num_pid_m_per_copy_chunk * num_ranks)
    rank_offset = pid_m % (num_pid_m_per_copy_chunk * num_ranks) // num_pid_m_per_copy_chunk
    block_offset = pid_m % num_pid_m_per_copy_chunk

    # rank_swizzle_offset = M_per_rank * nxt_rank // BLOCK_SIZE_M
    # pid_m = (pid_m + rank_swizzle_offset) % num_pid_m
    rank_offset = (rank_offset + rank + 1) % num_ranks
    pid_m = (rank_offset * M_per_rank + chunk_offset * M_PER_COPY_CHUNK + block_offset * BLOCK_SIZE_M) // BLOCK_SIZE_M

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

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
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    dtype = a_ptr.dtype.element_ty
    c = accumulator.to(dtype)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    target_m = ((pid_m * BLOCK_SIZE_M % M_per_rank) + M_per_rank * rank)
    offs_cm = target_m + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = tl.load(scatter_bufs_ptr + rank_offset).to(tl.pointer_type(dtype))
    c_ptr = tl.multiple_of(c_ptr, 16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def kernel_consumer_reduce(
    c_ptr,  # [M, N]
    out_ptr,  # [M_per_rank, N]
    # shape of matrix
    M_per_rank,
    N,
    rank,
    num_ranks: tl.constexpr,
    # tile size
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.where(offs < M_per_rank * N, offs, 0)

    out_ptrs = out_ptr + offs

    accum = tl.zeros((BLOCK_SIZE, ), dtype=out_ptr.dtype.element_ty)
    for i in range(0, num_ranks):
        cur_rank = (i + rank + 1) % num_ranks
        c_ptrs = c_ptr + offs + cur_rank * M_per_rank * N
        data = tl.load(c_ptrs)
        accum += data

    tl.store(out_ptrs, accum)


#####################################################


def ring_reduce_after_scatter(
    rank,
    num_ranks,
    scatter_out: torch.Tensor,  # [M, N]
    stream: torch.cuda.Stream,
):
    M, N = scatter_out.shape
    M_per_rank = M // num_ranks
    output = torch.empty((M_per_rank, N), dtype=scatter_out.dtype, device=scatter_out.device)
    grid = lambda META: (triton.cdiv(M_per_rank * N, META["BLOCK_SIZE"]), )
    with torch.cuda.stream(stream):
        kernel_consumer_reduce[grid](
            scatter_out,
            output,
            M_per_rank,
            N,
            rank=rank,
            num_ranks=num_ranks,
            BLOCK_SIZE=2048,
            num_warps=2,
        )

    return output


def matmul_fuse_scatter(A: torch.Tensor, B: torch.Tensor, scatter_bufs_ptr: torch.Tensor, rank, num_ranks,
                        gemm_config: triton.Config):
    # Check constraints.
    assert A.shape[1] == B.shape[1], "Incompatible dimensions"
    N, K = B.shape
    assert A.is_contiguous(), "Matrix A must be contiguous"
    M, K = A.shape

    alignment = 128
    assert M % alignment == 0 and N % alignment == 0 and K % alignment == 0

    # Allocates output.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    compiled = kernel_gemm_rs_producer_fuse_scatter[grid](
        A, B, scatter_bufs_ptr,  #
        rank, num_ranks, M, N, K,  #
        A.stride(0), A.stride(1),  #
        B.stride(1), B.stride(0),  #
        N, 1,  #
        **gemm_config.all_kwargs())
    return compiled


def prune_fn_by_shared_memory(config, A: torch.Tensor, *args, **kwargs):
    itemsize = A.itemsize
    gemm_config: triton.Config = config["gemm_config"]
    BLOCK_SIZE_M = gemm_config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = gemm_config.kwargs["BLOCK_SIZE_K"]
    num_stages = max(0, gemm_config.num_stages - 1)
    shared_mem_size = num_stages * (BLOCK_SIZE_M * BLOCK_SIZE_K * itemsize + BLOCK_SIZE_N * BLOCK_SIZE_K * itemsize)
    device = torch.cuda.current_device()
    if shared_mem_size > driver.active.utils.get_device_properties(device)["max_shared_mem"]:
        return False
    return True


def key_fn(A: torch.Tensor, B: torch.Tensor, output_dtype: torch.dtype, *args, **kwargs):
    return triton_dist.tune.to_hashable(A), triton_dist.tune.to_hashable(B), output_dtype


DEFAULT_GEMM_CONFIG = get_hip_autotune_config()[0]


@triton_dist.tune.autotune(config_space=[{"gemm_config": c} for c in get_hip_autotune_config()], key_fn=key_fn,
                           prune_fn=prune_fn_by_shared_memory)
def gemm_rs_intra_node_op(A: torch.Tensor, B: torch.Tensor, output_dtype: torch.dtype, rank, num_ranks,
                          scatter_bufs: List[torch.Tensor], scatter_bufs_ptr: torch.Tensor, sync_bufs: torch.Tensor,
                          fuse_scatter=True, gemm_config: triton.Config = DEFAULT_GEMM_CONFIG):
    """gemm reduce scatter for intra-node

    return C = reduce_scatter(A, B.T)

    Args:
        A (torch.Tensor<float>): local matmul A matrix. shape: [M, K_per_rank]
        B (torch.Tensor<float>): local matmul B matrix. shape: [N, K_per_rank]
        output_dtype (torch.dtype): output data type
        rank (int): current rank
        num_ranks (int): total number of ranks
        scatter_bufs (List[torch.Tensor<float>]): A list of symm-tensors used for inter-rank allgather.
            Each tensor shape: [maxM, N]. Created by `create_gemm_rs_intra_node_context`.
        scatter_bufs_ptr (torch.Tensor<int64>): data_ptr of each tensor in scatter_fufs
        sync_bufs_ptr (torch.Tensor<int32>): A symm-tensor used for global synchronization.
            Shape: [num_ranks]. Created by `create_gemm_rs_intra_node_context`.
        fuse_scatter: whether fuse scatter into gemm
        transpose_weight: if transpose_weight, weight shape is [K, N], otherwise it is [N, K]

    Returns:
        result matrix C. shape [M_per_rank, N]
    """
    if not fuse_scatter:
        raise NotImplementedError()

    M, local_K = A.shape
    N, _ = B.shape
    N, weight_local_K = B.shape
    assert weight_local_K == local_K

    assert A.dtype == B.dtype
    M_per_rank = M // num_ranks
    current_stream = torch.cuda.current_stream()
    barrier_all_on_stream(rank, num_ranks, sync_bufs, current_stream)

    output = torch.empty((M_per_rank, N), dtype=output_dtype, device=A.device)
    matmul_fuse_scatter(A, B, scatter_bufs_ptr, rank, num_ranks, gemm_config=gemm_config)
    scatter_out = scatter_bufs[rank][:M]

    barrier_all_on_stream(rank, num_ranks, sync_bufs, current_stream)
    output = ring_reduce_after_scatter(rank, num_ranks, scatter_out, current_stream)
    return output


@dataclasses.dataclass
class GEMMReduceScatterTensorParallelContext:
    rank: int
    num_ranks: int
    scatter_bufs: List[torch.Tensor]
    scatter_bufs_ptr: torch.Tensor
    sync_bufs_ptr: torch.Tensor
    sync_bufs: List[torch.Tensor]
    output_dtype: torch.dtype
    fuse_scatter: bool


def create_gemm_rs_intra_node_context(max_M, N, output_dtype, rank, num_ranks, tp_group: torch.distributed.ProcessGroup,
                                      fuse_scatter=True):
    """create context for gemm reduce-scatter intra-node

    Args:
        max_M (int): max M
        N(int): N
        output_dtype(torch.dtype): dtype of output
        rank (int): current rank
        num_ranks (int): total number of ranks
        tp_group: tp process_group
        fuse_scatter: whether fuse scatter into gemm
    Returns:
        GEMMReduceScatterTensorParallelContext
    """
    if not fuse_scatter:
        raise NotImplementedError()

    sync_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node([num_ranks], torch.int32)
    sync_bufs[rank].fill_(0)
    sync_bufs_ptr = torch.tensor([t.data_ptr() for t in sync_bufs], device=torch.cuda.current_device(),
                                 requires_grad=False)

    scatter_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node([max_M, N], output_dtype)
    scatter_bufs_ptr = torch.tensor([t.data_ptr() for t in scatter_bufs], device=torch.cuda.current_device(),
                                    requires_grad=False)
    torch.cuda.synchronize()
    torch.distributed.barrier(tp_group)
    ret = GEMMReduceScatterTensorParallelContext(
        rank=rank,
        num_ranks=num_ranks,
        scatter_bufs=scatter_bufs,
        scatter_bufs_ptr=scatter_bufs_ptr,
        sync_bufs_ptr=sync_bufs_ptr,
        sync_bufs=sync_bufs,
        output_dtype=output_dtype,
        fuse_scatter=fuse_scatter,
    )
    return ret


def gemm_rs_intra_node(A: torch.Tensor, B: torch.Tensor, ctx: GEMMReduceScatterTensorParallelContext):
    """GEMM Reduce-Scatter for Intra-Node

    return C = reduce_scatter(A @ B.T)

    Args:
        A (torch.Tensor<bfloat16/float16>): local matmul A matrix. shape: [M, local_K]
        B (torch.Tensor<bfloat16/float16>): local matmul B matrix. shape: [N, local_K]
        ctx(GEMMReduceScatterTensorParallelContext): context

    Returns:
        c (torch.Tensor<bfloat16/float16>): local matmul C matrix. shape: [M // world_size, N]
    """

    C = gemm_rs_intra_node_op(A, B, ctx.output_dtype, ctx.rank, ctx.num_ranks, ctx.scatter_bufs, ctx.scatter_bufs_ptr,
                              ctx.sync_bufs_ptr, ctx.fuse_scatter)

    return C
