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
from dataclasses import dataclass, field
from typing import Optional
import warnings

import torch

import triton
import triton.language as tl
import triton_dist.language as dl
from triton.language.extra.cuda.language_extra import __syncthreads, atomic_add, tid
from triton_dist.kernels.nvidia.common_ops import barrier_on_this_grid
from triton_dist.language.extra import libshmem_device
from triton_dist.kernels.nvidia.moe_utils import calc_gather_scatter_index_triton, reduce_topk_kernel
from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE, has_fullmesh_nvlink, launch_cooperative_grid_options, nvshmem_barrier_all_on_stream, nvshmem_create_tensor, nvshmem_free_tensor_sync


@dataclass
class MoEReduceRSContext:
    max_M: int
    N: int
    num_experts: int
    topk: int
    dtype: torch.dtype
    # distributed arguments
    rank: int
    num_ranks: int
    num_local_ranks: int
    n_chunks_max: int
    # barriers
    grid_barrier: torch.Tensor
    gemm_counter: torch.Tensor
    gemm_done_flag: torch.Tensor
    rs_counter: torch.Tensor

    # symmetric buffers or non-symmetric buffers
    symm_barrier: torch.Tensor = field(init=False)
    symm_reduce_scatter_buffer: torch.Tensor = field(init=False)

    local_rank: int = field(init=False)
    nnodes: int = field(init=False)
    # stream
    reduce_stream: torch.cuda.Stream = field(default_factory=lambda: torch.cuda.Stream(priority=-1))

    def __post_init__(self):
        assert self.dtype in [torch.bfloat16, torch.float16], "Currently only used for float16 or bfloat16"
        assert self.max_M % self.topk == 0, "M must be divisible by topk"
        self.local_rank = self.rank % self.num_local_ranks
        self.nnodes = self.num_ranks // self.num_local_ranks

        # Create a barrier for grid synchronization
        self.symm_barrier = nvshmem_create_tensor((self.n_chunks_max * self.num_ranks, ), NVSHMEM_SIGNAL_DTYPE)
        self.symm_barrier.zero_()
        ntokens = self.max_M // self.topk
        self.symm_reduce_scatter_buffer = nvshmem_create_tensor((ntokens, self.N), self.dtype)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    def finalize(self):
        nvshmem_free_tensor_sync(self.symm_barrier)
        nvshmem_free_tensor_sync(self.symm_reduce_scatter_buffer)


def create_moe_rs_context(rank, world_size, local_world_size, max_token_num, hidden_dim, num_experts, topk, input_dtype,
                          n_chunks_max=8):
    """
    Creates and initializes a context object for the MoE Reduce-Scatter operation.

    This context holds pre-allocated buffers (including symmetric nvshmem tensors)
    and synchronization primitives required by the kernels. It should be created
    once and reused for multiple kernel executions.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes in the group.
        local_world_size (int): The number of processes on the local node.
        max_token_num (int): The maximum number of tokens to be processed.
        hidden_dim (int): The hidden dimension size (N in GEMM).
        num_experts (int): The total number of experts.
        topk (int): The number of experts to route each token to.
        input_dtype (torch.dtype): The data type of the input tensors (e.g., torch.float16).
        n_chunks_max (int): The maximum number of chunks the N dimension can be split into.
                            This determines the size of pre-allocated synchronization buffers.

    Returns:
        MoEReduceRSContext: An initialized context object.
    """
    device = torch.cuda.current_device()
    grid_barrier = torch.zeros((1, ), dtype=torch.int32, device=device)
    gemm_counter = torch.zeros((n_chunks_max, ), dtype=torch.int32, device=device)
    gemm_done_flag = torch.zeros((n_chunks_max, ), dtype=torch.int32, device=device)
    rs_counter = torch.zeros((n_chunks_max * world_size, ), dtype=torch.int32, device=device)
    return MoEReduceRSContext(max_token_num, hidden_dim, num_experts, topk, dtype=input_dtype, rank=rank,
                              num_ranks=world_size, num_local_ranks=local_world_size, n_chunks_max=n_chunks_max,
                              grid_barrier=grid_barrier, gemm_counter=gemm_counter, gemm_done_flag=gemm_done_flag,
                              rs_counter=rs_counter)


@triton.jit
def swizzle_2d_by_group_n(pid, nblocks_m, nblocks_n, GROUP_SIZE_N: tl.constexpr):
    """ if we choose tile first in N within group_size_N, maybe each group with N = 1024, for BLOCK_SIZE_N = 64, then 16 tiles per tiled_m.
    maybe too much for L20. but never mind. maybe we can fix it later.
    """
    nblocks_per_group = GROUP_SIZE_N * nblocks_m
    group_id = pid // nblocks_per_group
    remainder = pid - group_id * nblocks_per_group
    pid_m = remainder // GROUP_SIZE_N
    pid_n = remainder % GROUP_SIZE_N + group_id * GROUP_SIZE_N
    return pid_m, pid_n, group_id


def _kernel_repr(proxy):
    constexprs = proxy.constants
    cap_major, cap_minor = torch.cuda.get_device_capability()
    a_dtype = proxy.signature["A_ptr"].lstrip("*")
    b_dtype = proxy.signature["B_ptr"].lstrip("*")
    c_dtype = proxy.signature["C_ptr"].lstrip("*")
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

    return f"triton3x_sm{cap_major}{cap_minor}_grouped_gemm_moe_rs_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_{a_trans}{b_trans}{c_trans}"


@triton.jit(repr=_kernel_repr)
def moe_gather_rs_grouped_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    A_scale_ptr,
    gather_index_ptr,
    expert_index_ptr,
    M_ptr,
    N,
    K,
    E,  # not used
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    counter_ptr,
    barrier_ptr,
    TOPK: tl.constexpr,  # not used
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    M = tl.load(M_ptr)

    num_block_m = tl.cdiv(M, BLOCK_SIZE_M)
    thread_idx = tid(0)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    if pid >= num_block_m * num_block_n:
        return

    pid_m, pid_n, group_id = swizzle_2d_by_group_n(pid, num_block_m, num_block_n, GROUP_SIZE_N)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_gather_a = tl.load(gather_index_ptr + offs_m)
    token_mask = offs_gather_a < M

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + offs_gather_a[:, None] * stride_am + offs_k[None, :] * stride_ak

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_be = tl.load(expert_index_ptr + pid_m)
    b_ptrs = B_ptr + offs_be * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    if A_ptr.dtype.element_ty == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
        tl.static_assert(False, "int8 is not supported in this kernel, please use float16 or bfloat16")
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K))

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + offs_gather_a[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if A_scale_ptr:
        accumulator = accumulator * tl.load(A_scale_ptr + offs_gather_a[:, None], mask=token_mask[:, None])
    accumulator = accumulator.to(A_ptr.dtype.element_ty)
    tl.store(c_ptrs, accumulator, mask=c_mask)

    thread_idx = tid(axis=0)
    __syncthreads()
    if thread_idx == 0:
        count = atomic_add(counter_ptr + group_id, 1, semantic="release", scope="gpu")
        if count == num_block_m * GROUP_SIZE_N - 1:
            atomic_add(barrier_ptr + group_id, 1, semantic="release", scope="sys")
            tl.store(counter_ptr + group_id, 0)  # reset counter


@triton.jit(repr=_kernel_repr)
def moe_grouped_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    A_scale_ptr,
    gather_index_ptr,
    expert_index_ptr,
    M_ptr,
    N,
    K,
    E,  # not used
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    TOPK: tl.constexpr,  # not used
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    M = tl.load(M_ptr)

    num_block_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    if pid >= num_block_m * num_block_n:
        return

    pid_m, pid_n, group_id = swizzle_2d_by_group_n(pid, num_block_m, num_block_n, GROUP_SIZE_N)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_gather_a = tl.load(gather_index_ptr + offs_m)
    token_mask = offs_gather_a < M

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + offs_gather_a[:, None] * stride_am + offs_k[None, :] * stride_ak

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_be = tl.load(expert_index_ptr + pid_m)
    b_ptrs = B_ptr + offs_be * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    if A_ptr.dtype.element_ty == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
        tl.static_assert(False, "int8 is not supported in this kernel, please use float16 or bfloat16")
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K))

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + offs_gather_a[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if A_scale_ptr:
        accumulator = accumulator * tl.load(A_scale_ptr + offs_gather_a[:, None], mask=token_mask[:, None])
    accumulator = accumulator.to(A_ptr.dtype.element_ty)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def moe_gather_rs_grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    gather_a_index: torch.Tensor,
    expert_id: torch.Tensor,
    M_pad: torch.Tensor,
    M_pad_approx: int,  # make sure M_pad_approx >= int(M_pad)
    N: int,
    K: int,
    E: int,
    topk: int,
    tile_counter: torch.Tensor,
    barrier: torch.Tensor,
    config: triton.Config,
):
    grid = lambda META: (triton.cdiv(M_pad_approx, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

    moe_gather_rs_grouped_gemm_kernel[grid](
        A,
        B,
        C,
        A_scale,
        gather_a_index,
        expert_id,
        M_pad,  # torch.Tensor on GPU
        N,
        K,
        E,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        tile_counter,
        barrier,
        TOPK=topk,
        **config.all_kwargs(),
    )
    return C


def moe_grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    gather_a_index: torch.Tensor,
    expert_id: torch.Tensor,
    M_pad: torch.Tensor,
    M_pad_approx: int,  # make sure M_pad_approx >= int(M_pad)
    N: int,
    K: int,
    E: int,
    topk: int,
    config: triton.Config,
):
    grid = lambda META: (triton.cdiv(M_pad_approx, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

    moe_grouped_gemm_kernel[grid](
        A,
        B,
        C,
        A_scale if A_scale is not None else None,  # can be None
        gather_a_index,
        expert_id,
        M_pad,  # torch.Tensor on GPU
        N,
        K,
        E,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        TOPK=topk,
        **config.all_kwargs(),
    )
    return C


@triton.jit(do_not_specialize=["rank"])
def reduce_topk_reduce_scatter_a2a_intra_node_kernel(
    input_ptr,  # of shape [ntokens * topk, N] with stride [stride_m, stride_n]
    expert_weight_ptr,  # not used actually. accumulate in Grouped GEMM
    # output
    symm_reduced_topk_ptr,  # of shape [ntokens, N]. symm_buffer = sum(input, axis=1)
    output_ptr,  # of shape [ntokens // num_ranks, N]. output = reduce_scatter(symm_buffer)
    # args
    ntokens,
    N,
    stride_m,
    stride_n,
    rank,
    num_ranks,
    # some barriers
    gemm_done_flag_ptr,
    grid_barrier_ptr,
    N_CHUNKS,
    TOPK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    use_cooperative: tl.constexpr,
):
    """
    output = reduce_scatter(symm_buffer)
    """
    pid = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)
    N_per_chunk = N // N_CHUNKS
    N_per_chunk = tl.multiple_of(N_per_chunk, 16)
    ntokens_per_rank = ntokens // num_ranks
    for n_chunk in tl.range(0, N_CHUNKS, step=1, loop_unroll_factor=1):
        token = dl.wait(gemm_done_flag_ptr + n_chunk, 1, scope="gpu", semantic="acquire", waitValue=1)
        offs_n_chunk = n_chunk * N_per_chunk * stride_n
        input_this_chunk_ptr = dl.consume_token(input_ptr + offs_n_chunk, token)
        reduced_topk_this_chunk_ptr = dl.consume_token(symm_reduced_topk_ptr + offs_n_chunk, token)

        for rid in range(0, num_ranks):
            peer = (rank + rid) % num_ranks
            out_ptr = dl.symm_at(reduced_topk_this_chunk_ptr, peer)
            out_ptr = tl.multiple_of(out_ptr, 16)
            reduce_topk_kernel(
                input_this_chunk_ptr + peer * ntokens_per_rank * TOPK * stride_m,
                expert_weight_ptr,
                0,  # no scale
                out_ptr + rank * ntokens_per_rank * stride_m,
                ntokens_per_rank,
                N_per_chunk,
                stride_m,
                stride_n,
                TOPK,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )

    # for intra-node, you may replace this with reduce flag to avoid nvshmem_barrier_block. which may cost a lot of registers and a little higher latency
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)
    if pid == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    # reduce symm_buffer_n_ptr to output_n_ptr
    blocks_n_per_chunk = tl.cdiv(N_per_chunk, BLOCK_SIZE_N)
    blocks_m_per_rank = tl.cdiv(ntokens_per_rank, BLOCK_SIZE_M)
    nblocks_per_chunk_per_rank = blocks_m_per_rank * blocks_n_per_chunk
    # want to save some registers, but loop_unroll_factor does not work here
    for n_chunk in tl.range(0, N_CHUNKS, step=1, loop_unroll_factor=1):
        offs_n_chunk = n_chunk * N_per_chunk * stride_n
        output_n_ptr = output_ptr + offs_n_chunk
        reduced_topk_this_chunk_ptr = symm_reduced_topk_ptr + offs_n_chunk
        for tile_id in range(pid, nblocks_per_chunk_per_rank, npid):
            tid_m = tile_id // blocks_n_per_chunk
            tid_n = tile_id % blocks_n_per_chunk
            offs_m = tid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask_m = offs_m < ntokens_per_rank
            mask_n = offs_n < N_per_chunk
            offs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            mask = mask_m[:, None] & mask_n[None, :]
            val = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 0, input_ptr.dtype.element_ty)
            for sid in range(0, num_ranks):
                segment = (rank + sid) % num_ranks
                offs_segment = segment * ntokens_per_rank * stride_m
                val += tl.load(reduced_topk_this_chunk_ptr + offs_segment + offs, mask=mask)
            tl.store(output_n_ptr + offs, val, mask=mask)


@triton.jit(do_not_specialize=["rank"])
def reduce_topk_reduce_scatter_ring_intra_node_kernel(
    input_ptr,  # of shape [ntokens * topk, N] with stride [stride_m, stride_n]
    expert_weight_ptr,  # not used actually. accumulate in Grouped GEMM
    symm_reduced_topk_ptr,  # of shape [ntokens, N]. symm_buffer = sum(input, axis=1)
    # output
    output_ptr,  # of shape [ntokens // num_ranks, N]. output = reduce_scatter(symm_buffer)
    # args
    ntokens,
    N,
    stride_m,
    stride_n,
    rank,
    num_ranks,
    # some barriers
    gemm_done_signal_ptr,
    symm_signal_ptr,
    counter_ptr,
    N_CHUNKS,  # N_CHUNKS is used to distinguish different groups
    TOPK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    output = reduce_scatter(symm_buffer)

    symm_signal should have at least N_CHUNKS * (1 + 2 * num_ranks) elems. per-chunk signals:
    * grouped gemm out signal: 1 per chunk (no need to be symmetric)
    * reduce scatter counter: num_ranks per chunk (no need to be symmetric)
    * reduce scatter ready signal: num_ranks per chunk (has to be symmetric)
    """
    npid = tl.num_programs(axis=0)
    thread_idx = tid(0)
    N_per_chunk = N // N_CHUNKS
    N_per_chunk = tl.multiple_of(N_per_chunk, 16)
    ntokens_per_rank = ntokens // num_ranks
    # always send to next
    peer = (rank + 1) % num_ranks
    peer_reduced_topk_ptr = dl.symm_at(symm_reduced_topk_ptr, peer)
    peer_reduced_topk_ptr = tl.multiple_of(peer_reduced_topk_ptr, 16)

    for n_chunk in tl.range(0, N_CHUNKS, step=1, loop_unroll_factor=1):
        gemm_done_signal_this_chunk_ptr = gemm_done_signal_ptr + n_chunk
        counter_this_chunk_ptr = counter_ptr + num_ranks * n_chunk
        symm_signal_this_chunk_ptr = symm_signal_ptr + num_ranks * n_chunk
        token = dl.wait(gemm_done_signal_this_chunk_ptr, 1, scope="gpu", semantic="acquire", waitValue=1)

        offs_n_by_chunk = n_chunk * N_per_chunk * stride_n
        input_this_chunk_ptr = dl.consume_token(input_ptr, token) + offs_n_by_chunk
        reduced_topk_this_chunk_ptr = dl.consume_token(symm_reduced_topk_ptr, token) + offs_n_by_chunk
        output_this_chunk_ptr = output_ptr + offs_n_by_chunk

        for rid in range(0, num_ranks):
            segment = (rank - rid - 1 + num_ranks) % num_ranks
            offs_m_by_segment = segment * ntokens_per_rank * stride_m

            if rid == num_ranks - 1:
                out_this_chunk_ptr = output_this_chunk_ptr  # directy to output.
            else:
                out_this_chunk_ptr = peer_reduced_topk_ptr + offs_n_by_chunk + offs_m_by_segment

            # wait from peer
            if rid != 0:
                if thread_idx == 0:
                    libshmem_device.signal_wait_until(symm_signal_this_chunk_ptr + segment,
                                                      libshmem_device.NVSHMEM_CMP_EQ, 1)
                __syncthreads()

            if rid == 0:
                reduce_topk_kernel(
                    input_this_chunk_ptr + offs_m_by_segment * TOPK,
                    expert_weight_ptr,
                    0,  # bias
                    out_this_chunk_ptr,
                    ntokens_per_rank,
                    N_per_chunk,
                    stride_m,
                    stride_n,
                    TOPK,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                )
            else:  # receive data from peer
                bias_ptr = reduced_topk_this_chunk_ptr + offs_m_by_segment
                bias_ptr = tl.multiple_of(bias_ptr, 16)
                # reduce to peer
                reduce_topk_kernel(
                    input_this_chunk_ptr + offs_m_by_segment * TOPK,
                    expert_weight_ptr,
                    bias_ptr,  # bias
                    out_this_chunk_ptr,
                    ntokens_per_rank,
                    N_per_chunk,
                    stride_m,
                    stride_n,
                    TOPK,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                )

            # notify peer: warp specialization helps a lot here.
            __syncthreads()
            if thread_idx == 0:
                value = atomic_add(counter_this_chunk_ptr + segment, 1, scope="sys", semantic="release")
                if value == npid - 1:  # all done.
                    libshmem_device.signal_op(symm_signal_this_chunk_ptr + segment, 1,
                                              libshmem_device.NVSHMEM_SIGNAL_SET, peer)
            __syncthreads()


def reduce_topk_reduce_scatter_a2a_intra_node(grouped_gemm_out: torch.Tensor, ctx: MoEReduceRSContext, ntokens,
                                              n_chunks: int, out: torch.Tensor, BLOCK_SIZE_M, BLOCK_SIZE_N):
    has_nvlink_fullmesh = has_fullmesh_nvlink()
    if not has_nvlink_fullmesh:
        warnings.warn(
            "reduce_topk_reduce_scatter_a2a_intra_node only works well on fullmesh nvlink. for PCI-e machines, try reduce_topk_reduce_scatter_ring_intra_node instead"
        )
    reduce_topk_reduce_scatter_a2a_intra_node_kernel[(32, )](
        grouped_gemm_out,
        None,  # group weight
        ctx.symm_reduce_scatter_buffer,
        out,
        ntokens,
        ctx.N,
        ctx.N,  # stride_m
        1,  # stride_n
        ctx.rank,
        ctx.num_ranks,
        ctx.gemm_done_flag,
        ctx.grid_barrier,
        TOPK=ctx.topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,  #max(1, 16 * 1024 // N_per_chunk // x.itemsize),  # each thread with a uint4 load
        BLOCK_SIZE_N=BLOCK_SIZE_N,  #N_per_chunk,
        N_CHUNKS=n_chunks,
        num_warps=32,
        use_cooperative=True,
        **launch_cooperative_grid_options(),
    )
    return out


def reduce_topk_reduce_scatter_ring_intra_node(grouped_gemm_out: torch.Tensor, ctx: MoEReduceRSContext, ntokens,
                                               n_chunks: int, out: torch.Tensor, BLOCK_SIZE_M, BLOCK_SIZE_N):
    reduce_topk_reduce_scatter_ring_intra_node_kernel[(6, )](
        grouped_gemm_out,
        None,  # group weight
        ctx.symm_reduce_scatter_buffer,
        out,
        ntokens,
        ctx.N,
        ctx.N,  # stride_m
        1,  # stride_n
        ctx.rank,
        ctx.num_ranks,
        ctx.gemm_done_flag,
        ctx.symm_barrier,
        ctx.rs_counter,
        TOPK=ctx.topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,  #max(1, 16 * 1024 // N_per_chunk // x.itemsize),  # each thread with a uint4 load
        BLOCK_SIZE_N=BLOCK_SIZE_N,  #N_per_chunk,
        N_CHUNKS=n_chunks,
        num_warps=32,
        **launch_cooperative_grid_options(),
    )
    return out


def reduce_topk_reduce_scatter_intra_node(grouped_gemm_out: torch.Tensor, ctx: MoEReduceRSContext, ntokens,
                                          n_chunks: int, out: torch.Tensor, BLOCK_SIZE_M, BLOCK_SIZE_N):
    has_nvlink_fullmesh = has_fullmesh_nvlink()
    func = reduce_topk_reduce_scatter_a2a_intra_node if has_nvlink_fullmesh else reduce_topk_reduce_scatter_ring_intra_node
    return func(grouped_gemm_out, ctx, ntokens, n_chunks, out, BLOCK_SIZE_M, BLOCK_SIZE_N)


def get_auto_triton_config(M, N, K, N_CHUNKS, dtype: torch.dtype) -> triton.Config:
    N_per_chunk = N // N_CHUNKS
    # TODO(houqi.1993) may relax this check
    assert N_per_chunk == triton.next_power_of_2(N_per_chunk), f"N_per_chunk({N_per_chunk}) must be power of 2"
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    # TODO(houqi.1993) maybe fill some GEMM-pruned configs
    config = triton.Config(
        kwargs={
            "BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N, "BLOCK_SIZE_K": BLOCK_SIZE_K, "GROUP_SIZE_N":
            N_per_chunk // BLOCK_SIZE_N
        })
    return config


def run_moe_reduce_rs_triton_non_overlap(x: torch.Tensor, weights: torch.Tensor, chosen_experts: torch.Tensor,
                                         expert_weight: torch.Tensor, n_chunks=2,
                                         config: Optional[triton.Config] = None):
    M, K_per_rank = x.shape
    N = weights.shape[-1]
    num_experts = weights.shape[0]
    topk = chosen_experts.shape[1]
    ntokens = M // topk
    ntokens_per_rank = ntokens // torch.distributed.get_world_size()
    config = config or get_auto_triton_config(M, N, K_per_rank, n_chunks, x.dtype)
    block_size_m = config.kwargs["BLOCK_SIZE_M"]
    _, _, gather_index, expert_index, M_pad_gpu = calc_gather_scatter_index_triton(chosen_experts, num_experts,
                                                                                   block_size_m)
    grouped_gemm_out = torch.empty(
        (M, N),
        dtype=x.dtype,
        device=torch.cuda.current_device(),
    )
    out = torch.empty((ntokens_per_rank, N), dtype=x.dtype, device=torch.cuda.current_device())
    M_pad_approx = (triton.cdiv(M, block_size_m) + num_experts) * block_size_m
    grid = lambda META: (triton.cdiv(M_pad_approx, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    moe_grouped_gemm_kernel[grid](x, weights, grouped_gemm_out, expert_weight, gather_index, expert_index, M_pad_gpu,
                                  N, x.shape[1], num_experts, x.stride(0), x.stride(1), weights.stride(0),
                                  weights.stride(1), weights.stride(2), grouped_gemm_out.stride(0),
                                  grouped_gemm_out.stride(1), topk, **config.all_kwargs())
    out_reduce_topk = torch.sum(grouped_gemm_out.reshape(ntokens, topk, N), dim=1, keepdim=False)
    torch.distributed.reduce_scatter_tensor(out, out_reduce_topk)
    return out


def run_moe_reduce_rs(
        # input
        x: torch.Tensor, weights: torch.Tensor, chosen_experts: torch.Tensor, expert_weight: torch.Tensor,
        ctx: MoEReduceRSContext, n_chunks=2, config: Optional[triton.Config] = None):
    """
    Runs a fused MoE computation with a reduce-scatter collective.

    This function orchestrates a complex sequence of operations:
    1. A grouped GEMM computes expert outputs, gathering inputs based on routing decisions.
    2. A custom reduce-scatter kernel sums the `topk` expert outputs for each token
       and then reduces the results across ranks, scattering the final output.
    This is performed on separate CUDA streams to overlap the GEMM computation
    with the reduce-scatter collective.

    Args:
        x (torch.Tensor): Input tensor of shape (M, K_per_rank), where M is `num_tokens * topk`.
        weights (torch.Tensor): Expert weights of shape (E, K_per_rank, N).
        chosen_experts (torch.Tensor): Routing decisions of shape (num_tokens, topk).
        expert_weight (torch.Tensor): Scaling factors for expert outputs.
        ctx (MoEReduceRSContext): The pre-initialized context object.
        n_chunks (int): n_chunks (int): The number of chunks to split the N dim into for pipelining the GEMM
                        and reduce-scatter operations. A larger value can improve the overlap
                        of computation and communication but may lead to smaller, less efficient
                        GEMMs. This value must be a power of two and cannot exceed `ctx.n_chunks_max`.
        config (triton.Config, optional): A Triton config for the GEMM kernel.
                                          If None, a default is chosen.

    Returns:
        torch.Tensor: The final output tensor of shape (ntokens_per_rank, N).
    """
    if n_chunks > ctx.n_chunks_max:
        warnings.warn(f"n_chunks({n_chunks}) must be less than or equal to ctx.n_chunks_max({ctx.n_chunks_max})")
        n_chunks = ctx.n_chunks_max

    assert x.ndim == 2 and x.is_cuda, "Input x must be a 2D CUDA tensor"
    assert weights.ndim == 3 and weights.is_cuda, "Weights must be a 3D CUDA tensor"
    assert chosen_experts.ndim == 2 and chosen_experts.is_cuda, "chosen experts must be a 2D CUDA tensor"
    M, K_per_rank = x.shape
    assert M <= ctx.max_M, f"Input M({M}) must be less than or equal to context M({ctx.max_M})"
    assert M % ctx.topk == 0, f"M({M}) must be divisible by topk({ctx.topk})"

    ntokens = M // ctx.topk
    assert chosen_experts.shape == (ntokens, ctx.topk), \
        f"chosen experts shape {chosen_experts.shape} must match (ntokens({ntokens}), topk({ctx.topk}))"

    assert weights.shape == (ctx.num_experts, K_per_rank, ctx.N), \
        f"Weights shape {weights.shape} must match context num_experts({ctx.num_experts}), K_per_rank({K_per_rank}), and N({ctx.N})"
    assert x.dtype == weights.dtype == ctx.dtype, f"Input x dtype({x.dtype}) must match context dtype({ctx.dtype}) and weights dtype({weights.dtype})"

    current_stream = torch.cuda.current_stream()
    ntokens_per_rank = ntokens // ctx.num_ranks
    N_per_chunk = ctx.N // n_chunks
    config = config or get_auto_triton_config(M, ctx.N, K_per_rank, n_chunks, x.dtype)
    block_size_m = config.kwargs["BLOCK_SIZE_M"]
    # TODO(houqi.1993) maybe we can pass this as argument and avoid recompute it again
    _, _, gather_index, expert_index, M_pad_gpu = calc_gather_scatter_index_triton(chosen_experts, ctx.num_experts,
                                                                                   block_size_m)

    grouped_gemm_out = torch.empty(
        (M, ctx.N),
        dtype=ctx.dtype,
        device=torch.cuda.current_device(),
    )
    out = torch.empty((ntokens_per_rank, ctx.N), dtype=ctx.dtype, device=torch.cuda.current_device())

    M_pad_approx = (triton.cdiv(M, block_size_m) + ctx.num_experts) * block_size_m

    # TODO(houqi.1993) move this to reduce_topk_reduce_scatter to hide latency
    ctx.symm_barrier.zero_()  # set to target value
    ctx.gemm_done_flag.zero_()
    ctx.rs_counter.zero_()
    nvshmem_barrier_all_on_stream(current_stream)

    ctx.reduce_stream.wait_stream(current_stream)
    moe_gather_rs_grouped_gemm(x, weights, grouped_gemm_out, expert_weight, gather_index, expert_index, M_pad_gpu,
                               M_pad_approx, ctx.N, K_per_rank, ctx.num_experts, ctx.topk, ctx.gemm_counter,
                               ctx.gemm_done_flag, config)

    # nvshmem_barrier_all_on_stream(current_stream)
    # ctx.reduce_stream.wait_stream(current_stream)

    with torch.cuda.stream(ctx.reduce_stream):
        block_size_m = max(1, 16 * 1024 // N_per_chunk // x.itemsize)  # each thread with a uint4 load
        block_size_n = N_per_chunk
        reduce_topk_reduce_scatter_intra_node(grouped_gemm_out, ctx, ntokens, n_chunks, out, block_size_m, block_size_n)

    current_stream.wait_stream(ctx.reduce_stream)
    return out
