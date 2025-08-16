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
from triton.language.extra.cuda.language_extra import (
    st,
    ld,
)
from triton_dist.kernels.nvidia.common_ops import barrier_on_this_grid
from triton_dist.kernels.nvidia.moe_utils import calc_gather_scatter_index_triton
from triton_dist.utils import (
    NVSHMEM_SIGNAL_DTYPE,
    launch_cooperative_grid_options,
    nvshmem_barrier_all_on_stream,
    nvshmem_create_tensor,
    nvshmem_free_tensor_sync,
)


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


@triton.jit()
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


def run_moe_reduce_ar_triton_non_overlap(x: torch.Tensor, weights: torch.Tensor, chosen_experts: torch.Tensor,
                                         expert_weight: torch.Tensor, n_chunks=2,
                                         config: Optional[triton.Config] = None):
    M, K_per_rank = x.shape
    N = weights.shape[-1]
    num_experts = weights.shape[0]
    topk = chosen_experts.shape[1]
    ntokens = M // topk
    config = config or get_auto_triton_config(M, N, K_per_rank, n_chunks, x.dtype)
    block_size_m = config.kwargs["BLOCK_SIZE_M"]
    _, _, gather_index, expert_index, M_pad_gpu = calc_gather_scatter_index_triton(chosen_experts, num_experts,
                                                                                   block_size_m)
    grouped_gemm_out = torch.empty(
        (M, N),
        dtype=x.dtype,
        device=torch.cuda.current_device(),
    )
    M_pad_approx = (triton.cdiv(M, block_size_m) + num_experts) * block_size_m
    grid = lambda META: (triton.cdiv(M_pad_approx, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    moe_grouped_gemm_kernel[grid](x, weights, grouped_gemm_out, expert_weight, gather_index, expert_index, M_pad_gpu,
                                  N, x.shape[1], num_experts, x.stride(0), x.stride(1), weights.stride(0),
                                  weights.stride(1), weights.stride(2), grouped_gemm_out.stride(0),
                                  grouped_gemm_out.stride(1), topk, **config.all_kwargs())
    out_reduce_topk = torch.sum(grouped_gemm_out.reshape(ntokens, topk, N), dim=1, keepdim=False)
    torch.distributed.all_reduce(out_reduce_topk)
    return out_reduce_topk


@dataclass
class MoEReduceARContext:
    max_M: int
    N: int
    num_experts: int
    topk: int
    dtype: torch.dtype
    rank: int
    num_ranks: int
    num_local_ranks: int
    n_chunks_max: int

    # Overlap synchronization
    symm_reduce_buffer: torch.Tensor = field(init=False)
    symm_output_buffer: torch.Tensor = field(init=False)
    gemm_counter: torch.Tensor = field(init=False)
    gemm_done_flag: torch.Tensor = field(init=False)
    topk_done_flag: torch.Tensor = field(init=False)
    multimem_barrier: torch.Tensor = field(init=False)

    local_rank: int = field(init=False)
    nnodes: int = field(init=False)
    compute_stream: torch.cuda.Stream = field(default_factory=lambda: torch.cuda.current_stream())
    reduce_stream: torch.cuda.Stream = field(default_factory=lambda: torch.cuda.Stream(priority=-1))

    def __post_init__(self):
        assert self.dtype in [torch.bfloat16, torch.float16]
        assert self.max_M % self.topk == 0
        self.local_rank = self.rank % self.num_local_ranks
        self.nnodes = self.num_ranks // self.num_local_ranks

        ntokens = self.max_M // self.topk
        self.symm_reduce_buffer = nvshmem_create_tensor((ntokens, self.N), self.dtype)
        self.symm_output_buffer = nvshmem_create_tensor((ntokens, self.N), self.dtype)
        self.gemm_counter = torch.zeros((self.n_chunks_max, ), dtype=torch.int32, device=torch.cuda.current_device())
        self.gemm_done_flag = torch.zeros((self.n_chunks_max, ), dtype=torch.int32, device=torch.cuda.current_device())
        self.topk_done_flag = nvshmem_create_tensor((self.n_chunks_max, ), NVSHMEM_SIGNAL_DTYPE)
        self.multimem_barrier = torch.zeros((1, ), dtype=torch.int32, device=torch.cuda.current_device())
        self.grid_barrier = torch.zeros((1, ), dtype=torch.int32, device=torch.cuda.current_device())

        self.symm_reduce_buffer.zero_()
        self.symm_output_buffer.zero_()
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    def finalize(self):
        nvshmem_free_tensor_sync(self.symm_reduce_buffer)
        nvshmem_free_tensor_sync(self.symm_output_buffer)


def create_moe_ar_context(rank, world_size, local_world_size, max_token_num, hidden_dim, num_experts, topk, input_dtype,
                          n_chunks_max=8):
    return MoEReduceARContext(max_token_num, hidden_dim, num_experts, topk, dtype=input_dtype, rank=rank,
                              num_ranks=world_size, num_local_ranks=local_world_size, n_chunks_max=n_chunks_max)


@triton.jit
def moe_gather_ar_grouped_gemm_kernel(
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


def moe_gather_ar_grouped_gemm(
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

    moe_gather_ar_grouped_gemm_kernel[grid](
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


@triton.jit(do_not_specialize=["rank"])
def reduce_topk_allreduce_kernel(
    grouped_gemm_out_ptr,  # [ntokens*topk, N]
    symm_reduce_buffer,  # [ntokens, N] - local reduction buffer
    symm_output_buffer,  # [ntokens, N] - final output buffer
    ntokens,
    N,
    stride_m,
    stride_n,
    rank,
    num_ranks,
    gemm_done_flag_ptr,  # [N_CHUNKS]
    topk_done_ptr,  # [N_CHUNKS]
    grid_barrier_ptr,  # [1] - grid barrier
    N_CHUNKS,
    TOPK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    use_cooperative: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)
    N_per_chunk = N // N_CHUNKS
    N_per_chunk = tl.multiple_of(N_per_chunk, 16)
    blocks_n_per_chunk = tl.cdiv(N_per_chunk, BLOCK_SIZE_N)
    blocks_m = tl.cdiv(ntokens, BLOCK_SIZE_M)

    for n_chunk in tl.range(0, N_CHUNKS, step=1, loop_unroll_factor=1):
        # Wait for GEMM completion of this chunk
        token = dl.wait(gemm_done_flag_ptr + n_chunk, 1, scope="gpu", semantic="acquire", waitValue=1)
        offs_n_chunk = n_chunk * N_per_chunk * stride_n
        input_this_chunk_ptr = dl.consume_token(grouped_gemm_out_ptr + offs_n_chunk, token)
        reduced_topk_this_chunk_ptr = dl.consume_token(symm_reduce_buffer + offs_n_chunk, token)

        # Local topk reduction for this chunk
        for tile_id in range(pid, blocks_m * blocks_n_per_chunk, npid):
            tid_m = tile_id // blocks_n_per_chunk
            tid_n = tile_id % blocks_n_per_chunk

            offs_m = tid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            mask_m = offs_m < ntokens
            mask_n = offs_n < N_per_chunk

            reduced_vals = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(TOPK):
                expert_offset = offs_m * TOPK + k
                expert_vals = tl.load(input_this_chunk_ptr + expert_offset[:, None] * N + offs_n[None, :],
                                      mask=mask_m[:, None] & mask_n[None, :])
                reduced_vals += expert_vals
            local_offset = offs_m[:, None] * N + offs_n[None, :]
            tl.store(reduced_topk_this_chunk_ptr + local_offset, reduced_vals, mask=mask_m[:, None] & mask_n[None, :])

        if pid == 0:
            st(topk_done_ptr + n_chunk, 1, scope="sys", semantic="release")

            for peer in range(num_ranks):
                peer_topk_done_ptr = dl.symm_at(topk_done_ptr, peer)
                while ld(peer_topk_done_ptr + n_chunk, scope="sys", semantic="acquire") != 1:
                    pass

        barrier_on_this_grid(grid_barrier_ptr, use_cooperative=use_cooperative)
        for tile_id in range(pid, blocks_m * blocks_n_per_chunk, npid):
            tid_m = tile_id // blocks_n_per_chunk
            tid_n = tile_id % blocks_n_per_chunk

            offs_m = tid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask_m = offs_m < ntokens
            mask_n = offs_n < N_per_chunk
            mask = mask_m[:, None] & mask_n[None, :]

            offset = offs_m[:, None] * N + n_chunk * N_per_chunk + offs_n[None, :]

            final_vals = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=symm_reduce_buffer.dtype.element_ty)
            for r in range(num_ranks):
                peer_ptr = dl.symm_at(symm_reduce_buffer, r)
                peer_vals = tl.load(peer_ptr + offset, mask=mask, other=0.0)
                final_vals += peer_vals
            tl.store(symm_output_buffer + offset, final_vals, mask=mask)


def run_moe_reduce_ar(x: torch.Tensor, weights: torch.Tensor, chosen_experts: torch.Tensor,
                      expert_weights: torch.Tensor, ctx: MoEReduceARContext, n_chunks: int = 4,
                      config: Optional[triton.Config] = None) -> torch.Tensor:
    if n_chunks > ctx.n_chunks_max:
        warnings.warn(f"n_chunks({n_chunks}) must be <= ctx.n_chunks_max({ctx.n_chunks_max})")
        n_chunks = ctx.n_chunks_max

    M, K_per_rank = x.shape
    ntokens = M // ctx.topk

    assert M <= ctx.max_M, f"Input M ({M}) exceeds max_M ({ctx.max_M}) in MoEReduceARContext"
    assert chosen_experts.shape == (ntokens, ctx.topk)
    assert weights.shape == (
        ctx.num_experts, K_per_rank, ctx.N
    ), f"Weights shape {weights.shape} does not match expected shape ({ctx.num_experts}, {K_per_rank}, {ctx.N})"

    config = config or get_auto_triton_config(M, ctx.N, K_per_rank, n_chunks, x.dtype)
    block_size_m = config.kwargs["BLOCK_SIZE_M"]

    _, _, gather_index, expert_index, M_pad_gpu = calc_gather_scatter_index_triton(chosen_experts, ctx.num_experts,
                                                                                   block_size_m)

    grouped_gemm_out = torch.empty((M, ctx.N), dtype=ctx.dtype, device=torch.cuda.current_device())
    M_pad_approx = (triton.cdiv(M, block_size_m) + ctx.num_experts) * block_size_m

    ctx.gemm_counter.zero_()
    ctx.gemm_done_flag.zero_()
    ctx.topk_done_flag.zero_()
    nvshmem_barrier_all_on_stream(ctx.compute_stream)
    ctx.reduce_stream.wait_stream(ctx.compute_stream)

    moe_gather_ar_grouped_gemm(x, weights, grouped_gemm_out, expert_weights, gather_index, expert_index, M_pad_gpu,
                               M_pad_approx, ctx.N, K_per_rank, ctx.num_experts, ctx.topk, ctx.gemm_counter,
                               ctx.gemm_done_flag, config)
    with torch.cuda.stream(ctx.reduce_stream):
        N_per_chunk = ctx.N // n_chunks
        block_size_m = max(1, 16 * 1024 // N_per_chunk // x.itemsize)
        block_size_n = N_per_chunk
        reduce_topk_allreduce_kernel[(32, )](grouped_gemm_out, ctx.symm_reduce_buffer, ctx.symm_output_buffer, ntokens,
                                             ctx.N, ctx.N, 1, ctx.rank, ctx.num_ranks, ctx.gemm_done_flag,
                                             ctx.topk_done_flag, ctx.grid_barrier, n_chunks, TOPK=ctx.topk,
                                             BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n, use_cooperative=True,
                                             **launch_cooperative_grid_options())

    ctx.compute_stream.wait_stream(ctx.reduce_stream)
    nvshmem_barrier_all_on_stream(ctx.compute_stream)
    return ctx.symm_output_buffer.reshape(-1)[:ntokens * ctx.N].reshape(ntokens, ctx.N)  # Reshape to (ntokens, N)
