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
from typing import Optional
import triton
import triton.language as tl
import torch
from triton_dist.kernels.nvidia.common_ops import next_power_of_2, bisect_right_kernel
from triton.language.extra.cuda.language_extra import __syncthreads


def calc_gather_index_torch(chosen_experts: torch.Tensor, stable=True):
    assert chosen_experts.ndim == 2 and chosen_experts.is_cuda
    ntokens, topk = chosen_experts.shape
    _, index_choosed_experts = chosen_experts.flatten().sort(stable=stable)
    gather_index = index_choosed_experts.to(torch.int32) // topk
    topk_index = torch.arange(0, topk, dtype=torch.int32, device="cuda").repeat(ntokens)[index_choosed_experts]
    return gather_index, topk_index


def calc_gather_index_from_scatter_index(
    scatter_index: torch.Tensor,
    row_start: int,
    row_end: int,
    BLOCK_SIZE: int = 1024,
):

    @triton.jit
    def calc_gather_index_from_scatter_index_kernel(
        scatter_index: torch.Tensor,
        # topk
        gather_index: torch.Tensor,
        topk_index: torch.Tensor,
        ntokens: int,
        topk: int,
        row_start: int,
        row_end: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < ntokens * topk
        scatter_idx = tl.load(scatter_index + offset, mask=mask, other=-1)
        token_idx = offset // topk
        topk_idx = offset % topk
        token_idx_mask = (scatter_idx >= row_start) & (scatter_idx < row_end)
        tl.store(gather_index + scatter_idx - row_start, token_idx, mask=token_idx_mask)
        tl.store(topk_index + scatter_idx - row_start, topk_idx, mask=token_idx_mask)

    assert scatter_index.is_cuda and scatter_index.ndim == 2
    ntokens, topk = scatter_index.shape
    gather_index = torch.zeros(row_end - row_start, dtype=torch.int32, device=scatter_index.device)
    topk_index = torch.zeros(row_end - row_start, dtype=torch.int32, device=scatter_index.device)
    grid = lambda META: (triton.cdiv(ntokens * topk, META["BLOCK_SIZE"]), )
    calc_gather_index_from_scatter_index_kernel[grid](
        scatter_index,
        gather_index,
        topk_index,
        ntokens,
        topk,
        row_start,
        row_end,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=min(32, BLOCK_SIZE // 32),
    )
    return gather_index, topk_index


def histogram_by_expert_torch(chosen_experts: torch.Tensor, nexperts: int):
    assert chosen_experts.is_cuda and chosen_experts.ndim == 2
    return torch.bincount(chosen_experts.flatten(), minlength=nexperts).to(torch.int32)


@triton.jit
def histogram_block_kernel(values_ptr, N, BLOCK_SIZE: tl.constexpr, NUM_BINS: tl.constexpr):
    # assert only 1 CTA
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    out = tl.zeros((NUM_BINS, ), dtype=tl.int32)
    for n in range(pid, num_blocks):
        offs = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        expert_idx = tl.load(values_ptr + offs, mask=offs < N, other=NUM_BINS)
        res = tl.histogram(expert_idx, NUM_BINS, mask=offs < N)
        out = out + res
    return out


def histogram_by_expert_triton(chosen_experts: torch.Tensor, nexperts: int):
    assert chosen_experts.is_cuda and chosen_experts.ndim == 2

    @triton.jit
    def _kernel(
        choosed_experts_ptr,
        ntokens_by_expert_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
        NEXPERTS: tl.constexpr,
    ):

        NEXPERTS_NEXT_POW_OF_2: tl.constexpr = next_power_of_2(NEXPERTS)
        ntokens_by_expert = histogram_block_kernel(choosed_experts_ptr, N, BLOCK_SIZE, NEXPERTS)
        offs = tl.arange(0, NEXPERTS_NEXT_POW_OF_2)
        mask = offs < NEXPERTS
        tl.store(ntokens_by_expert_ptr + offs, ntokens_by_expert, mask=mask)

    ntokens_by_expert = torch.zeros(nexperts, dtype=torch.int32, device=chosen_experts.device)
    _kernel[(1, )](
        chosen_experts,
        ntokens_by_expert,
        chosen_experts.numel(),
        BLOCK_SIZE=1024,
        NEXPERTS=nexperts,
        num_warps=32,
    )
    return ntokens_by_expert


def calc_scatter_index_torch(chosen_experts, stable=True):
    return (chosen_experts.flatten().argsort(stable=stable).argsort().int().view(chosen_experts.shape))


@triton.jit
def calc_gather_scatter_index_kernel(
    choosed_experts_ptr,
    # output
    ntokens_by_expert_ptr,
    gather_index_ptr,
    scatter_index_ptr,
    expert_index_ptr,
    M_pad_ptr,
    # workspace
    workspace_ptr,  # int32 of ntokens
    # args
    ntokens,
    topk,
    NEXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ALIGNMENT_BY_EXPERT: tl.constexpr,
):
    NEXPERTS_NEXT_POW_OF_2: tl.constexpr = next_power_of_2(NEXPERTS)
    offs_by_expert = tl.arange(0, NEXPERTS_NEXT_POW_OF_2)
    mask_by_expert = offs_by_expert < NEXPERTS
    # if ntokens_by_expert_ptr:
    #     ntokens_by_expert = tl.load(ntokens_by_expert_ptr + offs_by_expert, mask=mask_by_expert, other=0)
    # else:
    ntokens_by_expert = histogram_block_kernel(choosed_experts_ptr, ntokens * topk, BLOCK_SIZE, NEXPERTS)
    tl.store(ntokens_by_expert_ptr + offs_by_expert, ntokens_by_expert, mask=mask_by_expert)

    ntiles_by_expert = tl.cdiv(ntokens_by_expert, ALIGNMENT_BY_EXPERT)
    ntokens_by_expert_pad = ntiles_by_expert * ALIGNMENT_BY_EXPERT
    M_pad = tl.sum(ntokens_by_expert_pad, axis=0)
    ntokens_by_expert_pad_acc = tl.cumsum(ntokens_by_expert_pad, axis=0) - ntokens_by_expert_pad

    pid = tl.program_id(axis=0)
    M = ntokens * topk

    if M_pad_ptr:
        tl.store(M_pad_ptr, M_pad)

    if expert_index_ptr:
        ntiles_by_expert_acc = tl.cumsum(ntiles_by_expert, axis=0)
        tl.store(workspace_ptr + offs_by_expert, ntiles_by_expert_acc, mask=mask_by_expert)
        __syncthreads()
        ntiles = M_pad // ALIGNMENT_BY_EXPERT
        for n in range(tl.cdiv(ntiles, BLOCK_SIZE)):
            offs_expert_idx = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            expert_idx = bisect_right_kernel(workspace_ptr, offs_expert_idx, NEXPERTS)
            tl.store(expert_index_ptr + offs_expert_idx, expert_idx, mask=offs_expert_idx < ntiles)

    __syncthreads()
    tl.store(workspace_ptr + offs_by_expert, 0, mask=mask_by_expert)
    # get gather_index with padding to INT_MAX or 0x7fffffff
    if ALIGNMENT_BY_EXPERT > 1 and gather_index_ptr and M != M_pad:
        num_block_pad = tl.cdiv(M_pad, BLOCK_SIZE)
        for n in range(pid, num_block_pad, 1):
            offs = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            tl.store(gather_index_ptr + offs, 0x7FFFFFFF, mask=offs < M_pad)
    __syncthreads()

    num_blocks = tl.cdiv(M, BLOCK_SIZE)
    for n in range(pid, num_blocks, 1):
        offs = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < M
        expert_idx = tl.load(choosed_experts_ptr + offs, mask=mask)
        off_by_expert = tl.gather(ntokens_by_expert_pad_acc, expert_idx, axis=0)
        __syncthreads()
        off_in_expert = tl.atomic_add(workspace_ptr + expert_idx, 1, mask=mask, sem="relaxed", scope="gpu")
        if scatter_index_ptr:
            tl.store(scatter_index_ptr + offs, off_by_expert + off_in_expert, mask=mask)
        if gather_index_ptr:
            tl.store(
                gather_index_ptr + off_by_expert + off_in_expert,
                offs,
                mask=mask,
            )


def calc_gather_scatter_index_triton(
    chosen_experts: torch.Tensor,
    nexperts: int,
    alignment_by_expert: int = 1,
):
    assert chosen_experts.is_cuda and chosen_experts.ndim == 2
    ntokens, topk = chosen_experts.shape
    M = ntokens * topk
    ntokens_by_expert = torch.empty(nexperts, dtype=torch.int32, device="cuda")
    ntiles_approx = (triton.cdiv(M, alignment_by_expert) + nexperts)
    gather_index = torch.empty((ntiles_approx * alignment_by_expert, ), dtype=torch.int32, device="cuda")
    scatter_index = torch.empty((ntokens, topk), dtype=torch.int32, device="cuda")
    expert_index = torch.empty((ntiles_approx, ), dtype=torch.int32, device="cuda")
    M_pad = torch.empty((1, ), dtype=torch.int32, device="cuda")
    workspace = torch.empty(nexperts, dtype=torch.int32, device="cuda")
    calc_gather_scatter_index_kernel[(1, )](
        chosen_experts,
        ntokens_by_expert,
        gather_index,
        scatter_index,
        expert_index,
        M_pad,
        workspace,
        ntokens,
        topk,
        nexperts,
        BLOCK_SIZE=1024,
        ALIGNMENT_BY_EXPERT=alignment_by_expert,
        num_warps=32,
    )
    return ntokens_by_expert, scatter_index, gather_index, expert_index, M_pad


@triton.jit
def reduce_topk_tma_kernel(
    input_ptr,  # of shape [M * topk, H]
    scale_ptr,  # TODO(houqi.1993) not used now
    # output
    output_ptr,  # of shape [M // WORLD_SIZE, H]
    # args
    M,
    N,
    stride_m,
    stride_n,
    TOPK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    in_desc = tl.make_tensor_descriptor(input_ptr, shape=[M, TOPK, N], strides=[stride_m * TOPK, stride_m, stride_n],
                                        block_shape=[BLOCK_SIZE_M, TOPK, BLOCK_SIZE_N])
    out_desc = tl.make_tensor_descriptor(output_ptr, shape=[M, N], strides=[stride_m, stride_n],
                                         block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    pid = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)
    nblocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    nblocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    for tile_id in range(pid, nblocks_m * nblocks_n, npid):
        pid_m = tile_id // nblocks_n
        pid_n = tile_id % nblocks_n
        t_in = in_desc.load([pid_m * BLOCK_SIZE_M, 0, pid_n * BLOCK_SIZE_N])
        t_in = tl.sum(t_in, axis=1)
        out_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], t_in)


@triton.jit
def reduce_topk_kernel(
    input_ptr,  # of shape [M, topk, H]
    scale_ptr,  # weight = weight or torch.ones()
    bias_ptr,  # bias = bias or torch.zeros()
    # output
    output_ptr,  # of shape [M // WORLD_SIZE, H]
    # args
    M,
    N,
    stride_m,
    stride_n,
    TOPK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)
    nblocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    nblocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    nblocks = nblocks_m * nblocks_n
    for n in range(pid, nblocks, npid):
        pid_m = n // nblocks_n
        pid_n = n % nblocks_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < M
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < N
        offs_in = offs_m[:, None] * stride_m * TOPK + offs_n[None, :] * stride_n
        offs_out = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        inptrs = input_ptr + offs_in
        mask = mask_m[:, None] & mask_n[None, :]

        if scale_ptr:  # rely on the compiler to move scale_ptr out of for-loop
            reduced_topk = tl.load(inptrs, mask=mask)
            weight = tl.load(scale_ptr + offs_m, mask=mask_m)[:, None]
            reduced_topk = reduced_topk * weight
            for i in range(1, TOPK):
                val = tl.load(inptrs + i * stride_m, mask=mask)
                weight = tl.load(scale_ptr + offs_m + i, mask=mask_m)[:, None]
                reduced_topk += val * weight
        else:
            reduced_topk = tl.load(inptrs, mask=mask)
            for i in range(1, TOPK):
                val = tl.load(inptrs + i * stride_m, mask=mask)
                reduced_topk += val

        if bias_ptr:  # first local reduce, then add bias. don't reduce to bias
            bias = tl.load(bias_ptr + offs_out, mask=mask)
            reduced_topk += bias

        tl.store(output_ptr + offs_out, reduced_topk, mask=mask)


def reduce_topk_non_tma(data: torch.Tensor, scale: Optional[torch.Tensor], bias: Optional[torch.Tensor],
                        out: torch.Tensor):
    """
    data is organized by chosen_experts, that is data is of shape (ntokens, topk, H)
    this function do:
        data_topk_reduced = weighted_sum(data, weight, dim=1)
        out = reduce_scatter(data_topk_reduced)
    """
    assert data.ndim == 2 and data.is_cuda
    assert scale.ndim == 2 and scale.is_cuda
    assert data.stride(0) == out.stride(0)
    assert data.stride(1) == out.stride(1) == 1
    M, N = data.shape
    ntokens, topk = scale.shape
    assert M == ntokens * topk
    assert N == triton.next_power_of_2(N), f"N={N} should be power of 2"
    grid = lambda meta: (triton.cdiv(ntokens, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
    reduce_topk_kernel[grid](
        data,
        scale,  # expert_weight
        bias,
        out,
        ntokens,
        N,
        data.stride(0),
        data.stride(1),
        topk,
        BLOCK_SIZE_M=max(1, 16 * 1024 // data.itemsize // N),
        BLOCK_SIZE_N=N,
        num_warps=32,
    )


def reduce_topk_tma(data: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor], out: torch.Tensor):
    """
    data is organized by chosen_experts, that is data is of shape (ntokens, topk, H)
    this function do:
        data_topk_reduced = weighted_sum(data, weight, dim=1)
        out = reduce_scatter(data_topk_reduced)
    """
    assert data.ndim == 2 and data.is_cuda
    assert scale.ndim == 2 and scale.is_cuda
    assert data.stride(0) == out.stride(0)
    assert data.stride(1) == out.stride(1) == 1
    M, N = data.shape
    ntokens, topk = scale.shape
    assert M == ntokens * topk
    assert N == triton.next_power_of_2(N), f"N={N} should be power of 2"

    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)

    reduce_topk_tma_kernel[(64, )](
        data,
        scale,  # expert_weight
        out,
        ntokens,
        N,
        data.stride(0),
        data.stride(1),
        topk,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=1024,
        num_warps=8,
        num_stages=1,
    )
    return out
