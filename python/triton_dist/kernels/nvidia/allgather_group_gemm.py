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
from typing import List, Optional

import torch

import triton
import triton.language as tl
from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda
import triton_dist.language as dl
from triton.language.extra.cuda.language_extra import __syncthreads
from triton_dist.kernels.nvidia.allgather import (AllGatherMethod, cp_engine_producer_all_gather_inter_node,
                                                  cp_engine_producer_all_gather_intra_node, get_auto_all_gather_method)
from triton_dist.kernels.nvidia.common_ops import (barrier_on_this_grid, next_power_of_2)
from triton_dist.kernels.nvidia.threadblock_swizzle_ag_moe_triton import \
    threadblock_swizzle_ag_moe_kernel
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE, launch_cooperative_grid_options, nvshmem_barrier_all_on_stream, nvshmem_create_tensors, nvshmem_free_tensor_sync


@triton.jit(do_not_specialize=["rank"])
def _copy_and_reset_and_barrier_all_kernel(
    local_data_ptr,  # [M_per_rank, K]
    out_data_ptr,  # [M_per_rank, K]
    N,
    barrier_ptr,  # [num_ranks] set to 0 except for current rank
    rank,
    NUM_RANKS: tl.constexpr,
    grid_barrier_ptr,
    BLOCK_SIZE: tl.constexpr,
    use_cooperative: tl.constexpr,
):
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    # barrier_all
    if pid == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    # copy data
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    for n in range(pid, num_blocks, npid):
        off = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = off < N
        local_data = tl.load(local_data_ptr + off, mask=mask)
        tl.store(out_data_ptr + off, local_data, mask=mask)

    # reset barrier
    if pid == 0:
        NUM_RANKS_NEXT_POW_OF_2: tl.constexpr = next_power_of_2(NUM_RANKS)
        offs_by_rank = tl.arange(0, NUM_RANKS_NEXT_POW_OF_2)
        mask_by_rank = offs_by_rank < NUM_RANKS
        tl.store(barrier_ptr + offs_by_rank, tl.where(offs_by_rank == rank, 1, 0), mask_by_rank)

    # barrier_all
    if pid == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)


@triton.jit
def calc_sorted_gather_index_kernel(
    topk_ids_ptr,  # of shape (ntokens, TOPK)
    sorted_pad_gather_index_ptr,  # by_expert_by_rank, pad with TILE_SIZE_M
    ntokens_pad_by_expert_acc_ptr,  # of shape (NUM_EXPERTS,)
    ntokens_by_rank_by_expert_ptr,  # of shape (NUM_EXPERTS, TP_SIZE). as workspace buffer
    ntokens_by_expert_by_rank_acc_ptr,  # of shape (NUM_EXPERTS, TP_SIZE). as workspace buffer
    M_pad_ptr,
    ntokens: int,
    TP_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    NUM_EXPERTS_NEXT_POW_OF_2: tl.constexpr = next_power_of_2(NUM_EXPERTS)
    TP_SIZE_NEXT_POW_OF_2: tl.constexpr = next_power_of_2(TP_SIZE)
    ntokens_per_rank = ntokens // TP_SIZE
    M_per_rank = ntokens_per_rank * TOPK
    M = ntokens * TOPK
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(M, BLOCK_SIZE)
    offs_by_expert = tl.arange(0, NUM_EXPERTS_NEXT_POW_OF_2)
    mask_by_expert = offs_by_expert < NUM_EXPERTS
    offs_by_rank = tl.arange(0, TP_SIZE_NEXT_POW_OF_2)
    mask_by_rank = offs_by_rank < TP_SIZE
    offs_by_expert_by_rank = offs_by_expert[:, None] * TP_SIZE + offs_by_rank[None, :]
    mask_by_expert_by_rank = mask_by_expert[:, None] & mask_by_rank[None, :]
    offs_by_rank_by_expert = offs_by_rank[:, None] * NUM_EXPERTS + offs_by_expert[None, :]
    mask_by_rank_by_expert = mask_by_rank[:, None] & mask_by_expert[None, :]
    offs_ravel = tl.arange(0, TP_SIZE_NEXT_POW_OF_2 * NUM_EXPERTS_NEXT_POW_OF_2)
    mask_raval = offs_ravel < TP_SIZE * NUM_EXPERTS
    tl.store(ntokens_by_rank_by_expert_ptr + offs_ravel, 0, mask=mask_raval)
    __syncthreads()
    for n in range(pid, num_blocks, step=1):
        off = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = off < M
        expert_id = tl.load(topk_ids_ptr + off, mask=mask)
        rank = off // M_per_rank
        tl.atomic_add(ntokens_by_rank_by_expert_ptr + expert_id + rank * NUM_EXPERTS, 1, mask=mask, sem="relaxed",
                      scope="gpu")
    __syncthreads()
    ntokens_by_rank_by_expert = tl.load(ntokens_by_rank_by_expert_ptr + offs_by_rank_by_expert,
                                        mask=mask_by_rank_by_expert, other=0)
    __syncthreads()
    ntokens_by_expert_by_rank = ntokens_by_rank_by_expert.T

    ntokens_by_expert_by_rank_acc = tl.cumsum(ntokens_by_expert_by_rank, axis=1)
    ntokens_by_expert = tl.sum(ntokens_by_expert_by_rank, axis=1)
    ntokens_pad_by_expert = tl.cdiv(ntokens_by_expert, TILE_SIZE_M) * TILE_SIZE_M
    ntokens_pad_by_expert_acc = tl.cumsum(ntokens_pad_by_expert, axis=0)
    M_pad = tl.sum(ntokens_pad_by_expert)
    tl.store(M_pad_ptr, M_pad)
    __syncthreads()
    tl.store(ntokens_pad_by_expert_acc_ptr + offs_by_expert, ntokens_pad_by_expert_acc, mask=mask_by_expert)
    tl.store(ntokens_by_expert_by_rank_acc_ptr + offs_by_expert_by_rank, ntokens_by_expert_by_rank_acc,
             mask=mask_by_expert_by_rank)
    __syncthreads()
    # reset to zero
    tl.store(ntokens_by_rank_by_expert_ptr + offs_ravel, 0, mask=mask_raval)
    for n in range(pid, tl.cdiv(M_pad, BLOCK_SIZE), 1):
        off = tl.arange(0, BLOCK_SIZE) + n * BLOCK_SIZE
        tl.store(sorted_pad_gather_index_ptr + off, 0xffffffff, mask=off < M_pad)
    __syncthreads()

    for n in range(pid, num_blocks, 1):
        off = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = off < M
        expert_id = tl.load(topk_ids_ptr + off, mask=mask)
        rank = off // M_per_rank
        off_by_expert_pad = tl.where(expert_id == 0, 0,
                                     tl.load(ntokens_pad_by_expert_acc_ptr + expert_id - 1, mask=mask, other=0))
        off_in_expert_by_rank = tl.where(
            rank == 0, 0, tl.load(ntokens_by_expert_by_rank_acc_ptr + expert_id * TP_SIZE + rank - 1, mask=mask,
                                  other=0))
        token_index_in_rank_in_expert = tl.atomic_add(ntokens_by_rank_by_expert_ptr + rank * NUM_EXPERTS + expert_id, 1,
                                                      mask=mask, sem="relaxed", scope="gpu")
        tl.store(
            sorted_pad_gather_index_ptr + off_by_expert_pad + off_in_expert_by_rank + token_index_in_rank_in_expert,
            off,
            mask=mask,
        )


def calc_sorted_gather_index(
    topk_ids: torch.Tensor,
    tp_size,
    num_experts: int,
    block_size_m: int,
):
    ntokens, topk = topk_ids.shape
    ntokens_pad_by_expert_acc = torch.empty((num_experts, ), dtype=torch.int32, device="cuda")
    ntokens_by_rank_by_expert = torch.empty((tp_size, num_experts), dtype=torch.int32, device="cuda")
    ntokens_by_expert_by_rank_acc = torch.empty((num_experts, tp_size), dtype=torch.int32, device="cuda")
    ntokens_pad_approx = (triton.cdiv(ntokens, block_size_m) + num_experts) * block_size_m
    sorted_pad_gather_index = torch.empty((ntokens_pad_approx, topk), device=topk_ids.device, dtype=topk_ids.dtype)
    M_pad = torch.empty((1, ), device=topk_ids.device, dtype=topk_ids.dtype)
    BLOCK_SIZE = triton.next_power_of_2(min(1024, max(ntokens_pad_approx * topk, tp_size * num_experts)))
    calc_sorted_gather_index_kernel[(1, )](
        topk_ids,
        sorted_pad_gather_index,
        ntokens_pad_by_expert_acc,
        ntokens_by_rank_by_expert,
        ntokens_by_expert_by_rank_acc,
        M_pad,
        ntokens,
        tp_size,
        num_experts,
        topk,
        block_size_m,
        BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32,
    )
    return sorted_pad_gather_index, ntokens_by_rank_by_expert


@dataclass
class MoEAllGatherGroupGEMMTensorParallelContext:
    # problem size
    # local input [M_per_rank, K]
    # local weight [expert_num, K, N_per_rank]
    max_ntokens: int
    N_per_rank: int
    K: int
    num_experts: int
    topk: int
    dtype: torch.dtype
    # parallelism info
    rank: int
    num_ranks: int
    num_local_ranks: int = 8
    is_multinode: bool = field(init=False)
    n_nodes: int = field(init=False)
    node_rank: int = field(init=False)
    local_rank: int = field(init=False)
    # distributed mem
    symm_workspaces: List[torch.Tensor] = field(init=False)  # ag buffer
    symm_barriers: List[torch.Tensor] = field(init=False)
    symm_workspace: torch.Tensor = field(init=False)
    symm_barrier: torch.Tensor = field(init=False)
    barrier_target = 1
    grid_barrier: torch.Tensor = field(init=False)  # used for triton grid barrier
    # async streams
    ag_intranode_stream: Optional[torch.cuda.streams.Stream] = None
    ag_internode_stream: Optional[torch.cuda.streams.Stream] = None
    # triton compute kernel config
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    GROUP_SIZE_M: int = 8
    stages: int = 3
    warps: int = 8
    all_gather_method: AllGatherMethod = AllGatherMethod.Auto

    def __post_init__(self):
        assert self.num_ranks % self.num_local_ranks == 0, "num_ranks must be divisible by num_local_ranks"
        self.is_multinode = self.num_ranks > self.num_local_ranks
        self.n_nodes = self.num_ranks // self.num_local_ranks
        self.node_rank = self.rank // self.num_local_ranks
        self.local_rank = self.rank % self.num_local_ranks
        self.grid_barrier = torch.zeros((1, ), dtype=torch.int32, device="cuda")

        self.symm_workspaces = nvshmem_create_tensors((self.max_ntokens, self.K), self.dtype, self.rank,
                                                      self.num_local_ranks)
        self.symm_workspace = self.symm_workspaces[self.local_rank]

        self.symm_barriers = nvshmem_create_tensors((self.num_ranks, ), NVSHMEM_SIGNAL_DTYPE, self.rank,
                                                    self.num_local_ranks)
        self.symm_barrier = self.symm_barriers[self.local_rank]
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    def finalize(self):
        nvshmem_free_tensor_sync(self.symm_barrier)
        nvshmem_free_tensor_sync(self.symm_workspace)

    @staticmethod
    def sort_topk_ids_align_block_size(
        topk_ids: torch.Tensor,  # [ntokens, topk]
        num_experts: int,
        rank: int,
        num_ranks: int,
        num_local_ranks: int,
        block_size: int,
    ):
        sorted_gather_index, ntokens_by_rank_by_expert = calc_sorted_gather_index(topk_ids, num_ranks, num_experts,
                                                                                  block_size)
        ntokens, topk = topk_ids.shape
        #  maybe a little more than needed, but never mind
        ntiles_pad_approx = triton.cdiv(ntokens * topk, block_size) + num_experts

        expert_idx = torch.empty((ntiles_pad_approx, ), dtype=torch.int32, device="cuda")
        tile_index = torch.empty((ntiles_pad_approx, ), dtype=torch.int32, device="cuda")
        segment_start = torch.empty((ntiles_pad_approx, ), dtype=torch.int32, device="cuda")
        segment_end = torch.empty((ntiles_pad_approx, ), dtype=torch.int32, device="cuda")
        ntiles_pad_gpu = torch.empty((1, ), dtype=torch.int32, device="cuda")

        ntokens_by_expert_by_rank_acc = torch.empty((num_experts, num_ranks), dtype=torch.int32, device="cuda")
        ntiles_by_expert_acc = torch.empty((num_experts, ), dtype=torch.int32, device="cuda")
        ntiles_by_expert_by_stage = torch.empty((num_experts, num_ranks), dtype=torch.int32,
                                                device="cuda")  # this will be used as counter. zero before use.
        ntiles_by_expert_by_stage_acc = torch.empty((num_experts, num_ranks), dtype=torch.int32, device="cuda")

        threadblock_swizzle_ag_moe_kernel[(1, )](
            ntokens_by_rank_by_expert,
            # output
            expert_idx,
            tile_index,
            segment_start,
            segment_end,
            ntiles_pad_gpu,
            # workspace buffer
            ntokens_by_expert_by_rank_acc,
            ntiles_by_expert_acc,
            ntiles_by_expert_by_stage,
            ntiles_by_expert_by_stage_acc,
            rank,
            num_experts,
            num_ranks,
            num_local_ranks,
            triton.next_power_of_2(ntiles_pad_approx),
            BLOCK_SIZE_M=block_size,
            DEBUG=False,
        )

        return sorted_gather_index, expert_idx, tile_index, segment_start, segment_end, ntiles_pad_gpu

    def copy_and_reset_and_barrier_all(self, local_data):
        M_per_rank = local_data.shape[0]
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        self.symm_barrier.zero_()
        dst = self.symm_workspace[self.rank * M_per_rank:(self.rank + 1) * M_per_rank, :]
        dst.copy_(local_data)
        _set_signal_cuda(self.symm_barrier[self.rank], 1, torch.cuda.current_stream())
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    def copy_and_reset_and_barrier_all_triton(self, local_data):
        elems_per_rank = local_data.numel()
        assert local_data.is_contiguous() and local_data.is_cuda, "local_data must be contiguous"
        _copy_and_reset_and_barrier_all_kernel[(16, )](
            local_data,
            self.symm_workspace.flatten()[elems_per_rank * self.rank:elems_per_rank * (self.rank + 1)],
            elems_per_rank,
            self.symm_barrier,
            self.rank,
            self.symm_barrier.numel(),
            self.grid_barrier,
            BLOCK_SIZE=1024 * 16 // local_data.itemsize,
            num_warps=32,
            use_cooperative=True,
            **launch_cooperative_grid_options(),
        )


def create_ag_group_gemm_context(
    max_ntokens,
    N_per_rank,
    K,
    num_experts,
    topk,
    dtype: torch.dtype,
    rank,
    num_ranks,
    num_local_ranks,
    ag_intranode_stream: Optional[torch.cuda.Stream] = None,
    ag_internode_stream: Optional[torch.cuda.Stream] = None,
    BLOCK_SIZE_M=128,
    BLOCK_SIZE_N=256,
    BLOCK_SIZE_K=64,
    GROUP_SIZE_M=8,
    stages=3,
    num_warps=8,
) -> MoEAllGatherGroupGEMMTensorParallelContext:
    """Create context for allgather group gemm.

    Args:
        rank (int): current rank
        num_ranks (int): total number of ranks
        tensor_A (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        tensor_B (torch.Tensor<float>): local matmul B matrix. shape: [E, K, N_per_rank]
        full_topk_id (torch.Tensor<int32_t>): allgathered topk ids. shape: [M, topk]
        ag_intranode_stream (torch.cuda.streams.Stream, optional): The stream used for intranode allgather, if not provided, create a new one. Defaults to None.
        ag_internode_stream (torch.cuda.streams.Stream, optional): The stream used for internode allgather, if not provided, create a new one. Defaults to None.
        BLOCK_M (int, optional): Group GEMM tiling factor for M dim. Defaults to 128.
        BLOCK_N (int, optional): Group GEMM tiling factor for N dim. Defaults to 256.
        BLOCK_K (int, optional): Group GEMM tiling factor for K dim. Defaults to 64.
        GROUP_SIZE_M (int, optional): Group size of block for M dim (not size of group GEMM). Defaults to 8.
        stages (int, optional): GEMM async-copy stages. Defaults to 3.
        warps (int, optional): No.of used warps. Defaults to 8.

    Returns:
        MoEAllGatherGroupGEMMTensorParallelContext
    """

    ctx = MoEAllGatherGroupGEMMTensorParallelContext(
        max_ntokens=max_ntokens,
        N_per_rank=N_per_rank,
        K=K,
        num_experts=num_experts,
        topk=topk,
        dtype=dtype,
        rank=rank,
        num_ranks=num_ranks,
        num_local_ranks=num_local_ranks,
        ag_intranode_stream=ag_intranode_stream or torch.cuda.Stream(),
        ag_internode_stream=ag_internode_stream or torch.cuda.Stream(),
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_N=BLOCK_SIZE_N,
        BLOCK_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        stages=stages,
        warps=num_warps,
        all_gather_method=get_auto_all_gather_method(num_ranks, num_local_ranks),
    )

    return ctx


def ag_group_gemm(a: torch.Tensor, b: torch.Tensor, ctx: MoEAllGatherGroupGEMMTensorParallelContext, full_topk_ids):
    ntokens_per_rank, hidden = a.shape
    n_experts, K, hidden_b = b.shape
    assert n_experts == ctx.num_experts
    c = torch.empty(
        [ctx.topk * ntokens_per_rank * ctx.num_ranks, ctx.N_per_rank],
        dtype=ctx.dtype,
        device=a.device,
    )
    rowise_ag_scatter_group_gemm_dispatcher(a, b, c, ctx, full_topk_ids)
    return c


def rowise_ag_scatter_group_gemm_dispatcher(a,  # local tensor
                                            b,  # local weight
                                            c,  # output
                                            ctx: MoEAllGatherGroupGEMMTensorParallelContext, full_topk_ids):
    ctx.copy_and_reset_and_barrier_all_triton(a)

    current_stream = torch.cuda.current_stream()
    if ctx.is_multinode:
        ctx.ag_internode_stream.wait_stream(current_stream)
    ctx.ag_intranode_stream.wait_stream(current_stream)

    if not ctx.is_multinode:
        cp_engine_producer_all_gather_intra_node(
            ctx.rank,
            ctx.num_ranks,
            a,
            ctx.symm_workspaces,
            ctx.symm_barriers,
            ctx.ag_intranode_stream,
            all_gather_method=ctx.all_gather_method,
        )
    else:
        cp_engine_producer_all_gather_inter_node(a, ctx.symm_workspaces, ctx.symm_barriers, ctx.barrier_target,
                                                 ctx.rank, ctx.num_local_ranks, ctx.num_ranks, ctx.ag_intranode_stream,
                                                 ctx.ag_internode_stream, all_gather_method=ctx.all_gather_method)

    sorted_gather_index, expert_idx, tiled_m, segment_start, segment_end, ntiles_gpu = ctx.sort_topk_ids_align_block_size(
        full_topk_ids,
        ctx.num_experts,
        ctx.rank,
        ctx.num_ranks,
        ctx.num_local_ranks,
        ctx.BLOCK_M,
    )

    ntokens_per_rank, K = a.shape
    M = ntokens_per_rank * ctx.num_ranks * ctx.topk
    local_ag_buffer = ctx.symm_workspace[:M]

    grid = lambda META: ((triton.cdiv(M, META["BLOCK_SIZE_M"]) + ctx.num_experts - 1) * triton.cdiv(
        ctx.N_per_rank, META["BLOCK_SIZE_N"]), )
    compiled = kernel_consumer_m_parallel_scatter_group_gemm[grid](
        local_ag_buffer,
        b,
        c,
        ctx.symm_barrier,
        sorted_gather_index,
        expert_idx,
        tiled_m,
        segment_start,
        segment_end,
        ntiles_gpu,
        M,
        ctx.N_per_rank,
        ctx.K,
        local_ag_buffer.stride(0),
        local_ag_buffer.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        ctx.BLOCK_M,
        ctx.BLOCK_N,
        ctx.BLOCK_K,
        ctx.GROUP_SIZE_M,
        ctx.topk,
        num_stages=ctx.stages,
        num_warps=ctx.warps,
    )

    if ctx.is_multinode:
        current_stream.wait_stream(ctx.ag_internode_stream)
    current_stream.wait_stream(ctx.ag_intranode_stream)

    return compiled


def _kernel_consumer_gemm_non_persistent_repr(proxy):
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

    return f"triton3x_sm{cap_major}{cap_minor}_ag_group_gemm_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_{a_trans}{b_trans}{c_trans}"


@triton.jit
def swizzle_2d(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(do_not_specialize=["rank"], repr=_kernel_consumer_gemm_non_persistent_repr)
def kernel_consumer_m_parallel_scatter_group_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    block_barrier_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tiled_m_ptr,
    segment_start_ptr,
    segment_end_ptr,
    ntiles_pad_ptr,
    M,  # M = ntokens_per_rank * WORLD_SIZE
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    TOP_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    ntiles_pad = tl.load(ntiles_pad_ptr)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    # num_block_m = tl.cdiv(M, BLOCK_SIZE_M)
    npid = tl.num_programs(axis=0)
    num_block_m = npid // num_block_n

    pid_m, pid_n = swizzle_2d(pid, num_block_m, num_block_n, GROUP_SIZE_M)

    if pid_m >= ntiles_pad:
        return

    tiled_m = tl.load(tiled_m_ptr + pid_m)
    offs_token_id = tiled_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = (offs_token < M) & (offs_token >= 0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (a_ptr + offs_token[:, None] // TOP_K * stride_am + offs_k[None, :] * stride_ak)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_be = tl.load(expert_ids_ptr + pid_m)
    segment_start = tl.load(segment_start_ptr + pid_m)
    segment_end = tl.load(segment_end_ptr + pid_m)
    __syncthreads()

    b_ptrs = (b_ptr + offs_be * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    token = dl.wait(block_barrier_ptr + segment_start, segment_end - segment_start + 1, "gpu", "acquire")
    __syncthreads()
    a_ptrs = dl.consume_token(a_ptrs, token)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K))

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
