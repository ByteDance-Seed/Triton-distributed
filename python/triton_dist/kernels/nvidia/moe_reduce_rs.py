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
import triton_dist.language as dl
from triton.language.extra.cuda.language_extra import (__syncthreads, atomic_add, tid, ntid, multimem_ld_reduce_v4,
                                                       st_v4_b32)
from triton_dist.kernels.nvidia.common_ops import (BarrierAllContext, barrier_all_on_stream, barrier_on_this_grid,
                                                   set_signal, wait_eq)
from triton_dist.kernels.nvidia.reduce_scatter import ring_reduce
from triton_dist.language.extra import libshmem_device
from triton_dist.kernels.nvidia.moe_utils import calc_gather_scatter_index_triton, reduce_topk_kernel
from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE, nvshmem_barrier_all_on_stream, nvshmem_create_tensor, nvshmem_create_tensors, nvshmem_free_tensor_sync

################### helper functions ###################


def is_power_of_two(value):
    return ((value - 1) & value) == 0


def torch_dtype_to_triton_dtype(dtype):
    if dtype == torch.float32:
        return tl.float32
    elif dtype == torch.float16:
        return tl.float16
    elif dtype == torch.int32:
        return tl.int32
    elif dtype == torch.int8:
        return tl.int8
    else:
        raise RuntimeError(f"unsupported dtype: {dtype}")


################### compute ctx ###################
@dataclass
class MoEAgScatterGroupGemmPrecomputeContext:
    full_topk_weight: torch.Tensor = None
    full_topk_ids: torch.Tensor = None
    full_sorted_token_ids: torch.Tensor = None
    full_token_expert_ids: torch.Tensor = None
    block_wait_barriers: torch.Tensor = None
    rank_block_num: torch.Tensor = None
    full_num_tokens_post_padded_list: torch.Tensor = None
    EM: int = 0
    full_numel: int = 0
    topk: int = 0
    BLOCK_M: int = 0
    num_tokens_per_rank: int = 0


def full_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    num_ranks: int,
    num_tokens_per_rank: int,
):
    sorted_ids = torch.empty(
        ((num_tokens_per_rank * topk_ids.shape[1] + num_experts * (block_size - 1)) * num_ranks, ),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        ((num_tokens_per_rank * topk_ids.shape[1] + num_experts) * num_ranks, ),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    block_barrier_ids = torch.empty(
        ((num_tokens_per_rank * topk_ids.shape[1] + num_experts) * num_ranks, ),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    rank_block_num = torch.empty(
        num_ranks,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    # The part below adapted from the cuda kernel in Saber

    num_iterations = num_ranks
    num_tokens_per_iteration = num_tokens_per_rank * topk_ids.shape[1]
    numel = num_tokens_per_iteration
    tokens_per_thread = triton.cdiv(numel, num_experts)

    topk_ids_flatten = topk_ids.flatten()

    last_pad_tokens = 0
    num_tokens_post_pad[0] = 0
    sorted_ids_idx = 0
    expert_ids_idx = 0
    block_barrier_ids_idx = 0
    topk_ids_idx = 0
    for iter in range(num_iterations):
        sorted_ids_idx += last_pad_tokens
        expert_ids_idx += last_pad_tokens // block_size
        block_barrier_ids_idx += last_pad_tokens // block_size

        token_cnts = torch.zeros((num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device)
        cumsum = torch.zeros((num_experts + 1), dtype=torch.int32, device=topk_ids.device)

        for j in range(num_experts):
            start_idx = j * tokens_per_thread
            for i in range(start_idx, min(numel, start_idx + tokens_per_thread)):
                token_cnts[j + 1, topk_ids_flatten[topk_ids_idx + i]] += 1

        for j in range(num_experts):
            for i in range(1, num_experts + 1):
                token_cnts[i, j] += token_cnts[i - 1, j]

        for i in range(1, num_experts + 1):
            cumsum[i] = cumsum[i - 1] + triton.cdiv(token_cnts[num_experts, i - 1], block_size) * block_size
        num_tokens_post_pad[0] += cumsum[num_experts]
        rank_block_num[iter] = cumsum[num_experts] // block_size

        last_pad_tokens = cumsum[num_experts]

        for j in range(num_experts):
            for i in range(cumsum[j], cumsum[j + 1], block_size):
                expert_ids[expert_ids_idx + i // block_size] = j
                block_barrier_ids[block_barrier_ids_idx + i // block_size] = iter

        for j in range(num_experts):
            start_idx = j * tokens_per_thread
            for i in range(start_idx, min(numel, start_idx + tokens_per_thread)):
                expert_id = topk_ids_flatten[topk_ids_idx + i]
                rank_post_pad = token_cnts[j, expert_id] + cumsum[expert_id]
                sorted_ids[sorted_ids_idx + rank_post_pad] = i + iter * num_tokens_per_iteration
                token_cnts[j, expert_id] += 1

        topk_ids_idx += num_tokens_per_iteration

    return (
        sorted_ids,
        expert_ids,
        block_barrier_ids,
        rank_block_num,
        num_tokens_post_pad,
    )


def select_experts(pg, num_ranks, topk, input_dtype, device, router_logits):
    num_tokens_per_rank = router_logits.shape[0]
    num_tokens = num_tokens_per_rank * num_ranks
    full_topk_ids = torch.zeros(num_tokens, topk, dtype=torch.int32, device=device)
    full_topk_weight = torch.zeros(num_tokens, topk, dtype=input_dtype, device=device)
    score = torch.softmax(router_logits, dim=-1)
    local_topk_weight, local_topk_ids = torch.topk(score, topk)
    torch.distributed.all_gather_into_tensor(
        full_topk_weight,
        local_topk_weight,
        group=pg,
    )
    torch.distributed.all_gather_into_tensor(
        full_topk_ids,
        local_topk_ids.to(torch.int32),
        group=pg,
    )
    return full_topk_ids, full_topk_weight


def precompute_context_helper(
    pg,
    num_ranks: int,
    topk: int,
    num_tokens_per_rank: int,
    num_experts: int,
    input_dtype,
    device,
    router_logits,
    BLOCK_M: int = 128,
):
    ctx = MoEAgScatterGroupGemmPrecomputeContext()

    (ctx.full_topk_ids, ctx.full_topk_weight) = select_experts(pg, num_ranks, topk, input_dtype, device, router_logits)

    E = num_experts
    ctx.topk = topk
    ctx.BLOCK_M = BLOCK_M

    (
        full_sorted_token_ids,
        full_token_expert_ids,
        block_wait_barriers,
        rank_block_num,
        full_num_tokens_post_padded_list,
    ) = full_moe_align_block_size(ctx.full_topk_ids, BLOCK_M, E, num_ranks, num_tokens_per_rank)
    EM = full_num_tokens_post_padded_list.cpu().tolist()[0]  # full_sorted_token_ids.shape[0]
    full_numel = ctx.full_topk_ids.numel()

    ctx.full_sorted_token_ids = full_sorted_token_ids
    ctx.full_token_expert_ids = full_token_expert_ids
    ctx.full_num_tokens_post_padded_list = full_num_tokens_post_padded_list
    ctx.block_wait_barriers = block_wait_barriers
    ctx.rank_block_num = rank_block_num
    ctx.EM = EM
    ctx.full_numel = full_numel
    ctx.num_tokens_per_rank = num_tokens_per_rank
    return ctx


@dataclass
class DataflowConfig:
    GEMM_BLOCK_M: int
    GEMM_BLOCK_N: int
    GEMM_BLOCK_K: int
    GROUP_SIZE: int
    num_stages: int
    num_warps: int
    RS_BLOCK_M: int
    RS_BLOCK_N: int


@dataclass
class MoEReduceRSContext:
    rank: int
    world_size: int
    local_rank: int = field(init=False)
    local_world_size: int
    nnodes: int = field(init=False)

    precompute_ctx: MoEAgScatterGroupGemmPrecomputeContext

    symm_rs_buffers: List[torch.Tensor]
    symm_rs_buffer_ptrs: torch.Tensor
    symm_rs_per_node_buffer: torch.Tensor
    symm_p2p_buffer: torch.Tensor
    final_output_buffer: torch.Tensor
    barrier: BarrierAllContext

    rs_stream: torch.cuda.Stream
    reduction_stream: torch.cuda.Stream
    p2p_stream: torch.cuda.Stream

    dataflow_config: DataflowConfig

    symm_barrier_gemm_scatter_counter_ptrs: torch.Tensor
    symm_barrier_gemm_scatter_counter: torch.Tensor
    symm_barrier_gemm_scatter_ready_ptrs: torch.Tensor
    symm_barrier_gemm_scatter_ready: torch.Tensor
    symm_rs_per_node_signal: torch.Tensor

    def __post_init__(self):
        assert self.world_size % self.local_world_size == 0
        self.local_rank = self.rank % self.local_world_size
        self.nnodes = self.world_size // self.local_world_size

    def finalize(self):
        nvshmem_free_tensor_sync(self.symm_rs_buffers[self.local_rank])
        nvshmem_free_tensor_sync(self.symm_rs_per_node_buffer)
        nvshmem_free_tensor_sync(self.symm_p2p_buffer)
        nvshmem_free_tensor_sync(self.symm_barrier_gemm_scatter_counter)
        nvshmem_free_tensor_sync(self.symm_barrier_gemm_scatter_ready)
        nvshmem_free_tensor_sync(self.symm_rs_per_node_signal)


def create_moe_rs_context(pg: torch.distributed.ProcessGroup, local_world_size, max_token_num, hidden_dim, num_experts,
                          topk, input_dtype, output_dtype, device, moe_block_size, router_logits):
    num_tokens_per_rank = router_logits.shape[0]
    world_size = pg.size()
    rank = pg.rank()
    local_rank = rank % local_world_size
    precompute_ctx = precompute_context_helper(
        pg,
        world_size,
        topk,
        num_tokens_per_rank,
        num_experts,
        input_dtype,
        device,
        router_logits,
        BLOCK_M=moe_block_size,
    )

    symm_rs_buffers = nvshmem_create_tensors((max_token_num, hidden_dim), input_dtype, rank, local_world_size)
    symm_rs_buffer_ptrs: torch.Tensor = torch.tensor([t.data_ptr() for t in symm_rs_buffers], device=device)

    symm_rs_per_node_buffer = nvshmem_create_tensor(
        (max_token_num // local_world_size, hidden_dim),
        input_dtype,
    )
    symm_p2p_buffer = nvshmem_create_tensor(
        (max_token_num // local_world_size, hidden_dim),
        input_dtype,
    )
    final_output_buffer = torch.zeros(
        (max_token_num * topk, hidden_dim),
        dtype=output_dtype,
        device=device,
    )

    barrier = BarrierAllContext(True)

    # stream
    rs_stream = torch.cuda.Stream()
    reduction_stream = torch.cuda.Stream()
    p2p_stream = torch.cuda.Stream()

    # Setup metadata for kernel launch
    RS_BLOCK_M = max_token_num // world_size
    RS_BLOCK_N = hidden_dim
    GEMM_BLOCK_M = moe_block_size
    GEMM_BLOCK_N = 128
    GEMM_BLOCK_K = 32
    dataflow_config = DataflowConfig(GEMM_BLOCK_M, GEMM_BLOCK_N, GEMM_BLOCK_K, 8, 4, 4, RS_BLOCK_M, RS_BLOCK_N)

    # gemm_scatter
    symm_barrier_gemm_scatter_counters = nvshmem_create_tensors((world_size, 1), torch.int32, rank, local_world_size)
    symm_barrier_gemm_scatter_counter = symm_barrier_gemm_scatter_counters[local_rank]
    symm_barrier_gemm_scatter_counter.zero_()
    symm_barrier_gemm_scatter_counter_ptrs = torch.tensor([t.data_ptr()
                                                           for t in symm_barrier_gemm_scatter_counters]).cuda()

    symm_barrier_gemm_scatter_readys = nvshmem_create_tensors((world_size, 1), NVSHMEM_SIGNAL_DTYPE, rank,
                                                              local_world_size)
    symm_barrier_gemm_scatter_ready = symm_barrier_gemm_scatter_readys[local_rank]
    symm_barrier_gemm_scatter_ready.zero_()
    symm_barrier_gemm_scatter_ready_ptrs = torch.tensor([t.data_ptr() for t in symm_barrier_gemm_scatter_readys]).cuda()

    # intra_node - p2p
    symm_rs_per_node_signal = nvshmem_create_tensor((world_size, ), NVSHMEM_SIGNAL_DTYPE)
    symm_rs_per_node_signal.zero_()
    nvshmem_barrier_all_on_stream()

    return MoEReduceRSContext(
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        precompute_ctx=precompute_ctx,
        symm_rs_buffers=symm_rs_buffers,
        symm_rs_buffer_ptrs=symm_rs_buffer_ptrs,
        symm_rs_per_node_buffer=symm_rs_per_node_buffer,
        symm_p2p_buffer=symm_p2p_buffer,
        final_output_buffer=final_output_buffer,
        barrier=barrier,
        rs_stream=rs_stream,
        reduction_stream=reduction_stream,
        p2p_stream=p2p_stream,
        dataflow_config=dataflow_config,
        symm_barrier_gemm_scatter_counter_ptrs=symm_barrier_gemm_scatter_counter_ptrs,
        symm_barrier_gemm_scatter_counter=symm_barrier_gemm_scatter_counter,
        symm_barrier_gemm_scatter_ready_ptrs=symm_barrier_gemm_scatter_ready_ptrs,
        symm_barrier_gemm_scatter_ready=symm_barrier_gemm_scatter_ready,
        symm_rs_per_node_signal=symm_rs_per_node_signal,
    )


################### triton kernel ###################
@triton.jit
def kernel_producer_group_gemm_tp_scatter_input(
    local_world_size,
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    token_expert_ids_ptr,
    topk_weight_ptr,
    num_tokens_post_padded,
    block_wait_barrier_ptr,
    rank_block_num,
    barrier_counter,
    barriers_ready_ptrs,
    num_valid_tokens: int,
    EM,
    N,
    K_per_rank,
    E,
    stride_in_m,
    stride_in_k,
    stride_weight_e,
    stride_weight_k,
    stride_weight_n,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TOP_K: tl.constexpr,
    compute_dtype: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_block_m = tl.cdiv(EM, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)

    num_blocks_per_group = GROUP_M * num_block_n
    group_id = pid // num_blocks_per_group
    group_size = min(num_block_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % num_blocks_per_group // group_size

    rank = dl.rank()
    num_ranks = dl.num_ranks()
    local_rank = rank % local_world_size
    num_block_m_per_rank = num_block_m // num_ranks
    m_offset = num_block_m_per_rank * ((rank + 5) % num_ranks)
    pid_m = (pid_m + m_offset) % num_block_m

    num_tokens_post_padded_value = tl.load(num_tokens_post_padded)

    if pid_m * BLOCK_M >= num_tokens_post_padded_value:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (a_ptr + offs_token[:, None] * stride_in_m + offs_k[None, :] * stride_in_k)

    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_be = tl.load(token_expert_ids_ptr + pid_m)

    b_ptrs = (b_ptr + offs_be * stride_weight_e + offs_k[:, None] * stride_weight_k +
              offs_bn[None, :] * stride_weight_n)

    moe_weight = tl.load(topk_weight_ptr + offs_token, mask=token_mask, other=0)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K_per_rank, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K_per_rank - k * BLOCK_K),
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_bn[None, :] < N) & (offs_k[:, None] < K_per_rank - k * BLOCK_K),
        )

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_in_k
        b_ptrs += BLOCK_K * stride_weight_k

    accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_dtype)

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = (c_ptr + offs_token[:, None] * stride_out_m + offs_cn[None, :] * stride_out_n)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

    offs_counter = tl.load(block_wait_barrier_ptr + pid_m)
    offs_ready = tl.load(block_wait_barrier_ptr + pid_m)
    threshold = tl.load(rank_block_num + offs_counter) * num_block_n
    counter_ptr = barrier_counter + offs_counter
    remote_barrier_ready_ptr = tl.load(barriers_ready_ptrs + local_rank).to(tl.pointer_type(tl.uint64))
    __syncthreads()
    thread_id = tid(0)
    value = 1
    if thread_id == 0:
        if atomic_add(counter_ptr, value, "gpu", "relaxed") == threshold - 1:
            dl.notify(remote_barrier_ready_ptr + offs_ready, rank, signal=1, sig_op="add", comm_scope="gpu")


@triton.jit
def kernel_consumer_topk_reduce_scatter_intra_node(
    local_world_size,
    consumer_output_ptr,  # shape: [M * ReduceLength, N]
    remote_buffer_ptrs,  # each tensor shape: [M, N]
    barrier_gemm_scatter_ready,
    M,
    N,
    num_pid_m,
    num_pid_n,
    # constants
    REDUCE_LENGTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    perfect_tile: tl.constexpr,
    use_tl_reduce: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    local_rank = rank % local_world_size
    nnodes = world_size // local_world_size
    dtype = consumer_output_ptr.dtype.element_ty
    M_per_rank = M // world_size
    num_blocks_m_per_rank = tl.cdiv(M_per_rank, BLOCK_M)
    num_blocks_m_per_node = num_blocks_m_per_rank * local_world_size

    tl.static_assert(perfect_tile, "Consider perfect tiling now.")

    pid = tl.program_id(axis=0)
    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    num_block_per_pid_m = tl.cdiv(num_block_m, num_pid_m)
    num_block_per_pid_n = tl.cdiv(num_block_n, num_pid_n)

    for m in range(num_block_per_pid_m):
        for n in range(num_block_per_pid_n):
            mid = m * num_pid_m + pid_m
            nid = n * num_pid_n + pid_n

            mid = (mid + local_rank * nnodes * num_blocks_m_per_rank) % num_block_m

            to_rank = mid // num_blocks_m_per_rank
            to_rank_local = to_rank % local_world_size
            to_node = to_rank // local_world_size

            if use_tl_reduce:
                offs_m_reduce = tl.arange(0, BLOCK_M * REDUCE_LENGTH)
                offs_in_m = mid * BLOCK_M * REDUCE_LENGTH + offs_m_reduce
                offs_in_n = nid * BLOCK_N + tl.arange(0, BLOCK_N)
                src_ptrs = (consumer_output_ptr + offs_in_m[:, None] * N + offs_in_n[None, :])
                token = dl.wait(barrier_gemm_scatter_ready + to_rank, 1, "gpu", "acquire")
                src_ptrs = dl.consume_token(src_ptrs, token)
                data = tl.load(src_ptrs)
                to_reduce_data = tl.reshape(data, [BLOCK_M, REDUCE_LENGTH, BLOCK_N])
                reduce_data = tl.sum(to_reduce_data, axis=1)
            else:
                reduce_data = tl.zeros((BLOCK_M, BLOCK_N), dtype=dtype)
                token = dl.wait(barrier_gemm_scatter_ready + to_rank, 1, "gpu", "acquire")
                for i in range(REDUCE_LENGTH):
                    offs_m_reduce = tl.arange(0, BLOCK_M) + i
                    offs_in_m = (mid * BLOCK_M * REDUCE_LENGTH + offs_m_reduce * REDUCE_LENGTH)
                    offs_in_n = nid * BLOCK_N + tl.arange(0, BLOCK_N)
                    src_ptrs = (consumer_output_ptr + offs_in_m[:, None] * N + offs_in_n[None, :])
                    src_ptrs = dl.consume_token(src_ptrs, token)
                    data = tl.load(src_ptrs)
                    reduce_data += data

            # scatter
            dst_ptr = tl.load(remote_buffer_ptrs + to_rank_local).to(tl.pointer_type(dtype))
            offs_out_m = (to_node * num_blocks_m_per_node + local_rank * num_blocks_m_per_rank +
                          mid % num_blocks_m_per_rank) * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_out_n = nid * BLOCK_N + tl.arange(0, BLOCK_N)
            dst_ptrs = dst_ptr + offs_out_m[:, None] * N + offs_out_n[None, :]
            tl.store(dst_ptrs, reduce_data)


@triton.jit
def kernel_consumer_reduce(
    local_world_size,
    c_ptr,  # [M_per_node, N]
    out_ptr,  # [M_per_rank, N]
    # shape of matrix
    M,
    N,
    # strides
    stride_m,
    stride_n,
    # reduce tile shape
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    local_rank = rank % local_world_size
    m_per_rank = tl.cdiv(M, world_size)
    pid = tl.program_id(axis=0)
    reduce_n_blocks_per_rank = tl.cdiv(N, BLOCK_N)
    pid_m = pid // reduce_n_blocks_per_rank
    pid_n = pid % reduce_n_blocks_per_rank

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    out_ptrs = out_ptr + (offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)

    org_data = tl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.dtype.element_ty)

    for rid in range(0, local_world_size):
        swizzle_rid = (rid + local_rank) % local_world_size
        full_offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M) + swizzle_rid * m_per_rank) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        ptrs = c_ptr + (full_offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
        data = tl.load(ptrs)
        org_data += data

    tl.store(out_ptrs, org_data)


@triton.jit
def kernel_inter_node_p2p_for_same_local_rank(
    local_world_size,
    M_per_rank,
    N,
    input,  # [M_per_rank * nnodes, N]
    output,  # [M_per_rank * nnodes, N]
    rs_per_node_signal,
):
    pid = tl.program_id(axis=0)
    rank = dl.rank()
    world_size = dl.num_ranks()
    node_id = rank // local_world_size
    nnodes = world_size // local_world_size
    local_rank = rank % local_world_size
    num_pid = tl.num_programs(axis=0)
    nelem_per_rank = M_per_rank * N
    elem_size = tl.constexpr(input.dtype.element_ty.primitive_bitwidth) // 8

    for i in range(pid, nnodes - 1, num_pid):
        remote_node_id = (i + 1 + node_id) % nnodes
        remote_rank = local_rank + remote_node_id * local_world_size
        libshmem_device.signal_wait_until(
            rs_per_node_signal + remote_node_id,
            libshmem_device.NVSHMEM_CMP_EQ,
            1,
        )
        libshmem_device.putmem_block(
            output + node_id * nelem_per_rank,
            input + remote_node_id * nelem_per_rank,
            nelem_per_rank * elem_size,
            remote_rank,
        )


################### kernel calls ###################
def topk_reduce_scatter_reduce_for_each_node(
    rank,
    world_size,
    local_world_size,
    M,
    N,
    TOP_K,
    local_tensor: torch.Tensor,  # Output of GroupGEMM
    rs_buffers: List[torch.Tensor],  # [M, N] for each rank
    rs_buffer_ptrs: torch.Tensor,
    rs_per_node_buffer: torch.Tensor,  # [M // local_world_size, N]
    barrier_gemm_scatter_ready: torch.Tensor,
    rs_per_node_signal_buf: torch.Tensor,
    barrier: BarrierAllContext,
    rs_stream: torch.cuda.Stream,
    reduction_stream: torch.cuda.Stream,
):
    local_rank = rank % local_world_size
    nnodes = world_size // local_world_size
    node_id = rank // local_world_size
    M_per_node = M // nnodes
    M_per_rank = M // world_size

    with torch.cuda.stream(rs_stream):
        grid = lambda _: (128, 1, 1)

        kernel_consumer_topk_reduce_scatter_intra_node[grid](
            local_world_size,
            local_tensor,
            rs_buffer_ptrs,
            barrier_gemm_scatter_ready,
            M,
            N,
            16,  # num pid m
            8,  # num pid n
            TOP_K,  # REDUCE_LENGTH: tl.constexpr,
            128,  # BLOCK_M: tl.constexpr,
            128,  # BLOCK_N: tl.constexpr,
            True,  # perfect_tile: tl.constexpr,
            is_power_of_two(TOP_K),  # use_tl_reduce: tl.constexpr,
            num_warps=32,
        )

        barrier_all_on_stream(barrier, rs_stream)
        reduction_stream.wait_stream(rs_stream)

    with torch.cuda.stream(reduction_stream):

        for n in range(0, nnodes):
            cur_node_id = (node_id + n + 1) % nnodes
            rs_buffer_cur_node = rs_buffers[local_rank][cur_node_id * M_per_node:(cur_node_id + 1) * M_per_node]
            rs_per_node_buffer_cur_node = rs_per_node_buffer[cur_node_id * M_per_rank:(cur_node_id + 1) * M_per_rank]

            grid = lambda META: (triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )

            kernel_consumer_reduce[grid](
                local_world_size,
                rs_buffer_cur_node,  # c_ptr
                rs_per_node_buffer_cur_node,  # out_ptr
                M,
                N,
                N,  # stride_m
                1,  # stride_n
                128,  # BLOCK_M
                128,  # BLOCK_N
                num_warps=32,
            )

            set_signal(rs_per_node_signal_buf[cur_node_id].data_ptr(), 1, reduction_stream, require_i64=True)

    return rs_per_node_buffer[:M_per_rank * nnodes]


def p2p_inter_node(
    rank,
    world_size,
    local_world_size,
    input,
    output,
    rs_per_node_signal_buf,
    stream,
):
    nnodes = world_size // local_world_size
    node_id = rank // local_world_size
    if nnodes == 1:
        wait_eq(
            rs_per_node_signal_buf[node_id].data_ptr(),
            1,
            stream,
            require_i64=True,
        )
        return input
    M, N = input.shape
    M_per_rank = M // nnodes
    with torch.cuda.stream(stream):
        grid = lambda META: (nnodes - 1, )
        kernel_inter_node_p2p_for_same_local_rank[grid](
            local_world_size,
            M_per_rank,
            N,
            input,
            output,
            rs_per_node_signal_buf,
            num_warps=16,
        )
        wait_eq(rs_per_node_signal_buf[node_id].data_ptr(), 1, stream, require_i64=True)
        output[M_per_rank * node_id:M_per_rank * (node_id + 1)].copy_(input[M_per_rank * node_id:M_per_rank *
                                                                            (node_id + 1)])
    return output[:M_per_rank * nnodes]


def consumer_reduce_scatter_reduce_2d(
    rank,
    world_size,
    local_world_size,
    M,
    N,
    TOP_K,
    local_tensor: torch.Tensor,
    rs_buffers: List[torch.Tensor],
    rs_buffer_ptrs: torch.Tensor,
    rs_per_node_buffer: torch.Tensor,
    p2p_buffer: torch.Tensor,
    barrier_gemm_scatter_ready: torch.Tensor,
    symm_rs_per_node_signal_buffer: torch.Tensor,
    barrier: BarrierAllContext,
    rs_stream: torch.cuda.Stream,
    reduction_stream: torch.cuda.Stream,
    p2p_stream: torch.cuda.Stream,
):
    M_per_rank = M // world_size
    node_id = rank // local_world_size
    nnodes = world_size // local_world_size

    reduction_stream.wait_stream(rs_stream)
    nvshmem_barrier_all_on_stream(rs_stream)
    p2p_stream.wait_stream(rs_stream)
    rs_result_intra_node = topk_reduce_scatter_reduce_for_each_node(
        rank,
        world_size,
        local_world_size,
        M,
        N,
        TOP_K,
        local_tensor,
        rs_buffers,
        rs_buffer_ptrs,
        rs_per_node_buffer,
        barrier_gemm_scatter_ready,
        symm_rs_per_node_signal_buffer,
        barrier,
        rs_stream,
        reduction_stream,
    )
    p2p_result = p2p_inter_node(
        rank,
        world_size,
        local_world_size,
        rs_result_intra_node,
        p2p_buffer,
        symm_rs_per_node_signal_buffer,
        p2p_stream,
    )
    rs_stream.wait_stream(p2p_stream)
    nvshmem_barrier_all_on_stream(rs_stream)
    output = torch.empty((M_per_rank, N), dtype=local_tensor.dtype, device=local_tensor.device)
    with torch.cuda.stream(rs_stream):
        ring_reduce(
            p2p_result,
            output,
            node_id,
            nnodes,
        )
    return output


def moe_reduce_rs_rowise(
    rank: int,
    world_size: int,
    local_world_size: int,
    # input
    a: torch.Tensor,
    b: torch.Tensor,
    # context
    ctx: MoEReduceRSContext,
):
    padded_EM, K_per_rank = a.shape
    E = b.shape[0]
    M, topk = ctx.precompute_ctx.full_topk_ids.shape
    dtype = a.dtype
    assert dtype in [torch.bfloat16, torch.float16], "Currently only used for float16 or bfloat16"
    assert a.dtype == b.dtype
    assert a.ndim == 2 and a.is_cuda
    assert b.ndim == 3 and b.is_cuda

    GEMM_BLOCK_M = ctx.dataflow_config.GEMM_BLOCK_M
    GEMM_BLOCK_N = ctx.dataflow_config.GEMM_BLOCK_N
    GEMM_BLOCK_K = ctx.dataflow_config.GEMM_BLOCK_K
    GROUP_SIZE_M = ctx.dataflow_config.GROUP_SIZE
    num_stages = ctx.dataflow_config.num_stages
    num_warps = ctx.dataflow_config.num_warps

    ctx.symm_barrier_gemm_scatter_counter.zero_()
    ctx.symm_barrier_gemm_scatter_ready.zero_()
    ctx.rs_stream.wait_stream(torch.cuda.current_stream())
    ctx.reduction_stream.wait_stream(torch.cuda.current_stream())
    ctx.p2p_stream.wait_stream(torch.cuda.current_stream())
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    full_sorted_token_ids = ctx.precompute_ctx.full_sorted_token_ids
    full_token_expert_ids = ctx.precompute_ctx.full_token_expert_ids
    block_wait_barriers = ctx.precompute_ctx.block_wait_barriers
    rank_block_num = ctx.precompute_ctx.rank_block_num
    full_num_tokens_post_padded_list = (ctx.precompute_ctx.full_num_tokens_post_padded_list)
    EM = ctx.precompute_ctx.EM
    full_numel = ctx.precompute_ctx.full_numel

    (
        _,
        _,
        N,
    ) = b.shape

    grid = lambda META: (triton.cdiv(EM, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )

    kernel_producer_group_gemm_tp_scatter_input[grid](
        local_world_size,
        a,
        b,
        ctx.final_output_buffer,
        full_sorted_token_ids,
        full_token_expert_ids,
        ctx.precompute_ctx.full_topk_weight,
        full_num_tokens_post_padded_list,
        block_wait_barriers,
        rank_block_num,
        ctx.symm_barrier_gemm_scatter_counter,
        ctx.symm_barrier_gemm_scatter_ready_ptrs,
        full_numel,
        EM,
        N,
        K_per_rank,
        E,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        N,
        1,
        GEMM_BLOCK_M,
        GEMM_BLOCK_N,
        GEMM_BLOCK_K,
        GROUP_SIZE_M,
        topk,
        torch_dtype_to_triton_dtype(a.dtype),
        num_stages=num_stages,
        num_warps=num_warps,
    )

    with torch.cuda.stream(ctx.rs_stream):
        output = consumer_reduce_scatter_reduce_2d(
            rank,
            world_size,
            local_world_size,
            M,
            N,
            topk,
            ctx.final_output_buffer,
            ctx.symm_rs_buffers,
            ctx.symm_rs_buffer_ptrs,
            ctx.symm_rs_per_node_buffer,
            ctx.symm_p2p_buffer,
            ctx.symm_barrier_gemm_scatter_ready,
            ctx.symm_rs_per_node_signal,
            ctx.barrier,
            ctx.rs_stream,
            ctx.reduction_stream,
            ctx.p2p_stream,
        )

    torch.cuda.current_stream().wait_stream(ctx.rs_stream)
    torch.cuda.current_stream().wait_stream(ctx.reduction_stream)
    torch.cuda.current_stream().wait_stream(ctx.p2p_stream)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    return output


@dataclass
class MoEReduceRSColwiseContext:
    max_M: int
    N: int
    num_experts: int
    topk: int
    dtype: torch.dtype
    # distributed arguments
    rank: int
    num_ranks: int
    num_local_ranks: int
    local_rank: int = field(init=False)
    nnodes: int = field(init=False)
    n_split_max: int = 8
    # barriers
    grid_barrier: torch.Tensor = field(init=False)
    ntiles_counter: torch.Tensor = field(init=False)
    # symmetric buffers or non-symmetric buffers
    symm_barriers: List[torch.Tensor] = field(default_factory=list)
    symm_barrier: torch.Tensor = field(init=False)
    symm_reduce_scatter_buffers: List[torch.Tensor] = field(default_factory=list)
    symm_reduce_scatter_buffer: torch.Tensor = field(init=False)

    # stream
    reduce_stream: torch.cuda.Stream = field(default_factory=lambda: torch.cuda.Stream(priority=-1))

    def __post_init__(self):
        assert self.dtype in [torch.bfloat16, torch.float16], "Currently only used for float16 or bfloat16"
        assert self.max_M % self.topk == 0, "M must be divisible by topk"
        self.local_rank = self.rank % self.num_local_ranks
        self.nnodes = self.num_ranks // self.num_local_ranks
        # Create a barrier for grid synchronization
        self.grid_barrier = torch.zeros(
            (1, ),
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        self.ntiles_counter = torch.zeros((self.n_split_max, ), dtype=torch.int32, device=torch.cuda.current_device())
        self.symm_barriers = nvshmem_create_tensors((self.num_ranks, ), torch.int32, self.rank, self.num_local_ranks)
        self.symm_barrier = self.symm_barriers[self.local_rank]
        self.symm_barrier.zero_()
        ntokens = self.max_M // self.topk
        self.symm_reduce_scatter_buffers = nvshmem_create_tensors((ntokens, self.N), self.dtype, self.rank,
                                                                  self.num_local_ranks)
        self.symm_reduce_scatter_buffer = self.symm_reduce_scatter_buffers[self.local_rank]

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    def finalize(self):
        nvshmem_free_tensor_sync(self.symm_barrier)
        nvshmem_free_tensor_sync(self.symm_reduce_scatter_buffer)


def create_moe_rs_context_colwise(rank, world_size, local_world_size, max_token_num, hidden_dim, num_experts, topk,
                                  input_dtype):
    return MoEReduceRSColwiseContext(max_token_num, hidden_dim, num_experts, topk, dtype=input_dtype, rank=rank,
                                     num_ranks=world_size, num_local_ranks=local_world_size, n_split_max=8)


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


@triton.jit
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
    A_stride_m,
    A_stride_k,
    B_stride_e,
    B_stride_k,
    B_stride_n,
    C_stride_m,
    C_stride_n,
    tile_counter_ptr,
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
    a_ptrs = A_ptr + offs_gather_a[:, None] * A_stride_m + offs_k[None, :] * A_stride_k

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_be = tl.load(expert_index_ptr + pid_m)
    b_ptrs = B_ptr + offs_be * B_stride_e + offs_k[:, None] * B_stride_k + offs_bn[None, :] * B_stride_n

    if A_ptr.dtype.element_ty == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
        tl.static_assert(False, "int8 is not supported in this kernel, please use float16 or bfloat16")
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K))

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * A_stride_k
        b_ptrs += BLOCK_SIZE_K * B_stride_k

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + offs_gather_a[:, None] * C_stride_m + offs_cn[None, :] * C_stride_n
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if A_scale_ptr:
        accumulator = accumulator * tl.load(A_scale_ptr + offs_gather_a[:, None], mask=token_mask[:, None])
    accumulator = accumulator.to(A_ptr.dtype.element_ty)
    tl.store(c_ptrs, accumulator, mask=c_mask)

    thread_idx = tid(axis=0)
    __syncthreads()
    if thread_idx == 0:
        count = atomic_add(tile_counter_ptr + group_id, 1, semantic="release", scope="gpu")
        if count == num_block_m * GROUP_SIZE_N - 1:
            atomic_add(barrier_ptr + group_id, 1, semantic="release", scope="sys")


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


@triton.jit
def reduce_scatter_multimem_kernel(
    symm_input_ptr,
    output_ptr,
    M,
    N,
    stride_m,
    stride_n,
):
    # N % BLOCK_SIZE_N is expected.
    pid = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)
    thread_idx = tid(axis=0)
    block_dim = ntid(axis=0)
    mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_input_ptr)

    ELEMS_PER_THREAD: tl.constexpr = 128 // symm_input_ptr.dtype.element_ty.primitive_bitwidth

    # 16 byte per thread: as a uint4
    for k in range(pid * block_dim + thread_idx, M * N // ELEMS_PER_THREAD, block_dim * npid):
        offs_m = k * ELEMS_PER_THREAD // N
        offs_n = k * ELEMS_PER_THREAD % N
        if offs_m < M and offs_n < N:
            offs = offs_m * stride_m + offs_n * stride_n
            val0, val1, val2, val3 = multimem_ld_reduce_v4(mc_ptr + offs)
            st_v4_b32(output_ptr + offs, val0, val1, val2, val3)


@triton.jit
def reduce_scatter_a2a_kernel(
    symm_input_ptr,
    output_ptr,
    M,
    N,
    stride_m,
    stride_n,
    rank,
    num_ranks,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    nblocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    nblocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    nblocks = nblocks_m * nblocks_n
    pid = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)

    for bid in range(pid, nblocks, npid):
        bid_m = bid // nblocks_n
        bid_n = bid % nblocks_n
        offs_m = bid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < M
        offs_n = bid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < N
        offs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        mask = mask_m[:, None] & mask_n[None, :]
        val = tl.load(symm_input_ptr + offs, mask=mask)
        for n in range(1, num_ranks):
            peer = (rank + n) % num_ranks
            ptr_peer = libshmem_device.remote_ptr(symm_input_ptr, peer)
            ptr_peer = tl.multiple_of(ptr_peer, 16)
            x = tl.load(ptr_peer + offs, mask=mask)
            val += x

        tl.store(output_ptr + offs, val, mask=mask)


@triton.jit(do_not_specialize=["rank"])
def reduce_topk_reduce_scatter_intra_node(
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
    barrier_ptr,
    grid_barrier_ptr,
    N_CHUNKS,  # N_SPLIT is used to distinguish different groups
    TOPK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    symm_buffer = reduce_scatter(input)
    output = reduce_scatter(symm_buffer)
    """
    pid = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)
    N_per_chunk = N // N_CHUNKS
    N_per_chunk = tl.multiple_of(N_per_chunk, 16)
    ntokens_per_rank = ntokens // num_ranks
    for n_chunk in tl.range(0, N_CHUNKS, step=1, loop_unroll_factor=1):
        token = dl.wait(barrier_ptr + n_chunk, 1, scope="gpu", semantic="acquire", waitValue=1)
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
    barrier_on_this_grid(grid_barrier_ptr)
    if pid == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_barrier_ptr)

    # reduce symm_buffer_n_ptr to output_n_ptr
    blocks_n_per_chunk = tl.cdiv(N_per_chunk, BLOCK_SIZE_N)
    blocks_m_per_rank = tl.cdiv(ntokens_per_rank, BLOCK_SIZE_M)
    nblocks_per_split_per_rank = blocks_m_per_rank * blocks_n_per_chunk
    # want to save some registers, but loop_unroll_factor does not work here
    for n_chunk in tl.range(0, N_CHUNKS, step=1, loop_unroll_factor=1):
        offs_n_chunk = n_chunk * N_per_chunk * stride_n
        output_n_ptr = output_ptr + offs_n_chunk
        reduced_topk_this_chunk_ptr = symm_reduced_topk_ptr + offs_n_chunk
        for tile_id in range(pid, nblocks_per_split_per_rank, npid):
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
def reduce_topk_reduce_scatter_intra_node_a2a_read(
        input_ptr,  # of shape [ntokens, topk, N]
        expert_weight_ptr,
        # output
        symm_reduce_topk_ptr,  # of shape [ntokens, N]
        output_ptr,  # of shape [ntokens // num_rank, N]
        # args
    ntokens, N, stride_m, stride_n, rank, num_ranks,
        #
        barrier_ptr, grid_barrier_ptr, TOPK: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        N_CHUNKS: tl.constexpr,  # N_SPLIT is used to distinguish different groups
):
    """
    symm_buffer = reduce_scatter(input)
    output = reduce_scatter(symm_buffer)
    """
    pid = tl.program_id(axis=0)
    N_per_chunk = N // N_CHUNKS
    ntokens_per_rank = ntokens // num_ranks
    for chunk_id in range(0, N_CHUNKS, 1):
        token = dl.wait(barrier_ptr + chunk_id, 1, scope="gpu", semantic="acquire", waitValue=1)
        offs_n = chunk_id * N_per_chunk * stride_n
        input_this_chunk_ptr = dl.consume_token(input_ptr + offs_n * TOPK, token)
        output_this_chunk_ptr = dl.consume_token(output_ptr + offs_n, token)
        symm_reduced_topk_this_chunk_ptr = dl.consume_token(symm_reduce_topk_ptr + offs_n, token)
        reduce_topk_kernel(
            input_this_chunk_ptr,
            expert_weight_ptr,
            symm_reduced_topk_this_chunk_ptr,
            ntokens,
            N_per_chunk,
            stride_m,
            stride_n,
            TOPK,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )
        # TODO(houqi.1993) replace nvshmem_barrier_block with another memory flag
        barrier_on_this_grid(grid_barrier_ptr)
        if pid == 0:
            libshmem_device.barrier_all_block()
        barrier_on_this_grid(grid_barrier_ptr)
        # do reduce_scatter
        reduce_scatter_a2a_kernel(
            symm_reduced_topk_this_chunk_ptr + rank * ntokens_per_rank * stride_m,
            output_this_chunk_ptr,
            ntokens_per_rank,
            N_per_chunk,
            stride_m,
            stride_n,
            rank,
            num_ranks,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )

    if pid == 0:
        libshmem_device.barrier_all_block()


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


def moe_reduce_rs_colwise(
        # input
        x: torch.Tensor, weights: torch.Tensor, choosed_experts: torch.Tensor, expert_weight: torch.Tensor,
        # context
        ctx: MoEReduceRSColwiseContext, n_chunks=2, config: Optional[triton.Config] = None):
    assert x.ndim == 2 and x.is_cuda, "Input x must be a 2D CUDA tensor"
    assert weights.ndim == 3 and weights.is_cuda, "Weights must be a 3D CUDA tensor"
    assert choosed_experts.ndim == 2 and choosed_experts.is_cuda, "Choosed experts must be a 2D CUDA tensor"
    M, K_per_rank = x.shape
    assert M <= ctx.max_M, f"Input M({M}) must be less than or equal to context M({ctx.max_M})"
    assert M % ctx.topk == 0, f"M({M}) must be divisible by topk({ctx.topk})"

    ntokens = M // ctx.topk
    assert choosed_experts.shape == (ntokens, ctx.topk), \
        f"Choosed experts shape {choosed_experts.shape} must match (ntokens({ntokens}), topk({ctx.topk}))"

    assert weights.shape == (ctx.num_experts, K_per_rank, ctx.N), \
        f"Weights shape {weights.shape} must match context num_experts({ctx.num_experts}), K_per_rank({K_per_rank}), and N({ctx.N})"
    assert x.dtype == weights.dtype == ctx.dtype, f"Input x dtype({x.dtype}) must match context dtype({ctx.dtype}) and weights dtype({weights.dtype})"

    ntokens_per_rank = ntokens // ctx.num_ranks
    N_per_chunk = ctx.N // n_chunks
    config = config or get_auto_triton_config(M, ctx.N, K_per_rank, n_chunks, x.dtype)
    block_size_m = config.kwargs["BLOCK_SIZE_M"]
    # TODO(houqi.1993) maybe we can pass this as argument and avoid recompute it again
    _, _, gather_index, expert_index, M_pad_gpu = calc_gather_scatter_index_triton(choosed_experts, ctx.num_experts,
                                                                                   block_size_m)

    grouped_gemm_out = torch.empty(
        (M, ctx.N),
        dtype=ctx.dtype,
        device=torch.cuda.current_device(),
    )
    out = torch.empty((ntokens_per_rank, ctx.N), dtype=ctx.dtype, device=torch.cuda.current_device())

    M_pad_approx = (triton.cdiv(M, block_size_m) + ctx.num_experts) * block_size_m

    ctx.reduce_stream.wait_stream(torch.cuda.current_stream())
    moe_gather_rs_grouped_gemm(x, weights, grouped_gemm_out, expert_weight, gather_index, expert_index, M_pad_gpu,
                               M_pad_approx, ctx.N, K_per_rank, ctx.num_experts, ctx.topk, ctx.ntiles_counter,
                               ctx.symm_barrier, config)

    with torch.cuda.stream(ctx.reduce_stream):
        reduce_topk_reduce_scatter_intra_node[(32, )](
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
            ctx.symm_barrier,
            ctx.grid_barrier,
            TOPK=ctx.topk,
            BLOCK_SIZE_M=max(1, 16 * 1024 // N_per_chunk // x.itemsize),  # each thread with a uint4 load
            BLOCK_SIZE_N=N_per_chunk,
            N_CHUNKS=n_chunks,
            num_warps=32,
            launch_cooperative_grid=True,
        )
        # print("reduce_topk_reduce_scatter_intra_node done")

    torch.cuda.current_stream().wait_stream(ctx.reduce_stream)
    # print("expert_index", expert_index)
    # print("ntiles_counter", ctx.ntiles_counter)
    # print("symm_barrier", ctx.symm_barrier)
    # print("grid_barrier", ctx.grid_barrier)
    # torch.save(choosed_experts, f"choosed_experts_{ctx.rank}.pt")
    # torch.save(gather_index, f"gather_index_{ctx.rank}.pt")
    # torch.save(grouped_gemm_out, f"grouped_gemm_out_{ctx.rank}.pt")
    # torch.save(expert_weight, f"expert_weight_{ctx.rank}.pt")
    # torch.save(ctx.symm_reduce_scatter_buffer, f"reduced_triton_{ctx.rank}.pt")
    return out
