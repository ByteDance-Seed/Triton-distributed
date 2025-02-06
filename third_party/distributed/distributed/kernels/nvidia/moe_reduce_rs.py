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
import triton
import triton.language as tl
import triton.distributed.language as dl

from typing import List

from dataclasses import dataclass

from triton.language.extra.cuda.language_extra import (
    atomic_add,
    __syncthreads,
    __tid__,
)


################### helper functions ###################
def ceil_div(a, b):
    return (a + b - 1) // b


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


@triton.jit
def get_tid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def get_ntid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def get_flat_tid():
    tid_x, tid_y, tid_z = get_tid()
    ntid_x, ntid_y, _ = get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@tl.core.extern
def atomic_cas(
    ptr,
    value,
    target_value,
    scope: tl.constexpr,
    semantic: tl.constexpr,
    _builder=None,
):
    return tl.inline_asm_elementwise(
        asm=f"atom.{semantic.value}.{scope.value}.global.cas.b32 $0, [$1], $2, $3;",
        constraints=("=r,l,r,r"),
        args=[
            ptr,
            value,
            target_value,
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def barrier_all(rank, num_ranks, comm_buf_ptr):
    tid = __tid__(axis="x").to(tl.int32)
    sm_id = tl.program_id(axis=0)
    if tid < num_ranks:
        remote_ptr = dl.symm_at(comm_buf_ptr + sm_id * num_ranks + rank, tid)
        while atomic_cas(remote_ptr, 0, 1, "sys", "release") != 0:
            pass
        while (atomic_cas(comm_buf_ptr + sm_id * num_ranks + tid, 1, 0, "sys", "acquire") != 1):
            pass
    __syncthreads()


def barrier_all_on_stream(rank, num_ranks, sync_buf, stream):

    with torch.cuda.stream(stream):
        barrier_all[(1, )](rank, num_ranks, sync_buf)


################### compute ctx ###################
@dataclass
class MoEAgScatterGroupGemmPrecomputeContext:
    full_topk_weight = None
    full_topk_ids = None
    full_sorted_token_ids = None
    full_token_expert_ids = None
    block_wait_barriers = None
    rank_block_num = None
    full_num_tokens_post_padded_list = None
    EM: int = 0
    full_numel: int = 0
    TOP_K: int = 0
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
    tokens_per_thread = ceil_div(numel, num_experts)

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
            cumsum[i] = cumsum[i - 1] + ceil_div(token_cnts[num_experts, i - 1], block_size) * block_size
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
    ctx.TOP_K = topk
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


def create_moe_rs_context(pg, num_ranks, num_experts, topk, input_dtype, device, moe_block_size, router_logits):
    num_tokens_per_rank = router_logits.shape[0]
    return precompute_context_helper(
        pg,
        num_ranks,
        topk,
        num_tokens_per_rank,
        num_experts,
        input_dtype,
        device,
        router_logits,
        BLOCK_M=moe_block_size,
    )


################### triton kernel ###################
@triton.jit
def kernel_producer_group_gemm_tp(
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
    stride_a_m,
    stride_a_k,
    stride_b_e,
    stride_b_k,
    stride_b_n,
    stride_c_m,
    stride_c_n,
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
    num_block_m_per_rank = num_block_m // num_ranks
    m_offset = num_block_m_per_rank * ((rank + 3) % num_ranks)
    pid_m = (pid_m + m_offset) % num_block_m

    num_tokens_post_padded_value = tl.load(num_tokens_post_padded)

    if pid_m * BLOCK_M >= num_tokens_post_padded_value:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (a_ptr + offs_am[:, None] * stride_a_m + offs_k[None, :] * stride_a_k)

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_be = tl.load(token_expert_ids_ptr + pid_m)

    b_ptrs = (b_ptr + offs_be * stride_b_e + offs_k[:, None] * stride_b_k + offs_bn[None, :] * stride_b_n)

    moe_weight = tl.load(topk_weight_ptr + offs_token, mask=token_mask, other=0)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K_per_rank, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < EM) & (offs_k[None, :] < K_per_rank - k * BLOCK_K),
        )
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K_per_rank - k * BLOCK_K))

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_a_k
        b_ptrs += BLOCK_K * stride_b_k

    accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_dtype)

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = (c_ptr + offs_token[:, None] * stride_c_m + offs_cn[None, :] * stride_c_n)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

    offs_counter = tl.load(block_wait_barrier_ptr + pid_m)
    offs_ready = tl.load(block_wait_barrier_ptr + pid_m)
    threshold = tl.load(rank_block_num + offs_counter) * num_block_n
    counter_ptr = barrier_counter + offs_counter
    remote_barrier_ready_ptr = tl.load(barriers_ready_ptrs + rank).to(tl.pointer_type(tl.uint64))
    __syncthreads()
    tid = __tid__(axis="x")
    value = 1
    if tid == 0:
        if atomic_add(counter_ptr, value, "gpu", "relaxed") == threshold - 1:
            dl.notify(remote_barrier_ready_ptr + offs_ready, rank, signal=1, sig_op="add", comm_scope="gpu")


@triton.jit
def kernel_producer_group_gemm_tp_scatter_input(
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
    num_block_m_per_rank = num_block_m // num_ranks
    m_offset = num_block_m_per_rank * ((rank + 5) % num_ranks)
    pid_m = (pid_m + m_offset) % num_block_m

    num_tokens_post_padded_value = tl.load(num_tokens_post_padded)

    if pid_m * BLOCK_M >= num_tokens_post_padded_value:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    # offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

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
            mask=(offs_bn[None, :] < N and offs_k[:, None] < K_per_rank - k * BLOCK_K),
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
    remote_barrier_ready_ptr = tl.load(barriers_ready_ptrs + rank).to(tl.pointer_type(tl.uint64))
    __syncthreads()
    tid = __tid__(axis="x")
    value = 1
    if tid == 0:
        if atomic_add(counter_ptr, value, "gpu", "relaxed") == threshold - 1:
            dl.notify(remote_barrier_ready_ptr + offs_ready, rank, signal=1, sig_op="add", comm_scope="gpu")


@triton.jit
def kernel_consumer_topk_reduce_scatter(
    rank,
    num_ranks,
    consumer_output_ptr,  # shape: [M * ReduceLength, N]
    remote_buffer_ptrs,  # each tensor shape: [M, N]
    barrier_gemm_scatter_ready,
    barrier_scatter_reduce_counter,
    barriers_scatter_reduce_ready_ptrs,
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
    dtype = consumer_output_ptr.dtype.element_ty
    M_per_rank = M // num_ranks
    num_blocks_m_per_rank = tl.cdiv(M_per_rank, BLOCK_M)

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

            mid = (mid + rank * num_blocks_m_per_rank) % num_block_m

            to_rank = mid // num_blocks_m_per_rank

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
            # for to_rank in range(world_size):
            dst_ptr = tl.load(remote_buffer_ptrs + to_rank).to(tl.pointer_type(dtype))
            offs_out_m = (rank * num_blocks_m_per_rank + mid % num_blocks_m_per_rank) * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_out_n = nid * BLOCK_N + tl.arange(0, BLOCK_N)
            dst_ptrs = dst_ptr + offs_out_m[:, None] * N + offs_out_n[None, :]
            tl.store(dst_ptrs, reduce_data)

            barrier_ready_ptr = tl.load(barriers_scatter_reduce_ready_ptrs + to_rank).to(tl.pointer_type(tl.uint64))
            __syncthreads()
            value = 1
            if get_flat_tid() == 0:
                if (atomic_add(barrier_scatter_reduce_counter + to_rank, value, "gpu",
                               "release") == num_blocks_m_per_rank * num_block_n - 1):
                    dl.notify(barrier_ready_ptr + rank, rank, signal=1, sig_op="add", comm_scope="gpu")


@triton.jit
def kernel_consumer_reduce(
    c_ptr,
    out_ptr,
    # barrier
    barrier_ready,
    # shape of matrix
    M,
    N,
    # strides
    stride_m,
    stride_n,
    # reduce tile shape
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ignore_self_rank: tl.constexpr,
):
    rank = dl.rank()
    num_ranks = dl.num_ranks()
    m_per_rank = tl.cdiv(M, num_ranks)
    pid = tl.program_id(axis=0)
    # reduce_m_blocks_per_rank = tl.cdiv(m_per_rank, BLOCK_M)
    reduce_n_blocks_per_rank = tl.cdiv(N, BLOCK_N)
    pid_m = pid // reduce_n_blocks_per_rank
    pid_n = pid % reduce_n_blocks_per_rank

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    out_ptrs = out_ptr + (offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)

    if ignore_self_rank:
        org_data = tl.load(out_ptrs)
    else:
        org_data = tl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.dtype.element_ty)

    for rid in range(0, num_ranks):
        swizzle_rid = (rid + rank) % num_ranks
        if not ((swizzle_rid == rank) and ignore_self_rank):
            full_offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M) + swizzle_rid * m_per_rank) % M
            offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
            ptrs = c_ptr + (full_offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
            token = dl.wait(barrier_ready + swizzle_rid, 1, "gpu", "acquire")
            ptrs = dl.consume_token(ptrs, token)
            data = tl.load(ptrs)
            org_data += data

    tl.store(out_ptrs, org_data)


################### kernel calls ###################
def consumer_reduce_scatter_reduce_all2all_push(
    rank,
    num_ranks,
    M,
    N,
    TOP_K,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    remote_tensor_buffer_ptrs: torch.Tensor,
    barrier_gemm_scatter_ready: torch.Tensor,
    barrier_scatter_reduce_counter: torch.Tensor,
    barriers_scatter_reduce_ready_ptrs: torch.Tensor,
    barrier_scatter_reduce_ready: torch.Tensor,
    scatter_stream: torch.cuda.Stream,
    reduce_stream: torch.cuda.Stream,
):
    M_per_rank = M // num_ranks

    with torch.cuda.stream(scatter_stream):
        grid = lambda _: (128, 1, 1)

        kernel_consumer_topk_reduce_scatter[grid](
            rank,
            num_ranks,
            local_tensor,  # shape: [M * ReduceLength, N]
            remote_tensor_buffer_ptrs,
            # torch.tensor(
            #     [t.data_ptr() for t in remote_tensor_buffers]
            # ).cuda(),  # each tensor shape: [M, N]
            barrier_gemm_scatter_ready,
            barrier_scatter_reduce_counter,
            barriers_scatter_reduce_ready_ptrs,
            M,
            N,
            16,  # num pid m
            8,  # num pid n
            # constants
            TOP_K,  # REDUCE_LENGTH: tl.constexpr,
            128,  # BLOCK_M: tl.constexpr,
            128,  # BLOCK_N: tl.constexpr,
            True,  # perfect_tile: tl.constexpr,
            is_power_of_two(TOP_K),  # use_tl_reduce: tl.constexpr,
            num_warps=32,
        )

    with torch.cuda.stream(reduce_stream):
        grid = lambda META: (triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
        kernel_consumer_reduce[grid](
            remote_tensor_buffers[rank], local_tensor[rank * M_per_rank:(rank + 1) * M_per_rank],
            barrier_scatter_reduce_ready, M, N, N,  # stride_m
            1,  # stride_n
            128,  # BLOCK_M
            128,  # BLOCK_N
            False,  # ignore self rank
            num_warps=32)


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


def moe_reduce_rs_intra_node(
    rank,
    num_ranks,
    # input and output
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    b: torch.Tensor,
    c_local: torch.Tensor,
    c_buffers: List[torch.Tensor],
    c_buffer_ptrs: torch.Tensor,
    # stream
    scatter_stream: torch.cuda.Stream,
    reduce_stream: torch.cuda.Stream,
    # barrier
    barrier_gemm_scatter_counter: torch.Tensor,
    barriers_gemm_scatter_ready_ptrs: torch.Tensor,
    barrier_gemm_scatter_ready: torch.Tensor,
    barrier_scatter_reduce_counter: torch.Tensor,
    barriers_scatter_reduce_ready_ptrs: torch.Tensor,
    barrier_scatter_reduce_ready: torch.Tensor,
    sync_buf,
    # config
    dataflow_config: DataflowConfig,
    pre_computed_ctx: MoEAgScatterGroupGemmPrecomputeContext,
    # option
    dump_ir=False,
    debug_sync=False,
    bypass_comm=False,
    no_input_scatter=False,
    do_initial_sync=True,
    do_final_sync=True,
):
    padded_EM, K_per_rank = a.shape
    # K = (K_per_rank * num_ranks, )
    E = b.shape[0]
    M = topk_ids.shape[0]
    # M_per_rank = M // num_ranks
    TOP_K = topk_ids.shape[1]
    dtype = a.dtype
    assert (dtype == torch.float16 or dtype == torch.float8_e4m3fn), "Currently only used for float16 or float8_e4m3fn"
    assert a.dtype == b.dtype

    GEMM_BLOCK_M = dataflow_config.GEMM_BLOCK_M
    GEMM_BLOCK_N = dataflow_config.GEMM_BLOCK_N
    GEMM_BLOCK_K = dataflow_config.GEMM_BLOCK_K
    GROUP_SIZE_M = dataflow_config.GROUP_SIZE
    num_stages = dataflow_config.num_stages
    num_warps = dataflow_config.num_warps
    # RS_BLOCK_M = dataflow_config.RS_BLOCK_M
    # RS_BLOCK_N = dataflow_config.RS_BLOCK_N

    compiled = None

    if do_initial_sync:
        barrier_gemm_scatter_counter.zero_()
        barrier_gemm_scatter_ready.zero_()
        barrier_scatter_reduce_counter.zero_()
        barrier_scatter_reduce_ready.zero_()
        scatter_stream.wait_stream(torch.cuda.current_stream())
        reduce_stream.wait_stream(torch.cuda.current_stream())
        barrier_all_on_stream(rank, num_ranks, sync_buf, torch.cuda.current_stream())

    with torch.cuda.stream(torch.cuda.current_stream()):
        full_sorted_token_ids = pre_computed_ctx.full_sorted_token_ids
        full_token_expert_ids = pre_computed_ctx.full_token_expert_ids
        block_wait_barriers = pre_computed_ctx.block_wait_barriers
        rank_block_num = pre_computed_ctx.rank_block_num
        full_num_tokens_post_padded_list = (pre_computed_ctx.full_num_tokens_post_padded_list)
        EM = pre_computed_ctx.EM
        full_numel = pre_computed_ctx.full_numel
        # ctx = pre_computed_ctx

        (
            _,
            _,
            N,
        ) = b.shape

        grid = lambda META: (triton.cdiv(EM, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )

        if no_input_scatter:
            compiled = kernel_producer_group_gemm_tp[grid](
                a,
                b,
                c_local,
                full_sorted_token_ids,
                full_token_expert_ids,
                topk_weights,
                full_num_tokens_post_padded_list,
                block_wait_barriers,
                rank_block_num,
                barrier_gemm_scatter_counter,
                barriers_gemm_scatter_ready_ptrs,
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
                TOP_K,
                torch_dtype_to_triton_dtype(a.dtype),
                num_stages=num_stages,
                num_warps=num_warps,
            )
        else:
            compiled = kernel_producer_group_gemm_tp_scatter_input[grid](
                a,
                b,
                c_local,
                full_sorted_token_ids,
                full_token_expert_ids,
                topk_weights,
                full_num_tokens_post_padded_list,
                block_wait_barriers,
                rank_block_num,
                barrier_gemm_scatter_counter,
                barriers_gemm_scatter_ready_ptrs,
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
                TOP_K,
                torch_dtype_to_triton_dtype(a.dtype),
                num_stages=num_stages,
                num_warps=num_warps,
            )

    # debug sync
    if debug_sync:
        scatter_stream.wait_stream(torch.cuda.current_stream())
        reduce_stream.wait_stream(scatter_stream)
        torch.cuda.current_stream().wait_stream(scatter_stream)
        torch.cuda.current_stream().wait_stream(reduce_stream)
        barrier_all_on_stream(rank, num_ranks, sync_buf, torch.cuda.current_stream())

    if bypass_comm:
        pass
    else:
        consumer_reduce_scatter_reduce_all2all_push(
            rank,
            num_ranks,
            M,
            N,
            TOP_K,
            c_local,
            c_buffers,
            c_buffer_ptrs,
            barrier_gemm_scatter_ready,
            barrier_scatter_reduce_counter,
            barriers_scatter_reduce_ready_ptrs,
            barrier_scatter_reduce_ready,
            scatter_stream,
            reduce_stream,
        )

    if dump_ir:
        if rank == 0:
            if compiled is not None:
                for suffix in ["ptx", "ttir", "ttgir", "llir"]:
                    with open(f"trace_{compiled.name}.{suffix}", "w") as fout:
                        print(compiled.asm[suffix], file=fout)

    if do_final_sync:
        torch.cuda.current_stream().wait_stream(scatter_stream)
        torch.cuda.current_stream().wait_stream(reduce_stream)
        barrier_all_on_stream(rank, num_ranks, sync_buf, torch.cuda.current_stream())


def get_dataflowconfig(
    GEMM_BLOCK_M: int,
    GEMM_BLOCK_N: int,
    GEMM_BLOCK_K: int,
    GROUP_SIZE: int,
    num_stages: int,
    num_warps: int,
    RS_BLOCK_M: int,
    RS_BLOCK_N: int,
):
    return DataflowConfig(GEMM_BLOCK_M, GEMM_BLOCK_N, GEMM_BLOCK_K, GROUP_SIZE, num_stages, num_warps, RS_BLOCK_M,
                          RS_BLOCK_N)
