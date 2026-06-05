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
"""
AMD (HIP/ROCm) fused EP All-to-All + grouped GEMM "mega kernel".

Fuses MoE token *dispatch* (expert-parallel all-to-all) with the expert *grouped
GEMM* into a single persistent kernel so that, as soon as an expert's tokens have
arrived from every source rank, its GEMM tiles can start -- overlapping
communication with compute at tile granularity. (Reference design:
``kernels/nvidia/ep_all2all_fused.py``.)

The forward path is two mega kernels: ``dispatch + grouped GEMM(1)`` and
``grouped GEMM(2) + combine`` (serial / gather combine). Cross-rank data movement
and synchronization use mori_shmem device primitives via ``libshmem_device``
(``putmem_warp`` / ``signal_op`` / ``signal_wait_until`` / ``barrier_all``);
symmetric buffers come from ``mori_shmem_create_tensor`` and a per-expert
``MORI_SHMEM_SIGNAL_DTYPE`` (uint64) barrier; the wavefront width is 64 and
grouped-GEMM tiles are plain pointer-arithmetic ``tl.dot``. The cross-rank split
exchange and grouped-GEMM tiling metadata are built on-device by default (kernels
``kernel_build_fused_dispatch_metadata`` / ``kernel_build_gemm_tiling``); pass
``use_device_metadata=False`` to fall back to the host torch path. The data plane
stays inside the fused kernels.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.hip.language_extra import (
    tid,
    __syncthreads,
    ld,
    st,
    atomic_add,
    atomic_add_per_warp,
)
from triton_dist.language.extra.language_extra import threads_per_warp
from triton_dist.kernels.amd.ep_a2a import bincount
from triton_dist.kernels.amd.common_ops import barrier_on_this_grid
from triton_dist.utils import (
    MORI_SHMEM_SIGNAL_DTYPE,
    mori_shmem_create_tensor,
    mori_shmem_free_tensor_sync,
    mori_shmem_barrier_all_on_stream,
)


# ---------------------------------------------------------------------------
#  Device helpers
# ---------------------------------------------------------------------------
@triton_dist.jit
def dot_k_const(
    a_ptrs,
    b_ptrs,
    c_ptrs,
    M,
    N,
    K: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    need_mask: tl.constexpr,
):
    """A single (BLOCK_M, BLOCK_N) GEMM tile with a K reduction loop.

    Plain ``tl.dot`` accumulation in fp32.
    """
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if need_mask:
            a = tl.load(
                a_ptrs, mask=((tl.arange(0, BLOCK_SIZE_M) < M)[:, None] &
                              (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K)[None, :]))
        else:
            a = tl.load(a_ptrs, mask=(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K)[None, :])
        b = tl.load(b_ptrs, mask=(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K)[:, None])

        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator.to(c_ptrs.dtype.element_ty)
    if need_mask:
        c_mask = (tl.arange(0, BLOCK_SIZE_M) < M)[:, None] & (tl.arange(0, BLOCK_SIZE_N) < N)[None, :]
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        c_mask = (tl.arange(0, BLOCK_SIZE_N) < N)[None, :]
        tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
#  Dispatch tile (producer): scatter local tokens to remote expert ranks
# ---------------------------------------------------------------------------
@triton_dist.jit(do_not_specialize=["pid", "num_pid"])
def tile_kernel_dispatch_token_intra_node(
    pid,
    num_pid,
    counter_ptr,  # local int32 [num_experts]: per-global-expert sent counter
    barriers_ptr,  # symm uint64 [experts_per_rank * world_size]: per (local_expert, src_rank)
    recv_buf_offset_per_expert,  # int32 [world_size, experts_per_rank, world_size] exclusive offsets
    local_splits_buf,  # int32 [num_experts]: this rank's token count per global expert
    send_buf,  # symm: this rank's tokens [max_tokens, hidden] (putmem source)
    output_buf,  # symm: per-rank recv region [cap_tokens, hidden] (putmem dest)
    topk_indices_tensor,  # int32 [num_tokens, topk]
    token_dst_scatter_idx,  # int32 [num_tokens, topk]
    num_input_tokens_per_rank,  # int32 [world_size]
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    experts_per_rank: tl.constexpr,
    WITH_SCATTER_INDICES: tl.constexpr,
    num_warps: tl.constexpr,
):
    WARP_SIZE: tl.constexpr = 64
    ELEMENT_SIZE: tl.constexpr = tl.constexpr(send_buf.dtype.element_ty.primitive_bitwidth) // 8
    bytes_per_token: tl.constexpr = hidden_size * ELEMENT_SIZE

    rank = dl.rank()
    world_size = dl.num_ranks()
    thread_idx = tid(0)
    lane_idx = thread_idx % WARP_SIZE
    warp_id = thread_idx // WARP_SIZE
    total_warps = num_warps * num_pid
    global_warp_id = pid * num_warps + warp_id

    token_num = tl.load(num_input_tokens_per_rank + rank)

    # Each warp cooperatively transfers one (token, topk-slot) at a time.
    for flat_off in range(global_warp_id, token_num * topk, total_warps):
        token_offset = flat_off // topk
        j = flat_off % topk
        expert_idx = ld(topk_indices_tensor + token_offset * topk + j)
        expert_rank = expert_idx // experts_per_rank
        expert_idx_intra_rank = expert_idx % experts_per_rank

        if expert_rank < world_size:
            if not WITH_SCATTER_INDICES:
                # Atomically reserve a destination slot in the target rank's recv
                # region. `recv_buf_offset_per_expert` is seeded with exclusive
                # prefix offsets on the host and bumped here per token.
                store_idx = atomic_add_per_warp(
                    recv_buf_offset_per_expert + expert_rank * experts_per_rank * world_size +
                    expert_idx_intra_rank * world_size + rank, 1, scope="agent", semantic="relaxed")
                if lane_idx == 0:
                    st(token_dst_scatter_idx + token_offset * topk + j, store_idx)
            else:
                store_idx = ld(token_dst_scatter_idx + token_offset * topk + j)

            src_ptr = send_buf + token_offset * hidden_size
            dst_ptr = output_buf + store_idx.to(tl.int64) * hidden_size
            libshmem_device.putmem_warp(dst_ptr, src_ptr, bytes_per_token, expert_rank)

            # When this rank has sent its last token for `expert_idx`, release the
            # destination's per-(local_expert, src_rank) barrier so the consuming
            # GEMM tile can start.
            if lane_idx == 0:
                tokens_this_expert = ld(local_splits_buf + expert_idx)
                sent = atomic_add(counter_ptr + expert_idx, 1, scope="agent", semantic="relaxed")
                if sent == tokens_this_expert - 1:
                    libshmem_device.fence()
                    libshmem_device.signal_op(
                        barriers_ptr + expert_idx_intra_rank * world_size + rank,
                        1,
                        libshmem_device.MORI_SIGNAL_SET,
                        expert_rank,
                    )

    # Pre-signal barriers for experts this rank sends zero tokens to, otherwise a
    # consumer expecting tokens from this (empty) source rank would spin forever.
    # NOTE: indexes by expert_local (not expert_rank) so producer/consumer barrier
    # addressing stays consistent.
    if pid == 0:
        for i in range(thread_idx, experts_per_rank * world_size, num_warps * WARP_SIZE):
            if ld(local_splits_buf + i) == 0:
                empty_expert_rank = i // experts_per_rank
                empty_expert_local = i % experts_per_rank
                libshmem_device.signal_op(
                    barriers_ptr + empty_expert_local * world_size + rank,
                    1,
                    libshmem_device.MORI_SIGNAL_SET,
                    empty_expert_rank,
                )


# ---------------------------------------------------------------------------
#  Grouped GEMM tile (consumer): one (pid_m, pid_n) tile of the per-expert GEMM
# ---------------------------------------------------------------------------
@triton_dist.jit(do_not_specialize=["pid", "num_pid", "M"])
def tile_kernel_moe_grouped_gemm_nk_const(
    pid,
    num_pid,
    barriers_ptr,  # symm uint64 [experts_per_rank * world_size]
    a_ptr,  # this rank's received tokens [M, K] (symm output_buf)
    b_ptr,  # expert weights [experts_per_rank, K, N]
    c_ptr,  # gemm output [M, N]
    expert_ids_ptr,  # int32 [num_tiles_m]: local expert id of each m-tile
    split_size_ptr,  # int32 [experts_per_rank]: tokens per local expert
    split_size_cum_ptr,  # int32 [num_tiles_m]: token row_begin of the tile's expert
    tile_num_ptr,  # int32 [num_tiles_m]: #m-tiles of the tile's expert
    tile_num_cum_ptr,  # int32 [num_tiles_m]: inclusive cumulative #m-tiles
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NEED_WAIT: tl.constexpr,
):
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_block_n
    pid_n = pid % num_block_n

    expert_id = tl.load(expert_ids_ptr + pid_m)
    split_size = tl.load(split_size_ptr + expert_id)
    split_size_cum = tl.load(split_size_cum_ptr + pid_m)
    row_begin = split_size_cum
    tile_num = tl.load(tile_num_ptr + pid_m)
    tile_num_cum = tl.load(tile_num_cum_ptr + pid_m)
    tile_begin = tile_num_cum - tile_num
    local_pid_m = pid_m - tile_begin

    world_size = dl.num_ranks()
    thread_idx = tid(0)

    local_pid_m, pid_n = tl.swizzle2d(local_pid_m, pid_n, tile_num, num_block_n, GROUP_SIZE_M)

    # Wait until every source rank has delivered this expert's tokens, then issue a
    # system-scope fence before reading the peer-written rows.
    #
    # IMPORTANT: mori's signal_wait_until is a *relaxed* system wait
    # (ShmemTypeWaitUntilEquals -> while(AtomicLoadRelaxedSystem(addr) != val){}); it
    # does NOT carry acquire semantics, so it alone does not make the peer-written
    # `output_buf` rows visible to the following tl.load. NVIDIA uses ld_acquire here;
    # the mori port dropped the acquire, so we restore consumer-side visibility with an
    # explicit system fence after the wait. __syncthreads() only orders within the block
    # and does not invalidate system-scope (peer-GPU) caches.
    #
    # We use libshmem_device.fence() (= __threadfence_system, a seq_cst system fence that
    # subsumes acquire) on purpose: the dedicated language_extra.fence(scope="system")
    # lowers to `fence syncscope("system") ...`, an AMDGPU-unsupported scope string that
    # fails to compile on gfx942 ("Unsupported atomic synchronization scope"). The HIP
    # __threadfence_system builtin lowers to the correct system-scope fence.
    if NEED_WAIT:
        if thread_idx < world_size:
            libshmem_device.signal_wait_until(
                barriers_ptr + expert_id * world_size + thread_idx,
                libshmem_device.MORI_CMP_EQ,
                1,
            )
        __syncthreads()
        libshmem_device.fence()

    row_remain = split_size - local_pid_m * BLOCK_SIZE_M

    offs_bn = (pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = (b_ptr + expert_id.to(tl.int64) * stride_be + offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    offs_token = row_begin.to(tl.int64) + local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    a_ptrs = (a_ptr + offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    if row_remain >= BLOCK_SIZE_M:
        dot_k_const(a_ptrs, b_ptrs, c_ptrs, row_remain, min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N), K, stride_ak,
                    stride_bk, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, False)
    elif row_remain > 0:
        dot_k_const(a_ptrs, b_ptrs, c_ptrs, row_remain, min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N), K, stride_ak,
                    stride_bk, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, True)


# ---------------------------------------------------------------------------
#  Mega kernel: persistent scheduler over {dispatch tasks} U {gemm tiles}
# ---------------------------------------------------------------------------
@triton_dist.jit(do_not_specialize=["M"])
def mega_kernel_dispatch_token_moe_grouped_gemm(
    task_counter_ptr,  # global int32 [1], zero init

    # dispatch params
    recv_buf_offset_per_expert,
    local_splits_buf,
    send_buf,
    output_buf,
    topk_indices_tensor,
    token_dst_scatter_idx,
    num_input_tokens_per_rank,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    experts_per_rank: tl.constexpr,
    WITH_SCATTER_INDICES: tl.constexpr,
    num_dispatch_tasks: tl.constexpr,

    # grouped gemm params
    a_ptr,
    b_ptr,
    c_ptr,
    expert_ids_ptr,
    split_size_ptr,
    split_size_cum_ptr,
    tile_num_ptr,
    tile_num_cum_ptr,
    num_total_tiles_ptr,  # int32 [1]: total #m-tiles across experts
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,

    # synchronization
    counter_ptr,  # local int32 [num_experts]
    barriers_ptr,  # symm uint64 [experts_per_rank * world_size]
    NUM_WARPS: tl.constexpr,
):
    task_id = tl.atomic_add(task_counter_ptr, 1)
    group_gemm_total_tiles_m = tl.load(num_total_tiles_ptr)
    group_gemm_total_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    group_gemm_tasks = group_gemm_total_tiles_m * group_gemm_total_tiles_n
    total_tasks = num_dispatch_tasks + group_gemm_tasks

    while task_id < total_tasks:
        if task_id < num_dispatch_tasks:
            tile_kernel_dispatch_token_intra_node(
                task_id,
                num_dispatch_tasks,
                counter_ptr,
                barriers_ptr,
                recv_buf_offset_per_expert,
                local_splits_buf,
                send_buf,
                output_buf,
                topk_indices_tensor,
                token_dst_scatter_idx,
                num_input_tokens_per_rank,
                topk,
                hidden_size,
                experts_per_rank,
                WITH_SCATTER_INDICES,
                NUM_WARPS,
            )
        else:
            tile_kernel_moe_grouped_gemm_nk_const(
                task_id - num_dispatch_tasks,
                group_gemm_tasks,
                barriers_ptr,
                a_ptr,
                b_ptr,
                c_ptr,
                expert_ids_ptr,
                split_size_ptr,
                split_size_cum_ptr,
                tile_num_ptr,
                tile_num_cum_ptr,
                M,
                N,
                K,
                stride_am,
                stride_ak,
                stride_be,
                stride_bn,
                stride_bk,
                stride_cm,
                stride_cn,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                NEED_WAIT=True,
            )
        task_id = tl.atomic_add(task_counter_ptr, 1)


# ---------------------------------------------------------------------------
#  Host orchestration
# ---------------------------------------------------------------------------
@dataclass
class EpA2AFusedContext:
    group: torch.distributed.ProcessGroup
    rank: int
    world_size: int
    max_tokens: int
    hidden: int
    topk: int
    num_tot_experts: int
    experts_per_rank: int
    dtype: torch.dtype
    weight_dtype: torch.dtype  # Not in use; reserved for future path
    capacity: float
    cap_tokens: int

    # symmetric (mori) buffers
    send_buf: torch.Tensor
    output_buf: torch.Tensor
    barriers_buf: torch.Tensor
    combine_in_buf: torch.Tensor
    # symmetric buffers for the device-side metadata all-gather
    local_splits_buf: torch.Tensor
    full_splits_buf: torch.Tensor
    splits_signal_buf: torch.Tensor

    # scratch (local)
    task_counter: torch.Tensor
    expert_counter: torch.Tensor
    combine_grid_sync: torch.Tensor
    meta_grid_sync: torch.Tensor

    def ep_barrier(self):
        mori_shmem_barrier_all_on_stream(torch.cuda.current_stream())

    def finalize(self):
        mori_shmem_free_tensor_sync(self.send_buf)
        mori_shmem_free_tensor_sync(self.output_buf)
        mori_shmem_free_tensor_sync(self.barriers_buf)
        mori_shmem_free_tensor_sync(self.combine_in_buf)
        mori_shmem_free_tensor_sync(self.local_splits_buf)
        mori_shmem_free_tensor_sync(self.full_splits_buf)
        mori_shmem_free_tensor_sync(self.splits_signal_buf)


def create_ep_a2a_fused_context(
    group: torch.distributed.ProcessGroup,
    max_tokens: int,
    hidden: int,
    topk: int,
    num_tot_experts: int,
    rank: int,
    world_size: int,
    dtype: torch.dtype = torch.bfloat16,
    weight_dtype: torch.dtype = torch.float32,  # Not in use; reserved for future path
    capacity: float = 4.0,
) -> EpA2AFusedContext:
    assert num_tot_experts % world_size == 0, "num_tot_experts must be divisible by world_size"
    experts_per_rank = num_tot_experts // world_size
    # Upper bound on tokens a single rank can receive (all replicas may land here).
    cap_tokens = max(int(math.ceil(max_tokens * topk * capacity)), max_tokens)

    send_buf = mori_shmem_create_tensor([max_tokens, hidden], dtype)
    output_buf = mori_shmem_create_tensor([cap_tokens, hidden], dtype)
    barriers_buf = mori_shmem_create_tensor([experts_per_rank * world_size], MORI_SHMEM_SIGNAL_DTYPE)
    barriers_buf.zero_()
    # gemm2 output / combine source: symmetric so origin ranks can pull rows back.
    combine_in_buf = mori_shmem_create_tensor([cap_tokens, hidden], dtype)

    # device-side metadata all-gather buffers (symmetric)
    local_splits_buf = mori_shmem_create_tensor([num_tot_experts], torch.int32)
    local_splits_buf.zero_()
    full_splits_buf = mori_shmem_create_tensor([world_size, num_tot_experts], torch.int32)
    full_splits_buf.zero_()
    splits_signal_buf = mori_shmem_create_tensor([world_size], MORI_SHMEM_SIGNAL_DTYPE)
    splits_signal_buf.zero_()

    device = torch.cuda.current_device()
    task_counter = torch.zeros([1], dtype=torch.int32, device=device)
    expert_counter = torch.zeros([num_tot_experts], dtype=torch.int32, device=device)
    combine_grid_sync = torch.zeros([1], dtype=torch.int32, device=device)
    meta_grid_sync = torch.zeros([1], dtype=torch.int32, device=device)

    mori_shmem_barrier_all_on_stream(torch.cuda.current_stream())
    return EpA2AFusedContext(
        group=group, rank=rank, world_size=world_size, max_tokens=max_tokens, hidden=hidden, topk=topk,
        num_tot_experts=num_tot_experts, experts_per_rank=experts_per_rank, dtype=dtype, weight_dtype=weight_dtype,
        capacity=capacity, cap_tokens=cap_tokens, send_buf=send_buf, output_buf=output_buf,
        barriers_buf=barriers_buf, combine_in_buf=combine_in_buf, local_splits_buf=local_splits_buf,
        full_splits_buf=full_splits_buf, splits_signal_buf=splits_signal_buf, task_counter=task_counter,
        expert_counter=expert_counter, combine_grid_sync=combine_grid_sync, meta_grid_sync=meta_grid_sync)


def _build_gemm_tiling(split_size: torch.Tensor, block_m: int, device):
    """Build sorted per-m-tile grouped-GEMM metadata from per-expert token counts.

    Returns expert_ids / split_size_cum / tile_num / tile_num_cum (each indexed by
    global m-tile) plus the total tile count -- the layout the grouped-GEMM tile
    kernel consumes. Depends on ``block_m``; reused by both dispatch (gemm1) and
    combine (gemm2) so each can tile with its own BLOCK_M.
    """
    split_cpu = split_size.tolist()
    epr = len(split_cpu)
    tiles_per_expert = [(s + block_m - 1) // block_m for s in split_cpu]
    row_begin = [0] * epr
    acc = 0
    for g in range(epr):
        row_begin[g] = acc
        acc += split_cpu[g]
    expert_ids, split_size_cum, tile_num, tile_num_cum = [], [], [], []
    cum_tiles = 0
    for g in range(epr):
        cum_tiles += tiles_per_expert[g]
        for _ in range(tiles_per_expert[g]):
            expert_ids.append(g)
            split_size_cum.append(row_begin[g])
            tile_num.append(tiles_per_expert[g])
            tile_num_cum.append(cum_tiles)
    total_tiles = int(sum(tiles_per_expert))

    def _i32(lst):
        if len(lst) == 0:
            return torch.zeros([1], dtype=torch.int32, device=device)
        return torch.tensor(lst, dtype=torch.int32, device=device)

    return dict(
        block_m=block_m,
        expert_ids=_i32(expert_ids),
        split_size_cum=_i32(split_size_cum),
        tile_num=_i32(tile_num),
        tile_num_cum=_i32(tile_num_cum),
        total_tiles=torch.tensor([total_tiles], dtype=torch.int32, device=device),
    )


def _build_dispatch_metadata(ctx: EpA2AFusedContext, topk_indices: torch.Tensor, block_m: int):
    """Compute, in torch, everything the fused kernel needs that is cheaper /
    safer to derive on the host: the cross-rank split exchange, the destination
    recv offsets, and the grouped-GEMM tiling metadata for this rank's experts.
    """
    device = topk_indices.device
    ws = ctx.world_size
    epr = ctx.experts_per_rank
    num_experts = ctx.num_tot_experts

    # Validate at this public-boundary entry, consistently with the device path (torch.bincount
    # would raise on negatives but silently drop over-range ids via the [:num_experts] slice).
    assert bool(((topk_indices >= 0) & (topk_indices < num_experts)).all()), \
        "topk_indices out of range [0, num_tot_experts); drop-token / invalid ids are not supported"

    # local_splits[e] = #tokens this rank routes to global expert e
    local_splits = torch.bincount(topk_indices.reshape(-1), minlength=num_experts).to(torch.int32)[:num_experts]

    # all_gather -> full_splits[s, e]
    full_splits = torch.empty((ws, num_experts), dtype=torch.int32, device=device)
    torch.distributed.all_gather_into_tensor(full_splits, local_splits, group=ctx.group)

    # counts[dst_rank, expert_local, src_rank]; recall e = dst_rank * epr + expert_local
    counts = full_splits.view(ws, ws, epr).permute(1, 2, 0).contiguous()  # [dst, g, src]
    flat = counts.view(ws, epr * ws)
    recv_buf_offset = (flat.cumsum(dim=1) - flat).view(ws, epr, ws).to(torch.int32).contiguous()
    num_recv_tokens_per_rank = flat.sum(dim=1).to(torch.int32)  # [ws]
    num_input_tokens_per_rank = (full_splits.sum(dim=1) // ctx.topk).to(torch.int32)  # [ws]

    M_local = int(num_recv_tokens_per_rank[ctx.rank].item())
    split_size = counts[ctx.rank].sum(dim=1).to(torch.int32)  # [epr]

    meta = dict(
        local_splits=local_splits,
        recv_buf_offset=recv_buf_offset,
        num_input_tokens_per_rank=num_input_tokens_per_rank,
        num_recv_tokens_per_rank=num_recv_tokens_per_rank,
        M_local=M_local,
        split_size=split_size,
        metadata_backend="host",
    )
    meta.update(_build_gemm_tiling(split_size, block_m, device))
    return meta


# ---------------------------------------------------------------------------
#  Device-side metadata kernels (replace the host torch path above)
# ---------------------------------------------------------------------------
@triton_dist.jit
def kernel_build_fused_dispatch_metadata(
    local_splits_buf,  # symm int32 [num_experts]: this rank's per-global-expert token count
    full_splits_buf,  # symm int32 [world_size, num_experts]: all-gather destination
    splits_signal_buf,  # symm signal [world_size]: per-source all-gather done flag
    recv_buf_offset_per_expert,  # int32 [world_size, experts_per_rank, world_size] (out)
    num_input_tokens_per_rank,  # int32 [world_size] (out)
    num_recv_tokens_per_rank,  # int32 [world_size] (out)
    split_size,  # int32 [experts_per_rank] (out, MUST be zeroed on host: accumulated via atomic_add)
    grid_sync_counter,  # int32 [1]: grid barrier workspace (single slot, reused)
    experts_per_rank: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # power-of-two >= num_experts
    num_warps: tl.constexpr,
):
    """Device equivalent of ``_build_dispatch_metadata`` (intra-node).

    Mirrors the validated AMD pattern in ``kernel_get_ag_splits_and_recv_offset_intra_node``
    (``ep_a2a_intra_node.py``) -- the AMD/mori port of NVIDIA's
    ``kernel_get_ag_splits_and_recv_offset`` -- and additionally accumulates this rank's
    per-local-expert ``split_size``. Produces a byte-identical layout to the host path
    (``recv_buf_offset`` ordered g-major / src-minor; ``num_input = sum // topk``).
    """
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_experts = experts_per_rank * world_size
    elem_size = tl.constexpr(local_splits_buf.dtype.element_ty.primitive_bitwidth) // 8
    nbytes = num_experts * elem_size
    threads_per_block = num_warps * threads_per_warp()
    thread_idx = tid(0)

    # phase 1: all-gather local_splits -> full_splits_buf[rank] on every peer (incl. self)
    for remote_rank in range(pid, world_size, num_pid):
        libshmem_device.putmem_signal_nbi_block(
            full_splits_buf + rank * num_experts,
            local_splits_buf,
            nbytes,
            splits_signal_buf + rank,
            1,
            libshmem_device.MORI_SIGNAL_SET,
            remote_rank,
        )

    # ensure all all-gather traffic is globally visible before any consumer reads
    barrier_on_this_grid(grid_sync_counter, False)
    if pid == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_sync_counter, False)

    offs = tl.arange(0, BLOCK_SIZE)
    expert_mask = offs < num_experts

    # phase 2: scatter full_splits[target, e] -> recv_buf_offset[e//epr, e%epr, target];
    #          compute num_input_tokens[target]; accumulate this rank's split_size.
    for target_rank in range(pid, world_size, num_pid):
        token = dl.wait(splits_signal_buf + target_rank, 1, "sys", "acquire")
        full_splits_buf = dl.consume_token(full_splits_buf, token)
        __syncthreads()
        for expert_idx in range(thread_idx, num_experts, threads_per_block):
            val = ld(full_splits_buf + target_rank * num_experts + expert_idx, semantic="acquire")
            ep_rank = expert_idx // experts_per_rank
            expert_idx_intra_rank = expert_idx % experts_per_rank
            st(
                recv_buf_offset_per_expert + ep_rank * experts_per_rank * world_size +
                expert_idx_intra_rank * world_size + target_rank, val, semantic="release")
            if ep_rank == rank:
                atomic_add(split_size + expert_idx_intra_rank, val, scope="agent", semantic="relaxed")
        __syncthreads()
        splits_cur_rank = tl.load(full_splits_buf + target_rank * num_experts + offs, mask=expert_mask, other=0,
                                  volatile=True)
        total_topk_token_cur_rank = tl.sum(splits_cur_rank)
        tl.store(num_input_tokens_per_rank + target_rank, total_topk_token_cur_rank // topk)
        __syncthreads()

    # split_size (atomic) and recv_buf_offset (raw counts) are fully written across all pids
    barrier_on_this_grid(grid_sync_counter, False)

    # phase 3: per ep_rank, exclusive prefix sum over [experts_per_rank * world_size];
    #          num_recv_tokens_per_rank[ep_rank] = total.
    for ep_rank in range(pid, world_size, num_pid):
        splits_cur = tl.load(recv_buf_offset_per_expert + ep_rank * num_experts + offs, mask=expert_mask, other=0,
                             volatile=True)
        recv_tokens = tl.sum(splits_cur)
        cumsum_exclude = tl.cumsum(splits_cur) - splits_cur
        tl.store(recv_buf_offset_per_expert + ep_rank * num_experts + offs, cumsum_exclude, mask=expert_mask)
        tl.store(num_recv_tokens_per_rank + ep_rank, recv_tokens)
    __syncthreads()


@triton.jit
def _element_at(x: tl.tensor, idx) -> tl.tensor:
    return tl.sum(tl.where(tl.arange(0, x.numel) == idx, x, 0))


@triton.jit
def kernel_build_gemm_tiling(
    split_size_ptr,  # int32 [E]: per-local-expert token counts
    split_size_cum_ptr,  # int32 [num_tiles] (out): row_begin of each m-tile's expert
    expert_ids_ptr,  # int32 [num_tiles] (out): local expert id of each m-tile
    tile_num_ptr,  # int32 [num_tiles] (out): #m-tiles of each m-tile's expert
    tile_num_cum_ptr,  # int32 [num_tiles] (out): inclusive cumulative #m-tiles
    num_tiles_total_ptr,  # int32 [1] (out)
    E: tl.constexpr,
    E_PAD: tl.constexpr,  # power-of-two >= E
    BLOCK_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Device port of NVIDIA ``build_block_row_idx_info_kernel`` (``nvidia/group_gemm.py``).

    Expands per-expert token counts into the per-m-tile grouped-GEMM metadata consumed by
    ``tile_kernel_moe_grouped_gemm_nk_const`` -- the device equivalent of ``_build_gemm_tiling``.
    """
    sm_id = tl.program_id(0)
    idx = tl.arange(0, E_PAD)
    mask = idx < E
    row_splits = tl.load(split_size_ptr + idx, mask=mask, other=0)
    row_cumsums = tl.cumsum(row_splits, axis=0)
    row_offs = row_cumsums - row_splits
    tiles_splits = tl.cdiv(row_splits, BLOCK_SIZE_M)
    tiles_cumsum = tl.cumsum(tiles_splits, axis=0)
    num_tiles_total = tl.sum(tiles_splits, axis=0)

    if sm_id == 0:
        tl.store(num_tiles_total_ptr, num_tiles_total)

    for pid in range(sm_id, num_tiles_total, NUM_SMS):
        if pid < num_tiles_total:
            expert_idx = tl.argmax((pid < tiles_cumsum).to(tl.int1), axis=0, tie_break_left=True)
            if expert_idx == 0:
                row_offset = 0
                tile_split = _element_at(tiles_splits, 0)
                tile_cumsum = tile_split
            else:
                row_offset = _element_at(row_offs, expert_idx)
                tile_split = _element_at(tiles_splits, expert_idx)
                tile_cumsum = _element_at(tiles_cumsum, expert_idx)
            tl.store(expert_ids_ptr + pid, expert_idx)
            tl.store(split_size_cum_ptr + pid, row_offset)
            tl.store(tile_num_ptr + pid, tile_split)
            tl.store(tile_num_cum_ptr + pid, tile_cumsum)


def _build_gemm_tiling_device(split_size: torch.Tensor, block_m: int, epr: int, device, M_local: int, num_sms: int = 64):
    """Device-kernel version of ``_build_gemm_tiling`` (same dict layout)."""
    # An expert contributes at most cdiv(tokens, block_m) tiles; the +epr bound covers
    # the partial tile of every expert, so M_grid is a safe upper bound on total tiles.
    M_grid = max((M_local + block_m - 1) // block_m + epr, 1)
    expert_ids = torch.zeros([M_grid], dtype=torch.int32, device=device)
    split_size_cum = torch.zeros([M_grid], dtype=torch.int32, device=device)
    tile_num = torch.zeros([M_grid], dtype=torch.int32, device=device)
    tile_num_cum = torch.zeros([M_grid], dtype=torch.int32, device=device)
    total_tiles = torch.zeros([1], dtype=torch.int32, device=device)
    E_PAD = max(1 << (epr - 1).bit_length(), 1)  # power-of-two >= epr
    num_sms = min(num_sms, torch.cuda.get_device_properties(device).multi_processor_count)
    kernel_build_gemm_tiling[(num_sms, )](
        split_size,
        split_size_cum,
        expert_ids,
        tile_num,
        tile_num_cum,
        total_tiles,
        epr,
        E_PAD,
        block_m,
        num_sms,
    )
    return dict(
        block_m=block_m,
        expert_ids=expert_ids,
        split_size_cum=split_size_cum,
        tile_num=tile_num,
        tile_num_cum=tile_num_cum,
        total_tiles=total_tiles,
    )


def _build_dispatch_metadata_device(ctx: "EpA2AFusedContext", topk_indices: torch.Tensor, block_m: int,
                                    num_sms: int = 64):
    """Device-kernel version of ``_build_dispatch_metadata`` (same meta keys).

    Replaces the host ``bincount`` + ``all_gather_into_tensor`` + recv-offset / tiling torch
    ops with on-device kernels: only one device->host sync remains (reading ``M_local``).
    """
    device = topk_indices.device
    ws = ctx.world_size
    epr = ctx.experts_per_rank
    num_experts = ctx.num_tot_experts

    # Validate at this public-boundary entry (preprocess feeds user routing here): fail early and
    # clearly, consistently with the host path. Without it the device bincount would silently *skip*
    # out-of-range ids (the lower-bound guard), yielding a descriptor based on filtered routing.
    assert bool(((topk_indices >= 0) & (topk_indices < num_experts)).all()), \
        "topk_indices out of range [0, num_tot_experts); drop-token / invalid ids are not supported"

    # local_splits[e] via device bincount into the symmetric all-gather source buffer
    ctx.local_splits_buf.zero_()
    bincount(topk_indices.reshape(-1).contiguous(), num_experts, output=ctx.local_splits_buf)

    recv_buf_offset = torch.zeros((ws, epr, ws), dtype=torch.int32, device=device)
    num_input_tokens_per_rank = torch.zeros((ws, ), dtype=torch.int32, device=device)
    num_recv_tokens_per_rank = torch.zeros((ws, ), dtype=torch.int32, device=device)
    split_size = torch.zeros((epr, ), dtype=torch.int32, device=device)  # zeroed: atomic_add target

    # Stays on-stream: cross-rank sync is done entirely inside the kernel (grid barrier ->
    # barrier_all_block -> grid barrier), so NO host ep_barrier is needed here -- matching the
    # validated `get_ag_splits_and_recv_offset_for_dispatch_intra_node` (ep_a2a_intra_node.py),
    # which uses zero host barriers. ``barrier_all_block`` provides remote-op quiescence, so a
    # stale ``splits_signal_buf`` (SET=1 from a prior call) is harmless and needs no reset; it is
    # zeroed once at allocation. Only the grid-barrier counter must be reset per launch.
    ctx.meta_grid_sync.zero_()

    BLOCK_SIZE = max(1 << (num_experts - 1).bit_length(), 1)  # power-of-two >= num_experts
    kernel_build_fused_dispatch_metadata[(ws, )](
        ctx.local_splits_buf,
        ctx.full_splits_buf,
        ctx.splits_signal_buf,
        recv_buf_offset,
        num_input_tokens_per_rank,
        num_recv_tokens_per_rank,
        split_size,
        ctx.meta_grid_sync,
        epr,
        ctx.topk,
        BLOCK_SIZE,
        num_warps=8,
    )

    M_local = int(num_recv_tokens_per_rank[ctx.rank].item())
    meta = dict(
        # clone: ctx.local_splits_buf is shared/overwritten by the next preprocess, so a
        # descriptor must own a stable copy (dispatch reads local_splits for signal decisions).
        local_splits=ctx.local_splits_buf.clone(),
        recv_buf_offset=recv_buf_offset,
        num_input_tokens_per_rank=num_input_tokens_per_rank,
        num_recv_tokens_per_rank=num_recv_tokens_per_rank,
        M_local=M_local,
        split_size=split_size,
        metadata_backend="device",
    )
    meta.update(_build_gemm_tiling_device(split_size, block_m, epr, device, M_local, num_sms=num_sms))
    return meta


def fused_dispatch_token_moe_grouped_gemm(
    ctx: EpA2AFusedContext,
    input: torch.Tensor,  # [num_tokens, hidden] this rank's tokens
    topk_indices: torch.Tensor,  # [num_tokens, topk] int32 (global expert ids)
    gemm_weight: torch.Tensor,  # [experts_per_rank, hidden, N]
    *,  # keyword-only below: avoids any silent positional misbind of the new `meta` arg
    meta: Optional[dict] = None,  # precomputed layout metadata (e.g. from a layer.preprocess); built if None
    use_device_metadata: bool = True,  # build meta with device kernels (host torch fallback if False)
    num_sms: int = 64,
    num_dispatch_tasks: int = 20,
    BLOCK_SIZE_M: int = 128,
    BLOCK_SIZE_N: int = 128,
    BLOCK_SIZE_K: int = 64,
    GROUP_SIZE_M: int = 4,
    num_warps: int = 8,
    num_stages: int = 2,
    gemm_output: Optional[torch.Tensor] = None,
):
    """Run fused dispatch + grouped GEMM. Returns ``(gemm_output[M, N], meta)``.

    ``meta`` carries the layout (recv offsets, scatter idx, per-rank token
    counts) needed by a subsequent combine.

    Contract: ``topk_indices`` must be in ``[0, num_tot_experts)`` on **every** rank
    (drop-token / negative sentinels are unsupported). The range check is per-rank
    local, so passing invalid routing on only *some* ranks is undefined behavior and
    may hang at the next cross-rank step rather than failing cleanly — callers must
    guarantee valid routing on all ranks.
    """
    assert input.is_contiguous() and input.dtype == ctx.dtype
    assert topk_indices.dtype == torch.int32 and topk_indices.shape[1] == ctx.topk and topk_indices.is_contiguous()
    assert gemm_weight.dtype == ctx.dtype, "expert weights must match the activation dtype (ctx.dtype)"
    num_tokens, hidden = input.shape
    assert hidden == ctx.hidden and num_tokens <= ctx.max_tokens
    G, K, N = gemm_weight.shape
    assert G == ctx.experts_per_rank and K == ctx.hidden
    # Validate the routing the dispatch kernel will actually scatter on. This is the safety boundary
    # for the scatter pointer arithmetic, so it always runs on the *current* topk_indices: a precomputed
    # meta only proves some indices were validated at build time, not that these are them / unmutated.
    # Per-rank local: callers must pass in-range routing on every rank (see docstring).
    assert bool(((topk_indices >= 0) & (topk_indices < ctx.num_tot_experts)).all()), \
        "topk_indices out of range [0, num_tot_experts)"

    if meta is None:
        if use_device_metadata:
            meta = _build_dispatch_metadata_device(ctx, topk_indices, BLOCK_SIZE_M, num_sms=num_sms)
        else:
            meta = _build_dispatch_metadata(ctx, topk_indices, BLOCK_SIZE_M)
    else:
        assert meta.get("block_m") == BLOCK_SIZE_M, \
            "precomputed meta was built with a different BLOCK_SIZE_M than this launch"
    M = meta["M_local"]
    device = input.device
    # Global capacity check: every rank evaluates the same all-gathered per-rank recv counts,
    # so all ranks raise together. A per-rank ``M > cap_tokens`` check would let under-capacity
    # ranks fall through into the barrier/kernel while overflowing ranks bail out -> hang.
    max_recv = int(meta["num_recv_tokens_per_rank"].max().item())
    if max_recv > ctx.cap_tokens:
        raise RuntimeError(
            f"a rank would receive up to {max_recv} routed tokens > cap_tokens={ctx.cap_tokens}; "
            "increase `capacity` or `max_tokens`")
    num_sms = min(num_sms, torch.cuda.get_device_properties(device).multi_processor_count)

    # stage this rank's tokens into the symmetric send buffer (putmem source)
    ctx.send_buf[:num_tokens].copy_(input)

    token_dst_scatter_idx = torch.empty((num_tokens, ctx.topk), dtype=torch.int32, device=device)

    if gemm_output is None:
        gemm_output = torch.empty([max(M, 1), N], dtype=ctx.dtype, device=device)
    else:
        assert gemm_output.shape[0] >= M and gemm_output.shape[1] == N

    # reset per-call state
    ctx.task_counter.zero_()
    ctx.expert_counter.zero_()
    ctx.barriers_buf.zero_()
    recv_buf_offset = meta["recv_buf_offset"].clone()  # consumed (atomically bumped) by the kernel
    ctx.ep_barrier()

    # Do NOT early-return when M == 0: this rank may still own tokens to *send*
    # to other ranks (producer role) and must emit empty-expert pre-signals, so
    # skipping the launch would hang peers. M == 0 just yields 0 GEMM tiles.
    grid = (num_sms, )
    mega_kernel_dispatch_token_moe_grouped_gemm[grid](
        ctx.task_counter,
        recv_buf_offset,
        meta["local_splits"],
        ctx.send_buf,
        ctx.output_buf,
        topk_indices,
        token_dst_scatter_idx,
        meta["num_input_tokens_per_rank"],
        ctx.topk,
        ctx.hidden,
        ctx.experts_per_rank,
        False,  # WITH_SCATTER_INDICES (compute in-kernel)
        num_dispatch_tasks,
        ctx.output_buf,  # a_ptr: GEMM reads this rank's received tokens
        gemm_weight,
        gemm_output,
        meta["expert_ids"],
        meta["split_size"],
        meta["split_size_cum"],
        meta["tile_num"],
        meta["tile_num_cum"],
        meta["total_tiles"],
        M,
        N,
        K,
        ctx.output_buf.stride(0),
        ctx.output_buf.stride(1),
        gemm_weight.stride(0),
        gemm_weight.stride(2),
        gemm_weight.stride(1),
        gemm_output.stride(0),
        gemm_output.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        ctx.expert_counter,
        ctx.barriers_buf,
        num_warps,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    meta["token_dst_scatter_idx"] = token_dst_scatter_idx
    ctx.ep_barrier()
    return gemm_output[:M], meta


# ===========================================================================
#  Combine: grouped GEMM (gemm2 / down-proj) fused with gather + topk reduce
# ===========================================================================
@triton_dist.jit(do_not_specialize=["pid", "num_pid"])
def tile_kernel_gather_combine_token_intra_node(
    pid,
    num_pid,
    num_input_tokens_per_rank,  # int32 [world_size]
    input_buf,  # symm: per-rank gemm2 output rows [cap_tokens, hidden] (pull source)
    output_buf,  # local: combined output [num_tokens, hidden]
    topk_weights,  # local fp32 [num_tokens, topk] (routing weights) or dummy
    topk_indices_buf,  # int32 [num_tokens, topk]
    token_dst_scatter_idx,  # int32 [num_tokens, topk]
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    experts_per_rank: tl.constexpr,
    HAS_GATE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """Gather each origin token's topk expert outputs from the expert ranks (via
    ``symm_at`` peer pointers) and write the (weighted) sum. One warp per token,
    lanes stride the hidden dim. Relies on the caller having performed a cross-rank
    barrier so every ``input_buf`` row is already computed and globally visible.
    """
    WARP_SIZE: tl.constexpr = 64
    out_dtype = output_buf.dtype.element_ty
    rank = dl.rank()
    world_size = dl.num_ranks()
    thread_idx = tid(0)
    lane_id = thread_idx % WARP_SIZE
    warp_id = thread_idx // WARP_SIZE
    total_warps = num_warps * num_pid
    global_warp_id = pid * num_warps + warp_id

    num_tokens = tl.load(num_input_tokens_per_rank + rank)

    for token_idx in range(global_warp_id, num_tokens, total_warps):
        for i in range(lane_id, hidden_size, WARP_SIZE):
            acc = tl.zeros((), dtype=tl.float32)
            for j in range(topk):
                expert_idx = ld(topk_indices_buf + token_idx * topk + j)
                expert_rank = expert_idx // experts_per_rank
                if expert_rank < world_size:
                    scatter_idx = ld(token_dst_scatter_idx + token_idx * topk + j)
                    remote_ptr = dl.symm_at(input_buf, expert_rank)
                    # cross-device read of a peer's HBM: use system-scope acquire so the
                    # value written by the peer's gemm2 (before barrier_all) is observed.
                    val = ld(remote_ptr + scatter_idx.to(tl.int64) * hidden_size + i, scope="system",
                             semantic="acquire").to(tl.float32)
                    if HAS_GATE:
                        w = tl.load(topk_weights + token_idx * topk + j)
                        acc += w * val
                    else:
                        acc += val
            st(output_buf + token_idx.to(tl.int64) * hidden_size + i, acc.to(out_dtype))


@triton_dist.jit(do_not_specialize=["M"])
def mega_kernel_moe_grouped_gemm_combine_token(
    # grouped gemm (gemm2 / down-proj) params
    a_ptr,  # local: gemm2 input [M, K] (dispatched-layout activation)
    b_ptr,  # expert weights [experts_per_rank, K, N]
    c_ptr,  # symm: gemm2 output / combine source [cap_tokens, N]
    expert_ids_ptr,
    split_size_ptr,
    split_size_cum_ptr,
    tile_num_ptr,
    tile_num_cum_ptr,
    num_total_tiles_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,

    # combine params
    num_input_tokens_per_rank,  # int32 [world_size]
    output_buf,  # local: combined output [num_tokens, N]
    topk_weights,  # local fp32 [num_tokens, topk] or dummy
    topk_indices_buf,  # int32 [num_tokens, topk]
    token_dst_scatter_idx,  # int32 [num_tokens, topk]
    topk: tl.constexpr,
    experts_per_rank: tl.constexpr,
    HAS_GATE: tl.constexpr,

    # synchronization
    barriers_ptr,  # symm uint64 (unused here; NEED_WAIT=False) -- valid placeholder
    grid_sync_counter,  # local int32 [1]
    NUM_WARPS: tl.constexpr,
):
    """Serial-mode combine mega kernel:

      phase 1: every CTA strides over the grouped-GEMM(2) tiles, writing the
               down-proj output into the symmetric ``c_ptr``.
      barrier: grid barrier + cross-rank ``barrier_all`` so all ranks' rows are
               written and visible.
      phase 2: gather + topk-weighted reduce into ``output_buf``.
    """
    sm_id = tl.program_id(0)
    num_sms = tl.num_programs(0)
    group_gemm_total_tiles_m = tl.load(num_total_tiles_ptr)
    group_gemm_total_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    group_gemm_tasks = group_gemm_total_tiles_m * group_gemm_total_tiles_n

    for task_id in range(sm_id, group_gemm_tasks, num_sms):
        tile_kernel_moe_grouped_gemm_nk_const(
            task_id,
            group_gemm_tasks,
            barriers_ptr,
            a_ptr,
            b_ptr,
            c_ptr,
            expert_ids_ptr,
            split_size_ptr,
            split_size_cum_ptr,
            tile_num_ptr,
            tile_num_cum_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_be,
            stride_bn,
            stride_bk,
            stride_cm,
            stride_cn,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            NEED_WAIT=False,
        )

    # all local tiles done -> sync the grid, then sync all ranks (so every
    # rank's gemm2 output is complete and globally visible), then sync the grid.
    barrier_on_this_grid(grid_sync_counter, False)
    if sm_id == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_sync_counter, False)

    tile_kernel_gather_combine_token_intra_node(
        sm_id,
        num_sms,
        num_input_tokens_per_rank,
        c_ptr,
        output_buf,
        topk_weights,
        topk_indices_buf,
        token_dst_scatter_idx,
        topk,
        N,
        experts_per_rank,
        HAS_GATE,
        NUM_WARPS,
    )


def fused_group_gemm_combine_token(
    ctx: EpA2AFusedContext,
    gemm_input: torch.Tensor,  # [M, K2] dispatched-layout activation (e.g. act(gemm1 out))
    gemm_weight: torch.Tensor,  # [experts_per_rank, K2, N2] (N2 must == ctx.hidden)
    meta: dict,  # layout descriptor returned by fused_dispatch_token_moe_grouped_gemm
    topk_indices: torch.Tensor,  # [num_tokens, topk] int32 (same as dispatch)
    topk_weights: Optional[torch.Tensor] = None,  # [num_tokens, topk] fp32 routing weights
    use_device_metadata: Optional[bool] = None,  # None -> follow meta["metadata_backend"]; else override
    num_sms: int = 64,
    BLOCK_SIZE_M: int = 128,
    BLOCK_SIZE_N: int = 128,
    BLOCK_SIZE_K: int = 64,
    GROUP_SIZE_M: int = 4,
    num_warps: int = 8,
    num_stages: int = 2,
    combine_output: Optional[torch.Tensor] = None,
):
    """Fused gemm2 + combine. Returns ``combined_out[num_tokens, N2]``."""
    M = meta["M_local"]
    num_tokens = topk_indices.shape[0]
    G, K2, N2 = gemm_weight.shape
    assert G == ctx.experts_per_rank
    assert N2 == ctx.hidden, "combine maps expert outputs back to `hidden`; expected N2 == ctx.hidden"
    assert gemm_weight.dtype == ctx.dtype, "expert weights must match the activation dtype (ctx.dtype)"
    assert topk_indices.dtype == torch.int32 and topk_indices.shape[1] == ctx.topk and topk_indices.is_contiguous()
    # Validate the routing combine will gather on: the kernel computes expert_rank = expert_idx // epr
    # and reads peer buffers via symm_at(expert_rank), so an out-of-range / negative id would index an
    # invalid peer rank or address. Mirror the dispatch boundary check; per-rank local (see dispatch docstring).
    assert bool(((topk_indices >= 0) & (topk_indices < ctx.num_tot_experts)).all()), \
        "topk_indices out of range [0, num_tot_experts)"
    device = gemm_input.device
    # Global capacity check (see fused_dispatch_token_moe_grouped_gemm): raise on every rank together.
    max_recv = int(meta["num_recv_tokens_per_rank"].max().item())
    if max_recv > ctx.cap_tokens:
        raise RuntimeError(
            f"a rank would receive up to {max_recv} routed tokens > cap_tokens={ctx.cap_tokens}; "
            "increase `capacity` or `max_tokens`")
    num_sms = min(num_sms, torch.cuda.get_device_properties(device).multi_processor_count)

    has_gate = topk_weights is not None
    if has_gate:
        assert topk_weights.shape == (num_tokens, ctx.topk)
        topk_weights = topk_weights.to(torch.float32).contiguous()
    else:
        topk_weights = torch.empty((1, ), dtype=torch.float32, device=device)

    token_dst_scatter_idx = meta["token_dst_scatter_idx"]
    # When unset, follow how the dispatch metadata was built so combine stays consistent
    # with dispatch (avoids silently running device tiling after a host-metadata dispatch).
    if use_device_metadata is None:
        use_device_metadata = meta.get("metadata_backend", "device") == "device"
    if use_device_metadata:
        tiling = _build_gemm_tiling_device(meta["split_size"], BLOCK_SIZE_M, ctx.experts_per_rank, device,
                                           meta["M_local"], num_sms=num_sms)
    else:
        tiling = _build_gemm_tiling(meta["split_size"], BLOCK_SIZE_M, device)

    if combine_output is None:
        combine_output = torch.empty([num_tokens, N2], dtype=ctx.dtype, device=device)
    else:
        # The gather kernel writes rows linearly (st(output_buf + token_idx*hidden + i), no strides),
        # so a non-contiguous / wrong-dtype / wrong-device buffer would be written with the wrong layout.
        assert combine_output.shape == (num_tokens, N2)
        assert combine_output.is_contiguous(), "combine_output must be contiguous"
        assert combine_output.dtype == ctx.dtype, "combine_output dtype must match ctx.dtype"
        assert combine_output.device == device, "combine_output must be on the same device as gemm_input"

    ctx.combine_grid_sync.zero_()
    ctx.ep_barrier()

    # Do NOT early-return when M == 0: every rank must enter the kernel's
    # device-side barrier_all and still gather its own origin tokens from peers.
    assert gemm_input.shape == (M, K2) and gemm_input.is_contiguous()
    c_buf = ctx.combine_in_buf  # [cap_tokens, hidden]

    grid = (num_sms, )
    mega_kernel_moe_grouped_gemm_combine_token[grid](
        gemm_input,
        gemm_weight,
        c_buf,
        tiling["expert_ids"],
        meta["split_size"],
        tiling["split_size_cum"],
        tiling["tile_num"],
        tiling["tile_num_cum"],
        tiling["total_tiles"],
        M,
        N2,
        K2,
        gemm_input.stride(0),
        gemm_input.stride(1),
        gemm_weight.stride(0),
        gemm_weight.stride(2),
        gemm_weight.stride(1),
        c_buf.stride(0),
        c_buf.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        meta["num_input_tokens_per_rank"],
        combine_output,
        topk_weights,
        topk_indices,
        token_dst_scatter_idx,
        ctx.topk,
        ctx.experts_per_rank,
        has_gate,
        ctx.barriers_buf,
        ctx.combine_grid_sync,
        num_warps,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    ctx.ep_barrier()
    return combine_output
