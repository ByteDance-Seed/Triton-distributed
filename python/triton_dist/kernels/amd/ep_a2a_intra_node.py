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
AMD (HIP/ROCm) Expert Parallel All-to-All intra-node kernels.

This module uses mori_shmem for all SHMEM operations (no rocshmem).
NVIDIA uses NVSHMEM and CUDA-specific primitives; on AMD we use mori_shmem
device API (via libshmem_device when built for mori). Areas that need
mori-specific handling are marked with TODO(AMD).

Only intra-node path is implemented; inter-node is out of scope.
"""

import torch
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.hip.language_extra import (
    tid,
    __syncthreads,
    st,
    ld,
    atomic_add_per_warp,
)
from triton_dist.language.extra.language_extra import threads_per_warp
from triton_dist.kernels.amd.common_ops import barrier_on_this_grid
from triton_dist.utils import MORI_SHMEM_SIGNAL_DTYPE

########## Triton kernels (intra-node only) ##########


@triton_dist.jit(do_not_specialize=["dispatch_recv_token_num"])
def kernel_dispatch_token_intra_node(
    dispatch_recv_token_num,
    intra_node_dispatch_skipped_token_mapping_indices,
    intra_node_dispatch_skipped_token_topk_mapping_indices,
    recv_buf_offset_per_expert,
    input_buf,
    output_buf,
    weight_send_buf,
    weight_recv_buf,
    topk_indices_tensor,
    token_dst_scatter_idx,
    num_input_tokens_per_rank,
    topk: int,
    hidden_size: int,
    bytes_per_token: int,
    experts_per_rank: tl.constexpr,
    local_world_size: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WITH_SCATTER_INDICES: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Dispatch tokens to expert ranks (intra-node). Uses mori_shmem (symmetric
    memory) for put and signal. libshmem_device must be built for mori_shmem
    device API when using mori backend.
    """
    WARP_SIZE = threads_per_warp()
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    thread_idx = tid(0)
    warp_id = thread_idx // WARP_SIZE
    total_comm_warps = num_warps * num_pid
    global_warp_id = pid * num_warps + warp_id
    weight_elem_size = tl.constexpr(weight_send_buf.dtype.element_ty.primitive_bitwidth) // 8
    token_num = tl.load(num_input_tokens_per_rank + rank)

    for token_offset in range(global_warp_id, token_num, total_comm_warps):
        # Single fused topk loop: allocate scatter index + skip check + P2P transfer
        for j in range(topk):
            expert_idx = ld(topk_indices_tensor + token_offset * topk + j)
            expert_rank = expert_idx // experts_per_rank
            expert_idx_intra_rank = expert_idx % experts_per_rank

            if expert_rank < world_size:
                # Allocate or read scatter index
                if not WITH_SCATTER_INDICES:
                    store_idx = atomic_add_per_warp(
                        recv_buf_offset_per_expert + expert_rank * experts_per_rank * world_size +
                        expert_idx_intra_rank * world_size + rank,
                        1,
                        scope="agent",
                        semantic="monotonic",
                    )
                    st(
                        token_dst_scatter_idx + token_offset * topk + j,
                        store_idx,
                    )
                else:
                    store_idx = ld(token_dst_scatter_idx + token_offset * topk + j)

                # Cast to uint64 to match token_dst_scatter_idx element type
                store_idx = store_idx.to(tl.uint64)
                # Skip check: is this a duplicate rank within the same token?
                skip_this_token = False
                skipped_token_mapping_idx = store_idx
                for topk_idx in range(j):
                    if not skip_this_token and ld(topk_indices_tensor + token_offset * topk +
                                                  topk_idx) // experts_per_rank == expert_rank:
                        skip_this_token = True
                        skipped_token_mapping_idx = ld(token_dst_scatter_idx + token_offset * topk + topk_idx).to(
                            tl.uint64)

                src_ptr = input_buf + token_offset * hidden_size
                dst_ptr = output_buf + store_idx * hidden_size
                if not skip_this_token:
                    libshmem_device.putmem_signal_warp(
                        dst_ptr,
                        src_ptr,
                        bytes_per_token,
                        intra_node_dispatch_skipped_token_mapping_indices + store_idx,
                        skipped_token_mapping_idx,
                        libshmem_device.MORI_SIGNAL_SET,
                        expert_rank,
                    )
                else:
                    if thread_idx % WARP_SIZE == 0:
                        st(
                            dl.symm_at(
                                intra_node_dispatch_skipped_token_mapping_indices + store_idx,
                                expert_rank,
                            ),
                            skipped_token_mapping_idx,
                        )
                if thread_idx % WARP_SIZE == 0:
                    st(
                        dl.symm_at(
                            intra_node_dispatch_skipped_token_topk_mapping_indices + skipped_token_mapping_idx * topk +
                            j,
                            expert_rank,
                        ),
                        skipped_token_mapping_idx,
                    )
                if HAS_WEIGHT:
                    libshmem_device.putmem_warp(
                        weight_recv_buf + store_idx,
                        weight_send_buf + token_offset * topk + j,
                        weight_elem_size,
                        expert_rank,
                    )


@triton_dist.jit(do_not_specialize=["dispatch_recv_token_num"])
def kernel_skipped_token_local_dispatch_intra_node(
    dispatch_recv_token_num,
    intra_node_dispatch_skipped_token_mapping_indices,
    intra_node_dispatch_skipped_token_topk_mapping_indices,
    intra_node_dispatch_skipped_token_mapping_indices_copy,
    intra_node_dispatch_skipped_token_topk_mapping_indices_copy,
    dispatch_out_buf,
    hidden_size: tl.constexpr,
    bytes_per_token: int,
    topk: int,
    ENABLE_LOCAL_COMBINE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """Copy skipped token data from first-send slot to duplicate slots.

    SIMT warp-level: each warp handles one token, per-lane ld/st for
    hidden-dim data copy. Consistent with dispatch/combine kernels.
    """
    WARP_SIZE = threads_per_warp()
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    thread_idx = tid(0)
    warp_id = thread_idx // WARP_SIZE
    total_warps = num_warps * num_pid
    global_warp_id = pid * num_warps + warp_id
    lane_id = thread_idx % WARP_SIZE

    for recv_token_offset in range(global_warp_id, dispatch_recv_token_num, total_warps):
        skipped_token_mapping_idx = ld(intra_node_dispatch_skipped_token_mapping_indices + recv_token_offset)
        if skipped_token_mapping_idx != recv_token_offset:
            dst_ptr = dispatch_out_buf + recv_token_offset * hidden_size
            src_ptr = dispatch_out_buf + skipped_token_mapping_idx * hidden_size
            for i in range(lane_id, hidden_size, WARP_SIZE):
                st(dst_ptr + i, ld(src_ptr + i))
        if ENABLE_LOCAL_COMBINE:
            if lane_id == 0:
                st(
                    intra_node_dispatch_skipped_token_mapping_indices_copy + recv_token_offset,
                    skipped_token_mapping_idx,
                )
            for j in range(lane_id, topk, WARP_SIZE):
                st(
                    intra_node_dispatch_skipped_token_topk_mapping_indices_copy + recv_token_offset * topk + j,
                    ld(intra_node_dispatch_skipped_token_topk_mapping_indices + recv_token_offset * topk + j),
                )


@triton_dist.jit(do_not_specialize=["combine_token_num"])
def kernel_skipped_token_inplace_local_combine_intra_node(
    combine_token_num,
    intra_node_dispatch_skipped_token_mapping_indices,
    skipped_token_topk_mapping_indices,
    combine_input_buf,
    hidden_size: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Pre-combine local reduction for tokens with multiple same-rank experts.

    Block-level: pure local HBM operation (no P2P), so block-level
    tl.load/tl.store with tl.arange is optimal for access granularity.
    """
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    input_dtype = combine_input_buf.dtype.element_ty
    mapping_ptr = intra_node_dispatch_skipped_token_mapping_indices.to(tl.pointer_type(tl.int64))
    topk_mapping_ptr = skipped_token_topk_mapping_indices.to(tl.pointer_type(tl.int64))

    for combine_token_offset in range(pid, combine_token_num, num_pid):
        skipped_token_mapping_idx = tl.load(mapping_ptr + combine_token_offset)
        combine_offset_i64 = combine_token_offset.to(tl.int64)
        if skipped_token_mapping_idx == combine_offset_i64:
            local_combine_cnt = 0
            for j in range(topk):
                v = tl.load(topk_mapping_ptr + combine_token_offset * topk + j)
                if v != -1:
                    local_combine_cnt += 1
            if local_combine_cnt > 1:
                for block_start in range(0, hidden_size, BLOCK_H):
                    offs = tl.arange(0, BLOCK_H)
                    token_accum = tl.zeros([BLOCK_H], dtype=tl.float32)
                    for j in range(topk):
                        v = tl.load(topk_mapping_ptr + combine_token_offset * topk + j)
                        if v != -1:
                            val = tl.load(combine_input_buf + v * hidden_size + block_start + offs).to(tl.float32)
                            token_accum = token_accum + val
                    tl.store(
                        combine_input_buf + combine_token_offset * hidden_size + block_start + offs,
                        token_accum.to(input_dtype),
                    )


@triton_dist.jit(do_not_specialize=["num_dispatch_tokens"])
def kernel_combine_token_intra_node(
    num_dispatch_tokens,  # int — loop bound
    input_buf,  # symm buffer (dispatch_output_buf), bf16
    send_buf,  # [nnode, max_tokens, hidden] output (bf16)
    topk_indices_buf,  # [max_tokens, topk] int32
    token_dst_scatter_idx,  # [nnodes, max_tokens, topk] int32
    max_tokens: int,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    bytes_per_token: tl.constexpr,
    expert_per_rank: tl.constexpr,
    local_world_size: tl.constexpr,
    ENABLE_LOCAL_COMBINE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """Combine expert outputs within a single node (AMD).

    SIMT warp-level: each warp processes one token, each lane handles
    a slice of hidden dim. Topk loop is OUTER (index reads + symm_at
    computed once per topk entry), hidden loop is INNER (pure ld/st).
    """
    WARP_SIZE = threads_per_warp()
    rank = dl.rank()
    node_id = rank // local_world_size
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    output_dtype = send_buf.dtype.element_ty

    with dl.simt_exec_region() as (thread_id, block_size):
        warp_id = thread_id // WARP_SIZE
        lane_id = thread_id % WARP_SIZE
        warps_per_block = block_size // WARP_SIZE
        total_warps = warps_per_block * num_pid
        global_warp_id = pid * warps_per_block + warp_id

        for token_idx in range(global_warp_id, num_dispatch_tokens, total_warps):
            out_base = (node_id * max_tokens + token_idx) * hidden_size
            valid_j_count = 0

            if not ENABLE_LOCAL_COMBINE:
                for j in range(topk):
                    expert_idx = ld(topk_indices_buf + token_idx * topk + j)
                    expert_rank = expert_idx // expert_per_rank
                    expert_node_idx = expert_rank // local_world_size

                    if expert_node_idx == node_id:
                        token_scatter_idx = ld(token_dst_scatter_idx + token_idx * topk + j)
                        remote_ptr = dl.symm_at(input_buf, expert_rank)
                        src_base = remote_ptr + token_scatter_idx * hidden_size
                        # Pure ld/st hidden loop — no index reads inside
                        for i in range(lane_id, hidden_size, WARP_SIZE):
                            val = ld(src_base + i).to(tl.float32)
                            if valid_j_count == 0:
                                st(send_buf + out_base + i, val.to(output_dtype))
                            else:
                                prev = ld(send_buf + out_base + i).to(tl.float32)
                                st(send_buf + out_base + i, (prev + val).to(output_dtype))
                        valid_j_count += 1
            else:
                for j in range(topk):
                    expert_idx = ld(topk_indices_buf + token_idx * topk + j)
                    expert_rank = expert_idx // expert_per_rank
                    expert_node_idx = expert_rank // local_world_size

                    if expert_node_idx == node_id:
                        skip_this_token = False
                        for topk_idx in range(j):
                            prev_expert = ld(topk_indices_buf + token_idx * topk + topk_idx)
                            if not skip_this_token and prev_expert // expert_per_rank == expert_rank:
                                skip_this_token = True

                        if not skip_this_token:
                            token_scatter_idx = ld(token_dst_scatter_idx + token_idx * topk + j)
                            remote_ptr = dl.symm_at(input_buf, expert_rank)
                            src_base = remote_ptr + token_scatter_idx * hidden_size
                            for i in range(lane_id, hidden_size, WARP_SIZE):
                                val = ld(src_base + i).to(tl.float32)
                                if valid_j_count == 0:
                                    st(send_buf + out_base + i, val.to(output_dtype))
                                else:
                                    prev = ld(send_buf + out_base + i).to(tl.float32)
                                    st(send_buf + out_base + i, (prev + val).to(output_dtype))
                            valid_j_count += 1


@triton_dist.jit
def kernel_get_ag_splits_and_recv_offset_intra_node(
    topk_indices,
    local_splits_buf,  # symm buf, [num_experts + 1] (with drop token)
    full_splits_buf,  # symm buf, [world_size, num_experts + 1]
    splits_signal_buf,  # symm buf, [world_size]
    num_input_tokens_per_rank,  # [world_size]
    cumsum_input_tokens_per_rank,  # [world_size]
    num_recv_tokens_per_rank_cpu,  # pin memory, [world_size]
    cumsum_recv_tokens_per_rank,  # [world_size]
    recv_buf_offset_per_expert,  # [world_size, experts_per_rank, world_size]
    grid_sync_counter,  # [1] zero init
    full_scatter_indices,  # [num_total_tokens, topk]
    token_dst_scatter_idx,  # [nnodes, max_tokens, topk]
    full_splits_buf_expert_stride,
    local_world_size,
    max_tokens,
    experts_per_rank,
    topk: int,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_experts = experts_per_rank * world_size
    elem_size = (tl.constexpr(local_splits_buf.dtype.element_ty.primitive_bitwidth) // 8)
    nbytes = (full_splits_buf_expert_stride * elem_size)  # num_drop_token is counted in position `num_experts`
    threads_per_block = num_warps * threads_per_warp()
    thread_idx = tid(0)

    for remote_rank in range(pid, world_size, num_pid):
        libshmem_device.putmem_signal_nbi_block(
            full_splits_buf + rank * full_splits_buf_expert_stride,
            local_splits_buf,
            nbytes,
            splits_signal_buf + rank,
            1,
            libshmem_device.MORI_SIGNAL_SET,
            remote_rank,
        )

    # Ensure that all communication has been completed
    barrier_on_this_grid(grid_sync_counter, False)
    if pid == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_sync_counter, False)

    offs = tl.arange(0, BLOCK_SIZE)
    full_splits_mask = offs[:] < full_splits_buf_expert_stride
    recv_buf_offset_mask = offs[:] < num_experts  # do not count drop token

    for target_rank in range(pid, world_size, num_pid):
        token = dl.wait(splits_signal_buf + target_rank, 1, "sys", "acquire")
        full_splits_buf = dl.consume_token(full_splits_buf, token)
        __syncthreads()
        for expert_idx in range(thread_idx, num_experts, threads_per_block):
            val = ld(
                full_splits_buf + target_rank * full_splits_buf_expert_stride + expert_idx,
                semantic="acquire",
            )
            ep_rank = expert_idx // experts_per_rank
            expert_idx_intra_rank = expert_idx % experts_per_rank
            st(
                recv_buf_offset_per_expert + ep_rank * experts_per_rank * world_size +
                expert_idx_intra_rank * world_size + target_rank,
                val,
                semantic="release",
            )
        __syncthreads()
        splits_cur_rank = tl.load(
            full_splits_buf + target_rank * full_splits_buf_expert_stride + offs,
            mask=full_splits_mask,
            other=0,
            volatile=True,
        )
        total_topk_token_cur_rank = tl.sum(splits_cur_rank)
        num_input_tokens_cur_rank = total_topk_token_cur_rank // topk
        tl.store(num_input_tokens_per_rank + target_rank, num_input_tokens_cur_rank)
        tl.store(cumsum_input_tokens_per_rank + target_rank, num_input_tokens_cur_rank)
        __syncthreads()

    barrier_on_this_grid(grid_sync_counter, False)

    for ep_rank in range(pid, world_size, num_pid):
        splits_cur_rank = tl.load(
            recv_buf_offset_per_expert + ep_rank * num_experts + offs,
            mask=recv_buf_offset_mask,
            other=0,
            volatile=True,
        )
        recv_tokens = tl.sum(splits_cur_rank)
        cusum_splits_cur_rank = tl.cumsum(splits_cur_rank)
        cusum_splits_exclude = cusum_splits_cur_rank - splits_cur_rank
        tl.store(
            recv_buf_offset_per_expert + ep_rank * num_experts + offs,
            cusum_splits_exclude,
            mask=recv_buf_offset_mask,
        )
        tl.store(num_recv_tokens_per_rank_cpu + ep_rank, recv_tokens)
        tl.store(cumsum_recv_tokens_per_rank + ep_rank, recv_tokens)
    __syncthreads()

    # grid sync: wait all pid to save recv_tokens to cumsum_recv_tokens_per_rank
    barrier_on_this_grid(grid_sync_counter, False)

    if pid == 0:
        rank_mask = offs[:] < world_size
        recv_tokens_per_ranks = tl.load(
            cumsum_recv_tokens_per_rank + offs,
            mask=rank_mask,
            other=0,
            volatile=True,
        )
        cusum_recv_tokens = tl.cumsum(recv_tokens_per_ranks)
        cusum_recv_tokens_exclude = cusum_recv_tokens - recv_tokens_per_ranks
        tl.store(
            cumsum_recv_tokens_per_rank + offs,
            cusum_recv_tokens_exclude,
            mask=rank_mask,
        )

        input_tokens_per_ranks = tl.load(
            cumsum_input_tokens_per_rank + offs,
            mask=rank_mask,
            other=0,
            volatile=True,
        )
        cusum_input_tokens = tl.cumsum(input_tokens_per_ranks)
        cusum_input_tokens_exclude = cusum_input_tokens - input_tokens_per_ranks
        tl.store(
            cumsum_input_tokens_per_rank + offs,
            cusum_input_tokens_exclude,
            mask=rank_mask,
        )
        __syncthreads()

    barrier_on_this_grid(grid_sync_counter, False)

    if full_scatter_indices:
        tl.static_assert(token_dst_scatter_idx is not None)
        barrier_on_this_grid(grid_sync_counter, False)

        tokens_start = (tl.load(cumsum_input_tokens_per_rank + rank, volatile=True) * topk)
        num_tokens_target_rank = (tl.load(num_input_tokens_per_rank + rank, volatile=True) * topk)
        token_dst_scatter_idx_base_ptr = token_dst_scatter_idx
        __syncthreads()
        for token_idx in range(
                thread_idx + pid * threads_per_block,
                num_tokens_target_rank,
                threads_per_block * num_pid,
        ):
            scatter_idx = ld(full_scatter_indices + tokens_start + token_idx)
            expert_idx = ld(topk_indices + token_idx)
            expert_rank = expert_idx // experts_per_rank
            if expert_rank < world_size:
                global_out_rank_offset = ld(cumsum_recv_tokens_per_rank + expert_rank)
                st(
                    token_dst_scatter_idx_base_ptr + token_idx,
                    scatter_idx - global_out_rank_offset,
                )
        __syncthreads()


def get_ag_splits_and_recv_offset_for_dispatch_intra_node(
    topk_indices,
    local_splits,
    full_splits_buf,
    splits_signal_buf,
    topk,
    local_world_size,
    world_size,
    max_tokens,
    experts_per_rank,
    full_scatter_indices=None,
    cpu_default_val=-1,
    offset_dtype=torch.int32,
    num_sm=20,
):
    """
    Compute recv_buf_offset_per_expert and num_recv_tokens_per_rank for intra-node dispatch.

    AMD implementation uses mori_shmem device API (putmem_signal_nbi_block + wait + barrier)
    to all-gather local_splits into full_splits_buf, then computes:
      - num_input_tokens_per_rank
      - recv_buf_offset_per_expert (exclusive cumsum per ep_rank)
      - num_recv_tokens_per_rank_cpu
      - token_dst_scatter_idx (when full_scatter_indices is provided)
    """
    device = full_splits_buf.device
    num_recv_tokens_per_rank_cpu = torch.empty((world_size, ), dtype=torch.int32, device="cpu", pin_memory=True)
    num_recv_tokens_per_rank_cpu.fill_(cpu_default_val)
    num_input_tokens_per_rank = torch.empty((world_size, ), dtype=torch.int32, device=device)
    token_dst_scatter_idx = None
    if full_scatter_indices is not None:
        assert len(full_scatter_indices.shape) == 2
        nnodes = world_size // local_world_size
        assert full_scatter_indices.dtype == offset_dtype
        token_dst_scatter_idx = torch.empty(
            (nnodes, max_tokens, topk),
            dtype=full_scatter_indices.dtype,
            device=full_scatter_indices.device,
        )

    recv_buf_offset_per_expert = torch.zeros(
        (world_size, experts_per_rank, world_size),
        dtype=offset_dtype,
        device=device,
    )
    cumsum_recv_tokens_per_rank = torch.empty((world_size, ), dtype=torch.int32, device=device)
    cumsum_input_tokens_per_rank = torch.empty((world_size, ), dtype=torch.int32, device=device)
    # Use world_size blocks — most phases process per-rank data,
    # fewer blocks = cheaper grid sync (5 syncs in this kernel)
    grid = (world_size, )
    num_grid_sync = 8
    counter_workspace = torch.zeros((num_grid_sync, ), dtype=torch.int32, device=device)
    assert splits_signal_buf.dtype == MORI_SHMEM_SIGNAL_DTYPE
    assert (len(full_splits_buf.shape) == 2 and full_splits_buf.shape[1] == local_splits.shape[0])
    assert full_splits_buf.shape[0] == world_size

    BLOCK_SIZE = 1 << (full_splits_buf.shape[1]).bit_length()
    assert BLOCK_SIZE >= full_splits_buf.shape[1]

    kernel_get_ag_splits_and_recv_offset_intra_node[grid](
        topk_indices,
        local_splits,
        full_splits_buf,
        splits_signal_buf,
        num_input_tokens_per_rank,
        cumsum_input_tokens_per_rank,
        num_recv_tokens_per_rank_cpu,
        cumsum_recv_tokens_per_rank,
        recv_buf_offset_per_expert,
        counter_workspace,
        full_scatter_indices,
        token_dst_scatter_idx,
        full_splits_buf.shape[1],
        local_world_size,
        max_tokens,
        experts_per_rank,
        topk,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )

    return (
        recv_buf_offset_per_expert,
        num_recv_tokens_per_rank_cpu,
        num_input_tokens_per_rank,
        token_dst_scatter_idx,
    )
