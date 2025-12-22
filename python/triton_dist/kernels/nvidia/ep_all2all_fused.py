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
import triton_dist
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.cuda.language_extra import (tid, atomic_add, ld_acquire, __syncthreads, ld_b32,
                                                            atomic_add_per_warp, st, ld)
from .common_ops import barrier_on_this_grid, barrier_all_intra_node_atomic_cas_block, NVSHMEM_SIGNAL_DTYPE
from triton_dist.tools.profiler import Profiler
from triton.language import core
from .memory_ops import (load_v4, store_v4, zero_vec_f32, unpack_bf16x2_f32, pack_f32_bf16x2, copy_warp,
                         copy_1d_tilewise_kernel)


@core.extern
def sync_warp(_semantic=None):
    return tl.inline_asm_elementwise(
        asm="""
        bar.warp.sync 0xffffffff;
        mov.u32 $0, 0;
        """,
        constraints=("=r"),
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def consume_token(token, ptr, _semantic=None):
    ret_ptr = tl.inline_asm_elementwise(
        asm="""
        mov.b64 $0, $1;
        """,
        constraints=("=l,l,r"),
        args=[ptr, token],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )
    return tl.cast(ret_ptr, dtype=tl.pointer_type(ptr.dtype.element_ty), bitcast=True, _semantic=_semantic)


@triton_dist.jit(do_not_specialize=["pid", "num_pid"])
def tile_kernel_dispatch_token_intra_node(
    pid,
    num_pid,
    counter_ptr,
    barriers_ptr,
    recv_buf_offset_per_expert,
    local_splits_buf,
    input_buf,  # recv token from other nodes
    output_buf,
    weight_send_buf,
    weight_recv_buf,
    topk_indices_tensor,  # [nnodes, max_tokens, topk]
    token_dst_scatter_idx,  # [self.nnodes, self.max_tokens, self.topk]
    num_input_tokens_per_rank,  # [world_size]
    token_sort_indices,  # [nndoes, max_tokens * topk]
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    experts_per_rank: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WITH_SCATTER_INDICES: tl.constexpr,
    num_warps: tl.constexpr,
    profiler: Profiler,
    ENABLE_PROFILING: tl.constexpr,
):
    weight_elem_size = 4
    bytes_per_token = 2 * hidden_size

    WARP_SIZE = 32
    rank = dl.rank()
    world_size = dl.num_ranks()
    thread_idx = tid(0)
    lane_idx = thread_idx % WARP_SIZE
    warp_id = thread_idx // WARP_SIZE
    total_warps = num_warps * num_pid
    global_warp_id = pid * num_warps + warp_id

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=True, task_type=0)
    token_num = tl.load(num_input_tokens_per_rank + rank)
    for send_token_offset in range(global_warp_id, token_num * topk, total_warps):
        sort_token_offset = ld(token_sort_indices + send_token_offset)
        if sort_token_offset >= 0:  # ignore dropped tokens
            token_offset = sort_token_offset // topk
            expert_idx = ld(topk_indices_tensor + sort_token_offset)
            expert_rank = expert_idx // experts_per_rank
            expert_idx_intra_rank = expert_idx % experts_per_rank
            if not WITH_SCATTER_INDICES:
                store_idx = atomic_add_per_warp(
                    recv_buf_offset_per_expert + expert_rank * experts_per_rank * world_size +
                    expert_idx_intra_rank * world_size + rank, 1, scope="gpu", semantic="relaxed")
            else:
                store_idx = ld(token_dst_scatter_idx + sort_token_offset)

            src_ptr = input_buf + token_offset * hidden_size
            dst_ptr = output_buf + store_idx.to(tl.int64) * hidden_size

            libshmem_device.putmem_warp(dst_ptr, src_ptr, bytes_per_token, expert_rank)

            if not WITH_SCATTER_INDICES:
                st(token_dst_scatter_idx + sort_token_offset, store_idx)

            if HAS_WEIGHT:
                libshmem_device.putmem_warp(weight_recv_buf + store_idx, weight_send_buf + sort_token_offset,
                                            weight_elem_size, expert_rank)
            sync_warp()
            if lane_idx == 0:
                tokens_this_expert = ld(local_splits_buf + expert_idx)
                sent_tokens = atomic_add(counter_ptr + expert_idx, 1, scope="gpu", semantic="relaxed")
                if sent_tokens == tokens_this_expert - 1:
                    libshmem_device.fence()
                    libshmem_device.signal_op(
                        barriers_ptr + expert_idx_intra_rank * world_size + rank,
                        1,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        expert_rank,
                    )
    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=False, task_type=0)
        profiler = profiler.record(is_start=True, task_type=1)
    if pid == 0:
        for i in range(thread_idx, experts_per_rank * world_size, num_warps * WARP_SIZE):
            tokens_this_expert = ld(local_splits_buf + i)
            if tokens_this_expert == 0:
                libshmem_device.signal_op(
                    barriers_ptr + i // experts_per_rank * world_size + rank,
                    1,
                    libshmem_device.NVSHMEM_SIGNAL_SET,
                    i // experts_per_rank,
                )
    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=False, task_type=1)
    return profiler


@triton_dist.jit(do_not_specialize=["pid", "num_pid"])
def tile_kernel_dispatch_token_intra_node_two_stage(
    pid,
    num_pid,
    barriers_ptr,
    token_rank_table_buf,
    token_indirect_pos_buf,
    recv_buf_offset_per_expert,
    input_buf,  # recv token from other nodes
    output_buf,
    dispatch_output_local,
    weight_send_buf,
    weight_recv_buf,
    topk_indices_tensor,  # [nnodes, max_tokens, topk]
    token_dst_scatter_idx,  # [self.nnodes, self.max_tokens, self.topk]
    num_input_tokens_per_rank,  # [world_size]
    num_recv_tokens_per_expert,  # [experts_per_rank]
    expert_ids_ptr,
    split_size_ptr,
    split_size_cum_ptr,
    tile_num_ptr,
    tile_num_cum_ptr,
    gemm_total_tiles_m,
    expert_offs_ptr,  # [experts_per_rank]
    token_sort_indices,  # [nndoes, max_tokens * topk]
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    experts_per_rank: tl.constexpr,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WITH_SCATTER_INDICES: tl.constexpr,
    USE_BLOCK_WISE_BARRIER: tl.constexpr,
    num_warps: tl.constexpr,
    num_tail_sms: tl.constexpr,
    profiler: Profiler,
    ENABLE_PROFILING: tl.constexpr,
):
    weight_elem_size = 4
    bytes_per_token = 2 * hidden_size
    num_pid -= num_tail_sms

    WARP_SIZE = 32
    rank = dl.rank()
    world_size = dl.num_ranks()
    thread_idx = tid(0)
    lane_idx = thread_idx % WARP_SIZE
    warp_id = thread_idx // WARP_SIZE
    total_warps = num_warps * num_pid
    global_warp_id = pid * num_warps + warp_id

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=True, task_type=0)
    token_num = tl.load(num_input_tokens_per_rank + rank)

    if pid < num_pid:
        for send_token_offset in range(global_warp_id, token_num * topk, total_warps):
            sort_token_offset = ld(token_sort_indices + send_token_offset)
            if sort_token_offset >= 0:  # ignore dropped tokens
                token_offset = sort_token_offset // topk
                expert_idx = ld(topk_indices_tensor + sort_token_offset)
                expert_rank = expert_idx // experts_per_rank
                expert_idx_intra_rank = expert_idx % experts_per_rank
                if not WITH_SCATTER_INDICES:
                    store_idx = atomic_add_per_warp(
                        recv_buf_offset_per_expert + expert_rank * experts_per_rank * world_size +
                        expert_idx_intra_rank * world_size + rank, 1, scope="gpu", semantic="relaxed")
                else:
                    store_idx = ld(token_dst_scatter_idx + sort_token_offset)

                if not WITH_SCATTER_INDICES:
                    st(token_dst_scatter_idx + sort_token_offset, store_idx)

                if HAS_WEIGHT:
                    libshmem_device.putmem_warp(weight_recv_buf + store_idx, weight_send_buf + sort_token_offset,
                                                weight_elem_size, expert_rank)

                src_ptr = input_buf + token_offset * hidden_size
                dst_ptr = output_buf + store_idx.to(tl.int64) * hidden_size

                has_sent = ld(token_rank_table_buf + token_offset * world_size + expert_rank)
                remote_token_indirect_pos = dl.symm_at(token_indirect_pos_buf, expert_rank)
                if has_sent < 0:
                    has_sent = store_idx
                    libshmem_device.putmem_warp(dst_ptr, src_ptr, bytes_per_token, expert_rank)
                    st(token_rank_table_buf + token_offset * world_size + expert_rank, store_idx)
                sync_warp()
                if lane_idx == 0:
                    libshmem_device.fence()
                    st(remote_token_indirect_pos + store_idx, has_sent, scope="sys", semantic="release")

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=False, task_type=0)
        profiler = profiler.record(is_start=True, task_type=1)

    if pid >= num_pid:
        if USE_BLOCK_WISE_BARRIER:
            for tile_id in range(pid - num_pid, gemm_total_tiles_m, num_tail_sms):
                expert_id = tl.load(expert_ids_ptr + tile_id)
                split_size = tl.load(split_size_ptr + expert_id)
                split_size_cum = tl.load(split_size_cum_ptr + tile_id)
                row_begin = split_size_cum
                tile_num = tl.load(tile_num_ptr + tile_id)
                tile_num_cum = tl.load(tile_num_cum_ptr + tile_id)
                tile_begin = tile_num_cum - tile_num
                local_pid_m = tile_id - tile_begin
                num_tokens_this_tile = min(GEMM_BLOCK_SIZE_M, split_size - local_pid_m * GEMM_BLOCK_SIZE_M)
                for token_offset in range(warp_id, num_tokens_this_tile, num_warps):
                    real_offset = row_begin + local_pid_m * GEMM_BLOCK_SIZE_M + token_offset
                    has_sent = ld_acquire(token_indirect_pos_buf + real_offset, scope="sys")
                    while has_sent < 0:
                        has_sent = ld_acquire(token_indirect_pos_buf + real_offset, scope="sys")
                    copy_warp(dispatch_output_local + real_offset * hidden_size, output_buf + has_sent * hidden_size,
                              bytes_per_token)
                __syncthreads()
                if thread_idx == 0:
                    st(barriers_ptr + tile_id, 1, scope="gpu", semantic="release")
        else:
            for expert_idx in range(pid - num_pid, experts_per_rank, num_tail_sms):
                recv_token_cur_expert = ld(num_recv_tokens_per_expert + rank * experts_per_rank + expert_idx)
                recv_offset_cur_expert = ld(expert_offs_ptr + expert_idx)
                for recv_token_offset in range(warp_id, recv_token_cur_expert, num_warps):
                    real_offset = recv_offset_cur_expert + recv_token_offset
                    has_sent = ld_acquire(token_indirect_pos_buf + real_offset, scope="sys")
                    while has_sent < 0:
                        has_sent = ld_acquire(token_indirect_pos_buf + real_offset, scope="sys")
                    # let's keep this comment as a reminder
                    # the code in comments corresponds to no local copy
                    # if has_sent != real_offset:
                    #     copy_warp(output_buf + real_offset * hidden_size, output_buf + has_sent * hidden_size, bytes_per_token)
                    copy_warp(dispatch_output_local + real_offset * hidden_size, output_buf + has_sent * hidden_size,
                              bytes_per_token)
                __syncthreads()
                if thread_idx == 0:
                    st(barriers_ptr + expert_idx, 1, scope="gpu", semantic="release")

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=False, task_type=1)
    return profiler


@triton_dist.jit(do_not_specialize=["pid", "num_pid"])
def tile_kernel_gather_combine_token_intra_node(
    pid,
    num_pid,
    counter_ptr,  # symm buffer, [max_tokens, hidden_size // gemm_block_size_n]
    barriers_ptr,  # symm buffer, per token barrier [max_tokens * topk, hidden_size // gemm_block_size_n]
    num_input_tokens_per_rank,  # [world_size]
    input_buf,  # symm buffer (recv token in dispatch stage)
    output_buf,  #[max_tokens, hidden]
    gate_input_buf,  # symm buffer [dynamic_num_of_tokens]
    gate_output_buf,  # symm buffer [max_tokens, topk]
    topk_indices_buf,  # [max_tokens, topk]
    token_dst_scatter_idx,  # [max_tokens, topk]
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    expert_per_rank: tl.constexpr,
    BARRIER_TOKEN_BLOCK_SIZE: tl.constexpr,  # same as group gemm block_size_n
    HAS_GATE: tl.constexpr,
    num_warps: tl.constexpr,
    profiler: Profiler,
    NEED_WAIT: tl.constexpr,
    ENABLE_PROFILING: tl.constexpr,
):
    tl.static_assert(
        hidden_size % BARRIER_TOKEN_BLOCK_SIZE == 0,
        f"hidden_size={hidden_size} must be divisible by BARRIER_TOKEN_BLOCK_SIZE={BARRIER_TOKEN_BLOCK_SIZE}")
    N_BARRIERS_PER_TOKEN: tl.constexpr = hidden_size // BARRIER_TOKEN_BLOCK_SIZE
    WARP_SIZE = 32

    rank = dl.rank()
    world_size = dl.num_ranks()
    thread_idx = tid(0)
    lane_idx = thread_idx % WARP_SIZE
    total_warps = num_warps * num_pid
    warp_id = thread_idx // WARP_SIZE
    global_warp_id = pid * num_warps + warp_id

    tl.static_assert(output_buf.dtype.element_ty == tl.bfloat16, "output_buf must be bfloat16")
    tl.static_assert(gate_input_buf.dtype.element_ty == tl.float32, "gate_input_buf must be float32")
    VEC_SIZE: tl.constexpr = 128 // (output_buf.dtype.element_ty.primitive_bitwidth)
    tl.static_assert(BARRIER_TOKEN_BLOCK_SIZE % VEC_SIZE == 0,
                     f"BARRIER_TOKEN_BLOCK_SIZE={BARRIER_TOKEN_BLOCK_SIZE} must be divisible by VEC_SIZE={VEC_SIZE}")
    tl.static_assert(hidden_size % VEC_SIZE == 0, f"hidden_size={hidden_size} must be divisible by VEC_SIZE={VEC_SIZE}")

    num_dispatch_token_cur_rank = tl.load(num_input_tokens_per_rank + rank)

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=True, task_type=0)

    for token_idx in range(global_warp_id, num_dispatch_token_cur_rank, total_warps):
        for elem_idx in range(lane_idx, hidden_size // VEC_SIZE, WARP_SIZE):
            acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8 = zero_vec_f32(VEC_SIZE)
            for j in range(topk):
                expert_idx = ld_b32(topk_indices_buf + token_idx * topk + j)
                expert_rank = expert_idx // expert_per_rank

                if expert_rank // world_size == 0:  # ignore dropped tokens

                    token_scatter_idx = ld_b32(token_dst_scatter_idx + token_idx * topk + j)
                    remote_barriers_ptr = dl.symm_at(barriers_ptr, expert_rank)
                    remote_input_ptr = dl.symm_at(input_buf, expert_rank)

                    if HAS_GATE and elem_idx == 0:
                        remote_gate_input_ptr = dl.symm_at(gate_input_buf, expert_rank)
                        gate_val = ld_b32(remote_gate_input_ptr + token_scatter_idx)
                        st(
                            gate_output_buf.to(tl.pointer_type(tl.uint32)) + token_idx * topk + j,
                            tl.cast(gate_val, dtype=tl.uint32, bitcast=True))

                    if NEED_WAIT:
                        barrier_n_idx = elem_idx * VEC_SIZE // BARRIER_TOKEN_BLOCK_SIZE
                        barrier_idx = token_scatter_idx * N_BARRIERS_PER_TOKEN + barrier_n_idx
                        token = ld_acquire(remote_barriers_ptr + barrier_idx, scope="sys")
                        while token != 1:
                            token = ld_acquire(remote_barriers_ptr + barrier_idx, scope="sys")

                        remote_input_ptr = consume_token(token, remote_input_ptr)

                    t1, t2, t3, t4 = load_v4(
                        remote_input_ptr + token_scatter_idx.to(tl.int64) * hidden_size + elem_idx * VEC_SIZE, "b32")
                    u1, u2, u3, u4, u5, u6, u7, u8 = unpack_bf16x2_f32(t1, t2, t3, t4)
                    acc1 += u1
                    acc2 += u2
                    acc3 += u3
                    acc4 += u4
                    acc5 += u5
                    acc6 += u6
                    acc7 += u7
                    acc8 += u8

            v1, v2, v3, v4 = pack_f32_bf16x2((acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8))
            store_v4(output_buf + token_idx.to(tl.int64) * hidden_size + elem_idx * VEC_SIZE, v1, v2, v3, v4, "b32")

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=False, task_type=0)
    return profiler


@triton_dist.jit(do_not_specialize=["pid", "num_pid"])
def tile_kernel_scatter_token_intra_node(
    pid,
    num_pid,
    barriers_ptr,  # symm buffer, per token barrier [max_tokens * topk * local_world_size, hidden_size // gemm_block_size_n]
    num_recv_tokens_per_rank,
    input_buf,  # symm buffer (recv token in dispatch stage)
    scatter_send_buf,  #[max_tokens, topk, hidden]
    output_buf,  #[max_tokens, hidden]
    gate_input_buf,  # symm buffer [dynamic_num_of_tokens]
    gate_output_buf,  # symm buffer [max_tokens, topk]
    reversed_token_indices_buf,  # [max_tokens, topk, 2]
    scatter_output_barrier_buf,  # [max_tokens, topk, ]
    hidden_size: tl.constexpr,
    BARRIER_TOKEN_BLOCK_SIZE: tl.constexpr,  # same as group gemm block_size_n
    HAS_GATE: tl.constexpr,
    num_warps: tl.constexpr,
    profiler: Profiler,
    ENABLE_PROFILING: tl.constexpr,
):
    tl.static_assert(
        hidden_size % BARRIER_TOKEN_BLOCK_SIZE == 0,
        f"hidden_size={hidden_size} must be divisible by BARRIER_TOKEN_BLOCK_SIZE={BARRIER_TOKEN_BLOCK_SIZE}")
    N_BARRIERS_PER_TOKEN: tl.constexpr = hidden_size // BARRIER_TOKEN_BLOCK_SIZE
    WARP_SIZE = 32

    rank = dl.rank()
    thread_idx = tid(0)
    lane_idx = thread_idx % WARP_SIZE
    total_warps = num_warps * num_pid
    warp_id = thread_idx // WARP_SIZE
    global_warp_id = pid * num_warps + warp_id

    tl.static_assert(output_buf.dtype.element_ty == tl.bfloat16, "output_buf must be bfloat16")
    tl.static_assert(gate_input_buf.dtype.element_ty == tl.float32, "gate_input_buf must be float32")
    VEC_SIZE: tl.constexpr = 128 // (output_buf.dtype.element_ty.primitive_bitwidth)
    tl.static_assert(BARRIER_TOKEN_BLOCK_SIZE % VEC_SIZE == 0,
                     f"BARRIER_TOKEN_BLOCK_SIZE={BARRIER_TOKEN_BLOCK_SIZE} must be divisible by VEC_SIZE={VEC_SIZE}")
    tl.static_assert(hidden_size % VEC_SIZE == 0, f"hidden_size={hidden_size} must be divisible by VEC_SIZE={VEC_SIZE}")

    num_combine_token_cur_rank = tl.load(num_recv_tokens_per_rank + rank)

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=True, task_type=0)

    tl.static_assert(reversed_token_indices_buf is not None)

    for token_idx in range(global_warp_id, num_combine_token_cur_rank, total_warps):
        input_token_idx = ld_b32(reversed_token_indices_buf + token_idx * 2)
        from_rank = ld_b32(reversed_token_indices_buf + token_idx * 2 + 1)
        for elem_idx in range(lane_idx, hidden_size // VEC_SIZE, WARP_SIZE):
            barrier_n_idx = elem_idx * VEC_SIZE // BARRIER_TOKEN_BLOCK_SIZE
            barrier_idx = token_idx * N_BARRIERS_PER_TOKEN + barrier_n_idx

            if HAS_GATE and elem_idx == 0:
                remote_gate_output_ptr = dl.symm_at(gate_output_buf, from_rank)
                gate_val = ld_b32(gate_input_buf + token_idx)
                st(
                    remote_gate_output_ptr.to(tl.pointer_type(tl.uint32)) + input_token_idx,
                    tl.cast(gate_val, dtype=tl.uint32, bitcast=True))
            while ld_acquire(barriers_ptr + barrier_idx, scope="gpu") != 1:
                pass

            remote_output_ptr = dl.symm_at(scatter_send_buf, from_rank)
            t1, t2, t3, t4 = load_v4(input_buf + token_idx * hidden_size + elem_idx * VEC_SIZE, "b32")
            store_v4(remote_output_ptr + input_token_idx * hidden_size + elem_idx * VEC_SIZE, t1, t2, t3, t4, "b32")

        remote_scatter_output_barrier_buf = dl.symm_at(scatter_output_barrier_buf, from_rank)
        if scatter_output_barrier_buf is not None:
            sync_warp()
            if lane_idx == 0:
                libshmem_device.fence()
                st(remote_scatter_output_barrier_buf + input_token_idx, 1, scope="sys", semantic="release")

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=False, task_type=0)
    return profiler


@triton_dist.jit(do_not_specialize=["pid", "num_pid"])
def tile_kernel_topk_reduce_token_intra_node(
    pid,
    num_pid,
    num_input_tokens_per_rank,  # [world_size]
    scatter_send_buf,  #[max_tokens, topk, hidden]
    scatter_send_barrier_buf,  # [max_tokens, topk, ]
    output_buf,  #[max_tokens, hidden]
    gate_input_buf,  # symm buffer [dynamic_num_of_tokens]
    topk_indices_buf,  # [max_tokens, topk]
    BLOCK_SIZE: tl.constexpr,
    topk: tl.constexpr,
    num_experts,
    hidden_size: tl.constexpr,
    num_warps: tl.constexpr,
    profiler: Profiler,
    ENABLE_PROFILING: tl.constexpr,
):
    WARP_SIZE = 32

    rank = dl.rank()
    thread_idx = tid(0)
    lane_idx = thread_idx % WARP_SIZE
    total_warps = num_warps * num_pid
    warp_id = thread_idx // WARP_SIZE
    global_warp_id = pid * num_warps + warp_id

    tl.static_assert(output_buf.dtype.element_ty == tl.bfloat16, "output_buf must be bfloat16")
    tl.static_assert(gate_input_buf.dtype.element_ty == tl.float32, "gate_input_buf must be float32")
    VEC_SIZE: tl.constexpr = 128 // (output_buf.dtype.element_ty.primitive_bitwidth)
    tl.static_assert(hidden_size % VEC_SIZE == 0, f"hidden_size={hidden_size} must be divisible by VEC_SIZE={VEC_SIZE}")

    tl.static_assert(hidden_size % BLOCK_SIZE == 0,
                     f"hidden_size={hidden_size} must be divisible by BLOCK_SIZE={BLOCK_SIZE}")

    num_dispatch_token_cur_rank = tl.load(num_input_tokens_per_rank + rank)

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=True, task_type=1)

    for token_idx in range(global_warp_id, num_dispatch_token_cur_rank, total_warps):
        if scatter_send_barrier_buf is not None:
            for j in range(topk):
                topk_index = ld_b32(topk_indices_buf + (token_idx.to(tl.int64) * topk + j))
                if topk_index < num_experts:  # ignore dropped tokens
                    if lane_idx == 0:
                        # seg_idx = elem_idx * VEC_SIZE // BLOCK_SIZE
                        val = ld_acquire(scatter_send_barrier_buf + (token_idx * topk + j), scope="sys")
                        while val < 0:
                            val = ld_acquire(scatter_send_barrier_buf + (token_idx * topk + j), scope="sys")
            sync_warp()

        for elem_idx in range(lane_idx, hidden_size // VEC_SIZE, WARP_SIZE):

            acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8 = zero_vec_f32(VEC_SIZE)
            for j in range(topk):
                t1, t2, t3, t4 = load_v4(
                    scatter_send_buf + (token_idx.to(tl.int64) * topk + j) * hidden_size + elem_idx * VEC_SIZE, "b32")
                u1, u2, u3, u4, u5, u6, u7, u8 = unpack_bf16x2_f32(t1, t2, t3, t4)
                acc1 += u1
                acc2 += u2
                acc3 += u3
                acc4 += u4
                acc5 += u5
                acc6 += u6
                acc7 += u7
                acc8 += u8

            v1, v2, v3, v4 = pack_f32_bf16x2((acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8))
            store_v4(output_buf + token_idx.to(tl.int64) * hidden_size + elem_idx * VEC_SIZE, v1, v2, v3, v4, "b32")

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=False, task_type=1)
    return profiler


@triton_dist.jit(do_not_specialize=["M"])
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

    accumulator = accumulator.to(a_ptrs.dtype.element_ty)
    if need_mask:
        c_mask = (tl.arange(0, BLOCK_SIZE_M) < M)[:, None] & (tl.arange(0, BLOCK_SIZE_N) < N)[None, :]
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        c_mask = (tl.arange(0, BLOCK_SIZE_N) < N)[None, :]
        tl.store(c_ptrs, accumulator, mask=c_mask)


@triton_dist.jit(do_not_specialize=["pid", "num_pid", "M"])
def tile_kernel_moe_grouped_gemm_nk_const(
    pid,
    num_pid,
    counter_ptr,
    barriers_ptr,  # symm buf; producer: [max_num_tiles_m * num_block_n]; consumer: [num_experts_per_rank, num_ranks] = [num_experts]
    a_ptr,
    b_ptr,
    c_ptr,
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
    profiler: Profiler,
    NEED_WAIT: tl.constexpr,
    NEED_NOTIFY: tl.constexpr,
    USE_BLOCK_WISE_BARRIER: tl.constexpr,
    IS_DISPATCH_TWO_STAGET: tl.constexpr,
    ENABLE_PROFILING: tl.constexpr,
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

    if NEED_WAIT:
        if ENABLE_PROFILING:
            profiler = profiler.record(is_start=True, task_type=2)
        if IS_DISPATCH_TWO_STAGET:
            if USE_BLOCK_WISE_BARRIER:
                barrier_idx = local_pid_m + tile_begin
                if thread_idx == 0:
                    while ld_acquire(barriers_ptr + barrier_idx, scope="gpu") != 1:
                        pass
                __syncthreads()
            else:
                barrier_idx = expert_id
                while ld_acquire(barriers_ptr + barrier_idx, scope="gpu") != 1:
                    pass
        else:
            if thread_idx < world_size:
                barrier_idx = expert_id * world_size + thread_idx
                while ld_acquire(barriers_ptr + barrier_idx, scope="gpu") != 1:
                    pass
            __syncthreads()
        if ENABLE_PROFILING:
            profiler = profiler.record(is_start=False, task_type=2)

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=True, task_type=3)

    row_remain = split_size - local_pid_m * BLOCK_SIZE_M

    offs_bn = (pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = (b_ptr + expert_id.to(tl.int64) * stride_be + offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    offs_token = row_begin.to(tl.int64) + local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    a_ptrs = (a_ptr + offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=False, task_type=3)
        profiler = profiler.record(is_start=True, task_type=4)

    if row_remain >= BLOCK_SIZE_M:
        dot_k_const(a_ptrs, b_ptrs, c_ptrs, row_remain, min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N), K, stride_ak,
                    stride_bk, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, False)
    elif row_remain > 0:
        dot_k_const(a_ptrs, b_ptrs, c_ptrs, row_remain, min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N), K, stride_ak,
                    stride_bk, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, True)

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=False, task_type=4)

    if NEED_NOTIFY:
        if ENABLE_PROFILING:
            profiler = profiler.record(is_start=True, task_type=5)
        # per token barrier
        __syncthreads()
        token_begin = row_begin + local_pid_m * BLOCK_SIZE_M
        thread_idx = tid(0)
        valid_tokens = min(row_remain, BLOCK_SIZE_M)
        if thread_idx < valid_tokens:
            st(barriers_ptr + (token_begin + thread_idx) * num_block_n + pid_n, 1, scope="gpu", semantic="release")

        if ENABLE_PROFILING:
            profiler = profiler.record(is_start=False, task_type=5)
    return profiler


@triton_dist.jit
def transposed_dot(
    a_ptrs,
    b_ptrs,
    c_ptrs,
    split_size,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bm: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    for m in range(0, split_size, BLOCK_SIZE_M):
        a = tl.load(a_ptrs, mask=(m + tl.arange(0, BLOCK_SIZE_M) < split_size)[None, :])
        b = tl.load(b_ptrs, mask=(m + tl.arange(0, BLOCK_SIZE_M) < split_size)[:, None])

        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_M * stride_am
        b_ptrs += BLOCK_SIZE_M * stride_bm

    accumulator = accumulator.to(a_ptrs.dtype.element_ty)
    mask_c = (tl.arange(0, BLOCK_SIZE_N) < N)[:, None] & (tl.arange(0, BLOCK_SIZE_K) < K)[None, :]
    tl.store(c_ptrs, accumulator, mask=mask_c)


@triton_dist.jit(do_not_specialize=["pid", "num_pid", "M"])
def tile_kernel_transposed_moe_grouped_gemm_nk_const(
    pid,
    num_pid,
    grad_output_ptr,
    orig_input_ptr,
    grad_weight_ptr,
    split_size_ptr,
    split_size_cum_per_expert_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    stride_am: tl.constexpr,
    stride_an: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_ce: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_ck: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    profiler: Profiler,
    ENABLE_PROFILING: tl.constexpr,
):
    num_block_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    # pid = tl.program_id(axis=0)
    # pid_g = tl.program_id(axis=1)
    pid_g = pid // (num_block_n * num_block_k)
    pid = pid % (num_block_n * num_block_k)

    split_size = tl.load(split_size_ptr + pid_g)
    # if split_size <= 0:
    #     return
    split_begin = tl.load(split_size_cum_per_expert_ptr + pid_g)

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=True, task_type=6)

    pid_n = pid // num_block_k
    pid_k = pid % num_block_k
    pid_n, pid_k = tl.swizzle2d(pid_n, pid_k, num_block_n, num_block_k, GROUP_SIZE_M)

    offs_bk = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    b_ptrs = (orig_input_ptr + split_begin.to(tl.int64) * stride_bm + offs_m[:, None] * stride_bm +
              offs_bk[None, :] * stride_bk)

    offs_an = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = (grad_output_ptr + split_begin.to(tl.int64) * stride_am + offs_m[None, :] * stride_am +
              offs_an[:, None] * stride_an)

    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_ck = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
    c_ptrs = (grad_weight_ptr + pid_g * stride_ce + offs_cn[:, None] * stride_cn + offs_ck[None, :] * stride_ck)

    remain_n = min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N)
    remain_k = min(BLOCK_SIZE_K, K - pid_k * BLOCK_SIZE_K)

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=False, task_type=6)
        profiler = profiler.record(is_start=True, task_type=7)

    if split_size != 0:
        transposed_dot(
            a_ptrs,
            b_ptrs,
            c_ptrs,
            split_size,
            remain_n,
            remain_k,
            stride_am,
            stride_bm,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )
    else:
        mask_c = (tl.arange(0, BLOCK_SIZE_N) < remain_n)[:, None] & (tl.arange(0, BLOCK_SIZE_K) < remain_k)[None, :]
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=c_ptrs.dtype.element_ty), mask=mask_c)

    if ENABLE_PROFILING:
        profiler = profiler.record(is_start=False, task_type=7)
    return profiler


@triton_dist.jit(do_not_specialize=["M"])
def mega_kernel_dispatch_token_moe_grouped_gemm(
    task_counter_ptr,

    # dispatch token params
    recv_buf_offset_per_expert,
    local_splits_buf,
    input_buf,  # recv token from other nodes
    output_buf,
    weight_send_buf,
    weight_recv_buf,
    topk_indices_tensor,  # [nnodes, max_tokens, topk]
    token_dst_scatter_idx,  # [self.nnodes, self.max_tokens, self.topk]
    num_input_tokens_per_rank,  # [world_size]
    num_recv_tokens_per_expert,  # [world_size, experts_per_rank]
    token_sort_indices,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    experts_per_rank: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WITH_SCATTER_INDICES: tl.constexpr,

    #
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
    num_total_tiles_ptr,
    expert_offs_ptr,  # [experts_per_rank]
    G: tl.constexpr,
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

    # checkpoint params
    dispatch_output_local,

    #
    counter_ptr,  # local buf [max(num_experts, num_gemm_blocks_m)]
    barriers_ptr,  # symm buf [max(num_experts, num_gemm_blocks_m)]
    mega_token_rank_table_ptr,  # local buf [max_tokens, local_world_size]
    mega_token_indirect_pos_ptr,  # symm buf [max_tokens * topk * local_world_size]
    USE_BLOCK_WISE_BARRIER: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_TAIL_SMS: tl.constexpr,
    profiler_buffer,
    ENABLE_PROFILING: tl.constexpr,
):
    task_id = tl.atomic_add(task_counter_ptr, 1)
    group_gemm_total_tiles_m = tl.load(num_total_tiles_ptr)
    group_gemm_total_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    group_gemm_tasks = group_gemm_total_tiles_m * group_gemm_total_tiles_n
    total_tasks = num_dispatch_tasks + group_gemm_tasks

    is_leader = (tid(0) == 0)
    profiler = Profiler.create(profiler_buffer, group_id=0, num_groups=1, is_leader=is_leader,
                               ENABLE_PROFILING=ENABLE_PROFILING)

    while task_id < total_tasks:
        if task_id < num_dispatch_tasks:
            # dispatch token
            if NUM_TAIL_SMS > 0:
                profiler = tile_kernel_dispatch_token_intra_node_two_stage(
                    task_id,
                    num_dispatch_tasks,
                    barriers_ptr,
                    mega_token_rank_table_ptr,  # local buf [max_tokens, local_world_size]
                    mega_token_indirect_pos_ptr,  # symm buf [max_tokens * topk * local_world_size]
                    recv_buf_offset_per_expert,
                    input_buf,  # recv token from other nodes
                    output_buf,
                    dispatch_output_local,
                    weight_send_buf,
                    weight_recv_buf,
                    topk_indices_tensor,  # [nnodes, max_tokens, topk]
                    token_dst_scatter_idx,  # [self.nnodes, self.max_tokens, self.topk]
                    num_input_tokens_per_rank,  # [world_size]
                    num_recv_tokens_per_expert,  # [experts_per_rank]
                    expert_ids_ptr,
                    split_size_ptr,
                    split_size_cum_ptr,
                    tile_num_ptr,
                    tile_num_cum_ptr,
                    group_gemm_total_tiles_m,
                    expert_offs_ptr,  # [experts_per_rank]
                    token_sort_indices,
                    topk,
                    hidden_size,
                    experts_per_rank,
                    BLOCK_SIZE_M,
                    HAS_WEIGHT,
                    WITH_SCATTER_INDICES,
                    USE_BLOCK_WISE_BARRIER,
                    NUM_WARPS,
                    NUM_TAIL_SMS,
                    profiler,
                    ENABLE_PROFILING,
                )
            else:
                profiler = tile_kernel_dispatch_token_intra_node(
                    task_id,
                    num_dispatch_tasks,
                    counter_ptr,
                    barriers_ptr,
                    recv_buf_offset_per_expert,
                    local_splits_buf,
                    input_buf,  # recv token from other nodes
                    output_buf,
                    weight_send_buf,
                    weight_recv_buf,
                    topk_indices_tensor,  # [nnodes, max_tokens, topk]
                    token_dst_scatter_idx,  # [self.nnodes, self.max_tokens, self.topk]
                    num_input_tokens_per_rank,  # [world_size]
                    token_sort_indices,
                    topk,
                    hidden_size,
                    experts_per_rank,
                    HAS_WEIGHT,
                    WITH_SCATTER_INDICES,
                    NUM_WARPS,
                    profiler,
                    ENABLE_PROFILING,
                )
        else:
            # group gemm
            profiler = tile_kernel_moe_grouped_gemm_nk_const(
                task_id - num_dispatch_tasks,
                group_gemm_tasks,
                counter_ptr,
                barriers_ptr,
                a_ptr if NUM_TAIL_SMS <= 0 else dispatch_output_local,
                b_ptr,
                c_ptr,
                expert_ids_ptr,
                split_size_ptr,
                split_size_cum_ptr,
                tile_num_ptr,
                tile_num_cum_ptr,
                num_total_tiles_ptr,
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
                profiler,
                NEED_WAIT=True,
                NEED_NOTIFY=False,
                USE_BLOCK_WISE_BARRIER=USE_BLOCK_WISE_BARRIER,
                IS_DISPATCH_TWO_STAGET=NUM_TAIL_SMS > 0,
                ENABLE_PROFILING=ENABLE_PROFILING,
            )
        task_id = tl.atomic_add(task_counter_ptr, 1)


@triton_dist.jit(do_not_specialize=["M"])
def mega_kernel_moe_grouped_gemm_combine_token(
    task_counter_ptr,

    # grouped gemm params
    a_ptr,
    b_ptr,
    c_ptr,
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

    # combine token params
    num_input_tokens_per_rank,  # [world_size]
    num_recv_tokens_per_rank,  # [world_size]
    input_buf,  # symm buffer (recv token in dispatch stage)
    scatter_output_buf,  # symm buffer [max_tokens, topk, hidden]
    scatter_output_barrier_buf,  # [max_tokens, topk, ]
    output_buf,  #[max_tokens, hidden]
    gate_input_buf,  # symm buffer [dynamic_num_of_tokens]
    gate_output_buf,  # symm buffer [max_tokens, topk]
    topk_indices_buf,  # [max_tokens, topk]
    token_dst_scatter_idx,  # [max_tokens, topk]
    reversed_token_scatter_idx,  # [max_tokens, topk, 2]
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    expert_per_rank: tl.constexpr,
    HAS_GATE: tl.constexpr,
    USE_SCATTER_MODE: tl.constexpr,
    #
    num_combine_tasks: tl.constexpr,
    num_reduce_tasks: tl.constexpr,

    #
    counter_ptr,  # local buf [max_tokens * topk * local_world_size, hidden_size // gemm_block_size_n]
    barriers_ptr,  # symm buf [max_tokens * topk * local_world_size, hidden_size // gemm_block_size_n]
    barrier_all_workspace_ptr,  # symm buf [num_sms, num_ranks]
    grid_barrier_workspace_ptr,  # symm buf [num_sms, num_ranks]
    NUM_WARPS: tl.constexpr,
    profiler_buffer,
    ENABLE_PROFILING: tl.constexpr,
):
    group_gemm_total_tiles_m = tl.load(num_total_tiles_ptr)
    group_gemm_total_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    group_gemm_tasks = group_gemm_total_tiles_m * group_gemm_total_tiles_n

    is_leader = (tid(0) == 0)
    profiler = Profiler.create(profiler_buffer, group_id=0, num_groups=1, is_leader=is_leader,
                               ENABLE_PROFILING=ENABLE_PROFILING)

    sm_id = tl.program_id(0)
    num_sms = tl.num_programs(0)
    num_experts = expert_per_rank * dl.num_ranks()

    if not USE_SCATTER_MODE:
        if ENABLE_PROFILING:
            profiler = profiler.record(is_start=True, task_type=0)

        for task_id in range(sm_id, group_gemm_tasks, num_sms):
            profiler = tile_kernel_moe_grouped_gemm_nk_const(
                task_id,
                group_gemm_tasks,
                counter_ptr,
                barriers_ptr,
                a_ptr,
                b_ptr,
                c_ptr,
                expert_ids_ptr,
                split_size_ptr,
                split_size_cum_ptr,
                tile_num_ptr,
                tile_num_cum_ptr,
                num_total_tiles_ptr,
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
                profiler,
                NEED_WAIT=False,
                NEED_NOTIFY=False,
                IS_DISPATCH_TWO_STAGET=False,
                USE_BLOCK_WISE_BARRIER=False,
                ENABLE_PROFILING=False,
            )

        if ENABLE_PROFILING:
            profiler = profiler.record(is_start=False, task_type=0)
            profiler = profiler.record(is_start=True, task_type=1)

        rank = dl.rank()
        world_size = dl.num_ranks()
        barrier_on_this_grid(grid_barrier_workspace_ptr, False)
        if sm_id == 0:
            barrier_all_intra_node_atomic_cas_block(rank, rank, world_size, barrier_all_workspace_ptr)
        barrier_on_this_grid(grid_barrier_workspace_ptr, False)

        if ENABLE_PROFILING:
            profiler = profiler.record(is_start=False, task_type=1)
            profiler = profiler.record(is_start=True, task_type=2)

        profiler = tile_kernel_gather_combine_token_intra_node(
            sm_id,
            num_sms,
            counter_ptr,  # local buffer, [max_tokens, hidden_size // gemm_block_size_n]
            barriers_ptr,  # symm buffer, per token barrier [max_tokens * topk, hidden_size // gemm_block_size_n]
            num_input_tokens_per_rank,  # [world_size]
            c_ptr,  # symm buffer (recv token in dispatch stage)
            output_buf,  #[max_tokens, hidden]
            gate_input_buf,  # symm buffer [dynamic_num_of_tokens]
            gate_output_buf,  # symm buffer [max_tokens, topk]
            topk_indices_buf,  # [max_tokens, topk]
            token_dst_scatter_idx,  # [max_tokens, topk]
            topk,
            hidden_size,
            expert_per_rank,
            BLOCK_SIZE_N,  # same as group gemm block_size_n
            HAS_GATE,
            NUM_WARPS,
            profiler,
            NEED_WAIT=False,
            ENABLE_PROFILING=False,
        )

        if ENABLE_PROFILING:
            profiler = profiler.record(is_start=False, task_type=2)

    else:
        if scatter_output_barrier_buf is not None:
            num_reduce_tasks = num_reduce_tasks
        else:
            num_reduce_tasks = 0
        total_tasks = num_combine_tasks + group_gemm_tasks + num_reduce_tasks
        task_id = tl.atomic_add(task_counter_ptr, 1)
        while task_id < total_tasks:
            # put group gemm later to avoid overfill sms
            if task_id >= num_combine_tasks and task_id < num_combine_tasks + group_gemm_tasks:
                # group gemm
                profiler = tile_kernel_moe_grouped_gemm_nk_const(
                    task_id - num_combine_tasks,
                    group_gemm_tasks,
                    counter_ptr,
                    barriers_ptr,
                    a_ptr,
                    b_ptr,
                    c_ptr,
                    expert_ids_ptr,
                    split_size_ptr,
                    split_size_cum_ptr,
                    tile_num_ptr,
                    tile_num_cum_ptr,
                    num_total_tiles_ptr,
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
                    profiler,
                    NEED_WAIT=False,
                    NEED_NOTIFY=True,
                    IS_DISPATCH_TWO_STAGET=False,
                    USE_BLOCK_WISE_BARRIER=False,
                    ENABLE_PROFILING=ENABLE_PROFILING,
                )
            elif task_id < num_combine_tasks:
                profiler = tile_kernel_scatter_token_intra_node(
                    task_id,
                    num_combine_tasks,
                    # counter_ptr, # symm buffer, [max_tokens * topk * local_world_size, hidden_size // gemm_block_size_n]
                    barriers_ptr,  # symm buffer, per token barrier [max_tokens * topk * local_world_size, hidden_size // gemm_block_size_n]
                    num_recv_tokens_per_rank,
                    input_buf,  # symm buffer (recv token in dispatch stage)
                    scatter_output_buf,  #[max_tokens, topk, hidden]
                    output_buf,  #[max_tokens, hidden]
                    gate_input_buf,  # symm buffer [dynamic_num_of_tokens]
                    gate_output_buf,  # symm buffer [max_tokens, topk]
                    reversed_token_scatter_idx,  # [max_tokens, topk, 2]
                    scatter_output_barrier_buf,  # [max_tokens, topk, ]
                    # non_drop_token_count_buf,  # [max_tokens, ]
                    hidden_size,
                    BLOCK_SIZE_N,  # same as group gemm block_size_n
                    HAS_GATE,
                    NUM_WARPS,
                    profiler,
                    ENABLE_PROFILING=ENABLE_PROFILING,
                )
            else:  # task_id >= num_combine_tasks + group_gemm_tasks
                profiler = tile_kernel_topk_reduce_token_intra_node(
                    task_id - num_combine_tasks - group_gemm_tasks,
                    num_reduce_tasks,
                    num_input_tokens_per_rank,  # [world_size]
                    scatter_output_buf,  #[max_tokens, topk, hidden]
                    scatter_output_barrier_buf,  # [max_tokens, topk, ]
                    output_buf,  #[max_tokens, hidden]
                    gate_input_buf,  # symm buffer [dynamic_num_of_tokens]
                    topk_indices_buf,  # [max_tokens, topk]
                    BLOCK_SIZE_N,  # same as group gemm block_size_n
                    topk,
                    num_experts,
                    hidden_size,
                    NUM_WARPS,
                    profiler,
                    ENABLE_PROFILING=ENABLE_PROFILING,
                )
            task_id = tl.atomic_add(task_counter_ptr, 1)

        if scatter_output_barrier_buf is None:
            rank = dl.rank()
            world_size = dl.num_ranks()
            barrier_on_this_grid(grid_barrier_workspace_ptr, False)
            if sm_id == 0:
                barrier_all_intra_node_atomic_cas_block(rank, rank, world_size, barrier_all_workspace_ptr)
            barrier_on_this_grid(grid_barrier_workspace_ptr, False)

            profiler = tile_kernel_topk_reduce_token_intra_node(
                sm_id,
                num_sms,
                num_input_tokens_per_rank,  # [world_size]
                scatter_output_buf,  #[max_tokens, topk, hidden]
                scatter_output_barrier_buf,  # [max_tokens, topk, ]
                output_buf,  #[max_tokens, hidden]
                gate_input_buf,  # symm buffer [dynamic_num_of_tokens]
                topk_indices_buf,  # [max_tokens, topk]
                BLOCK_SIZE_N,  # same as group gemm block_size_n
                topk,
                num_experts,
                hidden_size,
                NUM_WARPS,
                profiler,
                ENABLE_PROFILING=ENABLE_PROFILING,
            )


@triton_dist.jit(do_not_specialize=["M"])
def mega_kernel_moe_grouped_gemm_combine_token_transposed_grouped_gemm(
    task_counter_ptr,

    # grouped gemm params
    a_ptr,
    b_ptr,
    c_ptr,
    expert_ids_ptr,
    split_size_ptr,
    split_size_cum_ptr,
    tile_num_ptr,
    tile_num_cum_ptr,
    num_total_tiles_ptr,
    G: tl.constexpr,
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

    # combine token params
    num_input_tokens_per_rank,  # [world_size]
    num_recv_tokens_per_rank,  # [world_size]
    input_buf,  # symm buffer (recv token in dispatch stage)
    scatter_output_buf,  # symm buffer [max_tokens, topk, hidden]
    scatter_output_barrier_buf,  # [max_tokens, topk, ]
    output_buf,  #[max_tokens, hidden]
    gate_input_buf,  # symm buffer [dynamic_num_of_tokens]
    gate_output_buf,  # symm buffer [max_tokens, topk]
    topk_indices_buf,  # [max_tokens, topk]
    token_dst_scatter_idx,  # [max_tokens, topk]
    reversed_token_scatter_idx,  # [max_tokens, topk, 2]
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    expert_per_rank: tl.constexpr,
    HAS_GATE: tl.constexpr,
    USE_SCATTER_MODE: tl.constexpr,
    #
    num_combine_tasks: tl.constexpr,
    num_reduce_tasks: tl.constexpr,

    # transposed group gemm params
    grad_output_ptr,
    orig_input_ptr,
    grad_weight_ptr,
    split_size_cum_per_expert_ptr,
    grad_N: tl.constexpr,
    grad_K: tl.constexpr,
    grad_stride_am: tl.constexpr,
    grad_stride_an: tl.constexpr,
    grad_stride_bm: tl.constexpr,
    grad_stride_bk: tl.constexpr,
    grad_stride_ce: tl.constexpr,
    grad_stride_cn: tl.constexpr,
    grad_stride_ck: tl.constexpr,
    grad_BLOCK_SIZE_M: tl.constexpr,
    grad_BLOCK_SIZE_N: tl.constexpr,
    grad_BLOCK_SIZE_K: tl.constexpr,
    grad_GROUP_SIZE_M: tl.constexpr,

    #
    counter_ptr,  # local buf [max_tokens * topk * local_world_size, hidden_size // gemm_block_size_n]
    barriers_ptr,  # symm buf [max_tokens * topk * local_world_size, hidden_size // gemm_block_size_n]
    barrier_all_workspace_ptr,  # symm buf [num_sms, num_ranks]
    grid_barrier_workspace_ptr,  # symm buf [num_sms, num_ranks]
    NUM_WARPS: tl.constexpr,
    profiler_buffer,
    ENABLE_PROFILING: tl.constexpr,
):
    group_gemm_total_tiles_m = tl.load(num_total_tiles_ptr)
    group_gemm_total_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    group_gemm_tasks = group_gemm_total_tiles_m * group_gemm_total_tiles_n
    transposed_group_gemm_tiles_n = tl.cdiv(grad_N, grad_BLOCK_SIZE_N)
    transposed_group_gemm_tiles_k = tl.cdiv(grad_K, grad_BLOCK_SIZE_K)
    transposed_group_gemm_tasks = transposed_group_gemm_tiles_n * transposed_group_gemm_tiles_k * G

    is_leader = (tid(0) == 0)
    profiler = Profiler.create(profiler_buffer, group_id=0, num_groups=1, is_leader=is_leader,
                               ENABLE_PROFILING=ENABLE_PROFILING)

    sm_id = tl.program_id(0)
    num_sms = tl.num_programs(0)
    num_experts = expert_per_rank * dl.num_ranks()

    if not USE_SCATTER_MODE:
        if ENABLE_PROFILING:
            profiler = profiler.record(is_start=True, task_type=0)

        for task_id in range(sm_id, group_gemm_tasks, num_sms):
            profiler = tile_kernel_moe_grouped_gemm_nk_const(
                task_id,
                group_gemm_tasks,
                counter_ptr,
                barriers_ptr,
                a_ptr,
                b_ptr,
                c_ptr,
                expert_ids_ptr,
                split_size_ptr,
                split_size_cum_ptr,
                tile_num_ptr,
                tile_num_cum_ptr,
                num_total_tiles_ptr,
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
                profiler,
                NEED_WAIT=False,
                NEED_NOTIFY=False,
                IS_DISPATCH_TWO_STAGET=False,
                USE_BLOCK_WISE_BARRIER=False,
                ENABLE_PROFILING=False,
            )

        if ENABLE_PROFILING:
            profiler = profiler.record(is_start=False, task_type=0)
            profiler = profiler.record(is_start=True, task_type=1)

        rank = dl.rank()
        world_size = dl.num_ranks()
        barrier_on_this_grid(grid_barrier_workspace_ptr, False)
        if sm_id == 0:
            barrier_all_intra_node_atomic_cas_block(rank, rank, world_size, barrier_all_workspace_ptr)
        barrier_on_this_grid(grid_barrier_workspace_ptr, False)

        if ENABLE_PROFILING:
            profiler = profiler.record(is_start=False, task_type=1)

        task_id = tl.atomic_add(task_counter_ptr, 1)
        while task_id < transposed_group_gemm_tasks + num_combine_tasks:
            if task_id < num_combine_tasks:
                if ENABLE_PROFILING:
                    profiler = profiler.record(is_start=True, task_type=2)
                profiler = tile_kernel_gather_combine_token_intra_node(
                    task_id,
                    num_combine_tasks,
                    counter_ptr,  # local buffer, [max_tokens, hidden_size // gemm_block_size_n]
                    barriers_ptr,  # symm buffer, per token barrier [max_tokens * topk, hidden_size // gemm_block_size_n]
                    num_input_tokens_per_rank,  # [world_size]
                    c_ptr,  # symm buffer (recv token in dispatch stage)
                    output_buf,  #[max_tokens, hidden]
                    gate_input_buf,  # symm buffer [dynamic_num_of_tokens]
                    gate_output_buf,  # symm buffer [max_tokens, topk]
                    topk_indices_buf,  # [max_tokens, topk]
                    token_dst_scatter_idx,  # [max_tokens, topk]
                    topk,
                    hidden_size,
                    expert_per_rank,
                    BLOCK_SIZE_N,  # same as group gemm block_size_n
                    HAS_GATE,
                    NUM_WARPS,
                    profiler,
                    NEED_WAIT=False,
                    ENABLE_PROFILING=False,
                )
                if ENABLE_PROFILING:
                    profiler = profiler.record(is_start=False, task_type=2)
            else:
                if ENABLE_PROFILING:
                    profiler = profiler.record(is_start=True, task_type=3)
                profiler = tile_kernel_transposed_moe_grouped_gemm_nk_const(
                    task_id - num_combine_tasks,
                    transposed_group_gemm_tasks,
                    grad_output_ptr,
                    orig_input_ptr,
                    grad_weight_ptr,
                    split_size_ptr,
                    split_size_cum_per_expert_ptr,
                    M,
                    grad_N,
                    grad_K,
                    G,
                    grad_stride_am,
                    grad_stride_an,
                    grad_stride_bm,
                    grad_stride_bk,
                    grad_stride_ce,
                    grad_stride_cn,
                    grad_stride_ck,
                    grad_BLOCK_SIZE_M,
                    grad_BLOCK_SIZE_N,
                    grad_BLOCK_SIZE_K,
                    grad_GROUP_SIZE_M,
                    profiler,
                    ENABLE_PROFILING=False,
                )
                if ENABLE_PROFILING:
                    profiler = profiler.record(is_start=False, task_type=3)
            task_id = tl.atomic_add(task_counter_ptr, 1)

    else:
        if scatter_output_barrier_buf is not None:
            num_reduce_tasks = num_reduce_tasks
        else:
            num_reduce_tasks = 0
        total_tasks = num_combine_tasks + group_gemm_tasks + num_reduce_tasks + transposed_group_gemm_tasks
        task_id = tl.atomic_add(task_counter_ptr, 1)
        while task_id < total_tasks:
            # put group gemm later to avoid overfill sms
            if task_id >= num_combine_tasks and task_id < num_combine_tasks + group_gemm_tasks:
                # group gemm
                profiler = tile_kernel_moe_grouped_gemm_nk_const(
                    task_id - num_combine_tasks,
                    group_gemm_tasks,
                    counter_ptr,
                    barriers_ptr,
                    a_ptr,
                    b_ptr,
                    c_ptr,
                    expert_ids_ptr,
                    split_size_ptr,
                    split_size_cum_ptr,
                    tile_num_ptr,
                    tile_num_cum_ptr,
                    num_total_tiles_ptr,
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
                    profiler,
                    NEED_WAIT=False,
                    NEED_NOTIFY=True,
                    IS_DISPATCH_TWO_STAGET=False,
                    USE_BLOCK_WISE_BARRIER=False,
                    ENABLE_PROFILING=ENABLE_PROFILING,
                )
            elif task_id < num_combine_tasks:
                profiler = tile_kernel_scatter_token_intra_node(
                    task_id,
                    num_combine_tasks,
                    # counter_ptr, # symm buffer, [max_tokens * topk * local_world_size, hidden_size // gemm_block_size_n]
                    barriers_ptr,  # symm buffer, per token barrier [max_tokens * topk * local_world_size, hidden_size // gemm_block_size_n]
                    num_recv_tokens_per_rank,
                    input_buf,  # symm buffer (recv token in dispatch stage)
                    scatter_output_buf,  #[max_tokens, topk, hidden]
                    output_buf,  #[max_tokens, hidden]
                    gate_input_buf,  # symm buffer [dynamic_num_of_tokens]
                    gate_output_buf,  # symm buffer [max_tokens, topk]
                    reversed_token_scatter_idx,  # [max_tokens, topk, 2]
                    scatter_output_barrier_buf,  # [max_tokens, topk, ]
                    # non_drop_token_count_buf,  # [max_tokens, ]
                    hidden_size,
                    BLOCK_SIZE_N,  # same as group gemm block_size_n
                    HAS_GATE,
                    NUM_WARPS,
                    profiler,
                    ENABLE_PROFILING=ENABLE_PROFILING,
                )
            elif task_id >= num_combine_tasks + group_gemm_tasks and task_id < num_combine_tasks + group_gemm_tasks + num_reduce_tasks:
                profiler = tile_kernel_topk_reduce_token_intra_node(
                    task_id - num_combine_tasks - group_gemm_tasks,
                    num_reduce_tasks,
                    num_input_tokens_per_rank,  # [world_size]
                    scatter_output_buf,  #[max_tokens, topk, hidden]
                    scatter_output_barrier_buf,  # [max_tokens, topk, ]
                    output_buf,  #[max_tokens, hidden]
                    gate_input_buf,  # symm buffer [dynamic_num_of_tokens]
                    topk_indices_buf,  # [max_tokens, topk]
                    BLOCK_SIZE_N,  # same as group gemm block_size_n
                    topk,
                    num_experts,
                    hidden_size,
                    NUM_WARPS,
                    profiler,
                    ENABLE_PROFILING=ENABLE_PROFILING,
                )
            else:  # task_id >= num_combine_tasks + group_gemm_tasks + num_reduce_tasks
                profiler = tile_kernel_transposed_moe_grouped_gemm_nk_const(
                    task_id - num_combine_tasks - group_gemm_tasks - num_reduce_tasks,
                    transposed_group_gemm_tasks,
                    grad_output_ptr,
                    orig_input_ptr,
                    grad_weight_ptr,
                    split_size_ptr,
                    split_size_cum_per_expert_ptr,
                    M,
                    grad_N,
                    grad_K,
                    G,
                    grad_stride_am,
                    grad_stride_an,
                    grad_stride_bm,
                    grad_stride_bk,
                    grad_stride_ce,
                    grad_stride_cn,
                    grad_stride_ck,
                    grad_BLOCK_SIZE_M,
                    grad_BLOCK_SIZE_N,
                    grad_BLOCK_SIZE_K,
                    grad_GROUP_SIZE_M,
                    profiler,
                    ENABLE_PROFILING=ENABLE_PROFILING,
                )
            task_id = tl.atomic_add(task_counter_ptr, 1)

        if scatter_output_barrier_buf is None:
            rank = dl.rank()
            world_size = dl.num_ranks()
            barrier_on_this_grid(grid_barrier_workspace_ptr, False)
            if sm_id == 0:
                barrier_all_intra_node_atomic_cas_block(rank, rank, world_size, barrier_all_workspace_ptr)
            barrier_on_this_grid(grid_barrier_workspace_ptr, False)

            profiler = tile_kernel_topk_reduce_token_intra_node(
                sm_id,
                num_sms,
                num_input_tokens_per_rank,  # [world_size]
                scatter_output_buf,  #[max_tokens, topk, hidden]
                scatter_output_barrier_buf,  # [max_tokens, topk, ]
                output_buf,  #[max_tokens, hidden]
                gate_input_buf,  # symm buffer [dynamic_num_of_tokens]
                topk_indices_buf,  # [max_tokens, topk]
                BLOCK_SIZE_N,  # same as group gemm block_size_n
                topk,
                num_experts,
                hidden_size,
                NUM_WARPS,
                profiler,
                ENABLE_PROFILING=ENABLE_PROFILING,
            )


@triton_dist.jit(do_not_specialize=["num_tokens"])
def kernel_get_ag_splits_and_recv_offset(
        num_tokens,  # int
        send_reqs_for_nodes, send_reqs_for_nodes_copy, send_reqs_recv_bufs,
        send_reqs_recv_bufs_copy,  # torch tensor, [nnodes, 2, max_tokens]
        token_sort_indices,  # [nnodes, max_tokens, topk]
        topk_indices_buf,  # symm buf, [nnodes, max_tokens, topk]
        topk_indices_buf_copy,  # torch tensor, [nnodes, max_tokens, topk]
        non_drop_token_count_buf,  # torch tensor, [nnodes, max_tokens, ]
        expert_indices_signal_buf, local_splits_buf,  # symm buf, [num_experts + 1, ] (with drop token)
        full_splits_buf,  # symm buf, [world_size, num_experts + 1]
        cumsum_full_splits_buf,  # symm buf, [world_size, num_experts + 1]
        splits_signal_buf,  # symm buf, [world_size, ]
        num_input_tokens_per_rank,  # [world_size, ]
        cumsum_input_tokens_per_rank,  # [world_size, ]
        num_recv_tokens_per_rank_cpu,  # pin memory, [world_size, ]
        cumsum_recv_tokens_per_rank,  # [world_size, ]
        send_buf_offset_per_expert,  # symm buf, [world_size, experts_per_rank, world_size]
        recv_buf_offset_per_expert,  # [world_size, experts_per_rank, world_size]
        recv_buf_tokens_per_expert,  # [world_size, experts_per_rank]
        grid_sync_counter,  #[1,] zero init
        full_global_scatter_indices,  # [num_total_tokens, topk]
        full_local_scatter_indices,  # symm buf [nnodes, max_tokens, topk]
        token_dst_scatter_idx,  #[nnodes, max_tokens, topk]
        reversed_token_scatter_idx,  # [self.nnodes, local_world_size, self.max_tokens, self.topk, 2]
        reversed_token_scatter_idx_copy,  # [self.nnodes, local_world_size, self.max_tokens, self.topk, 2]
        full_splits_buf_expert_stride, local_world_size, max_tokens, experts_per_rank, topk: int,
        BLOCK_SIZE: tl.constexpr,  # larger than num_experts
):
    # TODO: get topk_indices_buf
    rank = dl.rank()
    world_size = dl.num_ranks()
    nnodes = world_size // local_world_size
    local_rank = rank % local_world_size
    node_id = rank // local_world_size
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_experts = experts_per_rank * world_size
    elem_size = tl.constexpr(local_splits_buf.dtype.element_ty.primitive_bitwidth) // 8
    local_splits_nbytes = full_splits_buf_expert_stride * elem_size  # num_drop_token is counted in position `num_experts`
    index_elem_size = tl.constexpr(topk_indices_buf.dtype.element_ty.primitive_bitwidth) // 8
    n_warps: tl.constexpr = tl.extra.cuda.num_warps()
    threads_per_block = n_warps * 32
    thread_idx = tid(0)

    for remote_rank in range(pid, world_size, num_pid):
        libshmem_device.putmem_signal_nbi_block(
            full_splits_buf + rank * full_splits_buf_expert_stride,
            local_splits_buf,
            local_splits_nbytes,
            splits_signal_buf + rank,
            1,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            remote_rank,
        )

    # send expert indices and send_reqs
    if pid == 0:
        for node_offset in range(1, nnodes):
            target_node = (node_id + node_offset) % nnodes
            target_rank = local_rank + target_node * local_world_size
            indices_ptr = topk_indices_buf + node_id * max_tokens * topk
            send_req_src_ptr = send_reqs_for_nodes + target_node * max_tokens * 2
            send_req_dst_ptr = send_reqs_recv_bufs + node_id * max_tokens * 2
            msg_size = max_tokens * 2 * index_elem_size
            libshmem_device.putmem_nbi_block(send_req_dst_ptr, send_req_src_ptr, msg_size, target_rank)
            if full_local_scatter_indices:
                scatter_indices_ptr = full_local_scatter_indices + node_id * max_tokens * topk
                libshmem_device.putmem_nbi_block(
                    scatter_indices_ptr,
                    scatter_indices_ptr,
                    index_elem_size * max_tokens * topk,
                    target_rank,
                )
            libshmem_device.fence()
            libshmem_device.putmem_signal_nbi_block(
                indices_ptr,
                indices_ptr,
                index_elem_size * max_tokens * topk,
                expert_indices_signal_buf + rank,
                1,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                target_rank,
            )

    # Ensure that all communication has been completed
    barrier_on_this_grid(grid_sync_counter, False)
    if pid == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_sync_counter, False)

    offs = tl.arange(0, BLOCK_SIZE)
    full_splits_mask = (offs[:] < full_splits_buf_expert_stride)
    expert_mask = (offs[:] < num_experts)  # do not count drop token

    for target_rank in range(pid, world_size, num_pid):
        # libshmem_device.signal_wait_until(splits_signal_buf + target_rank, libshmem_device.NVSHMEM_CMP_EQ, 1)
        token = dl.wait(splits_signal_buf + target_rank, 1, "sys", "acquire")
        full_splits_buf = dl.consume_token(full_splits_buf, token)
        __syncthreads()
        for expert_idx in range(thread_idx, num_experts, threads_per_block):
            val = ld(full_splits_buf + target_rank * full_splits_buf_expert_stride + expert_idx, semantic="acquire")
            ep_rank = expert_idx // experts_per_rank
            expert_idx_intra_rank = expert_idx % experts_per_rank
            st(
                recv_buf_offset_per_expert + ep_rank * experts_per_rank * world_size +
                expert_idx_intra_rank * world_size + target_rank, val, semantic="release")
            st(
                send_buf_offset_per_expert + target_rank * world_size * experts_per_rank +
                expert_idx_intra_rank * world_size + ep_rank, val, scope="gpu", semantic="release")
            atomic_add(recv_buf_tokens_per_expert + ep_rank * experts_per_rank + expert_idx_intra_rank, val,
                       scope="gpu", semantic="release")
        __syncthreads()
        splits_cur_rank = tl.load(full_splits_buf + target_rank * full_splits_buf_expert_stride + offs,
                                  mask=full_splits_mask, other=0, volatile=True)
        total_topk_token_cur_rank = tl.sum(splits_cur_rank)
        total_topk_token_cur_rank_cum_sum = tl.cumsum(splits_cur_rank) - splits_cur_rank
        num_input_tokens_cur_rank = total_topk_token_cur_rank // topk
        tl.store(num_input_tokens_per_rank + target_rank, num_input_tokens_cur_rank)
        tl.store(cumsum_input_tokens_per_rank + target_rank, num_input_tokens_cur_rank)
        tl.store(cumsum_full_splits_buf + target_rank * full_splits_buf_expert_stride + offs,
                 total_topk_token_cur_rank_cum_sum, mask=full_splits_mask)
        __syncthreads()

    # recv full expert indices and send_reqs
    for node_offset in range(1, nnodes):
        src_node = (node_id + node_offset) % nnodes
        src_rank = local_rank + src_node * local_world_size
        # libshmem_device.signal_wait_until(expert_indices_signal_buf + src_rank, libshmem_device.NVSHMEM_CMP_EQ, 1)
        token = dl.wait(expert_indices_signal_buf + src_rank, 1, "sys", "acquire")
        send_reqs_recv_bufs = dl.consume_token(send_reqs_recv_bufs, token)
        topk_indices_buf = dl.consume_token(topk_indices_buf, token)
        if full_local_scatter_indices:
            full_local_scatter_indices = dl.consume_token(full_local_scatter_indices, token)

        __syncthreads()

    COPY_BLOCK_SIZE: tl.constexpr = 8192
    nelems_send_reqs_recv_bufs = nnodes * 2 * max_tokens
    copy_1d_tilewise_kernel(send_reqs_for_nodes_copy, send_reqs_for_nodes, nelems_send_reqs_recv_bufs, COPY_BLOCK_SIZE)
    copy_1d_tilewise_kernel(send_reqs_recv_bufs_copy, send_reqs_recv_bufs, nelems_send_reqs_recv_bufs, COPY_BLOCK_SIZE)
    nelems_topk_indices_buf = nnodes * max_tokens * topk
    copy_1d_tilewise_kernel(topk_indices_buf_copy, topk_indices_buf, nelems_topk_indices_buf, COPY_BLOCK_SIZE)

    # # grid sync
    barrier_on_this_grid(grid_sync_counter, False)

    for ep_rank in range(pid, world_size, num_pid):
        splits_cur_rank = tl.load(recv_buf_offset_per_expert + ep_rank * num_experts + offs, mask=expert_mask, other=0,
                                  volatile=True)
        recv_tokens = tl.sum(splits_cur_rank)
        cusum_splits_cur_rank = tl.cumsum(splits_cur_rank)
        cusum_splits_exclude = cusum_splits_cur_rank - splits_cur_rank
        tl.store(recv_buf_offset_per_expert + ep_rank * num_experts + offs, cusum_splits_exclude, mask=expert_mask)
        tl.store(num_recv_tokens_per_rank_cpu + ep_rank, recv_tokens)
        tl.store(cumsum_recv_tokens_per_rank + ep_rank, recv_tokens)

        num_send_tokens = tl.load(send_buf_offset_per_expert + ep_rank * num_experts + offs, mask=expert_mask, other=0,
                                  volatile=True)
        cumsum_send_tokens = tl.cumsum(num_send_tokens)
        offset_send_tokens = cumsum_send_tokens - num_send_tokens
        tl.store(send_buf_offset_per_expert + ep_rank * num_experts + offs, offset_send_tokens, mask=expert_mask)
    __syncthreads()

    # grid sync: wait all pid to save recv_tokens to cumsum_recv_tokens_per_rank
    barrier_on_this_grid(grid_sync_counter, False)

    # # compute cumsum of num_recv_tokens_per_rank_cpu
    if pid == 0:
        # BLOCK_SIZE is larger than num_experts
        rank_mask = (offs[:] < world_size)
        recv_tokens_per_ranks = tl.load(cumsum_recv_tokens_per_rank + offs, mask=rank_mask, other=0, volatile=True)
        cusum_recv_tokens = tl.cumsum(recv_tokens_per_ranks)
        cusum_recv_tokens_exclude = cusum_recv_tokens - recv_tokens_per_ranks
        tl.store(cumsum_recv_tokens_per_rank + offs, cusum_recv_tokens_exclude, mask=rank_mask)

        input_tokens_per_ranks = tl.load(cumsum_input_tokens_per_rank + offs, mask=rank_mask, other=0, volatile=True)
        cusum_input_tokens = tl.cumsum(input_tokens_per_ranks)
        cusum_input_tokens_exclude = cusum_input_tokens - input_tokens_per_ranks
        tl.store(cumsum_input_tokens_per_rank + offs, cusum_input_tokens_exclude, mask=rank_mask)
        __syncthreads()

    barrier_on_this_grid(grid_sync_counter, False)

    if full_global_scatter_indices:
        tl.static_assert(full_local_scatter_indices is None,
                         "Local scatter indices and global scatter indices cannot be used together")
        tl.static_assert(token_dst_scatter_idx is not None)

        # grid sync: wait cumsum_recv_tokens_per_rank computation
        barrier_on_this_grid(grid_sync_counter, False)

        for target_node_id in range(0, nnodes):
            target_rank = local_rank + target_node_id * local_world_size
            # tokens after topk gate
            tokens_start = tl.load(cumsum_input_tokens_per_rank + target_rank, volatile=True) * topk
            num_tokens_target_rank = tl.load(num_input_tokens_per_rank + target_rank, volatile=True) * topk
            token_dst_scatter_idx_base_ptr = token_dst_scatter_idx + target_node_id * max_tokens * topk
            __syncthreads()
            for token_idx in range(thread_idx + pid * threads_per_block, num_tokens_target_rank,
                                   threads_per_block * num_pid):
                scatter_idx = ld(full_global_scatter_indices + tokens_start + token_idx)
                expert_idx = ld(topk_indices_buf + (target_node_id * max_tokens * topk) + token_idx)
                expert_rank = expert_idx // experts_per_rank
                # skip drop token
                if expert_rank < world_size:
                    global_out_rank_offset = ld(cumsum_recv_tokens_per_rank + expert_rank)
                    """
                        scatter_idx is a global offset (outputs from all ranks are view as a single flatten buffer).
                        needs to be sub cumsum_recv_tokens_per_rank[expert_rank]
                        to get the index of local output buffer in expert_rank
                    """
                    to_idx = scatter_idx - global_out_rank_offset
                    st(token_dst_scatter_idx_base_ptr + token_idx, to_idx)
                    if reversed_token_scatter_idx:
                        reversed_token_scatter_idx_base_ptr = reversed_token_scatter_idx + target_node_id * local_world_size * max_tokens * topk * 2
                        remote_reversed_token_scatter_idx_base_ptr = dl.symm_at(reversed_token_scatter_idx_base_ptr,
                                                                                expert_rank)
                        st(remote_reversed_token_scatter_idx_base_ptr + to_idx * 2, token_idx)
                        st(remote_reversed_token_scatter_idx_base_ptr + to_idx * 2 + 1, target_rank)

                    if non_drop_token_count_buf:
                        atomic_add(non_drop_token_count_buf + target_node_id * max_tokens + token_idx // topk, 1,
                                   scope="gpu", semantic="relaxed")
            __syncthreads()
    elif full_local_scatter_indices:
        tl.static_assert(full_global_scatter_indices is None,
                         "Local scatter indices and global scatter indices cannot be used together")
        tl.static_assert(token_dst_scatter_idx is not None)

        # grid sync: wait cumsum_recv_tokens_per_rank computation
        barrier_on_this_grid(grid_sync_counter, False)

        for target_node_id in range(0, nnodes):
            target_rank = local_rank + target_node_id * local_world_size
            # tokens after topk gate
            tokens_start = tl.load(cumsum_input_tokens_per_rank + target_rank, volatile=True) * topk
            num_tokens_target_rank = tl.load(num_input_tokens_per_rank + target_rank, volatile=True) * topk
            token_dst_scatter_idx_base_ptr = token_dst_scatter_idx + target_node_id * max_tokens * topk
            __syncthreads()
            for token_idx in range(thread_idx + pid * threads_per_block, num_tokens_target_rank,
                                   threads_per_block * num_pid):
                scatter_idx = ld(full_local_scatter_indices + (target_node_id * max_tokens * topk) + token_idx)
                expert_idx = ld(topk_indices_buf + (target_node_id * max_tokens * topk) + token_idx)
                expert_rank = expert_idx // experts_per_rank
                expert_idx_intra_rank = expert_idx % experts_per_rank
                # skip drop token
                if expert_rank < world_size:
                    begin_idx_expert_from_input = ld(cumsum_full_splits_buf +
                                                     target_rank * full_splits_buf_expert_stride + expert_idx)
                    scatter_idx_intra_expert_from_input = scatter_idx - begin_idx_expert_from_input
                    begin_idx_expert_from_recv = ld(recv_buf_offset_per_expert +
                                                    expert_rank * experts_per_rank * world_size +
                                                    expert_idx_intra_rank * world_size + target_rank)
                    to_idx = scatter_idx_intra_expert_from_input + begin_idx_expert_from_recv
                    st(token_dst_scatter_idx_base_ptr + token_idx, to_idx)
                    if reversed_token_scatter_idx:
                        reversed_token_scatter_idx_base_ptr = reversed_token_scatter_idx + target_node_id * local_world_size * max_tokens * topk * 2
                        remote_reversed_token_scatter_idx_base_ptr = dl.symm_at(reversed_token_scatter_idx_base_ptr,
                                                                                expert_rank)
                        st(remote_reversed_token_scatter_idx_base_ptr + to_idx * 2, token_idx)
                        st(remote_reversed_token_scatter_idx_base_ptr + to_idx * 2 + 1, target_rank)

                    if token_sort_indices:
                        sort_offset = ld(send_buf_offset_per_expert + target_rank * world_size * experts_per_rank +
                                         expert_idx_intra_rank * world_size + expert_rank)
                        token_sort_indices_base_ptr = token_sort_indices + target_node_id * max_tokens * topk
                        st(token_sort_indices_base_ptr + scatter_idx_intra_expert_from_input + sort_offset, token_idx)

                    if non_drop_token_count_buf:
                        atomic_add(non_drop_token_count_buf + target_node_id * max_tokens + token_idx // topk, 1,
                                   scope="gpu", semantic="relaxed")
            __syncthreads()

    if reversed_token_scatter_idx:
        # Ensure that all communication has been completed
        barrier_on_this_grid(grid_sync_counter, False)
        if pid == 0:
            libshmem_device.barrier_all_block()
        barrier_on_this_grid(grid_sync_counter, False)
        copy_1d_tilewise_kernel(reversed_token_scatter_idx_copy, reversed_token_scatter_idx,
                                nnodes * local_world_size * max_tokens * topk * 2, COPY_BLOCK_SIZE)


def get_ag_splits_and_recv_offset_for_dispatch(
    num_tokens,
    send_reqs_for_nodes,
    send_reqs_recv_bufs,
    topk_indices_buf,
    expert_indices_signal_buf,
    local_splits,
    full_splits_buf,
    splits_signal_buf,
    reversed_token_scatter_idx_buf,
    topk,
    local_world_size,
    world_size,
    max_tokens,
    experts_per_rank,
    full_global_scatter_indices=None,
    full_local_scatter_indices=None,
    cpu_default_val=-1,
    offset_dtype=torch.int32,
    num_sm=20,
    need_reversed_token_scatter_idx=False,
    need_non_drop_token_count_buf=False,
):
    nnodes = world_size // local_world_size
    num_recv_tokens_per_rank_cpu = torch.empty((world_size, ), dtype=torch.int32, device="cpu", pin_memory=True)
    num_recv_tokens_per_rank_cpu.fill_(cpu_default_val)
    # gpu tensor
    num_input_tokens_per_rank = torch.empty((world_size, ), dtype=torch.int32, device=torch.cuda.current_device())
    cumsum_recv_tokens_per_rank = torch.empty((world_size, ), dtype=torch.int32, device=torch.cuda.current_device())
    cumsum_input_tokens_per_rank = torch.empty((world_size, ), dtype=torch.int32, device=torch.cuda.current_device())
    cumsum_full_splits_buf = torch.empty_like(full_splits_buf)
    topk_indices_buf_copy = torch.zeros(topk_indices_buf.size(), dtype=topk_indices_buf.dtype,
                                        device=torch.cuda.current_device())
    if need_non_drop_token_count_buf:
        non_drop_token_count_buf = torch.zeros((
            nnodes,
            max_tokens,
        ), dtype=torch.int32, device=torch.cuda.current_device())
    else:
        non_drop_token_count_buf = None
    send_reqs_for_nodes_copy = torch.zeros(send_reqs_for_nodes.size(), dtype=send_reqs_for_nodes.dtype,
                                           device=torch.cuda.current_device())
    send_reqs_recv_bufs_copy = torch.zeros(send_reqs_recv_bufs.size(), dtype=send_reqs_recv_bufs.dtype,
                                           device=torch.cuda.current_device())

    token_dst_scatter_idx = None
    if full_global_scatter_indices is not None:
        assert len(full_global_scatter_indices.shape) == 2  # [num_total_tokens, topk]
        assert full_global_scatter_indices.dtype == offset_dtype
        token_dst_scatter_idx = torch.empty((nnodes, max_tokens, topk), dtype=full_global_scatter_indices.dtype,
                                            device=full_global_scatter_indices.device)
        if need_reversed_token_scatter_idx:
            reversed_token_scatter_idx_copy = torch.empty((world_size * max_tokens * topk, 2),
                                                          dtype=full_global_scatter_indices.dtype,
                                                          device=full_global_scatter_indices.device)
        else:
            reversed_token_scatter_idx_copy = None
        token_sort_indices = None
    elif full_local_scatter_indices is not None:
        assert len(full_local_scatter_indices.shape) == 3  # [nnodes, max_tokens, topk]
        assert full_local_scatter_indices.dtype == offset_dtype
        token_dst_scatter_idx = torch.empty((nnodes, max_tokens, topk), dtype=full_local_scatter_indices.dtype,
                                            device=full_local_scatter_indices.device)
        if need_reversed_token_scatter_idx:
            reversed_token_scatter_idx_copy = torch.empty((world_size * max_tokens * topk, 2),
                                                          dtype=full_local_scatter_indices.dtype,
                                                          device=full_local_scatter_indices.device)
        else:
            reversed_token_scatter_idx_copy = None
        token_sort_indices = torch.empty((nnodes, max_tokens * topk), dtype=offset_dtype,
                                         device=full_local_scatter_indices.device)
        token_sort_indices.fill_(-1)
    else:
        token_dst_scatter_idx = None
        reversed_token_scatter_idx_copy = None
        token_sort_indices = None
    """
    recv_buf_offset_per_expert:
        recv_buf_offset_per_expert[i, j, k] represents the starting offset in the output of all tokens sent by `rank k` to `expert j` on `rank i`.
        This ensures that the tokens sent from all ranks to each expert are continuous in the output,
        which meet the layout requirements of group gemm and avoid post-processing.
    """
    send_buf_offset_per_expert = torch.zeros((world_size, experts_per_rank, world_size), dtype=offset_dtype,
                                             device=torch.cuda.current_device())
    recv_buf_offset_per_expert = torch.zeros((world_size, experts_per_rank, world_size), dtype=offset_dtype,
                                             device=torch.cuda.current_device())
    recv_buf_tokens_per_expert = torch.zeros((world_size, experts_per_rank), dtype=offset_dtype,
                                             device=torch.cuda.current_device())
    grid = (num_sm, )
    num_grid_sync = world_size
    counter_workspace = torch.zeros((num_grid_sync, ), dtype=torch.int32, device=torch.cuda.current_device())
    assert splits_signal_buf.dtype == NVSHMEM_SIGNAL_DTYPE
    assert len(full_splits_buf.shape) == 2 and full_splits_buf.shape[1] == local_splits.shape[0]
    assert full_splits_buf.shape[0] == world_size

    BLOCK_SIZE = 1 << (full_splits_buf.shape[1]).bit_length()  # the extra one is for drop token
    assert BLOCK_SIZE >= (full_splits_buf.shape[1])
    # num_tokens = exp_indices.shape[0]

    kernel_get_ag_splits_and_recv_offset[grid](
        num_tokens,
        send_reqs_for_nodes,
        send_reqs_for_nodes_copy,
        send_reqs_recv_bufs,
        send_reqs_recv_bufs_copy,
        token_sort_indices,
        topk_indices_buf,
        topk_indices_buf_copy,
        non_drop_token_count_buf,
        expert_indices_signal_buf,
        local_splits,
        full_splits_buf,
        cumsum_full_splits_buf,
        splits_signal_buf,
        num_input_tokens_per_rank,
        cumsum_input_tokens_per_rank,
        num_recv_tokens_per_rank_cpu,
        cumsum_recv_tokens_per_rank,
        send_buf_offset_per_expert,
        recv_buf_offset_per_expert,
        recv_buf_tokens_per_expert,
        counter_workspace,
        full_global_scatter_indices,
        full_local_scatter_indices,
        token_dst_scatter_idx,
        reversed_token_scatter_idx_buf,
        reversed_token_scatter_idx_copy,
        full_splits_buf.shape[1],
        local_world_size,
        max_tokens,
        experts_per_rank,
        topk,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )
    return (
        recv_buf_offset_per_expert,
        recv_buf_tokens_per_expert,
        num_recv_tokens_per_rank_cpu,
        num_input_tokens_per_rank,
        token_dst_scatter_idx,
        reversed_token_scatter_idx_copy,
        send_reqs_for_nodes_copy,
        send_reqs_recv_bufs_copy,
        topk_indices_buf_copy,
        non_drop_token_count_buf,
        token_sort_indices,
    )
