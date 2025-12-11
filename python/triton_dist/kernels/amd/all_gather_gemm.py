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

import torch.distributed
import triton
import triton.language as tl
import triton_dist.language as dl
import triton_dist.tune
from triton.runtime.driver import driver

import triton_dist
from triton_dist.language.extra.language_extra import st
from hip import hip
from triton_dist.utils import HIP_CHECK, launch_cooperative_grid_options
from typing import Optional, List
import pyrocshmem
from triton_dist.kernels.amd.common_ops import barrier_all_ipc_kernel_v2, barrier_on_this_grid


def _get_default_num_xcds():
    NUM_XCDS = 8 if torch.cuda.get_device_properties(0).multi_processor_count > 100 else 4
    return NUM_XCDS


@triton.jit(do_not_specialize=["rank"])
def copy_kernel_2d(
    src_tensor,
    dst_tensors_ptrs,
    barrier_ptrs,
    chunk_counters_ptr,
    rank,
    num_ranks,
    M,
    N,
    M_PER_CHUNK,
    stride_src_m,
    stride_src_n,
    stride_dst_m,
    stride_dst_n,
    dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_rank = tl.program_id(axis=0)  # Which target rank this block handles
    pid_block = tl.program_id(axis=1)  # Which block within that rank's data

    if pid_rank >= num_ranks:
        return

    target_rank = pid_rank
    if target_rank >= rank:
        target_rank += 1

    dst_tensor = tl.load(dst_tensors_ptrs + target_rank).to(tl.pointer_type(dtype))

    blocks_per_rank_m = tl.cdiv(M, BLOCK_SIZE_M)
    blocks_per_rank_n = tl.cdiv(N, BLOCK_SIZE_N)
    blocks_per_rank = blocks_per_rank_m * blocks_per_rank_n

    NUM_CHUNKS_PER_RANK_M = tl.cdiv(M, M_PER_CHUNK)
    BLOCKS_PER_CHUNK_M = tl.cdiv(M_PER_CHUNK, BLOCK_SIZE_M)

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    num_blocks_y = tl.num_programs(axis=1)

    prev_global_chunk_idx = tl.full((1, ), -1, dtype=tl.int32)[0]
    local_count = 0

    for block_id in range(pid_block, blocks_per_rank, num_blocks_y):
        block_m = block_id // blocks_per_rank_n
        block_n = block_id % blocks_per_rank_n

        src_m_start = block_m * BLOCK_SIZE_M
        src_n_start = block_n * BLOCK_SIZE_N

        dst_m_start = rank * M + src_m_start
        dst_n_start = src_n_start

        src_ptrs = src_tensor + (src_m_start + offs_m[:, None]) * stride_src_m + \
                   (src_n_start + offs_n[None, :]) * stride_src_n
        dst_ptrs = dst_tensor + (dst_m_start + offs_m[:, None]) * stride_dst_m + \
                   (dst_n_start + offs_n[None, :]) * stride_dst_n

        mask = (src_m_start + offs_m[:, None] < M) & (src_n_start + offs_n[None, :] < N)

        data = tl.load(src_ptrs, mask=mask)
        tl.store(dst_ptrs, data, mask=mask)

        chunk_idx_in_rank_m = block_m // BLOCKS_PER_CHUNK_M
        global_chunk_idx = target_rank * NUM_CHUNKS_PER_RANK_M + chunk_idx_in_rank_m

        if global_chunk_idx == prev_global_chunk_idx:
            local_count += 1
        else:
            if prev_global_chunk_idx != -1:
                # Calculate actual blocks for this chunk (handling non-perfect tiling)
                prev_chunk_idx_in_rank = prev_global_chunk_idx % NUM_CHUNKS_PER_RANK_M

                # Check if this is the last chunk
                if prev_chunk_idx_in_rank == NUM_CHUNKS_PER_RANK_M - 1:
                    # Last chunk: calculate actual blocks
                    prev_chunk_start_m = prev_chunk_idx_in_rank * M_PER_CHUNK
                    prev_chunk_end_m = tl.minimum(prev_chunk_start_m + M_PER_CHUNK, M)
                    prev_chunk_size_m = prev_chunk_end_m - prev_chunk_start_m
                    prev_chunk_blocks_m = tl.cdiv(prev_chunk_size_m, BLOCK_SIZE_M)
                else:
                    # Non-last chunks: use standard block count
                    prev_chunk_blocks_m = BLOCKS_PER_CHUNK_M

                prev_chunk_total_blocks = prev_chunk_blocks_m * blocks_per_rank_n

                old = tl.atomic_add(chunk_counters_ptr + prev_global_chunk_idx, local_count)
                if old + local_count == prev_chunk_total_blocks:
                    target_barrier_ptr = tl.load(barrier_ptrs + target_rank).to(tl.pointer_type(tl.int32))
                    st(
                        target_barrier_ptr + rank * NUM_CHUNKS_PER_RANK_M +
                        (prev_global_chunk_idx - target_rank * NUM_CHUNKS_PER_RANK_M), 1, semantic="release",
                        scope="system")
            prev_global_chunk_idx = global_chunk_idx
            local_count = 1

    if local_count > 0 and prev_global_chunk_idx != -1:
        # Calculate actual blocks for the last chunk (handling non-perfect tiling)
        last_chunk_idx_in_rank = prev_global_chunk_idx % NUM_CHUNKS_PER_RANK_M

        # Check if this is the last chunk
        if last_chunk_idx_in_rank == NUM_CHUNKS_PER_RANK_M - 1:
            # Last chunk: calculate actual blocks
            last_chunk_start_m = last_chunk_idx_in_rank * M_PER_CHUNK
            last_chunk_end_m = tl.minimum(last_chunk_start_m + M_PER_CHUNK, M)
            last_chunk_size_m = last_chunk_end_m - last_chunk_start_m
            last_chunk_blocks_m = tl.cdiv(last_chunk_size_m, BLOCK_SIZE_M)
        else:
            # Non-last chunks: use standard block count
            last_chunk_blocks_m = BLOCKS_PER_CHUNK_M

        last_chunk_total_blocks = last_chunk_blocks_m * blocks_per_rank_n

        old = tl.atomic_add(chunk_counters_ptr + prev_global_chunk_idx, local_count)
        if old + local_count == last_chunk_total_blocks:
            target_barrier_ptr = tl.load(barrier_ptrs + target_rank).to(tl.pointer_type(tl.int32))
            st(
                target_barrier_ptr + rank * NUM_CHUNKS_PER_RANK_M +
                (prev_global_chunk_idx - target_rank * NUM_CHUNKS_PER_RANK_M), 1, semantic="release", scope="system")


@triton.jit(do_not_specialize=["rank"])
def copy_kernel_2d_pull(
    src_tensor,
    dst_tensors_ptrs,
    barrier_ptrs,
    chunk_counters_ptr,
    rank,
    num_ranks,
    M,
    N,
    M_PER_CHUNK,
    stride_src_m,
    stride_src_n,
    stride_dst_m,
    stride_dst_n,
    dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_rank = tl.program_id(axis=0)  # Which target rank this block handles
    pid_block = tl.program_id(axis=1)  # Which block within that rank's data

    if pid_rank >= num_ranks:
        return

    src_rank = pid_rank
    if src_rank >= rank:
        src_rank += 1

    src_tensor = tl.load(dst_tensors_ptrs + src_rank).to(tl.pointer_type(dtype))
    dst_tensor = tl.load(dst_tensors_ptrs + rank).to(tl.pointer_type(dtype))

    blocks_per_rank_m = tl.cdiv(M, BLOCK_SIZE_M)
    blocks_per_rank_n = tl.cdiv(N, BLOCK_SIZE_N)
    blocks_per_rank = blocks_per_rank_m * blocks_per_rank_n

    NUM_CHUNKS_PER_RANK_M = tl.cdiv(M, M_PER_CHUNK)
    BLOCKS_PER_CHUNK_M = tl.cdiv(M_PER_CHUNK, BLOCK_SIZE_M)

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    num_blocks_y = tl.num_programs(axis=1)

    prev_global_chunk_idx = tl.full((1, ), -1, dtype=tl.int32)[0]
    local_count = 0

    for block_id in range(pid_block, blocks_per_rank, num_blocks_y):
        block_m = block_id // blocks_per_rank_n
        block_n = block_id % blocks_per_rank_n

        src_m_start = block_m * BLOCK_SIZE_M
        src_n_start = block_n * BLOCK_SIZE_N

        dst_m_start = src_rank * M + src_m_start
        dst_n_start = src_n_start

        src_ptrs = src_tensor + (dst_m_start + offs_m[:, None]) * stride_dst_m + \
                   (dst_n_start + offs_n[None, :]) * stride_dst_n
        dst_ptrs = dst_tensor + (dst_m_start + offs_m[:, None]) * stride_dst_m + \
                   (dst_n_start + offs_n[None, :]) * stride_dst_n

        mask = (src_m_start + offs_m[:, None] < M) & (src_n_start + offs_n[None, :] < N)

        data = tl.load(src_ptrs, mask=mask)
        tl.store(dst_ptrs, data, mask=mask)

        chunk_idx_in_rank_m = block_m // BLOCKS_PER_CHUNK_M
        global_chunk_idx = src_rank * NUM_CHUNKS_PER_RANK_M + chunk_idx_in_rank_m

        if global_chunk_idx == prev_global_chunk_idx:
            local_count += 1
        else:
            if prev_global_chunk_idx != -1:
                prev_chunk_idx_in_rank = prev_global_chunk_idx % NUM_CHUNKS_PER_RANK_M

                # Check if this is the last chunk
                if prev_chunk_idx_in_rank == NUM_CHUNKS_PER_RANK_M - 1:
                    # Last chunk: calculate actual blocks
                    prev_chunk_start_m = prev_chunk_idx_in_rank * M_PER_CHUNK
                    prev_chunk_end_m = tl.minimum(prev_chunk_start_m + M_PER_CHUNK, M)
                    prev_chunk_size_m = prev_chunk_end_m - prev_chunk_start_m
                    prev_chunk_blocks_m = tl.cdiv(prev_chunk_size_m, BLOCK_SIZE_M)
                else:
                    # Non-last chunks: use standard block count
                    prev_chunk_blocks_m = BLOCKS_PER_CHUNK_M

                prev_chunk_total_blocks = prev_chunk_blocks_m * blocks_per_rank_n

                old = tl.atomic_add(chunk_counters_ptr + prev_global_chunk_idx, local_count)
                if old + local_count == prev_chunk_total_blocks:
                    target_barrier_ptr = tl.load(barrier_ptrs + rank).to(tl.pointer_type(tl.int32))
                    chunk_idx_in_src_rank_m = prev_global_chunk_idx - src_rank * NUM_CHUNKS_PER_RANK_M
                    signal_ptr = target_barrier_ptr + src_rank * NUM_CHUNKS_PER_RANK_M + chunk_idx_in_src_rank_m
                    st(signal_ptr, 1, semantic="release", scope="system")
            prev_global_chunk_idx = global_chunk_idx
            local_count = 1

    if local_count > 0 and prev_global_chunk_idx != -1:
        # Calculate actual blocks for the last chunk (handling non-perfect tiling)
        last_chunk_idx_in_rank = prev_global_chunk_idx % NUM_CHUNKS_PER_RANK_M

        # Check if this is the last chunk
        if last_chunk_idx_in_rank == NUM_CHUNKS_PER_RANK_M - 1:
            # Last chunk: calculate actual blocks
            last_chunk_start_m = last_chunk_idx_in_rank * M_PER_CHUNK
            last_chunk_end_m = tl.minimum(last_chunk_start_m + M_PER_CHUNK, M)
            last_chunk_size_m = last_chunk_end_m - last_chunk_start_m
            last_chunk_blocks_m = tl.cdiv(last_chunk_size_m, BLOCK_SIZE_M)
        else:
            # Non-last chunks: use standard block count
            last_chunk_blocks_m = BLOCKS_PER_CHUNK_M

        last_chunk_total_blocks = last_chunk_blocks_m * blocks_per_rank_n

        old = tl.atomic_add(chunk_counters_ptr + prev_global_chunk_idx, local_count)
        if old + local_count == last_chunk_total_blocks:
            target_barrier_ptr = tl.load(barrier_ptrs + rank).to(tl.pointer_type(tl.int32))
            chunk_idx_in_src_rank_m = prev_global_chunk_idx - src_rank * NUM_CHUNKS_PER_RANK_M
            signal_ptr = target_barrier_ptr + src_rank * NUM_CHUNKS_PER_RANK_M + chunk_idx_in_src_rank_m
            st(signal_ptr, 1, semantic="release", scope="system")


def cp_engine_producer_all_gather_full_mesh_push_multi_stream(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    one: torch.Tensor,
    M_PER_CHUNK: int,
    ag_stream_pool: List[torch.cuda.Stream],
    barrier_buffers: List[torch.Tensor],
):
    M_per_rank, N = local_tensor.shape
    chunk_num_per_rank = M_per_rank // M_PER_CHUNK
    last_chunk_size = M_per_rank % M_PER_CHUNK
    total_chunks_per_rank = chunk_num_per_rank + (1 if last_chunk_size > 0 else 0)
    num_stream = len(ag_stream_pool)
    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    stream_offset = 0
    data_elem_size = local_tensor.element_size()
    barrier_elem_size = one.element_size()

    for idx, remote_rank in enumerate(rank_orders):
        if remote_rank == rank:
            stream_offset += total_chunks_per_rank
            continue
        for chunk_idx_intra_rank in range(total_chunks_per_rank):
            stream_offset += 1
            chunk_pos = rank * total_chunks_per_rank + chunk_idx_intra_rank
            stream_pos = idx % num_stream
            ag_stream = ag_stream_pool[stream_pos]
            M_src_start_pos = chunk_idx_intra_rank * M_PER_CHUNK
            M_dst_start_pos = rank * M_per_rank + M_src_start_pos
            # M_dst_end_pos = M_dst_start_pos + M_PER_CHUNK
            # dst = remote_tensor_buffers[remote_rank][M_dst_start_pos:M_dst_end_pos, :]
            #  The data pointer is used directly here to reduce the overhead of slice operation (which may cause GPU bubbles in small shapes)
            # M_src_end_pos = M_src_start_pos + M_PER_CHUNK
            # src = local_tensor[M_src_start_pos:M_src_end_pos, :]
            chunk_size = min(M_PER_CHUNK, M_per_rank - M_src_start_pos)
            src_ptr = local_tensor.data_ptr() + M_src_start_pos * N * data_elem_size
            dst_ptr = remote_tensor_buffers[remote_rank].data_ptr() + M_dst_start_pos * N * data_elem_size
            nbytes = chunk_size * N * data_elem_size
            cp_res = hip.hipMemcpyAsync(
                dst_ptr,
                src_ptr,
                nbytes,
                hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                ag_stream.cuda_stream,
            )
            HIP_CHECK(cp_res)
            """
                Why use memcpy to set signal:
                    Because driver API(waitValue/writeValue) on AMD will affect the perf of gemm. Memcpy also takes less than 5us.
            """
            # set_signal(barrier_buffers[remote_rank][chunk_pos].data_ptr(), 1, ag_stream)
            cp_res = hip.hipMemcpyAsync(
                barrier_buffers[remote_rank].data_ptr() + chunk_pos * barrier_elem_size,
                one.data_ptr(),
                barrier_elem_size,
                hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                ag_stream.cuda_stream,
            )
            HIP_CHECK(cp_res)


def copy_kernel_producer_all_gather(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    one: torch.Tensor,
    M_PER_CHUNK: int,
    copy_stream: torch.cuda.Stream,
    barrier_ptrs: torch.Tensor,
    dst_tensor_ptrs: torch.Tensor,
    chunk_counters: torch.Tensor,
    comm_sms: int = 0,
    BLOCK_SIZE_M: int = 128,
    BLOCK_SIZE_N: int = 256,
):
    M_per_rank, N = local_tensor.shape
    assert comm_sms % (num_ranks - 1) == 0

    # 2D grid: (num_ranks - 1, comm_sms / (num_ranks - 1))
    grid = (num_ranks - 1, comm_sms // (num_ranks - 1))

    with torch.cuda.stream(copy_stream):
        copy_kernel_2d[grid](
            local_tensor,
            dst_tensor_ptrs,
            barrier_ptrs,
            chunk_counters,
            rank,
            num_ranks,
            M_per_rank,
            N,
            M_PER_CHUNK,
            local_tensor.stride(0),
            local_tensor.stride(1),
            remote_tensor_buffers[0].stride(0),
            remote_tensor_buffers[0].stride(1),
            dtype=tl.float16 if local_tensor.dtype == torch.float16 else tl.bfloat16,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )


def matmul_get_configs():
    configs = []
    for BM in [128, 256]:
        for BN in [128, 256]:
            for BK in [64, 128]:
                for WAVES in [0, 2]:
                    kwargs = {'num_stages': 2, 'num_warps': 8}
                    first_arg = {
                        'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 1, 'waves_per_eu':
                        WAVES, 'matrix_instr_nonkdim': 16, 'kpack': 1, "NUM_XCDS": _get_default_num_xcds()
                    }
                    configs.append(triton.Config(first_arg, **kwargs))
    return configs


@triton_dist.jit(do_not_specialize=["rank"])
def local_copy_and_barrier_all_ipc_kernel(
    rank,
    local_buf_ptr,
    global_buf_ptr,
    M_per_rank,
    N,
    stride_local_m,
    stride_local_n,
    stride_global_m,
    stride_global_n,
    num_ranks,
    comm_buf_base_ptrs,
    barrier_ptr,
    chunk_counters_ptr,
    sync_grid_buf_ptr,
    m_chunk_num: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    sm_id = tl.program_id(axis=0)
    num_sms = tl.num_programs(axis=0)

    offs_m = tl.max_contiguous(tl.arange(0, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.arange(0, BLOCK_SIZE_N), BLOCK_SIZE_N)
    num_iters_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    num_iters_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_iters = num_iters_m * num_iters_n

    for i in range(sm_id, num_iters, num_sms):
        pid_m = i // num_iters_n
        pid_n = i % num_iters_n
        data_ptr = local_buf_ptr + (pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_local_m + (
            pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_local_n
        dst_ptr = global_buf_ptr + (rank * M_per_rank + pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_global_m + (
            pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_global_n
        mask_data = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (pid_n * BLOCK_SIZE_N + offs_n[None, :] < N)
        mask_dst = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (pid_n * BLOCK_SIZE_N + offs_n[None, :] < N)

        data = tl.load(data_ptr, mask=mask_data)
        tl.store(dst_ptr, data, mask=mask_dst)

    chunks_per_rank = m_chunk_num // num_ranks
    for i in range(sm_id, m_chunk_num, num_sms):
        chunk_rank = i // chunks_per_rank
        if chunk_rank == rank:
            tl.store(barrier_ptr + i, 1)
        else:
            tl.store(barrier_ptr + i, 0)

    for i in range(sm_id, num_chunks, num_sms):
        tl.store(chunk_counters_ptr + i, 0)

    barrier_on_this_grid(sync_grid_buf_ptr, use_cooperative=False)
    if sm_id == 0:
        barrier_all_ipc_kernel_v2(rank, num_ranks, comm_buf_base_ptrs)
        # barrier_all_ipc_kernel(rank, num_ranks, comm_buf_base_ptrs)


@triton.jit
def row_to_chunk_idx(row, M_PER_CHUNK, M_per_rank, chunks_per_rank):
    rank = row // M_per_rank
    offset = row - rank * M_per_rank
    chunk_in_rank = offset // M_PER_CHUNK
    chunk_idx = rank * chunks_per_rank + chunk_in_rank
    return chunk_idx


@triton.jit
def swizzle_ag_gemm_imperfect(original_pid_m, M, rank, world_size, CHUNK_SIZE_M: tl.constexpr,
                              BLOCK_SIZE_M: tl.constexpr):
    M_per_rank = M // world_size

    if CHUNK_SIZE_M < BLOCK_SIZE_M:
        effective_chunk_size = BLOCK_SIZE_M
    elif CHUNK_SIZE_M % BLOCK_SIZE_M != 0:
        # Round DOWN to nearest multiple of BLOCK_SIZE_M
        blocks_per_chunk = CHUNK_SIZE_M // BLOCK_SIZE_M  # Integer division rounds down
        effective_chunk_size = blocks_per_chunk * BLOCK_SIZE_M
    else:
        effective_chunk_size = CHUNK_SIZE_M

    m_addr = original_pid_m * BLOCK_SIZE_M
    group_id = m_addr // effective_chunk_size

    start_addr_of_group = group_id * effective_chunk_size
    first_pid_of_group = (start_addr_of_group + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    id_within_group = original_pid_m - first_pid_of_group

    nchunk_per_rank_min = M_per_rank // effective_chunk_size
    nchunk_per_rank_max = (M_per_rank + effective_chunk_size - 1) // effective_chunk_size

    if nchunk_per_rank_min != nchunk_per_rank_max and \
       (group_id + world_size - 1) // world_size >= nchunk_per_rank_max:
        last_trial = group_id - (nchunk_per_rank_max - 1) * world_size
        m_rem = M_per_rank % effective_chunk_size
        if m_rem == 0:
            rank_in_group_dist = group_id % world_size
        else:
            rank_in_group_dist = (last_trial * effective_chunk_size) // m_rem
        chunk_in_group_dist = nchunk_per_rank_max - 1
    else:
        rank_in_group_dist = group_id % world_size
        chunk_in_group_dist = group_id // world_size

    base_group_offset = 0
    groups_to_shift = 0
    for r in range(world_size):
        m_rank_start = r * M_per_rank
        m_rank_end = (r + 1) * M_per_rank
        chunk_id_start = m_rank_start // effective_chunk_size
        chunk_id_end = (m_rank_end - 1) // effective_chunk_size
        if r > 0:
            if (m_rank_start - 1) // effective_chunk_size == chunk_id_start:
                chunk_id_start += 1
        nchunks = chunk_id_end - chunk_id_start + 1
        if nchunks < 0:
            nchunks = 0
        if r < rank_in_group_dist:
            base_group_offset += nchunks
        if r < rank:
            groups_to_shift += nchunks

    reordered_group_id = base_group_offset + chunk_in_group_dist
    total_chunks = (M + effective_chunk_size - 1) // effective_chunk_size
    final_swizzled_group_id = (reordered_group_id + groups_to_shift) % total_chunks
    start_addr_of_new_group = final_swizzled_group_id * effective_chunk_size
    first_pid_of_new_group = (start_addr_of_new_group + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    swizzled_pid_m = first_pid_of_new_group + id_within_group
    return swizzled_pid_m


@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0})
@triton.jit(do_not_specialize=["rank"])
def kernel_consumer_gemm_persistent(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                                    rank, world_size: tl.constexpr, barrier_ptr, BLOCK_SIZE_M: tl.constexpr,
                                    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                                    M_PER_CHUNK: tl.constexpr, NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr,
                                    EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    M_per_rank = M // world_size
    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    chunks_per_rank = tl.cdiv(M_per_rank, M_PER_CHUNK)
    for tile_id in range(pid, total_tiles, NUM_SMS):

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # Swizzle
        pid_m = swizzle_ag_gemm_imperfect(pid_m, M, rank, world_size, M_PER_CHUNK, BLOCK_SIZE_M)
        offs_am = pid_m * BLOCK_SIZE_M
        block_m_start = offs_am
        block_m_end = min(offs_am + BLOCK_SIZE_M, M)
        start_rank = block_m_start // M_per_rank
        end_rank = (block_m_end - 1) // M_per_rank

        if start_rank != rank or end_rank != rank:
            start_chunk_idx = row_to_chunk_idx(block_m_start, M_PER_CHUNK, M_per_rank, chunks_per_rank)
            end_chunk_idx = row_to_chunk_idx(block_m_end - 1, M_PER_CHUNK, M_per_rank, chunks_per_rank)
            signal_count = end_chunk_idx - start_chunk_idx + 1
            token = dl.wait(barrier_ptr + start_chunk_idx, signal_count, "sys", "relaxed", waitValue=1)
            A = dl.consume_token(A, token)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        tl.assume(pid_m > 0)
        tl.assume(pid_n > 0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)


def key_fn(A: torch.Tensor, B: torch.Tensor, *args, **kwargs):
    return triton_dist.tune.to_hashable(A), triton_dist.tune.to_hashable(B)


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


@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0})
@triton.jit(do_not_specialize=["rank"])
def kernel_fused_ag_gemm(A, localA,  # Local tensor for this rank [M_per_rank, K]
                         B,  # Weight tensor [N, K]
                         C,  # Output tensor [M, N]
                         dst_tensors_ptrs,  # Pointers to all workspace tensors
                         barrier_ptrs,  # Barrier pointers for synchronization
                         barrier_ptr, chunk_counters_ptr,  # Atomic counters for chunk completion
                         rank, world_size, M,  # Total M dimension (M_per_rank * world_size)
                         N, K, stride_local_am, stride_local_ak, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                         stride_cn, CP_BLOCK_SIZE_M: tl.constexpr, CP_BLOCK_SIZE_K: tl.constexpr,
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                         GROUP_SIZE_M: tl.constexpr, M_PER_CHUNK: tl.constexpr, NUM_COMM_SMS: tl.constexpr,
                         NUM_GEMM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr, EVEN_K: tl.constexpr, dtype: tl.constexpr):
    """Fused all-gather and GEMM kernel - SMs split between communication and computation"""
    global_pid = tl.program_id(0)
    M_per_rank = M // world_size
    if global_pid < NUM_COMM_SMS:
        comm_pid = global_pid

        pid_rank = comm_pid % (world_size - 1)  # Which target rank this block handles
        pid_block = comm_pid // (world_size - 1)  # Which block within that rank's data

        if pid_rank >= world_size:
            return

        target_rank = pid_rank
        if target_rank >= rank:
            target_rank += 1

        dst_tensor = tl.load(dst_tensors_ptrs + target_rank).to(tl.pointer_type(dtype))

        blocks_per_rank_m = tl.cdiv(M_per_rank, CP_BLOCK_SIZE_M)
        blocks_per_rank_k = tl.cdiv(K, CP_BLOCK_SIZE_K)
        blocks_per_rank = blocks_per_rank_m * blocks_per_rank_k

        NUM_CHUNKS_PER_RANK_M = tl.cdiv(M_per_rank, M_PER_CHUNK)
        BLOCKS_PER_CHUNK_M = tl.cdiv(M_PER_CHUNK, CP_BLOCK_SIZE_M)

        offs_m = tl.arange(0, CP_BLOCK_SIZE_M)
        offs_k = tl.arange(0, CP_BLOCK_SIZE_K)

        num_blocks_y = NUM_COMM_SMS // (world_size - 1)

        prev_global_chunk_idx = tl.full((1, ), -1, dtype=tl.int32)[0]
        local_count = 0

        for block_id in range(pid_block, blocks_per_rank, num_blocks_y):
            block_m = block_id // blocks_per_rank_k
            block_k = block_id % blocks_per_rank_k

            src_m_start = block_m * CP_BLOCK_SIZE_M
            src_k_start = block_k * CP_BLOCK_SIZE_K

            dst_m_start = rank * M_per_rank + src_m_start
            dst_k_start = src_k_start

            src_ptrs = localA + (src_m_start + offs_m[:, None]) * stride_local_am + \
                    (src_k_start + offs_k[None, :]) * stride_local_ak
            dst_ptrs = dst_tensor + (dst_m_start + offs_m[:, None]) * stride_am + \
                    (dst_k_start + offs_k[None, :]) * stride_ak

            mask = (src_m_start + offs_m[:, None] < M_per_rank) & (src_k_start + offs_k[None, :] < K)

            data = tl.load(src_ptrs, mask=mask)
            tl.store(dst_ptrs, data, mask=mask)
            chunk_idx_in_rank_m = block_m // BLOCKS_PER_CHUNK_M
            global_chunk_idx = target_rank * NUM_CHUNKS_PER_RANK_M + chunk_idx_in_rank_m

            if global_chunk_idx == prev_global_chunk_idx:
                local_count += 1
            else:
                if prev_global_chunk_idx != -1:
                    # Calculate actual blocks for this chunk (handling non-perfect tiling)
                    prev_chunk_idx_in_rank = prev_global_chunk_idx % NUM_CHUNKS_PER_RANK_M

                    # Check if this is the last chunk
                    if prev_chunk_idx_in_rank == NUM_CHUNKS_PER_RANK_M - 1:
                        # Last chunk: calculate actual blocks
                        prev_chunk_start_m = prev_chunk_idx_in_rank * M_PER_CHUNK
                        prev_chunk_end_m = tl.minimum(prev_chunk_start_m + M_PER_CHUNK, M_per_rank)
                        prev_chunk_size_m = prev_chunk_end_m - prev_chunk_start_m
                        prev_chunk_blocks_m = tl.cdiv(prev_chunk_size_m, CP_BLOCK_SIZE_M)
                    else:
                        # Non-last chunks: use standard block count
                        prev_chunk_blocks_m = BLOCKS_PER_CHUNK_M

                    prev_chunk_total_blocks = prev_chunk_blocks_m * blocks_per_rank_k

                    old = tl.atomic_add(chunk_counters_ptr + prev_global_chunk_idx, local_count)
                    if old + local_count == prev_chunk_total_blocks:
                        target_barrier_ptr = tl.load(barrier_ptrs + target_rank).to(tl.pointer_type(tl.int32))
                        st(
                            target_barrier_ptr + rank * NUM_CHUNKS_PER_RANK_M +
                            (prev_global_chunk_idx - target_rank * NUM_CHUNKS_PER_RANK_M), 1, semantic="release",
                            scope="system")
                prev_global_chunk_idx = global_chunk_idx
                local_count = 1

        if local_count > 0 and prev_global_chunk_idx != -1:
            # Calculate actual blocks for the last chunk (handling non-perfect tiling)
            last_chunk_idx_in_rank = prev_global_chunk_idx % NUM_CHUNKS_PER_RANK_M

            # Check if this is the last chunk
            if last_chunk_idx_in_rank == NUM_CHUNKS_PER_RANK_M - 1:
                # Last chunk: calculate actual blocks
                last_chunk_start_m = last_chunk_idx_in_rank * M_PER_CHUNK
                last_chunk_end_m = tl.minimum(last_chunk_start_m + M_PER_CHUNK, M_per_rank)
                last_chunk_size_m = last_chunk_end_m - last_chunk_start_m
                last_chunk_blocks_m = tl.cdiv(last_chunk_size_m, CP_BLOCK_SIZE_M)
            else:
                # Non-last chunks: use standard block count
                last_chunk_blocks_m = BLOCKS_PER_CHUNK_M

            last_chunk_total_blocks = last_chunk_blocks_m * blocks_per_rank_k

            old = tl.atomic_add(chunk_counters_ptr + prev_global_chunk_idx, local_count)
            if old + local_count == last_chunk_total_blocks:
                target_barrier_ptr = tl.load(barrier_ptrs + target_rank).to(tl.pointer_type(tl.int32))
                st(
                    target_barrier_ptr + rank * NUM_CHUNKS_PER_RANK_M +
                    (prev_global_chunk_idx - target_rank * NUM_CHUNKS_PER_RANK_M), 1, semantic="release",
                    scope="system")

    else:
        gemm_pid = global_pid - NUM_COMM_SMS

        if NUM_XCDS != 1:
            gemm_pid = (gemm_pid % NUM_XCDS) * (NUM_GEMM_SMS // NUM_XCDS) + (gemm_pid // NUM_XCDS)

        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        total_tiles = num_pid_m * num_pid_n

        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bn > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_cm > 0)
        tl.assume(stride_cn > 0)

        M_per_rank = M // world_size
        acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
        chunks_per_rank = tl.cdiv(M_per_rank, M_PER_CHUNK)
        for tile_id in range(gemm_pid, total_tiles, NUM_GEMM_SMS):

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # Swizzle
            pid_m = swizzle_ag_gemm_imperfect(pid_m, M, rank, world_size, M_PER_CHUNK, BLOCK_SIZE_M)
            offs_am = pid_m * BLOCK_SIZE_M
            block_m_start = offs_am
            block_m_end = min(offs_am + BLOCK_SIZE_M, M)
            start_rank = block_m_start // M_per_rank
            end_rank = (block_m_end - 1) // M_per_rank

            if start_rank != rank or end_rank != rank:
                start_chunk_idx = row_to_chunk_idx(block_m_start, M_PER_CHUNK, M_per_rank, chunks_per_rank)
                end_chunk_idx = row_to_chunk_idx(block_m_end - 1, M_PER_CHUNK, M_per_rank, chunks_per_rank)
                signal_count = end_chunk_idx - start_chunk_idx + 1
                token = dl.wait(barrier_ptr + start_chunk_idx, signal_count, "sys", "relaxed", waitValue=1)
                A = dl.consume_token(A, token)

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rk = tl.arange(0, BLOCK_SIZE_K)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            tl.assume(pid_m > 0)
            tl.assume(pid_n > 0)

            loop_k = tl.cdiv(K, BLOCK_SIZE_K)
            if not EVEN_K:
                loop_k -= 1

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for k in range(0, loop_k):
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
                acc += tl.dot(a, b)
                A_BASE += BLOCK_SIZE_K * stride_ak
                B_BASE += BLOCK_SIZE_K * stride_bk

            if not EVEN_K:
                k = loop_k
                rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
                B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                A_BASE = tl.multiple_of(A_BASE, (1, 16))
                B_BASE = tl.multiple_of(B_BASE, (16, 1))
                a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
                b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
                acc += tl.dot(a, b)

            c = acc.to(C.type.element_ty)
            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            c_mask = (rm[:, None] < M) & (rn[None, :] < N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            tl.store(C_, c, c_mask)


@dataclasses.dataclass
class AllGatherGEMMTensorParallelContext:
    rank: int
    num_ranks: int
    workspace_tensors: List[torch.Tensor]
    barrier_tensors: List[torch.Tensor]
    one: torch.Tensor
    comm_bufs: List[torch.Tensor]
    comm_buf_ptr: torch.Tensor
    ag_streams: List[torch.cuda.streams.Stream] = None
    M_PER_CHUNK: int = 1024
    use_copy_kernel: bool = False
    comm_sms: int = 0

    dst_tensor_ptrs: torch.Tensor = dataclasses.field(init=False)
    barrier_ptrs: torch.Tensor = dataclasses.field(init=False)
    chunk_counters: torch.Tensor = dataclasses.field(init=False)
    sync_grid_buf: torch.Tensor = dataclasses.field(init=False)

    gemm_stream_torch: Optional[torch.cuda.streams.Stream] = None

    def __post_init__(self):
        M = self.workspace_tensors[0].shape[0]
        self.dst_tensor_ptrs = torch.zeros(self.num_ranks, dtype=torch.int64, device='cuda')
        self.barrier_ptrs = torch.zeros(self.num_ranks, dtype=torch.int64, device='cuda')
        # Calculate total number of chunks across all ranks
        M_per_rank = M // self.num_ranks
        chunks_per_rank = (M_per_rank + self.M_PER_CHUNK - 1) // self.M_PER_CHUNK
        total_chunks = chunks_per_rank * self.num_ranks
        self.chunk_counters = torch.zeros(total_chunks, dtype=torch.int32, device='cuda')
        for i in range(self.num_ranks):
            self.dst_tensor_ptrs[i] = self.workspace_tensors[i].data_ptr()
            self.barrier_ptrs[i] = self.barrier_tensors[i].data_ptr()

        self.sync_grid_buf = torch.zeros((1, ), dtype=torch.uint32, device="cuda")


def create_ag_gemm_intra_node_context(max_M, N, K, input_dtype: torch.dtype, output_dtype: torch.dtype, rank, num_ranks,
                                      tp_group: torch.distributed.ProcessGroup, ag_streams=None, M_PER_CHUNK=1024,
                                      use_copy_kernel=False, comm_sms=0):
    """create context for allgather gemm intra-node

    Args:
        max_M: max number of M shape
        N(int): N
        K(int): K
        input_dtype(torch.dtype): dtype of input
        output_dtype(torch.dtype): dtype of output
        rank (int): current rank
        num_ranks (int): total number of ranks
        ag_streams (torch.cuda.streams.Stream, optional): The stream used for allgather, if not provided, create a new one. Defaults to None.
        serial (bool, optional): Make the execution serialized, for debug. Defaults to False.
        use_copy_kernel (bool, optional): whether to use SM-based copy kernel. Defaults to False.
        comm_sms (int, optional): number of SMs to reserve for comm. Defaults to 0.

    Returns:
        AllGatherGEMMTensorParallelContext
    """
    assert max_M % num_ranks == 0
    M_per_rank = max_M // num_ranks
    dtype = input_dtype
    workspaces = pyrocshmem.rocshmem_create_tensor_list_intra_node([max_M, K], dtype)

    chunks_per_rank = (M_per_rank + M_PER_CHUNK - 1) // M_PER_CHUNK
    barriers = pyrocshmem.rocshmem_create_tensor_list_intra_node([num_ranks * chunks_per_rank], torch.int32)
    barriers[rank].fill_(0)

    comm_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node([num_ranks], torch.int32)
    comm_bufs[rank].fill_(0)
    comm_buf_ptr = torch.tensor([t.data_ptr() for t in comm_bufs], device=torch.cuda.current_device(),
                                requires_grad=False)

    torch.cuda.synchronize()
    torch.distributed.barrier()
    if ag_streams is None:
        if use_copy_kernel:
            gemm_stream_torch = None
            _ag_streams = [torch.cuda.Stream(priority=-1)]
        else:
            # CP engine needs multiple streams for concurrent operations
            _ag_streams = [torch.cuda.Stream(priority=-1) for i in range(num_ranks)]
    else:
        _ag_streams = ag_streams
    one = torch.ones((1024, ), dtype=torch.int32, device=torch.cuda.current_device())

    ret = AllGatherGEMMTensorParallelContext(
        rank=rank,
        num_ranks=num_ranks,
        workspace_tensors=workspaces,
        barrier_tensors=barriers,
        comm_bufs=comm_bufs,
        one=one,
        comm_buf_ptr=comm_buf_ptr,
        ag_streams=_ag_streams,
        gemm_stream_torch=gemm_stream_torch if use_copy_kernel else None,
        M_PER_CHUNK=M_PER_CHUNK,
        use_copy_kernel=use_copy_kernel,
        comm_sms=comm_sms,
    )
    return ret


def fused_ag_gemm_intra_node_op(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                                ctx: AllGatherGEMMTensorParallelContext, gemm_config: triton.Config):
    """Fused all-gather and GEMM operation in a single kernel"""
    assert A.shape[1] == B.shape[1], "Incompatible dimensions"
    assert A.dtype == B.dtype, "Incompatible dtypes"

    M_per_rank, K = A.shape
    M = M_per_rank * ctx.num_ranks
    N_per_rank, _ = B.shape
    NUM_COMM_SMS = ctx.comm_sms
    total_sms = torch.cuda.get_device_properties(0).multi_processor_count
    NUM_GEMM_SMS = total_sms - NUM_COMM_SMS

    grid = (NUM_COMM_SMS + NUM_GEMM_SMS, )
    full_input = ctx.workspace_tensors[ctx.rank][:M]

    kernel_fused_ag_gemm[grid](
        full_input,
        A,  # localA
        B,  # B
        C,  # C
        ctx.dst_tensor_ptrs,  # pointers to all workspace tensors
        ctx.barrier_ptrs,  # barrier pointers
        ctx.barrier_tensors[ctx.rank],  # my barrier pointer
        ctx.chunk_counters,  # chunk counters
        ctx.rank,
        ctx.num_ranks,
        M,
        N_per_rank,
        K,
        A.stride(0),
        A.stride(1),  # local A strides
        full_input.stride(0),
        full_input.stride(1),
        B.stride(1),
        B.stride(0),  # B strides (transposed)
        C.stride(0),
        C.stride(1),  # C strides
        CP_BLOCK_SIZE_M=64,
        CP_BLOCK_SIZE_K=512,
        M_PER_CHUNK=ctx.M_PER_CHUNK,
        NUM_COMM_SMS=NUM_COMM_SMS,
        NUM_GEMM_SMS=NUM_GEMM_SMS,
        dtype=tl.float16 if A.dtype == torch.float16 else tl.bfloat16,
        **gemm_config.all_kwargs(),
    )
    return C


DEFAULT_CONFIG = triton.Config(
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "NUM_XCDS": 8,
        "matrix_instr_nonkdim": 16,
        "waves_per_eu": 0,
    }, num_stages=0, num_warps=8, num_ctas=1)


@triton_dist.tune.autotune(
    config_space=[{"gemm_config": c} for c in matmul_get_configs()],
    key_fn=key_fn,
    prune_fn=prune_fn_by_shared_memory,
)
def ag_gemm_intra_node_op(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, ctx: AllGatherGEMMTensorParallelContext,
                          gemm_config: triton.Config = DEFAULT_CONFIG, use_persistent_gemm=True, use_fused_kernel=False,
                          serial: bool = False):
    # Check constraints.
    assert A.shape[1] == B.shape[1], "Incompatible dimensions"
    assert A.dtype == B.dtype, "Incompatible dtypes"

    M_per_rank, K = A.shape
    M = M_per_rank * ctx.num_ranks
    N_per_rank, _ = B.shape
    current_stream = torch.cuda.current_stream()
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    local_copy_and_barrier_all_ipc_kernel[(num_sms, )](
        ctx.rank, A, ctx.workspace_tensors[ctx.rank], M_per_rank, K, A.stride(0), A.stride(1),
        ctx.workspace_tensors[ctx.rank].stride(0), ctx.workspace_tensors[ctx.rank].stride(1), ctx.num_ranks,
        ctx.comm_buf_ptr, barrier_ptr=ctx.barrier_tensors[ctx.rank], chunk_counters_ptr=ctx.chunk_counters,
        sync_grid_buf_ptr=ctx.sync_grid_buf, m_chunk_num=ctx.barrier_tensors[ctx.rank].shape[0],
        num_chunks=ctx.chunk_counters.shape[0], BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, num_warps=16,
        **launch_cooperative_grid_options())

    if use_fused_kernel:
        # Use the new fused kernel
        return fused_ag_gemm_intra_node_op(A, B, C, ctx, gemm_config)

    # Original implementation with separate kernels
    for ag_stream in ctx.ag_streams:
        ag_stream.wait_stream(current_stream)

    if ctx.gemm_stream_torch is not None:
        ctx.gemm_stream_torch.wait_stream(current_stream)

    def call_ag():
        if ctx.use_copy_kernel:
            # SM copy uses single stream
            copy_kernel_producer_all_gather(ctx.rank, ctx.num_ranks, A, ctx.workspace_tensors, ctx.one, ctx.M_PER_CHUNK,
                                            ctx.ag_streams[0], ctx.barrier_ptrs, ctx.dst_tensor_ptrs,
                                            ctx.chunk_counters, comm_sms=ctx.comm_sms, BLOCK_SIZE_M=64,
                                            BLOCK_SIZE_N=512)
        else:
            # CP engine uses multiple streams
            cp_engine_producer_all_gather_full_mesh_push_multi_stream(ctx.rank, ctx.num_ranks, A, ctx.workspace_tensors,
                                                                      ctx.one, ctx.M_PER_CHUNK, ctx.ag_streams,
                                                                      ctx.barrier_tensors)

    if serial:
        call_ag()
        for ag_stream in ctx.ag_streams:
            current_stream.wait_stream(ag_stream)
        if ctx.gemm_stream_torch is not None:
            current_stream.wait_stream(ctx.gemm_stream_torch)
    else:
        call_ag()

    gemm_stream = ctx.gemm_stream_torch if ctx.gemm_stream_torch else torch.cuda.current_stream()
    with torch.cuda.stream(gemm_stream):
        if use_persistent_gemm:
            NUM_SMS = torch.cuda.get_device_properties(
                0).multi_processor_count - ctx.comm_sms if ctx.use_copy_kernel else torch.cuda.get_device_properties(
                    0).multi_processor_count
            # TODO(houqi.1993) this may be tuned
            BLOCK_SIZE_M = gemm_config.kwargs["BLOCK_SIZE_M"]
            BLOCK_SIZE_N = gemm_config.kwargs["BLOCK_SIZE_N"]
            total_tiles = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N_per_rank, BLOCK_SIZE_N)
            NUM_SMS = min(NUM_SMS, total_tiles)
            grid = (NUM_SMS, )
            full_input = ctx.workspace_tensors[ctx.rank][:M]

            kernel_consumer_gemm_persistent[grid](full_input, B, C, M, N_per_rank, K, full_input.stride(0),
                                                  full_input.stride(1), B.stride(1), B.stride(0), C.stride(0),
                                                  C.stride(1), ctx.rank, ctx.num_ranks, ctx.barrier_tensors[ctx.rank],
                                                  M_PER_CHUNK=ctx.M_PER_CHUNK, NUM_SMS=NUM_SMS,
                                                  **gemm_config.all_kwargs())
        else:
            raise NotImplementedError("Non-persistent gemm is not yet supported")

    if torch.cuda.is_current_stream_capturing():
        for ag_stream in ctx.ag_streams:
            current_stream.wait_stream(ag_stream)
        if ctx.gemm_stream_torch is not None:
            current_stream.wait_stream(ctx.gemm_stream_torch)


def ag_gemm_intra_node(A: torch.Tensor, B: torch.Tensor, ctx: AllGatherGEMMTensorParallelContext,
                       use_fused_kernel=False, autotune: bool = True):
    """allgather gemm for intra-node

    return C = all_gather(A) @ B.T

    Args:
        A (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        B (torch.Tensor<float>): local matmul B matrix. shape: [N_per_rank, K]
        ctx: (AllGatherGEMMTensorParallelContext):
        use_fused_kernel (bool): whether to use the fused all-gather-gemm kernel

    Returns:
        C (torch.Tensor<float>): local matmul C matrix. shape: [M, N_per_rank]
    """
    M_per_rank, K = A.shape
    N_per_rank, _ = B.shape
    C = torch.empty([ctx.num_ranks * M_per_rank, N_per_rank], dtype=A.dtype, device=A.device)
    ag_gemm_intra_node_op(A, B, C, ctx, use_persistent_gemm=True, use_fused_kernel=use_fused_kernel, autotune=autotune)
    return C


@triton_dist.tune.autotune(
    config_space=[{"gemm_config": c} for c in matmul_get_configs()],
    key_fn=key_fn,
    prune_fn=prune_fn_by_shared_memory,
)
def gemm_only(A: torch.Tensor, B: torch.Tensor, ctx: AllGatherGEMMTensorParallelContext, NUM_SMS: int,
              gemm_config: triton.Config = DEFAULT_CONFIG):
    M_per_rank, K = A.shape
    N_per_rank, _ = B.shape
    C = torch.empty([ctx.num_ranks * M_per_rank, N_per_rank], dtype=A.dtype, device=A.device)

    M = M_per_rank * ctx.num_ranks
    BLOCK_SIZE_M = gemm_config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.kwargs["BLOCK_SIZE_N"]
    total_tiles = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N_per_rank, BLOCK_SIZE_N)
    grid = (min(NUM_SMS, total_tiles), )
    full_input = ctx.workspace_tensors[ctx.rank][:M]

    kernel_consumer_gemm_persistent[grid](full_input, B, C, M, N_per_rank, K,
                                          full_input.stride(0), full_input.stride(1), B.stride(1), B.stride(0),
                                          C.stride(0), C.stride(1), ctx.rank, ctx.num_ranks,
                                          ctx.barrier_tensors[ctx.rank], M_PER_CHUNK=ctx.M_PER_CHUNK, NUM_SMS=NUM_SMS,
                                          **gemm_config.all_kwargs())

    return C


def allgather(A: torch.Tensor, ctx):
    current_stream = torch.cuda.current_stream()
    if ctx.use_copy_kernel:
        # SM copy uses single stream
        # Use the same BLOCK_SIZE_M as consumer kernel to ensure consistent chunk partitioning
        copy_kernel_producer_all_gather(ctx.rank, ctx.num_ranks, A, ctx.workspace_tensors, ctx.one, ctx.M_PER_CHUNK,
                                        current_stream, ctx.barrier_ptrs, ctx.dst_tensor_ptrs, ctx.chunk_counters,
                                        comm_sms=ctx.comm_sms, BLOCK_SIZE_M=128, BLOCK_SIZE_N=256)
    else:
        # CP engine uses multiple streams
        cp_engine_producer_all_gather_full_mesh_push_multi_stream(ctx.rank, ctx.num_ranks, A, ctx.workspace_tensors,
                                                                  ctx.one, ctx.M_PER_CHUNK, [current_stream],
                                                                  ctx.barrier_tensors)

    barrier_all_ipc_kernel_v2[(1, )](ctx.rank, ctx.num_ranks, ctx.comm_buf_ptr)
