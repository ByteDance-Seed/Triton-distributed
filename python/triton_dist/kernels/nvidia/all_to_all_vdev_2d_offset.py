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
import os
import math
import triton
import ctypes
import triton_dist
import triton.language as tl
import triton_dist.language as dl
from typing import Union
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.cuda.language_extra import (__syncthreads, ld, st, tid, laneid, __shfl_sync_i32)
from triton_dist.kernels.nvidia.common_ops import barrier_on_this_grid, barrier_all_intra_node_atomic_cas_block
from triton_dist.kernels.nvidia.gemm_rs_threadblock_swizzle import warp_prefix_sum_kernel
import torch
import dataclasses
from triton_dist.utils import nvshmem_create_tensor, nvshmem_free_tensor_sync, nvshmem_barrier_all_on_stream, launch_cooperative_grid_options
from triton_dist.tools.profiler import (Profiler, alloc_profiler_buffer, export_to_perfetto_trace)


@triton.jit(do_not_specialize=[])
def single_block_prefix_sum_kernel_scan_scan(
    split_ptr,
    partial_sum_ptr,
    res_ptr,
    M,
    N,
    num_warps: tl.constexpr,
    exclusive: tl.constexpr = True,
    MAJOR_ALIGN: tl.constexpr = 1,
):
    WARP_SIZE: tl.constexpr = 32
    nsplits = M * N
    thread_idx = tid(0)
    lane_id = laneid()
    warp_id = thread_idx // WARP_SIZE
    prefix_target = tl.cast(0, split_ptr.dtype.element_ty)

    if MAJOR_ALIGN <= 1:
        total_tiles = tl.cdiv(nsplits, WARP_SIZE)
        iters = tl.cdiv(total_tiles, num_warps)
        for i in range(iters):
            warp_task_id = i * num_warps + warp_id
            tile_start = warp_task_id * WARP_SIZE
            valid_len = tl.minimum(WARP_SIZE, nsplits - tile_start)
            if valid_len > 0:
                val = ld(split_ptr + tile_start + lane_id)
                prefix_inclusive = warp_prefix_sum_kernel(val, lane_id, valid_len)
                prefix_target = prefix_inclusive - val if exclusive else prefix_inclusive
                if lane_id == valid_len - 1:
                    st(partial_sum_ptr + warp_task_id, prefix_inclusive)
            __syncthreads()

            # using the first warp to do inclusive prefix sum for partial sum
            num_tiles_this_iter = tl.minimum(num_warps, total_tiles - i * num_warps)
            if warp_id == 0:
                val = ld(partial_sum_ptr + i * num_warps + lane_id)
                prefix_inclusive = warp_prefix_sum_kernel(val, lane_id, num_tiles_this_iter)
                if i > 0:
                    prefix_inclusive += ld(partial_sum_ptr + i * num_warps - 1)  # add back the offset from last iter
                st(partial_sum_ptr + i * num_warps + lane_id, prefix_inclusive)  # TODO: caution: illegal mem access
            __syncthreads()

            # add back the offset for each elem
            if lane_id < valid_len:
                if warp_task_id > 0:
                    prefix_target += ld(partial_sum_ptr + warp_task_id - 1)
                st(res_ptr + tile_start + lane_id, prefix_target)
    else:
        # we have to identify the last warp of each row to handle the alignment
        num_tiles_per_row = tl.cdiv(N, WARP_SIZE)
        total_tiles = M * num_tiles_per_row
        iters = tl.cdiv(total_tiles, num_warps)

        for i in range(iters):
            warp_task_id = i * num_warps + warp_id
            row_idx = warp_task_id // num_tiles_per_row
            col_tile_idx = warp_task_id % num_tiles_per_row
            tile_start_flat = row_idx * N + col_tile_idx * WARP_SIZE
            valid_len = tl.minimum(WARP_SIZE, N - col_tile_idx * WARP_SIZE)
            is_valid_tile = warp_task_id < total_tiles and valid_len > 0

            if is_valid_tile:
                val = ld(split_ptr + tile_start_flat + lane_id)
                prefix_inclusive = warp_prefix_sum_kernel(val, lane_id, valid_len)
                prefix_target = prefix_inclusive - val if exclusive else prefix_inclusive
                if lane_id == valid_len - 1:
                    st(partial_sum_ptr + warp_task_id, prefix_inclusive)
            __syncthreads()

            num_tiles_this_iter = tl.minimum(num_warps, total_tiles - i * num_warps)
            if warp_id == 0:
                val = ld(partial_sum_ptr + i * num_warps + lane_id)
                prefix_inclusive = warp_prefix_sum_kernel(
                    val, lane_id, num_tiles_this_iter)  # inter-warp: add offset from previous warps
                if i > 0:
                    prefix_inclusive += ld(partial_sum_ptr + i * num_warps -
                                           1)  # inter-block: add back the offset from last iter

                for k in range(num_tiles_this_iter):
                    g_tile_id = i * num_warps + k
                    if (g_tile_id + 1) % num_tiles_per_row == 0:  #
                        row_end_sum = __shfl_sync_i32(0xFFFFFFFF, prefix_inclusive, k)
                        aligned_val = (row_end_sum + MAJOR_ALIGN - 1) // MAJOR_ALIGN * MAJOR_ALIGN
                        aligned_val = tl.maximum(aligned_val, MAJOR_ALIGN)
                        delta = aligned_val - row_end_sum
                        if lane_id >= k:
                            prefix_inclusive += delta
                st(partial_sum_ptr + i * num_warps + lane_id, prefix_inclusive)  # TODO: caution: illegal mem access
            __syncthreads()

            # add back the offset for each elem
            if lane_id < valid_len:
                if warp_task_id > 0:
                    prefix_target += ld(partial_sum_ptr + warp_task_id - 1)
                st(res_ptr + tile_start_flat + lane_id, prefix_target)


@triton_dist.jit(do_not_specialize=[])
def exchange_split_offset_2d_intra_node(
    in_splits_offsets,  # (2, nsplits)
    out_splits_offsets,  # (2, nsplits)
    partial_sum_ptr,  # (num_tiles,)
    symm_signal_ptr,
    num_expert_per_rank: tl.constexpr,
    rank_in_row: tl.constexpr = True,
    has_input_off: tl.constexpr = False,
    extra_barrier: tl.constexpr = False,
    num_warps: tl.constexpr = 32,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    nsplits = world_size * num_expert_per_rank

    in_splits = in_splits_offsets
    in_offset = in_splits_offsets + nsplits
    out_splits = out_splits_offsets
    source_offset = out_splits_offsets + nsplits
    if not has_input_off:
        single_block_prefix_sum_kernel_scan_scan(
            in_splits,
            partial_sum_ptr,
            in_offset,  # as the output
            world_size if rank_in_row else num_expert_per_rank,
            num_expert_per_rank if rank_in_row else world_size,
            num_warps=num_warps,
        )
        __syncthreads()

    if extra_barrier:
        barrier_all_intra_node_atomic_cas_block(rank, rank, world_size, symm_signal_ptr)

    with dl.simt_exec_region() as (thread_idx, threads_per_block):
        for idx in range(thread_idx, nsplits, threads_per_block):
            if rank_in_row:
                peer_off = idx // num_expert_per_rank
                peer = (rank + peer_off) % world_size
                e = idx % num_expert_per_rank
                dst_offset = e * world_size + rank
                ld_idx = peer * num_expert_per_rank + e
            else:
                peer = idx % world_size
                e = idx // world_size
                dst_offset = rank * num_expert_per_rank + e
                ld_idx = idx
            split_val = ld(in_splits + ld_idx)
            offset_val = ld(in_offset + ld_idx)
            remote_out_splits = dl.symm_at(out_splits, peer)
            remote_source_offset = dl.symm_at(source_offset, peer)
            st(remote_out_splits + dst_offset, split_val)
            st(remote_source_offset + dst_offset, offset_val)

    barrier_all_intra_node_atomic_cas_block(rank, rank, world_size, symm_signal_ptr)


@triton_dist.jit(do_not_specialize=["num_expert_per_rank"])
def all_to_all_v_2d_kernel(
    profiler_buf,
    in_splits_offsets,  # (2, nsplits), symmetric
    out_splits_offsets,  # (2, nsplits), symmetric
    local_split_offset,  # (nsplits,)
    partial_sum_ptr,  # (num_tiles,)
    signal_pad_ptr,
    recv_data,
    send_data,
    token_stride: tl.constexpr,  # token len elem
    num_expert_per_rank: tl.constexpr,
    rank_in_row: tl.constexpr,
    has_input_off: tl.constexpr = False,
    MAJOR_ALIGN: tl.constexpr = 1,
    launch_cooperative_grid: tl.constexpr = False,
    num_warps: tl.constexpr = 32,
    profiling: tl.constexpr = False,
):
    profiler = Profiler.create(
        profiler_buffer=profiler_buf,
        group_id=0,
        num_groups=1,
        is_leader=(tid(0) == 0),
        ENABLE_PROFILING=profiling,
    )
    rank = dl.rank()
    world_size = dl.num_ranks()
    nsplits = world_size * num_expert_per_rank
    if rank_in_row:
        M, N = world_size, num_expert_per_rank
    else:
        M, N = num_expert_per_rank, world_size
    out_splits = out_splits_offsets
    source_offset = out_splits_offsets + nsplits
    barrier_ptr = signal_pad_ptr
    grid_barrier = signal_pad_ptr + world_size
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    if pid == 0:
        barrier_all_intra_node_atomic_cas_block(rank, rank, world_size, barrier_ptr)

        profiler = profiler.record(is_start=True, task_type=0)
        exchange_split_offset_2d_intra_node(
            in_splits_offsets,
            out_splits_offsets,
            partial_sum_ptr,
            barrier_ptr,
            num_expert_per_rank,
            rank_in_row=rank_in_row,
            has_input_off=has_input_off,
            extra_barrier=False,
            num_warps=num_warps,
        )  # this kernel has a barrier after exchange
        profiler = profiler.record(is_start=False, task_type=0)

        profiler = profiler.record(is_start=True, task_type=1)
        single_block_prefix_sum_kernel_scan_scan(
            out_splits,
            partial_sum_ptr,
            local_split_offset,  # as the output
            N,
            M,  # important: now `M` is the minor size after transpose
            num_warps=num_warps,
            MAJOR_ALIGN=MAJOR_ALIGN,
        )
        profiler = profiler.record(is_start=False, task_type=1)

    profiler = profiler.record(is_start=True, task_type=2)
    if num_pid != 1:
        barrier_on_this_grid(grid_barrier, launch_cooperative_grid)
    else:
        __syncthreads()  # make sure `local_split_offset` is ready
    profiler = profiler.record(is_start=False, task_type=2)

    profiler = profiler.record(is_start=True, task_type=3)
    elem_size = tl.constexpr(send_data.dtype.element_ty.primitive_bitwidth) // 8

    rr_shift = rank if rank_in_row else rank * num_expert_per_rank  # round-robin
    for idx in range(pid, nsplits, num_pid):
        task_id = (idx + rr_shift) % nsplits
        chunk_size_bytes = tl.load(out_splits + task_id) * token_stride * elem_size
        read_offset = tl.load(source_offset + task_id) * token_stride
        local_off = tl.load(local_split_offset + task_id)
        write_offset = local_off * token_stride
        peer = task_id % M if rank_in_row else task_id // M  # important: splits/offs are transposed, using `M` instead of `N` to compute peer
        libshmem_device.getmem_nbi_block(recv_data + write_offset, send_data + read_offset, chunk_size_bytes, peer)
        tl.store(source_offset + task_id, local_off)  # write back
    profiler = profiler.record(is_start=False, task_type=3)


@triton_dist.jit(do_not_specialize=["num_expert_per_rank", "stage"])
def all_to_all_v_2d_kernel_v2(
    profiler_buf,
    input,
    output,
    input_splits,  # (nsplits,), local
    output_splits,  # (nsplits,), local
    send_recv_num_tokens,  # (2,), pinned memory
    symm_splits_workspace,  # (2, nsplits), symmetric
    symm_data_workspace,  # (max_num_token * token_len,), symmetric
    partial_sum_ptr,  # (num_tiles,)
    signal_pad_ptr,
    stage,
    total_stages: tl.constexpr,
    rank_in_row: tl.constexpr,
    token_len: tl.constexpr,  # token len elem, must be a power of 2
    num_expert_per_rank: tl.constexpr,
    profiling: tl.constexpr = False,
    copy_to_symm_buffer: tl.constexpr = True,
    launch_cooperative_grid: tl.constexpr = False,
    num_warps: tl.constexpr = 32,
    BS_M: tl.constexpr = 1,
):
    profiler = Profiler.create(
        profiler_buffer=profiler_buf,
        group_id=0,
        num_groups=1,
        is_leader=(tid(0) == 0),
        ENABLE_PROFILING=profiling,
    )
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    rank = dl.rank()
    world_size = dl.num_ranks()
    thread_idx = tid(0)
    nsplits = world_size * num_expert_per_rank
    if rank_in_row:
        M, N = world_size, num_expert_per_rank
    else:
        M, N = num_expert_per_rank, world_size

    barrier_ptr = signal_pad_ptr
    grid_barrier = signal_pad_ptr + world_size
    in_offset = symm_splits_workspace
    store_offset = symm_splits_workspace + nsplits
    source_offset = symm_splits_workspace + 2 * nsplits
    send_data = symm_data_workspace
    recv_data = output

    if pid == 0:
        last_stage = (stage + total_stages - 1) % total_stages
        tl.store(send_recv_num_tokens + last_stage * 2 + tl.arange(0, 2), -1)
        profiler = profiler.record(is_start=True, task_type=1)  # prefix sum
        single_block_prefix_sum_kernel_scan_scan(
            input_splits,
            partial_sum_ptr,
            in_offset,  # as the output
            M,
            N,
            num_warps=num_warps,
        )
        __syncthreads()
        single_block_prefix_sum_kernel_scan_scan(
            output_splits,
            partial_sum_ptr,
            store_offset,  # as the output
            N,
            M,  # important: now `M` is the minor size after transpose
            num_warps=num_warps,
        )
        profiler = profiler.record(is_start=False, task_type=1)  # prefix sum
    barrier_on_this_grid(grid_barrier, launch_cooperative_grid)

    num_send_tokens = tl.load(in_offset + nsplits - 1) + tl.load(input_splits + nsplits - 1)
    num_send_tokens = tl.cast(num_send_tokens, tl.int32)
    num_recv_tokens = tl.load(store_offset + nsplits - 1) + tl.load(output_splits + nsplits - 1)
    num_recv_tokens = tl.cast(num_recv_tokens, tl.int32)
    if copy_to_symm_buffer:
        profiler = profiler.record(is_start=True, task_type=3)  # copy to symm buffer
        tokens_per_pid = num_send_tokens // num_pid
        remainder = num_send_tokens % num_pid
        start_token = pid * tokens_per_pid + tl.minimum(pid, remainder)
        count = tokens_per_pid + (1 if pid < remainder else 0)
        end_token = start_token + count
        num_iters = tl.cdiv(count, BS_M)
        start_elem = start_token * token_len
        for iter in range(num_iters):
            offs = start_elem + iter * BS_M * token_len + tl.arange(0, BS_M * token_len)
            mask = (offs < end_token * token_len)
            val = tl.load(input + offs, mask=mask)
            tl.store(send_data + offs, val, mask=mask)
        profiler = profiler.record(is_start=False, task_type=3)  # copy to symm buffer
        barrier_on_this_grid(grid_barrier, launch_cooperative_grid)

    profiler = profiler.record(is_start=True, task_type=2)  # exchange source offset
    if pid == 0:
        tl.store(send_recv_num_tokens + stage * 2, num_send_tokens)
        tl.store(send_recv_num_tokens + stage * 2 + 1, num_recv_tokens)
        profiler = profiler.record(is_start=True, task_type=5)
        with dl.simt_exec_region() as (thread_idx, threads_per_block):
            for idx in range(thread_idx, nsplits, threads_per_block):
                in_split_val = ld(input_splits + idx)
                if in_split_val > 0:  # in case of sparse splits
                    if rank_in_row:
                        peer = idx // num_expert_per_rank
                        e = idx % num_expert_per_rank
                        dst_offset = e * world_size + rank
                    else:
                        peer = idx % world_size
                        e = idx // world_size
                        dst_offset = rank * num_expert_per_rank + e
                    offset_val = ld(in_offset + idx)
                    remote_source_offset = dl.symm_at(source_offset, peer)
                    st(remote_source_offset + dst_offset, offset_val)
            __syncthreads()
        profiler = profiler.record(is_start=False, task_type=5)

        profiler = profiler.record(is_start=True, task_type=6)
        barrier_all_intra_node_atomic_cas_block(rank, rank, world_size, barrier_ptr)
        profiler = profiler.record(is_start=False, task_type=6)
    barrier_on_this_grid(grid_barrier, launch_cooperative_grid)
    profiler = profiler.record(is_start=False, task_type=2)  # exchange source offset

    profiler = profiler.record(is_start=True, task_type=4)  # exchange data
    elem_size = tl.constexpr(send_data.dtype.element_ty.primitive_bitwidth) // 8
    tokens_per_pid = num_recv_tokens // num_pid
    remainder = num_recv_tokens % num_pid
    start_token = pid * tokens_per_pid + tl.minimum(pid, remainder)
    count = tokens_per_pid + (1 if pid < remainder else 0)
    end_token = start_token + count

    if count > 0:
        left = 0
        right = nsplits
        while left < right:
            mid = (left + right) // 2
            mid_val = tl.load(store_offset + mid)
            if mid_val <= start_token:
                left = mid + 1
            else:
                right = mid
        curr_task_idx = tl.maximum(0, left - 1)
        cur_token_pos = start_token

        while cur_token_pos < end_token and curr_task_idx < nsplits:
            task_base_token = tl.load(store_offset + curr_task_idx)
            task_num_tokens = tl.load(output_splits + curr_task_idx)
            task_end_token_global = task_base_token + task_num_tokens
            chunk_start = tl.maximum(cur_token_pos, task_base_token)
            chunk_end = tl.minimum(end_token, task_end_token_global)
            valid_chunk_len = chunk_end - chunk_start

            if valid_chunk_len > 0:
                in_split_offset = chunk_start - task_base_token
                read_base = tl.load(source_offset + curr_task_idx)
                read_ptr = (read_base + in_split_offset) * token_len
                write_ptr = (task_base_token + in_split_offset) * token_len
                chunk_size_bytes = valid_chunk_len * token_len * elem_size
                if rank_in_row:
                    peer = curr_task_idx % M
                else:
                    peer = curr_task_idx // M
                libshmem_device.getmem_nbi_block(recv_data + write_ptr, send_data + read_ptr, chunk_size_bytes, peer)
                cur_token_pos += valid_chunk_len
            curr_task_idx += 1

    profiler = profiler.record(is_start=False, task_type=4)  # exchange data


@dataclasses.dataclass
class AllToAllContext:
    rank: int
    world_size: int
    ne: int
    k: int  # max val of each splits
    max_num_token: int
    token_len_elem: int
    use_v2_kernel: bool
    # Symmetric Buffers
    split_meta_buf: torch.Tensor  # [in_splits (2*nsplits) | out_splits (2*nsplits)]
    data_meta_buf: torch.Tensor
    signal_buf: torch.Tensor  # [barrier (world_size) | grid_barrier (1 or aligned)]
    # Local Buffers
    local_offset_buf: torch.Tensor
    partial_sum_buf: torch.Tensor
    token_dtype: torch.dtype
    num_send_recv_tokens: torch.Tensor

    def finalize(self):
        attrs_to_free = ['split_meta_buf', 'signal_buf', 'data_meta_buf']
        for attr in attrs_to_free:
            if hasattr(self, attr):
                t = getattr(self, attr)
                nvshmem_free_tensor_sync(t)
                t = None
                del t
                setattr(self, attr, None)
                delattr(self, attr)

    def __del__(self):
        self.finalize()

    def __post_init__(self):
        self.nsplits = self.world_size * self.ne
        self.dispatch_splits_offsets = self.split_meta_buf[0:2 * self.nsplits]
        self.combine_splits_offsets = self.split_meta_buf[2 * self.nsplits:]
        self.dispatch_data = self.data_meta_buf[0:self.max_num_token * self.token_len_elem]
        self.combine_data = self.data_meta_buf[self.max_num_token * self.token_len_elem:]
        self.profile_buf = alloc_profiler_buffer(max_num_profile_slots=1000000)
        self.total_stages = self.num_send_recv_tokens.numel() // 2
        self.current_stage = 0

    def get_stage(self) -> int:
        ret = self.current_stage
        self.current_stage = (self.current_stage + 1) % self.total_stages
        return ret

    def dump_profiler_trace(self, info: str = None):
        profiler_dir = "./prof"
        if info is not None and info != "":
            profiler_dir += f"/{info}"
        os.makedirs(profiler_dir, exist_ok=True)
        name = f"all_to_all_dev_RANK_{self.rank}"
        trace_file = os.path.join(profiler_dir, name)
        if self.use_v2_kernel:
            task_names = [
                "1st barrier", "prefix sum", "exchange source offset", "copy local", "get_mem", "pure source offset",
                "intra barrier", "reset & record"
            ]
        else:
            task_names = ["split_exchange", "out_splits_scan", "wait", "get_mem"]
        export_to_perfetto_trace(
            profiler_buffer=self.profile_buf,
            task_names=task_names,
            file_name=trace_file,
        )


def create_context(
    rank: int,
    world_size: int,
    ne: int,
    k: int,
    token_len_elem: int,
    token_dtype: torch.dtype,
    overflow_factor: Union[int, float] = None,
    max_token_num_per_rank: int = None,
    use_v2_kernel: bool = False,
) -> AllToAllContext:
    # Buffer layout for split_meta_buf: [in_splits (2*nsplits) | out_splits (2*nsplits)]
    # Size: 2 * nsplits * sizeof(int32) * 2 (for in and out)
    nsplits = world_size * ne
    meta_num_elems = 4 * nsplits
    split_meta_buf = nvshmem_create_tensor((meta_num_elems, ), dtype=torch.int32)

    # Buffer layout for signal_buf: [barrier (world_size) | grid_barrier (1 or aligned)]
    signal_num_elems = world_size + 8
    signal_buf = nvshmem_create_tensor((signal_num_elems, ), dtype=torch.int64)
    signal_buf.zero_()

    # send and recv buffer
    max_token_num_per_rank = k * nsplits if max_token_num_per_rank is None else max_token_num_per_rank
    overflow_factor = world_size if overflow_factor is None else overflow_factor  # worst case: one rank receives all data
    max_num_token = max_token_num_per_rank * math.ceil(overflow_factor)
    if use_v2_kernel:
        data_meta_buf = nvshmem_create_tensor((max_num_token * token_len_elem, ), token_dtype)
    else:
        data_meta_buf = nvshmem_create_tensor((2 * max_num_token * token_len_elem, ), token_dtype)

    # Local buffers
    local_offset_buf = torch.empty((nsplits, ), dtype=torch.int32, device="cuda")
    num_tiles = triton.cdiv(nsplits, 32)
    partial_sum_buf = torch.zeros((2 * num_tiles, ), dtype=torch.int32, device="cuda")
    num_send_recv_tokens = torch.empty((8, ), dtype=torch.int32, device="cpu", pin_memory=True)
    num_send_recv_tokens.fill_(-1)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    return AllToAllContext(rank=rank, world_size=world_size, ne=ne, k=k, max_num_token=max_num_token,
                           use_v2_kernel=use_v2_kernel, token_len_elem=token_len_elem, token_dtype=token_dtype,
                           split_meta_buf=split_meta_buf, data_meta_buf=data_meta_buf, signal_buf=signal_buf,
                           local_offset_buf=local_offset_buf, partial_sum_buf=partial_sum_buf,
                           num_send_recv_tokens=num_send_recv_tokens)


def all_to_all_v_offset_op_v2(ctx: AllToAllContext, rank_in_row: bool, input: torch.Tensor, output: torch.Tensor,
                              in_splits: torch.Tensor, out_splits: torch.Tensor, copy_to_symm_buffer: bool = True,
                              return_tensor_sliced: bool = True, return_stat: bool = False, grid_size: int = None,
                              profiling: bool = False):
    assert in_splits.dtype == ctx.split_meta_buf.dtype and out_splits.dtype == ctx.split_meta_buf.dtype
    assert in_splits.numel() == ctx.nsplits and out_splits.numel() == ctx.nsplits

    assert input.dtype == ctx.token_dtype and output.dtype == ctx.token_dtype
    assert input.is_contiguous() and output.is_contiguous()
    assert input.size(1) == output.size(1) and input.size(1) == ctx.token_len_elem
    assert input.size(0) <= ctx.max_num_token and output.size(0) <= ctx.max_num_token

    grid = (ctx.ne, ) if grid_size is None else (grid_size, )
    BS_M = (1024 * 32) // (ctx.token_len_elem * input.element_size())
    current_stage = ctx.get_stage()
    all_to_all_v_2d_kernel_v2[grid](
        ctx.profile_buf,
        input,
        output,
        in_splits,
        out_splits,
        ctx.num_send_recv_tokens,
        ctx.split_meta_buf,
        ctx.data_meta_buf,
        stage=current_stage,
        total_stages=ctx.total_stages,
        partial_sum_ptr=ctx.partial_sum_buf,
        signal_pad_ptr=ctx.signal_buf,
        num_expert_per_rank=ctx.ne,
        rank_in_row=rank_in_row,
        profiling=profiling,
        token_len=ctx.token_len_elem,
        copy_to_symm_buffer=copy_to_symm_buffer,
        **launch_cooperative_grid_options(),
        num_warps=32,
        BS_M=BS_M,
    )

    if return_stat:
        base_ptr = ctx.num_send_recv_tokens[current_stage * 2].data_ptr()
        elem_bytes = ctx.num_send_recv_tokens.itemsize
        while ctypes.c_int32.from_address(base_ptr).value == -1:  # num recv token
            pass
        while ctypes.c_int32.from_address(base_ptr + elem_bytes).value == -1:  # num recv token
            pass
        send_token_num = ctypes.c_int32.from_address(base_ptr).value
        recv_token_num = ctypes.c_int32.from_address(base_ptr + elem_bytes).value
    else:
        send_token_num = None
        recv_token_num = None

    if return_tensor_sliced:
        num_elem = recv_token_num * ctx.token_len_elem
        if copy_to_symm_buffer:
            return output,
        else:
            return ctx.combine_data[:num_elem].view(-1, ctx.token_len_elem), send_token_num, recv_token_num
    else:
        ret = None
    return ret, send_token_num, recv_token_num


def all_to_all_v_offset_op(ctx: AllToAllContext, rank_in_row: bool, input: torch.Tensor = None,
                           output: torch.Tensor = None, in_splits: torch.Tensor = None, in_offset: torch.Tensor = None,
                           copy_to_symm_buffer: bool = True,  # False: data already in comm buffer
                           return_tensor: bool = True, major_align: int = 1, grid_size: int = None,
                           has_input_offset: bool = False, profiling: bool = False):
    if rank_in_row:
        in_splits_offsets = ctx.dispatch_splits_offsets
        out_splits_offsets = ctx.combine_splits_offsets
        send_data = ctx.dispatch_data
        recv_data = ctx.combine_data
    else:
        in_splits_offsets = ctx.combine_splits_offsets
        out_splits_offsets = ctx.dispatch_splits_offsets
        send_data = ctx.combine_data
        recv_data = ctx.dispatch_data

    if copy_to_symm_buffer:
        assert in_splits is not None
        assert in_splits.dim() == 2
        assert in_splits.dtype == in_splits_offsets.dtype
        assert in_splits.numel() == ctx.nsplits
        in_splits_offsets[:ctx.nsplits].copy_(in_splits.view(-1))

        if has_input_offset:
            assert in_offset is not None
            assert in_offset.dtype == in_splits_offsets.dtype
            assert in_offset.numel() == ctx.nsplits
            in_splits_offsets[ctx.nsplits:].copy_(in_offset.view(-1))

        assert input is not None
        assert input.dtype == ctx.token_dtype
        assert input.numel() <= ctx.max_num_token * ctx.token_len_elem
        if not input.is_contiguous():
            raise Warning("input tensor is not contiguous")
        input_flat = input.contiguous().view(-1)
        send_data[:input_flat.numel()].copy_(input_flat)

    grid = (ctx.ne, ) if grid_size is None else (grid_size, )
    all_to_all_v_2d_kernel[grid](
        in_splits_offsets=in_splits_offsets,
        out_splits_offsets=out_splits_offsets,
        partial_sum_ptr=ctx.partial_sum_buf,
        num_expert_per_rank=ctx.ne,
        rank_in_row=rank_in_row,
        has_input_off=has_input_offset,
        profiler_buf=ctx.profile_buf,
        profiling=profiling,
        local_split_offset=ctx.local_offset_buf,
        signal_pad_ptr=ctx.signal_buf,
        recv_data=recv_data,
        send_data=send_data,
        token_stride=ctx.token_len_elem,
        MAJOR_ALIGN=major_align,
        **launch_cooperative_grid_options(),
        num_warps=32,
    )

    last_split_size = out_splits_offsets[ctx.nsplits - 1]
    num_token_aligned = ctx.local_offset_buf[-1] + last_split_size
    num_elem = num_token_aligned * ctx.token_len_elem

    if copy_to_symm_buffer:
        assert output is not None
        assert output.dtype == ctx.token_dtype
        assert output.numel() <= ctx.max_num_token * ctx.token_len_elem
        if not output.is_contiguous():
            raise Warning("output tensor is not contiguous")
        output_flat = output.contiguous().view(-1)
        output_flat[:num_elem].copy_(recv_data.view(-1)[:num_elem])

    if return_tensor:
        ret = output if copy_to_symm_buffer else recv_data[:num_elem].view(-1, ctx.token_len_elem)
        return ret, num_token_aligned
    else:
        return None, None


def all_to_all_vdev_2d(ctx: AllToAllContext, input: torch.Tensor = None, output: torch.Tensor = None,
                       in_splits: torch.Tensor = None, copy_to_symm_buffer: bool = True, major_align: int = 1,
                       grid_size: int = None, profiling: bool = False):
    return all_to_all_v_offset_op(
        ctx=ctx,
        rank_in_row=True,
        input=input,
        output=output,
        in_splits=in_splits,
        in_offset=None,
        copy_to_symm_buffer=copy_to_symm_buffer,
        major_align=major_align,
        grid_size=grid_size,
        has_input_offset=False,
        profiling=profiling,
        return_tensor=True,
    )


def all_to_all_vdev_2d_offset(ctx: AllToAllContext, input: torch.Tensor = None, output: torch.Tensor = None,
                              in_splits: torch.Tensor = None, in_offset: torch.Tensor = None,
                              copy_to_symm_buffer: bool = True, grid_size: int = None, profiling: bool = False):
    return all_to_all_v_offset_op(
        ctx=ctx,
        rank_in_row=False,
        input=input,
        output=output,
        in_splits=in_splits,
        in_offset=in_offset,
        copy_to_symm_buffer=copy_to_symm_buffer,
        major_align=1,  # no major alignment allowed
        grid_size=grid_size,
        has_input_offset=True,  # input offsets must be provided
        profiling=profiling,
        return_tensor=True,
    )
