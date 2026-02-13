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
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.cuda.language_extra import (__syncthreads, ld, st, tid, membar, laneid, ld_acquire,
                                                            __shfl_sync_i32)
from triton_dist.kernels.nvidia.common_ops import barrier_on_this_grid, barrier_all_intra_node_atomic_cas_block
from triton_dist.kernels.nvidia.gemm_rs_threadblock_swizzle import warp_prefix_sum_kernel
import torch
import dataclasses
import triton_dist
from triton_dist.utils import nvshmem_create_tensor, nvshmem_free_tensor_sync, nvshmem_barrier_all_on_stream, launch_cooperative_grid_options
from triton_dist.tools.profiler import (Profiler, alloc_profiler_buffer, export_to_perfetto_trace)
from triton_dist.kernels.nvidia.all_to_all_vdev_2d_offset import single_block_prefix_sum_kernel_scan_scan

SPLIT_DTYPE = torch.int32
SIGNAL_DTYPE = torch.int64


@triton.jit
def transpose_kernel(ptr_x, ptr_y, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    n_block_m = tl.cdiv(M, BLOCK_M)
    n_block_n = tl.cdiv(N, BLOCK_N)

    for bid in range(pid, n_block_m * n_block_n, num_pid):
        pid_m = bid // n_block_n
        pid_n = bid % n_block_n
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_x = (rm[:, None] < M) & (rn[None, :] < N)
        offs_x = rm[:, None] * N + rn[None, :]
        val = tl.load(ptr_x + offs_x, mask=mask_x, other=0.0)

        val_t = tl.trans(val)
        row_y = rn
        col_y = rm
        mask_y = (row_y[:, None] < N) & (col_y[None, :] < M)
        offs_y = row_y[:, None] * M + col_y[None, :]
        tl.store(ptr_y + offs_y, val_t, mask=mask_y)


@triton_dist.jit
def scatter_tile_intra_node(
    splits_buf,
    offset_buf,
    out_split,
    out_source_offset,
    rank_in_row: tl.constexpr,
    local_world_size: tl.constexpr,
    num_expert_per_rank: tl.constexpr,
    elem_size: tl.constexpr,
    from_node: tl.constexpr,
    num_warps: tl.constexpr,
):
    WARP_SIZE: tl.constexpr = 32
    thread_idx = tid(0)
    warp_id = thread_idx // WARP_SIZE
    rank = dl.rank()
    world_size = dl.num_ranks()
    local_rank = rank % local_world_size
    node_id = rank // local_world_size
    splits_per_node = local_world_size * num_expert_per_rank
    from_rank = local_rank + from_node * local_world_size
    if rank_in_row:
        with dl.simt_exec_region() as (thread_idx, threads_per_block):
            for idx in range(thread_idx, splits_per_node, threads_per_block):
                split_val = ld(splits_buf + from_node * splits_per_node + idx)
                offset_val = ld(offset_buf + from_node * splits_per_node + idx)
                to_local_rank = idx // num_expert_per_rank
                exp_id = idx % num_expert_per_rank
                to_rank = to_local_rank + node_id * local_world_size
                # send to `to_rank` and do transpose
                write_pos = exp_id * world_size + from_rank
                remote_out_splits = dl.symm_at(out_split, to_rank)
                remote_out_offset = dl.symm_at(out_source_offset, to_rank)
                st(remote_out_splits + write_pos, split_val)
                st(remote_out_offset + write_pos, offset_val)
    else:
        # use putmem_warp to send data to `to_rank`
        for to_local_rank in range(warp_id, local_world_size, num_warps):
            to_rank = to_local_rank + node_id * local_world_size
            libshmem_device.putmem_nbi_warp(
                out_split + from_rank * num_expert_per_rank,
                splits_buf + from_node * splits_per_node + to_local_rank * num_expert_per_rank,
                num_expert_per_rank * elem_size, to_rank)
            libshmem_device.putmem_nbi_warp(
                out_source_offset + from_rank * num_expert_per_rank,
                offset_buf + from_node * splits_per_node + to_local_rank * num_expert_per_rank,
                num_expert_per_rank * elem_size, to_rank)


@triton.jit
def per_row_prefix_kernel(
    split_ptr,
    partial_sum_ptr,
    res_ptr,
    M,
    N,
    num_warps: tl.constexpr,
):
    WARP_SIZE: tl.constexpr = 32
    thread_idx = tid(0)
    lane_id = laneid()
    warp_id = thread_idx // WARP_SIZE
    prefix_target = tl.cast(0, split_ptr.dtype.element_ty)
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
            prefix_target = prefix_inclusive - val
            if lane_id == valid_len - 1:
                st(partial_sum_ptr + warp_task_id, prefix_inclusive)
        __syncthreads()

        num_tiles_this_iter = tl.minimum(num_warps, total_tiles - i * num_warps)
        tile_prefix_row_wise = tl.cast(0, split_ptr.dtype.element_ty)
        if warp_id == 0:
            my_tile_sum = ld(partial_sum_ptr + i * num_warps + lane_id)
            my_tile_row = (i * num_warps + lane_id) // num_tiles_per_row
            for k in range(num_tiles_this_iter):
                k_tile_row = (i * num_warps + k) // num_tiles_per_row
                k_tile_sum = __shfl_sync_i32(0xFFFFFFFF, my_tile_sum, k)
                if lane_id >= k and k_tile_row == my_tile_row:
                    tile_prefix_row_wise += k_tile_sum
            if i > 0 and (i * num_warps - 1) // num_tiles_per_row == my_tile_row:
                tile_prefix_row_wise += ld(partial_sum_ptr + i * num_warps -
                                           1)  # inter-block: add back the offset from last iter
            st(partial_sum_ptr + i * num_warps + lane_id, tile_prefix_row_wise)
        __syncthreads()

        # add back the offset for each elem
        if is_valid_tile and lane_id < valid_len:
            if col_tile_idx > 0:
                prefix_target += ld(partial_sum_ptr + warp_task_id - 1)
            st(res_ptr + tile_start_flat + lane_id, prefix_target)


@triton_dist.jit(do_not_specialize=[])
def exchange_metadata_inter_node_kernel(
    in_splits_offsets,
    trans_in_splits_offsets,  # for transpose
    recv_splits_offsets,
    out_splits_offsets,
    per_node_offsets,
    node_barrier_signal,
    intra_node_barrier_signal,
    partial_sum_ptr,  # [num_tiles]
    local_world_size: tl.constexpr,
    num_expert_per_rank: tl.constexpr,
    rank_in_row: tl.constexpr = True,
    has_input_off: tl.constexpr = False,
    extra_barrier: tl.constexpr = False,
    num_warps: tl.constexpr = 32,
):
    rank = dl.rank()
    local_rank = rank % local_world_size
    world_size = dl.num_ranks()
    nsplits = world_size * num_expert_per_rank
    splits_per_node = local_world_size * num_expert_per_rank
    nnodes = world_size // local_world_size
    node_id = rank // local_world_size
    thread_idx = tid(0)
    elem_size = tl.constexpr(in_splits_offsets.dtype.element_ty.primitive_bitwidth) // 8

    in_splits = in_splits_offsets
    in_offsets = in_splits_offsets + nsplits
    trans_in_splits = trans_in_splits_offsets
    trans_in_offsets = trans_in_splits_offsets + nsplits
    recv_splits = recv_splits_offsets
    recv_offsets = recv_splits_offsets + nsplits
    out_split = out_splits_offsets
    out_source_offset = out_splits_offsets + nsplits

    if not has_input_off:
        single_block_prefix_sum_kernel_scan_scan(
            in_splits,
            partial_sum_ptr,
            in_offsets,  # as the output
            world_size if rank_in_row else num_expert_per_rank,
            num_expert_per_rank if rank_in_row else world_size,
            num_warps=num_warps,
        )
        __syncthreads()

    if not rank_in_row:
        transpose_kernel(
            in_splits,
            trans_in_splits,
            num_expert_per_rank,
            world_size,
            BLOCK_M=16,
            BLOCK_N=32,
        )
        transpose_kernel(
            in_offsets,
            trans_in_offsets,
            num_expert_per_rank,
            world_size,
            BLOCK_M=16,
            BLOCK_N=32,
        )
        __syncthreads()
        rank_major_in_splits = trans_in_splits
        rank_major_in_offsets = trans_in_offsets
    else:
        rank_major_in_splits = in_splits
        rank_major_in_offsets = in_offsets

    per_row_prefix_kernel(
        rank_major_in_splits,
        partial_sum_ptr,
        per_node_offsets,  # as the output
        nnodes,  # as the num of rows
        splits_per_node,
        num_warps=num_warps,
    )
    __syncthreads()
    membar(scope="cta")

    if extra_barrier:
        libshmem_device.barrier_all_block()

    for node_offset in range(1, nnodes):
        target_node = (node_id + node_offset) % nnodes
        target_rank = local_rank + target_node * local_world_size
        libshmem_device.putmem_nbi_block(
            recv_splits + node_id * splits_per_node,
            rank_major_in_splits + target_node * splits_per_node,
            splits_per_node * elem_size,
            target_rank,
        )
        libshmem_device.fence()
        libshmem_device.putmem_signal_nbi_block(
            recv_offsets + node_id * splits_per_node,
            per_node_offsets + target_node * splits_per_node,
            splits_per_node * elem_size,
            node_barrier_signal + node_id,
            1,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            target_rank,
        )

    for node_offset in range(0, nnodes):
        from_node = (node_id - node_offset + nnodes) % nnodes
        if node_offset > 0:
            if thread_idx == 0:
                libshmem_device.signal_wait_until(node_barrier_signal + from_node, libshmem_device.NVSHMEM_CMP_EQ, 1)
            __syncthreads()
        src_splits = rank_major_in_splits if node_offset == 0 else recv_splits
        src_offsets = rank_major_in_offsets if node_offset == 0 else recv_offsets
        scatter_tile_intra_node(
            src_splits,
            src_offsets,
            out_split,
            out_source_offset,
            rank_in_row,
            local_world_size,
            num_expert_per_rank,
            elem_size,
            from_node,
            num_warps,
        )
        __syncthreads()
    barrier_all_intra_node_atomic_cas_block(local_rank, rank, local_world_size, intra_node_barrier_signal)


@triton_dist.jit(do_not_specialize=["num_expert_per_rank"])
def all_to_all_v_2d_inter_node_kernel(
    profiler_buf,
    in_splits_offsets,  # (2, nsplits), symmetric
    trans_in_splits_offsets,  # (2, nsplits), symmetric
    recv_splits_offsets,  # (2, nsplits), symmetric
    out_splits_offsets,  # (2, nsplits), symmetric
    per_node_offsets,  # (nsplits,), symmetric
    local_store_offset,  # (nsplits,), local
    partial_sum_ptr,  # (num_tiles,), local
    signal_pad,  # (2 * nnodes + local_world_size,), symmetric
    grid_barrier_signal,  # (1,), symmetric / local
    send_data,  # max_tokens_per_rank * token_len
    transfer_buffer,  # max_tokens_per_rank * token_len
    recv_data,  # max_tokens_per_rank * token_len
    K: tl.constexpr,  # max token num per split
    sig_pad_len: tl.constexpr,
    rank_in_row: tl.constexpr,
    token_len_elem: tl.constexpr,
    local_world_size: tl.constexpr,
    num_expert_per_rank: tl.constexpr,
    launch_cooperative_grid: tl.constexpr = False,
    has_input_off: tl.constexpr = False,
    profiling: tl.constexpr = False,
    num_warps: tl.constexpr = 32,
    MAJOR_ALIGN: tl.constexpr = 1,
):
    profiler = Profiler.create(
        profiler_buffer=profiler_buf,
        group_id=0,
        num_groups=1,
        is_leader=(tid(0) == 0),
        ENABLE_PROFILING=profiling,
    )
    thread_idx = tid(0)
    warp_id = thread_idx // 32
    lane_id = laneid()
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    global_warp_id = warp_id + pid * num_warps
    rank = dl.rank()
    world_size = dl.num_ranks()
    node_id = rank // local_world_size
    local_rank = rank % local_world_size
    nnodes = world_size // local_world_size
    nsplits = world_size * num_expert_per_rank
    splits_per_node = num_expert_per_rank * local_world_size
    max_tokens_per_node = splits_per_node * K
    elem_size = tl.constexpr(send_data.dtype.element_ty.primitive_bitwidth) // 8

    out_splits = out_splits_offsets
    out_source_offset = out_splits_offsets + nsplits
    node_exchange_signal = signal_pad
    node_transfer_signal = signal_pad + nnodes
    intra_exchange_barrier_signal = signal_pad + 2 * nnodes  # (local_world_size,)

    profiler = profiler.record(is_start=True, task_type=0)  # exchange data
    if pid == 0:
        offs = tl.arange(0, sig_pad_len)
        tl.store(signal_pad + offs, 0)
        libshmem_device.barrier_all_block()
        exchange_metadata_inter_node_kernel(
            in_splits_offsets,
            trans_in_splits_offsets,
            recv_splits_offsets,
            out_splits_offsets,
            per_node_offsets,
            node_exchange_signal,
            intra_exchange_barrier_signal,
            partial_sum_ptr,
            local_world_size,
            num_expert_per_rank,
            rank_in_row=rank_in_row,
            has_input_off=has_input_off,
            extra_barrier=False,
            num_warps=num_warps,
        )  # there is a intra-node barrier after exchange
        single_block_prefix_sum_kernel_scan_scan(
            out_splits,
            partial_sum_ptr,
            local_store_offset,  # as the output
            num_expert_per_rank if rank_in_row else world_size,  # row and col are reversed
            world_size if rank_in_row else num_expert_per_rank,
            num_warps=num_warps,
            MAJOR_ALIGN=MAJOR_ALIGN,
        )
    profiler = profiler.record(is_start=False, task_type=0)  # exchange data

    profiler = profiler.record(is_start=True, task_type=1)  # grid_barrier
    barrier_on_this_grid(grid_barrier_signal, launch_cooperative_grid)
    membar(scope="gl")
    profiler = profiler.record(is_start=False, task_type=1)  # grid_barrier

    if rank_in_row:
        rank_major_in_splits = in_splits_offsets
        rank_major_in_offsets = in_splits_offsets + nsplits
    else:
        rank_major_in_splits = trans_in_splits_offsets
        rank_major_in_offsets = trans_in_splits_offsets + nsplits

    profiler = profiler.record(is_start=True, task_type=2)  # inter-node nbi push
    for node_offset in range(1, nnodes):
        target_node = (node_id + node_offset) % nnodes
        target_rank = local_rank + target_node * local_world_size  # transfer at this rank
        for warp_task_id in range(global_warp_id, splits_per_node, num_warps * num_pid):
            tgt_local_rank = warp_task_id // num_expert_per_rank
            tgt_global_rank = tgt_local_rank + target_node * local_world_size  # the final rank to send to
            eid = warp_task_id % num_expert_per_rank
            read_off = ld(rank_major_in_offsets + tgt_global_rank * num_expert_per_rank + eid)
            splits = ld(rank_major_in_splits + tgt_global_rank * num_expert_per_rank + eid)
            write_off = ld(per_node_offsets + tgt_global_rank * num_expert_per_rank + eid)
            libshmem_device.putmem_nbi_warp(
                transfer_buffer + (node_id * max_tokens_per_node + write_off) * token_len_elem,
                send_data + read_off * token_len_elem,
                splits * token_len_elem * elem_size,
                target_rank,
            )

        membar(scope="gl")
        __syncthreads()
        libshmem_device.fence()

        if thread_idx == 0:
            libshmem_device.signal_op(
                node_transfer_signal + node_id,
                1,
                libshmem_device.NVSHMEM_SIGNAL_ADD,
                target_rank,
            )
    profiler = profiler.record(is_start=False, task_type=2)  # inter-node nbi push

    profiler = profiler.record(is_start=True, task_type=3)  # intra-node pull
    for warp_task_id in range(global_warp_id, splits_per_node * nnodes, num_warps * num_pid):
        node_offset = warp_task_id // splits_per_node
        idx_in_node = warp_task_id % splits_per_node
        src_send_node = (node_id - node_offset + nnodes) % nnodes

        tgt_local_rank = (idx_in_node // num_expert_per_rank + local_rank) % local_world_size
        tgt_global_rank = tgt_local_rank + src_send_node * local_world_size
        tgt_transfer_rank = tgt_local_rank + node_id * local_world_size
        eid = idx_in_node % num_expert_per_rank

        if src_send_node != node_id:
            remote_signal = dl.symm_at(node_transfer_signal, tgt_transfer_rank)
            if lane_id == 0:
                while ld_acquire(remote_signal + src_send_node, "sys") != num_pid:
                    pass
            __shfl_sync_i32(0xFFFFFFFF, tl.cast(0, tl.int32), lane_id)

        idx = eid * world_size + tgt_global_rank if rank_in_row else tgt_global_rank * num_expert_per_rank + eid
        st_off = ld(local_store_offset + idx)
        num_token = ld(out_splits + idx)
        source_off = ld(out_source_offset + idx)

        if node_offset > 0:
            pull_buffer = transfer_buffer
            rd_off = src_send_node * max_tokens_per_node + source_off
        else:
            pull_buffer = send_data
            rd_off = source_off

        libshmem_device.getmem_nbi_warp(
            recv_data + st_off * token_len_elem,
            pull_buffer + rd_off * token_len_elem,
            num_token * token_len_elem * elem_size,
            tgt_transfer_rank,
        )
        st(out_source_offset + idx, st_off)  # write back

    __syncthreads()
    profiler = profiler.record(is_start=False, task_type=3)


@dataclasses.dataclass
class AllToAllContext:
    rank: int
    world_size: int
    ne: int
    k: int
    token_len_elem: int
    local_world_size: int
    signal_pad_len: int

    # symmetric buffers
    split_meta_buf: torch.Tensor
    data_meta_buf: torch.Tensor
    signal_buf: torch.Tensor

    # local buffers
    partial_sum_buf: torch.Tensor  # may differ between cases

    # others
    need_profile: bool
    token_dtype: torch.dtype

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
        assert self.split_meta_buf.dtype == SPLIT_DTYPE, "split_meta_buf must be int32"
        assert self.world_size % self.local_world_size == 0
        self.node_id = self.rank // self.local_world_size
        self.nnodes = self.world_size // self.local_world_size
        self.nsplits = self.world_size * self.ne
        self.local_rank = self.rank % self.local_world_size
        assert self.split_meta_buf.numel() >= 9 * self.nsplits
        assert self.signal_buf.dtype == SIGNAL_DTYPE
        assert self.signal_buf.numel() >= 2 * self.nnodes + 2 * self.local_world_size, "signal_buf must be large enough"

        # splits and offsets
        self.in_splits_offsets = self.split_meta_buf[0:2 * self.nsplits]
        self.trans_in_splits_offsets = self.split_meta_buf[2 * self.nsplits:4 * self.nsplits]
        self.recv_splits_offsets = self.split_meta_buf[4 * self.nsplits:6 * self.nsplits]
        self.out_splits_offsets = self.split_meta_buf[6 * self.nsplits:8 * self.nsplits]
        self.per_node_offsets = self.split_meta_buf[8 * self.nsplits:9 * self.nsplits]
        self.local_store_offset = torch.empty((self.nsplits, ), dtype=SPLIT_DTYPE, device="cuda")

        # signals
        self.node_exchang_signal = self.signal_buf[0:self.nnodes]
        self.node_push_signal = self.signal_buf[self.nnodes:2 * self.nnodes]
        self.intra_exchange_barrier_signal = self.signal_buf[2 * self.nnodes:2 * self.nnodes + self.local_world_size]
        self.intra_push_barrier_signal = self.signal_buf[2 * self.nnodes + self.local_world_size:2 * self.nnodes +
                                                         2 * self.local_world_size]
        self.grid_barrier = torch.zeros((1, ), dtype=SIGNAL_DTYPE, device="cuda")

        # data buffers
        self.max_tokens_per_rank = self.k * self.nsplits
        self.max_token_elem = self.max_tokens_per_rank * self.token_len_elem  # for send/ transfer/ recv, the max num of tokens are the same
        assert self.data_meta_buf.numel() >= 3 * self.max_token_elem, "data_meta_buf must be large enough"
        self.send_data = self.data_meta_buf[0:self.max_token_elem]
        self.transfer_buffer = self.data_meta_buf[self.max_token_elem:2 * self.max_token_elem]
        self.recv_data = self.data_meta_buf[2 * self.max_token_elem:3 * self.max_token_elem]
        if not hasattr(self, 'profile_buf'):
            self.profile_buf = alloc_profiler_buffer(max_num_profile_slots=1000000)

    def dump_profiler_trace(self, info: str = None):
        profiler_dir = "./prof"
        os.makedirs(profiler_dir, exist_ok=True)
        name = f"all_to_all_dev_RANK_{self.rank}"
        if info is not None and info != "":
            name += f"_{info}"
        trace_file = os.path.join(profiler_dir, name)
        export_to_perfetto_trace(
            profiler_buffer=self.profile_buf,
            task_names=[
                "meta_exchange",
                "grid_barrier",
                "intermediate push",
                "pull local data",
            ],
            file_name=trace_file,
        )


def create_context(
    rank: int,
    world_size: int,
    ne: int,
    k: int,
    token_len_elem: int,
    token_dtype: torch.dtype,
    local_world_size: int = 8,
    need_profile: bool = True,
) -> AllToAllContext:

    nsplits = world_size * ne
    nnodes = world_size // local_world_size
    split_meta_buf = nvshmem_create_tensor((9 * nsplits, ), dtype=SPLIT_DTYPE)
    signal_len = triton.next_power_of_2(2 * nnodes + 2 * local_world_size)
    signal_buf = nvshmem_create_tensor((signal_len, ), dtype=SIGNAL_DTYPE)
    signal_buf.zero_()

    max_tokens_per_rank = k * nsplits
    max_token_elem = max_tokens_per_rank * token_len_elem
    data_meta_buf = nvshmem_create_tensor((3 * max_token_elem, ), dtype=token_dtype)

    WARP_TILE_SIZE = 32
    if ne > world_size:
        aligned_N = (world_size + WARP_TILE_SIZE - 1) // WARP_TILE_SIZE * WARP_TILE_SIZE
        partial_sum_buf = torch.empty((ne * aligned_N, ), dtype=SPLIT_DTYPE, device="cuda")
    else:
        aligned_N = (ne + WARP_TILE_SIZE - 1) // WARP_TILE_SIZE * WARP_TILE_SIZE
        partial_sum_buf = torch.empty((world_size * aligned_N, ), dtype=SPLIT_DTYPE, device="cuda")

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    return AllToAllContext(
        rank=rank,
        world_size=world_size,
        ne=ne,
        k=k,
        token_len_elem=token_len_elem,
        token_dtype=token_dtype,
        local_world_size=local_world_size,
        need_profile=need_profile,
        split_meta_buf=split_meta_buf,
        data_meta_buf=data_meta_buf,
        signal_buf=signal_buf,
        signal_pad_len=signal_len,
        partial_sum_buf=partial_sum_buf,
    )


# below is mostly the same with intranode version
def all_to_all_v_offset_op(ctx: AllToAllContext, rank_in_row: bool, input: torch.Tensor = None,
                           output: torch.Tensor = None, in_splits: torch.Tensor = None, in_offset: torch.Tensor = None,
                           copy_to_symm_buffer: bool = True,  # False: data already in comm buffer
                           return_tensor: bool = True, major_align: int = 1, grid_size: int = None,
                           has_input_offset: bool = False, profiling: bool = False):
    if copy_to_symm_buffer:
        assert in_splits is not None
        assert in_splits.dim() == 2
        assert in_splits.dtype == ctx.in_splits_offsets.dtype
        assert in_splits.numel() == ctx.nsplits
        ctx.in_splits_offsets[:ctx.nsplits].copy_(in_splits.view(-1))

        if has_input_offset:
            assert in_offset is not None
            assert in_offset.dtype == ctx.in_splits_offsets.dtype
            assert in_offset.numel() == ctx.nsplits
            ctx.in_splits_offsets[ctx.nsplits:].copy_(in_offset.view(-1))

        assert input is not None
        assert input.dtype == ctx.token_dtype
        if not input.is_contiguous():
            raise Warning("input tensor is not contiguous")
        input_flat = input.contiguous().view(-1)
        ctx.send_data[:input_flat.numel()].copy_(input_flat)

    # ctx.signal_buf.zero_()
    # nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    num_sms = min(ctx.ne, ctx.nsplits // 32)  # max num_warps=32 per block
    grid = (num_sms, ) if grid_size is None else (grid_size, )
    all_to_all_v_2d_inter_node_kernel[grid](
        profiler_buf=ctx.profile_buf,
        in_splits_offsets=ctx.in_splits_offsets,
        trans_in_splits_offsets=ctx.trans_in_splits_offsets,
        recv_splits_offsets=ctx.recv_splits_offsets,
        out_splits_offsets=ctx.out_splits_offsets,
        per_node_offsets=ctx.per_node_offsets,
        local_store_offset=ctx.local_store_offset,
        partial_sum_ptr=ctx.partial_sum_buf,
        signal_pad=ctx.signal_buf,
        grid_barrier_signal=ctx.grid_barrier,
        send_data=ctx.send_data,
        transfer_buffer=ctx.transfer_buffer,
        recv_data=ctx.recv_data,
        K=ctx.k,
        sig_pad_len=ctx.signal_pad_len,
        rank_in_row=rank_in_row,
        token_len_elem=ctx.token_len_elem,
        local_world_size=ctx.local_world_size,
        num_expert_per_rank=ctx.ne,
        **launch_cooperative_grid_options(),
        has_input_off=has_input_offset,
        profiling=profiling,
        num_warps=32,
        MAJOR_ALIGN=major_align,
    )

    if return_tensor:
        torch.cuda.current_stream().synchronize()
        last_split_size = ctx.out_splits_offsets[ctx.nsplits - 1]
        num_token_aligned = ctx.local_store_offset[-1] + last_split_size
        num_elem = num_token_aligned * ctx.token_len_elem
        if copy_to_symm_buffer:
            assert output is not None
            assert output.dtype == ctx.token_dtype
            if not output.is_contiguous():
                raise Warning("output tensor is not contiguous")
            output_flat = output.contiguous().view(-1)
            output_flat[:num_elem].copy_(ctx.recv_data.view(-1)[:num_elem])
        ret = output if copy_to_symm_buffer else ctx.recv_data[:num_elem].view(-1, ctx.token_len_elem)
        return ret, num_token_aligned
    else:
        return None, None


def all_to_all_vdev_2d(ctx: AllToAllContext, input: torch.Tensor = None, output: torch.Tensor = None,
                       in_splits: torch.Tensor = None, return_tensor: bool = True, copy_to_symm_buffer: bool = True,
                       major_align: int = 1, grid_size: int = None, profiling: bool = False):
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
        return_tensor=return_tensor,
    )


def all_to_all_vdev_2d_offset(ctx: AllToAllContext, input: torch.Tensor = None, output: torch.Tensor = None,
                              in_splits: torch.Tensor = None, in_offset: torch.Tensor = None,
                              return_tensor: bool = True, copy_to_symm_buffer: bool = True, grid_size: int = None,
                              profiling: bool = False):
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
        return_tensor=return_tensor,
    )
