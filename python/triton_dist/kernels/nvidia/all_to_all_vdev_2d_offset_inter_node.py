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
import ctypes
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.cuda.language_extra import (__syncthreads, ld, st, tid, membar, ld_acquire)
from triton_dist.kernels.nvidia.common_ops import barrier_on_this_grid, barrier_all_intra_node_atomic_cas_block
import torch
from typing import Union
from math import ceil
import dataclasses
import triton_dist
from triton_dist.utils import nvshmem_create_tensor, nvshmem_free_tensor_sync, nvshmem_barrier_all_on_stream, launch_cooperative_grid_options
from triton_dist.tools.profiler import (Profiler, alloc_profiler_buffer, export_to_perfetto_trace)
from triton_dist.kernels.nvidia.all_to_all_vdev_2d_offset import single_block_prefix_sum_kernel_scan_scan

SPLIT_DTYPE = torch.int32
SIGNAL_DTYPE = torch.int64


@triton.jit
def transpose_kernel(ptr_x, ptr_y, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    n_block_m = tl.cdiv(M, BLOCK_M)
    n_block_n = tl.cdiv(N, BLOCK_N)

    for bid in range(n_block_m * n_block_n):
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


@triton_dist.jit(do_not_specialize=["num_expert_per_rank", "stage"])
def all_to_all_v_2d_inter_node_kernel(
    profiler_buf,
    in_splits_offsets,  # (2, nsplits), symmetric
    trans_in_splits_offsets,  # (2, nsplits), symmetric
    recv_splits_offsets,  # (2, nsplits), symmetric
    out_splits_offsets,  # (2, nsplits), symmetric
    per_node_offsets,  # (nsplits,), symmetric
    local_store_offset,  # (nsplits,), local
    partial_sum_ptr,  # (num_tiles,), local
    send_recv_num_tokens,  # (2 * num_stages,), pinned memory
    stage,
    total_stages: tl.constexpr,
    signal_pad,  # (2 * nnodes + local_world_size,), symmetric
    grid_barrier_signal,  # (1,), symmetric / local
    input,  # max_tokens_per_rank * token_len
    output,
    transfer_buffer,  # max_tokens_per_rank * token_len
    max_num_token: tl.constexpr,  # receiver's view, max num of tokens may receive from all senders
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
    COPY_BLOCK_SIZE: tl.constexpr = 1,
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
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    rank = dl.rank()
    world_size = dl.num_ranks()
    node_id = rank // local_world_size
    local_rank = rank % local_world_size
    nnodes = world_size // local_world_size
    nsplits = world_size * num_expert_per_rank
    splits_per_node = num_expert_per_rank * local_world_size
    data_elem_size = tl.constexpr(input.dtype.element_ty.primitive_bitwidth) // 8
    splits_elem_size = tl.constexpr(in_splits_offsets.dtype.element_ty.primitive_bitwidth) // 8
    token_len_bytes_i64 = tl.cast(token_len_elem * data_elem_size, tl.int64)
    token_len_elem_i64 = tl.cast(token_len_elem, tl.int64)

    in_splits = in_splits_offsets
    in_offsets = in_splits_offsets + nsplits  # local output offset
    out_splits = out_splits_offsets
    out_source_offset = out_splits_offsets + nsplits
    trans_in_splits = trans_in_splits_offsets
    per_node_offsets = trans_in_splits_offsets + nsplits
    recv_splits = recv_splits_offsets  # received out_splits
    recv_offsets = recv_splits_offsets + nsplits  # per-node offsets
    exchange_meta_signal = signal_pad
    per_node_data_signal = signal_pad + 1
    intra_barrier_signal = signal_pad + 1 + nnodes
    rank_major_in_splits = in_splits

    profiler = profiler.record(is_start=True, task_type=0)  # build meta
    if pid == 0:
        offs = tl.arange(0, sig_pad_len)
        tl.store(signal_pad + offs, 0)
        last_stage = (stage + total_stages - 1) % total_stages
        tl.store(send_recv_num_tokens + last_stage * 2 + tl.arange(0, 2), -1)
        libshmem_device.barrier_all_block()
        if not rank_in_row:
            transpose_kernel(
                in_splits,
                trans_in_splits,
                num_expert_per_rank,
                world_size,
                BLOCK_M=16,
                BLOCK_N=32,
            )
            rank_major_in_splits = trans_in_splits
        __syncthreads()
        for nid in range(nnodes):
            single_block_prefix_sum_kernel_scan_scan(
                rank_major_in_splits + nid * splits_per_node,
                partial_sum_ptr,
                per_node_offsets + nid * splits_per_node,  # as the output
                local_world_size,
                num_expert_per_rank,
                num_warps=num_warps,
            )
            __syncthreads()
        __syncthreads()
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
    profiler = profiler.record(is_start=False, task_type=0)  # build meta

    profiler = profiler.record(is_start=True, task_type=1)  # grid_barrier
    barrier_on_this_grid(grid_barrier_signal, launch_cooperative_grid)
    membar(scope="gl")
    profiler = profiler.record(is_start=False, task_type=1)  # grid_barrier

    profiler = profiler.record(is_start=True, task_type=2)  # exchange
    if pid == 0:
        for tgt_rank in range(warp_id, world_size, num_warps):
            libshmem_device.putmem_signal_nbi_warp(
                recv_splits + rank * num_expert_per_rank,
                rank_major_in_splits + tgt_rank * num_expert_per_rank,
                num_expert_per_rank * splits_elem_size,
                exchange_meta_signal,
                1,
                libshmem_device.NVSHMEM_SIGNAL_ADD,
                tgt_rank,
            )
            libshmem_device.putmem_signal_nbi_warp(
                recv_offsets + rank * num_expert_per_rank,
                per_node_offsets + tgt_rank * num_expert_per_rank,
                num_expert_per_rank * splits_elem_size,
                exchange_meta_signal,
                1,
                libshmem_device.NVSHMEM_SIGNAL_ADD,
                tgt_rank,
            )
    else:
        for tgt_node_offset in range(nnodes):
            tgt_node = (node_id + tgt_node_offset) % nnodes
            tgt_relay = local_rank + tgt_node * local_world_size
            for idx in range(pid - 1, splits_per_node, num_pid - 1):
                tgt_local_rank = idx // num_expert_per_rank
                eid = idx % num_expert_per_rank
                tgt_rank = tgt_local_rank + tgt_node * local_world_size
                split_id = tgt_rank * num_expert_per_rank + eid if rank_in_row else eid * world_size + tgt_rank
                split_val = ld(in_splits + split_id)
                offset_val = ld(in_offsets + split_id)
                per_node_off_val = ld(per_node_offsets + tgt_rank * num_expert_per_rank + eid)
                libshmem_device.putmem_nbi_block(
                    transfer_buffer +
                    tl.cast(node_id * max_num_token + per_node_off_val, tl.int64) * token_len_elem_i64,
                    input + tl.cast(offset_val, tl.int64) * token_len_elem_i64,
                    tl.cast(split_val, tl.int64) * token_len_bytes_i64,
                    tgt_relay,
                )
            membar(scope="gl")
            __syncthreads()
            libshmem_device.fence()
            if thread_idx == 0:
                libshmem_device.signal_op(
                    per_node_data_signal + node_id,
                    1,
                    libshmem_device.NVSHMEM_SIGNAL_ADD,
                    tgt_relay,
                )
    profiler = profiler.record(is_start=False, task_type=2)  # exchange

    profiler = profiler.record(is_start=True, task_type=3)  # wait metadata + compute offsets
    if pid == 0:
        while ld_acquire(exchange_meta_signal, "sys") != 2 * world_size:
            pass
        if rank_in_row:
            transpose_kernel(
                recv_splits,
                out_splits,
                world_size,
                num_expert_per_rank,
                BLOCK_M=16,
                BLOCK_N=32,
            )
        else:
            with dl.simt_exec_region() as (simt_tid, threads_per_block):
                for idx in range(simt_tid, nsplits, threads_per_block):
                    out_splits_val = ld(recv_splits + idx)
                    st(out_splits + idx, out_splits_val)
        __syncthreads()
        single_block_prefix_sum_kernel_scan_scan(
            out_splits,
            partial_sum_ptr,
            out_source_offset,  # as the output
            num_expert_per_rank if rank_in_row else world_size,  # row and col are reversed
            world_size if rank_in_row else num_expert_per_rank,
            num_warps=num_warps,
            MAJOR_ALIGN=MAJOR_ALIGN,
        )
        __syncthreads()
        if thread_idx == 0:
            num_send_tokens = ld(in_offsets + nsplits - 1, scope='gpu', semantic='acquire') + ld(
                in_splits + nsplits - 1, scope='gpu', semantic='acquire')
            num_send_tokens = tl.cast(num_send_tokens, tl.int32)
            num_recv_tokens = ld(out_source_offset + nsplits - 1, scope='gpu', semantic='acquire') + ld(
                out_splits + nsplits - 1, scope='gpu', semantic='acquire')
            num_recv_tokens = tl.cast(num_recv_tokens, tl.int32)
            st(send_recv_num_tokens + stage * 2, num_send_tokens, scope='gpu', semantic='release')
            st(send_recv_num_tokens + stage * 2 + 1, num_recv_tokens, scope='gpu', semantic='release')
        barrier_all_intra_node_atomic_cas_block(local_rank, rank, local_world_size, intra_barrier_signal)
    barrier_on_this_grid(grid_barrier_signal, launch_cooperative_grid)
    membar(scope="sys")
    profiler = profiler.record(is_start=False, task_type=3)  # wait metadata + compute offsets

    profiler = profiler.record(is_start=True, task_type=4)  # pull local data
    num_push_pids = num_pid - 1
    for src_node_offset in range(nnodes):
        src_node = (node_id + src_node_offset) % nnodes
        for idx in range(pid, splits_per_node, num_pid):
            tgt_local_rank = idx // num_expert_per_rank
            eid = idx % num_expert_per_rank
            tgt_rank = tgt_local_rank + src_node * local_world_size
            tgt_transfer_rank = tgt_local_rank + node_id * local_world_size
            split_id = eid * world_size + tgt_rank if rank_in_row else tgt_rank * num_expert_per_rank + eid

            if src_node != node_id:
                remote_signal = dl.symm_at(per_node_data_signal, tgt_transfer_rank)
                if thread_idx == 0:
                    while ld_acquire(remote_signal + src_node, "sys") < num_push_pids:
                        pass
                __syncthreads()

            store_offset_val = ld(out_source_offset + split_id)
            split_val = ld(out_splits + split_id)
            per_node_off_val = ld(recv_offsets + tgt_rank * num_expert_per_rank + eid)

            libshmem_device.getmem_block(
                output + tl.cast(store_offset_val, tl.int64) * token_len_elem_i64,
                transfer_buffer + tl.cast(src_node * max_num_token + per_node_off_val, tl.int64) * token_len_elem_i64,
                tl.cast(split_val, tl.int64) * token_len_bytes_i64,
                tgt_transfer_rank,
            )
    profiler = profiler.record(is_start=False, task_type=4)  # pull local data

    profiler = profiler.record(is_start=True, task_type=5)  # barrier
    barrier_on_this_grid(grid_barrier_signal, launch_cooperative_grid)
    if pid == 0:
        libshmem_device.barrier_all_block()
    profiler = profiler.record(is_start=False, task_type=5)  # barrier


@dataclasses.dataclass
class AllToAllContext:
    rank: int
    world_size: int
    ne: int
    k: int
    max_tokens_per_rank: int  # sender's view, max num of tokens per rank
    max_recv_tokens_per_rank: int  # receiver's view, max num of tokens may receive from all senders
    token_len_elem: int
    local_world_size: int
    signal_pad_len: int

    # symmetric buffers
    split_meta_buf: torch.Tensor
    data_meta_buf: torch.Tensor
    signal_buf: torch.Tensor

    # local buffers
    partial_sum_buf: torch.Tensor  # may differ between cases
    num_send_recv_tokens: torch.Tensor  # pinned memory

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

        # signal layout (managed by kernel via signal_pad offsets):
        #   [0]                  : exchange_meta_signal (1 element)
        #   [1 .. nnodes]        : per_node_data_signal (nnodes elements)
        #   [1+nnodes .. ]       : intra_barrier_signal (local_world_size elements)
        self.grid_barrier = torch.zeros((1, ), dtype=SIGNAL_DTYPE, device="cuda")

        # data buffers
        self.max_recv_elem_per_rank = self.max_recv_tokens_per_rank * self.token_len_elem
        assert self.data_meta_buf.numel() >= (self.nnodes +
                                              1) * self.max_recv_elem_per_rank, "data_meta_buf must be large enough"
        self.send_data = self.data_meta_buf[self.nnodes * self.max_recv_elem_per_rank:]
        self.transfer_buffer = self.data_meta_buf

        if not hasattr(self, 'profile_buf'):
            self.profile_buf = alloc_profiler_buffer(max_num_profile_slots=1000000)

        self.total_stages = self.num_send_recv_tokens.numel() // 2
        self.current_stage = 0

    def get_stage(self) -> int:
        ret = self.current_stage
        self.current_stage = (self.current_stage + 1) % self.total_stages
        return ret

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
                "build_meta",  # task_type=0
                "grid_barrier",  # task_type=1
                "exchange (meta+data)",  # task_type=2
                "wait metadata + compute offsets",  # task_type=3
                "pull local data (per-node)",  # task_type=4
                "barrier",  # task_type=5
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
    overflow_factor: Union[int, float] = None,
    max_token_num_per_rank: int = None,
) -> AllToAllContext:
    """
    max_token_num_per_rank: view of senders' max num of tokens per rank
    max_recv_tokens_per_rank: view of receivers' max num of tokens may receive from all senders, 
            if overflow_factor is not None, it is the max num of tokens may receive from all senders
    """

    nsplits = world_size * ne
    nnodes = world_size // local_world_size
    split_meta_buf = nvshmem_create_tensor((9 * nsplits, ), dtype=SPLIT_DTYPE)
    signal_len = triton.next_power_of_2(2 * nnodes + 2 * local_world_size)
    signal_buf = nvshmem_create_tensor((signal_len, ), dtype=SIGNAL_DTYPE)
    signal_buf.zero_()

    # overflow worst case: one rank receives all data from all senders
    overflow_factor = world_size if overflow_factor is None else max(1, overflow_factor)
    if max_token_num_per_rank is None:
        assert k >= 1, "k must be no less than 1 if max_token_num_per_rank is not provided"
        max_tokens_per_rank = k * nsplits
    else:
        assert max_token_num_per_rank > 0, "max_token_num_per_rank must be greater than 0"
        max_tokens_per_rank = max_token_num_per_rank
    max_recv_tokens_per_rank = ceil(max_tokens_per_rank * overflow_factor)
    max_recv_elem_per_rank = max_recv_tokens_per_rank * token_len_elem
    data_elem = (nnodes + 1) * max_recv_elem_per_rank  # send(1) + recv(1) + transfer(nnodes)
    data_meta_buf = nvshmem_create_tensor((data_elem, ), dtype=token_dtype)

    num_send_recv_tokens = torch.empty((8, ), dtype=torch.int32, device="cpu", pin_memory=True)
    num_send_recv_tokens.fill_(-1)

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
        max_recv_tokens_per_rank=max_recv_tokens_per_rank,
        max_tokens_per_rank=max_tokens_per_rank,
        token_len_elem=token_len_elem,
        token_dtype=token_dtype,
        local_world_size=local_world_size,
        need_profile=need_profile,
        split_meta_buf=split_meta_buf,
        data_meta_buf=data_meta_buf,
        signal_buf=signal_buf,
        signal_pad_len=signal_len,
        partial_sum_buf=partial_sum_buf,
        num_send_recv_tokens=num_send_recv_tokens,
    )


# below is mostly the same with intranode version
def all_to_all_v_offset_op(ctx: AllToAllContext, rank_in_row: bool, input: torch.Tensor = None,
                           output: torch.Tensor = None, in_splits: torch.Tensor = None, in_offset: torch.Tensor = None,
                           copy_to_symm_buffer: bool = True,  # False: data already in comm buffer
                           return_stat: bool = True, major_align: int = 1, grid_size: int = None,
                           has_input_offset: bool = False, profiling: bool = False):
    assert in_splits.dtype == ctx.split_meta_buf.dtype
    assert in_splits.numel() == ctx.nsplits
    assert input is not None and input.dtype == ctx.token_dtype
    assert input.is_contiguous()
    if input.dim() > 1:
        assert input.size(1) == ctx.token_len_elem
    if output is not None:
        assert output.dtype == ctx.token_dtype
        assert output.is_contiguous()
    if copy_to_symm_buffer:
        assert output is not None

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
    copy_block_size = (1024 * 32) // input.element_size()
    copy_block_size = 1 << (copy_block_size.bit_length() - 1)
    current_stage = ctx.get_stage()
    all_to_all_v_2d_inter_node_kernel[grid](
        profiler_buf=ctx.profile_buf,
        in_splits_offsets=ctx.in_splits_offsets,
        trans_in_splits_offsets=ctx.trans_in_splits_offsets,
        recv_splits_offsets=ctx.recv_splits_offsets,
        out_splits_offsets=ctx.out_splits_offsets,
        per_node_offsets=ctx.per_node_offsets,
        local_store_offset=ctx.local_store_offset,
        partial_sum_ptr=ctx.partial_sum_buf,
        send_recv_num_tokens=ctx.num_send_recv_tokens,
        stage=current_stage,
        total_stages=ctx.total_stages,
        signal_pad=ctx.signal_buf,
        grid_barrier_signal=ctx.grid_barrier,
        input=ctx.send_data,
        transfer_buffer=ctx.transfer_buffer,
        output=output,
        max_num_token=ctx.max_recv_tokens_per_rank,
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
        COPY_BLOCK_SIZE=copy_block_size,
    )

    base_ptr = ctx.num_send_recv_tokens[current_stage * 2].data_ptr()
    elem_bytes = ctx.num_send_recv_tokens.itemsize
    while ctypes.c_int32.from_address(base_ptr).value == -1:  # num recv token
        pass
    while ctypes.c_int32.from_address(base_ptr + elem_bytes).value == -1:  # num recv token
        pass
    send_token_num = ctypes.c_int32.from_address(base_ptr).value
    recv_token_num = ctypes.c_int32.from_address(base_ptr + elem_bytes).value
    print(f"[DEBUG] Rank {ctx.rank} dynamic get send token num: {send_token_num}, recv token num: {recv_token_num}")
    return output, send_token_num, recv_token_num


def all_to_all_vdev_2d(ctx: AllToAllContext, input: torch.Tensor = None, output: torch.Tensor = None,
                       in_splits: torch.Tensor = None, return_stat: bool = True, copy_to_symm_buffer: bool = True,
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
        return_stat=return_stat,
    )


def all_to_all_vdev_2d_offset(ctx: AllToAllContext, input: torch.Tensor = None, output: torch.Tensor = None,
                              in_splits: torch.Tensor = None, in_offset: torch.Tensor = None, return_stat: bool = True,
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
        return_stat=return_stat,
    )
