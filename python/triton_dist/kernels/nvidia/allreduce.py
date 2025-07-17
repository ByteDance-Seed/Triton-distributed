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
import dataclasses
import functools
import math
import warnings
from typing import List

import torch
from torch import Tensor
from cuda import cudart

import triton
import triton.language as tl
from triton_dist.kernels.allreduce import AllReduceMethod
from triton.language.extra.cuda.language_extra import (__syncthreads, atomic_cas, load_v2_b64, multimem_st_b64, ntid,
                                                       pack_b32_v2, st_v4_b32, tid, multimem_ld_reduce_v4)
from triton.language.extra.cuda.utils import num_warps
from triton_dist.kernels.nvidia.common_ops import (add_v8_bf16, barrier_on_this_grid, get_flat_tid, load_b64_v2)
from triton_dist.kernels.nvidia.reduce_scatter import copy_continuous_kernel, kernel_ring_reduce_tma, kernel_ring_reduce_non_tma
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import (CUDA_CHECK, NVSHMEM_SIGNAL_DTYPE, get_device_property, is_tma_support,
                               nvshmem_barrier_all_on_stream, nvshmem_create_tensors, nvshmem_free_tensor_sync,
                               requires, is_nvshmem_multimem_supported)

SIGNAL_TARGET = 1
MAX_DOUBLE_TREE_BLOCKS = 1024  # for double tree op


def workspace_bytes_per_in_byte(world_size, method: AllReduceMethod) -> int:
    if method in [AllReduceMethod.OneShot, AllReduceMethod.OneShot_TMA]:
        return world_size
    if method in [AllReduceMethod.TwoShot]:
        return 2
    if method in [AllReduceMethod.OneShot_Multimem, AllReduceMethod.TwoShot_Multimem, AllReduceMethod.DoubleTree]:
        return 1
    raise Exception(f"Unknown allreduce method {method}")
    return 0  # to make lint happy


def get_max_chunk_nbytes(workspace_nbytes, world_size, method: AllReduceMethod) -> int:
    return workspace_nbytes // workspace_bytes_per_in_byte(world_size, method)


def _memcpy_async_unsafe(dst: torch.Tensor, src: torch.Tensor, nbytes, stream: torch.cuda.Stream):
    """ no check dtype. no check device/host. no check tensor size. no check contiguous.
    """
    err, = cudart.cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), nbytes, cudart.cudaMemcpyKind.cudaMemcpyDefault,
                                  stream.cuda_stream)
    CUDA_CHECK(err)


@dataclasses.dataclass
class AllReduceContext:
    workspace_nbytes: int
    rank: int
    world_size: int
    local_world_size: int

    # comm buffer
    symm_scatter_bufs: List[Tensor]
    symm_scatter_buf: Tensor = dataclasses.field(init=False)

    # barrier bufs
    symm_signals: List[Tensor]
    symm_signal: Tensor = dataclasses.field(init=False)

    # use this to sync all grids
    grid_barrier: torch.Tensor = dataclasses.field(init=False)

    local_rank: int = dataclasses.field(init=False)
    node_id: int = dataclasses.field(init=False)
    nnodes: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        assert self.world_size % self.local_world_size == 0
        self.nnodes = self.world_size // self.local_world_size
        self.grid_barrier = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.symm_signal = self.symm_signals[self.local_rank]
        self.symm_scatter_buf = self.symm_scatter_bufs[self.local_rank]

    def finalize(self):
        nvshmem_free_tensor_sync(self.symm_scatter_buf)
        nvshmem_free_tensor_sync(self.symm_signal)

    def get_symm_list(self):
        return self.symm_scatter_bufs, self.symm_signals


def create_allreduce_ctx(
    workspace_nbytes,
    rank,
    world_size,
    local_world_size,
) -> AllReduceContext:
    """
    symmetric buffer requirement for input tensor x with x.nbytes = N.

    method                 |  symmetric buffer size
    double_tree            | N (for scatter)
    one_shot/one_shot_tma  | N * world_size (for all-gather)
    two_shot               | N + N (N for scatter, N for output)
    one_shot_multimem      | N (for ld_reduce)
    two_shot_multimem      | N (for ld_reduce/scatter)
    two_shot_multimem_st   | N + N (N for scatter, N for output)
    """
    local_rank = rank % local_world_size
    symm_scatter_bufs = nvshmem_create_tensors((workspace_nbytes, ), torch.int8, rank, local_world_size)
    symm_signals = nvshmem_create_tensors((MAX_DOUBLE_TREE_BLOCKS * world_size, ), NVSHMEM_SIGNAL_DTYPE, rank,
                                          local_world_size)
    symm_signal = symm_signals[local_rank]
    symm_signal.fill_(0)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    ctx = AllReduceContext(workspace_nbytes=workspace_nbytes, rank=rank, world_size=world_size,
                           local_world_size=local_world_size, symm_scatter_bufs=symm_scatter_bufs,
                           symm_signals=symm_signals)
    return ctx


def _run_straggler(ctx, straggler_option):
    if straggler_option:
        rank, cyles = straggler_option
        if rank == ctx.rank:
            torch.cuda._sleep(cyles)


@functools.lru_cache()
def get_tree_parent_and_children(N, rank):
    """
    Calculate parent and children nodes for a given rank in two complementary binary trees
    Integrated from https://github.com/cchan/tccl/blob/main/tree.py

    Args:
        N (int): Total number of nodes, must be a power of 2
        rank (int): Node identifier

    Returns:
        tuple: (tree_a_parent, tree_a_left, tree_a_right, tree_b_parent, tree_b_left, tree_b_right)
               -1 indicates no parent or child node exists
    """

    # Check if N is a power of 2
    assert (N & (N - 1)) == 0, "len(ranks) is not a power of 2!"

    # Build first tree
    height = int(math.log2(N))
    tree_a = []
    prev_row = None
    for h in range(height):
        if prev_row is None:
            row = [i for i in range(N) if i % 2 != 0]
        else:
            row = []
            for i in range(0, len(prev_row), 2):
                row.append((prev_row[i] + prev_row[i + 1]) // 2)
        tree_a = row + tree_a
        prev_row = row
    tree_a = [0] + tree_a

    # Build second complementary tree
    tree_b = list(map(lambda e: e - 1, tree_a))
    tree_b[0] = N - 1

    # Calculate parent and children in tree_a
    rank_index_a = tree_a.index(rank)
    if rank_index_a == 0:
        parent_a = -1
        left_child_a = tree_a[rank_index_a + 1]
        right_child_a = -1
    else:
        parent_idx_a = rank_index_a // 2
        left_child_idx_a = rank_index_a * 2
        right_child_idx_a = left_child_idx_a + 1

        parent_a = tree_a[parent_idx_a]
        left_child_a = tree_a[left_child_idx_a] if left_child_idx_a < len(tree_a) else -1
        right_child_a = tree_a[right_child_idx_a] if right_child_idx_a < len(tree_a) else -1

    # Calculate parent and children in tree_b
    rank_index_b = tree_b.index(rank)
    if rank_index_b == 0:
        parent_b = -1
        left_child_b = tree_b[rank_index_b + 1]
        right_child_b = -1
    else:
        parent_idx_b = rank_index_b // 2
        left_child_idx_b = rank_index_b * 2
        right_child_idx_b = left_child_idx_b + 1

        parent_b = tree_b[parent_idx_b]
        left_child_b = tree_b[left_child_idx_b] if left_child_idx_b < len(tree_b) else -1
        right_child_b = tree_b[right_child_idx_b] if right_child_idx_b < len(tree_b) else -1

    return parent_a, left_child_a, right_child_a, parent_b, left_child_b, right_child_b


@triton.jit(do_not_specialize=["rank", "world_size"])
def allreduce_double_tree_intra_node_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    output_ptr,
    tree0_parent,
    tree0_child0,
    tree0_child1,
    tree1_parent,
    tree1_child0,
    tree1_child1,
    numel,
    rank,
    world_size,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
):
    tl.static_assert(output_ptr.dtype.element_ty == tl.bfloat16, "Only supports dtype=Bf16")
    block_id = (tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0) + tl.program_id(1) * tl.num_programs(0) +
                tl.program_id(0))
    pid = tl.program_id(axis=0)
    thread_idx = get_flat_tid()
    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.int64))
    output_ptr = output_ptr.to(tl.pointer_type(tl.int64))
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    block_start = pid * BLOCK_SIZE

    if pid < tl.num_programs(axis=0) // 2:
        tree_child0 = tree0_child0
        tree_child1 = tree0_child1
        tree_parent = tree0_parent
    else:
        tree_child0 = tree1_child0
        tree_child1 = tree1_child1
        tree_parent = tree1_parent

    if tree_child0 != -1:
        local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint64))
        wait_addrs = local_signal_pad_addr + block_id * world_size + tree_child0
        if thread_idx == 0:  # wait
            while (atomic_cas(wait_addrs, 1, 0, "sys", "acquire") != 1):
                pass
        __syncthreads()
    if tree_child1 != -1:
        local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint64))
        wait_addrs = local_signal_pad_addr + block_id * world_size + tree_child1
        if thread_idx == 0:  # wait
            while (atomic_cas(wait_addrs, 1, 0, "sys", "acquire") != 1):
                pass
        __syncthreads()

    while block_start < (numel // NUMEL_PER_THREAD):

        offsets = (block_start + tl.arange(0, BLOCK_SIZE)) * 2
        mask = block_start + tl.arange(0, BLOCK_SIZE) < numel // NUMEL_PER_THREAD

        acc_hi = tl.zeros((BLOCK_SIZE, ), tl.int64)
        acc_lo = tl.zeros((BLOCK_SIZE, ), tl.int64)
        if tree_child0 != -1:
            buffer_ptr = tl.load(buffer_ptrs + tree_child0).to(tl.pointer_type(tl.int64))
            (hi, lo) = load_b64_v2(buffer_ptr + offsets, mask=mask)
            (acc_hi, acc_lo) = add_v8_bf16(acc_hi, acc_lo, hi, lo)
        # All the computation in else Region will be eliminated by DSE pass, but it's necessary in triton's frontend parser now
        else:
            buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.int64))
            hi = tl.zeros((BLOCK_SIZE, ), tl.int64)
            lo = tl.zeros((BLOCK_SIZE, ), tl.int64)
            (acc_hi, acc_lo) = (acc_hi, acc_lo)

        if tree_child1 != -1:
            buffer_ptr = tl.load(buffer_ptrs + tree_child1).to(tl.pointer_type(tl.int64))
            (hi, lo) = load_b64_v2(buffer_ptr + offsets, mask=mask)
            (acc_hi, acc_lo) = add_v8_bf16(acc_hi, acc_lo, hi, lo)
        else:
            buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.int64))
            hi = tl.zeros((BLOCK_SIZE, ), tl.int64)
            lo = tl.zeros((BLOCK_SIZE, ), tl.int64)
            (acc_hi, acc_lo) = (acc_hi, acc_lo)

        buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.int64))
        (hi, lo) = load_b64_v2(buffer_ptr + offsets, mask=mask)
        (acc_hi, acc_lo) = add_v8_bf16(acc_hi, acc_lo, hi, lo)
        tl.store(buffer_ptr + offsets + 0, acc_hi, mask=mask)
        tl.store(buffer_ptr + offsets + 1, acc_lo, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    if tree_parent != -1:
        remote_signal_pad_addr = tl.load(signal_pad_ptrs + tree_parent).to(tl.pointer_type(tl.uint64))
        send_addr = remote_signal_pad_addr + block_id * world_size + rank
        __syncthreads()
        if thread_idx == 0:  # send
            while atomic_cas(send_addr, 0, 1, "sys", "release") != 0:
                pass

        local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint64))
        wait_addrs = local_signal_pad_addr + block_id * world_size + tree_parent
        if thread_idx == 0:  # wait
            while (atomic_cas(wait_addrs, 1, 0, "sys", "acquire") != 1):
                pass
        __syncthreads()

    block_start = pid * BLOCK_SIZE
    while block_start < (numel // NUMEL_PER_THREAD):
        # Each thread processes 128 bits. Since Triton doesn't yet natively
        # support 128-bit dtypes, we achieve this by having each thread process
        # two 64-bit elements.

        offsets = (block_start + tl.arange(0, BLOCK_SIZE)) * 2
        mask = block_start + tl.arange(0, BLOCK_SIZE) < numel // NUMEL_PER_THREAD

        if tree_parent != -1:
            buffer_ptr = tl.load(buffer_ptrs + tree_parent).to(tl.pointer_type(tl.int64))
            (hi, lo) = load_b64_v2(buffer_ptr + offsets, mask=mask)
            buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.int64))
            tl.store(buffer_ptr + offsets + 0, hi, mask=mask)
            tl.store(buffer_ptr + offsets + 1, lo, mask=mask)
            tl.store(output_ptr + offsets + 0, hi, mask=mask)
            tl.store(output_ptr + offsets + 1, lo, mask=mask)
        else:
            buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.int64))
            (hi, lo) = load_b64_v2(buffer_ptr + offsets, mask=mask)
            tl.store(output_ptr + offsets + 0, hi, mask=mask)
            tl.store(output_ptr + offsets + 1, lo, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    if tree_child0 != -1:
        remote_signal_pad_addr = tl.load(signal_pad_ptrs + tree_child0).to(tl.pointer_type(tl.uint64))
        send_addr = remote_signal_pad_addr + block_id * world_size + rank
        __syncthreads()
        if thread_idx == 0:  # send
            while atomic_cas(send_addr, 0, 1, "sys", "release") != 0:
                pass
    if tree_child1 != -1:
        remote_signal_pad_addr = tl.load(signal_pad_ptrs + tree_child1).to(tl.pointer_type(tl.uint64))
        send_addr = remote_signal_pad_addr + block_id * world_size + rank
        __syncthreads()
        if thread_idx == 0:  # send
            while atomic_cas(send_addr, 0, 1, "sys", "release") != 0:
                pass


@triton.jit(do_not_specialize=["rank"])
def allreduce_one_shot_push_intra_node_kernel(
    input_ptr,
    output_ptr,
    symm_signal_ptr,
    symm_buffer_ptr,
    grid_barrier_ptr,
    rank,
    world_size: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    thread_idx = tid(0)
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)
    elem_size = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8
    # cast as input_ptr dtype
    symm_buffer_ptr = tl.cast(symm_buffer_ptr, input_ptr.dtype)

    # reset signals and then barrier all
    if pid == 0:
        offs = tl.arange(0, world_size)
        tl.store(symm_signal_ptr + offs, 0)
        libshmem_device.barrier_all_block()

    barrier_on_this_grid(grid_barrier_ptr)

    # all-gather with push
    for peer in range(pid, world_size, num_pid):
        libshmem_device.putmem_signal_block(
            symm_buffer_ptr + n_elements * rank,
            input_ptr,
            n_elements * elem_size,
            symm_signal_ptr + rank,
            1,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )

    if thread_idx < world_size:
        libshmem_device.signal_wait_until(symm_signal_ptr + thread_idx, libshmem_device.NVSHMEM_CMP_EQ, 1)
    __syncthreads()

    kernel_ring_reduce_non_tma(
        symm_buffer_ptr,
        output_ptr,
        n_elements,
        rank,
        world_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit(do_not_specialize=["rank"])
def allreduce_one_shot_tma_push_intra_node_kernel(
    M,
    N,
    input_ptr,
    output_ptr,
    symm_signal_ptr,
    symm_recv_ptr,
    grid_barrier_ptr,
    rank,
    world_size: tl.constexpr,
    n_elements,
    BLOCK_SIZE_REDUCE_M: tl.constexpr,
    BLOCK_SIZE_REDUCE_N: tl.constexpr,
):
    tl.static_assert(input_ptr.dtype == output_ptr.dtype)
    symm_recv_ptr = tl.cast(symm_recv_ptr, input_ptr.dtype)

    thread_idx = tid(0)
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)
    elem_size = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8

    # reset signals and then barrier all
    if pid == 0:
        offs = tl.arange(0, world_size)
        tl.store(symm_signal_ptr + offs, 0)
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_barrier_ptr)

    # all-gather with push
    for peer in range(pid, world_size, num_pid):
        libshmem_device.putmem_signal_nbi_block(
            symm_recv_ptr + n_elements * rank,
            input_ptr,
            n_elements * elem_size,
            symm_signal_ptr + rank,
            1,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )

    if thread_idx < world_size:
        libshmem_device.signal_wait_until(symm_signal_ptr + thread_idx, libshmem_device.NVSHMEM_CMP_EQ, 1)
    __syncthreads()

    # reduce with TMA
    kernel_ring_reduce_tma(
        symm_recv_ptr,
        output_ptr,
        M,
        N,
        rank,
        world_size,
        BLOCK_SIZE_M=BLOCK_SIZE_REDUCE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_REDUCE_N,
    )


@triton.jit(do_not_specialize=["rank"])
def allreduce_two_shot_push_intra_node_kernel(
    input_ptr,
    symm_out_ptr,
    symm_signal_ptr,
    grid_barrier_ptr,
    # output
    out_ptr,
    rank,
    world_size: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    thread_idx = tid(0)
    pid = tl.program_id(0)
    elem_size = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8
    elem_per_rank = tl.cdiv(n_elements, world_size)
    symm_out_ptr = tl.cast(symm_out_ptr, input_ptr.dtype)
    symm_recv_ptr = symm_out_ptr + n_elements

    # reset signals and then barrier all
    if pid == 0:
        offs = tl.arange(0, world_size * 2)
        tl.store(symm_signal_ptr + offs, 0)
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_barrier_ptr)

    # this is an allgather with push
    if pid < world_size:
        peer = (rank + pid + 1) % world_size
        libshmem_device.putmem_signal_nbi_block(
            symm_recv_ptr + rank * elem_per_rank,
            input_ptr + peer * elem_per_rank,
            elem_per_rank * elem_size,
            symm_signal_ptr + rank,
            1,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )

    if thread_idx < world_size:
        libshmem_device.signal_wait_until(symm_signal_ptr + thread_idx, libshmem_device.NVSHMEM_CMP_EQ, 1)
    __syncthreads()
    libshmem_device.fence()

    # reduce and store to symm_out_ptr: ready to send again.
    kernel_ring_reduce_non_tma(
        symm_recv_ptr,
        symm_out_ptr + elem_per_rank * rank,
        elem_per_rank,
        rank,
        world_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    barrier_on_this_grid(grid_barrier_ptr)

    symm_signal_ptr += world_size
    # the second shot:
    if pid < world_size - 1:
        peer = (rank + pid + 1) % world_size
        libshmem_device.putmem_signal_nbi_block(
            symm_out_ptr + rank * elem_per_rank,
            symm_out_ptr + rank * elem_per_rank,
            elem_per_rank * elem_size,
            symm_signal_ptr + rank,
            1,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )

    if thread_idx < world_size and thread_idx != rank:
        libshmem_device.signal_wait_until(symm_signal_ptr + thread_idx, libshmem_device.NVSHMEM_CMP_EQ, 1)
    __syncthreads()
    libshmem_device.fence()

    # copy to output
    if out_ptr:
        copy_continuous_kernel(symm_out_ptr, out_ptr, n_elements, BLOCK_SIZE)


@triton.jit(do_not_specialize=["rank"])
def allreduce_two_shot_multimem_st_intra_node_kernel(
    input_ptr,
    symm_out_ptr,
    symm_signal_ptr,
    grid_barrier_ptr,
    # output
    out_ptr,
    rank,
    world_size,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    thread_idx = tid(0)
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)
    num_total_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    elem_size = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8
    tiles_per_rank = num_total_tiles // world_size
    elem_per_rank = tl.cdiv(n_elements, world_size)

    symm_out_ptr = tl.cast(symm_out_ptr, tl.pointer_type(input_ptr.dtype))
    symm_recv_ptr = symm_out_ptr + n_elements

    # reset signals and then barrier all
    if pid == 0:
        offs = tl.arange(0, world_size)
        tl.store(symm_signal_ptr + offs, 0)
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_barrier_ptr)

    # this is a allgather with push
    if pid < world_size - 1:
        peer = (rank + pid + 1) % world_size
        libshmem_device.putmem_signal_nbi_block(
            symm_recv_ptr + rank * elem_per_rank,
            input_ptr + peer * elem_per_rank,
            elem_per_rank * elem_size,
            symm_signal_ptr + rank,
            1,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )

    if thread_idx < world_size and thread_idx != rank:
        libshmem_device.signal_wait_until(symm_signal_ptr + thread_idx, libshmem_device.NVSHMEM_CMP_EQ, 1)

    __syncthreads()
    for i in range(pid, tiles_per_rank, num_pid):
        tile_id = tiles_per_rank * rank + i
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        acc = tl.load(input_ptr + offsets, mask=mask)
        for stage in range(1, world_size):
            from_rank = (rank - stage + world_size) % world_size
            buffer_offsets = (tiles_per_rank * from_rank + i) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            buffer_mask = buffer_offsets < n_elements
            val = tl.load(symm_recv_ptr + buffer_offsets, mask=buffer_mask)
            acc += val
        tl.store(symm_out_ptr + offsets, acc, mask=mask)
        __syncthreads()
        block_N = ntid(axis=0)
        ptr = tl.cast(symm_out_ptr + tile_id * BLOCK_SIZE, tl.pointer_type(tl.int8))
        mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, ptr)
        for n in range(thread_idx, BLOCK_SIZE * elem_size // 16, block_N):
            val0, val1 = load_v2_b64(ptr + n * 16)
            multimem_st_b64(mc_ptr + n * 16, val0)
            multimem_st_b64(mc_ptr + n * 16 + 8, val1)

    barrier_on_this_grid(grid_barrier_ptr)
    if pid == 0:
        libshmem_device.barrier_all_block()
    if out_ptr:
        barrier_on_this_grid(grid_barrier_ptr)
        copy_continuous_kernel(symm_out_ptr, out_ptr, n_elements, BLOCK_SIZE)


@triton.jit
def allreduce_one_shot_multimem_intra_node_kernel(symm_in_ptr, out_ptr, elems, grid_barrier_ptr):
    symm_in_ptr = tl.cast(symm_in_ptr, out_ptr.dtype)
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)

    if pid == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_barrier_ptr)

    data_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_in_ptr)
    VEC_SIZE = 128 // tl.constexpr(symm_in_ptr.dtype.element_ty.primitive_bitwidth)

    thread_idx = tid(axis=0)
    block_dim = num_warps() * 32
    for idx in range(thread_idx + block_dim * pid, elems // VEC_SIZE, num_pid * block_dim):
        val0, val1, val2, val3 = multimem_ld_reduce_v4(data_mc_ptr + idx * VEC_SIZE)
        st_v4_b32(out_ptr + idx * VEC_SIZE, val0, val1, val2, val3)

    barrier_on_this_grid(grid_barrier_ptr)
    if pid == 0:
        libshmem_device.barrier_all_block()


@triton.jit(do_not_specialize=["rank"])
def allreduce_two_shot_multimem_intra_node_kernel(symm_ptr, grid_barrier_ptr, elems, output_ptr, rank, world_size,
                                                  BLOCK_SIZE: tl.constexpr):
    elems_per_rank = elems // world_size
    # each rank do all-reduce for elems_per_rank
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)
    data_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_ptr)
    data_mc_ptr += rank * elems_per_rank
    VEC_SIZE = 128 // tl.constexpr(symm_ptr.dtype.element_ty.primitive_bitwidth)
    if pid == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_barrier_ptr)

    thread_idx = tid(0)
    block_dim = num_warps() * 32
    for idx in range(thread_idx + block_dim * pid, elems_per_rank // VEC_SIZE, num_pid * block_dim):
        val0, val1, val2, val3 = multimem_ld_reduce_v4(data_mc_ptr + idx * VEC_SIZE)
        multimem_st_b64(data_mc_ptr + idx * VEC_SIZE, pack_b32_v2(val0, val1))
        multimem_st_b64(data_mc_ptr + idx * VEC_SIZE + VEC_SIZE // 2, pack_b32_v2(val2, val3))

    barrier_on_this_grid(grid_barrier_ptr)
    if pid == 0:
        libshmem_device.barrier_all_block()

    # copy to non-symmetric buffer if needed
    if output_ptr:
        barrier_on_this_grid(grid_barrier_ptr)
        # copy to output
        nblocks = tl.cdiv(elems, BLOCK_SIZE)
        pid = tl.program_id(axis=0)
        npid = tl.num_programs(axis=0)
        for i in range(pid, nblocks, npid):
            offs = tl.arange(0, BLOCK_SIZE) + BLOCK_SIZE * i
            x = tl.load(symm_ptr + offs, offs < elems)
            tl.store(output_ptr + offs, x, mask=offs < elems)


def allreduce_one_shot_push_intra_node(
    ctx: AllReduceContext,
    x: Tensor,
    output: Tensor,
    straggler_option=None,
    max_sm: int = -1,
    num_warps: int = 32,
):
    """ Performs a reduction operation using a simple, native implementation.
    This op uses the most basic reduction algorithm, which is only efficient
    for small data size. However, it causes symmetric memory capacity issues
    when processing large inputs.

    """
    assert x.is_cuda and x.is_contiguous()
    assert output.is_cuda and output.is_contiguous()
    assert x.dtype == output.dtype and x.shape == output.shape, f"x.dtype({x.dtype}) == output.dtype({output.dtype}) and x.shape({x.shape}) == output.shape({output.shape})"
    # one_shot requires a lot of memory
    assert x.nbytes <= ctx.workspace_nbytes // ctx.world_size

    block_size = num_warps * 32 * 16 // x.itemsize
    num_elem = x.numel()
    num_tiles = triton.cdiv(num_elem, block_size)
    _run_straggler(ctx, straggler_option)
    # the grid can't be too large: cooperative_launchs
    if max_sm > 0:
        num_tiles = min(max_sm, num_tiles)
    num_tiles = min(get_device_property().multi_processor_count - 4, num_tiles)
    allreduce_one_shot_push_intra_node_kernel[(num_tiles, )](
        x,
        output,
        ctx.symm_signal,
        ctx.symm_scatter_buf,
        ctx.grid_barrier,
        ctx.rank,
        ctx.world_size,
        num_elem,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        launch_cooperative_grid=True,
    )
    return output


def allreduce_two_shot_push_intra_node(
    ctx: AllReduceContext,
    x: Tensor,
    output: Tensor,
    straggler_option=None,
    max_sm: int = -1,
    num_warps: int = 32,
):
    """ this function requires x.nbytes as

    """
    assert x.is_cuda and x.is_contiguous()
    assert output.is_cuda and output.is_contiguous()
    assert x.dtype == output.dtype and x.nbytes == output.nbytes
    # two_shot requires x.nbytes for symmetric scatter and x.nbytes for symmetric output
    assert x.nbytes <= ctx.workspace_nbytes // 2

    block_size = num_warps * 32 * 16 // x.itemsize
    num_elem = x.numel()
    num_tiles = triton.cdiv(num_elem, block_size)
    _run_straggler(ctx, straggler_option)
    # the grid can't be too large: cooperative_launchs
    if max_sm > 0:
        num_tiles = min(max_sm, num_tiles)
    num_tiles = max(ctx.world_size, min(get_device_property().multi_processor_count, num_tiles))
    allreduce_two_shot_push_intra_node_kernel[(num_tiles, )](
        x,
        ctx.symm_scatter_buf,
        ctx.symm_signal,
        ctx.grid_barrier,
        # output
        output,
        ctx.rank,
        ctx.world_size,
        num_elem,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        launch_cooperative_grid=True,
    )
    return output


def allreduce_one_shot_tma_push_intra_node(ctx: AllReduceContext, x: Tensor, output: Tensor, straggler_option=None,
                                           max_sm: int = -1, num_warps: int = 32,  # dummy arg
                                           ):
    """ P2P sends full-length copy of local tensor to other ranks, then uses TMA reduce.

    Args:
        straggler_option (_type_, optional): Simulates the straggler. Defaults to None.
    """
    assert x.is_cuda and x.is_contiguous()
    assert output.is_cuda and output.is_contiguous()
    assert x.nbytes == output.nbytes and x.dtype == output.dtype
    # check the workspace size
    assert x.nbytes <= ctx.workspace_nbytes // ctx.world_size
    num_elem = x.numel()
    # heuristic choose M, N, and block_size_m, block_size_n with a reshape (numel, ) => (M_per_rank * N // 64, N) with N % 64 == 0
    assert num_elem * x.itemsize % 32 == 0, "TMA requires 32-byte alignment."
    for alignment in [256, 128, 64, 32]:
        if num_elem * x.itemsize % alignment == 0:
            N = alignment // x.itemsize
            M = num_elem // N
            break

    block_size_n = 64
    block_size_m = 128

    _run_straggler(ctx, straggler_option)
    num_sms = 32 if max_sm < 0 else max_sm
    allreduce_one_shot_tma_push_intra_node_kernel[(num_sms, )](
        M,
        N,
        x,
        output,
        ctx.symm_signal,
        ctx.symm_scatter_buf,
        ctx.grid_barrier,
        ctx.rank,
        ctx.world_size,
        num_elem,
        BLOCK_SIZE_REDUCE_M=block_size_m,
        BLOCK_SIZE_REDUCE_N=block_size_n,
        num_warps=num_warps,
        launch_cooperative_grid=True,
    )
    return output


@requires(is_nvshmem_multimem_supported)
def allreduce_one_shot_multimem_intra_node(
    ctx: AllReduceContext,
    x: torch.Tensor,
    output: torch.Tensor,
    straggler_option=None,
    num_warps=32,
    max_sm=-1,
):
    """ One-shot reduces symm buffer using the multimem.ld_reduce instruction.

    Raises:
        NotImplementedError: Currently supports bf16 and float32.
    """
    assert x.is_cuda and x.is_contiguous() and output.is_cuda and output.is_contiguous()
    assert x.dtype == output.dtype and x.numel() == output.numel()
    # TODO(houqi.1993) int is not implemented yet
    assert x.dtype in [torch.float, torch.bfloat16, torch.float16]
    assert x.nbytes <= ctx.workspace_nbytes

    _run_straggler(ctx, straggler_option)
    # TODO(houqi.1993) fuse into triton kernel for better performance
    _memcpy_async_unsafe(ctx.symm_scatter_buf, x, x.nbytes, torch.cuda.current_stream())
    num_blocks = 4 if max_sm < 0 else max_sm
    allreduce_one_shot_multimem_intra_node_kernel[(num_blocks, )](
        ctx.symm_scatter_buf,
        output,
        x.numel(),
        ctx.grid_barrier,
        num_warps=num_warps,
        launch_cooperative_grid=True,
    )
    return output


def allreduce_double_tree_intra_node(
    ctx: AllReduceContext,
    x: Tensor,
    output: Tensor,
    max_sm: int = -1,
    num_warps: int = 32,
    straggler_option=None,
):
    """
    Args:
        max_sm (int, optional): Max grid size. Defaults to -1.
        num_warps (int, optional): The This operator does not specify a block size directly;
                                instead, it calculates the block size by multiplying num_warps by 32. Defaults to 8.
        straggler_option (_type_, optional): _description_. Defaults to None.

    """
    world_size, local_world_size = ctx.world_size, ctx.local_world_size
    rank = ctx.rank

    block_size = num_warps * 32
    NUMEL_PER_THREAD = 8

    assert x.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert x.is_cuda and x.is_contiguous()
    assert (x.numel() % NUMEL_PER_THREAD == 0), "The number of elements must be 128-bit aligned."

    if max_sm > 0:
        num_blocks = min(
            triton.cdiv(triton.cdiv(x.numel(), NUMEL_PER_THREAD), block_size),
            max_sm,
        )
    else:
        num_blocks = triton.cdiv(triton.cdiv(x.numel(), NUMEL_PER_THREAD), block_size)
    assert world_size == local_world_size, "Inter-node is not currently supported"
    assert ctx.symm_signal.numel() >= world_size * num_blocks, "Alloc at least one signal for each block"
    assert x.nbytes <= ctx.workspace_nbytes, f"Workspace size {ctx.workspace_nbytes} is not enough for {x.nbytes} bytes"

    tree0_parent, tree0_child0, tree0_child1, tree1_parent, tree1_child0, tree1_child1 = get_tree_parent_and_children(
        world_size, rank)

    def list2tensor_ptr(l: List[Tensor]):
        ptrs = []
        for t in l:
            ptrs.append(t.data_ptr())
        ptrs = torch.tensor(ptrs, ).cuda()
        return ptrs

    buf_ptrs = list2tensor_ptr(ctx.symm_scatter_bufs)  # construct a C-style pointer array
    sig_ptrs = list2tensor_ptr(ctx.symm_signals)

    _memcpy_async_unsafe(ctx.symm_scatter_buf, x, x.nbytes, torch.cuda.current_stream())
    ctx.symm_signal.zero_()

    _run_straggler(ctx, straggler_option)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    allreduce_double_tree_intra_node_kernel[(num_blocks, 1, 1)](
        buf_ptrs,
        sig_ptrs,
        output,
        tree0_parent,
        tree0_child0,
        tree0_child1,
        tree1_parent,
        tree1_child0,
        tree1_child1,
        numel=x.numel(),
        rank=rank,
        world_size=world_size,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=NUMEL_PER_THREAD,
        num_warps=num_warps,
    )
    return output


# TODO(houqi.1993) support very small shape. now only support 16 Byte aligned
@requires(is_nvshmem_multimem_supported)
def allreduce_two_shot_multimem_intra_node(
    ctx: AllReduceContext,
    x: torch.Tensor,
    output: torch.Tensor,
    straggler_option=None,
    num_warps=8,
    max_sm=-1,
):
    """ Reduces corresponding shard using PTX instruction multimem.ld_reduce then multimem.st

        Notes:
            - if `output` parameter is None, this function will return the symmetric buffer directly.
            - this function requires x.nbytes of symmetric buffer.
    """
    elems = x.numel()
    elems_per_rank = elems // ctx.world_size
    assert x.is_cuda and x.is_contiguous()
    assert elems % ctx.world_size == 0
    assert x.dtype in [torch.float, torch.bfloat16, torch.float16]
    assert elems_per_rank * x.itemsize % 16 == 0, "use vectorize of 128bit mulitmem.ld_reduce/st"
    assert x.nbytes <= ctx.workspace_nbytes

    _memcpy_async_unsafe(ctx.symm_scatter_buf, x, x.nbytes, torch.cuda.current_stream())
    _run_straggler(ctx, straggler_option)

    #  each thread got 4 float or 8 bfloat16
    block_size = num_warps * 32 * 16 // x.itemsize
    ngrids = 4 if max_sm < 0 else max_sm  # TODO(houqi.1993) maybe H20 need more than 4
    ngrids = min(triton.cdiv(elems_per_rank, block_size), ngrids)

    allreduce_two_shot_multimem_intra_node_kernel[(ngrids, )](
        ctx.symm_scatter_buf.view(x.dtype),
        ctx.grid_barrier,
        elems,
        output,
        ctx.rank,
        ctx.world_size,
        block_size,
        num_warps=num_warps,
        launch_cooperative_grid=True,
    )
    return output


@requires(is_nvshmem_multimem_supported)
def allreduce_two_shot_multimem_st_intra_node(
    ctx: AllReduceContext,
    x: Tensor,
    output: torch.Tensor,
    straggler_option=None,
    num_warps: int = 32,
    max_sm: int = -1,
):
    """ This function is DEPRECATED. Please use `allreduce_two_shot_multimem_intra_node` instead.

    this function requires the most symmetric buffer:
        x.nbytes for receive from other ranks.
        x.nbytes for symmetric output.

    Notes:
        - Out-of-place op, results are stored in symm buffer (see `return ctx.scatter_buf`)
        - Uses PTX instruction multimem.st only.
        - NVLink communication cannot be fused with reduce, leading to lower bandwidth utilization
        - Relies on nvshmem for per-block data transfer, which limits scalability when using multiple ranks per peer.
          Especially problematic for high-bandwidth hardware like H20.
    """
    warnings.warn(
        "This function is deprecated and will be removed in a future version. "
        "Use 'allreduce_two_shot_multimem_intra_node' instead for better performance.", DeprecationWarning,
        stacklevel=2)

    assert x.nbytes <= ctx.workspace_nbytes // 2
    assert x.is_cuda and x.is_contiguous

    rank, world_size = ctx.rank, ctx.world_size
    num_elem = x.numel()
    block_size = 16 * 32 * num_warps // x.itemsize  # a wavefront of warps
    num_tiles = triton.cdiv(triton.cdiv(num_elem, world_size), block_size)
    _run_straggler(ctx, straggler_option)
    if max_sm > 0:
        num_tiles = min(max_sm, num_tiles)
    grid_size = max(min(get_device_property(0).multi_processor_count, num_tiles), world_size - 1)

    allreduce_two_shot_multimem_st_intra_node_kernel[(grid_size, )](
        x,
        ctx.symm_scatter_buf,
        ctx.symm_signal,
        ctx.grid_barrier,
        output,
        rank,
        world_size,
        num_elem,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        launch_cooperative_grid=True,
    )
    return output if output is not None else ctx.symm_scatter_buf[:num_elem].view_as(x)


def get_auto_allreduce_method(nbytes):
    if is_nvshmem_multimem_supported():
        if nbytes > 64 * 1024:
            return AllReduceMethod.TwoShot_Multimem
        else:
            return AllReduceMethod.OneShot_Multimem

    # TODO(houqi.1993) re-determine nbytes
    if is_tma_support() and nbytes < 16 * 1024:
        return AllReduceMethod.OneShot_TMA

    # TODO(houqi.1993) no two-shot without TMA implementation
    return AllReduceMethod.OneShot


def all_reduce(
    x: Tensor,
    output: Tensor,
    method: AllReduceMethod,
    ctx: AllReduceContext,
    max_sm: int = -1,
    straggler_option=None,
):
    """ TODO(houqi.1993) if use multiple chunks, does not support CUDAGraph.
    """
    method = method or get_auto_allreduce_method(x.nbytes)
    # method naming: allreduce_${algo}_${arch}_${impl}_${protocol}_${extra}
    #  algo: double_tree / one_shot / two_shot / ring
    #  arch: arch related such as multicast/tma/null
    #  impl: push / pull
    #  protocol: null / LL. currently only null supported
    #  extra: intra_node / inter_node (or null)
    op_handle = {
        AllReduceMethod.OneShot: allreduce_one_shot_push_intra_node,
        AllReduceMethod.TwoShot: allreduce_two_shot_push_intra_node,
        AllReduceMethod.OneShot_TMA: allreduce_one_shot_tma_push_intra_node,
        AllReduceMethod.OneShot_Multimem: allreduce_one_shot_multimem_intra_node,
        AllReduceMethod.DoubleTree: allreduce_double_tree_intra_node,
        AllReduceMethod.TwoShot_Multimem: allreduce_two_shot_multimem_intra_node,
        AllReduceMethod.TwoShot_Multimem_ST: allreduce_two_shot_multimem_st_intra_node,
    }[method]

    nbytes_per_chunk = ctx.workspace_nbytes // workspace_bytes_per_in_byte(ctx.world_size, method)
    nchunks = triton.cdiv(x.nbytes, nbytes_per_chunk)
    elems_per_chunk = nbytes_per_chunk // x.itemsize

    if nchunks > 1:
        assert not torch.cuda.is_current_stream_capturing(
        ), "allreduce does not support CUDAGraph if use multiple chunks"

    if output is None:
        output = torch.empty_like(x)

    for n in range(nchunks):
        op_handle(
            ctx=ctx,
            x=x.flatten()[elems_per_chunk * n:elems_per_chunk * (n + 1)],
            output=output.flatten()[elems_per_chunk * n:elems_per_chunk * (n + 1)],
            max_sm=max_sm,
            num_warps=32,
            straggler_option=straggler_option,
        )

    return output
