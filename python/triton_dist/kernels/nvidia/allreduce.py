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
from typing import Optional

import torch
from cuda import cudart

import triton
import triton.language as tl
from triton.language.extra.cuda import libnvshmem_device
from triton_dist.kernels.allreduce import AllReduceMethod
from triton.language.extra.cuda.language_extra import (__syncthreads, load_v2_b64, multimem_st_b64, ntid, pack_b32_v2,
                                                       st_v4_b32, tid, multimem_ld_reduce_v4)
from triton.language.extra.cuda.utils import num_warps
from triton_dist.kernels.nvidia.common_ops import barrier_all_intra_node_atomic_cas_block, barrier_all_intra_node_non_atomic_block, barrier_on_this_grid
from triton_dist.kernels.nvidia.reduce_scatter import copy_continuous_kernel, kernel_ring_reduce_tma, kernel_ring_reduce_non_tma
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import (CUDA_CHECK, NVSHMEM_SIGNAL_DTYPE, supports_p2p_native_atomic, get_device_property,
                               launch_cooperative_grid_options, nvshmem_barrier_all_on_stream, nvshmem_create_tensor,
                               nvshmem_free_tensor_sync, requires, is_nvshmem_multimem_supported, has_tma)

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
    err, = cudart.cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream.cuda_stream)
    CUDA_CHECK(err)


@dataclasses.dataclass
class AllReduceContext:
    workspace_nbytes: int
    rank: int
    world_size: int
    local_world_size: int

    # comm buffer
    symm_scatter_buf: torch.Tensor

    # barrier bufs
    symm_signal: torch.Tensor

    # faster barrier-all
    phase: int = 0
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

    def finalize(self):
        nvshmem_free_tensor_sync(self.symm_scatter_buf)
        nvshmem_free_tensor_sync(self.symm_signal)


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
    symm_scatter_buf = nvshmem_create_tensor((workspace_nbytes, ), torch.int8)
    symm_signal = nvshmem_create_tensor((MAX_DOUBLE_TREE_BLOCKS * world_size, ), NVSHMEM_SIGNAL_DTYPE)
    symm_signal.fill_(0)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    ctx = AllReduceContext(workspace_nbytes=workspace_nbytes, rank=rank, world_size=world_size,
                           local_world_size=local_world_size, symm_scatter_buf=symm_scatter_buf,
                           symm_signal=symm_signal)
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
    symm_input_ptr,
    symm_signal_ptr,
    output_ptr,
    tree0_parent: tl.constexpr,
    tree0_child0: tl.constexpr,
    tree0_child1: tl.constexpr,
    tree1_parent: tl.constexpr,
    tree1_child0: tl.constexpr,
    tree1_child1: tl.constexpr,
    numel,
    rank,
    world_size,
    CHUNK_SIZE: tl.constexpr,  # data in a chunk size share a signal
    BLOCK_SIZE: tl.constexpr,
):
    symm_input_ptr = tl.cast(symm_input_ptr, output_ptr.dtype)
    chunk_id = tl.program_id(axis=0)
    pid = tl.program_id(axis=0)
    thread_idx = tid(axis=0)

    if pid < tl.num_programs(axis=0) // 2:
        tree_child0 = tree0_child0
        tree_child1 = tree0_child1
        tree_parent = tree0_parent
    else:
        tree_child0 = tree1_child0
        tree_child1 = tree1_child1
        tree_parent = tree1_parent

    NULLPTR: tl.constexpr = tl.cast(tl.cast(0, tl.int64), symm_input_ptr.dtype, bitcast=True)
    if tree_child0 != -1:
        tree_child0_input_ptr = libnvshmem_device.remote_ptr(symm_input_ptr, tree_child0)
        tree_child0_input_ptr = tl.multiple_of(tree_child0_input_ptr, 16)
    else:
        tree_child0_input_ptr = NULLPTR  # not used
    if tree_child1 != -1:
        tree_child1_input_ptr = libnvshmem_device.remote_ptr(symm_input_ptr, tree_child1)
        tree_child1_input_ptr = tl.multiple_of(tree_child1_input_ptr, 16)
    else:
        tree_child1_input_ptr = NULLPTR  # not used
    if tree_parent != -1:
        tree_parent_input_ptr = libnvshmem_device.remote_ptr(symm_input_ptr, tree_parent)
        tree_parent_input_ptr = tl.multiple_of(tree_parent_input_ptr, 16)
    else:
        tree_parent_input_ptr = NULLPTR  # not used

    SIGNAL_VALUE = chunk_id + 1  # more robust value for debug only

    if tree_child0 != -1:
        if thread_idx == 0:  # wait for child ready
            libshmem_device.signal_wait_until(symm_signal_ptr + chunk_id * world_size + tree_child0,
                                              libshmem_device.NVSHMEM_CMP_EQ, SIGNAL_VALUE)
        __syncthreads()
    if tree_child1 != -1:
        if thread_idx == 0:  # wait for child ready
            libshmem_device.signal_wait_until(symm_signal_ptr + chunk_id * world_size + tree_child1,
                                              libshmem_device.NVSHMEM_CMP_EQ, SIGNAL_VALUE)
        __syncthreads()

    has_child = (tree_child0 != -1) or (tree_child1 != -1)  # `has_child` mean it's node leaf node
    has_parent = tree_parent != -1  # `has_parent` means it's not root node

    chunk_offset = chunk_id * CHUNK_SIZE
    # UP-pass: reduce data from children
    if has_child:  # skip load/store if has no child
        for n in range(CHUNK_SIZE // BLOCK_SIZE):
            offsets = chunk_offset + n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            x = tl.load(symm_input_ptr + offsets, mask=mask)
            if tree_child0 != -1:
                x += tl.load(tree_child0_input_ptr + offsets, mask=mask)
            if tree_child1 != -1:
                x += tl.load(tree_child1_input_ptr + offsets, mask=mask)

            tl.store(symm_input_ptr + offsets, x, mask=mask)

    # UP-pass: reduce data from children
    if has_parent:
        __syncthreads()
        if thread_idx == 0:  # notify parent that i'm ready for pull
            libshmem_device.signal_op(symm_signal_ptr + chunk_id * world_size + rank, SIGNAL_VALUE,
                                      libshmem_device.NVSHMEM_SIGNAL_SET, tree_parent)

    # DOWN-PASS: wait for parent
    if has_parent:
        if thread_idx == 0:  # wait for parent ready
            libshmem_device.signal_wait_until(symm_signal_ptr + chunk_id * world_size + tree_parent,
                                              libshmem_device.NVSHMEM_CMP_EQ, SIGNAL_VALUE)
        __syncthreads()

    # DOWN-PASS: copy data from parent
    for n in range(CHUNK_SIZE // BLOCK_SIZE):
        offsets = chunk_offset + n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        if has_parent:
            x = tl.load(tree_parent_input_ptr + offsets, mask=mask)
            if has_child:
                tl.store(symm_input_ptr + offsets, x, mask=mask)
            tl.store(output_ptr + offsets, x, mask=mask)
        else:
            x = tl.load(symm_input_ptr + offsets, mask=mask)
            tl.store(output_ptr + offsets, x, mask=mask)

    # DOWN-PASS: signal children
    if tree_child0 != -1:
        __syncthreads()
        if thread_idx == 0:  # notify child that parent is ready for pull
            libshmem_device.signal_op(symm_signal_ptr + chunk_id * world_size + rank, SIGNAL_VALUE,
                                      libshmem_device.NVSHMEM_SIGNAL_SET, tree_child0)
    if tree_child1 != -1:
        __syncthreads()
        if thread_idx == 0:  # notify child that parent is ready for pull
            libshmem_device.signal_op(symm_signal_ptr + chunk_id * world_size + rank, SIGNAL_VALUE,
                                      libshmem_device.NVSHMEM_SIGNAL_SET, tree_child1)


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
    use_cooperative: tl.constexpr,
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

    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

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
    use_cooperative: tl.constexpr,
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
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

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
    use_cooperative: tl.constexpr,
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
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

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
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

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
    rank,
    world_size,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    use_cooperative: tl.constexpr,
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
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

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

    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)
    if pid == 0:
        libshmem_device.barrier_all_block()


@triton.heuristics({"HAS_ATOMIC_CAS": lambda args: supports_p2p_native_atomic()})
@triton.jit(do_not_specialize=["rank", "phase"])
def allreduce_one_shot_multimem_intra_node_kernel(
    in_ptr,
    symm_in_ptr,
    symm_signal_ptr,
    out_ptr,
    elems,
    grid_barrier_ptr,
    rank,
    phase,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # used for memcpy
    HAS_ATOMIC_CAS: tl.constexpr,
    use_cooperative: tl.constexpr,
):
    symm_in_ptr = tl.cast(symm_in_ptr, out_ptr.dtype)
    pid = tl.program_id(0)
    thread_idx = tid(axis=0)
    num_pid = tl.num_programs(axis=0)

    # memcpy from in_ptr to symm_in_ptr: save a kernel call
    if in_ptr:
        nblocks = tl.cdiv(elems, BLOCK_SIZE)
        for bid in range(pid, nblocks, num_pid):
            offs = tl.arange(0, BLOCK_SIZE) + bid * BLOCK_SIZE
            mask = offs < elems
            val = tl.load(in_ptr + offs, mask=mask)
            tl.store(symm_in_ptr + offs, val, mask=mask)

    if pid == 0:
        # this runs a little faster than libshmem_device.barrier_all_block
        if HAS_ATOMIC_CAS:
            barrier_all_intra_node_atomic_cas_block(rank, rank, WORLD_SIZE, symm_signal_ptr)
        else:
            barrier_all_intra_node_non_atomic_block(rank, rank, WORLD_SIZE, symm_signal_ptr, phase)

    if num_pid != 1:
        barrier_on_this_grid(grid_barrier_ptr, use_cooperative)  # zero_ signal here

    data_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_in_ptr)
    VEC_SIZE = 128 // tl.constexpr(symm_in_ptr.dtype.element_ty.primitive_bitwidth)

    block_dim = num_warps() * 32
    for idx in range(thread_idx + block_dim * pid, elems // VEC_SIZE, num_pid * block_dim):
        val0, val1, val2, val3 = multimem_ld_reduce_v4(data_mc_ptr + idx * VEC_SIZE)
        st_v4_b32(out_ptr + idx * VEC_SIZE, val0, val1, val2, val3)

    # wait for all ranks to finish
    if num_pid != 1:
        barrier_on_this_grid(grid_barrier_ptr, use_cooperative)
    if pid == 0:
        symm_signal_ptr += WORLD_SIZE * 2
        if thread_idx < WORLD_SIZE and thread_idx != rank:
            libshmem_device.signal_op(symm_signal_ptr + rank, phase, libshmem_device.NVSHMEM_SIGNAL_SET, thread_idx)
            libshmem_device.signal_wait_until(symm_signal_ptr + thread_idx, libshmem_device.NVSHMEM_CMP_EQ, phase)
        __syncthreads()


@triton.jit(do_not_specialize=["rank"])
def allreduce_two_shot_multimem_intra_node_kernel(symm_ptr, grid_barrier_ptr, elems, rank, world_size,
                                                  use_cooperative: tl.constexpr):
    elems_per_rank = elems // world_size
    # each rank do all-reduce for elems_per_rank
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)
    data_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_ptr)
    data_mc_ptr += rank * elems_per_rank
    VEC_SIZE = 128 // tl.constexpr(symm_ptr.dtype.element_ty.primitive_bitwidth)
    if pid == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    thread_idx = tid(0)
    block_dim = num_warps() * 32
    for idx in range(thread_idx + block_dim * pid, elems_per_rank // VEC_SIZE, num_pid * block_dim):
        val0, val1, val2, val3 = multimem_ld_reduce_v4(data_mc_ptr + idx * VEC_SIZE)
        multimem_st_b64(data_mc_ptr + idx * VEC_SIZE, pack_b32_v2(val0, val1))
        multimem_st_b64(data_mc_ptr + idx * VEC_SIZE + VEC_SIZE // 2, pack_b32_v2(val2, val3))

    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)
    if pid == 0:
        libshmem_device.barrier_all_block()


def allreduce_one_shot_push_intra_node(
    ctx: AllReduceContext,
    x: torch.Tensor,
    output: Optional[torch.Tensor] = None,
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
    if output is None:
        output = torch.empty_like(x)
    else:
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
        use_cooperative=True,
        **launch_cooperative_grid_options(),
    )
    return output


def allreduce_two_shot_push_intra_node(
    ctx: AllReduceContext,
    x: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    straggler_option=None,
    max_sm: int = -1,
    num_warps: int = 32,
):
    """ Performs an intra-node all-reduce using a two-shot algorithm.

    This algorithm consists of two main phases:
    1. All-to-All and reduce
    2. All-Gather: Each rank gathers the reduced chunks from all other ranks to
       reconstruct the full, all-reduced tensor.

    This method requires a symmetric workspace of at least `2 * x.nbytes`.

    Args:
        ctx (AllReduceContext): The all-reduce context containing buffers and metadata.
        x (torch.Tensor): The input tensor to be all-reduced.
        output (Optional[torch.Tensor]): The tensor to store the output. If None, a new tensor is allocated.
        max_sm (int, optional): The maximum number of streaming multiprocessors to use.
                                Defaults to -1 (unlimited).
        num_warps (int, optional): The number of warps per block. Defaults to 32.
        straggler_option (optional): Used for stress test to simulate a slow rank. Defaults to None.

    Returns:
        torch.Tensor: The tensor containing the all-reduced result.
    """
    assert x.is_cuda and x.is_contiguous()
    if output is not None:
        assert output.is_cuda and output.is_contiguous()
        assert x.dtype == output.dtype and x.nbytes == output.nbytes
    else:
        output = torch.empty_like(x)
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
        use_cooperative=True,
        **launch_cooperative_grid_options(),
    )
    return output


def allreduce_one_shot_tma_push_intra_node(ctx: AllReduceContext, x: torch.Tensor,
                                           output: Optional[torch.Tensor] = None, straggler_option=None,
                                           max_sm: int = -1, num_warps: int = 32):
    """ P2P sends full-length copy of local tensor to other ranks, then uses TMA reduce.

    Args:
        straggler_option (_type_, optional): Simulates the straggler. Defaults to None.
    """
    assert x.is_cuda and x.is_contiguous()
    if output is None:
        output = torch.empty_like(x)
    else:
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
        use_cooperative=True,
        **launch_cooperative_grid_options(),
    )
    return output


@requires(is_nvshmem_multimem_supported)
def allreduce_one_shot_multimem_intra_node(
    ctx: AllReduceContext,
    x: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    straggler_option=None,
    num_warps=32,
    max_sm=-1,
):
    """ One-shot reduces symm buffer using the multimem.ld_reduce instruction.

    Raises:
        NotImplementedError: Currently supports bfloat16 and float16 and float32.
    """
    assert x.is_cuda and x.is_contiguous()
    # TODO(houqi.1993) int is not implemented yet
    assert x.dtype in [torch.float, torch.bfloat16, torch.float16]
    assert x.nbytes <= ctx.workspace_nbytes
    if output is None:
        output = torch.empty_like(x)
    else:
        assert output.is_cuda and output.is_contiguous()
        assert x.dtype == output.dtype and x.numel() == output.numel()

    _run_straggler(ctx, straggler_option)
    # _memcpy_async_unsafe(ctx.symm_scatter_buf, x, x.nbytes, torch.cuda.current_stream())
    num_blocks = min(8, triton.cdiv(x.numel(), num_warps * 32 * 16 // x.itemsize))
    num_blocks = num_blocks if max_sm < 0 else max_sm
    ctx.phase += 1
    allreduce_one_shot_multimem_intra_node_kernel[(num_blocks, )](
        x,
        ctx.symm_scatter_buf,
        ctx.symm_signal,
        output,
        x.numel(),
        ctx.grid_barrier,
        ctx.rank,
        ctx.phase,
        WORLD_SIZE=ctx.world_size,
        BLOCK_SIZE=num_warps * 32 * 16 // x.itemsize,
        num_warps=num_warps,
        use_cooperative=True,
        **launch_cooperative_grid_options(),
    )
    return output


def allreduce_double_tree_intra_node(
    ctx: AllReduceContext,
    x: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    max_sm: int = -1,
    num_warps: int = 32,
    straggler_option=None,
):
    """ WARNING: this function has performance issus to be fixed. try to verify the double-tree method.

    Args:
        ctx (AllReduceContext): The all-reduce context.
        x (torch.Tensor): The input tensor. Must be of dtype `torch.bfloat16`.
        output (torch.Tensor): The output tensor to store the result.
        max_sm (int, optional): The maximum number of streaming multiprocessors to use.
                                Defaults to -1 (unlimited).
        num_warps (int, optional): The number of warps per block. Defaults to 32.
        straggler_option (optional): Used for stress test to simulate a slow rank. Defaults to None.

    Returns:
        torch.Tensor: The output tensor with the all-reduced data.
    """
    warnings.warn(
        "`allreduce_double_tree_intra_node` should only be treated as a prototype. still a lot of performance issues to be fixed.",
        stacklevel=1)
    world_size, local_world_size = ctx.world_size, ctx.local_world_size
    rank = ctx.rank

    assert x.is_cuda and x.is_contiguous()
    if output is None:
        output = torch.empty_like(x)
    else:
        assert output.is_cuda and output.is_contiguous()
        assert output.dtype == x.dtype and output.numel() == x.numel()

    CHUNK_SIZE = 128 * 1024 // x.itemsize  # 128KB data per chunk
    CHUNK_SIZE = max(CHUNK_SIZE, triton.next_power_of_2(triton.cdiv(x.numel(), MAX_DOUBLE_TREE_BLOCKS)))
    BLOCK_SIZE = num_warps * 32 * 16 // x.itemsize
    num_blocks = triton.cdiv(x.numel(), CHUNK_SIZE)
    assert world_size == local_world_size, "Inter-node is not currently supported"
    assert ctx.symm_signal.numel() >= world_size * num_blocks, "Alloc at least one signal for each block"
    assert x.nbytes <= ctx.workspace_nbytes, f"Workspace size {ctx.workspace_nbytes} is not enough for {x.nbytes} bytes"

    tree0_parent, tree0_child0, tree0_child1, tree1_parent, tree1_child0, tree1_child1 = get_tree_parent_and_children(
        world_size, rank)

    _memcpy_async_unsafe(ctx.symm_scatter_buf, x, x.nbytes, torch.cuda.current_stream())
    ctx.symm_signal.zero_()
    _run_straggler(ctx, straggler_option)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    # 1024 * 1024 // 16 = 64 * 1024
    allreduce_double_tree_intra_node_kernel[(num_blocks, )](
        ctx.symm_scatter_buf,
        ctx.symm_signal,
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
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    return output


# TODO(houqi.1993) support very small shape. now only support 16 Byte aligned
@requires(is_nvshmem_multimem_supported)
def allreduce_two_shot_multimem_intra_node(
    ctx: AllReduceContext,
    x: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    straggler_option=None,
    num_warps=8,
    max_sm=-1,
):
    """
    Performs intra-node all-reduce in two shots using `multimem.ld_reduce` and `multimem.st`.

    Each rank does in-place all-reduce to a shard of the data using `multimem.ld_reduce` and `multimem.st`.
    This function requires x.nbytes of symmetric buffer.

    Args:
        ctx (AllReduceContext): The all-reduce context.
        x (torch.Tensor): The input tensor.
        output (Optional[torch.Tensor], optional): The output tensor. If None, the result is returned
                                                    in the symmetric buffer, which is valid until the next
                                                    collective call. Defaults to None.
        straggler_option (optional): Used for stress test to simulate a slow rank. Defaults to None.
        num_warps (int, optional): The number of warps per block. Defaults to 8.
        max_sm (int, optional): The maximum number of streaming multiprocessors to use. Defaults to -1.

    Returns:
        torch.Tensor: The output tensor with the all-reduced data. If `output` is None, this is a view
                      into the symmetric buffer.
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
        ctx.rank,
        ctx.world_size,
        num_warps=num_warps,
        use_cooperative=True,
        **launch_cooperative_grid_options(),
    )
    if output is None:
        output = ctx.symm_scatter_buf.view(x.dtype)[:elems].view_as(x)
    elif output.data_ptr() != ctx.symm_scatter_buf.data_ptr():
        _memcpy_async_unsafe(output, ctx.symm_scatter_buf, x.nbytes, torch.cuda.current_stream())
    return output


@requires(is_nvshmem_multimem_supported)
def allreduce_two_shot_multimem_st_intra_node(
    ctx: AllReduceContext,
    x: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    straggler_option=None,
    num_warps: int = 32,
    max_sm: int = -1,
):
    """
    DEPRECATED: This method is less efficient and you'd better use `allreduce_two_shot_multimem_intra_node` instead.

    This method use libnvshmem.putmem for all-to-all stage, and multimem.st for all-gather stage.
    It requires multimem features, but not fully utilizes it. Compared to `allreduce_two_shot_multimem_intra_node`,
    this function is less efficient and requires 2x symmetric memory.

    Args:
        ctx (AllReduceContext): The all-reduce context.
        x (torch.Tensor): The input tensor.
        output (Optional[torch.Tensor], optional): The output tensor. If None, the result is returned
                                                    in the symmetric buffer. Defaults to None.
        straggler_option (optional): Used for stress test to simulate a slow rank. Defaults to None.
        num_warps (int, optional): The number of warps per block. Defaults to 32.
        max_sm (int, optional): The maximum number of streaming multiprocessors to use. Defaults to -1.

    Returns:
        torch.Tensor: The output tensor with the all-reduced data.

    This function is DEPRECATED. Please use `allreduce_two_shot_multimem_intra_node` instead.

    this function requires the most symmetric buffer:
        x.nbytes for receive from other ranks.
        x.nbytes for symmetric output.
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
        rank,
        world_size,
        num_elem,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        use_cooperative=True,
        **launch_cooperative_grid_options(),
    )
    if output is None:
        output = ctx.symm_scatter_buf.view(x.dtype)[:num_elem].view_as(x)
    elif output.data_ptr() != ctx.symm_scatter_buf.data_ptr():
        _memcpy_async_unsafe(output, ctx.symm_scatter_buf, num_elem * x.itemsize, torch.cuda.current_stream())
    return output


def get_auto_allreduce_method(nbytes):
    """
    Automatically selects the best all-reduce method based on the input size and hardware capabilities.

    The selection logic prefers multimem-based methods if supported, falling back to TMA-based
    or standard one-shot methods otherwise. The choice also depends on the tensor size.

    Args:
        nbytes (int): The size of the input tensor in bytes.

    Returns:
        AllReduceMethod: The recommended all-reduce method.
    """
    if is_nvshmem_multimem_supported():
        # TODO(houqi.1993) maybe better for H800, not for H20 or other machines.
        if nbytes > 64 * 1024:
            return AllReduceMethod.TwoShot_Multimem
        else:
            return AllReduceMethod.OneShot_Multimem

    # TODO(houqi.1993) re-determine nbytes
    if has_tma() and nbytes < 16 * 1024:
        return AllReduceMethod.OneShot_TMA

    # TODO(houqi.1993) no two-shot without TMA implementation
    return AllReduceMethod.OneShot


def all_reduce(
    x: torch.Tensor,
    method: AllReduceMethod,
    ctx: AllReduceContext,
    output: Optional[torch.Tensor] = None,
    max_sm: int = -1,
    straggler_option=None,
):
    """
    Performs an all-reduce operation on the input tensor `x`.

    This is the main entry point for the all-reduce collective. It selects the specified
    all-reduce implementation and handles chunking if the input tensor is larger than
    the available workspace for a single operation.

    Args:
        x (torch.Tensor): The input tensor to be all-reduced.
        method (AllReduceMethod): The all-reduce algorithm to use. If None, a method is
                                  chosen automatically based on tensor size and hardware.
        ctx (AllReduceContext): The all-reduce context containing buffers and metadata.
        output (Optional[torch.Tensor], optional): The tensor to store the output. If None, a new tensor
                                             is allocated, or for some methods, a buffer from the
                                             context is returned. This buffer is valid only until
                                             the next all_reduce call. Defaults to None.
        max_sm (int, optional): The maximum number of streaming multiprocessors to use.
                                Defaults to -1 (unlimited).
        straggler_option (optional): Used for stress test to simulate a slow rank. Defaults to None.

    Returns:
        torch.Tensor: The tensor containing the all-reduced result.

    TODO(houqi.1993) if use multiple chunks, does not support CUDAGraph.
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

    if nchunks == 1:
        return op_handle(
            ctx=ctx,
            x=x,
            output=output,
            max_sm=max_sm,
            num_warps=32,
            straggler_option=straggler_option,
        )
    else:
        assert not torch.cuda.is_current_stream_capturing(
        ), "allreduce does not support CUDAGraph if use multiple chunks"
        if output is None:
            output = torch.empty_like(x)
        for n in range(nchunks):
            # TODO(houqi.1993) flatten invalidate shape check. please add back shape check again
            op_handle(
                ctx=ctx,
                x=x.flatten()[elems_per_chunk * n:elems_per_chunk * (n + 1)],
                output=output.flatten()[elems_per_chunk * n:elems_per_chunk * (n + 1)],
                max_sm=max_sm,
                num_warps=32,
                straggler_option=straggler_option,
            )

        return output
