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
""" NOTE: allgather.py is for high-throughput. while low_latency_allgather.py is for low-latency.
"""

import functools
from enum import Enum
from typing import List

import nvshmem.bindings.nvshmem as pynvshmem
import nvshmem.core
import torch
from cuda import cuda, cudart

import triton
import triton.language as tl
from triton.language.extra.cuda.language_extra import __syncthreads, tid
from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda, _wait_eq_cuda
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import (CUDA_CHECK, NVSHMEM_SIGNAL_DTYPE, has_fullmesh_nvlink, get_numa_world_size,
                               nvshmem_barrier_all_on_stream, sleep_async)


class AllGatherMethod(Enum):
    Auto = 0
    All2All_IntraNode = 1
    All2All_InterNode = 2
    Ring1D_IntraNode = 3
    Ring2D_IntraNode = 4
    Ring1D_InterNode = 5
    Ring2D_InterNode = 6


@functools.lru_cache()
def get_auto_all_gather_method(num_ranks, num_local_ranks):
    if has_fullmesh_nvlink():
        if num_ranks == num_local_ranks:
            return AllGatherMethod.All2All_IntraNode
        else:
            return AllGatherMethod.All2All_InterNode
    else:
        numa_world_size = get_numa_world_size()
        if num_local_ranks == num_ranks:
            if numa_world_size == num_ranks:
                return AllGatherMethod.Ring1D_IntraNode
            else:
                return AllGatherMethod.Ring2D_IntraNode
        else:
            return AllGatherMethod.Ring2D_InterNode


def _add_noise_workload_debug():
    import random

    if random.random() > 0.3:
        sleep_async(100)


def cp_engine_producer_all_gather_full_mesh_push(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    stream: torch.cuda.Stream,
):
    M_per_rank, N_per_rank = local_tensor.shape
    push_order = [(rank + i) % num_ranks for i in range(num_ranks)]
    src = local_tensor
    with torch.cuda.stream(stream):
        for dst_rank in push_order:
            dst = remote_tensor_buffers[dst_rank][rank * M_per_rank:(rank + 1) * M_per_rank, :]
            dst.copy_(src)

            (err, ) = cuda.cuStreamWriteValue32(
                stream.cuda_stream,
                barrier_buffers[dst_rank][rank].data_ptr(),
                1,
                cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )
            CUDA_CHECK(err)


def cp_engine_producer_all_gather_full_mesh_pull(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    stream: torch.cuda.Stream,
    for_correctness=False,
):
    M_per_rank, N = local_tensor.shape

    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    with torch.cuda.stream(stream):
        if for_correctness:
            # fake a slow communication case
            # test if the computation is waiting for the correct communication
            _add_noise_workload_debug()
        for src_rank in rank_orders:
            if src_rank == rank:
                continue
            dst = remote_tensor_buffers[rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
            src = remote_tensor_buffers[src_rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
            dst.copy_(src)

            (err, ) = cuda.cuStreamWriteValue32(
                stream.cuda_stream,
                barrier_buffers[rank][src_rank].data_ptr(),
                1,
                cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )
            CUDA_CHECK(err)


def cp_engine_producer_all_gather_ring_push_1d(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    stream: torch.cuda.Stream,
    for_correctness=False,
):
    flag_dtype = barrier_buffers[0].dtype
    assert flag_dtype in [torch.int32, NVSHMEM_SIGNAL_DTYPE], flag_dtype
    if flag_dtype == torch.int32:
        wait_value_fn = cuda.cuStreamWaitValue32
        write_value_fn = cuda.cuStreamWriteValue32
    else:
        wait_value_fn = cuda.cuStreamWaitValue64
        write_value_fn = cuda.cuStreamWriteValue64

    def wait_ready(rank: int, segment: int, stream: torch.cuda.Stream):
        (err, ) = wait_value_fn(
            stream.cuda_stream,
            barrier_buffers[rank][segment].data_ptr(),
            1,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        CUDA_CHECK(err)

    def set_ready(rank, segment, stream: torch.cuda.Stream):
        (err, ) = write_value_fn(
            stream.cuda_stream,
            barrier_buffers[rank][segment].data_ptr(),
            1,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
        CUDA_CHECK(err)

    M_per_rank, N = local_tensor.shape
    to_rank = (rank - 1 + num_ranks) % num_ranks
    with torch.cuda.stream(stream):
        if for_correctness:
            # fake a slow communication case
            # test if the computation is waiting for the correct communication
            _add_noise_workload_debug()

        for stage in range(num_ranks - 1):
            send_segment = (rank + stage) % num_ranks
            M_start = send_segment * M_per_rank
            M_end = M_start + M_per_rank
            if stage != 0:
                wait_ready(rank, send_segment, stream)
            dst = remote_tensor_buffers[to_rank][M_start:M_end, :]
            src = remote_tensor_buffers[rank][M_start:M_end, :]
            dst.copy_(src)
            set_ready(to_rank, send_segment, stream)


def cp_engine_producer_all_gather_ring_push_numa_2d(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    stream: torch.cuda.Stream,
    for_correctness=False,
):
    flag_dtype = barrier_buffers[0].dtype
    assert flag_dtype in [torch.int32, NVSHMEM_SIGNAL_DTYPE], flag_dtype
    if flag_dtype == torch.int32:
        wait_value_fn = cuda.cuStreamWaitValue32
        write_value_fn = cuda.cuStreamWriteValue32
    else:
        wait_value_fn = cuda.cuStreamWaitValue64
        write_value_fn = cuda.cuStreamWriteValue64

    def wait_ready(rank: int, segment: int, stream: torch.cuda.Stream):
        (err, ) = wait_value_fn(
            stream.cuda_stream,
            barrier_buffers[rank][segment].data_ptr(),
            1,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        CUDA_CHECK(err)

    def set_ready(rank, segment, stream: torch.cuda.Stream):
        (err, ) = write_value_fn(
            stream.cuda_stream,
            barrier_buffers[rank][segment].data_ptr(),
            1,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
        CUDA_CHECK(err)

    NUMA_WORLD_SIZE = get_numa_world_size()
    assert (num_ranks % NUMA_WORLD_SIZE == 0), f"num_ranks {num_ranks} should be divisible by NUMA {NUMA_WORLD_SIZE}"
    n_numa_nodes = num_ranks // NUMA_WORLD_SIZE
    assert n_numa_nodes == 2, f"n_numa_nodes {n_numa_nodes} should be 2"

    M_per_rank, N = local_tensor.shape
    to_rank = (rank - 1 + num_ranks) % num_ranks
    numa_node_id = rank // NUMA_WORLD_SIZE
    to_rank_numa = (rank - 1 + NUMA_WORLD_SIZE) % NUMA_WORLD_SIZE + numa_node_id * NUMA_WORLD_SIZE
    with torch.cuda.stream(stream):
        if for_correctness:
            # fake a slow communication case
            # test if the computation is waiting for the correct communication
            _add_noise_workload_debug()

        for stage in range(num_ranks - 1):
            send_segment = (rank + stage) % num_ranks
            is_2d_stage = stage >= NUMA_WORLD_SIZE and rank % NUMA_WORLD_SIZE == 0
            if is_2d_stage:
                send_segment = (send_segment + NUMA_WORLD_SIZE) % num_ranks
                to_rank = to_rank_numa
            M_start = send_segment * M_per_rank
            M_end = M_start + M_per_rank
            if stage != 0:
                wait_ready(rank, send_segment, stream)
            dst = remote_tensor_buffers[to_rank][M_start:M_end, :]
            src = remote_tensor_buffers[rank][M_start:M_end, :]
            dst.copy_(src)
            set_ready(to_rank, send_segment, stream)


def cp_engine_producer_all_gather_intra_node(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    stream: torch.cuda.Stream,
    for_correctness=False,
    all_gather_method: AllGatherMethod = AllGatherMethod.All2All_IntraNode,
):
    if all_gather_method == AllGatherMethod.All2All_IntraNode:
        fn = cp_engine_producer_all_gather_full_mesh_pull
    elif all_gather_method == AllGatherMethod.Ring1D_IntraNode:
        fn = cp_engine_producer_all_gather_ring_push_1d
    elif all_gather_method == AllGatherMethod.Ring2D_IntraNode:
        fn = cp_engine_producer_all_gather_ring_push_numa_2d
    else:
        raise Exception(f"Unsupported allgather method: {all_gather_method}")

    fn(
        rank,
        num_ranks,
        local_tensor,
        remote_tensor_buffers,
        barrier_buffers,
        stream,
        for_correctness=for_correctness,
    )


def cp_engine_producer_all_gather_ring_push_2d_inter_node(
    rank,
    num_local_ranks,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    intranode_stream: torch.cuda.Stream,
    internode_stream: torch.cuda.Stream = None,
    for_correctness=False,
):
    flag_dtype = barrier_buffers[0].dtype
    assert flag_dtype in [torch.int32, NVSHMEM_SIGNAL_DTYPE], flag_dtype
    if flag_dtype == torch.int32:
        wait_value_fn = cuda.cuStreamWaitValue32
        write_value_fn = cuda.cuStreamWriteValue32
    else:
        wait_value_fn = cuda.cuStreamWaitValue64
        write_value_fn = cuda.cuStreamWriteValue64

    def wait_ready(rank: int, segment: int, stream: torch.cuda.Stream):
        (err, ) = wait_value_fn(
            stream.cuda_stream,
            barrier_buffers[rank][segment].data_ptr(),
            1,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        CUDA_CHECK(err)

    def set_ready(rank, segment, stream: torch.cuda.Stream):
        (err, ) = write_value_fn(
            stream.cuda_stream,
            barrier_buffers[rank][segment].data_ptr(),
            1,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
        CUDA_CHECK(err)

    nnodes = num_ranks // num_local_ranks
    M_per_rank, N = local_tensor.shape
    nbytes_per_rank = M_per_rank * remote_tensor_buffers[0].itemsize * N
    local_rank = rank % num_local_ranks
    node_id = rank // num_local_ranks
    to_rank = (local_rank - 1 + num_local_ranks) % num_local_ranks
    with torch.cuda.stream(intranode_stream):
        if for_correctness:
            # fake a slow communication case
            # test if the computation is waiting for the correct communication
            _add_noise_workload_debug()

        for n in range(nnodes):
            rank_base = ((n + node_id) % nnodes) * num_local_ranks
            if n != 0:  # inter node comm
                # DON't overlap communication of INTER-NODE with INTRA-NODE: slow!
                nvshmem_barrier_all_on_stream(intranode_stream)
                rank_to_inter_node = (rank - n * num_local_ranks + num_ranks) % num_ranks
                segment = rank
                M_start = segment * M_per_rank
                M_end = M_start + M_per_rank
                src = remote_tensor_buffers[local_rank][M_start:M_end, :]
                # with gridDim = 1
                pynvshmem.putmem_signal_on_stream(
                    src.data_ptr(),
                    src.data_ptr(),
                    nbytes_per_rank,
                    barrier_buffers[local_rank][rank].data_ptr(),
                    1,
                    nvshmem.core.SignalOp.SIGNAL_SET,
                    rank_to_inter_node,
                    intranode_stream.cuda_stream,
                )
                nvshmem_barrier_all_on_stream(intranode_stream)

            # intra node comm
            for stage in range(num_local_ranks - 1):
                segment = (rank + stage) % num_local_ranks + rank_base
                M_start = segment * M_per_rank
                M_end = M_start + M_per_rank
                if stage != 0 or n != 0:
                    wait_ready(local_rank, segment, intranode_stream)
                dst = remote_tensor_buffers[to_rank][M_start:M_end, :]
                src = remote_tensor_buffers[local_rank][M_start:M_end, :]
                dst.copy_(src)
                set_ready(to_rank, segment, intranode_stream)


@triton.jit(do_not_specialize=["rank", "local_world_size", "world_size"])
def nvshmem_device_producer_all_gather_2d_put_block_kernel(
    ag_buffer_ptr,
    signal_buffer_ptr,
    elem_per_rank,
    size_per_elem,
    signal_target,
    rank,
    local_world_size,
    world_size,
    DISPATCH_BLOCK_NUM: tl.constexpr,
    SEND_BLOCK_NUM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    thread_idx = tid(axis=0)

    n_nodes = world_size // local_world_size
    n_nodes = world_size // local_world_size
    local_rank = rank % local_world_size
    node_rank = rank // local_world_size

    if pid < DISPATCH_BLOCK_NUM:  # intra dispatch block
        peer = (local_rank + pid + 1) % local_world_size + node_rank * local_world_size
        for i in range(n_nodes):
            segment = local_rank + ((node_rank + i) % n_nodes) * local_world_size
            if thread_idx == 0:
                libshmem_device.signal_wait_until(
                    signal_buffer_ptr + segment,
                    libshmem_device.NVSHMEM_CMP_GE,
                    signal_target,
                )
            __syncthreads()
            libshmem_device.putmem_signal_block(
                ag_buffer_ptr + segment * elem_per_rank,
                ag_buffer_ptr + segment * elem_per_rank,
                elem_per_rank * size_per_elem,
                signal_buffer_ptr + segment,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )
    else:  # inter send block
        if thread_idx == 0:
            libshmem_device.signal_wait_until(
                signal_buffer_ptr + rank,
                libshmem_device.NVSHMEM_CMP_GE,
                signal_target,
            )
        __syncthreads()
        global_send_pid = pid % SEND_BLOCK_NUM + 1
        peer = local_rank + (node_rank + global_send_pid) % n_nodes * local_world_size
        libshmem_device.putmem_signal_block(
            ag_buffer_ptr + rank * elem_per_rank,
            ag_buffer_ptr + rank * elem_per_rank,
            elem_per_rank * size_per_elem,
            signal_buffer_ptr + rank,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


@triton.jit
def nvshmem_device_producer_p2p_put_block_kernel(
    ag_buffer_ptr,
    signal_buffer_ptr,
    elem_per_rank,
    size_per_elem,
    signal_target,
    rank,
    local_world_size,
    world_size,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)

    n_nodes = world_size // local_world_size
    local_rank = rank % local_world_size
    node_rank = rank // local_world_size

    for i in range(pid, n_nodes - 1, num_pid):
        peer = local_rank + (node_rank + i + 1) % n_nodes * local_world_size
        libshmem_device.putmem_signal_block(
            ag_buffer_ptr + rank * elem_per_rank,
            ag_buffer_ptr + rank * elem_per_rank,
            elem_per_rank * size_per_elem,
            signal_buffer_ptr + rank,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


def cp_engine_producer_all_gather_full_mesh_pull_inter_node(
    rank,
    local_world_size,
    world_size,
    local_tensor: torch.Tensor,
    ag_buffer: list[torch.Tensor],
    signal_buffer: list[torch.Tensor],
    intranode_ag_stream=None,
    internode_ag_stream=None,
    signal_target=1,
    for_correctness=False,
):
    local_rank = rank % local_world_size
    n_nodes = world_size // local_world_size
    node_rank = rank // local_world_size
    M_per_rank, N = local_tensor.shape
    with torch.cuda.stream(internode_ag_stream):
        grid = lambda META: (int(n_nodes - 1), )
        nvshmem_device_producer_p2p_put_block_kernel[grid](
            ag_buffer[local_rank],
            signal_buffer[local_rank],
            M_per_rank * N,
            local_tensor.element_size(),
            signal_target,
            rank,
            local_world_size,
            world_size,
            num_warps=32,
        )

    with torch.cuda.stream(intranode_ag_stream):
        for i in range(1, local_world_size):
            segment = rank * M_per_rank * N
            local_dst_rank = (local_rank + local_world_size - i) % local_world_size
            src_ptr = (ag_buffer[local_rank].data_ptr() + segment * local_tensor.element_size())
            dst_ptr = (ag_buffer[local_dst_rank].data_ptr() + segment * local_tensor.element_size())
            (err, ) = cudart.cudaMemcpyAsync(
                dst_ptr,
                src_ptr,
                M_per_rank * N * local_tensor.element_size(),
                cudart.cudaMemcpyKind.cudaMemcpyDefault,
                intranode_ag_stream.cuda_stream,
            )
            CUDA_CHECK(err)
            _set_signal_cuda(signal_buffer[local_dst_rank][rank], signal_target, intranode_ag_stream)

        for i in range(1, n_nodes):
            recv_rank = (local_rank + (node_rank + n_nodes - i) % n_nodes * local_world_size)
            recv_segment = recv_rank * M_per_rank * N
            _wait_eq_cuda(signal_buffer[local_rank][recv_rank], signal_target, intranode_ag_stream)
            src_ptr = (ag_buffer[local_rank].data_ptr() + recv_segment * local_tensor.element_size())
            for j in range(1, local_world_size):
                local_dst_rank = (local_rank + local_world_size - j) % local_world_size
                dst_ptr = (ag_buffer[local_dst_rank].data_ptr() + recv_segment * local_tensor.element_size())
                (err, ) = cudart.cudaMemcpyAsync(
                    dst_ptr,
                    src_ptr,
                    M_per_rank * N * local_tensor.element_size(),
                    cudart.cudaMemcpyKind.cudaMemcpyDefault,
                    intranode_ag_stream.cuda_stream,
                )
                CUDA_CHECK(err)
                _set_signal_cuda(signal_buffer[local_dst_rank][recv_rank], signal_target, intranode_ag_stream)

    intranode_ag_stream.wait_stream(internode_ag_stream)


def cp_engine_producer_all_gather_inter_node(
    local_tensor: torch.Tensor,
    ag_buffer: list[torch.Tensor],
    signal_buffer: list[torch.Tensor],
    signal_target,
    rank,
    local_world_size,
    world_size,
    intranode_ag_stream=None,
    internode_ag_stream=None,
    for_correctness: bool = False,
    all_gather_method: AllGatherMethod = AllGatherMethod.All2All_InterNode,
):
    if all_gather_method == AllGatherMethod.All2All_InterNode:
        cp_engine_producer_all_gather_full_mesh_pull_inter_node(
            rank,
            local_world_size,
            world_size,
            local_tensor,
            ag_buffer,
            signal_buffer,
            intranode_ag_stream,
            internode_ag_stream,
            signal_target=signal_target,
            for_correctness=for_correctness,
        )
    elif all_gather_method == AllGatherMethod.Ring2D_InterNode:
        cp_engine_producer_all_gather_ring_push_2d_inter_node(
            rank,
            local_world_size,
            world_size,
            local_tensor,
            ag_buffer,
            signal_buffer,
            intranode_ag_stream,
            None,
            for_correctness=for_correctness,
        )
    else:
        raise Exception(f"Unsupported allgather method: {all_gather_method}")
