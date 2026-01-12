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
import triton
import triton_dist
import triton.language as tl
import triton_dist.language as dl
# TODO:(MACA UPGRADE): support from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.maca import libmxshmem_device as libshmem_device
from triton_dist.language.extra.language_extra import tid, __syncthreads
import pymxshmem
import time
from typing import Optional, List
from maca import macart
from dataclasses import dataclass
from triton_dist.utils import MACA_CHECK
from triton_dist.kernels.metax.utils import get_numa_world_size, has_fullmesh_mxlink_ngpus
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def cp_engine_producer_all_gather_full_mesh_push(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    ag_stream: torch.cuda.Stream,
    barrier_buffers: List[torch.Tensor],
    num_chunks_per_rank: int,
    for_correctness=False,
    signal_t=None,
):
    M_per_rank, K = local_tensor.shape
    M_per_chunk = M_per_rank // num_chunks_per_rank
    ele_byte = torch.finfo(local_tensor.dtype).bits // 8

    rank_orders = [(rank - i + num_ranks) % num_ranks for i in range(1, num_ranks)]

    with torch.cuda.stream(ag_stream):
        if for_correctness:
            # fake a slow communication case
            # test if the computation is waiting for the correct communication
            time.sleep(3)

        # if full mesh
        dst_arr = [None] * num_chunks_per_rank * (num_ranks - 1)
        src_arr = [None] * num_chunks_per_rank * (num_ranks - 1)
        engine = [None] * num_chunks_per_rank * (num_ranks - 1)
        count = [None] * num_chunks_per_rank * (num_ranks - 1)
        write_flag = [None] * num_chunks_per_rank * (num_ranks - 1)
        write_value = [None] * num_chunks_per_rank * (num_ranks - 1)
        for chunk_idx in range(num_chunks_per_rank):
            chunk_offset = chunk_idx * M_per_chunk

            for rank_idx, src_rank in enumerate(rank_orders):
                rank_base = rank * M_per_rank
                dst = remote_tensor_buffers[src_rank][rank_base + chunk_offset:rank_base + chunk_offset +
                                                      M_per_chunk, :]
                src = local_tensor[chunk_offset:chunk_offset + M_per_chunk, :]

                req_idx = chunk_idx * (num_ranks - 1) + rank_idx
                dst_arr[req_idx] = dst.data_ptr()
                src_arr[req_idx] = src.data_ptr()
                engine[req_idx] = macart.mcParallelCopyEngine.ParallelCopyEngineDefault
                count[req_idx] = M_per_chunk * K * ele_byte
                write_flag[req_idx] = barrier_buffers[src_rank][rank * num_chunks_per_rank + chunk_idx].data_ptr()
                write_value[req_idx] = signal_t

        (err, ) = macart.mcExtBatchCopyFlagAndWait(dst_arr,  # dst addr
                                                   src_arr,  # src addr
                                                   engine,  # cp engine
                                                   count,  # data size
                                                   write_flag,  # barrier addr
                                                   write_value,  # barrier value
                                                   [], [], ag_stream.cuda_stream  # stream
                                                   )
        MACA_CHECK(err)


def cp_engine_producer_all_gather_numa_node_push(
    rank,
    num_ranks,
    numa_world_size,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    intranode_ag_stream: torch.cuda.Stream,
    internode_ag_stream: torch.cuda.Stream,
    barrier_buffers: List[torch.Tensor],
    num_chunks_per_rank: int,
    for_correctness=False,
    signal_t=None,
):
    local_rank = rank % numa_world_size  # numa node内的rank
    n_numa_nodes = num_ranks // numa_world_size  # numa node个数
    numa_id = rank // numa_world_size  # rank 所在的numa id
    M_per_rank, K = local_tensor.shape
    M_per_chunk = M_per_rank // num_chunks_per_rank
    ele_byte = torch.finfo(local_tensor.dtype).bits // 8
    # TODO : n_numa_nodes == 2

    with torch.cuda.stream(internode_ag_stream):
        # inter_numa_node_p2p
        inter_dst_arr = [None] * num_chunks_per_rank * (n_numa_nodes - 1)
        inter_src_arr = [None] * num_chunks_per_rank * (n_numa_nodes - 1)
        inter_engine = [None] * num_chunks_per_rank * (n_numa_nodes - 1)
        inter_count = [None] * num_chunks_per_rank * (n_numa_nodes - 1)
        inter_write_flag = [None] * num_chunks_per_rank * (n_numa_nodes - 1)
        inter_write_value = [None] * num_chunks_per_rank * (n_numa_nodes - 1)

        rank_base = rank * M_per_rank
        for chunk_idx in range(num_chunks_per_rank):
            chunk_offset = chunk_idx * M_per_chunk
            for i in range(1, n_numa_nodes):
                p2p_send_rank = local_rank + (numa_id - i + n_numa_nodes) % n_numa_nodes * numa_world_size
                dst = remote_tensor_buffers[p2p_send_rank][rank_base + chunk_offset:rank_base + chunk_offset +
                                                           M_per_chunk, :]
                src = local_tensor[chunk_offset:chunk_offset + M_per_chunk, :]

                req_idx = chunk_idx * (n_numa_nodes - 1) + (i - 1)
                inter_dst_arr[req_idx] = dst.data_ptr()
                inter_src_arr[req_idx] = src.data_ptr()
                inter_engine[req_idx] = macart.mcParallelCopyEngine.ParallelCopyEngine0
                inter_count[req_idx] = M_per_chunk * K * ele_byte
                inter_write_flag[req_idx] = barrier_buffers[p2p_send_rank][rank * num_chunks_per_rank +
                                                                           chunk_idx].data_ptr()
                inter_write_value[req_idx] = signal_t
        (err, ) = macart.mcExtBatchCopyFlagAndWait(inter_dst_arr,  # dst addr
                                                   inter_src_arr,  # src addr
                                                   inter_engine,  # cp engine
                                                   inter_count,  # data size
                                                   inter_write_flag,  # barrier addr
                                                   inter_write_value,  # barrier value
                                                   [], [], internode_ag_stream.cuda_stream  # stream
                                                   )
        MACA_CHECK(err)

    with torch.cuda.stream(intranode_ag_stream):
        # intra numa node, local numa node data:
        rank_orders = [(local_rank - i + numa_world_size) % numa_world_size for i in range(1, numa_world_size)]
        engines = [
            macart.mcParallelCopyEngine.ParallelCopyEngine1, macart.mcParallelCopyEngine.ParallelCopyEngine2,
            macart.mcParallelCopyEngine.ParallelCopyEngine3, macart.mcParallelCopyEngine.ParallelCopyEngine4
        ]
        engines_size = len(engines)

        intra_dst_arr = [None] * num_chunks_per_rank * (numa_world_size - 1)
        intra_src_arr = [None] * num_chunks_per_rank * (numa_world_size - 1)
        intra_engine = [None] * num_chunks_per_rank * (numa_world_size - 1)
        intra_count = [None] * num_chunks_per_rank * (numa_world_size - 1)
        intra_write_flag = [None] * num_chunks_per_rank * (numa_world_size - 1)
        intra_write_value = [None] * num_chunks_per_rank * (numa_world_size - 1)
        intra_wait_flag = [None] * num_chunks_per_rank * (numa_world_size - 1)
        intra_wait_value = [None] * num_chunks_per_rank * (numa_world_size - 1)
        rank_base = rank * M_per_rank

        for chunk_idx in range(num_chunks_per_rank):
            chunk_offset = chunk_idx * M_per_chunk
            for rank_idx, src_local_rank in enumerate(rank_orders):
                src_rank = src_local_rank + numa_id * numa_world_size
                dst = remote_tensor_buffers[src_rank][rank_base + chunk_offset:rank_base + chunk_offset +
                                                      M_per_chunk, :]
                src = local_tensor[chunk_offset:chunk_offset + M_per_chunk, :]

                req_idx = chunk_idx * (numa_world_size - 1) + rank_idx
                intra_dst_arr[req_idx] = dst.data_ptr()
                intra_src_arr[req_idx] = src.data_ptr()
                intra_engine[req_idx] = engines[req_idx % engines_size]
                intra_count[req_idx] = M_per_chunk * K * ele_byte
                intra_write_flag[req_idx] = barrier_buffers[src_rank][rank * num_chunks_per_rank + chunk_idx].data_ptr()
                intra_write_value[req_idx] = signal_t

        (err, ) = macart.mcExtBatchCopyFlagAndWait(intra_dst_arr,  # dst addr
                                                   intra_src_arr,  # src addr
                                                   intra_engine,  # cp engine
                                                   intra_count,  # data size
                                                   intra_write_flag,  # barrier addr
                                                   intra_write_value,  # barrier value
                                                   [], [], intranode_ag_stream.cuda_stream  # stream
                                                   )
        MACA_CHECK(err)

        p2p_recv_rank = local_rank + (numa_id + 1 + n_numa_nodes) % n_numa_nodes * numa_world_size
        for chunk_idx in range(num_chunks_per_rank):
            chunk_offset = chunk_idx * M_per_chunk
            for rank_idx, src_local_rank in enumerate(rank_orders):
                src_rank = src_local_rank + numa_id * numa_world_size
                rank_base = p2p_recv_rank * M_per_rank
                src = remote_tensor_buffers[rank][rank_base + chunk_offset:rank_base + chunk_offset + M_per_chunk, :]
                dst = remote_tensor_buffers[src_rank][rank_base + chunk_offset:rank_base + chunk_offset +
                                                      M_per_chunk, :]
                req_idx = chunk_idx * (numa_world_size - 1) + rank_idx
                intra_dst_arr[req_idx] = dst.data_ptr()
                intra_src_arr[req_idx] = src.data_ptr()
                intra_engine[req_idx] = engines[req_idx % engines_size]
                intra_count[req_idx] = M_per_chunk * K * ele_byte
                intra_write_flag[req_idx] = barrier_buffers[src_rank][p2p_recv_rank * num_chunks_per_rank +
                                                                      chunk_idx].data_ptr()
                intra_write_value[req_idx] = signal_t
                intra_wait_flag[req_idx] = barrier_buffers[rank][p2p_recv_rank * num_chunks_per_rank +
                                                                 chunk_idx].data_ptr()
                intra_wait_value[req_idx] = signal_t

        (err, ) = macart.mcExtBatchCopyFlagAndWait(intra_dst_arr,  # dst addr
                                                   intra_src_arr,  # src addr
                                                   intra_engine,  # cp engine
                                                   intra_count,  # data size
                                                   intra_write_flag,  # barrier addr
                                                   intra_write_value,  # barrier value
                                                   intra_wait_flag,  # wait addr
                                                   intra_wait_value,  # wait value
                                                   intranode_ag_stream.cuda_stream  # stream
                                                   )
        MACA_CHECK(err)

    intranode_ag_stream.wait_stream(internode_ag_stream)


@triton_dist.jit
def mxshmem_device_producer_all_gather_2d_put_block_kernel(
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
                    libshmem_device.MXSHMEM_CMP_GE,
                    signal_target,
                )
            __syncthreads()
            libshmem_device.putmem_signal_block(
                ag_buffer_ptr + segment * elem_per_rank,
                ag_buffer_ptr + segment * elem_per_rank,
                elem_per_rank * size_per_elem,
                signal_buffer_ptr + segment,
                signal_target,
                libshmem_device.MXSHMEM_SIGNAL_SET,
                peer,
            )
    else:  # inter send block
        if thread_idx == 0:
            libshmem_device.signal_wait_until(
                signal_buffer_ptr + rank,
                libshmem_device.MXSHMEM_CMP_GE,
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
            libshmem_device.MXSHMEM_SIGNAL_SET,
            peer,
        )


@triton_dist.jit
def mxshmem_device_producer_p2p_put_block_kernel(
    ag_buffer_ptr,
    signal_buffer_ptr,
    size_per_elem,
    signal_target,
    rank,
    local_world_size,
    world_size,
    elem_per_chunk: tl.constexpr,
    num_chunks_per_rank: tl.constexpr = 1,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)

    n_nodes = world_size // local_world_size
    local_rank = rank % local_world_size
    node_rank = rank // local_world_size
    elem_per_rank = elem_per_chunk * num_chunks_per_rank

    for i in range(pid, (n_nodes - 1) * num_chunks_per_rank, num_pid):
        rank_idx = i // num_chunks_per_rank
        chunk_idx = i % num_chunks_per_rank
        peer = local_rank + (node_rank + rank_idx + 1) % n_nodes * local_world_size
        libshmem_device.putmem_signal_block(
            ag_buffer_ptr + rank * elem_per_rank + chunk_idx * elem_per_chunk,
            ag_buffer_ptr + rank * elem_per_rank + chunk_idx * elem_per_chunk,
            elem_per_chunk * size_per_elem,
            signal_buffer_ptr + rank * num_chunks_per_rank + chunk_idx,
            signal_target,
            libshmem_device.MXSHMEM_SIGNAL_SET,
            peer,
        )


def inter_node_allgather(local_tensor: torch.Tensor, ag_buffer: list[torch.Tensor], signal_buffer: list[torch.Tensor],
                         numa_world_size, signal_target, rank, num_chunks_per_rank, local_world_size, world_size,
                         intranode_ag_stream=None, internode_ag_stream=None, cpengine_dispatch=False, signal_t=None,
                         numanode_ag_stream=None):
    local_rank = rank % local_world_size
    n_nodes = world_size // local_world_size
    node_rank = rank // local_world_size
    M_per_rank, N = local_tensor.shape
    M_per_chunk = M_per_rank // num_chunks_per_rank
    n_numa_nodes = local_world_size // numa_world_size
    numa_id = local_rank // numa_world_size
    local_numa_rank = local_rank % numa_world_size

    if not cpengine_dispatch:
        assert num_chunks_per_rank != 1
        with torch.cuda.stream(internode_ag_stream):
            grid = lambda META: (int(local_world_size + n_nodes - 2), )
            mxshmem_device_producer_all_gather_2d_put_block_kernel[grid](
                ag_buffer[local_rank],
                signal_buffer[local_rank],
                M_per_rank * N,
                local_tensor.element_size(),
                signal_target,
                rank,
                local_world_size,
                world_size,
                tl.constexpr(local_world_size - 1),
                tl.constexpr(n_nodes - 1),
                num_warps=16,
            )
    else:
        # internode comm
        with torch.cuda.stream(internode_ag_stream):
            if n_numa_nodes > 1:
                # numanode comm with local tensor
                arr_size = num_chunks_per_rank * (n_numa_nodes - 1)
                dst_arr = [None] * arr_size
                src_arr = [None] * arr_size
                engine = [None] * arr_size
                count = [None] * arr_size
                write_flag = [None] * arr_size
                write_value = [None] * arr_size
                rank_base = rank * M_per_rank
                for chunk_idx in range(num_chunks_per_rank):
                    chunk_offset = chunk_idx * M_per_chunk
                    for i in range(1, n_numa_nodes):
                        p2p_send_rank = local_numa_rank + (numa_id - i + n_numa_nodes) % n_numa_nodes * numa_world_size
                        dst = ag_buffer[p2p_send_rank].data_ptr() + (rank_base +
                                                                     chunk_offset) * N * local_tensor.element_size()
                        src = ag_buffer[local_rank].data_ptr() + (rank_base +
                                                                  chunk_offset) * N * local_tensor.element_size()
                        req_idx = chunk_idx * (n_numa_nodes - 1) + (i - 1)
                        dst_arr[req_idx] = dst
                        src_arr[req_idx] = src
                        engine[req_idx] = macart.mcParallelCopyEngine.ParallelCopyEngine0
                        count[req_idx] = M_per_chunk * N * local_tensor.element_size()
                        write_flag[req_idx] = signal_buffer[p2p_send_rank].data_ptr() + (
                            rank * num_chunks_per_rank + chunk_idx) * signal_buffer[0].element_size()
                        write_value[req_idx] = signal_t

                (err, ) = macart.mcExtBatchCopyFlagAndWait(
                    dst_arr,  # dst addr
                    src_arr,  # src addr
                    engine,  # cp engine
                    count,  # data size
                    write_flag,  # barrier addr
                    write_value,  # barrier value
                    [],  # barrier addr
                    [],  # barrier value
                    #numanode_ag_stream.cuda_stream   # stream
                    internode_ag_stream.cuda_stream  # stream
                )
                MACA_CHECK(err)
            # internode comm
            grid = lambda META: (int(n_nodes - 1) * num_chunks_per_rank, )
            mxshmem_device_producer_p2p_put_block_kernel[grid](
                #internode_workspaces[local_rank],
                #inter_signal_buffer[local_rank],
                ag_buffer[local_rank],
                signal_buffer[local_rank],
                local_tensor.element_size(),
                signal_target,
                rank,
                local_world_size,
                world_size,
                num_warps=16,
                elem_per_chunk=M_per_chunk * N,
                num_chunks_per_rank=num_chunks_per_rank,
            )
            if n_numa_nodes > 1:
                for node_id in range(n_nodes):
                    if node_id != node_rank:
                        # numanode comm with inter tensor
                        arr_size = num_chunks_per_rank * (n_numa_nodes - 1)
                        dst_arr = [None] * arr_size
                        src_arr = [None] * arr_size
                        engine = [None] * arr_size
                        count = [None] * arr_size
                        wait_flag = [None] * arr_size
                        wait_value = [None] * arr_size
                        write_flag = [None] * arr_size
                        write_value = [None] * arr_size
                        rank_idx = local_rank + node_id * local_world_size
                        rank_base = rank_idx * M_per_rank
                        for chunk_idx in range(num_chunks_per_rank):
                            chunk_offset = chunk_idx * M_per_chunk
                            for i in range(1, n_numa_nodes):
                                p2p_send_rank = local_numa_rank + (numa_id - i +
                                                                   n_numa_nodes) % n_numa_nodes * numa_world_size
                                dst = ag_buffer[p2p_send_rank].data_ptr() + (
                                    rank_base + chunk_offset) * N * local_tensor.element_size()
                                src = ag_buffer[local_rank].data_ptr() + (
                                    rank_base + chunk_offset) * N * local_tensor.element_size()
                                req_idx = chunk_idx * (n_numa_nodes - 1) + (i - 1)
                                dst_arr[req_idx] = dst
                                src_arr[req_idx] = src
                                engine[req_idx] = macart.mcParallelCopyEngine.ParallelCopyEngine0
                                count[req_idx] = M_per_chunk * N * local_tensor.element_size()
                                wait_flag[req_idx] = signal_buffer[local_rank].data_ptr() + (
                                    rank_idx * num_chunks_per_rank + chunk_idx) * signal_buffer[0].element_size()
                                wait_value[req_idx] = signal_t
                                write_flag[req_idx] = signal_buffer[p2p_send_rank].data_ptr() + (
                                    rank_idx * num_chunks_per_rank + chunk_idx) * signal_buffer[0].element_size()
                                write_value[req_idx] = signal_t

                        (err, ) = macart.mcExtBatchCopyFlagAndWait(
                            dst_arr,  # dst addr
                            src_arr,  # src addr
                            engine,  # cp engine
                            count,  # data size
                            write_flag,  # barrier addr
                            write_value,  # barrier value
                            wait_flag,  # barrier addr
                            wait_value,  # barrier value
                            #numanode_ag_stream.cuda_stream   # stream
                            internode_ag_stream.cuda_stream  # stream
                        )
                        MACA_CHECK(err)
        if n_numa_nodes > 1:
            engine_list = [
                macart.mcParallelCopyEngine.ParallelCopyEngine1, macart.mcParallelCopyEngine.ParallelCopyEngine2,
                macart.mcParallelCopyEngine.ParallelCopyEngine3, macart.mcParallelCopyEngine.ParallelCopyEngine4
            ]
        else:
            engine_list = [macart.mcParallelCopyEngine.ParallelCopyEngineDefault]
        with torch.cuda.stream(intranode_ag_stream):
            idx = 0
            arr_size = num_chunks_per_rank * (numa_world_size - 1)
            dst_arr = [None] * arr_size
            src_arr = [None] * arr_size
            engine = [None] * arr_size
            count = [None] * arr_size
            wait_flag = [None] * arr_size
            wait_value = [None] * arr_size
            write_flag = [None] * arr_size
            write_value = [None] * arr_size
            # intranode comm with local tensor
            for chunk_idx in range(num_chunks_per_rank):
                for i in range(1, numa_world_size):
                    local_dst_rank = numa_id * numa_world_size + (local_numa_rank + numa_world_size -
                                                                  i) % numa_world_size
                    segment = (rank * M_per_rank + chunk_idx * M_per_chunk) * N
                    src_ptr = ag_buffer[local_rank].data_ptr() + segment * local_tensor.element_size()
                    dst_ptr = ag_buffer[local_dst_rank].data_ptr() + segment * local_tensor.element_size()
                    dst_arr[idx] = dst_ptr
                    src_arr[idx] = src_ptr
                    engine[idx] = engine_list[idx % len(engine_list)]
                    count[idx] = M_per_chunk * N * local_tensor.element_size()
                    write_flag[idx] = signal_buffer[local_dst_rank].data_ptr() + (
                        rank * num_chunks_per_rank + chunk_idx) * signal_buffer[0].element_size()
                    write_value[idx] = signal_t
                    idx += 1

            if idx > 0:
                (err, ) = macart.mcExtBatchCopyFlagAndWait(dst_arr,  # dst addr
                                                           src_arr,  # src addr
                                                           engine,  # cp engine
                                                           count,  # data size
                                                           write_flag,  # barrier addr
                                                           write_value,  # barrier value
                                                           [], [], intranode_ag_stream.cuda_stream  # stream
                                                           )
                MACA_CHECK(err)

            arr_size = num_chunks_per_rank * (numa_world_size - 1) * (n_nodes * n_numa_nodes - 1)
            dst_arr = [None] * arr_size
            src_arr = [None] * arr_size
            engine = [None] * arr_size
            count = [None] * arr_size
            wait_flag = [None] * arr_size
            wait_value = [None] * arr_size
            write_flag = [None] * arr_size
            write_value = [None] * arr_size
            idx = 0
            # intranode comm with numanode & internode tensor, first numanode second internode.
            for i in range(1, n_nodes * n_numa_nodes):
                numa_idx = (numa_id + n_numa_nodes - i % n_numa_nodes) % n_numa_nodes
                node_idx = (node_rank + n_nodes - i // n_numa_nodes) % n_nodes
                recv_rank = local_numa_rank + numa_idx * numa_world_size + node_idx * local_world_size
                for chunk_idx in range(num_chunks_per_rank):
                    recv_segment = (recv_rank * M_per_rank + chunk_idx * M_per_chunk) * N
                    src_ptr = ag_buffer[local_rank].data_ptr() + recv_segment * local_tensor.element_size()
                    for j in range(1, numa_world_size):
                        local_dst_rank = numa_id * numa_world_size + (local_numa_rank + numa_world_size -
                                                                      j) % numa_world_size
                        dst_ptr = ag_buffer[local_dst_rank].data_ptr() + recv_segment * local_tensor.element_size()
                        dst_arr[idx] = dst_ptr
                        src_arr[idx] = src_ptr
                        engine[idx] = engine_list[idx % len(engine_list)]
                        count[idx] = M_per_chunk * N * local_tensor.element_size()
                        write_flag[idx] = signal_buffer[local_dst_rank].data_ptr() + (
                            recv_rank * num_chunks_per_rank + chunk_idx) * signal_buffer[0].element_size()
                        write_value[idx] = signal_t
                        wait_flag[idx] = signal_buffer[local_rank].data_ptr() + (
                            recv_rank * num_chunks_per_rank + chunk_idx) * signal_buffer[0].element_size()
                        wait_value[idx] = signal_t
                        idx += 1
            if idx > 0:
                (err, ) = macart.mcExtBatchCopyFlagAndWait(dst_arr,  # dst addr
                                                           src_arr,  # src addr
                                                           engine,  # cp engine
                                                           count,  # data size
                                                           write_flag,  # barrier addr
                                                           write_value,  # barrier value
                                                           wait_flag,  # barrier addr
                                                           wait_value,  # barrier value
                                                           intranode_ag_stream.cuda_stream  # stream
                                                           )
                MACA_CHECK(err)

        intranode_ag_stream.wait_stream(internode_ag_stream)
        if numanode_ag_stream:
            intranode_ag_stream.wait_stream(numanode_ag_stream)


@triton_dist.jit
def kernel_local_copy_and_barrier_all(
    rank,
    num_ranks,
    local_buf_ptr,
    global_buf_ptr,
    barrier_ptr,
    M_per_rank,
    N,
    stride_local_m,
    stride_local_n,
    stride_global_m,
    stride_global_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    sm_id = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = sm_id // num_pid_n
    pid_n = sm_id % num_pid_n

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    data_ptr = local_buf_ptr + (pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_local_m + (
        pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_local_n
    dst_ptr = global_buf_ptr + (rank * M_per_rank + pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_global_m + (
        pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_global_n
    mask_data = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (pid_n * BLOCK_SIZE_N + offs_n[None, :] < N)
    mask_dst = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (pid_n * BLOCK_SIZE_N + offs_n[None, :] < N)

    data = tl.load(data_ptr, mask=mask_data)
    tl.store(dst_ptr, data, mask=mask_dst)


def local_copy_and_barrier_all(rank, num_ranks, local_data, global_data, comm_buf, barrier_ptr, M_per_rank, N,
                               is_internode: bool = False):
    assert is_internode
    barrier_ptr.fill_(0)
    pymxshmem.mxshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    grid = lambda META: (triton.cdiv(M_per_rank, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    kernel_local_copy_and_barrier_all[grid](rank, num_ranks, local_data, global_data, barrier_ptr, M_per_rank, N,
                                            local_data.stride(0), local_data.stride(1), global_data.stride(0),
                                            global_data.stride(1), 128, 256)
    # TODO: fix set_signal
    # set_signal(barrier_ptr[rank].data_ptr(), 1, torch.cuda.current_stream(), is_internode)
    barrier_ptr[rank].fill_(1)


@triton_dist.jit(do_not_specialize=["rank"])
def kernel_consumer_gemm(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    local_ptr,
    # Distributed parameters
    rank,
    num_ranks,
    ready_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    is_fp8: tl.constexpr,
    needs_wait: tl.constexpr,
    ready_value: tl.constexpr = 1,
    local_world_size: tl.constexpr = 8,
    nnodes: tl.constexpr = 1,
    num_chunks_per_rank: tl.constexpr = 1,
    nnumas: tl.constexpr = 1,
    numa_world_size: tl.constexpr = 8,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    dtype = tl.float16 if not is_fp8 else tl.float8e4nv
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    M_per_rank = M // num_ranks
    pid_ms_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    # read A from local or symmetric memory
    is_local = (pid_m < pid_ms_per_rank)

    if is_local:
        pid_m_local = pid_m
        # read local tensor A with local M offset
        offs_am = (pid_m_local * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = local_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[None, :] * stride_bk + offs_bn[:, None] * stride_bn)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
        for k in range(0, num_k_blocks):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.trans(b)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(dtype)
        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        # write tensor C with global M offset
        offs_cm = M_per_rank * rank + pid_m_local * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        # block idx start from current rank
        if nnodes == 1:
            m_offset = M_per_rank * rank
            pid_m_offset = tl.cdiv(m_offset, BLOCK_SIZE_M)
            pid_m = (pid_m + pid_m_offset) % num_pid_m
        elif nnumas == 1:
            node_id = rank // local_world_size
            m_rank = pid_m // pid_ms_per_rank
            pid_m_intra_rank = pid_m - m_rank * pid_ms_per_rank
            m_node_id = m_rank // local_world_size
            m_local_rank = m_rank % local_world_size
            swizzle_m_node_id = (m_node_id + node_id) % nnodes
            swizzle_m_local_rank = (m_local_rank + rank) % local_world_size
            swizzle_m_rank = swizzle_m_node_id * local_world_size + swizzle_m_local_rank
            pid_m = swizzle_m_rank * pid_ms_per_rank + pid_m_intra_rank
        else:
            local_rank = rank % local_world_size
            node_id = rank // local_world_size
            numa_id = local_rank // numa_world_size
            m_rank = pid_m // pid_ms_per_rank
            pid_m_intra_rank = pid_m - m_rank * pid_ms_per_rank
            m_node_id = m_rank // local_world_size
            m_numa_id = (m_rank % local_world_size) // numa_world_size
            m_local_rank = m_rank % numa_world_size
            swizzle_m_node_id = (m_node_id + node_id) % nnodes
            swizzle_m_numa_id = (m_numa_id + numa_id) % nnumas
            swizzle_m_local_rank = (m_local_rank + rank) % numa_world_size
            swizzle_m_rank = swizzle_m_node_id * local_world_size + swizzle_m_local_rank + swizzle_m_numa_id * numa_world_size
            pid_m = swizzle_m_rank * pid_ms_per_rank + pid_m_intra_rank

        M_per_chunk = M_per_rank // num_chunks_per_rank
        offs_am_b = pid_m * BLOCK_SIZE_M
        chunk_beg = offs_am_b // M_per_chunk
        chunk_end = (min(offs_am_b + BLOCK_SIZE_M, M) - 1) // M_per_chunk
        token = dl.wait(ready_ptr + chunk_beg, chunk_end - chunk_beg + 1, "gpu", "acquire", waitValue=ready_value)

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[None, :] * stride_bk + offs_bn[:, None] * stride_bn)
        a_ptrs = dl.consume_token(a_ptrs, token)
        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
        for k in range(0, num_k_blocks):
            # You can also put barriers here (but with performance drawback)
            # if needs_wait:
            #     num_barriers_to_wait = 1
            #     token = dl.wait(ready_ptr + ((k * BLOCK_SIZE_K) // K_per_barrier), num_barriers_to_wait, "gpu", "acquire")
            #     a_ptrs = dl.consume_token(a_ptrs, token)
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.trans(b)
            # We accumulate along the K dimension.
            accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        c = accumulator.to(dtype)
        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def ag_gemm_intra_node_op(a, b, c, rank, num_ranks, fullmesh_world_size, num_chunks_per_rank, workspace_tensors,
                          barrier_tensors, comm_buf, for_correctness=False, ag_stream=None, gemm_stream=None,
                          internode_ag_stream=None, serial=False, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, stages=4,
                          pipeline="cpasync", autotune=False, use_pull=True, signal_t=None):
    """no-tma allgather gemm for intra-node

    Allgather global matrix A and do matmul with local matrix B, produces local matrix C

    Args:
        a (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        b (torch.Tensor<float>): local matmul B matrix. shape: [N_per_rank, K]
        c (torch.Tensor<float>): local matmul C matrix. shape: [M, N_per_rank]
        rank (int): current rank
        num_ranks (int): total number of ranks
        workspace_tensors (List[torch.Tensor<float>]): A list of symm-tensors used for inter-rank allgather.
            Each tensor shape: [maxM, K]. Created by `create_ag_gemm_intra_node_context`.
        barrier_tensors (List[torch.Tensor<int32>]): A list of symm-tensors used for allgather.
            Each tensor shape: [num_ranks]. Created by `create_ag_gemm_intra_node_context`.
        comm_buf (torch.Tensor<int32>): A symm-tensor used for global synchronization.
            Shape: [MAX_NUM_BLOCKS_ON_GPU(65536)*num_ranks]. Created by `create_ag_gemm_intra_node_context`.
        for_correctness (bool, optional): if only for correctness, communication would sleep some seconds to
            trigger possible synchronization and dependency bugs. Defaults to False.
        ag_stream (torch.cuda.streams.Stream, optional): The stream used for allgather, if not provided, create a new one. Defaults to None.
        gemm_stream (torch.cuda.streams.Stream, optional): The stream used for gemm, if not provided, use current stream. Defaults to None.
        serial (bool, optional): Make the execution serialized, for debug. Defaults to False.
        BLOCK_M (int, optional): GEMM tiling factor for M dim. Defaults to 128.
        BLOCK_N (int, optional): GEMM tiling factor for N dim. Defaults to 256.
        BLOCK_K (int, optional): GEMM tiling factor for K dim. Defaults to 64.
        stages (int, optional): GEMM async-copy stages. Defaults to 3.
        autotune (bool, optional): whether to enable autotune. Defaults to False.

    Returns:
        Triton compiled code: used for debug
    """
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M_per_rank, K = a.shape
    M = M_per_rank * num_ranks
    N_per_rank, K = b.shape
    assert M_per_rank % BLOCK_M == 0, "M_per_rank should be divided by BLOCK_M"

    assert fullmesh_world_size > 1, "Cant find fullmesh nodes in topo"
    fullmesh_nodes_num = max(num_ranks // fullmesh_world_size, 1)

    ag_stream = torch.cuda.Stream() if ag_stream is None else ag_stream
    gemm_stream = torch.cuda.current_stream() if gemm_stream is None else gemm_stream
    current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(current_stream)
    gemm_stream.wait_stream(current_stream)
    internode_ag_stream.wait_stream(current_stream)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_per_rank, META["BLOCK_SIZE_N"]), )

    def call_ag(use_pull):
        if fullmesh_nodes_num == 1:
            cp_engine_producer_all_gather_full_mesh_push(
                rank,
                num_ranks,
                a,
                workspace_tensors,
                ag_stream,
                barrier_tensors,
                num_chunks_per_rank,
                for_correctness=for_correctness,
                signal_t=signal_t,
            )
        else:
            cp_engine_producer_all_gather_numa_node_push(
                rank,
                num_ranks,
                fullmesh_world_size,
                a,
                workspace_tensors,
                ag_stream,
                internode_ag_stream,
                barrier_tensors,
                num_chunks_per_rank,
                for_correctness=for_correctness,
                signal_t=signal_t,
            )

    if serial:
        call_ag(use_pull)
        current_stream.wait_stream(ag_stream)
        torch.cuda.synchronize()
    else:
        call_ag(use_pull)
    with torch.cuda.stream(gemm_stream):
        compiled = kernel_consumer_gemm[grid](
            workspace_tensors[rank][:M], b, c,  #
            a, rank, num_ranks, barrier_tensors[rank], M, N_per_rank, K,  #
            workspace_tensors[rank][:M].stride(0), workspace_tensors[rank][:M].stride(1), b.stride(1), b.stride(0),
            c.stride(0), c.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, 8, False, True, num_stages=stages, num_warps=4,
            pipeline=pipeline, scenario="noaddropt", num_chunks_per_rank=num_chunks_per_rank, nnodes=fullmesh_nodes_num,
            local_world_size=fullmesh_world_size)

    current_stream.wait_stream(ag_stream)
    current_stream.wait_stream(gemm_stream)

    return compiled


def ag_gemm_inter_node_op(a, b, c, rank, num_ranks, fullmesh_world_size, num_chunks_per_rank, workspace_tensors,
                          barrier_tensors, comm_buf, ag_stream=None, internode_ag_stream=None, gemm_stream=None,
                          BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, stages=4, pipeline="cpasync", local_world_size=8,
                          signal_target=1, signal_t=None, copy_engine_dispatch=False):
    """allgather gemm for inter-node

    Allgather global matrix A and do matmul with local matrix B, produces local matrix C

    Args:
        a (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        b (torch.Tensor<float>): local matmul B matrix. shape: [N_per_rank, K]
        c (torch.Tensor<float>): local matmul C matrix. shape: [M, N_per_rank]
        rank (int): current rank
        num_ranks (int): total number of ranks
        workspace_tensors (List[torch.Tensor<float>]): A list of symm-tensors used for inter-rank allgather.
            Each tensor shape: [maxM, K]. Created by `create_ag_gemm_intra_node_context`.
        barrier_tensors (List[torch.Tensor<int32>]): A list of symm-tensors used for allgather.
            Each tensor shape: [num_ranks]. Created by `create_ag_gemm_intra_node_context`.
        comm_buf (torch.Tensor<int32>): A symm-tensor used for global synchronization.
            Shape: [MAX_NUM_BLOCKS_ON_GPU(65536)*num_ranks]. Created by `create_ag_gemm_intra_node_context`.
        for_correctness (bool, optional): if only for correctness, communication would sleep some seconds to
            trigger possible synchronization and dependency bugs. Defaults to False.
        ag_stream (torch.cuda.streams.Stream, optional): The stream used for allgather, if not provided, create a new one. Defaults to None.
        gemm_stream (torch.cuda.streams.Stream, optional): The stream used for gemm, if not provided, use current stream. Defaults to None.
        serial (bool, optional): Make the execution serialized, for debug. Defaults to False.
        BLOCK_M (int, optional): GEMM tiling factor for M dim. Defaults to 128.
        BLOCK_N (int, optional): GEMM tiling factor for N dim. Defaults to 256.
        BLOCK_K (int, optional): GEMM tiling factor for K dim. Defaults to 64.
        stages (int, optional): GEMM async-copy stages. Defaults to 3.
        autotune (bool, optional): whether to enable autotune. Defaults to False.
        copy_engine_dispatch (bool, optional): whether to use copy enginer for intra-node dispatch. Defaults to True.

    Returns:
        Triton compiled code: used for debug
    """
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M_per_rank, K = a.shape
    M = M_per_rank * num_ranks
    N_per_rank, K = b.shape

    local_rank = rank % local_world_size
    n_nodes = num_ranks // local_world_size

    ag_stream = torch.cuda.Stream() if ag_stream is None else ag_stream
    gemm_stream = torch.cuda.current_stream() if gemm_stream is None else gemm_stream
    current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(current_stream)
    gemm_stream.wait_stream(current_stream)
    internode_ag_stream.wait_stream(current_stream)
    numa_nodes_num = max(local_world_size // fullmesh_world_size, 1)

    inter_node_allgather(a, workspace_tensors, barrier_tensors, fullmesh_world_size, signal_target, rank,
                         num_chunks_per_rank, local_world_size, num_ranks, ag_stream, internode_ag_stream,
                         copy_engine_dispatch, signal_t=signal_t, numanode_ag_stream=None)

    compiled = None
    with torch.cuda.stream(gemm_stream):

        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_per_rank, META["BLOCK_SIZE_N"]), )
        compiled = kernel_consumer_gemm[grid](
            workspace_tensors[local_rank][:M],
            b,
            c,  #
            a,
            rank,
            num_ranks,
            barrier_tensors[local_rank],
            M,
            N_per_rank,
            K,  #
            workspace_tensors[local_rank][:M].stride(0),
            workspace_tensors[local_rank][:M].stride(1),
            b.stride(1),
            b.stride(0),
            c.stride(0),
            c.stride(1),
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            8,
            False,
            True,
            ready_value=signal_target,
            num_stages=stages,
            num_warps=4,
            pipeline=pipeline,
            nnodes=n_nodes,
            local_world_size=local_world_size,
            num_chunks_per_rank=num_chunks_per_rank,
            nnumas=numa_nodes_num,
            numa_world_size=fullmesh_world_size,
        )

    current_stream.wait_stream(ag_stream)
    current_stream.wait_stream(gemm_stream)

    return compiled


def get_intranode_fullmesh_world_size(num_ranks):
    # get numa_world_size and fullmesh_world_size
    try:
        numa_world_size = get_numa_world_size()  # expensive so only do once
    except AssertionError:
        numa_world_size = num_ranks

    fullmesh_world_size = numa_world_size
    while (fullmesh_world_size > 1 and not has_fullmesh_mxlink_ngpus(fullmesh_world_size)):
        fullmesh_world_size //= 2

    return fullmesh_world_size


@dataclass
class AllGatherGEMMTensorParallelContext:
    rank: int
    num_ranks: int
    num_local_ranks: int
    local_rank: int
    num_chunks_per_rank: int
    workspace_tensors: List[torch.Tensor]
    barrier_tensors: List[torch.Tensor]
    fake_barrier_tensor: torch.Tensor
    comm_buf: torch.Tensor
    for_correctness: bool = False
    ag_stream: Optional[torch.cuda.streams.Stream] = None
    gemm_stream: Optional[torch.cuda.streams.Stream] = None
    internode_ag_stream: Optional[torch.cuda.streams.Stream] = None
    serial: bool = False
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    stages: int = 3
    autotune: bool = False
    fullmesh_world_size: int = 1

    def update(self, rank, num_ranks, num_local_ranks=8, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, stages=3,
               for_correctness=False, ag_stream=None, internode_ag_stream=None, gemm_stream=None, serial=False,
               autotune=False, fullmesh_world_size=1):
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_local_ranks = num_local_ranks
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.stages = stages
        self.for_correctness = for_correctness
        self.ag_stream = ag_stream
        self.internode_ag_stream = internode_ag_stream
        self.gemm_stream = gemm_stream
        self.serial = serial
        self.autotune = autotune
        self.fullmesh_world_size = fullmesh_world_size


def create_ag_gemm_intra_node_context(tensor_A, tensor_B, rank, num_ranks, max_M=2**14, max_blocks=65536, BLOCK_M=128,
                                      BLOCK_N=256, BLOCK_K=64, stages=3, num_chunks_per_rank=1, for_correctness=False,
                                      ag_stream=None, gemm_stream=None, serial=False, autotune=False, use_tma=False):
    """create context for allgather gemm intra-node

    Args:
        tensor_A (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        tensor_B (torch.Tensor<float>): local matmul B matrix. shape: [N_per_rank, K]
        rank (int): current rank
        num_ranks (int): total number of ranks
        max_M: max number of M shape
        max_blocks: max number of blocks on GPU
        BLOCK_M (int, optional): GEMM tiling factor for M dim. Defaults to 128.
        BLOCK_N (int, optional): GEMM tiling factor for N dim. Defaults to 256.
        BLOCK_K (int, optional): GEMM tiling factor for K dim. Defaults to 64.
        stages (int, optional): GEMM async-copy stages. Defaults to 3.
        for_correctness (bool, optional): if only for correctness, communication would sleep some seconds to
            trigger possible synchronization and dependency bugs. Defaults to False.
        ag_stream (torch.cuda.streams.Stream, optional): The stream used for allgather, if not provided, create a new one. Defaults to None.
        gemm_stream (torch.cuda.streams.Stream, optional): The stream used for gemm, if not provided, use current stream. Defaults to None.
        serial (bool, optional): Make the execution serialized, for debug. Defaults to False.
        autotune (bool, optional): whether to enable autotune. Defaults to False.

    Returns:
        AllGatherGEMMTensorParallelContext
    """
    M_per_rank, K = tensor_A.shape
    assert tensor_B.shape[
        1] == K, f"tensor_B should has shape (col_major) [N_per_rank, {K}], but get [{tensor_B.shape}]"
    assert tensor_A.dtype == tensor_B.dtype
    assert M_per_rank % num_chunks_per_rank == 0
    dtype = tensor_A.dtype
    fake_barrier = torch.ones([num_ranks], dtype=torch.int32, device=tensor_A.device)
    barriers = pymxshmem.mxshmem_create_tensor_list_intra_node([num_ranks * num_chunks_per_rank], torch.uint64)
    if use_tma:
        assert num_chunks_per_rank == 1
        comm_buf = pymxshmem.mxshmem_create_tensor([max_blocks * num_ranks], torch.int32)
        comm_buf.fill_(0)
    else:
        comm_buf = None
    # workspaces = pymxshmem.mxshmem_create_tensor_list_intra_node([max_M, K], dtype)
    symm_ag_a = symm_mem.empty([M_per_rank * num_ranks, K], dtype=dtype, device=tensor_A.device)
    symm_mem_hdl = symm_mem.rendezvous(symm_ag_a, group=dist.group.WORLD)
    workspaces = []
    for r in range(num_ranks):
        if r == rank:
            workspaces.append(symm_ag_a)
        else:
            workspaces.append(symm_mem_hdl.get_buffer(r, [M_per_rank * num_ranks, K], dtype, 0))

    barriers[rank].fill_(0)
    current_stream = torch.cuda.current_stream()
    pymxshmem.mxshmem_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    ret = AllGatherGEMMTensorParallelContext(
        rank=rank, num_ranks=num_ranks, local_rank=rank, num_chunks_per_rank=num_chunks_per_rank,
        num_local_ranks=num_ranks, workspace_tensors=workspaces, barrier_tensors=barriers,
        fake_barrier_tensor=fake_barrier, comm_buf=comm_buf, for_correctness=for_correctness, ag_stream=ag_stream,
        internode_ag_stream=torch.cuda.Stream(), gemm_stream=gemm_stream, serial=serial, BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, stages=stages, autotune=autotune,
        fullmesh_world_size=get_intranode_fullmesh_world_size(num_ranks))

    return ret


def ag_gemm_intra_node(a, b, ctx=None, rank=None, num_ranks=None, use_tma=True):
    """allgather gemm for intra-node

    Allgather global matrix A and do matmul with local matrix B, produces local matrix C

    Args:
        a (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        b (torch.Tensor<float>): local matmul B matrix. shape: [N_per_rank, K]
        ctx: (Optional[AllGatherGEMMTensorParallelContext]): if not provided, created immediately

    Returns:
        c (torch.Tensor<float>): local matmul C matrix. shape: [M, N_per_rank]
    """
    if ctx is None:
        assert rank is not None and num_ranks is not None
        ctx = create_ag_gemm_intra_node_context(a, b, rank, num_ranks)

    M_per_rank, K = a.shape
    N_per_rank, _ = b.shape
    C = torch.empty([ctx.num_ranks * M_per_rank, N_per_rank], dtype=a.dtype, device=a.device)

    use_pull = False  # default cp_engine_allgather_push
    ctx.barrier_tensors[ctx.rank].fill_(0)
    current_stream = torch.cuda.current_stream()

    signal_t = 1
    ctx.barrier_tensors[ctx.rank][ctx.rank * ctx.num_chunks_per_rank:(ctx.rank + 1) *
                                  ctx.num_chunks_per_rank].fill_(signal_t)
    # flush L2
    pymxshmem.flush_l2c(current_stream.cuda_stream)
    # make sure local copy and flush_l2c finished on all devices
    pymxshmem.mxshmem_barrier_all_on_stream(current_stream.cuda_stream)

    ag_gemm_intra_node_op(a, b, C, ctx.rank, ctx.num_ranks, ctx.fullmesh_world_size, ctx.num_chunks_per_rank,
                          ctx.workspace_tensors, ctx.barrier_tensors, ctx.comm_buf, for_correctness=ctx.for_correctness,
                          ag_stream=ctx.ag_stream, gemm_stream=ctx.gemm_stream,
                          internode_ag_stream=ctx.internode_ag_stream, serial=ctx.serial, autotune=ctx.autotune,
                          use_pull=use_pull, signal_t=signal_t)

    pymxshmem.mxshmem_barrier_all_on_stream(current_stream.cuda_stream)

    return C


def create_ag_gemm_inter_node_context(tensor_A, tensor_B, rank, num_ranks, num_local_ranks=8, max_M=2**14,
                                      max_blocks=65536, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, stages=3,
                                      num_chunks_per_rank=1, for_correctness=False, ag_stream=None, gemm_stream=None,
                                      serial=False, autotune=False, use_tma=False):
    """create context for allgather gemm inter-node

    Args:
        tensor_A (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        tensor_B (torch.Tensor<float>): local matmul B matrix. shape: [N_per_rank, K]
        rank (int): current rank
        num_ranks (int): total number of ranks
        max_M: max number of M shape
        max_blocks: max number of blocks on GPU
        BLOCK_M (int, optional): GEMM tiling factor for M dim. Defaults to 128.
        BLOCK_N (int, optional): GEMM tiling factor for N dim. Defaults to 256.
        BLOCK_K (int, optional): GEMM tiling factor for K dim. Defaults to 64.
        stages (int, optional): GEMM async-copy stages. Defaults to 3.
        ag_stream (torch.cuda.streams.Stream, optional): The stream used for allgather, if not provided, create a new one. Defaults to None.
        gemm_stream (torch.cuda.streams.Stream, optional): The stream used for gemm, if not provided, use current stream. Defaults to None.
        serial (bool, optional): Make the execution serialized, for debug. Defaults to False.
        autotune (bool, optional): whether to enable autotune. Defaults to False.

    Returns:
        AllGatherGEMMTensorParallelContext
    """
    M_per_rank, K = tensor_A.shape
    assert tensor_B.shape[
        1] == K, f"tensor_B should has shape (col_major) [N_per_rank, {K}], but get [{tensor_B.shape}]"
    assert tensor_A.dtype == tensor_B.dtype
    assert M_per_rank % num_chunks_per_rank == 0
    dtype = tensor_A.dtype

    local_rank = rank % num_local_ranks

    fake_barrier = torch.ones([num_ranks], dtype=torch.int32, device=tensor_A.device)
    workspaces = pymxshmem.mxshmem_create_tensor_list_intra_node([M_per_rank * num_ranks, K], dtype)
    barriers = pymxshmem.mxshmem_create_tensor_list_intra_node([num_ranks * num_chunks_per_rank], torch.uint64)

    if use_tma:
        assert num_chunks_per_rank == 1
        comm_buf = pymxshmem.mxshmem_create_tensor([max_blocks * num_ranks], torch.int32)
        comm_buf.fill_(0)
    else:
        comm_buf = None
    barriers[local_rank].fill_(0)
    current_stream = torch.cuda.current_stream()
    pymxshmem.mxshmem_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    ret = AllGatherGEMMTensorParallelContext(
        rank=rank, num_ranks=num_ranks, local_rank=local_rank, num_chunks_per_rank=num_chunks_per_rank,
        num_local_ranks=num_local_ranks, workspace_tensors=workspaces, barrier_tensors=barriers,
        fake_barrier_tensor=fake_barrier, comm_buf=comm_buf, for_correctness=for_correctness, ag_stream=ag_stream,
        internode_ag_stream=torch.cuda.Stream(), gemm_stream=gemm_stream, serial=serial, BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, stages=stages, autotune=autotune,
        fullmesh_world_size=get_intranode_fullmesh_world_size(num_local_ranks))

    return ret


def ag_gemm_inter_node(a, b, ctx=None, rank=None, num_ranks=None, local_world_size=8, signal_target=1, use_tma=True):
    """allgather gemm for inter-node

    Allgather global matrix A and do matmul with local matrix B, produces local matrix C

    Args:
        a (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        b (torch.Tensor<float>): local matmul B matrix. shape: [N_per_rank, K]
        ctx: (Optional[AllGatherGEMMTensorParallelContext]): if not provided, created immediately

    Returns:
        c (torch.Tensor<float>): local matmul C matrix. shape: [M, N_per_rank]
    """
    if ctx is None:
        assert rank is not None and num_ranks is not None
        ctx = create_ag_gemm_inter_node_context(a, b, rank, num_ranks)
    M_per_rank, K = a.shape
    N_per_rank, _ = b.shape
    C = torch.empty([ctx.num_ranks * M_per_rank, N_per_rank], dtype=a.dtype, device=a.device)
    local_rank = ctx.rank % local_world_size
    ctx.barrier_tensors[local_rank].fill_(0)
    current_stream = torch.cuda.current_stream()
    pymxshmem.mxshmem_barrier_all_on_stream(current_stream.cuda_stream)
    ctx.workspace_tensors[local_rank][ctx.rank * M_per_rank:(ctx.rank + 1) * M_per_rank, :].copy_(a)
    signal_t = 1
    ctx.barrier_tensors[local_rank][ctx.rank * ctx.num_chunks_per_rank:(ctx.rank + 1) *
                                    ctx.num_chunks_per_rank].fill_(signal_t)

    # flush L2
    pymxshmem.flush_l2c(current_stream.cuda_stream)
    # make sure local copy and flush_l2c finished on all devices
    pymxshmem.mxshmem_barrier_all_on_stream(current_stream.cuda_stream)

    ag_gemm_inter_node_op(a, b, C, ctx.rank, ctx.num_ranks, ctx.fullmesh_world_size, ctx.num_chunks_per_rank,
                          ctx.workspace_tensors, ctx.barrier_tensors, ctx.comm_buf, ag_stream=ctx.ag_stream,
                          internode_ag_stream=ctx.internode_ag_stream, gemm_stream=ctx.gemm_stream,
                          local_world_size=local_world_size, signal_target=signal_target, signal_t=signal_t,
                          copy_engine_dispatch=True)

    return C


@triton.jit
def kernel_consumer_gemm_triton(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    is_fp8: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    dtype = tl.float16 if not is_fp8 else tl.float8e4nv
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[None, :] * stride_bk + offs_bn[:, None] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_blocks):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.trans(b)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def gemm(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, stages=4, pipeline="cpasync", autotune=False):
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, _ = b.shape
    c = torch.empty([M, N], dtype=a.dtype, device=a.device)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    kernel_consumer_gemm_triton[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        c.stride(0),
        c.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        8,
        False,
        num_stages=stages,
        pipeline="cpasync",
        num_warps=4,
    )

    return c
