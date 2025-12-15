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
Inter-node AllGather
====================
In this tutorial, you will write a low latency all gather kernel using using Triton-distributed.

.. code-block:: bash

    # To run this tutorial
    source ./scripts/setenv.sh
    bash ./scripts/launch.sh tutorials/03-inter-node-allgather.py

"""
import os
from dataclasses import dataclass

import torch

import triton_dist
import triton.language as tl
from triton_dist.language.extra.language_extra import __syncthreads, tid
from triton_dist.language.extra import libshmem_device
from triton_dist.profiler_utils import perf_func
from triton_dist.utils import finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream, NVSHMEM_SIGNAL_DTYPE, nvshmem_free_tensor_sync, nvshmem_create_tensor, sleep_async


@dataclass
class AllGatherContext:
    rank: int
    node: int
    num_ranks: int
    num_nodes: int
    symm_signals: torch.Tensor
    signal_value: int = 15
    max_buffer_size: int = 2 * 32 * 1024 * 1024


@triton_dist.jit(do_not_specialize=["rank", "signal_value"])
def all_gather_push_1d_kernel(symm_ptr, bytes_per_rank, symm_flag,
                              WORLD_SIZE: tl.constexpr, rank, signal_value):
    pid = tl.program_id(0)
    thread_idx = tid(0)
    # there are WORLD_SIZE programs processing different data.
    if pid == rank:  # 1 program waitings for all other peers putmem done
        peer = thread_idx
        if peer < WORLD_SIZE and peer != rank:
            # rank `peer` is responsible for putmem segment `peer`.
            # wait for rank `peer` putmem done, then quit the kernel
            libshmem_device.signal_wait_until(
                symm_flag + peer,
                libshmem_device.NVSHMEM_CMP_EQ,
                signal_value,
            )
        # wait for all peers done
        __syncthreads()
    else:  # other `WORLD_SIZE - 1` programs putmem with signals to other peers
        peer = pid
        segment = rank
        # libshmem_device.putmem_signal_block calls NVSHMEM API `nvshmemx_putmem_signal_block`, which
        # sends `segment` at `rank` to `segment` at `peer` and set `flag[segment]` at `peer` to `signal_value`
        # NVSHMEM takes care of memory barrier semantic, so don't worry.
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) +
            segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) +
            segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_value,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )  # write and tell peer remote that remote copy is done


def all_gather_push_1d(ctx: AllGatherContext, symm_buffer: torch.Tensor):
    ctx.signal_value += 1
    all_gather_push_1d_kernel[(ctx.num_ranks, )](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ctx.symm_signals[ctx.signal_value % 2],
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_value,
    )
    return symm_buffer


@triton_dist.jit(do_not_specialize=["rank", "signal_value"])
def all_gather_push_2d_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_flag,
    NNODES: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    rank,
    signal_value,
):
    pid = tl.program_id(0)
    thread_idx = tid(0)
    # Here we represent rank or segment in 2d fashion:
    #  (node_id, local_rank) => node_id * local_world_size + local_rank = rank
    LOCAL_WORLD_SIZE = WORLD_SIZE // NNODES
    node_id = rank // LOCAL_WORLD_SIZE
    local_rank = rank % LOCAL_WORLD_SIZE

    peer_rank = pid
    peer_node_id = peer_rank // LOCAL_WORLD_SIZE
    peer_local_rank = peer_rank % LOCAL_WORLD_SIZE
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    if peer_local_rank == local_rank:
        if peer_rank != rank:  # cross NODE communication
            # send segment `(node_id, local_rank)` at rank `(node_id, local_rank)` to segment `(node_id, local_rank)` at rank `(peer_node_id, local_rank)`
            peer = peer_node_id * LOCAL_WORLD_SIZE + local_rank
            segment = rank
            libshmem_device.putmem_signal_nbi_block(
                symm_ptr + segment * bytes_per_rank,
                symm_ptr + segment * bytes_per_rank,
                bytes_per_rank,
                symm_flag + segment,
                signal_value,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )
        else:  # wait for all peers done done
            if thread_idx < WORLD_SIZE and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_value,
                )
            __syncthreads()
    else:  # intra-NODE communication
        peer = node_id * LOCAL_WORLD_SIZE + peer_local_rank
        segment = peer_node_id * LOCAL_WORLD_SIZE + local_rank
        # wait for inter-NODE putmem_signal done from other nodes
        if peer_node_id != node_id:
            if thread_idx == 0:
                libshmem_device.signal_wait_until(
                    symm_flag + segment,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_value,
                )
            __syncthreads()
        # send segment (i, local_rank) to (i, peer_local_rank) for all i and local_rank != peer_rank
        libshmem_device.putmem_signal_block(
            symm_ptr + segment * bytes_per_rank,
            symm_ptr + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_value,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


def all_gather_push_2d(ctx: AllGatherContext, symm_buffer: torch.Tensor):
    ctx.signal_value += 1
    all_gather_push_2d_kernel[(ctx.num_ranks, )](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ctx.symm_signals[ctx.signal_value % 2],
        ctx.num_nodes,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_value,
        num_warps=32,  # use as many threads as possible
    )

    return symm_buffer


def perf_ag(func, ag_buffers: torch.Tensor, nbytes: int,
            ctx: AllGatherContext):
    nbytes_per_rank = nbytes // WORLD_SIZE

    ref_tensor = torch.arange(nbytes, dtype=torch.int8).cuda()
    ref_tensor = (torch.randint(0, 9999, [nbytes // 4],
                                dtype=torch.int32).view(torch.int8).cuda())
    torch.distributed.broadcast(ref_tensor, src=0)

    # local copy
    ag_buffer = ag_buffers[ctx.signal_value % 2]
    # suppose You already write to ag_bufferp[index_start:index_end], so here copy does not count in profile
    index_start, index_end = nbytes_per_rank * RANK, nbytes_per_rank * (RANK +
                                                                        1)
    ag_buffer[index_start:index_end].copy_(ref_tensor[index_start:index_end])

    def _run_all_gather_triton():
        ag_buffer = ag_buffers[ctx.signal_value % 2][:nbytes]
        return func(ctx, ag_buffer)

    def _run_all_gather_nccl():
        torch.distributed.all_gather_into_tensor(
            ref_tensor, ref_tensor[index_start:index_end], group=TP_GROUP)

    result = _run_all_gather_triton()

    # verify
    torch.testing.assert_close(result, ref_tensor, atol=0, rtol=0)
    print(f"✅ RANK[{RANK}] check passed")

    # perf all-gather by NCCL
    sleep_async(1000)  # in case CPU bound
    _, duration_per_iter_ms = perf_func(
        _run_all_gather_nccl,
        warmup_iters=5,
        iters=10,
    )
    gbps = nbytes * 1e-9 / (duration_per_iter_ms * 1e-3) * (WORLD_SIZE -
                                                            1) / WORLD_SIZE
    print(
        f"[NCCL] RANK = {RANK}, {nbytes // 1024} KB, Latency {duration_per_iter_ms * 1000:0.2f} us, Bus bandwith = {gbps:0.2f} GB/S"
    )

    # perf all-gather by triton-distributed
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    sleep_async(1000)  # in case CPU bound
    _, duration_per_iter_ms = perf_func(
        _run_all_gather_triton,
        warmup_iters=5,
        iters=10,
    )

    gbps = nbytes * 1e-9 / (duration_per_iter_ms * 1e-3) * (WORLD_SIZE -
                                                            1) / WORLD_SIZE
    print(
        f"[Triton] RANK = {RANK}, {nbytes // 1024} KB, Latency {duration_per_iter_ms * 1000:0.2f} us, Bus bandwith = {gbps:0.2f} GB/S"
    )


# get all distributed arguments from environment. which is set by torchrun
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE
TP_GROUP = initialize_distributed()

nbytes = 8 * 1024  # total bytes for AllGather

# use nvshmem_create_tensor as a torch-friendly wrapper of nvshmem_malloc.
# since our implementation does not wait for other peers done, so a double buffer is
# used to avoid data corupt when all_gather kernels are in different phases.
symm_ag_buffer = nvshmem_create_tensor((2, nbytes), torch.int8)

# keep some variables here
ctx = AllGatherContext(
    rank=TP_GROUP.rank(),
    node=RANK // LOCAL_WORLD_SIZE,
    num_ranks=WORLD_SIZE,
    num_nodes=NNODES,
    symm_signals=[
        nvshmem_create_tensor((1, ), NVSHMEM_SIGNAL_DTYPE) for _ in range(2)
    ],
    signal_value=10,
)
print("using push 1d...")
perf_ag(
    all_gather_push_1d,
    symm_ag_buffer,
    nbytes,
    ctx,
)
print("using push 2d...")
perf_ag(
    all_gather_push_2d,
    symm_ag_buffer,
    nbytes,
    ctx,
)
nvshmem_free_tensor_sync(symm_ag_buffer)
nvshmem_free_tensor_sync(ctx.symm_signals[0])
nvshmem_free_tensor_sync(ctx.symm_signals[1])

finalize_distributed()
