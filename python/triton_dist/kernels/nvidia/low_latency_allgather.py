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
from dataclasses import dataclass
from typing import List

import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device

from triton.language.extra.cuda.language_extra import (
    __syncthreads,
    pack_b32_v2,
    tid,
    ntid,
    load_v4_u32,
    load_v2_b64,
    st_v2_u32,
    st,
    multimem_st_b64,
)
from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE, nvshmem_free_tensor_sync, nvshmem_create_tensor


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_pull_kernel(symm_ptr, bytes_per_rank, symm_flag, world_size, rank, signal_target):
    pid = tl.program_id(0)
    thread_idx = tid(0)
    if pid == rank:
        if thread_idx != rank and thread_idx < world_size:
            libshmem_device.signal_op(
                symm_flag + rank,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                thread_idx,
            )
        __syncthreads()
    else:
        peer = pid
        if thread_idx == 0:
            libshmem_device.signal_wait_until(symm_flag + peer, libshmem_device.NVSHMEM_CMP_EQ, signal_target)
        __syncthreads()
        libshmem_device.getmem_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + peer * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + peer * bytes_per_rank,
            bytes_per_rank,
            peer,
        )


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_numa_2d_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_flag,
    n_numa_nodes: tl.constexpr,
    world_size: tl.constexpr,
    rank,
    signal_target,
):
    tl.static_assert(n_numa_nodes == 2, "only support NUMA node == 2")
    numa_world_size = world_size // n_numa_nodes
    local_rank = rank % numa_world_size
    nid = rank // numa_world_size

    pid = tl.program_id(0)
    peer_nid = pid // numa_world_size
    peer_local_rank = pid % numa_world_size
    thread_idx = tid(0)

    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    if peer_local_rank == local_rank:  # remote push
        if peer_nid != nid:  # pnid: peer node id. each block recv from pnid
            peer_to = peer_nid * numa_world_size + local_rank
            libshmem_device.putmem_signal_block(
                symm_ptr + rank * bytes_per_rank,
                symm_ptr + rank * bytes_per_rank,
                bytes_per_rank,
                symm_flag + rank,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer_to,
            )  # write and tell peer remote that remote copy is done
        else:  # pack ll data
            # wait for all write done
            if thread_idx < world_size and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()
    else:  # local push
        peer = nid * numa_world_size + peer_local_rank
        segment = peer_nid * numa_world_size + local_rank
        if peer_nid != nid:  # wait for recv_ll done
            if thread_idx == 0:
                libshmem_device.signal_wait_until(symm_flag + segment, libshmem_device.NVSHMEM_CMP_EQ, signal_target)
            __syncthreads()
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )  # write and tell peer remote that remote copy is done


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_numa_2d_ll_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_flag,
    symm_ll_buffer,
    n_numa_nodes: tl.constexpr,
    world_size: tl.constexpr,
    rank,
    signal_target,
):
    tl.static_assert(n_numa_nodes == 2, "only support NUMA node == 2")
    numa_world_size = world_size // n_numa_nodes
    local_rank = rank % numa_world_size
    nid = rank // numa_world_size

    pid = tl.program_id(0)
    peer_nid = pid // numa_world_size
    peer_local_rank = pid % numa_world_size
    thread_idx = tid(0)
    num_ints = bytes_per_rank // 4

    symm_ll_buffer = tl.cast(symm_ll_buffer, tl.pointer_type(tl.int8))
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    if peer_local_rank == local_rank:  # remote push
        if peer_nid != nid:  # pnid: peer node id. each block recv from pnid
            segment = peer_nid * numa_world_size + local_rank
            _recv_ll_block(
                symm_ptr + segment * bytes_per_rank,
                symm_ll_buffer + segment * bytes_per_rank * 2,
                num_ints,
                signal_target,
            )  # magic number here
            __syncthreads()
            if thread_idx == 0:
                st(symm_flag + segment, signal_target, scope="gpu", semantic="release")
        else:  # pack ll data
            _pack_ll_block(
                symm_ll_buffer + rank * bytes_per_rank * 2,
                symm_ptr + rank * bytes_per_rank,
                num_ints,
                signal_target,
                2048,
            )  # magic number here
            __syncthreads()

            peer_to_nid = 1 - nid  # only for n_numa_nodes == 2
            peer_to = peer_to_nid * numa_world_size + local_rank
            libshmem_device.putmem_block(
                symm_ll_buffer + rank * bytes_per_rank * 2,
                symm_ll_buffer + rank * bytes_per_rank * 2,
                bytes_per_rank * 2,
                peer_to,
            )  # write and tell peer remote that remote copy is done
            # wait for all write done
            if thread_idx < world_size and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()

    else:  # local push
        peer = nid * numa_world_size + peer_local_rank
        segment = peer_nid * numa_world_size + local_rank
        if peer_nid != nid:  # wait for recv_ll done
            if thread_idx == 0:
                libshmem_device.signal_wait_until(symm_flag + segment, libshmem_device.NVSHMEM_CMP_EQ, signal_target)
            __syncthreads()
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )  # write and tell peer remote that remote copy is done


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_numa_2d_ll_multinode_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_flag,
    symm_ll_buffer,
    nnodes: tl.constexpr,
    n_numa_nodes: tl.constexpr,
    world_size: tl.constexpr,
    rank,
    signal_target,
):
    # the communication pattern
    #  for rank (node_id, numa_id, numa_rank) with (i0, j0, k0)
    #  (i0, j0, k0) => (i0, j0, k_x) with intra NUMA communication with nvshmem_putmem_signal_block
    #  (i0, j1, k0) => (i0, j_x, k0) with inter NUMA communication with nvshmem_putmem_signal_block
    #  (i0, j1, k0) => (i_x, j0, k0) with intra NODE communication with nvshmem_putmem_nbi_warp with LL protocol
    # BUT the difference: NIC can be done with nbi, but NUMA can't (or better not)
    tl.static_assert(n_numa_nodes == 2, "only support NUMA node == 2")
    local_world_size = world_size // nnodes
    node_id = rank // local_world_size
    local_rank = rank % local_world_size
    numa_world_size = local_world_size // n_numa_nodes
    numa_rank = local_rank % numa_world_size
    local_numa_id = local_rank // numa_world_size
    global_numa_id = rank // numa_world_size

    pid = tl.program_id(0)
    peer_node_id = pid // local_world_size
    peer_local_rank = pid % local_world_size
    peer_numa_rank = peer_local_rank % numa_world_size
    peer_local_numa_id = peer_local_rank // numa_world_size
    peer_global_numa_id = pid // numa_world_size
    thread_idx = tid(0)
    num_ints = bytes_per_rank // 4

    symm_ll_buffer = tl.cast(symm_ll_buffer, tl.pointer_type(tl.int8))
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    # (i0, j0, k0) 2 conditions to put internode => (node_id, local_numa_id, numa_rank)
    #  1. send intra NUMA (i0, j0, k_x) x!=0
    #  2. send inter NUMA (i0, j_x, k0) x!=0
    is_intra_numa = numa_rank != peer_numa_rank
    is_inter_numa = node_id == peer_node_id and (local_numa_id != peer_local_numa_id and numa_rank == peer_numa_rank)

    if (is_intra_numa and global_numa_id == peer_global_numa_id):  # no need to wait, just send
        peer = global_numa_id * numa_world_size + peer_numa_rank
        segment = rank
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )  # write and tell peer remote that remote copy is done
    elif is_intra_numa and global_numa_id != peer_global_numa_id:
        peer = global_numa_id * numa_world_size + peer_numa_rank
        segment = (peer_node_id * local_world_size + peer_local_numa_id * numa_world_size + numa_rank)
        # wait for segment ready
        if thread_idx == 0:
            libshmem_device.signal_wait_until(symm_flag + segment, libshmem_device.NVSHMEM_CMP_EQ, signal_target)
        __syncthreads()
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )  # write and tell peer remote that remote copy is done
    elif is_inter_numa:
        peer = (node_id * local_world_size + peer_local_numa_id * numa_world_size + peer_numa_rank)
        segment = rank
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )  # write and tell peer remote that remote copy is done
    else:  # remote push
        if peer_node_id != node_id:  # pnid: peer node id. each block recv from pnid
            segment = peer_global_numa_id * numa_world_size + peer_numa_rank
            _recv_ll_block(
                symm_ptr + segment * bytes_per_rank,
                symm_ll_buffer + segment * bytes_per_rank * 2,
                num_ints,
                signal_target,
            )  # magic number here
            __syncthreads()

            if thread_idx == 0:
                st(symm_flag + segment, signal_target, scope="gpu", semantic="release")
        else:  # pack ll data and send to peer
            _pack_ll_block(
                symm_ll_buffer + rank * bytes_per_rank * 2,
                symm_ptr + rank * bytes_per_rank,
                num_ints,
                signal_target,
                2048,
            )  # magic number here
            __syncthreads()

            for i in range(world_size // numa_world_size):
                if i // n_numa_nodes != node_id:  # only send to other nodes
                    peer_to = numa_rank + i * numa_world_size
                    libshmem_device.putmem_nbi_warp(
                        symm_ll_buffer + rank * bytes_per_rank * 2,
                        symm_ll_buffer + rank * bytes_per_rank * 2,
                        bytes_per_rank * 2,
                        peer_to,
                    )  # write and tell peer remote that remote copy is done

            # wait for all write done
            if thread_idx < world_size and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_2d_kernel(symm_ptr, bytes_per_rank, symm_flag, NNODES, WORLD_SIZE, rank, signal_target):
    LOCAL_WORLD_SIZE = WORLD_SIZE // NNODES
    local_rank = rank % LOCAL_WORLD_SIZE
    node_id = rank // LOCAL_WORLD_SIZE
    rank_base = node_id * LOCAL_WORLD_SIZE

    pid = tl.program_id(0)
    peer_rank = pid
    peer_node_id = peer_rank // LOCAL_WORLD_SIZE
    peer_local_rank = peer_rank % LOCAL_WORLD_SIZE
    thread_idx = tid(0)
    if peer_local_rank == local_rank:  # remote push
        if peer_rank != rank:
            peer = peer_node_id * LOCAL_WORLD_SIZE + local_rank
            segment = rank
            libshmem_device.putmem_signal_nbi_block(
                tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
                tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
                bytes_per_rank,
                symm_flag + segment,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )  # write and tell peer remote that remote copy is done
        else:
            if thread_idx < WORLD_SIZE and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()
    else:  # local push
        peer = rank_base + peer_local_rank
        segment = peer_node_id * LOCAL_WORLD_SIZE + local_rank
        if peer_node_id != node_id:  # wait for data from other nodes
            if thread_idx == 0:
                libshmem_device.signal_wait_until(
                    symm_flag + segment,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )  # write and tell peer remote that remote copy is done


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_3d_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_ll_buffer,
    symm_flag,
    NNODES,
    N_NUMA_NODES,
    WORLD_SIZE,
    rank,
    signal_target,
    INTER_NODE_WITH_LL: tl.constexpr = False,
):
    """inter-node / inter-NUMA / intra-NUMA"""
    LOCAL_WORLD_SIZE = WORLD_SIZE // NNODES
    NUMA_WORLD_SIZE = LOCAL_WORLD_SIZE // N_NUMA_NODES
    local_rank = rank % LOCAL_WORLD_SIZE
    node_id = rank // LOCAL_WORLD_SIZE
    numa_rank = local_rank % NUMA_WORLD_SIZE
    local_numa_id = local_rank // NUMA_WORLD_SIZE

    pid = tl.program_id(0)
    peer_rank = pid
    peer_node_id = peer_rank // LOCAL_WORLD_SIZE
    peer_local_rank = peer_rank % LOCAL_WORLD_SIZE
    peer_numa_rank = peer_local_rank % NUMA_WORLD_SIZE
    peer_local_numa_id = peer_local_rank // NUMA_WORLD_SIZE

    thread_idx = tid(0)
    num_ints = bytes_per_rank // 4
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))
    symm_ll_buffer = tl.cast(symm_ll_buffer, tl.pointer_type(tl.int8))
    if peer_local_rank == local_rank:
        if peer_node_id != node_id:
            if INTER_NODE_WITH_LL:
                segment = peer_node_id * LOCAL_WORLD_SIZE + local_rank
                _recv_ll_block(
                    symm_ptr + segment * bytes_per_rank,
                    symm_ll_buffer + segment * bytes_per_rank * 2,
                    num_ints,
                    signal_target,
                )
                __syncthreads()
                if thread_idx == 0:
                    st(symm_flag + segment, signal_target, scope="gpu", semantic="release")
        else:
            wid = thread_idx // 32
            if INTER_NODE_WITH_LL:
                segment = rank
                _pack_ll_block(symm_ll_buffer + rank * bytes_per_rank * 2, symm_ptr + rank * bytes_per_rank, num_ints,
                               signal_target, 2048)
                __syncthreads()

                if wid < NNODES and wid != node_id:
                    peer = wid * LOCAL_WORLD_SIZE + local_rank
                    libshmem_device.putmem_nbi_warp(
                        symm_ll_buffer + segment * bytes_per_rank * 2,
                        symm_ll_buffer + segment * bytes_per_rank * 2,
                        bytes_per_rank * 2,
                        peer,
                    )  # write and tell peer remote that remote copy is done
            else:
                if wid < NNODES and wid != node_id:
                    peer = wid * LOCAL_WORLD_SIZE + local_rank
                    segment = rank
                    libshmem_device.putmem_signal_nbi_warp(
                        symm_ptr + segment * bytes_per_rank,
                        symm_ptr + segment * bytes_per_rank,
                        bytes_per_rank,
                        symm_flag + segment,
                        signal_target,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        peer,
                    )  # write and tell peer remote that remote copy is done

            __syncthreads()
            if thread_idx < WORLD_SIZE and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()
    else:  # local push with NUMA opt
        # NIC consume all the PCI-e bandwidth. don't overlap with inter-NODE communication
        # inter/intra NODE communication overlap pattern is too complex.
        if NNODES > 1:  # no if for single node.
            if thread_idx < WORLD_SIZE and (thread_idx % LOCAL_WORLD_SIZE == local_rank and thread_idx != rank):
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()

        if peer_numa_rank == numa_rank:  # NUMA write
            peer = (node_id * LOCAL_WORLD_SIZE + peer_local_numa_id * NUMA_WORLD_SIZE + numa_rank)
            segment = peer_node_id * LOCAL_WORLD_SIZE + local_rank
            libshmem_device.putmem_signal_block(
                symm_ptr + segment * bytes_per_rank,
                symm_ptr + segment * bytes_per_rank,
                bytes_per_rank,
                symm_flag + segment,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )
        else:
            peer = (node_id * LOCAL_WORLD_SIZE + local_numa_id * NUMA_WORLD_SIZE + peer_numa_rank)
            segment = (peer_node_id * LOCAL_WORLD_SIZE + peer_local_numa_id * NUMA_WORLD_SIZE + numa_rank)

            if peer_local_numa_id != local_numa_id:  # wait for data from other NUMA
                if thread_idx == 0:
                    libshmem_device.signal_wait_until(
                        symm_flag + segment,
                        libshmem_device.NVSHMEM_CMP_EQ,
                        signal_target,
                    )
                __syncthreads()

            libshmem_device.putmem_signal_block(
                symm_ptr + segment * bytes_per_rank,
                symm_ptr + segment * bytes_per_rank,
                bytes_per_rank,
                symm_flag + segment,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )


@triton.jit
def _recv_ll_block(dest_ptr, src_ptr, num_ints, ll_flag):
    """split src/dest outside of _recv_ll. this function is designed for a threadblock

    num_ints: of the pre-LL-packed num_ints.
    """
    thread_idx = tid(0)
    block_size = ntid(0)
    src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.int32))
    dest_ptr = tl.cast(dest_ptr, tl.pointer_type(tl.int32))
    # manual load per vec
    for n in range(thread_idx, num_ints // 2, block_size):
        data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        while flag1 != ll_flag or flag2 != ll_flag:
            data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        st_v2_u32(dest_ptr + n * 2, data1, data2)


@triton.jit(do_not_specialize=["ll_flag"])
def _pack_ll_block(dest_ptr, src_ptr, num_ints, ll_flag, BLOCK_SIZE: tl.constexpr):
    """split src/dest outside of _recv_ll. this function is designed for a threadblock

    nbytes: of the pre-LL-packed bytes.
    BLOCK_SIZE: count by ints, not bytes.
    """
    iters = tl.cdiv(num_ints, BLOCK_SIZE)
    src_ptr = tl.cast(src_ptr, dtype=tl.pi32_t)
    dest_ptr = tl.cast(dest_ptr, dtype=tl.pi32_t)
    for n in range(iters):
        src_offsets = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        src_mask = src_offsets < num_ints
        src = tl.load(src_ptr + src_offsets, mask=src_mask)
        flags = tl.full((BLOCK_SIZE, ), ll_flag, tl.int32)
        dst = tl.interleave(src, flags)
        dest_offset = n * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2)
        dest_mask = dest_offset < num_ints * 2
        tl.store(dest_ptr + dest_offset, dst, mask=dest_mask)


@triton.jit
def _recv_ll_and_multimem_st_block(dest_ptr, src_ptr, num_ints, ll_flag):
    """split src/dest outside of _recv_ll. this function is designed for a threadblock

    num_ints: of the pre-LL-packed num_ints.
    """
    thread_idx = tid(0)
    block_size = ntid(0)
    src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.int32))
    dest_ptr = tl.cast(dest_ptr, tl.pointer_type(tl.int32))
    dest_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, dest_ptr)
    # manual load per vec
    for n in range(thread_idx, num_ints // 2, block_size):
        data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        while flag1 != ll_flag or flag2 != ll_flag:
            data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        multimem_st_b64(dest_mc_ptr + n * 2, pack_b32_v2(data1, data2))


@triton.jit(do_not_specialize=["ll_flag"])
def _recv_ll_and_multimem_st_ll_block(dest_ptr, src_ptr, num_ints, ll_flag):
    """split src/dest outside of _recv_ll. this function is designed for a threadblock

    num_ints: of the pre-LL-packed num_ints.
    """
    thread_idx = tid(0)
    block_size = ntid(0)
    src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.int32))
    dest_ptr = tl.cast(dest_ptr, tl.pointer_type(tl.int32))
    dest_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, dest_ptr)
    # manual load per vec
    for n in range(thread_idx, num_ints // 2, block_size):
        data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        while flag1 != ll_flag or flag2 != ll_flag:
            data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        multimem_st_b64(dest_mc_ptr + n * 4, pack_b32_v2(data1, flag1))
        multimem_st_b64(dest_mc_ptr + n * 4 + 2, pack_b32_v2(data2, flag2))


@triton.jit
def broadcast_naive_block(dst_ptr, src_ptr, nbytes):
    thread_idx = tid(axis=0)
    block_dim = ntid(axis=0)
    src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.int8))
    dst_ptr = tl.cast(dst_ptr, tl.pointer_type(tl.int8))
    dst_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, dst_ptr)
    num_int4 = nbytes // 16
    for n in range(thread_idx, num_int4, block_dim):
        val0, val1 = load_v2_b64(src_ptr + 16 * n)
        multimem_st_b64(dst_mc_ptr + n * 16, val0)
        multimem_st_b64(dst_mc_ptr + n * 16 + 8, val1)


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_2d_ll_multimem_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_ll_buffer,
    nnodes: tl.constexpr,
    world_size: tl.constexpr,
    rank,
    signal_target,
):
    """
    pack_ll and nvshmem_putmem_nbi, then recv_ll and multimem.st
    """
    local_world_size = world_size // nnodes
    local_rank = rank % local_world_size
    nid = rank // local_world_size

    pid = tl.program_id(0)
    peer_nid = pid // local_world_size
    peer_local_rank = pid % local_world_size
    num_ints = bytes_per_rank // 4
    thread_idx = tid(axis=0)

    ll_buffer_int8 = tl.cast(symm_ll_buffer, tl.pointer_type(tl.int8))
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    if peer_local_rank == local_rank:
        if nid != peer_nid:
            segment = peer_nid * local_world_size + local_rank
            _recv_ll_and_multimem_st_ll_block(
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                num_ints,
                signal_target,
            )  # magic number here
            _recv_ll_block(
                symm_ptr + segment * bytes_per_rank,
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                num_ints,
                signal_target,
            )  # magic number here
        else:  # already has data. pack only
            _pack_ll_block(
                ll_buffer_int8 + rank * bytes_per_rank * 2,
                symm_ptr + rank * bytes_per_rank,
                num_ints,
                signal_target,
                2048,
            )  # magic number here
            __syncthreads()
            wid = thread_idx // 32
            # send
            if wid < nnodes and wid != nid:
                peer_to = wid * local_world_size + local_rank
                libshmem_device.putmem_nbi_warp(
                    ll_buffer_int8 + rank * bytes_per_rank * 2,
                    ll_buffer_int8 + rank * bytes_per_rank * 2,
                    bytes_per_rank * 2,
                    peer_to,
                )  # write and tell peer remote that remote copy is done

            segment = peer_nid * local_world_size + local_rank
            broadcast_naive_block(
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                bytes_per_rank * 2,
            )
    else:
        segment_recv_local = peer_nid * local_world_size + peer_local_rank
        _recv_ll_block(
            symm_ptr + segment_recv_local * bytes_per_rank,
            ll_buffer_int8 + segment_recv_local * bytes_per_rank * 2,
            num_ints,
            signal_target,
        )  # magic number here


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_2d_ll_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_flag,
    symm_ll_buffer,
    nnodes: tl.constexpr,
    world_size: tl.constexpr,
    rank,
    signal_target,
):
    local_world_size = world_size // nnodes
    local_rank = rank % local_world_size
    nid = rank // local_world_size

    pid = tl.program_id(0)
    peer_nid = pid // local_world_size
    peer_local_rank = pid % local_world_size
    thread_idx = tid(0)
    num_ints = bytes_per_rank // 4

    ll_buffer_int8 = tl.cast(symm_ll_buffer, tl.pointer_type(tl.int8))
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    if peer_local_rank == local_rank:  # remote push
        if peer_nid != nid:  # pnid: peer node id. each block recv from pnid
            segment = peer_nid * local_world_size + local_rank
            _recv_ll_block(
                symm_ptr + segment * bytes_per_rank,
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                num_ints,
                signal_target,
            )  # magic number here
            __syncthreads()
            if thread_idx == 0:
                st(symm_flag + segment, signal_target, scope="gpu", semantic="release")
        else:  # pack ll data
            _pack_ll_block(
                ll_buffer_int8 + rank * bytes_per_rank * 2,
                symm_ptr + rank * bytes_per_rank,
                num_ints,
                signal_target,
                2048,
            )  # magic number here
            __syncthreads()
            wid = thread_idx // 32
            if wid < nnodes and wid != nid:  # wid -> peer node id
                peer_to = wid * local_world_size + local_rank
                libshmem_device.putmem_nbi_warp(
                    ll_buffer_int8 + rank * bytes_per_rank * 2,
                    ll_buffer_int8 + rank * bytes_per_rank * 2,
                    bytes_per_rank * 2,
                    peer_to,
                )  # write and tell peer remote that remote copy is done
            # wait for all write done
            if thread_idx < world_size and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()

    else:  # local push
        peer = nid * local_world_size + peer_local_rank
        segment = peer_nid * local_world_size + local_rank
        if peer_nid != nid:  # wait for recv_ll done
            if thread_idx == 0:
                libshmem_device.signal_wait_until(symm_flag + segment, libshmem_device.NVSHMEM_CMP_EQ, signal_target)
            __syncthreads()
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )  # write and tell peer remote that remote copy is done


@dataclass
class FastAllGatherContext:
    rank: int
    node: int
    num_ranks: int
    num_nodes: int
    signal_tensor: torch.Tensor
    ll_buffers: List[torch.Tensor]  # double buffer
    grid_barrier: torch.Tensor
    max_buffer_size: int = 2 * 32 * 1024 * 1024
    signal_target: int = 1

    def finalize(self):
        nvshmem_free_tensor_sync(self.signal_tensor)
        for ll_buffer in self.ll_buffers:
            nvshmem_free_tensor_sync(ll_buffer)


def create_fast_allgather_context(rank, node, num_ranks, num_nodes, max_buffer_size: int = 2 * 32 * 1024 * 1024):
    signal_tensor = nvshmem_create_tensor((num_ranks, ), NVSHMEM_SIGNAL_DTYPE)
    signal_tensor.zero_()
    ll_buffers = [nvshmem_create_tensor((max_buffer_size, ), torch.int8) for _ in range(2)]
    grid_barrier = torch.zeros((1, ), dtype=torch.uint32, device="cuda")

    ctx = FastAllGatherContext(
        rank=rank,
        node=node,
        num_ranks=num_ranks,
        num_nodes=num_nodes,
        signal_tensor=signal_tensor,
        ll_buffers=ll_buffers,
        grid_barrier=grid_barrier,
        max_buffer_size=max_buffer_size,
        signal_target=15,
    )

    return ctx


def fast_allgather_pull(ctx: FastAllGatherContext, symm_buffer: torch.Tensor):
    ctx.signal_target += 1
    return _forward_pull_kernel[(ctx.num_ranks, )](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ctx.signal_tensor,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )


def fast_allgather_push_2d(ctx: FastAllGatherContext, symm_buffer: torch.Tensor):
    ctx.signal_target += 1
    _forward_push_2d_kernel[(ctx.num_ranks, )](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ctx.signal_tensor,
        ctx.num_nodes,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )
    return symm_buffer


def fast_allgather_push_3d(ctx: FastAllGatherContext, symm_buffer: torch.Tensor):
    ctx.signal_target += 1
    _forward_push_3d_kernel[(ctx.num_ranks, )](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ctx.ll_buffers[ctx.signal_target % 2],
        ctx.signal_tensor,
        ctx.num_nodes,
        2,  # TODO(houqi.1993)
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        INTER_NODE_WITH_LL=False,
        num_warps=32,
    )
    return symm_buffer


def fast_allgather_push_2d_ll(ctx: FastAllGatherContext, symm_buffer: torch.Tensor):
    assert symm_buffer.nbytes * 2 < ctx.max_buffer_size
    ctx.signal_target += 1
    ll_buffer = ctx.ll_buffers[ctx.signal_target % 2]
    _forward_push_2d_ll_kernel[(ctx.num_ranks, )](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ctx.signal_tensor,
        ll_buffer,
        ctx.num_nodes,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )

    return symm_buffer


def fast_allgather_push_2d_ll_multimem(ctx: FastAllGatherContext, symm_buffer: torch.Tensor):
    assert symm_buffer.nbytes * 2 < ctx.max_buffer_size
    ctx.signal_target += 1
    ll_buffer = ctx.ll_buffers[ctx.signal_target % 2]
    _forward_push_2d_ll_multimem_kernel[(ctx.num_ranks, )](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ll_buffer,
        ctx.num_nodes,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )

    return symm_buffer


def fast_allgather_push_numa_2d(ctx: FastAllGatherContext, symm_buffer: torch.Tensor):
    assert symm_buffer.nbytes * 2 < ctx.max_buffer_size
    signal = ctx.signal[ctx.signal_target % 2]
    _forward_push_numa_2d_kernel[(ctx.num_ranks, )](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        signal,
        2,  # TODO(houqi.1993) 2 NUMA nodes supported
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )
    ctx.signal_target += 1
    return symm_buffer


def fast_allgather_push_numa_2d_ll(ctx: FastAllGatherContext, symm_buffer: torch.Tensor):
    assert symm_buffer.nbytes * 2 < ctx.max_buffer_size
    assert ctx.num_nodes == 1
    signal = ctx.signal[ctx.signal_target % 2]
    _forward_push_numa_2d_ll_kernel[(ctx.num_ranks, )](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        signal,
        ctx.ll_buffers[ctx.signal_target % 2],
        2,  # TODO(houqi.1993) 2 NUMA nodes supported
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )
    ctx.signal_target += 1
    return symm_buffer


def fast_allgather_push_numa_2d_ll_multinode(ctx: FastAllGatherContext, symm_buffer: torch.Tensor):
    assert symm_buffer.nbytes * 2 < ctx.max_buffer_size
    signal = ctx.signal[ctx.signal_target % 2]
    _forward_push_numa_2d_ll_multinode_kernel[(ctx.num_ranks, )](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        signal,
        ctx.ll_buffers[ctx.signal_target % 2],
        ctx.num_nodes,
        2,  # TODO(houqi.1993) 2 NUMA nodes supported
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )
    ctx.signal_target += 1
    return symm_buffer


FAST_ALLGATHER_FUNC_DISPATCH = {
    "pull": fast_allgather_pull,
    "push2d": fast_allgather_push_2d,
    "push2d_ll": fast_allgather_push_2d_ll,
    "push2d_ll_multimem": fast_allgather_push_2d_ll_multimem,
    "push_numa_2d": fast_allgather_push_numa_2d,
    "push_numa_2d_ll": fast_allgather_push_numa_2d_ll,
    "push_numa_2d_ll_multinode": fast_allgather_push_numa_2d_ll_multinode,
}


def fast_allgather(
    symm_buffer: torch.Tensor,
    ctx: FastAllGatherContext = None,
    rank=None,
    node=None,
    num_ranks=None,
    num_nodes=None,
    mode="pull",
):
    assert mode in FAST_ALLGATHER_FUNC_DISPATCH
    if ctx is None:
        assert rank is not None and node is not None
        assert num_ranks is not None and num_nodes is not None
        ctx = create_fast_allgather_context(
            rank,
            node,
            num_ranks,
            num_nodes,
        )
    return FAST_ALLGATHER_FUNC_DISPATCH[mode](ctx, symm_buffer)
