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
from dataclasses import dataclass
import torch
from triton_dist.utils import nvshmem_create_tensor, nvshmem_free_tensor_sync, NVSHMEM_SIGNAL_DTYPE, nvshmem_barrier_all_on_stream

import triton
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra.cuda.language_extra import tid, __syncthreads, membar
from triton_dist.language.extra import libshmem_device


# assume that bs == 1
@triton_dist.jit(do_not_specialize=["bs", "seq"])
def kernel_pre_attn_qkv_pack_a2a(
    q,  # [bs, seq // P, q_nheads, k_head_dim]
    k,  # [bs, seq // P, kv_nheads, k_head_dim]
    v,  # [bs, seq // P, kv_nheads, v_head_dim]
    out_q,  # [bs, seq, local_q_nheads, k_head_dim]
    out_k,  # [bs, seq, local_kv_nheads, k_head_dim]
    out_v,  # [bs, seq, local_kv_nheads, v_head_dim]
    send_q_buf,  # [bs, q_nheads, seq // P, k_head_dim]
    send_k_buf,  # [bs, kv_nheads, seq // P, k_head_dim]
    send_v_buf,  # [bs, kv_nheads, seq // P, v_head_dim]
    recv_p2p_q_buf,  # [nnodes, bs, q_nheads_per_node, seq // P, k_head_dim]
    recv_p2p_k_buf,  # [nnodes, bs, max_kv_nheads_per_node, seq // P, k_head_dim]
    recv_p2p_v_buf,  # [nnodes, bs, max_kv_nheads_per_node, seq // P, v_head_dim]
    recv_out_q_buf,  # [bs, seq, local_q_nheads, k_head_dim]
    recv_out_k_buf,  # [bs, seq, local_kv_nheads, k_head_dim]
    recv_out_v_buf,  # [bs, seq, local_kv_nheads, v_head_dim]
    p2p_signal_buf,  # [nnodes * 2, ]
    intra_node_sync_signal_buf,  # [local_world_size, ]
    grid_sync_buf,  # [1, ]
    bs,
    local_seq,  # local_seq = seq // P
    max_kv_nheads_per_node,
    Q_NHEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    K_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    SKIP_Q_A2A: tl.constexpr,
):
    tl.static_assert(WORLD_SIZE % LOCAL_WORLD_SIZE == 0)
    tl.static_assert(Q_NHEADS % WORLD_SIZE == 0)
    tl.static_assert(Q_NHEADS % GROUP_SIZE == 0)

    rank = dl.rank()
    local_rank = rank % LOCAL_WORLD_SIZE
    NNODES: tl.constexpr = WORLD_SIZE // LOCAL_WORLD_SIZE
    node_id = rank // LOCAL_WORLD_SIZE
    LOCAL_Q_NHEADS: tl.constexpr = Q_NHEADS // WORLD_SIZE
    LOCAL_KV_NHEADS: tl.constexpr = max(LOCAL_Q_NHEADS // GROUP_SIZE, 1)
    tl.static_assert(LOCAL_Q_NHEADS % GROUP_SIZE == 0 or GROUP_SIZE % LOCAL_Q_NHEADS == 0)
    Q_NHEADS_PER_NODE = LOCAL_Q_NHEADS * LOCAL_WORLD_SIZE

    num_tiles_seq = tl.cdiv(local_seq, BLOCK_SEQ)
    num_pid = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)

    q_elem_size: tl.constexpr = tl.constexpr(send_q_buf.dtype.element_ty.primitive_bitwidth) // 8
    k_elem_size: tl.constexpr = tl.constexpr(send_k_buf.dtype.element_ty.primitive_bitwidth) // 8
    v_elem_size: tl.constexpr = tl.constexpr(send_v_buf.dtype.element_ty.primitive_bitwidth) // 8
    tl.static_assert(k_elem_size == v_elem_size)

    thread_idx = tid(0)
    ALIGNMENT: tl.constexpr = 16
    # todo: copy q/k/v to send_q/k/v comm buf
    for node_offset in range(0, NNODES):
        target_node_id = (node_id + node_offset + 1) % NNODES
        target_rank = target_node_id * LOCAL_WORLD_SIZE + local_rank
        q_nheads_start = target_node_id * LOCAL_WORLD_SIZE * LOCAL_Q_NHEADS
        q_nheads_end = q_nheads_start + LOCAL_WORLD_SIZE * LOCAL_Q_NHEADS
        kv_nheads_start = q_nheads_start // GROUP_SIZE
        kv_nheads_end = tl.cdiv(q_nheads_end, GROUP_SIZE)
        # step 0: inter-node p2p(between gpu with same id)
        if target_node_id != node_id:
            if pid == 0:
                # skip first round
                if node_offset > 0:
                    if thread_idx == 0:
                        libshmem_device.signal_wait_until(p2p_signal_buf + NNODES + target_node_id,
                                                          libshmem_device.NVSHMEM_CMP_EQ, 1)
                    __syncthreads()

                q_nbytes = (q_nheads_end - q_nheads_start) * local_seq * K_HEAD_DIM * q_elem_size
                k_nbytes = (kv_nheads_end - kv_nheads_start) * local_seq * K_HEAD_DIM * k_elem_size
                v_nbytes = (kv_nheads_end - kv_nheads_start) * local_seq * V_HEAD_DIM * v_elem_size

                # for q: [q_nheads_start:q_nheads_end, :, :]
                recv_p2p_q_buf_base = recv_p2p_q_buf + node_id * bs * Q_NHEADS_PER_NODE * local_seq * K_HEAD_DIM
                if not SKIP_Q_A2A:
                    libshmem_device.putmem_nbi_block(
                        recv_p2p_q_buf_base,
                        send_q_buf + q_nheads_start * local_seq * K_HEAD_DIM,
                        q_nbytes,
                        target_rank,
                    )

                # for k: [kv_nheads_start:kv_nheads_end, :, :]
                recv_p2p_k_buf_base = recv_p2p_k_buf + node_id * bs * max_kv_nheads_per_node * local_seq * K_HEAD_DIM
                libshmem_device.putmem_nbi_block(
                    recv_p2p_k_buf_base,
                    send_k_buf + kv_nheads_start * local_seq * K_HEAD_DIM,
                    k_nbytes,
                    target_rank,
                )

                recv_p2p_v_buf_base = recv_p2p_v_buf + node_id * bs * max_kv_nheads_per_node * local_seq * V_HEAD_DIM
                libshmem_device.putmem_nbi_block(
                    recv_p2p_v_buf_base,
                    send_v_buf + kv_nheads_start * local_seq * V_HEAD_DIM,
                    v_nbytes,
                    target_rank,
                )

                libshmem_device.fence()
                if thread_idx == 0:
                    libshmem_device.signal_op(
                        p2p_signal_buf + node_id,
                        1,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        target_rank,
                    )
                __syncthreads()

        # intra-node dispatch start from current node
        src_node_id = (node_id - node_offset + NNODES) % NNODES
        q_nheads_start_src_node = node_id * LOCAL_WORLD_SIZE * LOCAL_Q_NHEADS
        q_nheads_end_src_node = q_nheads_start_src_node + LOCAL_WORLD_SIZE * LOCAL_Q_NHEADS
        kv_nheads_start_src_node = q_nheads_start_src_node // GROUP_SIZE
        kv_nheads_end_src_node = tl.cdiv(q_nheads_end_src_node, GROUP_SIZE)
        if src_node_id != node_id:
            recv_p2p_q_buf_src_node_base = recv_p2p_q_buf + src_node_id * bs * Q_NHEADS_PER_NODE * local_seq * K_HEAD_DIM
            recv_p2p_k_buf_src_node_base = recv_p2p_k_buf + src_node_id * bs * max_kv_nheads_per_node * local_seq * K_HEAD_DIM
            recv_p2p_v_buf_src_node_base = recv_p2p_v_buf + src_node_id * bs * max_kv_nheads_per_node * local_seq * V_HEAD_DIM
            if thread_idx == 0:
                libshmem_device.signal_wait_until(p2p_signal_buf + src_node_id, libshmem_device.NVSHMEM_CMP_EQ, 1)
            __syncthreads()
        else:
            # add offset in nheads dim
            recv_p2p_q_buf_src_node_base = send_q_buf + q_nheads_start_src_node * local_seq * K_HEAD_DIM
            recv_p2p_k_buf_src_node_base = send_k_buf + kv_nheads_start_src_node * local_seq * K_HEAD_DIM
            recv_p2p_v_buf_src_node_base = send_v_buf + kv_nheads_start_src_node * local_seq * K_HEAD_DIM

        membar(scope="gl")

        # for nxt round
        if pid == 0:
            if node_offset + 2 < NNODES:
                nxt_step_src_node = (node_id - node_offset - 2 + NNODES) % NNODES
                if thread_idx == 0:
                    libshmem_device.signal_op(
                        p2p_signal_buf + NNODES + node_id,
                        1,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        local_rank + nxt_step_src_node * LOCAL_WORLD_SIZE,
                    )
                __syncthreads()

        tl.static_assert(K_HEAD_DIM == triton.next_power_of_2(K_HEAD_DIM))
        tl.static_assert(V_HEAD_DIM == triton.next_power_of_2(V_HEAD_DIM))

        #step 1: intra node dispatch, recv_p2p_q/k/v_buf_src_node -> recv_out_q/k/v_buf
        num_q_heads_src_node = q_nheads_end_src_node - q_nheads_start_src_node
        num_kv_heads_src_node = kv_nheads_end_src_node - kv_nheads_start_src_node
        num_q_tiles = num_q_heads_src_node * num_tiles_seq
        num_kv_tiles = num_kv_heads_src_node * num_tiles_seq

        # [q_nheads_start_src_node:q_nheads_end_src_node, local_seq, K_HEAD_DIM] -> [src_rank * local_seq:(src_rank + 1) * local_seq, local_q_nheads, K_HEAD_DIM]
        offs_k_hd = tl.arange(0, K_HEAD_DIM)
        offs_v_hd = tl.arange(0, V_HEAD_DIM)
        src_rank = src_node_id * LOCAL_WORLD_SIZE + local_rank
        # assume that bs = 1, skip offset in batch dim
        if not SKIP_Q_A2A:
            for tile_id in range(pid, num_q_tiles, num_pid):
                q_nheads_offset = tile_id % num_q_heads_src_node
                seq_tile_id = tile_id // num_q_heads_src_node
                q_nheads_idx_src_node = q_nheads_offset + q_nheads_start_src_node
                target_rank = q_nheads_idx_src_node // LOCAL_Q_NHEADS
                q_nheads_idx_target = q_nheads_idx_src_node % LOCAL_Q_NHEADS
                offs_seq = tl.arange(0, BLOCK_SEQ) + seq_tile_id * BLOCK_SEQ
                mask_seq = offs_seq < local_seq
                q_tile_data = tl.load(
                    recv_p2p_q_buf_src_node_base + q_nheads_offset * local_seq * K_HEAD_DIM +
                    offs_seq[:, None] * K_HEAD_DIM + offs_k_hd[None, :], mask=mask_seq[:, None])
                target_recv_out_q_buf = dl.symm_at(recv_out_q_buf, target_rank)
                target_recv_out_q_buf = tl.multiple_of(target_recv_out_q_buf, ALIGNMENT)
                tl.store(
                    target_recv_out_q_buf + (src_rank * local_seq + offs_seq[:, None]) * LOCAL_Q_NHEADS * K_HEAD_DIM +
                    q_nheads_idx_target * K_HEAD_DIM + offs_k_hd[None, :], q_tile_data, mask=mask_seq[:, None])

        # k && v
        for tile_id in range(pid, num_kv_tiles, num_pid):
            kv_nheads_offset = tile_id % num_kv_heads_src_node
            seq_tile_id = tile_id // num_kv_heads_src_node
            kv_nheads_idx_src_node = kv_nheads_offset + kv_nheads_start_src_node
            offs_seq = tl.arange(0, BLOCK_SEQ) + seq_tile_id * BLOCK_SEQ
            mask_seq = offs_seq < local_seq
            k_tile_data = tl.load(
                recv_p2p_k_buf_src_node_base + kv_nheads_offset * local_seq * K_HEAD_DIM +
                offs_seq[:, None] * K_HEAD_DIM + offs_k_hd[None, :], mask=mask_seq[:, None])
            v_tile_data = tl.load(
                recv_p2p_v_buf_src_node_base + kv_nheads_offset * local_seq * V_HEAD_DIM +
                offs_seq[:, None] * V_HEAD_DIM + offs_v_hd[None, :], mask=mask_seq[:, None])

            if GROUP_SIZE > LOCAL_Q_NHEADS:
                # tl.static_assert(LOCAL_KV_NHEADS == 1)
                tl.static_assert(GROUP_SIZE % LOCAL_Q_NHEADS == 0)
                NUM_BROADCAST_RANKS = GROUP_SIZE // LOCAL_Q_NHEADS
                target_rank_start = kv_nheads_idx_src_node * NUM_BROADCAST_RANKS
                target_rank_end = (kv_nheads_idx_src_node + 1) * NUM_BROADCAST_RANKS
                if target_rank_start < node_id * LOCAL_WORLD_SIZE:
                    target_rank_start = node_id * LOCAL_WORLD_SIZE
                if target_rank_end > (node_id + 1) * LOCAL_WORLD_SIZE:
                    target_rank_end = (node_id + 1) * LOCAL_WORLD_SIZE
                target_range = target_rank_end - target_rank_start
                rank_offset = local_rank % target_range
                for idx in range(target_range):
                    target_rank = target_rank_start + (idx + rank_offset) % target_range
                    target_recv_out_k_buf = dl.symm_at(recv_out_k_buf, target_rank)
                    target_recv_out_v_buf = dl.symm_at(recv_out_v_buf, target_rank)
                    target_recv_out_k_buf = tl.multiple_of(target_recv_out_k_buf, ALIGNMENT)
                    target_recv_out_v_buf = tl.multiple_of(target_recv_out_v_buf, ALIGNMENT)

                    # only one head, offset in nhead dim always is zero
                    tl.store(
                        target_recv_out_k_buf +
                        (src_rank * local_seq + offs_seq[:, None]) * LOCAL_KV_NHEADS * K_HEAD_DIM + offs_k_hd[None, :],
                        k_tile_data, mask=mask_seq[:, None])
                    tl.store(
                        target_recv_out_v_buf +
                        (src_rank * local_seq + offs_seq[:, None]) * LOCAL_KV_NHEADS * V_HEAD_DIM + offs_v_hd[None, :],
                        v_tile_data, mask=mask_seq[:, None])

            else:
                tl.static_assert(LOCAL_Q_NHEADS % GROUP_SIZE == 0)
                target_rank = kv_nheads_idx_src_node // LOCAL_KV_NHEADS
                kv_nheads_idx_target = kv_nheads_offset % LOCAL_KV_NHEADS
                target_recv_out_k_buf = dl.symm_at(recv_out_k_buf, target_rank)
                target_recv_out_v_buf = dl.symm_at(recv_out_v_buf, target_rank)
                target_recv_out_k_buf = tl.multiple_of(target_recv_out_k_buf, ALIGNMENT)
                target_recv_out_v_buf = tl.multiple_of(target_recv_out_v_buf, ALIGNMENT)
                tl.store(
                    target_recv_out_k_buf + (src_rank * local_seq + offs_seq[:, None]) * LOCAL_KV_NHEADS * K_HEAD_DIM +
                    kv_nheads_idx_target * K_HEAD_DIM + offs_k_hd[None, :], k_tile_data, mask=mask_seq[:, None])
                tl.store(
                    target_recv_out_v_buf + (src_rank * local_seq + offs_seq[:, None]) * LOCAL_KV_NHEADS * V_HEAD_DIM +
                    kv_nheads_idx_target * V_HEAD_DIM + offs_v_hd[None, :], v_tile_data, mask=mask_seq[:, None])


# src/dst has added offset in batch and nheads dimension
@triton_dist.jit
def _kernel_inner_tile_copy(
    src_base_ptr,
    dst_base_ptr,
    seq,
    tile_id_seq,
    stride_src_seq,
    stride_src_hd,
    stride_dst_seq,
    stride_dst_hd,
    BLOCK_SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    tl.static_assert(HEAD_DIM == triton.next_power_of_2(HEAD_DIM))
    offs_seq = tl.arange(0, BLOCK_SEQ) + tile_id_seq * BLOCK_SEQ
    offs_hd = tl.arange(0, HEAD_DIM)
    mask_seq = offs_seq < seq

    src_ptrs = src_base_ptr + offs_seq[:, None] * stride_src_seq + offs_hd[None, :] * stride_src_hd
    dst_ptrs = dst_base_ptr + offs_seq[:, None] * stride_dst_seq + offs_hd[None, :] * stride_dst_hd

    data = tl.load(src_ptrs, mask=mask_seq[:, None])
    tl.store(dst_ptrs, data, mask=mask_seq[:, None])


# [batch, seq, nheads, head_dim] -> [batch, nheads, seq, head_dim]
@triton_dist.jit(do_not_specialize=["bs", "seq"])
def kernel_qkv_bsnd_to_bnsd(
    q,  # [bs, seq, q_nheads, k_head_dim]
    k,  # [bs, seq, kv_nheads, k_head_dim]
    v,  # [bs, seq, kv_nheads, v_head_dim]
    send_q_buf,  # [bs, q_nheads, seq, k_head_dim]
    send_k_buf,  # [bs, kv_nheads, seq, k_head_dim]
    send_v_buf,  # [bs, kv_nheads, seq, v_head_dim]
    stride_q_hd,
    stride_q_nh,
    stride_q_seq,
    stride_q_bs,
    stride_k_hd,
    stride_k_nh,
    stride_k_seq,
    stride_k_bs,
    stride_v_hd,
    stride_v_nh,
    stride_v_seq,
    stride_v_bs,
    bs,
    seq,
    Q_NHEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    K_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    SKIP_Q_A2A: tl.constexpr,
):
    tl.static_assert(Q_NHEADS % GROUP_SIZE == 0)

    KV_NHEADS: tl.constexpr = Q_NHEADS // GROUP_SIZE

    num_tiles_seq = tl.cdiv(seq, BLOCK_SEQ)
    num_pid = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)

    k_elem_size: tl.constexpr = tl.constexpr(send_k_buf.dtype.element_ty.primitive_bitwidth) // 8
    v_elem_size: tl.constexpr = tl.constexpr(send_v_buf.dtype.element_ty.primitive_bitwidth) // 8
    tl.static_assert(k_elem_size == v_elem_size)

    num_tiles = bs * (Q_NHEADS + KV_NHEADS * 2) * num_tiles_seq
    stride_send_q_nh = seq * K_HEAD_DIM
    stride_send_q_bs = stride_send_q_nh * Q_NHEADS

    stride_send_k_nh = seq * K_HEAD_DIM
    stride_send_k_bs = stride_send_k_nh * KV_NHEADS

    stride_send_v_nh = seq * V_HEAD_DIM
    stride_send_v_bs = stride_send_v_nh * KV_NHEADS

    # each tile response to copy [BLOCK_SEQ, HEAD_DIM]
    for tile_id in range(pid, num_tiles, num_pid):
        tile_id_seq = tile_id % num_tiles_seq
        nh_idx = (tile_id // num_tiles_seq) % (Q_NHEADS + KV_NHEADS * 2)
        bs_idx = tile_id // (num_tiles_seq * (Q_NHEADS + KV_NHEADS * 2))
        if nh_idx < Q_NHEADS:
            src_base_ptr = q + bs_idx * stride_q_bs + nh_idx * stride_q_nh
            dst_base_ptr = send_q_buf + bs_idx * stride_send_q_bs + nh_idx * stride_send_q_nh
            if not SKIP_Q_A2A:
                _kernel_inner_tile_copy(
                    src_base_ptr,
                    dst_base_ptr,
                    seq,
                    tile_id_seq,
                    stride_q_seq,
                    stride_q_hd,
                    stride_dst_seq=K_HEAD_DIM,
                    stride_dst_hd=1,
                    BLOCK_SEQ=BLOCK_SEQ,
                    HEAD_DIM=K_HEAD_DIM,
                )
        elif nh_idx < Q_NHEADS + KV_NHEADS:
            k_nh_idx = nh_idx - Q_NHEADS
            src_base_ptr = k + bs_idx * stride_k_bs + k_nh_idx * stride_k_nh
            dst_base_ptr = send_k_buf + bs_idx * stride_send_k_bs + k_nh_idx * stride_send_k_nh
            _kernel_inner_tile_copy(
                src_base_ptr,
                dst_base_ptr,
                seq,
                tile_id_seq,
                stride_k_seq,
                stride_k_hd,
                stride_dst_seq=K_HEAD_DIM,
                stride_dst_hd=1,
                BLOCK_SEQ=BLOCK_SEQ,
                HEAD_DIM=K_HEAD_DIM,
            )
        else:
            v_nh_idx = nh_idx - Q_NHEADS - KV_NHEADS
            src_base_ptr = v + bs_idx * stride_v_bs + v_nh_idx * stride_v_nh
            dst_base_ptr = send_v_buf + bs_idx * stride_send_v_bs + v_nh_idx * stride_send_v_nh
            _kernel_inner_tile_copy(
                src_base_ptr,
                dst_base_ptr,
                seq,
                tile_id_seq,
                stride_v_seq,
                stride_v_hd,
                stride_dst_seq=V_HEAD_DIM,
                stride_dst_hd=1,
                BLOCK_SEQ=BLOCK_SEQ,
                HEAD_DIM=V_HEAD_DIM,
            )


def qkv_bsnd_to_bnsd(q, k, v, send_q_buf, send_k_buf, send_v_buf, BLOCK_SEQ=256, skip_q_a2a=False, num_sms=0):

    bs, seq, q_nheads, q_head_dim = q.shape
    bs, seq, kv_nheads, k_head_dim = k.shape
    bs, seq, kv_nheads, v_head_dim = v.shape

    def _check_buf(src, dst):
        assert src.dtype == dst.dtype
        assert dst.is_contiguous()
        assert dst.numel() >= src.numel()

    assert q_head_dim == k_head_dim
    assert q_nheads % kv_nheads == 0
    _check_buf(q, send_q_buf)
    _check_buf(k, send_k_buf)
    _check_buf(v, send_v_buf)

    group_size = q_nheads // kv_nheads

    if num_sms > 0:
        grid = (num_sms, )
        num_warps = 32
    else:
        grid = (bs * (q_nheads + kv_nheads * 2) * triton.cdiv(seq, BLOCK_SEQ), )
        num_warps = 8

    kernel_qkv_bsnd_to_bnsd[grid](
        q,  # [bs, seq, q_nheads, k_head_dim]
        k,  # [bs, seq, kv_nheads, k_head_dim]
        v,  # [bs, seq, kv_nheads, v_head_dim]
        send_q_buf,  # [bs, q_nheads, seq, k_head_dim]
        send_k_buf,  # [bs, kv_nheads, seq, k_head_dim]
        send_v_buf,  # [bs, kv_nheads, seq, v_head_dim]
        stride_q_hd=q.stride(3),
        stride_q_nh=q.stride(2),
        stride_q_seq=q.stride(1),
        stride_q_bs=q.stride(0),
        stride_k_hd=k.stride(3),
        stride_k_nh=k.stride(2),
        stride_k_seq=k.stride(1),
        stride_k_bs=k.stride(0),
        stride_v_hd=v.stride(3),
        stride_v_nh=v.stride(2),
        stride_v_seq=v.stride(1),
        stride_v_bs=v.stride(0),
        bs=bs,
        seq=seq,
        Q_NHEADS=q_nheads,
        GROUP_SIZE=group_size,
        K_HEAD_DIM=k_head_dim,
        V_HEAD_DIM=v_head_dim,
        BLOCK_SEQ=BLOCK_SEQ,
        SKIP_Q_A2A=skip_q_a2a,
        num_warps=num_warps,
    )
    return send_q_buf, send_k_buf, send_v_buf


@dataclass
class UlyssesSPPreAttnCommContext:
    send_q_buf: torch.Tensor  # [bs, q_nheads, seq // P, k_head_dim]
    send_k_buf: torch.Tensor  # [bs, kv_nheads, seq // P, k_head_dim]
    send_v_buf: torch.Tensor  # [bs, kv_nheads, seq // P, v_head_dim]

    recv_p2p_q_buf: torch.Tensor  # [nnodes, bs, q_nheads_per_node, seq // P, k_head_dim]
    recv_p2p_k_buf: torch.Tensor  # [nnodes, bs, max_kv_nheads_per_node, seq // P, k_head_dim]
    recv_p2p_v_buf: torch.Tensor  # [nnodes, bs, max_kv_nheads_per_node, seq // P, v_head_dim]

    recv_out_q_buf: torch.Tensor  # [bs, seq, local_q_nheads, k_head_dim]
    recv_out_k_buf: torch.Tensor  # [bs, seq, local_kv_nheads, k_head_dim]
    recv_out_v_buf: torch.Tensor  # [bs, seq, local_kv_nheads, v_head_dim]

    p2p_signal_buf: torch.Tensor  # [2 * nnodes, ], zero init
    intra_node_sync_signal_buf: torch.Tensor  # [local_world_size] zero init
    grid_sync_buf: torch.Tensor  # [1, ]

    bs: int
    max_seq: int
    group_size: int
    q_nheads: int
    kv_nheads: int
    k_head_dim: int
    v_head_dim: int

    max_kv_nheads_per_node: int

    rank: int
    local_world_size: int
    world_size: int

    num_sms: int

    def finalize(self):
        nvshmem_free_tensor_sync(self.send_q_buf)
        nvshmem_free_tensor_sync(self.send_k_buf)
        nvshmem_free_tensor_sync(self.send_v_buf)

        nvshmem_free_tensor_sync(self.recv_p2p_q_buf)
        nvshmem_free_tensor_sync(self.recv_p2p_k_buf)
        nvshmem_free_tensor_sync(self.recv_p2p_v_buf)

        nvshmem_free_tensor_sync(self.recv_out_q_buf)
        nvshmem_free_tensor_sync(self.recv_out_k_buf)
        nvshmem_free_tensor_sync(self.recv_out_v_buf)

        nvshmem_free_tensor_sync(self.p2p_signal_buf)
        nvshmem_free_tensor_sync(self.intra_node_sync_signal_buf)

    def local_q_nheads(self, q_nheads):
        assert q_nheads % self.world_size == 0
        local_q_nheads = q_nheads // self.world_size
        return local_q_nheads

    def local_kv_nheads(self, q_nheads):
        local_kv_nheads = max(self.local_q_nheads(q_nheads) // self.group_size, 1)
        return local_kv_nheads

    def check_context(self, bs, seq, q_nheads, kv_nheads, k_head_dim, v_head_dim):
        assert bs <= self.bs
        assert seq <= self.max_seq
        assert q_nheads <= self.q_nheads
        assert kv_nheads <= self.kv_nheads
        assert k_head_dim <= self.k_head_dim
        assert v_head_dim <= self.v_head_dim
        assert seq % self.world_size == 0
        assert q_nheads % self.world_size == 0
        local_q_nheads = q_nheads // self.world_size
        assert q_nheads % kv_nheads == 0 and q_nheads // kv_nheads == self.group_size
        assert local_q_nheads % self.group_size == 0 or self.group_size % local_q_nheads == 0

    def reset_all_signal(self):
        # no need to reset intra_node_sync_signal_buf
        self.p2p_signal_buf.zero_()


def create_ulysses_sp_pre_attn_comm_context(bs: int, max_seq: int, q_nheads: int, k_head_dim: int, v_head_dim: int,
                                            group_size: int, dtype, local_world_size,
                                            pg: torch.distributed.ProcessGroup,
                                            num_sms=0) -> UlyssesSPPreAttnCommContext:
    assert bs == 1

    world_size = pg.size()
    rank = pg.rank()

    assert local_world_size > 0 and world_size % local_world_size == 0
    assert q_nheads > 0 and q_nheads % group_size == 0
    kv_nheads = q_nheads // group_size
    assert q_nheads % world_size == 0
    local_q_nheads = q_nheads // world_size
    local_kv_nheads = max(local_q_nheads // group_size, 1)
    assert local_q_nheads % group_size == 0 or group_size % local_q_nheads == 0

    nnodes = world_size // local_world_size
    assert max_seq % world_size == 0
    max_local_seq = (max_seq + world_size - 1) // world_size

    send_q_buf = nvshmem_create_tensor([bs, q_nheads, max_local_seq, k_head_dim], dtype)
    send_k_buf = nvshmem_create_tensor([bs, kv_nheads, max_local_seq, k_head_dim], dtype)
    send_v_buf = nvshmem_create_tensor([bs, kv_nheads, max_local_seq, v_head_dim], dtype)

    q_nheads_per_node = local_q_nheads * local_world_size
    max_kv_nheads_per_node = (q_nheads_per_node + group_size - 1) // group_size + 1
    recv_p2p_q_buf = nvshmem_create_tensor([nnodes, bs, q_nheads_per_node, max_local_seq, k_head_dim], dtype)
    recv_p2p_k_buf = nvshmem_create_tensor([nnodes, bs, max_kv_nheads_per_node, max_local_seq, k_head_dim], dtype)
    recv_p2p_v_buf = nvshmem_create_tensor([nnodes, bs, max_kv_nheads_per_node, max_local_seq, v_head_dim], dtype)

    recv_out_q_buf = nvshmem_create_tensor([bs, max_seq, local_q_nheads, k_head_dim], dtype)
    recv_out_k_buf = nvshmem_create_tensor([bs, max_seq, local_kv_nheads, k_head_dim], dtype)
    recv_out_v_buf = nvshmem_create_tensor([bs, max_seq, local_kv_nheads, v_head_dim], dtype)
    """
        p2p-signal-buf[:nnodes] is used to notify the receiver,
        p2p-signal-buf[nnodes:] is used to notify the next round sender.
    """
    p2p_signal_buf = nvshmem_create_tensor([
        nnodes * 2,
    ], NVSHMEM_SIGNAL_DTYPE)
    p2p_signal_buf.zero_()
    intra_node_sync_signal_buf = nvshmem_create_tensor([
        local_world_size,
    ], NVSHMEM_SIGNAL_DTYPE)
    intra_node_sync_signal_buf.zero_()
    grid_sync_buf = torch.zeros((1, ), dtype=torch.int32, device=torch.cuda.current_device())

    nvshmem_barrier_all_on_stream()

    return UlyssesSPPreAttnCommContext(
        send_q_buf=send_q_buf, send_k_buf=send_k_buf, send_v_buf=send_v_buf, recv_p2p_q_buf=recv_p2p_q_buf,
        recv_p2p_k_buf=recv_p2p_k_buf, recv_p2p_v_buf=recv_p2p_v_buf, recv_out_q_buf=recv_out_q_buf,
        recv_out_k_buf=recv_out_k_buf, recv_out_v_buf=recv_out_v_buf, p2p_signal_buf=p2p_signal_buf,
        intra_node_sync_signal_buf=intra_node_sync_signal_buf, grid_sync_buf=grid_sync_buf, bs=bs, max_seq=max_seq,
        group_size=group_size, q_nheads=q_nheads, kv_nheads=kv_nheads, k_head_dim=k_head_dim, v_head_dim=v_head_dim,
        max_kv_nheads_per_node=max_kv_nheads_per_node, rank=rank, local_world_size=local_world_size,
        world_size=world_size, num_sms=num_sms)


def pre_attn_qkv_pack_a2a_op(ctx: UlyssesSPPreAttnCommContext, q, k, v, skip_q_a2a=False, return_comm_buf=False):
    bs, q_local_seq, q_nheads, k_head_dim = q.shape
    _, k_local_seq, kv_nheads, k_head_dim = k.shape
    _, v_local_seq, kv_nheads, v_head_dim = v.shape
    assert q_local_seq == k_local_seq and k_local_seq == v_local_seq
    local_seq = q_local_seq
    seq = q_local_seq * ctx.world_size
    ctx.check_context(bs, seq, q_nheads, kv_nheads, k_head_dim, v_head_dim)
    num_sms = ctx.num_sms
    if num_sms <= 0:
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (num_sms, )
    local_q_nheads = ctx.local_q_nheads(q_nheads)
    local_kv_nheads = ctx.local_kv_nheads(q_nheads)

    out_q = torch.empty((bs, seq, local_q_nheads, k_head_dim), dtype=q.dtype, device=q.device)
    out_k = torch.empty((bs, seq, local_kv_nheads, k_head_dim), dtype=k.dtype, device=k.device)
    out_v = torch.empty((bs, seq, local_kv_nheads, v_head_dim), dtype=v.dtype, device=v.device)

    BLOCK_SEQ = 128

    def _permute_and_copy(src, dst):
        permuted_src = src.permute(0, 2, 1, 3)
        src_numel = src.numel()
        dst_numel = dst.numel()
        shape_permuted_src = permuted_src.shape
        if src_numel > dst_numel:
            raise ValueError("comm buf mismatch")
        dst_buf = dst.reshape(-1)[:src_numel]
        dst_buf.copy_(permuted_src.reshape(-1))
        return dst_buf.view(shape_permuted_src)

    def _inplace_copy_from_comm_buf_to_out(src, dst):
        src_numel = src.numel()
        dst_numel = dst.numel()
        if src_numel < dst_numel:
            raise ValueError("comm buf mismatch")

        src_buf = src.reshape(-1)[:dst_numel]
        dst.reshape(-1).copy_(src_buf)
        return dst

    qkv_bsnd_to_bnsd(q, k, v, ctx.send_q_buf, ctx.send_k_buf, ctx.send_v_buf, skip_q_a2a=skip_q_a2a,
                     num_sms=ctx.num_sms)

    current_stream = torch.cuda.current_stream()
    nvshmem_barrier_all_on_stream(current_stream)

    kernel_pre_attn_qkv_pack_a2a[grid](
        q,  # [bs, seq // P, q_nheads, k_head_dim]
        k,  # [bs, seq // P, kv_nheads, k_head_dim]
        v,  # [bs, seq // P, kv_nheads, v_head_dim]
        out_q,  # [bs, seq, local_q_nheads, k_head_dim]
        out_k,  # [bs, seq, local_kv_nheads, k_head_dim]
        out_v,  # [bs, seq, local_kv_nheads, v_head_dim]
        ctx.send_q_buf,  # [bs, q_nheads, seq // P, k_head_dim]
        ctx.send_k_buf,  # [bs, kv_nheads, seq // P, k_head_dim]
        ctx.send_v_buf,  # [bs, kv_nheads, seq // P, v_head_dim]
        ctx.recv_p2p_q_buf,  # [nnodes, bs, q_nheads_per_node, seq // P, k_head_dim]
        ctx.recv_p2p_k_buf,  # [nnodes, bs, max_kv_nheads_per_node, seq // P, k_head_dim]
        ctx.recv_p2p_v_buf,  # [nnodes, bs, max_kv_nheads_per_node, seq // P, v_head_dim]
        ctx.recv_out_q_buf,  # [bs, seq, local_q_nheads, k_head_dim]
        ctx.recv_out_k_buf,  # [bs, seq, local_kv_nheads, k_head_dim]
        ctx.recv_out_v_buf,  # [bs, seq, local_kv_nheads, v_head_dim]
        ctx.p2p_signal_buf,  # [nnodes * 2, ]
        ctx.intra_node_sync_signal_buf,
        ctx.grid_sync_buf,
        bs,
        local_seq,  # local_seq = seq // P
        ctx.max_kv_nheads_per_node,
        Q_NHEADS=q_nheads,
        GROUP_SIZE=ctx.group_size,
        K_HEAD_DIM=k_head_dim,
        V_HEAD_DIM=v_head_dim,
        BLOCK_SEQ=BLOCK_SEQ,
        LOCAL_WORLD_SIZE=ctx.local_world_size,
        WORLD_SIZE=ctx.world_size,
        SKIP_Q_A2A=skip_q_a2a,
        num_warps=32,
    )

    ctx.reset_all_signal()
    nvshmem_barrier_all_on_stream(current_stream)

    if return_comm_buf:

        def _slice_and_reshape(src, dst):
            n_dst = dst.numel()
            assert src.numel() >= n_dst
            return src.reshape(-1)[:n_dst].reshape(dst.shape)

        return _slice_and_reshape(ctx.recv_out_q_buf, out_q) if not skip_q_a2a else None, _slice_and_reshape(
            ctx.recv_out_k_buf, out_k), _slice_and_reshape(ctx.recv_out_v_buf, out_v),

    if not return_comm_buf:
        _inplace_copy_from_comm_buf_to_out(ctx.recv_out_q_buf, out_q)
        _inplace_copy_from_comm_buf_to_out(ctx.recv_out_k_buf, out_k)
        _inplace_copy_from_comm_buf_to_out(ctx.recv_out_v_buf, out_v)

    if skip_q_a2a:
        out_q = None
    return out_q, out_k, out_v
