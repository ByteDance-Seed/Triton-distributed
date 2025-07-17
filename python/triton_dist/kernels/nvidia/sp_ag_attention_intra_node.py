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
import triton.language as tl
import triton_dist.language as dl

from typing import List
import math

from dataclasses import dataclass
from cuda import cudart

from triton_dist.utils import CUDA_CHECK, nvshmem_create_tensors, nvshmem_free_tensor_sync
from triton_dist.kernels.nvidia.common_ops import barrier_all_on_stream, BarrierAllContext

##################################################


@dataclass
class SPAllGatherAttentionContextIntraNode:
    ag_k_buffers: List[torch.Tensor]
    ag_k_buffer: torch.Tensor
    ag_k_buffers_ptr: torch.Tensor
    ag_v_buffers: List[torch.Tensor]
    ag_v_buffer: torch.Tensor
    ag_v_buffers_ptr: torch.Tensor
    attn_output_buffer: torch.Tensor
    ag_stream: torch.cuda.Stream
    barrier: BarrierAllContext

    def finalize(self):
        nvshmem_free_tensor_sync(self.ag_k_buffer)
        nvshmem_free_tensor_sync(self.ag_v_buffer)


def create_sp_ag_attention_context_intra_node(
    batch_size,
    q_head,
    kv_head,
    max_seqlen_k,
    max_q_shard_len,
    head_dim,
    input_dtype,
    output_dtype,
    rank,
    world_size,
    device,
):
    ag_k_buffers = nvshmem_create_tensors((batch_size * max_seqlen_k, kv_head, head_dim), input_dtype, rank, world_size)
    ag_k_buffer = ag_k_buffers[rank]
    ag_k_buffers_ptr: torch.Tensor = torch.tensor([t.data_ptr() for t in ag_k_buffers], device=device)

    ag_v_buffers = nvshmem_create_tensors((batch_size * max_seqlen_k, kv_head, head_dim), input_dtype, rank, world_size)
    ag_v_buffer = ag_v_buffers[rank]
    ag_v_buffers_ptr: torch.Tensor = torch.tensor([t.data_ptr() for t in ag_v_buffers], device=device)

    attn_output_buffer = torch.empty(
        batch_size * max_q_shard_len,
        q_head,
        head_dim,
        dtype=output_dtype,
        device=device,
    )

    # stream for copy
    ag_stream = torch.cuda.Stream()

    barrier = BarrierAllContext(True)

    ctx = SPAllGatherAttentionContextIntraNode(ag_k_buffers=ag_k_buffers, ag_k_buffer=ag_k_buffer,
                                               ag_k_buffers_ptr=ag_k_buffers_ptr, ag_v_buffers=ag_v_buffers,
                                               ag_v_buffer=ag_v_buffer, ag_v_buffers_ptr=ag_v_buffers_ptr,
                                               attn_output_buffer=attn_output_buffer, ag_stream=ag_stream,
                                               barrier=barrier)

    return ctx


##################################################


def cp_engine_producer_kv_all_gather(
    k_shard: torch.Tensor,  # [total_kv_shard, kv_head, head_dim]
    v_shard: torch.Tensor,  # [total_kv_shard, kv_head, head_dim]
    k_buffer: torch.Tensor,  # [total_kv, kv_head, head_dim]
    v_buffer: torch.Tensor,  # [total_kv, kv_head, head_dim]
    k_buffers: List[torch.Tensor],
    v_buffers: List[torch.Tensor],
    cu_seqlens_k: torch.Tensor,  # kv_full_lens
    rank: int,
    world_size: int,
    ag_stream: torch.cuda.Stream,
    compute_stream: torch.cuda.Stream,
    barrier: BarrierAllContext,
):
    assert k_buffer.is_contiguous()
    assert v_buffer.is_contiguous()
    assert k_shard.is_contiguous()
    assert v_shard.is_contiguous()

    total_kv_shard, kv_head, head_dim = k_shard.shape
    batch_size = cu_seqlens_k.shape[0] - 1

    byte_per_token = kv_head * head_dim * k_shard.dtype.itemsize

    def _cp_engine_copy_data(dst_ptr, src_ptr, cp_size, stream):
        (err, ) = cudart.cudaMemcpyAsync(
            dst_ptr,
            src_ptr,
            cp_size,
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            stream.cuda_stream,
        )

        CUDA_CHECK(err)

    # local copy in compute stream
    with torch.cuda.stream(compute_stream):
        for i in range(batch_size):
            cu_seqlens_k_start = cu_seqlens_k[i].item()
            cu_seqlens_k_end = cu_seqlens_k[i + 1].item()
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            k_shard_len = seqlen_k // world_size
            byte_start = cu_seqlens_k_start * byte_per_token
            byte_per_rank = k_shard_len * byte_per_token
            cp_size = byte_per_rank

            k_dst_ptr = k_buffers[rank].data_ptr() + byte_start + rank * byte_per_rank
            k_src_ptr = k_shard.data_ptr() + byte_start // world_size
            _cp_engine_copy_data(k_dst_ptr, k_src_ptr, cp_size, compute_stream)

            v_dst_ptr = v_buffers[rank].data_ptr() + byte_start + rank * byte_per_rank
            v_src_ptr = v_shard.data_ptr() + byte_start // world_size
            _cp_engine_copy_data(v_dst_ptr, v_src_ptr, cp_size, compute_stream)

    barrier_all_on_stream(barrier, compute_stream)
    ag_stream.wait_stream(compute_stream)

    with torch.cuda.stream(ag_stream):
        for i in range(batch_size):
            cu_seqlens_k_start = cu_seqlens_k[i].item()
            cu_seqlens_k_end = cu_seqlens_k[i + 1].item()
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            k_shard_len = seqlen_k // world_size
            byte_start = cu_seqlens_k_start * byte_per_token
            byte_per_rank = k_shard_len * byte_per_token
            cp_size = byte_per_rank
            for offset in range(1, world_size):
                src_rank = (rank + offset) % world_size

                k_src_ptr = (k_buffers[src_rank].data_ptr() + byte_start + src_rank * byte_per_rank)
                k_dst_ptr = (k_buffers[rank].data_ptr() + byte_start + src_rank * byte_per_rank)
                _cp_engine_copy_data(k_dst_ptr, k_src_ptr, cp_size, ag_stream)

                v_src_ptr = (v_buffers[src_rank].data_ptr() + byte_start + src_rank * byte_per_rank)
                v_dst_ptr = (v_buffers[rank].data_ptr() + byte_start + src_rank * byte_per_rank)
                _cp_engine_copy_data(v_dst_ptr, v_src_ptr, cp_size, ag_stream)

    barrier_all_on_stream(barrier, ag_stream)
    compute_stream.wait_stream(ag_stream)


@triton.jit
def _flash_attn_forward_inner(
    acc,
    l_i,
    m_i,
    q,
    global_offset_q,  #
    K_block_ptr,
    V_block_ptr,  #
    start_m,
    qk_scale,  #
    q_len,
    kv_len,
    kv_len_per_sp_block,
    world_size,
    offset_per_rank,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
):
    assert q_len <= kv_len
    prefix_len = kv_len - q_len
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, prefix_len + global_offset_q + start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = (
            prefix_len + global_offset_q + start_m * BLOCK_M,
            prefix_len + global_offset_q + (start_m + 1) * BLOCK_M,
        )
    else:
        lo, hi = 0, kv_len

    for start_n in range(lo, hi, BLOCK_N):
        wait_offset = start_n
        sp_block_idx = wait_offset // kv_len_per_sp_block
        wait_rank = (sp_block_idx if sp_block_idx < world_size else 2 * world_size - sp_block_idx - 1)
        kv_load_offset = (wait_offset % kv_len_per_sp_block + sp_block_idx // world_size * kv_len_per_sp_block +
                          wait_rank * offset_per_rank)
        k_load_block_ptr = tl.advance(K_block_ptr, (0, kv_load_offset))
        k = tl.load(k_load_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = (prefix_len + global_offset_q + offs_m[:, None]) >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v_load_block_ptr = tl.advance(V_block_ptr, (kv_load_offset, 0))
        v = tl.load(v_load_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def kernel_consumer_flash_attn_forward(
    Q,  # [total_q_shard, q_head, head_dim]
    K,  # [total_kv, kv_head, head_dim]
    V,  # [total_kv, kv_head, head_dim]
    sm_scale,
    Out,  # [total_q_shard, q_head, head_dim]
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    cu_seqlens_q,  # q_shard_lens
    cu_seqlens_k,  # kv_full_lens
    HQ,
    HK,
    enable_zig_zag: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)

    rank = dl.rank()
    world_size = dl.num_ranks()

    cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
    cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
    q_shard_len = cu_seqlens_q_end - cu_seqlens_q_start
    q_len = q_shard_len * world_size
    if start_m * BLOCK_M > q_shard_len:
        return

    cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
    cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
    kv_len = cu_seqlens_k_end - cu_seqlens_k_start
    kv_shard_len = kv_len // world_size

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    dtype = Out.dtype.element_ty
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    group_size = HQ // HK
    off_h_kv = off_h_q // group_size if group_size != 1 else off_h_q

    q_offset = (off_z.to(tl.int64) * stride_qz + off_h_q.to(tl.int64) * stride_qh +
                (cu_seqlens_q_start + start_m * BLOCK_M) * stride_qm)
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(q_shard_len, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    o_offset = (off_z.to(tl.int64) * stride_oz + off_h_q.to(tl.int64) * stride_oh +
                (cu_seqlens_q_start + start_m * BLOCK_M) * stride_om)
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(q_shard_len, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    k_offset = (off_z.to(tl.int64) * stride_kz + off_h_kv.to(tl.int64) * stride_kh + cu_seqlens_k_start * stride_kn)
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, kv_len),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    v_offset = (off_z.to(tl.int64) * stride_vz + off_h_kv.to(tl.int64) * stride_vh + cu_seqlens_k_start * stride_vk)
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(kv_len, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    if not enable_zig_zag:
        global_offset_q = q_shard_len * rank
        kv_len_per_sp_block = kv_len // world_size
    else:
        half_q_shard_len = q_shard_len // 2
        if start_m * BLOCK_M < half_q_shard_len:
            global_offset_q = rank * half_q_shard_len
        else:
            global_offset_q = q_len - (rank + 1) * half_q_shard_len
            # correct the extra offset of `start_m`
            global_offset_q -= half_q_shard_len
        kv_len_per_sp_block = kv_len // (2 * world_size)

    q = tl.load(Q_block_ptr)

    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _flash_attn_forward_inner(
            acc,
            l_i,
            m_i,
            q,
            global_offset_q,
            K_block_ptr,
            V_block_ptr,
            start_m,
            qk_scale,  #
            q_len,
            kv_len,  #
            kv_len_per_sp_block,
            world_size,
            kv_shard_len,
            BLOCK_M,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
        )

    # stage 1: off-band
    if STAGE & 1:
        acc, l_i, m_i = _flash_attn_forward_inner(
            acc,
            l_i,
            m_i,
            q,
            global_offset_q,
            K_block_ptr,
            V_block_ptr,
            start_m,
            qk_scale,  #
            q_len,
            kv_len,  #
            kv_len_per_sp_block,
            world_size,
            kv_shard_len,
            BLOCK_M,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
        )

    # epilogue
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(dtype))


def get_compute_config():
    return (128, 64, 8, 3)


def fused_sp_ag_attn_intra_node(
    ctx: SPAllGatherAttentionContextIntraNode,
    q_shard: torch.Tensor,  # [total_q_shard, q_head, head_dim]
    k_shard: torch.Tensor,  # [total_kv_shard, kv_head, head_dim]
    v_shard: torch.Tensor,  # [total_kv_shard, kv_head, head_dim]
    output: torch.Tensor,  # [total_q_shard, q_head, head_dim]
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    rank: int,
    world_size: int,
    is_causal: bool = True,
    enable_zig_zag: bool = True,
):
    BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES = get_compute_config()

    compute_stream = torch.cuda.current_stream()
    ag_k = ctx.ag_k_buffers[rank]
    ag_v = ctx.ag_v_buffers[rank]

    ctx.ag_stream.wait_stream(compute_stream)
    # kv all gather
    cp_engine_producer_kv_all_gather(
        k_shard,
        v_shard,
        ag_k,
        ag_v,
        ctx.ag_k_buffers,
        ctx.ag_v_buffers,
        cu_seqlens_k,
        rank,
        world_size,
        ctx.ag_stream,
        compute_stream,
        ctx.barrier,
    )

    # flash attn
    stage = 3 if is_causal else 1
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q_shard.shape[-1], k_shard.shape[-1]
    HEAD_DIM_V = v_shard.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    sm_scale = 1 / math.sqrt(HEAD_DIM_Q)

    with torch.cuda.stream(compute_stream):
        grid = lambda args: (
            triton.cdiv(max_seqlen_q, args["BLOCK_M"]),  # max_num_blocks_m
            q_shard.shape[1],  # q_head
            cu_seqlens_q.shape[0] - 1,  # batch_size
        )
        kernel_consumer_flash_attn_forward[grid](
            q_shard,  # [total_q_shard, q_head, head_dim]
            ag_k,  # [total_kv, kv_head, head_dim]
            ag_v,  # [total_kv, kv_head, head_dim]
            sm_scale,
            output,  # [total_q_shard, q_head, head_dim]
            0,
            q_shard.stride(1),
            q_shard.stride(0),
            q_shard.stride(2),
            0,
            ag_k.stride(1),
            ag_k.stride(0),
            ag_k.stride(2),
            0,
            ag_v.stride(1),
            ag_v.stride(0),
            ag_v.stride(2),
            0,
            output.stride(1),
            output.stride(0),
            output.stride(2),
            cu_seqlens_q,
            cu_seqlens_k,
            q_shard.shape[1],  # HQ
            ag_k.shape[1],  # HK
            enable_zig_zag,
            HEAD_DIM_K,
            BLOCK_M,
            BLOCK_N,
            stage,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )

    compute_stream.wait_stream(ctx.ag_stream)
    barrier_all_on_stream(ctx.barrier, compute_stream)
