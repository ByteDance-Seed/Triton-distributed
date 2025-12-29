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
from cuda import cudart
from typing import Optional, List
from dataclasses import dataclass
from triton_dist.language.extra.language_extra import tid, st, __syncthreads, atomic_add
from triton_dist.kernels.nvidia.common_ops import _wait_eq_cuda, BarrierAllContext, barrier_all_on_stream

from triton_dist.utils import (nvshmem_create_tensors, nvshmem_free_tensor_sync, NVSHMEM_SIGNAL_DTYPE,
                               nvshmem_create_tensor)
from triton_dist.kernels.nvidia.common_ops import nvshmem_barrier_all_on_stream
from triton_dist.utils import CUDA_CHECK


@dataclass
class SPConfig:

    max_seq: int
    q_nheads: int
    kv_nheads: int
    k_head_dim: int
    v_head_dim: int

    rank: int
    world_size: int
    local_world_size: int

    data_dtype: torch.dtype
    scale_dtype: Optional[torch.dtype]

    def has_scale(self):
        return self.scale_dtype is not None


@dataclass
class UlyssesSpInferPreAttnContext:
    sp_config: SPConfig

    recv_bufs: List[torch.Tensor]  # [seq_len, local_nheads * head_dim]
    recv_buf_ptrs: torch.Tensor  # [local_world_size, ]
    recv_scale_bufs: List[torch.Tensor]  # [seq_len, local_nheads * head_dim]
    all_signal_buf: torch.Tensor  # [2 * local_world_size,], i64

    barrier_ctx: BarrierAllContext

    a2a_stream: torch.cuda.Stream

    @staticmethod
    def create(max_seq, q_nheads, kv_nheads, k_head_dim, v_head_dim, rank, world_size, local_world_size, data_dtype,
               scale_dtype=torch.float32, pack_scale=False, a2a_stream=None) -> "UlyssesSpInferPreAttnContext":
        sp_config = SPConfig(max_seq=max_seq, q_nheads=q_nheads, kv_nheads=kv_nheads, k_head_dim=k_head_dim,
                             v_head_dim=v_head_dim, rank=rank, world_size=world_size, local_world_size=local_world_size,
                             data_dtype=data_dtype, scale_dtype=scale_dtype)

        min_scale_block_size = 16
        assert world_size == local_world_size, "only support single node"
        assert q_nheads % world_size == 0 and kv_nheads % world_size == 0
        local_q_nheads = q_nheads // world_size
        local_kv_nheads = kv_nheads // world_size
        head_msg_size = local_q_nheads * k_head_dim + local_kv_nheads * k_head_dim + local_kv_nheads * v_head_dim
        barrier_ctx = BarrierAllContext(is_intra_node=True)
        recv_bufs = nvshmem_create_tensors([max_seq, head_msg_size], data_dtype, rank, local_world_size)
        recv_scale_bufs = nvshmem_create_tensors([max_seq, head_msg_size // min_scale_block_size], scale_dtype, rank,
                                                 local_world_size)
        recv_buf_ptrs = torch.tensor([t.data_ptr() for t in recv_bufs], dtype=torch.int64).cuda()
        all_signal_buf = nvshmem_create_tensor([
            2 * world_size,
        ], NVSHMEM_SIGNAL_DTYPE)  # signal_buf | counter_buf
        all_signal_buf.zero_()

        if a2a_stream is None:
            a2a_stream = torch.cuda.Stream()
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        return UlyssesSpInferPreAttnContext(
            sp_config=sp_config,
            recv_bufs=recv_bufs,
            recv_buf_ptrs=recv_buf_ptrs,
            recv_scale_bufs=recv_scale_bufs,
            all_signal_buf=all_signal_buf,
            barrier_ctx=barrier_ctx,
            a2a_stream=a2a_stream,
        )

    @property
    def recv_buf(self):
        return self.recv_bufs[self.sp_config.rank % self.sp_config.local_world_size]

    @property
    def recv_scale_buf(self):
        return self.recv_scale_bufs[self.sp_config.rank % self.sp_config.local_world_size]

    @property
    def signal_buf(self):
        return self.all_signal_buf[:self.sp_config.local_world_size]

    @property
    def counter_buf(self):
        return self.all_signal_buf[self.sp_config.local_world_size:].view(torch.int32)

    def finalize(self):
        nvshmem_free_tensor_sync(self.recv_buf)
        nvshmem_free_tensor_sync(self.recv_scale_buf)
        nvshmem_free_tensor_sync(self.all_signal_buf)
        self.barrier_ctx.finalize()

    def reset_signal(self):
        self.all_signal_buf.zero_()

    def check_ctx(self, seq_len, out_feat):
        assert seq_len <= self.sp_config.max_seq, f"seq_len {seq_len} exceeds max_seq {self.sp_config.max_seq}"
        assert self.sp_config.q_nheads * self.sp_config.k_head_dim + self.sp_config.kv_nheads * (
            self.sp_config.v_head_dim + self.sp_config.k_head_dim) == out_feat


@triton.jit(do_not_specialize=["M"])
def kernel_gemm_a2a_producer_gemm_with_quant_persistent(
    input_data_ptr,
    weight_ptr,
    output_ptr,
    output_scale_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    barrier_ptr,
    counter_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Data type flags
    USE_INT8: tl.constexpr,
    USE_FP8: tl.constexpr,
    # Group size for swizzling
    GROUP_SIZE_N: tl.constexpr = 8,
    INPUT_IS_E5M2: tl.constexpr = True,
    FP8_FAST_ACCUM: tl.constexpr = False,
    USE_CHUNK_LAYOUT: tl.constexpr = False,
    QUNAT_OUT: tl.constexpr = False,
):
    tl.static_assert(N % world_size == 0)
    LOCAL_N: tl.constexpr = N // world_size
    tl.static_assert(LOCAL_N % BLOCK_N == 0, f"LOCAL_N={LOCAL_N} must be divisible by BLOCK_N={BLOCK_N}")

    block_id = tl.program_id(0)
    num_sm = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    for tile_id in range(block_id, num_pid_m * num_pid_n, num_sm):
        # rank swizzle
        rank_pid_n_offset = (rank + 1) % world_size * LOCAL_N // BLOCK_N
        pid = (tile_id + rank_pid_n_offset * num_pid_m) % (num_pid_m * num_pid_n)

        num_pid_in_group = GROUP_SIZE_N * num_pid_m
        group_id = pid // num_pid_in_group
        first_pid_n = group_id * GROUP_SIZE_N
        group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)

        pid_m = (pid % num_pid_in_group) // group_size_n
        pid_n = first_pid_n + (pid % num_pid_in_group) % group_size_n

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        if INPUT_IS_E5M2:
            FP8_TYPE = tl.float8e5
        else:
            FP8_TYPE = tl.float8e4nv

        if USE_INT8:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        if USE_INT8 or USE_FP8:
            a_scale = tl.load(input_scale_ptr + rm, mask=rm < M, other=1.0).to(tl.float32)
            b_scale = tl.load(weight_scale_ptr + rn, mask=rn < N, other=1.0).to(tl.float32)

        for k in range(0, K, BLOCK_K):
            rk = k + tl.arange(0, BLOCK_K)

            a_ptrs = input_data_ptr + rm[:, None] * K + rk[None, :]
            a_mask = (rm[:, None] < M) & (rk[None, :] < K)

            if USE_INT8:
                a = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.int8)
            elif USE_FP8:
                a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(FP8_TYPE)
            else:
                a = tl.load(a_ptrs, mask=a_mask, other=0)

            b_ptrs = weight_ptr + rn[:, None] * K + rk[None, :]
            b_mask = (rn[:, None] < N) & (rk[None, :] < K)

            if USE_INT8:
                b = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.int8)
            elif USE_FP8:
                b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(FP8_TYPE)
            else:
                b = tl.load(b_ptrs, mask=b_mask, other=0)

            if USE_FP8 and FP8_FAST_ACCUM:
                acc = tl.dot(a, tl.trans(b), acc=acc)
            else:
                acc += tl.dot(a, tl.trans(b))

        if USE_INT8:
            combined_scale = a_scale[:, None] * b_scale[None, :]
            acc_fp = acc.to(tl.float32)
            c = (acc_fp * combined_scale)
        elif USE_FP8:
            combined_scale = a_scale[:, None] * b_scale[None, :]
            acc = acc * combined_scale
            c = acc
        else:
            c = acc

        if QUNAT_OUT:
            # bf16 -> fp8
            c = c.to(tl.bfloat16)

            # OUT_TYPE_AFTER_QUANT = tl.pointer_type(tl.float8e4nv)
            FP8_MAX_INV = tl.constexpr(1 / 448.)
            scale = tl.max(tl.abs(c), 1, keep_dims=True).to(tl.float32) * FP8_MAX_INV
            c = (c.to(tl.float32) / scale)
        else:
            c = c.to(tl.bfloat16)

        if USE_CHUNK_LAYOUT:
            out_offset_per_chunk = M * LOCAL_N
            chunk_id = pid_n * BLOCK_N // LOCAL_N
            rn_of_chunk = rn % LOCAL_N

            c_ptrs = output_ptr + chunk_id * out_offset_per_chunk + rm[:, None] * LOCAL_N + rn_of_chunk[None, :]
        else:
            c_ptrs = output_ptr + rm[:, None] * N + rn[None, :]

        if QUNAT_OUT:
            if USE_CHUNK_LAYOUT:
                chunk_id = pid_n * BLOCK_N // LOCAL_N
                c_scale_ptrs = output_scale_ptr + chunk_id * M * LOCAL_N // BLOCK_N + rm * (
                    LOCAL_N // BLOCK_N) + pid_n % (LOCAL_N // BLOCK_N)
            else:
                c_scale_ptrs = output_scale_ptr + rm * (N // BLOCK_N)
            c_scale_mask = rm < M
            tl.store(c_scale_ptrs, scale.reshape(BLOCK_M), mask=c_scale_mask)

        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

        __syncthreads()
        PID_N_PER_RANK = LOCAL_N // BLOCK_N
        signal_idx = pid_n // PID_N_PER_RANK

        if tid(0) == 0:
            val = atomic_add(counter_ptr + signal_idx, 1, scope='gpu', semantic='release')
            if val == (num_pid_m * PID_N_PER_RANK - 1):
                st(barrier_ptr + signal_idx, 1, scope='gpu', semantic='release')


def ulysses_gemm_a2a_produer_gemm(
    ctx: UlyssesSpInferPreAttnContext,
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    fp8_fast_acc: bool = False,
    quant_out: bool = False,
    use_chunk_layout: bool = False,
    num_sms: int = 0,
) -> torch.Tensor:
    M, K = input.shape
    N, K_w = weight.shape
    assert K == K_w, f"Dimension mismatch: input K={K}, weight K={K_w}"

    dtype = input.dtype
    use_int8 = dtype == torch.int8
    use_fp8 = dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    if not quant_out:
        output_dtype = torch.bfloat16 if (use_int8 or use_fp8) else input.dtype
    else:
        output_dtype = torch.float8_e4m3fn
    output = torch.empty((M, N), dtype=output_dtype, device=input.device)

    if use_int8 or use_fp8:
        assert input_scale is not None, "Input scale required for quantized types"
        assert weight_scale is not None, "Weight scale required for quantized types"
        if input_scale.dim() == 2:
            input_scale = input_scale.squeeze(1)
        if weight_scale.dim() == 2:
            weight_scale = weight_scale.squeeze(0)

    world_size = ctx.sp_config.world_size
    assert N % world_size == 0, f"N={N} must be divisible by world_size={world_size}"
    local_N = N // world_size
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    assert local_N % BLOCK_N == 0, f"local_N={local_N} must be divisible by BLOCK_N={BLOCK_N}"
    # GROUP_SIZE_N = local_N // BLOCK_N
    GROUP_SIZE_N = 1
    assert BLOCK_N == ctx.sp_config.k_head_dim
    if not quant_out:
        output_scale = None
    else:
        output_scale = torch.empty((M, N // BLOCK_N), dtype=torch.float32, device=input.device)

    if num_sms <= 0:
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    else:
        grid = (num_sms, )

    if weight.dtype == torch.float8_e5m2:
        input_is_e5m2_flag = True
    else:
        input_is_e5m2_flag = False

    producer_gemm = kernel_gemm_a2a_producer_gemm_with_quant_persistent

    producer_gemm[grid](
        input,
        weight,
        output,
        output_scale,
        input_scale if input_scale is not None else input,
        weight_scale if weight_scale is not None else weight,
        ctx.signal_buf,
        ctx.counter_buf,
        M,
        N,
        K,
        ctx.sp_config.rank,  # rank
        ctx.sp_config.world_size,  # world_size
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        use_int8,
        use_fp8,
        GROUP_SIZE_N=GROUP_SIZE_N,
        INPUT_IS_E5M2=input_is_e5m2_flag,
        FP8_FAST_ACCUM=fp8_fast_acc,
        USE_CHUNK_LAYOUT=use_chunk_layout,
        QUNAT_OUT=quant_out,
    )

    return output, output_scale


def _ce_p2p(src_ptr, dst_ptr, nbytes):
    """ no check dtype. no check device/host. no check tensor size. no check contiguous.
    """
    current_stream = torch.cuda.current_stream()
    err, = cudart.cudaMemcpyAsync(dst_ptr, src_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                                  current_stream.cuda_stream)
    CUDA_CHECK(err)


def sp_intra_node_barrier(ctx):
    # nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    barrier_all_on_stream(ctx.barrier_ctx, torch.cuda.current_stream())


def pre_attn_a2a_comm_only(ctx: UlyssesSpInferPreAttnContext, input: torch.Tensor,  # [local_seq, out_feat]
                           input_scale: Optional[torch.Tensor] = None, signal_buf: Optional[torch.Tensor] = None,
                           use_chunk_layout: bool = False):
    rank = ctx.sp_config.rank
    sp_size = ctx.sp_config.world_size
    local_seq, out_feat = input.shape
    input_dtype = input.dtype
    seq_len = local_seq * sp_size
    assert out_feat % sp_size == 0
    local_out_feat = out_feat // sp_size
    assert local_out_feat <= ctx.recv_buf.shape[1]

    if use_chunk_layout:
        input = input.reshape(sp_size, local_seq, local_out_feat)
    else:
        input = input.reshape(local_seq, sp_size, local_out_feat)

    nbytes_per_rank = local_seq * local_out_feat * input_dtype.itemsize
    if input_scale is not None:
        if use_chunk_layout:
            input_scale = input_scale.reshape(sp_size, local_seq, -1)
        else:
            input_scale = input_scale.reshape(local_seq, sp_size, -1)
        scale_dtype = input_scale.dtype
        nbytes_scale_per_rank = local_seq * input_scale.shape[-1] * scale_dtype.itemsize

    recv_scale_bufs_ptr = [ctx.recv_scale_bufs[tgt_rank].data_ptr() for tgt_rank in range(sp_size)]
    recv_bufs_ptr = [ctx.recv_bufs[tgt_rank].data_ptr() for tgt_rank in range(sp_size)]

    if sp_size > 1:
        sp_intra_node_barrier(ctx)
    for step in range(sp_size):
        target_rank = (rank + 1 + step) % sp_size
        if signal_buf is not None:
            _wait_eq_cuda(signal_buf[target_rank], 1)
        if use_chunk_layout:
            src_data = input[target_rank].contiguous()
        else:
            src_data = input[:, target_rank, :].contiguous()
        dst_buf_ptr = recv_bufs_ptr[target_rank] + rank * nbytes_per_rank
        _ce_p2p(src_data.data_ptr(), dst_buf_ptr, nbytes_per_rank)
        if input_scale is not None:
            if use_chunk_layout:
                src_scale_data = input_scale[target_rank].contiguous()
            else:
                src_scale_data = input_scale[:, target_rank, :].contiguous()
            dst_scale_buf_ptr = recv_scale_bufs_ptr[target_rank] + rank * nbytes_scale_per_rank
            _ce_p2p(src_scale_data.data_ptr(), dst_scale_buf_ptr, nbytes_scale_per_rank)

        sp_intra_node_barrier(ctx)

    output = torch.empty((seq_len, local_out_feat), dtype=input.dtype, device=input.device)
    if input_scale is None:
        output_scale = None
    else:
        output_scale = torch.empty((seq_len, input_scale.shape[-1]), dtype=input_scale.dtype, device=input.device)
        _ce_p2p(recv_scale_bufs_ptr[rank], output_scale.data_ptr(), nbytes_scale_per_rank * sp_size)
    _ce_p2p(recv_bufs_ptr[rank], output.data_ptr(), nbytes_per_rank * sp_size)
    return output, output_scale


def ulysses_sp_infer_gemm_a2a_op(
    ctx: UlyssesSpInferPreAttnContext,
    input: torch.Tensor,  # [local_seq, hidden]
    weight: torch.Tensor,  # [out_feat, hidden]
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    fp8_fast_acc: bool = False,
    quant_out: bool = False,
):
    current_stream = torch.cuda.current_stream()
    a2a_stream = ctx.a2a_stream
    a2a_stream.wait_stream(current_stream)

    assert input.shape[1] == weight.shape[1]
    ctx.check_ctx(input.shape[0], weight.shape[0])
    num_total_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_producer_gemm_sm = num_total_sm - 1
    output, output_scale = ulysses_gemm_a2a_produer_gemm(ctx, input, weight, input_scale, weight_scale,
                                                         fp8_fast_acc=fp8_fast_acc, quant_out=quant_out,
                                                         use_chunk_layout=True, num_sms=num_producer_gemm_sm)
    with torch.cuda.stream(a2a_stream):
        output, output_scale = pre_attn_a2a_comm_only(ctx, output, output_scale, signal_buf=ctx.signal_buf,
                                                      use_chunk_layout=True)
    current_stream.wait_stream(a2a_stream)
    ctx.reset_signal()
    return output, output_scale
