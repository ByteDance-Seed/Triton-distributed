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
from torch import nn
import torch.distributed

from triton_dist.kernels.amd.all_gather_gemm import create_ag_gemm_intra_node_context, ag_gemm_intra_node
from triton_dist.kernels.amd.gemm_reduce_scatter import create_gemm_rs_intra_node_context, gemm_rs_intra_node

from flash_attn import flash_attn_with_kvcache
import triton
import triton.language as tl


def shard_local(tensor: torch.Tensor, world_size: int, dim: int, local_rank: int):
    tensor_dim = tensor.shape[dim]
    tensor_slice = tensor_dim // world_size
    if tensor_dim % world_size != 0:
        raise ValueError(f"Tensor dimension {tensor_dim} is not divisible by world size {world_size}.")
    if local_rank < 0 or local_rank >= world_size:
        raise ValueError(f"Local rank {local_rank} is out of bounds for world size {world_size}.")
    if dim < 0 or dim >= tensor.dim():
        raise ValueError(f"Dimension {dim} is out of bounds for tensor with {tensor.dim()} dimensions.")
    if tensor_slice == 0:
        raise ValueError(f"Tensor slice size is zero for tensor dimension {tensor_dim} and world size {world_size}.")
    return tensor.split(tensor_slice, dim=dim)[local_rank].contiguous()


def layer_norm(
    hidden_states: torch.Tensor,
    eps: float,
    w: torch.Tensor,
):
    """Applies RMS Normalization"""
    return nn.functional.rms_norm(hidden_states.view(-1, hidden_states.size(-1)),
                                  normalized_shape=(hidden_states.size(-1), ), weight=w, eps=eps).view_as(hidden_states)


def _set_cos_sin_cache(inv_freq: torch.Tensor, max_length: int):
    """Precomputes cosine and sine cache for rotary position embeddings."""
    t = torch.arange(max_length, device="cuda", dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_sin_cache = torch.cat((emb.cos()[:, :64], emb.sin()[:, :64]), dim=-1)
    return cos_sin_cache


@triton.jit
def apply_rotary_pos_emb_kernel(
    X_ptr,
    Cos_Sin_ptr,
    Position_ids_ptr,
    Output_ptr,
    stride_xb,
    stride_xs,
    stride_xh,
    stride_xe,
    stride_cos_sin,
    stride_pid_b,
    stride_pid_s,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for applying Rotary Position Embedding.
    New layout: x is [bsz, seq, head, head_dim], pos_ids is [bsz, seq]
    """
    # Grid is (bsz, seq, heads)
    b_idx = tl.program_id(0)
    s_idx = tl.program_id(1)
    h_idx = tl.program_id(2)

    pid_offset = b_idx * stride_pid_b + s_idx * stride_pid_s
    pid = tl.load(Position_ids_ptr + pid_offset)

    # Offset for [bsz, seq, head, head_dim]
    x_offset = b_idx * stride_xb + s_idx * stride_xs + h_idx * stride_xh
    x_ptr = X_ptr + x_offset
    output_ptr = Output_ptr + x_offset
    cos_sin_ptr = Cos_Sin_ptr + pid * stride_cos_sin
    half_dim = head_dim // 2

    # Create a range of offsets for the head dimension
    range_offs = tl.arange(0, BLOCK_SIZE)

    # Create masks to avoid out-of-bounds access
    mask = range_offs < half_dim

    # Load the first half of the head vector
    x1 = tl.load(x_ptr + range_offs, mask=mask)

    # Load the second half of the head vector
    x2 = tl.load(x_ptr + range_offs + half_dim, mask=mask)

    # Load cos and sin values
    cos = tl.load(cos_sin_ptr + range_offs, mask=mask)
    sin = tl.load(cos_sin_ptr + range_offs + half_dim, mask=mask)

    output1 = x1 * cos - x2 * sin
    output2 = x2 * cos + x1 * sin
    tl.store(output_ptr + range_offs, output1, mask=mask)
    tl.store(output_ptr + range_offs + half_dim, output2, mask=mask)


def apply_rotary_pos_emb(x: torch.Tensor, cos_sin: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embedding to the input tensor with shape [bsz, seq, head, head_dim].

    Args:
        x: Input tensor of shape (batch_size, seq_len, heads, head_dim)
        cos_sin: Tensor with precomputed cos and sin values of shape (max_seq_len, head_dim)
        position_ids: Tensor indicating the position of each token, shape (batch_size, seq_len).

    Returns:
        Output tensor with rotary embeddings applied.
    """
    # Input shapes validation
    assert x.dim() == 4, "Input tensor x must be 4-dimensional"
    assert position_ids.dim() == 2, "position_ids must be 2-dimensional"

    batch_size, seq_len, heads, head_dim = x.shape

    # Ensure contiguity for optimal performance
    x = x.contiguous()
    cos_sin = cos_sin.contiguous()
    position_ids = position_ids.contiguous()

    output = torch.empty_like(x)

    # Define the grid for launching the kernel. Each program handles one head vector.
    grid = (batch_size, seq_len, heads)

    # BLOCK_SIZE should be a power of 2 and cover half of the head_dim for optimal performance
    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)

    apply_rotary_pos_emb_kernel[grid](
        x, cos_sin, position_ids, output,
        # Strides for x: [bsz, seq, head, head_dim]
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        # Stride for cos_sin cache
        cos_sin.stride(0),
        # Strides for position_ids: [bsz, seq]
        position_ids.stride(0), position_ids.stride(1), head_dim=head_dim, BLOCK_SIZE=BLOCK_SIZE)

    return output


class TP_Attn:
    """
    Tensor Parallel Attention.
    QKV Projection: Column Parallelism on weights (sharded over head dimension).
    Output Projection: Row Parallelism on weights.
    """

    def __init__(self, rank=0, world_size=8, group=None):
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.head_dim = 128
        self.wqkv = None
        self.wo = None

    def _init_parameters(self, self_attn: nn.Module, verbose=False):
        self.q_size = self_attn.q_proj.weight.shape[0] // self.world_size
        self.kv_size = self_attn.k_proj.weight.shape[0] // self.world_size
        wq = shard_local(self_attn.q_proj.weight.detach(), self.world_size, 0, self.rank)
        wk = shard_local(self_attn.k_proj.weight.detach(), self.world_size, 0, self.rank)
        wv = shard_local(self_attn.v_proj.weight.detach(), self.world_size, 0, self.rank)
        self.wqkv = torch.cat((wq, wk, wv), dim=0).to("cuda", non_blocking=True)  # [qkv_dim, hidden_size]
        self.wo = shard_local(self_attn.o_proj.weight.detach(), self.world_size, 1,
                              self.rank).to("cuda", non_blocking=True)

        self.ag_N_per_rank = self.wqkv.shape[0]
        self.K = self.wqkv.shape[1]
        self.dtype = self.wqkv.dtype

        if hasattr(self_attn, "q_norm"):
            self.q_norm_eps = self_attn.q_norm.variance_epsilon
            self.q_norm_w = self_attn.q_norm.weight.detach().to("cuda", non_blocking=True)
        if hasattr(self_attn, "k_norm"):
            self.k_norm_eps = self_attn.k_norm.variance_epsilon
            self.k_norm_w = self_attn.k_norm.weight.detach().to("cuda", non_blocking=True)

        if verbose:
            print(f"[RANK {self.rank}] Attn initialized with parameters: qkv ({self.wqkv.shape}, o ({self.wo.shape}))")

    def _init_ctx(self, max_M, ag_intranode_stream, BLOCK_M, BLOCK_N, BLOCK_K, serial=False, ag_internode_stream=None,
                  stages=None):
        if serial:
            print(f"[RANK {self.rank}] Using serial mode for AG-GEMM.")
        self.ag_ctx = create_ag_gemm_intra_node_context(max_M=max_M, N=self.ag_N_per_rank, K=self.K, rank=self.rank,
                                                        num_ranks=self.world_size, input_dtype=self.dtype,
                                                        output_dtype=self.dtype, tp_group=self.group,
                                                        ag_streams=ag_intranode_stream, serial=serial, autotune=True,
                                                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                                        M_PER_CHUNK=256)
        self.rs_ctx = create_gemm_rs_intra_node_context(
            max_M=max_M,
            N=self.K,
            rank=self.rank,
            num_ranks=self.world_size,
            output_dtype=self.dtype,
            tp_group=self.group,
            fuse_scatter=True,
        )
        torch.cuda.synchronize()

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor,
                             cos_sin_cache: torch.Tensor):
        """Applies Rotary Position Embedding inplace."""
        bsz, seq, _ = q.shape
        q = q.view(bsz, seq, -1, self.head_dim)
        k = k.view(bsz, seq, -1, self.head_dim)
        q = apply_rotary_pos_emb(q, cos_sin_cache, position_ids).view(bsz, seq, -1, self.head_dim)
        k = apply_rotary_pos_emb(k, cos_sin_cache, position_ids).view(bsz, seq, -1, self.head_dim)
        return q, k

    @torch.inference_mode()
    def torch_fwd(self, x, position_ids, cos_sin_cache, kv_cache, layer_idx: int):
        """
        Reference PyTorch forward pass for attention with Tensor Parallelism.
        Activations related to head dimensions are sharded. Final output is AllReduced.
        x: input tensor, shape [batch_size, q_len, hidden_size_in] (replicated on each rank)
        """
        bsz, q_len, _ = x.size()
        qkv = torch.nn.functional.linear(x, self.wqkv)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        v = v.view(bsz, q_len, -1, self.head_dim)

        # qk norm
        if hasattr(self, 'q_norm_eps'):
            q = layer_norm(q.contiguous().view(bsz, q_len, -1, self.head_dim), self.q_norm_eps,
                           self.q_norm_w).view(bsz, q_len, -1)
        if hasattr(self, 'k_norm_eps'):
            k = layer_norm(k.contiguous().view(bsz, q_len, -1, self.head_dim), self.k_norm_eps,
                           self.k_norm_w).view(bsz, q_len, -1)
        # RoPE
        q, k = self.apply_rotary_pos_emb(q, k, position_ids, cos_sin_cache)
        k_cache, v_cache, kv_offset = kv_cache.update_kv_cache(k, v, layer_idx)

        # FlashAttn
        out = flash_attn_with_kvcache(q=q, k_cache=k_cache, v_cache=v_cache, k=k, v=v, cache_seqlens=kv_offset,
                                      causal=True)

        out = torch.nn.functional.linear(out.view(bsz, q_len, -1), self.wo)
        if self.world_size > 1:
            torch.distributed.all_reduce(out, torch.distributed.ReduceOp.SUM, group=self.group)
        return out

    @torch.inference_mode()
    def dist_triton_fwd(self, x, position_ids, cos_sin_cache, kv_cache, layer_idx: int):
        """
        triton_dist forward pass.
        Input x is batch-sharded. Output is also batch-sharded.
        x: input tensor, shape [batch_size_per_rank, q_len, hidden_size_in]
        """
        bsz, q_len, d = x.size()

        # ag + gemm
        qkv = ag_gemm_intra_node(x.view(-1, d), self.wqkv, transe_b=False,
                                 ctx=self.ag_ctx).view(bsz * self.world_size, q_len, -1)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        v = v.view(bsz * self.world_size, q_len, -1, self.head_dim)

        # qk norm
        if hasattr(self, 'q_norm_eps'):
            q = layer_norm(q.contiguous().view(bsz * self.world_size, q_len, -1, self.head_dim), self.q_norm_eps,
                           self.q_norm_w).view(bsz * self.world_size, q_len, -1)
        if hasattr(self, 'k_norm_eps'):
            k = layer_norm(k.contiguous().view(bsz * self.world_size, q_len, -1, self.head_dim), self.k_norm_eps,
                           self.k_norm_w).view(bsz * self.world_size, q_len, -1)
        # RoPE
        q, k = self.apply_rotary_pos_emb(q, k, position_ids, cos_sin_cache)
        k_cache, v_cache, kv_offset = kv_cache.update_kv_cache(k, v, layer_idx)

        # FlashAttn
        out = flash_attn_with_kvcache(q=q, k_cache=k_cache, v_cache=v_cache, k=k, v=v, cache_seqlens=kv_offset,
                                      causal=True)

        # gemm + rs
        out = gemm_rs_intra_node(out.view(bsz * self.world_size * q_len, -1), self.wo, self.rs_ctx).view(bsz, q_len, -1)
        return out

    def fwd(self, x: torch.Tensor, position_ids: torch.Tensor, cos_sin_cache: torch.Tensor, kv_cache, layer_idx: int):
        raise NotImplementedError("Please use torch_fwd or dist_triton_fwd instead.")
