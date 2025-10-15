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
import flashinfer

from triton_dist.kernels.allreduce import AllReduceMethod
from triton_dist.kernels.nvidia.allgather_gemm import AllGatherGEMMTensorParallelContext, get_auto_all_gather_method, ag_gemm
from triton_dist.kernels.nvidia import create_gemm_rs_context, gemm_rs
from triton_dist.utils import nvshmem_barrier_all_on_stream
from triton_dist.kernels.nvidia.allreduce import (create_allreduce_ctx, all_reduce)
from triton_dist.layers.nvidia import GemmARLayer

try:
    from flash_attn_interface import flash_attn_with_kvcache
    msg = "Using flash_attn_interface, which is faster for sm90."
except ImportError:
    from flash_attn import flash_attn_with_kvcache
    msg = "Using flash_attn, which is much slower than flash_attn_interface for sm90"
print(msg)


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
    """Applies RMS Normalization using flashinfer."""
    return flashinfer.norm.rmsnorm(hidden_states.view(-1, hidden_states.size(-1)), w, eps).view_as(hidden_states)


def _set_cos_sin_cache(inv_freq: torch.Tensor, max_length: int):
    """Precomputes cosine and sine cache for rotary position embeddings."""
    t = torch.arange(max_length, device="cuda", dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_sin_cache = torch.cat((emb.cos()[:, :64], emb.sin()[:, :64]), dim=-1)
    return cos_sin_cache


class TP_Attn:
    """
    Tensor Parallel Attention.
    QKV Projection: Column Parallelism on weights (sharded over head dimension).
    Output Projection: Row Parallelism on weights.
    """

    def __init__(self, rank=0, world_size=8, group=None):
        # TODO does not support multiple node
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.head_dim = 128
        self.wqkv = None
        self.wo = None
        self.ag_ctx = None
        self.rs_ctx = None
        self.ar_ctx = None

    def _init_parameters(self, self_attn: nn.Module, verbose=False):
        self.q_size = self_attn.q_proj.weight.shape[0] // self.world_size
        self.kv_size = self_attn.k_proj.weight.shape[0] // self.world_size
        wq = shard_local(self_attn.q_proj.weight.detach(), self.world_size, 0, self.rank)
        wk = shard_local(self_attn.k_proj.weight.detach(), self.world_size, 0, self.rank)
        wv = shard_local(self_attn.v_proj.weight.detach(), self.world_size, 0, self.rank)
        self.wqkv: torch.Tensor = torch.cat((wq, wk, wv), dim=0).to("cuda", non_blocking=True)  # [qkv_dim, hidden_size]
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

        # bias
        if self_attn.q_proj.bias is not None:
            bq = shard_local(self_attn.q_proj.bias.detach(), self.world_size, 0, self.rank)
            bk = shard_local(self_attn.k_proj.bias.detach(), self.world_size, 0, self.rank)
            bv = shard_local(self_attn.v_proj.bias.detach(), self.world_size, 0, self.rank)
            self.bqkv = torch.cat((bq, bk, bv), dim=0).to("cuda", non_blocking=True)  # [qkv_dim]

        if verbose:
            print(f"[RANK {self.rank}] Attn initialized with parameters: qkv ({self.wqkv.shape}, o ({self.wo.shape}))")

    def _init_ctx(self, max_M, ag_intranode_stream: torch.cuda.Stream | None,
                  ag_internode_stream: torch.cuda.Stream | None):
        self.ag_ctx = AllGatherGEMMTensorParallelContext(
            N_per_rank=self.ag_N_per_rank, K=self.K, dtype=self.dtype, rank=self.rank, num_ranks=self.world_size,
            num_local_ranks=self.world_size, max_M=max_M, ag_intranode_stream=ag_intranode_stream,
            ag_internode_stream=ag_internode_stream,
            all_gather_method=get_auto_all_gather_method(self.world_size, self.world_size))
        self.rs_ctx = create_gemm_rs_context(
            max_M=max_M,
            N=self.K,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=self.world_size,
            output_dtype=self.dtype,
            rs_stream=ag_intranode_stream,
        )
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    def _init_AR_ctx(self, max_M, method: AllReduceMethod, dtype=torch.bfloat16):
        self.ar_method = method
        self.ar_ctx = create_allreduce_ctx(
            workspace_nbytes=max_M * self.K * dtype.itemsize, rank=self.rank, world_size=self.world_size,
            local_world_size=self.world_size,  # TODO(houqi.1993) does not support multiple nodes now.
        )

    def finalize(self):
        if self.ag_ctx:
            self.ag_ctx.finalize()
        if self.rs_ctx:
            self.rs_ctx.finalize()
        if self.ar_ctx:
            self.ar_ctx.finalize()

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor,
                             cos_sin_cache: torch.Tensor):
        """Applies Rotary Position Embedding inplace."""
        if cos_sin_cache.dtype != torch.float32:
            cos_sin_cache = cos_sin_cache.to(torch.float32)
        bsz, seq, _ = q.shape
        flashinfer.apply_rope_with_cos_sin_cache_inplace(position_ids.contiguous(), q.view(bsz * seq, -1),
                                                         k.view(bsz * seq, -1), self.head_dim, cos_sin_cache, True),
        q = q.view(bsz, seq, -1, self.head_dim)
        k = k.view(bsz, seq, -1, self.head_dim)
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
        if hasattr(self, 'bqkv'):
            qkv = qkv + self.bqkv.view(1, 1, -1).expand(bsz, q_len, -1)

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
    def dist_triton_fwd(self, x, position_ids, cos_sin_cache, kv_cache, layer_idx: int, autotune=True):
        """
        triton_dist forward pass.
        Input x is batch-sharded. Output is also batch-sharded.
        x: input tensor, shape [batch_size_per_rank, q_len, hidden_size_in]
        """
        bsz, q_len, d = x.size()

        # ag + gemm
        qkv = ag_gemm(x.view(-1, d), self.wqkv.T, ctx=self.ag_ctx,
                      autotune=autotune).view(bsz * self.world_size, q_len, -1)
        if hasattr(self, 'bqkv'):
            qkv = qkv + self.bqkv.view(1, 1, -1).expand(bsz * self.world_size, q_len, -1)

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
        out = gemm_rs(out.view(bsz * self.world_size * q_len, -1), self.wo.T, self.rs_ctx,
                      autotune=True).view(bsz, q_len, -1)
        return out

    @torch.inference_mode()
    def dist_triton_AR_fwd(self, x, position_ids, cos_sin_cache, kv_cache, layer_idx: int):
        """
        triton_dist AR forward pass for attention with Tensor Parallelism.
        Activations related to head dimensions are sharded. Final output is AllReduced.
        x: input tensor, shape [batch_size, q_len, hidden_size_in] (replicated on each rank)
        """
        bsz, q_len, _ = x.size()
        qkv = torch.nn.functional.linear(x, self.wqkv)
        if hasattr(self, 'bqkv'):
            qkv = qkv + self.bqkv.view(1, 1, -1).expand(bsz, q_len, -1)

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

        out = torch.nn.functional.linear(out.view(bsz, q_len, -1), self.wo).view(bsz * q_len, -1)
        if self.world_size > 1:
            out_allreduce = torch.empty_like(out)
            out = all_reduce(x=out.contiguous(), output=out_allreduce, method=self.ar_method, ctx=self.ar_ctx)
        return out.view(bsz, q_len, -1)

    def _init_gemm_ar_ctx(self, max_M, dtype=torch.bfloat16):
        N = self.wo.shape[0]
        K = self.wo.shape[1]
        self.gemm_ar_ctx = GemmARLayer(self.group, max_M, N, K, dtype, dtype, self.world_size, persistent=True,
                                       use_ll_kernel=max_M <= 256, copy_to_local=False,
                                       NUM_COMM_SMS=16 if max_M <= 256 else 4)

    @torch.inference_mode()
    def dist_triton_gemm_ar_fwd(self, x, position_ids, cos_sin_cache, kv_cache, layer_idx: int):
        bsz, q_len, _ = x.size()
        qkv = torch.nn.functional.linear(x, self.wqkv)
        if hasattr(self, 'bqkv'):
            qkv = qkv + self.bqkv.view(1, 1, -1).expand(bsz, q_len, -1)

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
        out = self.gemm_ar_ctx.forward(out.view(bsz * q_len, -1), self.wo).view(bsz, q_len, -1)
        return out

    def fwd(self, x: torch.Tensor, position_ids: torch.Tensor, cos_sin_cache: torch.Tensor, kv_cache, layer_idx: int):
        raise NotImplementedError("Please use torch_fwd or dist_triton_fwd instead.")
