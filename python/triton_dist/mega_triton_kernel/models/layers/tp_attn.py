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
from ..paged_kv_cache import PagedKVCache


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


# adapt from triton_dist/layers/nvidia/tp_attn.py
class TPAttnBuilder:
    """
    Tensor Parallel Attention.
    QKV Projection: Column Parallelism on weights (sharded over head dimension).
    Output Projection: Row Parallelism on weights.
    """

    def __init__(self, builder, layer_idx, head_dim=128, rank=0, world_size=8, group=None):
        self._builder = builder
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.head_dim = head_dim
        self.wqkv = None
        self.wo = None
        self.kv_cache = None
        self.layer_idx = layer_idx
        self.soft_cap = 0.0
        self.sm_scale = self.head_dim**-0.5

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
        assert hasattr(self_attn, "q_norm") and hasattr(self_attn, "k_norm")

        if verbose:
            print(f"[RANK {self.rank}] Attn initialized with parameters: qkv ({self.wqkv.shape}, o ({self.wo.shape}))")

    def build_fwd(self, x, cos_cache, sin_cache, kv_cache: PagedKVCache):
        """
        x: input tensor, shape [batch_size, q_len, hidden_size] (replicated on each rank)
        """
        key_cache, value_cache, block_tables, kv_lens = kv_cache.get_layer_kv_cache(self.layer_idx)
        assert hasattr(self, 'q_norm_eps') and hasattr(self, 'k_norm_eps')
        assert len(x.shape) == 3 and x.dtype == torch.bfloat16
        batch_size, q_len, hidden_size = x.shape
        x = x.reshape(-1, hidden_size)
        num_tokens = batch_size * q_len
        qkv_proj_out = torch.empty((num_tokens, self.wqkv.shape[0]), dtype=x.dtype, device=x.device)
        q_norm_rope = torch.empty((batch_size, q_len, self.q_size // self.head_dim, self.head_dim), dtype=x.dtype,
                                  device=x.device)
        attn_out = torch.empty_like(q_norm_rope)
        if self.world_size > 1:
            o_proj_out = self._builder.create_symm_tensor((num_tokens, hidden_size), x.dtype)
            ar_out = torch.empty(num_tokens, hidden_size, dtype=x.dtype, device=x.device)
        else:
            o_proj_out = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
        self._builder.make_qkv_proj(x, self.wqkv, qkv_proj_out)
        qkv_proj_out_bsnh = qkv_proj_out.reshape(batch_size, q_len, -1, self.head_dim)
        self._builder.make_qk_norm_rope_update_kvcache(qkv_proj_out_bsnh, key_cache, value_cache, block_tables, kv_lens,
                                                       self.q_norm_w, self.k_norm_w, cos_cache, sin_cache, q_norm_rope,
                                                       self.q_norm_eps, self.k_norm_eps)
        self._builder.make_flash_decode(q_norm_rope, key_cache, value_cache, block_tables, kv_lens, attn_out,
                                        self.sm_scale, self.soft_cap)
        attn_out_2d = attn_out.reshape(num_tokens, self.q_size)
        self._builder.make_o_proj(attn_out_2d, self.wo, o_proj_out)
        if self.world_size > 1:
            self._builder.make_allreduce(o_proj_out, ar_out, double_input_buffer=True)
            return ar_out.reshape(batch_size, q_len, hidden_size)
        return o_proj_out.reshape(batch_size, q_len, hidden_size)
