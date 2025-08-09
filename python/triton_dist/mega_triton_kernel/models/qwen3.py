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
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from triton_dist.models import ModelConfig
from triton_dist.models.utils import init_model_cpu
from .utils import prepare_cos_sin_cache
from .layers import TPMLPBuilder, TPAttnBuilder
from .paged_kv_cache import PagedKVCache
from .model_builder import ModelBuilder


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


# adapt from triton_dist/models/qwen.py
class Qwen3LayerBuilder:
    """
    A single layer of Qwen3 model, containing self-attention and MLP.
    This layer is designed to be used in a tensor parallel setting.
    It initializes the parameters and sets the forward pass method based on the mode.
    """

    def __init__(self, builder, layer_idx, head_dim, rank: int = 0, world_size: int = 1) -> None:
        self._builder = builder
        self.attn = None
        self.mlp = None
        self.input_norm_eps = None
        self.input_norm_w = None
        self.post_norm_eps = None
        self.post_norm_w = None
        self.rank = rank
        self.world_size = world_size
        self.layer_idx = layer_idx
        self.head_dim = head_dim

    def init_parameters(self, hf_layer: Qwen3DecoderLayer):
        self.mlp = TPMLPBuilder(builder=self._builder, rank=self.rank, world_size=self.world_size)
        self.mlp._init_parameters(hf_layer.mlp)

        self.attn = TPAttnBuilder(builder=self._builder, layer_idx=self.layer_idx, head_dim=self.head_dim,
                                  rank=self.rank, world_size=self.world_size)
        self.attn._init_parameters(hf_layer.self_attn)

        self.input_norm_eps = hf_layer.input_layernorm.variance_epsilon
        self.input_norm_w = hf_layer.input_layernorm.weight.detach().cuda()
        self.post_norm_eps = hf_layer.post_attention_layernorm.variance_epsilon
        self.post_norm_w = hf_layer.post_attention_layernorm.weight.detach().cuda()

    def build_fwd(self, hidden_states: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor,
                  kv_cache: PagedKVCache):
        assert len(hidden_states.shape) == 3 and hidden_states.dtype == torch.bfloat16
        batch_size, seq_len, hidden_size = hidden_states.shape
        input_norm_out = torch.empty_like(hidden_states)
        attn_residual_out = torch.empty_like(hidden_states)

        post_norm_out = torch.empty_like(hidden_states)
        mlp_residual_out = torch.empty_like(hidden_states)

        # attn
        self._builder.make_rms_norm(hidden_states, self.input_norm_w, input_norm_out, self.input_norm_eps)
        attn_out = self.attn.build_fwd(input_norm_out, cos_cache, sin_cache, kv_cache)
        self._builder.make_add(hidden_states, attn_out, attn_residual_out)

        # mlp
        self._builder.make_rms_norm(
            attn_residual_out, self.post_norm_w, post_norm_out,
            self.post_norm_eps)  # post_norm_out = rms_norm(attn_residual_out, post_norm_out, self.post_norm_eps)
        mlp_out = self.mlp.build_fwd(post_norm_out)  # mlp_out = mlp(post_norm_out)
        self._builder.make_add(attn_residual_out, mlp_out,
                               mlp_residual_out)  # mlp_residual_out = attn_residual_out + mlp_out
        return mlp_residual_out


class Qwen3Model:
    """
    Qwen3 model implementation for tensor parallel training.
    This model initializes the parameters, sets the forward pass method, and provides an inference method.
    It supports both torch and triton_dist modes for forward pass.
    """

    def __init__(self, batch_size, model_config: ModelConfig, builder: 'ModelBuilder', build_lm_head=True) -> None:
        self._builder = builder
        self.dtype = model_config.dtype
        self.config = Qwen3Config.from_pretrained(model_config.model_name, local_files_only=model_config.local_only)
        self.model_name = model_config.model_name
        self.max_length = model_config.max_length
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim
        self.num_key_value_heads = self.config.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.batch_size = batch_size
        self.rank = model_config.rank
        self.world_size = model_config.world_size
        self.eos_token_id = self.config.eos_token_id
        self.build_lm_head = build_lm_head
        self.init_parameters()
        self.hidden_state_buffer = torch.empty((batch_size, 1, self.hidden_size), dtype=self.dtype,
                                               device=torch.cuda.current_device())
        self.kv_cache = PagedKVCache(num_layers=self.num_layers, batch_size=self.batch_size, max_length=self.max_length,
                                     num_kv_heads=self.num_key_value_heads // self.world_size, head_dim=self.head_dim,
                                     dtype=self.dtype)
        self.mega_out = self.build_fwd(self.hidden_state_buffer, self.kv_cache)
        self._builder.compile()
        torch.cuda.synchronize()
        if self.world_size > 1:
            torch.distributed.barrier()

    def init_parameters(self):
        hf_model = init_model_cpu(self.model_name, dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().cuda()
        self.lm_head = hf_model.lm_head.weight.detach().cuda()
        self.norm_weight = hf_model.model.norm.weight.detach().cuda()
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        cos_cache, sin_cache = prepare_cos_sin_cache(self.head_dim, max_position_embeddings=self.max_length,
                                                     rope_theta=self.rope_theta)

        self.sin_cache = sin_cache.to(torch.float32).unsqueeze(0)
        self.cos_cache = cos_cache.to(torch.float32).unsqueeze(0)

        self.layers: list[Qwen3LayerBuilder] = []

        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = Qwen3LayerBuilder(builder=self._builder, layer_idx=idx, head_dim=self.head_dim, rank=self.rank,
                                      world_size=self.world_size)
            layer.init_parameters(hf_layer=hf_layer)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None

        self.num_layers = len(self.layers)

    def build_fwd(self, hidden_states: torch.Tensor, kv_cache: PagedKVCache):

        batch_size, seq_len, hidden_size = hidden_states.shape
        assert seq_len == 1, "currently only support decode"
        assert batch_size == self.batch_size
        for idx in range(self.num_layers):
            hidden_states = self.layers[idx].build_fwd(
                hidden_states=hidden_states,
                cos_cache=self.cos_cache,
                sin_cache=self.sin_cache,
                kv_cache=kv_cache,
            )

        rms_norm_out = torch.empty_like(hidden_states)
        self._builder.make_rms_norm(hidden_states, self.norm_weight, rms_norm_out, self.norm_variance_epsilon)
        if self.build_lm_head:
            logits = torch.empty((batch_size, seq_len, self.lm_head.shape[0]), dtype=rms_norm_out.dtype,
                                 device=rms_norm_out.device)
            self._builder.make_linear(rms_norm_out.reshape(-1, hidden_size), self.lm_head,
                                      logits.reshape(-1, self.lm_head.shape[0]))
            return logits
        else:
            return rms_norm_out

    def mega_forwrad(self, input_ids: torch.LongTensor):
        batch_size, seq_len = input_ids.size()
        hidden_states = torch.nn.functional.embedding(input_ids, self.embed_tokens)
        self.hidden_state_buffer.copy_(hidden_states)
        # inplace write to mega out tensor
        self._builder.run()
        if self.build_lm_head:
            return self.mega_out
        else:
            logits = torch.nn.functional.linear(self.mega_out, self.lm_head).float()
            return logits
