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
import torch.nn.functional as F
import gc

from transformers import Qwen3ForCausalLM, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from triton_dist.kernels.allreduce import AllReduceMethod
from triton_dist.models.kv_cache import KV_Cache

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")
try:
    if torch.version.cuda:
        from triton_dist.layers.nvidia.tp_mlp import TP_MLP
        from triton_dist.layers.nvidia.tp_attn import TP_Attn, layer_norm, _set_cos_sin_cache
        PLATFORM = 'nvidia'
    elif torch.version.hip:
        from triton_dist.layers.amd.tp_mlp import TP_MLP
        from triton_dist.layers.amd.tp_attn import TP_Attn, layer_norm, _set_cos_sin_cache
        PLATFORM = 'amd'
except ImportError as e:
    raise ImportError(
        "Required Triton Dist layers not found. Please ensure you have the correct Triton Dist package installed."
    ) from e


class Qwen3Layer:
    """
    A single layer of Qwen3 model, containing self-attention and MLP.
    This layer is designed to be used in a tensor parallel setting.
    It initializes the parameters and sets the forward pass method based on the mode.
    """

    def __init__(self, layer_idx, group) -> None:

        self.attn: TP_Attn = None
        self.mlp: TP_MLP = None
        self.input_norm_eps = None
        self.input_norm_w = None
        self.post_norm_eps = None
        self.post_norm_w = None

        self.layer_idx = layer_idx
        self.group = group

    def init_parameters(self, hf_layer: Qwen3DecoderLayer, rank: int, world_size: int):
        self.mlp = TP_MLP(rank=rank, world_size=world_size, group=self.group)
        self.mlp._init_parameters(hf_layer.mlp)

        self.attn = TP_Attn(rank=rank, world_size=world_size, group=self.group)
        self.attn._init_parameters(hf_layer.self_attn)

        self.input_norm_eps = hf_layer.input_layernorm.variance_epsilon
        self.input_norm_w = hf_layer.input_layernorm.weight.detach().cuda()
        self.post_norm_eps = hf_layer.post_attention_layernorm.variance_epsilon
        self.post_norm_w = hf_layer.post_attention_layernorm.weight.detach().cuda()

    def set_fwd(self, mode: str = 'torch'):
        if mode == 'triton_dist':
            self.attn.fwd = self.attn.dist_triton_fwd
            self.mlp.fwd = self.mlp.dist_triton_fwd
        elif mode == 'torch':
            self.attn.fwd = self.attn.torch_fwd
            self.mlp.fwd = self.mlp.torch_fwd
        elif mode == 'triton_dist_AR':
            self.attn.fwd = self.attn.dist_triton_AR_fwd
            self.mlp.fwd = self.mlp.dist_triton_AR_fwd
        else:
            raise ValueError(f"Unsupported mode: {mode}, choose from ['dist_triton', 'torch']")

    @torch.inference_mode()
    def fwd(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, cos_sin_cache: torch.Tensor,
            kv_cache: KV_Cache):
        residual = hidden_states
        # self-attention
        hidden_states = layer_norm(hidden_states, self.input_norm_eps, self.input_norm_w)
        hidden_states = self.attn.fwd(hidden_states, position_ids, cos_sin_cache, kv_cache, self.layer_idx)
        hidden_states = residual + hidden_states

        residual = hidden_states
        # mlp
        hidden_states = layer_norm(hidden_states, self.post_norm_eps, self.post_norm_w)
        hidden_states = self.mlp.fwd(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3:
    """
    Qwen3 model implementation for tensor parallel training.
    This model initializes the parameters, sets the forward pass method, and provides an inference method.
    It supports both torch and triton_dist modes for forward pass.
    """

    def __init__(self, model_config, group) -> None:
        self.dtype = model_config.dtype
        self.config = Qwen3Config.from_pretrained(model_config.model_name, local_files_only=model_config.local_only)
        self.model_name = model_config.model_name
        self.max_length = model_config.max_length
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta

        self.rank = model_config.rank
        self.world_size = model_config.world_size
        self.group = group

        self.init_parameters()
        self.set_fwd()
        self.use_ar = False
        self.model_type = 'dense'

    def set_fwd(self, mode: str = 'torch'):
        for layer in self.layers:
            layer.set_fwd(mode)

    def init_parameters(self):
        hf_model = Qwen3ForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().cuda()
        self.lm_head = hf_model.lm_head.weight.detach().cuda()
        self.norm_weight = hf_model.model.norm.weight.detach().cuda()
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        self.cos_sin_cache = _set_cos_sin_cache(hf_model.model.rotary_emb.inv_freq.cuda(), max_length=self.max_length)

        self.layers: list[Qwen3Layer] = []

        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = Qwen3Layer(idx, self.group)
            layer.init_parameters(hf_layer=hf_layer, rank=self.rank, world_size=self.world_size)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    def init_triton_dist_ctx(self, max_M: int = 4096):
        # init ctx
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 128
        stages = 3
        if PLATFORM == 'nvidia':
            self.ag_intranode_stream = torch.cuda.Stream(priority=-1)
        elif PLATFORM == 'amd':
            self.ag_intranode_stream = [torch.cuda.Stream(priority=-1) for i in range(self.world_size)]
        else:
            raise RuntimeError(f"Unsupported platform: {PLATFORM}. Supported platforms are 'nvidia' and 'amd'.")
        self.ag_internode_stream = torch.cuda.Stream()
        self.layers[0].attn._init_ctx(max_M=max_M, ag_intranode_stream=self.ag_intranode_stream,
                                      ag_internode_stream=self.ag_internode_stream, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                      BLOCK_K=BLOCK_K, stages=stages)
        self.layers[0].mlp._init_ctx(max_M=max_M, ag_intranode_stream=self.ag_intranode_stream,
                                     ag_internode_stream=self.ag_internode_stream, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                     BLOCK_K=BLOCK_K, stages=stages)
        for layer in self.layers[1:]:
            layer.attn.ag_ctx = self.layers[0].attn.ag_ctx
            layer.attn.rs_ctx = self.layers[0].attn.rs_ctx
            layer.mlp.ag_ctx = self.layers[0].mlp.ag_ctx
            layer.mlp.rs_ctx = self.layers[0].mlp.rs_ctx

        self.use_ar = False

    def init_triton_dist_AR_ctx(self, max_M: int = 128, ar_method: AllReduceMethod = AllReduceMethod.DoubleTree):
        self.layers[0].attn._init_AR_ctx(max_M=max_M, method=ar_method, dtype=self.dtype)
        self.layers[0].mlp._init_AR_ctx(max_M=max_M, method=ar_method, dtype=self.dtype)

        for layer in self.layers[1:]:
            layer.attn.ar_ctx = self.layers[0].attn.ar_ctx
            layer.attn.ar_method = self.layers[0].attn.ar_method
            layer.mlp.ar_ctx = self.layers[0].mlp.ar_ctx
            layer.mlp.ar_method = self.layers[0].mlp.ar_method
        self.use_ar = True

    def finalize(self):
        self.layers[0].attn.finalize()
        self.layers[0].mlp.finalize()

    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, kv_cache: KV_Cache,
                  wo_lm_head=False):

        bsz, seq_len = input_ids.size()
        hidden_states = F.embedding(input_ids, self.embed_tokens)
        for idx in range(self.num_layers):
            hidden_states = self.layers[idx].fwd(
                hidden_states=hidden_states,
                position_ids=position_ids,
                cos_sin_cache=self.cos_sin_cache,
                kv_cache=kv_cache,
            )

        hidden_states = layer_norm(hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon)

        if seq_len > 1:  # prefill
            hidden_states = hidden_states[:, -1:]
        if wo_lm_head:  # for benchmark
            return hidden_states
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits
