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

from triton_dist.kernels.nvidia import allgather_group_gemm, ag_group_gemm, create_ag_group_gemm_context, moe_reduce_rs
from triton_dist.kernels.nvidia.moe_reduce_rs import create_moe_rs_context, run_moe_reduce_rs


def shard_local(tensor: torch.Tensor, world_size: int, dim: int, local_rank: int) -> torch.Tensor:
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


class TP_MoE:
    """
    Tensor Parallel MoE.
    This MoE uses a common TP strategy:
    1. First linear layer (gate/up) weights are column-parallel.
    2. Second linear layer (down) weights are row-parallel.
    """

    def __init__(self, rank=0, world_size=8, group=None, autotune=False):
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.act_fn = None
        self.gate_up_proj = None  # [experts, hidden_size, MLP_size * 2 // world_size]
        self.down_proj = None  # [experts, MLP_size // world_size, hidden_size]
        self.ag_ctx = None
        self.rs_ctx = None

        self.top_k = None
        self.num_experts = None
        self.gate = None

        if autotune:
            # auto-tune
            import triton
            from triton_dist.autotuner import contextual_autotune
            configs = [
                triton.Config({"BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK}, num_stages=s, num_warps=w)
                for BN in [128, 256]
                for BK in [32, 64]
                for s in [3, 4]
                for w in [4, 8]
            ]
            allgather_group_gemm.kernel_consumer_m_parallel_scatter_group_gemm = triton.autotune(
                configs=configs, key=["M", "N",
                                      "K"])(allgather_group_gemm.kernel_consumer_m_parallel_scatter_group_gemm)
            self.ag_group_gemm = contextual_autotune(is_dist=True)(ag_group_gemm)
            moe_reduce_rs.moe_gather_rs_grouped_gemm_kernel = triton.autotune(configs=configs, key=["M", "N", "K"])(
                moe_reduce_rs.moe_gather_rs_grouped_gemm_kernel)
            self.run_moe_reduce_rs = contextual_autotune(is_dist=True)(run_moe_reduce_rs)
        else:
            self.ag_group_gemm = ag_group_gemm
            self.run_moe_reduce_rs = run_moe_reduce_rs

    def _init_parameters(self, mlp: nn.Module, verbose=False):
        """
        Initializes and shards MoE parameters for Tensor Parallelism.
        moe: A standard nn.Module MoE (e.g., from HuggingFace Transformers).
             Expected to have mlp.gate, mlp.experts[0].gate_proj, mlp.experts[0].up_proj, mlp.experts[0].down_proj.
        """
        self.num_experts = mlp.num_experts
        self.top_k = mlp.top_k
        self.gate = mlp.gate.weight.detach().to("cuda")  # [num_experts, hidden_size]
        hidden_size = self.gate.shape[1]
        self.hidden_size = hidden_size
        MLP_size = mlp.experts[0].gate_proj.weight.detach().shape[0]  # [MLP_size, hidden_size]
        dtype = mlp.experts[0].gate_proj.weight.dtype
        self.gate_up_proj = torch.zeros(self.num_experts, hidden_size, MLP_size * 2 // self.world_size, dtype=dtype,
                                        device="cuda")
        self.down_proj = torch.zeros(self.num_experts, MLP_size // self.world_size, hidden_size, dtype=dtype,
                                     device="cuda")

        for e in range(self.num_experts):
            gate_proj = shard_local(mlp.experts[e].gate_proj.weight.detach(), self.world_size, 0,
                                    self.rank)  # [MLP_size // world_size, hidden_size]
            up_proj = shard_local(mlp.experts[e].up_proj.weight.detach(), self.world_size, 0,
                                  self.rank)  # [MLP_size // world_size, hidden_size]
            self.gate_up_proj[e] = torch.cat(
                (gate_proj, up_proj), dim=0).t().to("cuda",
                                                    non_blocking=True)  # [MLP_size * 2 // world_size, hidden_size]
            self.down_proj[e] = shard_local(mlp.experts[e].down_proj.weight.detach(), self.world_size, 1,
                                            self.rank).t().to(
                                                "cuda", non_blocking=True)  # [MLP_size // world_size, hidden_size]

        self.act_fn = mlp.experts[0].act_fn
        self.dtype = self.gate_up_proj.dtype

        assert mlp.experts[0].gate_proj.bias is None, "We do not support bias for now."

        if verbose:
            print(
                f"[RANK {self.rank}] MoE initialized with parameters: gate_up_proj shape: {self.gate_up_proj.shape}, down_proj shape: {self.down_proj.shape}"
            )

    def _init_ctx(self, M):
        """Initializes contexts for triton_dist AllGather-GEMM and GEMM-ReduceScatter operations."""
        self.ag_ctx = create_ag_group_gemm_context(
            M,
            self.gate_up_proj.shape[2],
            self.gate_up_proj.shape[1],
            self.num_experts,
            self.top_k,
            self.dtype,
            self.rank,
            self.world_size,
            self.world_size,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8,
            stages=4,
            num_warps=8,
        )

        self.rs_ctx = create_moe_rs_context(
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=self.world_size,
            max_token_num=M * self.top_k,
            hidden_dim=self.hidden_size,
            num_experts=self.num_experts,
            topk=self.top_k,
            input_dtype=self.dtype,
        )
        torch.cuda.synchronize()
        if self.world_size > 1:
            torch.distributed.barrier(group=self.group)

    def finalize(self):
        if self.rs_ctx:
            self.rs_ctx.finalize()

    @torch.inference_mode()
    def torch_fwd(self, x):
        '''
        Reference PyTorch forward pass using Tensor Parallelism.
        modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
        Final output is AllReduced.
        x: input tensor, shape [batch_size, seq_len, hidden_size]
        '''

        assert len(x.size()) == 3
        bsz, seq, hidden_dim = x.size()
        x = x.view(-1, hidden_dim)

        router_logits = torch.nn.functional.linear(x, self.gate)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        out = torch.zeros((bsz * seq, hidden_dim), dtype=x.dtype, device=x.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = x[None, top_x].reshape(-1, hidden_dim)
            cur_out_fused = current_state @ self.gate_up_proj[expert_idx].squeeze(0)
            wg, w1 = torch.chunk(cur_out_fused, 2, dim=-1)
            cur_out = self.act_fn(wg) * w1
            cur_out = cur_out @ self.down_proj[expert_idx].squeeze(0)
            cur_out = cur_out * routing_weights[top_x, idx, None]
            out.index_add_(0, top_x, cur_out)

        if self.world_size > 1:
            torch.distributed.all_reduce(out, torch.distributed.ReduceOp.SUM, group=self.group)
        return out.view(bsz, seq, hidden_dim)

    @torch.inference_mode()
    def torch_fwd_no_loop(self, x):
        '''
        Reference PyTorch forward pass using Tensor Parallelism.
        This version does not use a loop over experts. But it requires extra FLOPs.
        Final output is AllReduced.
        x: input tensor, shape [batch_size, seq_len, hidden_size]
        '''

        assert len(x.size()) == 3
        bsz, seq, hidden_dim = x.size()
        x = x.view(-1, hidden_dim)

        router_logits = torch.nn.functional.linear(x, self.gate)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        topk_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        normalized_topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        full_routing_weights = torch.zeros_like(routing_weights)
        full_routing_weights.scatter_(1, selected_experts, normalized_topk_weights)
        routing_weights = full_routing_weights.to(x.dtype)

        up_gate = torch.einsum("mn,enk->mek", x, self.gate_up_proj)
        wg, w1 = torch.chunk(up_gate, 2, dim=-1)
        out = self.act_fn(wg) * w1
        out = torch.einsum("mek,ekn->emn", out, self.down_proj).transpose(0, 1)

        out = (out * routing_weights.unsqueeze(-1)).sum(dim=1)
        if self.world_size > 1:
            torch.distributed.all_reduce(out, torch.distributed.ReduceOp.SUM, group=self.group)
        return out.view(bsz, seq, hidden_dim)

    @torch.inference_mode()
    def dist_triton_fwd(self, x: torch.Tensor):
        """
        triton_dist forward pass for TP.
        This version uses ag_gemm and gemm_rs.
        x: input tensor, shape [batch_size, seq_len, hidden_size]
        """
        assert len(x.size()) == 3
        bsz, seq, hidden_dim = x.size()
        x = x.view(-1, hidden_dim)

        router_logits = torch.nn.functional.linear(x, self.gate)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        local_topk_weight, local_topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        local_topk_weight /= local_topk_weight.sum(dim=-1, keepdim=True)

        full_topk_weight = torch.zeros(bsz * seq * self.world_size, self.top_k, dtype=local_topk_weight.dtype,
                                       device="cuda")
        full_topk_ids = torch.zeros(bsz * seq * self.world_size, self.top_k, dtype=torch.int32, device="cuda")
        torch.distributed.all_gather_into_tensor(full_topk_weight, local_topk_weight, group=self.group)
        torch.distributed.all_gather_into_tensor(full_topk_ids, local_topk_ids.to(torch.int32), group=self.group)

        # ag moe
        out_fused = self.ag_group_gemm(x.contiguous(), self.gate_up_proj.contiguous(), ctx=self.ag_ctx,
                                       full_topk_ids=full_topk_ids.contiguous())

        wg, w1 = torch.chunk(out_fused, 2, dim=-1)
        out = self.act_fn(wg) * w1

        # moe rs
        out = self.run_moe_reduce_rs(
            out.contiguous(),
            self.down_proj.contiguous(),
            full_topk_ids.contiguous(),
            full_topk_weight.contiguous(),
            ctx=self.rs_ctx,
            n_chunks=4,
        )

        return out.view(bsz, seq, hidden_dim)

    @torch.inference_mode()
    def fwd(self, x: torch.Tensor):
        raise NotImplementedError("Please use torch_fwd or dist_triton_fwd instead.")
