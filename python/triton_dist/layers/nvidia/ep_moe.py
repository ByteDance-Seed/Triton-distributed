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
import torch.distributed as dist
import triton

from triton_dist.function.nvidia.common import init_triton_dist_ep_op, deinit_triton_dist_ep_op
from triton_dist.function.nvidia.ep_moe_fused import TritonDistFusedEpMoeFunction
from triton_dist.kernels.nvidia.group_gemm import (moe_grouped_gemm, build_block_row_idx_info_kernel,
                                                   GROUP_GEMM_BLOCK_SIZE_M)


def prepare_moe_metadata_using_kernel(expert_counts, num_experts):
    device = expert_counts.device
    num_sms = 32

    M = expert_counts.sum().item()
    M_grid = triton.cdiv(M, GROUP_GEMM_BLOCK_SIZE_M) + num_experts
    E_PAD = triton.next_power_of_2(num_experts)

    split_size_cum_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=device)
    expert_idx_to_tile_offset = torch.zeros(num_experts, dtype=torch.int32, device=device)

    block_row_idx_to_expert_idx = torch.zeros(M_grid, dtype=torch.int32, device=device)
    block_row_idx_to_row_offset = torch.zeros(M_grid, dtype=torch.int32, device=device)
    block_row_idx_to_tile_split = torch.zeros(M_grid, dtype=torch.int32, device=device)
    block_row_idx_to_tile_cumsum = torch.zeros(M_grid, dtype=torch.int32, device=device)

    num_tiles_total = torch.zeros(1, dtype=torch.int32, device=device)

    grid = (num_sms, )
    build_block_row_idx_info_kernel[grid](expert_counts, split_size_cum_per_expert, block_row_idx_to_expert_idx,
                                          block_row_idx_to_row_offset, block_row_idx_to_tile_split,
                                          block_row_idx_to_tile_cumsum, expert_idx_to_tile_offset, num_tiles_total,
                                          num_experts, E_PAD, GROUP_GEMM_BLOCK_SIZE_M, num_sms)

    return (split_size_cum_per_expert, block_row_idx_to_expert_idx, block_row_idx_to_row_offset,
            block_row_idx_to_tile_split, block_row_idx_to_tile_cumsum, num_tiles_total)


class EP_MoE:
    """
    EP MoE
    """

    def __init__(self, rank=0, world_size=8, group=None):
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.act_fn = None
        self.gate_up_proj = None  # [experts // world_size, hidden_size, MLP_size * 2]
        self.down_proj = None  # [experts // world_size, MLP_size, hidden_size]
        self.ep_op_ctx = None

        self.top_k = None
        self.num_experts = None
        self.gate = None

    def _init_parameters(self, mlp: nn.Module, verbose=False):
        self.num_experts = mlp.num_experts
        self.top_k = mlp.top_k
        self.gate = mlp.gate.weight.detach().to("cuda")  # [num_experts, hidden_size]
        hidden_size = self.gate.shape[1]
        self.hidden_size = hidden_size
        MLP_size = mlp.experts[0].gate_proj.weight.detach().shape[0]  # [MLP_size, hidden_size]
        dtype = mlp.experts[0].gate_proj.weight.dtype
        self.gate_up_proj = torch.zeros(self.num_experts // self.world_size, MLP_size * 2, hidden_size, dtype=dtype,
                                        device="cuda")
        self.down_proj = torch.zeros(self.num_experts // self.world_size, hidden_size, MLP_size, dtype=dtype,
                                     device="cuda")
        assert mlp.num_experts % self.world_size == 0, "num_experts must be divisible by world_size."

        start_expert_id = self.rank * self.num_experts // self.group.size()
        end_expert_id = (self.rank + 1) * self.num_experts // self.group.size()

        for e in range(self.num_experts):
            if not (start_expert_id <= e < end_expert_id):
                continue
            local_idx = e - start_expert_id

            gate_proj = mlp.experts[e].gate_proj.weight.detach()
            up_proj = mlp.experts[e].up_proj.weight.detach()
            self.gate_up_proj[local_idx] = torch.cat((gate_proj, up_proj), dim=0).to("cuda", non_blocking=True)
            self.down_proj[local_idx] = mlp.experts[e].down_proj.weight.detach().to("cuda", non_blocking=True)

        self.act_fn = mlp.experts[0].act_fn
        self.dtype = self.gate_up_proj.dtype

        assert mlp.experts[0].gate_proj.bias is None, "We do not support bias for now."

        if verbose:
            print(
                f"[RANK {self.rank}] MoE initialized with parameters: gate_up_proj shape: {self.gate_up_proj.shape}, down_proj shape: {self.down_proj.shape}"
            )

    def _init_ctx(self, EP_GROUP, max_tokens_per_rank):
        init_triton_dist_ep_op(
            EP_GROUP,
            max_tokens_per_rank,
            self.hidden_size,
            self.top_k,
            self.rank,
            self.num_experts,
            self.world_size,
            dtype=self.dtype,
            weight_dtype=torch.float32,
            num_sm=64,
            num_buffers=1,
            capacity=4.0,
        )
        torch.cuda.synchronize()
        if self.world_size > 1:
            torch.distributed.barrier(group=self.group)

    def finalize(self):
        deinit_triton_dist_ep_op()

    @torch.inference_mode()
    def torch_fwd(self, hidden_states: torch.Tensor):

        assert len(hidden_states.size()) == 3
        bsz, seq, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = torch.nn.functional.linear(hidden_states, self.gate)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        selected_experts = selected_experts.to(torch.int32)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        num_experts_per_rank = self.num_experts // self.group.size()
        batch_size, topk = selected_experts.shape
        hidden_dim = hidden_states.shape[1]

        # --- Dispatch ---
        hidden_states_rep = hidden_states.repeat_interleave(topk, dim=0)
        routing_weights_flat = routing_weights.flatten()
        selected_experts_flat = selected_experts.flatten()

        dest_ranks = selected_experts_flat // num_experts_per_rank
        sort_idxs = torch.argsort(dest_ranks, stable=True)

        tokens_to_send = hidden_states_rep[sort_idxs]
        weights_to_send = routing_weights_flat[sort_idxs]
        expert_ids_to_send = selected_experts_flat[sort_idxs]

        splits_send_tensor = torch.bincount(dest_ranks, minlength=self.group.size())
        splits_send = splits_send_tensor.tolist()
        splits_recv_tensor = torch.empty(self.group.size(), dtype=torch.long, device=hidden_states.device)
        dist.all_to_all_single(splits_recv_tensor, splits_send_tensor, group=self.group)
        splits_recv = splits_recv_tensor.tolist()
        total_recv = sum(splits_recv)

        tokens_recv = torch.empty((total_recv, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        weights_recv = torch.empty((total_recv, ), dtype=routing_weights.dtype, device=hidden_states.device)
        expert_ids_recv = torch.empty((total_recv, ), dtype=torch.int32, device=hidden_states.device)

        dist.all_to_all_single(tokens_recv, tokens_to_send, output_split_sizes=splits_recv,
                               input_split_sizes=splits_send, group=self.group)
        dist.all_to_all_single(weights_recv, weights_to_send, output_split_sizes=splits_recv,
                               input_split_sizes=splits_send, group=self.group)
        dist.all_to_all_single(expert_ids_recv, expert_ids_to_send, output_split_sizes=splits_recv,
                               input_split_sizes=splits_send, group=self.group)

        local_expert_ids = expert_ids_recv % num_experts_per_rank
        local_sort_idxs = torch.argsort(local_expert_ids, stable=True)

        tokens_sorted = tokens_recv[local_sort_idxs]
        weights_sorted = weights_recv[local_sort_idxs]
        expert_counts = torch.bincount(local_expert_ids, minlength=num_experts_per_rank).to(torch.int32)

        (split_size_cum_per_expert, meta_expert_ids, meta_split_cum, meta_tile_num, meta_tile_num_cum,
         num_tiles_total) = prepare_moe_metadata_using_kernel(expert_counts, num_experts_per_rank)

        # compute
        fc1_out = moe_grouped_gemm(tokens_sorted, self.gate_up_proj, meta_expert_ids, expert_counts, meta_split_cum,
                                   meta_tile_num, meta_tile_num_cum, num_tiles_total, input_reduce_last_dim=True,
                                   weight_reduce_last_dim=True)

        gate, val = fc1_out.chunk(2, dim=-1)
        gate_f = gate.float()
        val_f = val.float()
        weights_f = weights_sorted.float().unsqueeze(-1)
        swiglu_out_f = torch.nn.functional.silu(gate_f) * val_f
        swiglu_out_weighted = (swiglu_out_f * weights_f).to(hidden_states.dtype)

        fc2_out = moe_grouped_gemm(swiglu_out_weighted, self.down_proj, meta_expert_ids, expert_counts, meta_split_cum,
                                   meta_tile_num, meta_tile_num_cum, num_tiles_total, input_reduce_last_dim=True,
                                   weight_reduce_last_dim=True)

        # Combine
        inv_local_sort_idxs = torch.argsort(local_sort_idxs)
        fc2_out_unsorted = fc2_out[inv_local_sort_idxs]

        combined_out_flat = torch.empty((batch_size * topk, hidden_dim), dtype=hidden_states.dtype,
                                        device=hidden_states.device)
        dist.all_to_all_single(combined_out_flat, fc2_out_unsorted, output_split_sizes=splits_send,
                               input_split_sizes=splits_recv, group=self.group)

        inv_sort_idxs = torch.argsort(sort_idxs)
        final_out_ordered = combined_out_flat[inv_sort_idxs]

        output = final_out_ordered.view(batch_size, topk, hidden_dim).sum(dim=1)
        return output.view(bsz, seq, hidden_dim)

    @torch.inference_mode()
    def dist_triton_fwd(self, hidden_states: torch.Tensor):

        assert len(hidden_states.size()) == 3
        bsz, seq, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = torch.nn.functional.linear(hidden_states, self.gate)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        output = TritonDistFusedEpMoeFunction.apply(self.num_experts, routing_weights, selected_experts.to(torch.int32),
                                                    hidden_states, self.gate_up_proj, None, self.down_proj, self.group)
        return output.view(bsz, seq, hidden_dim)

    @torch.inference_mode()
    def fwd(self, hidden_states: torch.Tensor):
        raise NotImplementedError("Please use torch_fwd or dist_triton_fwd instead.")
