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
"""
AMD (HIP/ROCm) model-facing fused EP MoE.

Runs the full fused MoE forward on top of an ``EpAll2AllFusedOp``:

    dispatch + grouped GEMM-1 (up)  ->  activation  ->  grouped GEMM-2 (down) + combine

  * expert weights are this rank's local slices (EP sharding):
    ``w1_local = [experts_per_rank, hidden, inter]``,
    ``w2_local = [experts_per_rank, inter, hidden]``;
  * the caller passes a pre-built ``EpAll2AllFusedOp`` (which owns the symmetric buffers).
"""

from typing import Callable, Optional

import torch

from triton_dist.layers.amd.ep_a2a_fused_layer import EpAll2AllFusedOp


def fused_ep_moe(
    layer: EpAll2AllFusedOp,
    hidden_states: torch.Tensor,  # [num_tokens, hidden]
    selected_experts: torch.Tensor,  # [num_tokens, topk] int32, global expert ids
    w1_local: torch.Tensor,  # [experts_per_rank, hidden, inter]
    w2_local: torch.Tensor,  # [experts_per_rank, inter, hidden]
    routing_weights: Optional[torch.Tensor] = None,  # [num_tokens, topk]
    activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
) -> torch.Tensor:
    """Functional (inference) entry: full fused MoE forward; returns ``[num_tokens, hidden]``."""
    return layer.forward(hidden_states, selected_experts, w1_local, w2_local, topk_weights=routing_weights,
                         activation=activation)


class TritonDistFusedEpMoeFunction(torch.autograd.Function):
    """``torch.autograd.Function`` wrapper around the fused MoE forward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: Optional[torch.Tensor],
        w1_local: torch.Tensor,
        w2_local: torch.Tensor,
        layer: EpAll2AllFusedOp,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
    ) -> torch.Tensor:
        ctx.set_materialize_grads(False)
        return layer.forward(hidden_states, selected_experts, w1_local, w2_local, topk_weights=routing_weights,
                             activation=activation)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("backward is not implemented")


def fused_ep_moe_autograd(
    layer: EpAll2AllFusedOp,
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    routing_weights: Optional[torch.Tensor] = None,
    activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
) -> torch.Tensor:
    """Same as ``fused_ep_moe`` but routed through the autograd ``Function``."""
    return TritonDistFusedEpMoeFunction.apply(hidden_states, selected_experts, routing_weights, w1_local, w2_local,
                                              layer, activation)
