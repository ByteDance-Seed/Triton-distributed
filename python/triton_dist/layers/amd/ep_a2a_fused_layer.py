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
AMD (HIP/ROCm) fused EP All-to-All + grouped GEMM MoE layer.

``EpAll2AllFusedOp`` exposes the fused MoE as an ``nn.Module`` with a
``preprocess`` -> ``mega_dispatch_group_gemm`` -> ``mega_group_gemm_combine`` flow
(plus ``dispatch_postprocess`` / ``combine_preprocess`` / ``ep_barrier_all`` /
``sync`` / ``finalize`` and an ``EPAllToAllLayoutDesc`` descriptor), built on the
AMD mega kernels in ``kernels/amd/ep_all2all_fused.py``.

  * combine uses serial / gather mode;
  * symmetric buffers are allocated eagerly via ``EpA2AFusedContext`` with a fixed
    ``capacity`` (the host raises ``RuntimeError`` if a rank would exceed it);
  * weights are this rank's local expert slices
    (``w1 = [experts_per_rank, hidden, inter]``, ``w2 = [experts_per_rank, inter, hidden]``);
    combine maps expert outputs back to ``hidden``.
"""

import dataclasses
from typing import Callable, Optional

import torch
import torch.distributed

from triton_dist.kernels.amd.ep_all2all_fused import (
    create_ep_a2a_fused_context,
    fused_dispatch_token_moe_grouped_gemm,
    fused_group_gemm_combine_token,
    _build_dispatch_metadata,
)


@dataclasses.dataclass
class EPAllToAllLayoutDesc:
    """AMD layout descriptor produced by ``EpAll2AllFusedOp.preprocess`` and
    consumed by ``mega_dispatch_group_gemm`` / ``mega_group_gemm_combine``.

    The dispatch/combine path only reads ``meta`` and ``topk_indices_tensor`` (plus
    ``token_dst_scatter_idx`` via ``meta`` after dispatch). The per-expert / per-rank
    count tensors below are populated for descriptor parity only -- see their notes.
    """
    num_dispatch_token_cur_rank: int  # this rank's input-token count (T)
    recv_buf_offset_per_expert: torch.Tensor  # int32 [ws, epr, ws]. Not in use; reserved for future path
    recv_buf_tokens_per_expert: torch.Tensor  # int32 [epr]. Not in use; reserved for future path
    num_recv_tokens_per_rank: torch.Tensor  # int32 [ws]. Not in use; reserved for future path
    num_input_tokens_per_rank: torch.Tensor  # int32 [ws]. Not in use; reserved for future path
    topk_indices_tensor: torch.Tensor  # int32 [num_tokens, topk]
    num_recv_tokens_cur_rank: int  # this rank's received-token count (M_local)
    meta: dict  # internal metadata dict consumed by the AMD kernels
    token_dst_scatter_idx: Optional[torch.Tensor] = None  # int32 [num_tokens, topk]; filled by dispatch


class EpAll2AllFusedOp(torch.nn.Module):
    """Fused EP All-to-All + grouped GEMM MoE op for AMD GPUs."""

    def __init__(
        self,
        ep_group: torch.distributed.ProcessGroup,
        max_tokens: int,
        hidden: int,
        topk: int,
        num_tot_experts: int,
        local_world_size: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
        weight_dtype: torch.dtype = torch.float32,  # Not in use; reserved for future path
        num_sm: int = 64,
        capacity: Optional[float] = None,
        FWD_GEMM_BLOCK_SIZE_N: int = 128,
        COMBINE_GEMM_BLOCK_SIZE_N: int = 128,
        GROUP_GEMM_BLOCK_SIZE_M: int = 128,
        gemm_BLOCK_SIZE_K: int = 64,
        gemm_GROUP_SIZE_M: int = 4,
        num_warps: int = 8,
        num_stages: int = 2,
        num_dispatch_tasks: int = 16,
    ):
        super().__init__()
        self.ep_group = ep_group
        self.rank = ep_group.rank()
        self.world_size = ep_group.size()
        self.local_world_size = self.world_size if local_world_size is None else local_world_size
        assert self.world_size == self.local_world_size, "EpAll2AllFusedOp requires world_size == local_world_size"
        assert num_tot_experts % self.world_size == 0, "num_tot_experts must be divisible by world_size"

        self.max_tokens = max_tokens
        self.hidden = hidden
        self.topk = topk
        self.num_tot_experts = num_tot_experts
        self.experts_per_rank = num_tot_experts // self.world_size
        self.dtype = dtype
        self.weight_dtype = weight_dtype  # Not in use; reserved for future path
        # `capacity` defaults to world_size (covers the worst-case all-to-one receive).
        self.capacity = float(self.world_size) if capacity is None else float(capacity)

        # tuning knobs (forwarded to the mega kernels)
        self.num_sm = num_sm
        self.num_dispatch_tasks = num_dispatch_tasks
        self.FWD_GEMM_BLOCK_SIZE_N = FWD_GEMM_BLOCK_SIZE_N
        self.COMBINE_GEMM_BLOCK_SIZE_N = COMBINE_GEMM_BLOCK_SIZE_N
        self.GROUP_GEMM_BLOCK_SIZE_M = GROUP_GEMM_BLOCK_SIZE_M
        self.gemm_BLOCK_SIZE_K = gemm_BLOCK_SIZE_K
        self.gemm_GROUP_SIZE_M = gemm_GROUP_SIZE_M
        self.num_warps = num_warps
        self.num_stages = num_stages

        # eager symmetric-buffer allocation.
        self.ctx = create_ep_a2a_fused_context(
            ep_group, max_tokens=max_tokens, hidden=hidden, topk=topk, num_tot_experts=num_tot_experts,
            rank=self.rank, world_size=self.world_size, dtype=dtype, weight_dtype=weight_dtype,
            capacity=self.capacity)

    # ===================== buffer-management API (parity) =====================
    def sync(self):
        """Not in use; reserved for future path (symmetric buffers are allocated eagerly in ``__init__``)."""

    def materialize(self):
        """Not in use; reserved for future path."""
        self.sync()

    def finalize(self):
        self.ctx.finalize()

    def ep_barrier_all(self):
        self.ctx.ep_barrier()

    # ============================= preprocess =================================
    def preprocess(self, exp_indices: torch.Tensor) -> EPAllToAllLayoutDesc:
        """Cross-rank split exchange + recv-offset + grouped-GEMM tiling (computed host-side in torch)."""
        assert exp_indices.dtype == torch.int32 and exp_indices.shape[1] == self.topk and exp_indices.is_contiguous()
        meta = _build_dispatch_metadata(self.ctx, exp_indices, self.GROUP_GEMM_BLOCK_SIZE_M)
        return EPAllToAllLayoutDesc(
            num_dispatch_token_cur_rank=exp_indices.shape[0],
            recv_buf_offset_per_expert=meta["recv_buf_offset"],
            recv_buf_tokens_per_expert=meta["split_size"],
            num_recv_tokens_per_rank=meta["num_recv_tokens_per_rank"],
            num_input_tokens_per_rank=meta["num_input_tokens_per_rank"],
            topk_indices_tensor=exp_indices,
            num_recv_tokens_cur_rank=meta["M_local"],
            meta=meta,
        )

    def dispatch_postprocess(self):
        """Not in use; reserved for future path (the forward path resets these counters/barriers in-kernel)."""
        self.ctx.task_counter.zero_()
        self.ctx.expert_counter.zero_()
        self.ctx.barriers_buf.zero_()

    def combine_preprocess(self, M_recv: Optional[int] = None):  # M_recv: Not in use; reserved for future path
        """Not in use; reserved for future path (the forward path resets the grid-sync counter in-kernel)."""
        self.ctx.combine_grid_sync.zero_()

    # ===================== dispatch + grouped GEMM (gemm1) ====================
    def mega_dispatch_group_gemm(
        self,
        input: torch.Tensor,  # [num_tokens, hidden]
        exp_indices: torch.Tensor,  # [num_tokens, topk] int32 (must be the tensor passed to preprocess)
        ep_a2a_layout_desc: EPAllToAllLayoutDesc,
        gemm_weight: torch.Tensor,  # [experts_per_rank, hidden, inter]
        gemm_output_data: Optional[torch.Tensor] = None,
        gemm_BLOCK_SIZE_N: Optional[int] = None,
    ):
        """Fused dispatch + grouped GEMM-1. Returns ``(gemm1_out[M, inter], desc)``."""
        # `meta` was built from the desc's indices; bind to that single source of truth.
        assert exp_indices is ep_a2a_layout_desc.topk_indices_tensor, \
            "exp_indices must be the tensor passed to preprocess (use ep_a2a_layout_desc.topk_indices_tensor)"
        gemm1_out, meta = fused_dispatch_token_moe_grouped_gemm(
            self.ctx, input, ep_a2a_layout_desc.topk_indices_tensor, gemm_weight, meta=ep_a2a_layout_desc.meta,
            num_sms=self.num_sm, num_dispatch_tasks=self.num_dispatch_tasks, BLOCK_SIZE_M=self.GROUP_GEMM_BLOCK_SIZE_M,
            BLOCK_SIZE_N=(gemm_BLOCK_SIZE_N or self.FWD_GEMM_BLOCK_SIZE_N), BLOCK_SIZE_K=self.gemm_BLOCK_SIZE_K,
            GROUP_SIZE_M=self.gemm_GROUP_SIZE_M, num_warps=self.num_warps, num_stages=self.num_stages,
            gemm_output=gemm_output_data)
        ep_a2a_layout_desc.token_dst_scatter_idx = meta.get("token_dst_scatter_idx")
        return gemm1_out, ep_a2a_layout_desc

    # ===================== grouped GEMM (gemm2) + combine =====================
    def mega_group_gemm_combine(
        self,
        gemm_input_data: torch.Tensor,  # [M, inter] (e.g. act(gemm1 out))
        gemm_weight: torch.Tensor,  # [experts_per_rank, inter, hidden]
        ep_a2a_layout_desc: EPAllToAllLayoutDesc,
        gate_input: Optional[torch.Tensor] = None,  # [num_tokens, topk] routing weights
        combine_output: Optional[torch.Tensor] = None,
        gemm_BLOCK_SIZE_N: Optional[int] = None,
        combine_mode: str = "serial",
    ) -> torch.Tensor:
        """Fused grouped GEMM-2 + combine (serial/gather). Returns ``[num_tokens, hidden]``."""
        assert combine_mode == "serial", "AMD combine only supports serial/gather mode"
        return fused_group_gemm_combine_token(
            self.ctx, gemm_input_data, gemm_weight, ep_a2a_layout_desc.meta,
            ep_a2a_layout_desc.topk_indices_tensor, topk_weights=gate_input, num_sms=self.num_sm,
            BLOCK_SIZE_M=self.GROUP_GEMM_BLOCK_SIZE_M,
            BLOCK_SIZE_N=(gemm_BLOCK_SIZE_N or self.COMBINE_GEMM_BLOCK_SIZE_N), BLOCK_SIZE_K=self.gemm_BLOCK_SIZE_K,
            GROUP_SIZE_M=self.gemm_GROUP_SIZE_M, num_warps=self.num_warps, num_stages=self.num_stages,
            combine_output=combine_output)

    # ============================ full forward ================================
    def forward(
        self,
        hidden_states: torch.Tensor,  # [num_tokens, hidden]
        topk_indices: torch.Tensor,  # [num_tokens, topk] int32, global expert ids
        w1: torch.Tensor,  # [experts_per_rank, hidden, inter]
        w2: torch.Tensor,  # [experts_per_rank, inter, hidden]
        topk_weights: Optional[torch.Tensor] = None,  # [num_tokens, topk]
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
    ) -> torch.Tensor:
        """Convenience full MoE forward: preprocess -> dispatch+gemm1 -> act -> gemm2+combine."""
        desc = self.preprocess(topk_indices)
        gemm1_out, desc = self.mega_dispatch_group_gemm(hidden_states, topk_indices, desc, w1)
        act = activation(gemm1_out).contiguous()
        return self.mega_group_gemm_combine(act, w2, desc, gate_input=topk_weights)


# AMD-naming alias (consistent with EPAll2AllLayer / EPLowLatencyAllToAllLayer).
EPAll2AllFusedLayer = EpAll2AllFusedOp
