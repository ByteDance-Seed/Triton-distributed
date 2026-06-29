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

from typing import Dict, Optional

import torch
import torch.distributed as dist

from flash_comm.buffer import SymmetricTensor
from flash_comm.ep import EPConfig

from ._ops._base import GEMM_CLUSTER_TILE_M

# Sentinel the layout C kernel writes into ``recv_token_count_cpu``
# before publishing the real count; kernel re-arms it on every call.
_PENDING_RECV_COUNT_SENTINEL = -1


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


class EPOverlapContext:
    """CuTeDSL EP-overlap buffer container; construct via :meth:`create`."""

    def __init__(self, config: EPConfig, group: dist.ProcessGroup, *, num_worst_tokens: int = -1,
                 capacity_coeff: float = 1.2, check_num_worst_tokens: bool = False, alloc_alignment: int = 1024,
                 expert_alignment: int = 1):
        if config.nnodes != 1:
            raise NotImplementedError("EPOverlapContext currently only supports intranode (nnodes==1)")
        if expert_alignment < 1:
            raise ValueError(f"expert_alignment must be >= 1, got {expert_alignment}")
        # GEMM invariant: padded expert M must be a multiple of the
        # M-contig group-GEMM per-cluster tile, else the scheduler would
        # issue a partial tile spanning the next expert.
        if expert_alignment > 1 and expert_alignment % GEMM_CLUSTER_TILE_M != 0:
            raise ValueError(f"expert_alignment={expert_alignment} must be a multiple of "
                             f"GEMM_CLUSTER_TILE_M={GEMM_CLUSTER_TILE_M} (the M-contig "
                             f"group-GEMM per-cluster tile)")
        self.config = config
        self.group = group
        self.num_worst_tokens = int(num_worst_tokens)
        self.capacity_coeff = float(capacity_coeff)
        self.check_num_worst_tokens = bool(check_num_worst_tokens)
        self.alloc_alignment = int(alloc_alignment)
        self.expert_alignment = int(expert_alignment)

        # Symmetric eager buffers (peers / layout kernel reach into them).
        self.nvl_barrier_buf: torch.Tensor = None
        self.nvl_barrier_buf_ptrs: torch.Tensor = None
        self.full_splits_buf_ptrs: torch.Tensor = None
        self.token_src_rank_topk_and_indices_buf: torch.Tensor = None
        self.token_src_rank_topk_and_indices_buf_ptrs: torch.Tensor = None
        # Worst-case recv token count; used as the dummy compile-time M
        # by ops that ``mark_dynamic`` the M dimension.
        self.max_recv_tokens: int = 0

        # Local eager buffers (read/written only by THIS rank).
        self.recv_token_count_cpu: torch.Tensor = None
        self.recv_token_count: torch.Tensor = None
        # ``expert_signal_state`` is (2, experts_per_rank) int32:
        # row 0 = producer-ready flags, row 1 = CTA-arrival counters.
        # Single backing tensor so :meth:`reset_expert_signals` issues
        # one zero kernel; the row views alias and stay contiguous.
        self.expert_signal_state: torch.Tensor = None
        self.expert_signals: torch.Tensor = None
        self.expert_signal_counters: torch.Tensor = None

        # Lazy CuTeDSL staging buffers.
        self.dispatch_input_symm_tensor: Optional[SymmetricTensor] = None
        self.dispatch_input_buf: Optional[torch.Tensor] = None
        self.dispatch_input_ptrs: Optional[torch.Tensor] = None

        # Optional symmetric weight staging used by the dispatch
        # ``has_weight`` side channel. Lazy-allocated on first use:
        # peers pull (max_m, topk) FP32 weights and the kernel
        # side-writes per-row scalars into the caller-provided local
        # output tensor.
        self.dispatch_input_weight_symm_tensor: Optional[SymmetricTensor] = None
        self.dispatch_input_weight_buf: Optional[torch.Tensor] = None
        self.dispatch_input_weight_ptrs: Optional[torch.Tensor] = None

        self.combine_output_symm_tensor: Optional[SymmetricTensor] = None
        self.combine_output_buf: Optional[torch.Tensor] = None
        self.combine_output_ptrs: Optional[torch.Tensor] = None

        self.group_gemm_combine_output_symm_tensor: Optional[SymmetricTensor] = None
        self.group_gemm_combine_output_buf: Optional[torch.Tensor] = None
        self.group_gemm_combine_output_ptrs: Optional[torch.Tensor] = None
        self.group_gemm_combine_output_n_out: Optional[int] = None

        # Optional symmetric output-weight staging used by the
        # group_gemm_combine ``has_weight`` side channel. Lazy-allocated
        # on first use: peers push a 4 B FP32 weight per dispatched row
        # back to the source rank's ``(max_m, topk)`` symmetric slot.
        self.group_gemm_combine_output_weight_symm_tensor: Optional[SymmetricTensor] = None
        self.group_gemm_combine_output_weight_buf: Optional[torch.Tensor] = None
        self.group_gemm_combine_output_weight_ptrs: Optional[torch.Tensor] = None

        # Strong refs so SymmetricTensor GPU allocations survive GC.
        self._symm_tensors: Dict[str, SymmetricTensor] = {}

    @classmethod
    def create(cls, *, max_m: int, hidden: int, topk: int, num_experts: int, group: dist.ProcessGroup,
               local_world_size: int, capacity_coeff: float = 1.2, num_worst_tokens: int = -1,
               check_num_worst_tokens: bool = False, expert_alignment: int = 1) -> "EPOverlapContext":
        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        config = EPConfig(max_m=max_m, hidden=hidden, topk=topk, num_experts=num_experts, rank=rank,
                          world_size=world_size, local_world_size=local_world_size)
        if config.num_experts % config.world_size != 0:
            raise ValueError(f"num_experts ({config.num_experts}) must be divisible by "
                             f"world_size ({config.world_size})")

        ctx = cls(config=config, group=group, num_worst_tokens=num_worst_tokens, capacity_coeff=capacity_coeff,
                  check_num_worst_tokens=check_num_worst_tokens, expert_alignment=expert_alignment)
        ctx._init_eager_buffers()
        return ctx

    def _init_eager_buffers(self) -> None:
        cfg = self.config
        group = self.group

        # nvl_barrier must start zeroed group-wide; stale non-zero
        # would deadlock the next barrier kernel.
        nvl_barrier_symm = SymmetricTensor(
            shape=(cfg.local_world_size, ),
            dtype=torch.int32,
            group=group,
        )
        nvl_barrier_symm.get_local_tensor().fill_(0)
        self.nvl_barrier_buf = nvl_barrier_symm.get_local_tensor()
        self.nvl_barrier_buf_ptrs = nvl_barrier_symm.ptrs
        self._symm_tensors["nvl_barrier"] = nvl_barrier_symm

        full_splits_symm = SymmetricTensor(
            shape=(cfg.world_size, cfg.num_experts + 1),
            dtype=cfg.offset_dtype,
            group=group,
        )
        self.full_splits_buf_ptrs = full_splits_symm.ptrs
        self._symm_tensors["full_splits"] = full_splits_symm

        # Pinned page; layout kernel writes -1 -> real count via two
        # async copies, so the poll loop sees the latest value without
        # a CPU-side reset.
        self.recv_token_count_cpu = torch.empty(
            (cfg.world_size, ),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.recv_token_count_cpu.fill_(_PENDING_RECV_COUNT_SENTINEL)
        self.recv_token_count = torch.empty(
            (cfg.world_size, ),
            dtype=torch.int32,
            device="cuda",
        )

        if self.num_worst_tokens > 0:
            dispatch_recv_tokens = self.num_worst_tokens
        else:
            dispatch_recv_tokens = _round_up(
                int(cfg.max_m * cfg.topk * self.capacity_coeff),
                self.alloc_alignment,
            )
        self._alloc_token_src_meta(dispatch_recv_tokens)

        # Per-expert producer/consumer signals for fused dispatch+gemm.
        # Single (2, experts_per_rank) int32 backing tensor; the two
        # row views alias storage and are zeroed in one kernel by
        # :meth:`reset_expert_signals` before each fused call.
        experts_per_rank = cfg.num_experts // cfg.world_size
        self.expert_signal_state = torch.zeros(
            (2, experts_per_rank),
            dtype=torch.int32,
            device="cuda",
        )
        self.expert_signals = self.expert_signal_state[0]
        self.expert_signal_counters = self.expert_signal_state[1]

    def _alloc_token_src_meta(self, num_alloc_tokens: int) -> None:
        """Allocate the symmetric receiver-side dispatch metadata.

        Symmetric because peers write into this rank's slots from the
        layout C kernel. Sized once to the worst-case receive count.
        """
        token_src_symm = SymmetricTensor(
            shape=(num_alloc_tokens, ),
            dtype=torch.int64,
            group=self.group,
        )
        # -1 sentinel: padded receive-slots (expert_alignment > 1) read
        # as "no source" so CuTeDSL skips them via the src_rank check.
        token_src_symm.get_local_tensor().fill_(-1)
        self.token_src_rank_topk_and_indices_buf = (token_src_symm.get_local_tensor())
        self.token_src_rank_topk_and_indices_buf_ptrs = token_src_symm.ptrs
        self._symm_tensors["token_src_rank_topk_and_indices"] = token_src_symm
        self.max_recv_tokens = int(num_alloc_tokens)

    def ensure_dispatch_input(self) -> None:
        """Lazy alloc of the symmetric dispatch staging buffer.

        First call host-barriers so peers see the new ptrs; subsequent
        calls are no-ops.
        """
        if self.dispatch_input_symm_tensor is not None:
            return
        cfg = self.config
        symm = SymmetricTensor(
            shape=(cfg.max_m, cfg.hidden),
            dtype=cfg.token_dtype,
            group=self.group,
        )
        self.dispatch_input_symm_tensor = symm
        self.dispatch_input_buf = symm.get_local_tensor()
        self.dispatch_input_ptrs = symm.ptrs
        self._symm_tensors["dispatch_input"] = symm
        dist.barrier(group=self.group)

    def ensure_dispatch_input_weight(self) -> None:
        """Lazy alloc of the symmetric topk-weight staging buffer.

        Same alloc-time barrier discipline as
        :meth:`ensure_dispatch_input`. Buffer holds the full
        ``(max_m, topk)`` FP32 weight matrix so peers can pull a
        4 B scalar per ``(token, topk_idx)`` row.
        """
        if self.dispatch_input_weight_symm_tensor is not None:
            return
        cfg = self.config
        symm = SymmetricTensor(
            shape=(cfg.max_m, cfg.topk),
            dtype=cfg.weight_dtype,
            group=self.group,
        )
        self.dispatch_input_weight_symm_tensor = symm
        self.dispatch_input_weight_buf = symm.get_local_tensor()
        self.dispatch_input_weight_ptrs = symm.ptrs
        self._symm_tensors["dispatch_input_weight"] = symm
        dist.barrier(group=self.group)

    def ensure_combine_output(self) -> None:
        """Lazy alloc of the symmetric combine staging buffer.

        Same alloc-time barrier discipline as :meth:`ensure_dispatch_input`.
        """
        if self.combine_output_symm_tensor is not None:
            return
        cfg = self.config
        symm = SymmetricTensor(
            shape=(cfg.max_m * cfg.topk, cfg.hidden),
            dtype=cfg.token_dtype,
            group=self.group,
        )
        self.combine_output_symm_tensor = symm
        self.combine_output_buf = symm.get_local_tensor()
        self.combine_output_ptrs = symm.ptrs
        self._symm_tensors["combine_output"] = symm
        dist.barrier(group=self.group)

    def ensure_group_gemm_combine_output_weight(self) -> None:
        """Lazy alloc of the symmetric output-weight buffer.

        Same alloc-time barrier discipline as
        :meth:`ensure_dispatch_input`. Holds the per-rank
        ``(max_m, topk)`` FP32 weight matrix; the fused
        group_gemm_combine kernel push-writes one 4 B FP32 scalar per
        dispatched row to its source rank's ``(src_token_idx,
        src_topk_idx)`` slot (one write per token, gated to the
        leftmost N-tile so the multi-N-tile epilogue does not re-emit
        the same scalar).
        """
        if self.group_gemm_combine_output_weight_symm_tensor is not None:
            return
        cfg = self.config
        symm = SymmetricTensor(
            shape=(cfg.max_m, cfg.topk),
            dtype=cfg.weight_dtype,
            group=self.group,
        )
        self.group_gemm_combine_output_weight_symm_tensor = symm
        self.group_gemm_combine_output_weight_buf = symm.get_local_tensor()
        self.group_gemm_combine_output_weight_ptrs = symm.ptrs
        self._symm_tensors["group_gemm_combine_output_weight"] = symm
        dist.barrier(group=self.group)

    def ensure_group_gemm_combine_output(self, n_out: int) -> None:
        """Lazy alloc of the fused group-GEMM + combine staging buffer.

        Re-allocates on ``n_out`` change (each alloc triggers a
        cross-rank sync via SymmetricTensor).
        """
        if (self.group_gemm_combine_output_symm_tensor is not None and self.group_gemm_combine_output_n_out == n_out):
            return
        cfg = self.config
        symm = SymmetricTensor(
            shape=(cfg.max_m * cfg.topk, n_out),
            dtype=cfg.token_dtype,
            group=self.group,
        )
        self.group_gemm_combine_output_symm_tensor = symm
        self.group_gemm_combine_output_buf = symm.get_local_tensor()
        self.group_gemm_combine_output_ptrs = symm.ptrs
        self.group_gemm_combine_output_n_out = n_out
        self._symm_tensors["group_gemm_combine_output"] = symm
        dist.barrier(group=self.group)

    def reset_expert_signals(self) -> None:
        """Single-kernel zero of both signal rows (ready flags + CTA counters)."""
        self.expert_signal_state.zero_()
