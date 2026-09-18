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

import os
import time
from typing import Optional, Tuple

import torch

import flash_comm._C.ep_intranode as _ep
import flash_comm._C.ep_internode as _ep_inter
from flash_comm.ep import EPCommLayoutDesc
from flash_comm.utils import get_global_cutedsl_kernel_cache

from ._ops import (
    CuTeDSLCombineInterOp,
    CuTeDSLCombinePushOp,
    CuTeDSLCombineTilePushOp,
    CuTeDSLDispatchGroupGemmInterOp,
    CuTeDSLDispatchGroupGemmOp,
    CuTeDSLDispatchInterOp,
    CuTeDSLDispatchOp,
    CuTeDSLGroupGemmCombineOp,
    CuTeDSLGroupGemmCombineInterOp,
    CuTeDSLGroupGemmOp,
    CuTeDSLTopkReduceOp,
)
from ._ops._base import GEMM_CLUSTER_TILE_M
from .context import _PENDING_RECV_COUNT_SENTINEL, EPOverlapContext
from .kernels.triton_weight_mask_copy import masked_weight_copy


class EPOverlapKernels:
    """CuTeDSL-only EP overlap operator surface.

    JIT artifacts live in :attr:`cutedsl_cache`. Use
    :meth:`clear_cutedsl_cache` to drop everything.
    """

    # Per-cluster M tile of the GEMM kernels (constant); exposed so
    # callers can compute padding strides without an instance.
    cluster_tile_m: int = GEMM_CLUSTER_TILE_M

    def __init__(self, max_m: int, hidden: int, topk: int, num_experts: int, local_world_size: int,
                 ep_group: torch.distributed.ProcessGroup, capacity: float = 1.2, num_worst_tokens: int = -1,
                 expert_alignment: int = 1, check_num_worst_tokens: bool = False):
        self.ep_group = ep_group
        self.num_worst_tokens = int(num_worst_tokens)
        self.check_num_worst_tokens = bool(check_num_worst_tokens)
        self.capacity_coeff = float(capacity)
        self._finalized = False
        cpu_poll_sleep_us = int(os.environ.get("FLASH_COMM_EP_CPU_POLL_SLEEP_US", "50"))
        if cpu_poll_sleep_us < 0:
            raise ValueError(f"FLASH_COMM_EP_CPU_POLL_SLEEP_US must be >= 0, got {cpu_poll_sleep_us}")
        self.cpu_poll_sleep_us = cpu_poll_sleep_us
        self.cpu_poll_sleep_s = cpu_poll_sleep_us / 1_000_000

        self.overlap_context = EPOverlapContext.create(
            max_m=max_m,
            hidden=hidden,
            topk=topk,
            num_experts=num_experts,
            group=ep_group,
            local_world_size=local_world_size,
            capacity_coeff=capacity,
            num_worst_tokens=num_worst_tokens,
            check_num_worst_tokens=check_num_worst_tokens,
            expert_alignment=expert_alignment,
        )
        # Device SM count is the natural default for GEMM scheduling
        # when the caller does not override ``gemm_num_sm`` per call.
        # Read once here so we don't query torch per launch.
        self._device_sm_count = int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
        # Make peers' symmetric allocations visible before any kernel launch.
        torch.distributed.barrier(group=ep_group)

        self.cutedsl_cache = get_global_cutedsl_kernel_cache()

        op_kwargs = {"rank": self.rank, "world_size": self.world_size}
        self._dispatch_op = CuTeDSLDispatchOp(**op_kwargs, expert_alignment=self.expert_alignment)
        self._dispatch_inter_op = CuTeDSLDispatchInterOp(
            **op_kwargs,
            local_world_size=self.overlap_context.config.local_world_size,
        )
        self._combine_push_op = CuTeDSLCombinePushOp(**op_kwargs)
        self._combine_tile_push_op = CuTeDSLCombineTilePushOp(**op_kwargs)
        self._combine_inter_op = CuTeDSLCombineInterOp(
            **op_kwargs,
            local_world_size=self.overlap_context.config.local_world_size,
        )
        self._topk_reduce_op = CuTeDSLTopkReduceOp(**op_kwargs)
        self._group_gemm_op = CuTeDSLGroupGemmOp(**op_kwargs)
        self._group_gemm_combine_op = CuTeDSLGroupGemmCombineOp(**op_kwargs)
        self._group_gemm_combine_inter_op = CuTeDSLGroupGemmCombineInterOp(
            **op_kwargs,
            local_world_size=self.overlap_context.config.local_world_size,
        )
        self._dispatch_group_gemm_op = CuTeDSLDispatchGroupGemmOp(**op_kwargs, expert_alignment=self.expert_alignment)
        self._dispatch_group_gemm_inter_op = CuTeDSLDispatchGroupGemmInterOp(
            **op_kwargs,
            expert_alignment=self.expert_alignment,
        )

    # ------------------------------------------------------------------
    # Properties and shared utilities.
    # ------------------------------------------------------------------

    @property
    def config(self):
        return self.overlap_context.config

    @property
    def rank(self) -> int:
        return self.overlap_context.config.rank

    @property
    def world_size(self) -> int:
        return self.overlap_context.config.world_size

    @property
    def expert_alignment(self) -> int:
        """Per-expert row alignment used by dispatch/group-GEMM ops."""
        return self.overlap_context.expert_alignment

    @property
    def device_sm_count(self) -> int:
        """Total SMs on this device (queried once via ``torch.cuda``).

        Default for ``gemm_num_sm`` when the caller does not override.
        """
        return self._device_sm_count

    # Minimum GEMM SM budget is the cluster size of the M-contig
    # GEMM family (cluster shape (2, 2) -> 4 SMs per cluster); see
    # ``_GEMM_CLUSTER_M`` / ``_GEMM_CTA_GROUP`` in ``_ops/_base.py``.
    _MIN_GEMM_NUM_SM: int = 4

    def _resolve_comm_num_sm(self, comm_num_sm: Optional[int]) -> int:
        """Validate and resolve a per-call non-GEMM SM budget.

        ``None`` means use the full device. Runtime-SM kernels take the
        value as an argument, so this does not affect CuTeDSL cache keys.
        """
        if comm_num_sm is None:
            return self._device_sm_count
        n = int(comm_num_sm)
        if n < 1:
            raise ValueError(f"comm_num_sm must be >= 1; got {n}")
        if n > self._device_sm_count:
            n = self._device_sm_count
        return n

    def _resolve_gemm_num_sm(self, gemm_num_sm: Optional[int]) -> int:
        """Validate and resolve the per-call GEMM SM budget.

        ``None`` -> full device SM count. Values are clamped at the
        cluster-size lower bound and the device SM upper bound so users
        can sweep without worrying about cache-key blow-up from
        out-of-range inputs producing distinct compiled kernels.
        """
        if gemm_num_sm is None:
            return self._device_sm_count
        n = int(gemm_num_sm)
        if n < self._MIN_GEMM_NUM_SM:
            raise ValueError(f"gemm_num_sm must be >= {self._MIN_GEMM_NUM_SM} (cluster "
                             f"size of the M-contig GEMM); got {n}")
        if n > self._device_sm_count:
            n = self._device_sm_count
        return n

    def clear_cutedsl_cache(self) -> None:
        self.cutedsl_cache.clear()

    def finalize(self) -> None:
        """Collectively retire kernels and release all EP-overlap resources.

        Every rank in ``ep_group`` must call this in the same order. Teardown is
        explicit rather than ``__del__`` because Python GC cannot safely run a
        cross-rank barrier or coordinate NCCL window/communicator destruction.
        """
        if self._finalized:
            return
        ctx = self.overlap_context
        if ctx.config.nnodes > 1:
            ctx.release_internode_nccl_resources()
            torch.cuda.synchronize()
            torch.distributed.barrier(group=self.ep_group)
            ctx.destroy_internode_nccl_gin()
            torch.cuda.synchronize()
            torch.distributed.barrier(group=self.ep_group)

        # Raw VMM/CUDA-IPC backends free immediately, so explicitly retire every
        # stream and rank before dropping peer mappings. This also makes lazy and
        # resized buffers obey the same lifecycle as eager allocations.
        torch.cuda.synchronize()
        torch.distributed.barrier(group=self.ep_group)
        ctx.free_buffers()
        torch.cuda.synchronize()
        torch.distributed.barrier(group=self.ep_group)
        self._finalized = True

    close = finalize

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()

    # ------------------------------------------------------------------
    # Low-level CUDA building blocks.
    # ------------------------------------------------------------------

    def _reset_dispatch_signals_barrier(self) -> None:
        rng = self.overlap_context._dispatch_signal_range
        if rng is None:
            raise RuntimeError("dispatch signal range is not initialized")
        _ep_inter.reset_signals_barrier_all_on_stream(*rng)

    def _reset_combine_signals_barrier(self) -> None:
        rng = self.overlap_context._combine_signal_range
        if rng is None:
            raise RuntimeError("combine signal range is not initialized")
        _ep_inter.reset_signals_barrier_all_on_stream(*rng)

    def ep_group_barrier(self) -> None:
        """Symmetric-memory cross-rank barrier (stream-side for intranode)."""
        cfg = self.overlap_context.config
        if cfg.nnodes == 1:
            _ep.barrier_all_on_stream(
                self.overlap_context.nvl_barrier_buf_ptrs,
                cfg.rank,
                cfg.world_size,
            )
        else:
            _ep_inter.barrier_all_on_stream()

    def compute_token_within_expert_offset_and_expert_counts(
        self,
        topk_indices: torch.Tensor,
        *,
        comm_num_sm: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(token_within_expert_offset, expert_counts)``.

        The intermediate block-cumulative histogram is dropped (unused
        by callers).
        """
        cfg = self.overlap_context.config
        resolved_comm_num_sm = self._resolve_comm_num_sm(comm_num_sm)
        (token_within_expert_offset, _block_cumsum_hist,
         expert_counts) = (_ep.compute_stable_local_token_within_expert_offset_and_expert_counts(
             topk_indices,
             cfg.num_experts,
             resolved_comm_num_sm,
         ))
        return token_within_expert_offset, expert_counts

    # ------------------------------------------------------------------
    # Internal: dispatch layout + staging buffer preamble.
    # ------------------------------------------------------------------

    def _validate_dispatch_weights(self, topk_weights: torch.Tensor, input: torch.Tensor, cfg, *, op_name: str) -> None:
        """Validate the optional ``topk_weights`` side channel."""
        if topk_weights.dtype != cfg.weight_dtype:
            raise TypeError(f"{op_name} topk_weights dtype {topk_weights.dtype} != "
                            f"config weight_dtype {cfg.weight_dtype}")
        if not topk_weights.is_contiguous():
            raise ValueError(f"{op_name} requires a contiguous topk_weights")
        if topk_weights.ndim != 2:
            raise ValueError(f"{op_name} topk_weights must be 2D (num_tokens, topk);"
                             f" got shape {tuple(topk_weights.shape)}")
        if topk_weights.shape[0] != input.shape[0]:
            raise ValueError(f"{op_name} topk_weights.shape[0] != input.shape[0]: "
                             f"{topk_weights.shape[0]} vs {input.shape[0]}")
        if topk_weights.shape[1] != cfg.topk:
            raise ValueError(f"{op_name} topk_weights.shape[1] != config.topk: "
                             f"{topk_weights.shape[1]} vs {cfg.topk}")

    def _validate_dispatch_input(self, input: torch.Tensor, cfg, *, op_name: str) -> None:
        if input.dtype != cfg.token_dtype:
            raise TypeError(f"{op_name} input dtype {input.dtype} != "
                            f"config token_dtype {cfg.token_dtype}")
        if not input.is_contiguous():
            raise ValueError(f"{op_name} requires a contiguous input")
        if input.shape[0] > cfg.max_m:
            raise ValueError(f"input rows {input.shape[0]} exceed configured max_m={cfg.max_m}")
        if input.shape[1] != cfg.hidden:
            raise ValueError(f"input hidden {input.shape[1]} != configured hidden={cfg.hidden}")

    def _ensure_dispatch_layout(self, layout_desc: EPCommLayoutDesc, topk_indices: torch.Tensor, *,
                                comm_num_sm: Optional[int] = None) -> None:
        """Idempotently (re)compute the dispatch layout."""
        if layout_desc.need_recompute_token_within_expert_offset_and_expert_counts(topk_indices):
            (layout_desc.token_within_expert_offset,
             layout_desc.expert_counts) = (self.compute_token_within_expert_offset_and_expert_counts(
                 topk_indices,
                 comm_num_sm=comm_num_sm,
             ))

        cfg = self.overlap_context.config
        num_tokens = int(topk_indices.shape[0])
        recompute = layout_desc.need_recompute_dispatch_layout(self.expert_alignment, num_tokens)
        if not recompute:
            if cfg.nnodes > 1:
                if layout_desc.num_tokens_per_rank is None:
                    raise ValueError("layout_desc.num_tokens_per_rank is required for internode dispatch layout reuse")
                self._stage_internode_sender_layout(layout_desc, topk_indices)
            return

        if self.expert_alignment > 1:
            # Re-arm the -1 sentinel for padded slots; barrier so peers
            # cannot overwrite our fill via cross-rank layout writes.
            self.overlap_context.token_src_rank_topk_and_indices_buf.fill_(-1)
            self.ep_group_barrier()

        resolved_comm_num_sm = self._resolve_comm_num_sm(comm_num_sm)
        if cfg.nnodes == 1:
            (
                layout_desc.recv_base_offset,
                layout_desc.token_dst_scatter_indices,
                layout_desc.token_topk_send_mask,
                layout_desc.recv_token_count_cpu,
                layout_desc.recv_token_count,
                layout_desc.recv_aligned_token_count_cpu,
                layout_desc.recv_aligned_token_count,
                layout_desc.recv_expert_counts,
            ) = _ep.compute_dispatch_layout(
                topk_indices,
                layout_desc.token_within_expert_offset,
                layout_desc.expert_counts,
                self.overlap_context.full_splits_buf_ptrs,
                self.overlap_context.nvl_barrier_buf_ptrs,
                cfg.num_experts,
                cfg.rank,
                cfg.world_size,
                resolved_comm_num_sm,
                token_src_rank_topk_and_indices_ptrs=(self.overlap_context.token_src_rank_topk_and_indices_buf_ptrs),
                expert_alignment=self.expert_alignment,
            )
        else:
            layout_desc.num_tokens_per_rank = torch.empty(
                (cfg.world_size, ),
                dtype=torch.int32,
                device=topk_indices.device,
            )
            # compute_dispatch_layout is stateless: it allocates layout-private
            # recv-count buffers and does not stage the RDMA rail slot.
            (
                layout_desc.recv_base_offset,
                sender_token_dst_scatter_indices,
                sender_topk_send_mask,
                layout_desc.recv_token_count_cpu,
                layout_desc.recv_token_count,
                layout_desc.recv_aligned_token_count_cpu,
                layout_desc.recv_aligned_token_count,
                layout_desc.recv_expert_counts,
            ) = _ep_inter.compute_dispatch_layout(
                topk_indices,
                layout_desc.token_within_expert_offset,
                layout_desc.expert_counts,
                self.overlap_context.full_splits_win_handle,
                layout_desc.num_tokens_per_rank,
                cfg.num_experts,
                resolved_comm_num_sm,
                expert_alignment=self.expert_alignment,
            )
            layout_desc.token_dst_scatter_indices = sender_token_dst_scatter_indices
            layout_desc.token_topk_send_mask = sender_topk_send_mask
            layout_desc.internode_sender_topk_send_mask = sender_topk_send_mask
            layout_desc.internode_sender_token_dst_scatter_indices = sender_token_dst_scatter_indices
            # Stage sender-space metadata into the RDMA rail every compute so a
            # later dispatch cannot observe a stale slot (and so layout reuse
            # via _stage_internode_sender_layout stays consistent).
            self._stage_internode_sender_layout(layout_desc, topk_indices)
        layout_desc.expert_alignment = self.expert_alignment
        layout_desc.num_tokens = num_tokens

    def _stage_internode_sender_layout(
        self,
        layout_desc: EPCommLayoutDesc,
        topk_indices: torch.Tensor,
    ) -> None:
        """Restage sender-token-space layout metadata for cached internode layouts."""
        cfg = self.overlap_context.config
        sender_mask = layout_desc.internode_sender_topk_send_mask
        sender_scatter = layout_desc.internode_sender_token_dst_scatter_indices

        if (sender_mask is None and layout_desc.token_topk_send_mask is not None
                and layout_desc.token_topk_send_mask.dim() == 2):
            sender_mask = layout_desc.token_topk_send_mask
        if (sender_scatter is None and layout_desc.token_dst_scatter_indices is not None
                and layout_desc.token_dst_scatter_indices.dim() == 2):
            sender_scatter = layout_desc.token_dst_scatter_indices

        if sender_mask is None or sender_scatter is None:
            raise ValueError("internode dispatch layout reuse requires sender-side "
                             "topk_send_mask and token_dst_scatter metadata")

        expected_shape = tuple(topk_indices.shape)
        if tuple(sender_mask.shape) != expected_shape:
            raise ValueError("internode sender_topk_send_mask shape "
                             f"{tuple(sender_mask.shape)} != topk_indices shape {expected_shape}")
        if tuple(sender_scatter.shape) != expected_shape:
            raise ValueError("internode sender_token_dst_scatter_indices shape "
                             f"{tuple(sender_scatter.shape)} != topk_indices shape {expected_shape}")

        rdma_rail_send_views = self.overlap_context.rdma_rail_send_slot_views(
            topk_indices.shape[0],
            cfg.hidden,
            cfg.topk,
            max_slot_num_token=cfg.max_m,
        )
        rdma_rail_send_views["topk_send_mask"].copy_(sender_mask)
        rdma_rail_send_views["token_dst_scatter"].copy_(sender_scatter)
        layout_desc.internode_sender_topk_send_mask = sender_mask
        layout_desc.internode_sender_token_dst_scatter_indices = sender_scatter

    def _select_recv_count(self, layout_desc: EPCommLayoutDesc):
        if layout_desc.expert_alignment > 1:
            if layout_desc.recv_aligned_token_count_cpu is None:
                raise ValueError("expert_alignment > 1 requires recv_aligned_token_count_cpu")
            return (layout_desc.recv_aligned_token_count_cpu, layout_desc.recv_aligned_token_count)
        return (layout_desc.recv_token_count_cpu, layout_desc.recv_token_count)

    def _poll_local_recv_count(
        self,
        recv_token_count_cpu: torch.Tensor,
        recv_token_count: torch.Tensor,
    ) -> Tuple[int, int]:
        """Busy-wait until the layout kernel publishes the recv count.

        Returns ``(this_rank_count, all_rank_max)``. For a fixed positive
        ``num_worst_tokens`` the configured bound is returned without a CPU
        poll, matching :class:`EPKernels`.
        """
        if not recv_token_count_cpu.is_cpu:
            raise ValueError("recv_token_count_cpu must be a CPU pinned tensor")
        if recv_token_count_cpu.dtype != torch.int32:
            raise TypeError("recv_token_count_cpu must be int32; got "
                            f"{recv_token_count_cpu.dtype}")
        if self.num_worst_tokens > 0:
            if self.check_num_worst_tokens:
                torch._assert_async(
                    recv_token_count[self.rank] <= self.num_worst_tokens,
                    f"num_worst_tokens = {self.num_worst_tokens} is not valid",
                )
            return int(self.num_worst_tokens), int(self.num_worst_tokens)

        arr = recv_token_count_cpu.numpy()
        while int(arr.min()) == _PENDING_RECV_COUNT_SENTINEL:
            if self.cpu_poll_sleep_us > 0:
                time.sleep(self.cpu_poll_sleep_s)
        cur_recv = int(arr[self.rank])
        max_recv = int(arr.max())
        return cur_recv, max_recv

    def _grow_dispatch_buffers(self, max_recv_tokens: int) -> bool:
        """Collectively grow receive buffers after layout count publication."""
        ctx = self.overlap_context
        max_recv_tokens = int(max_recv_tokens)
        if (self.num_worst_tokens > 0 or ctx.config.nnodes == 1 or max_recv_tokens <= ctx.max_recv_tokens):
            return False

        # The layout kernel only publishes counts and sender metadata for the
        # inter-node data path. Retire it before replacing receive buffers.
        self.ep_group_barrier()
        torch.cuda.synchronize()
        aligned = ((max_recv_tokens + ctx.alloc_alignment - 1) // ctx.alloc_alignment * ctx.alloc_alignment)
        growth_target = int(aligned * self.capacity_coeff)
        alloc_tokens = max(max_recv_tokens, growth_target)
        alloc_tokens = ((alloc_tokens + ctx.alloc_alignment - 1) // ctx.alloc_alignment * ctx.alloc_alignment)
        old_capacity = ctx.max_recv_tokens
        if self.rank == 0:
            print(f"reallocate EP-overlap recv buffers from {old_capacity} to {alloc_tokens}")
        ctx.reallocate_dispatch_buffers(alloc_tokens)
        self.ep_group_barrier()
        torch.cuda.synchronize()
        return True

    def _prepare_dispatch_layout(
        self,
        layout_desc: EPCommLayoutDesc,
        topk_indices: torch.Tensor,
        *,
        comm_num_sm: Optional[int] = None,
    ) -> Tuple[int, torch.Tensor]:
        """Compute layout and grow inter-node receive buffers if needed."""
        self._ensure_dispatch_layout(layout_desc, topk_indices, comm_num_sm=comm_num_sm)
        buf_count_cpu, kernel_recv_count = self._select_recv_count(layout_desc)
        cur_recv, max_recv = self._poll_local_recv_count(buf_count_cpu, kernel_recv_count)
        self._grow_dispatch_buffers(max_recv)
        return cur_recv, kernel_recv_count

    def _prepare_dispatch_buffers(
        self,
        input: torch.Tensor,
        topk_indices: torch.Tensor,
        layout_desc: EPCommLayoutDesc,
        topk_weights: Optional[torch.Tensor] = None,
        comm_num_sm: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, EPCommLayoutDesc]:
        """Common preamble for standalone and fused CuTeDSL dispatch.

        Stages ``input`` into the symmetric dispatch buffer, refreshes
        the layout, polls the receive count, and allocates the local
        pull-mode output. Returns
        ``(output, kernel_recv_count, layout_desc)``.

        Post-copy ``ep_group_barrier`` guarantees peers can pull-read
        the staged input before their dispatch kernel starts. When
        ``topk_weights`` is supplied the symmetric weight buffer is
        staged in the same window, sharing the single ``ep_group_barrier``
        so the has_weight path costs exactly one extra ``copy_`` (no
        extra cross-rank synchronisation, no extra fence).
        """
        ctx = self.overlap_context
        cfg = ctx.config
        ctx.ensure_dispatch_input()
        if topk_weights is not None:
            ctx.ensure_dispatch_input_weight()

        dispatch_recv_token_count, kernel_recv_count = self._prepare_dispatch_layout(
            layout_desc,
            topk_indices,
            comm_num_sm=comm_num_sm,
        )

        ctx.dispatch_input_buf[:input.shape[0]].copy_(input)
        if topk_weights is not None:
            ctx.dispatch_input_weight_buf[:topk_weights.shape[0]].copy_(topk_weights, )
        self.ep_group_barrier()

        layout_desc.token_src_rank_topk_and_indices = (
            ctx.token_src_rank_topk_and_indices_buf[:dispatch_recv_token_count])
        layout_desc.topk_indices = topk_indices
        # Pull-mode output is local-only (peers don't need its address),
        # so we use a fresh per-call torch.empty.
        output = torch.empty(
            (dispatch_recv_token_count, cfg.hidden),
            dtype=cfg.token_dtype,
            device=input.device,
        )
        return output, kernel_recv_count, layout_desc

    # ------------------------------------------------------------------
    # CuTeDSL pull-mode dispatch.
    # ------------------------------------------------------------------

    def dispatch_cutedsl(self, input: torch.Tensor, topk_indices: torch.Tensor,
                         layout_desc: Optional[EPCommLayoutDesc] = None, enable_expert_signals: bool = False,
                         comm_num_sm: Optional[int] = None):
        """CuTeDSL pull-mode dispatch; returns ``(output, layout_desc)``.

        Pull writes land on the local rank only -- the next call's
        pre-copy barrier is the sole peer-wait point.
        """
        cfg = self.overlap_context.config
        if cfg.nnodes != 1:
            if enable_expert_signals:
                raise NotImplementedError("inter-node dispatch_cutedsl does not support expert signals yet")
            return self._dispatch_cutedsl_inter(
                input,
                topk_indices,
                layout_desc,
                comm_num_sm=comm_num_sm,
            )
        if layout_desc is None:
            layout_desc = EPCommLayoutDesc()
        else:
            layout_desc.check_layout_desc(
                num_tokens=topk_indices.shape[0],
                topk=topk_indices.shape[1],
                num_experts=cfg.num_experts,
                world_size=cfg.world_size,
                local_world_size=cfg.local_world_size,
                max_slot_num_token=cfg.max_m,
            )
        self._validate_dispatch_input(input, cfg, op_name="dispatch_cutedsl")

        output, kernel_recv_count, layout_desc = self._prepare_dispatch_buffers(
            input,
            topk_indices,
            layout_desc,
            comm_num_sm=comm_num_sm,
        )

        resolved_comm_num_sm = self._resolve_comm_num_sm(comm_num_sm)
        if enable_expert_signals:
            self.overlap_context.reset_expert_signals()
        self._dispatch_op.run(
            dispatch_input_ptrs=self.overlap_context.dispatch_input_ptrs,
            token_src_rank_topk_and_indices=layout_desc.token_src_rank_topk_and_indices,
            recv_count=kernel_recv_count,
            output_buf=output,
            recv_expert_counts=layout_desc.recv_expert_counts,
            expert_signals=self.overlap_context.expert_signals,
            expert_signal_counters=self.overlap_context.expert_signal_counters,
            enable_expert_signals=enable_expert_signals,
            comm_num_sm=resolved_comm_num_sm,
        )
        return output, layout_desc

    def _dispatch_cutedsl_inter(
        self,
        input: torch.Tensor,
        topk_indices: torch.Tensor,
        layout_desc: Optional[EPCommLayoutDesc] = None,
        *,
        comm_num_sm: Optional[int] = None,
    ):
        cfg = self.overlap_context.config
        if layout_desc is None:
            layout_desc = EPCommLayoutDesc()
        else:
            layout_desc.check_layout_desc(
                num_tokens=topk_indices.shape[0],
                topk=topk_indices.shape[1],
                num_experts=cfg.num_experts,
                world_size=cfg.world_size,
                local_world_size=cfg.local_world_size,
                max_slot_num_token=cfg.max_m,
            )
        self._validate_dispatch_input(input, cfg, op_name="dispatch_cutedsl_inter")

        dispatch_recv_token_count, kernel_recv_count = self._prepare_dispatch_layout(
            layout_desc,
            topk_indices,
            comm_num_sm=comm_num_sm,
        )

        ctx = self.overlap_context
        if ctx.dispatch_output_buf is None or ctx.dispatch_topk_scatter_indices_buf is None:
            raise RuntimeError("inter-node dispatch output buffers are not initialized")
        if dispatch_recv_token_count > ctx.dispatch_output_buf.shape[0]:
            raise RuntimeError(f"dispatch recv token count {dispatch_recv_token_count} exceeds "
                               f"preallocated inter-node output capacity "
                               f"{ctx.dispatch_output_buf.shape[0]}")

        rdma_rail_send_views = ctx.rdma_rail_send_slot_views(
            input.shape[0],
            cfg.hidden,
            cfg.topk,
            max_slot_num_token=cfg.max_m,
        )
        rdma_rail_send_views["x"].copy_(input)
        rdma_rail_send_views["topk_indices"].copy_(topk_indices)
        ctx.dispatch_topk_scatter_indices_buf[:dispatch_recv_token_count].fill_(-1)
        self.ep_group_barrier()

        meta_shape = (cfg.nnodes, cfg.max_m, cfg.topk)
        node_topk_indices = torch.empty(
            meta_shape,
            dtype=topk_indices.dtype,
            device=topk_indices.device,
        )
        node_topk_indices.fill_(cfg.num_experts)
        node_topk_send_mask = torch.empty(
            meta_shape,
            dtype=torch.int32,
            device=topk_indices.device,
        )
        node_topk_send_mask.zero_()
        node_token_dst_scatter_indices = torch.empty(
            meta_shape,
            dtype=cfg.offset_dtype,
            device=topk_indices.device,
        )
        node_token_dst_scatter_indices.fill_(-1)

        resolved_comm_num_sm = self._resolve_comm_num_sm(comm_num_sm)
        self._dispatch_inter_op.run(
            dev_comm_ptr=ctx.nccl_gin_dev_comm_ptr(),
            rdma_rail_send_win_handle=ctx.rdma_rail_send_win_handle,
            num_tokens_per_rank=layout_desc.num_tokens_per_rank,
            recv_x_ptrs=ctx.dispatch_output_ptrs,
            recv_topk_scatter_indices_ptrs=ctx.dispatch_topk_scatter_indices_ptrs,
            node_topk_indices=node_topk_indices,
            node_topk_send_mask=node_topk_send_mask,
            node_token_dst_scatter_indices=node_token_dst_scatter_indices,
            output_buf=ctx.dispatch_output_buf,
            max_slot_num_token=cfg.max_m,
            max_recv_tokens=int(ctx.dispatch_output_buf.shape[0]),
            experts_per_rank=cfg.num_experts // cfg.world_size,
            num_sm=resolved_comm_num_sm,
        )
        self._reset_dispatch_signals_barrier()

        # Master's internode C++ combine consumes node_* (receiver-view) metadata.
        # Also mirror into token_* so existing CuTeDSL combine call sites that still
        # read the node-view through those names keep working.
        layout_desc.node_topk_indices = node_topk_indices
        layout_desc.node_topk_send_mask = node_topk_send_mask
        layout_desc.node_token_dst_scatter_indices = node_token_dst_scatter_indices
        layout_desc.topk_indices = node_topk_indices
        layout_desc.token_topk_send_mask = node_topk_send_mask
        layout_desc.token_dst_scatter_indices = node_token_dst_scatter_indices
        layout_desc.recv_topk_scatter_indices = (ctx.dispatch_topk_scatter_indices_buf[:dispatch_recv_token_count])
        layout_desc.token_src_rank_topk_and_indices = None
        return ctx.dispatch_output_buf[:dispatch_recv_token_count], layout_desc

    # ------------------------------------------------------------------
    # CuTeDSL push-mode combine.
    # ------------------------------------------------------------------

    def combine_cutedsl(self, input: torch.Tensor, layout_desc: EPCommLayoutDesc = None, *, push_mode: str = "1d",
                        tile_m: int = 128, tile_n: int = 128, profile_stages: bool = False,
                        comm_num_sm: Optional[int] = None, dispatched_weights: Optional[torch.Tensor] = None):
        """CuTeDSL push-mode combine.

        ``push_mode``: ``"1d"`` or ``"tile"`` (2D blocked push).
        ``dispatched_weights`` is currently only accepted by the
        inter-node scaffold, where it is staged as the optional combine
        weight side channel.
        ``profile_stages=True`` returns
        ``(output, {push_ms, barrier_ms, reduce_ms})``; otherwise the
        dense output tensor is returned alone.
        """
        if layout_desc is None:
            raise ValueError("layout_desc is required for combine_cutedsl")
        cfg = self.overlap_context.config
        if cfg.nnodes != 1:
            return self._combine_cutedsl_inter(
                input=input,
                layout_desc=layout_desc,
                profile_stages=profile_stages,
                comm_num_sm=comm_num_sm,
                dispatched_weights=dispatched_weights,
            )
        if dispatched_weights is not None:
            raise NotImplementedError("combine_cutedsl dispatched_weights are currently inter-node only")
        if input.dtype != cfg.token_dtype:
            raise TypeError(f"combine_cutedsl input dtype {input.dtype} != "
                            f"config token_dtype {cfg.token_dtype}")
        if not input.is_contiguous():
            raise ValueError("combine_cutedsl requires a contiguous input")
        if input.shape[1] != cfg.hidden:
            raise ValueError(f"combine_cutedsl input hidden {input.shape[1]} != "
                             f"configured hidden={cfg.hidden}")
        if layout_desc.token_src_rank_topk_and_indices is None:
            raise ValueError("layout_desc.token_src_rank_topk_and_indices is required")
        if layout_desc.topk_indices is None:
            raise ValueError("layout_desc.topk_indices is required")
        if layout_desc.topk_indices.shape[1] != cfg.topk:
            raise ValueError(f"layout_desc.topk_indices topk {layout_desc.topk_indices.shape[1]} "
                             f"!= configured topk={cfg.topk}")

        self.overlap_context.ensure_combine_output()
        _, kernel_recv_count = self._select_recv_count(layout_desc)
        resolved_comm_num_sm = self._resolve_comm_num_sm(comm_num_sm)

        events = _StageEvents(profile_stages)
        events.record("push_start")
        if push_mode == "1d":
            self._combine_push_op.run(
                input_buf=input,
                meta=layout_desc.token_src_rank_topk_and_indices,
                recv_count=kernel_recv_count,
                output_ptrs=self.overlap_context.combine_output_ptrs,
                topk=int(layout_desc.topk_indices.shape[1]),
                comm_num_sm=resolved_comm_num_sm,
            )
        elif push_mode == "tile":
            self._combine_tile_push_op.run(
                input_buf=input,
                meta=layout_desc.token_src_rank_topk_and_indices,
                recv_count=kernel_recv_count,
                output_ptrs=self.overlap_context.combine_output_ptrs,
                tile_m=tile_m,
                tile_n=tile_n,
                topk=int(layout_desc.topk_indices.shape[1]),
                comm_num_sm=resolved_comm_num_sm,
            )
        else:
            raise ValueError(f"unknown cutedsl combine push_mode: {push_mode}")
        events.record("push_end")

        # Peers must finish writing our staging stripe before reduce.
        self.ep_group_barrier()
        events.record("barrier_end")

        num_tokens = layout_desc.topk_indices.shape[0]
        output = torch.empty((num_tokens, cfg.hidden), dtype=cfg.token_dtype, device=input.device)
        staging = self.overlap_context.combine_output_buf[:num_tokens * cfg.topk]
        self._topk_reduce_op.run(
            staging=staging,
            topk_indices=layout_desc.topk_indices,
            output=output,
            num_experts=cfg.num_experts,
        )
        events.record("reduce_end")
        return events.finalize(output)

    def _combine_cutedsl_inter(
        self,
        *,
        input: torch.Tensor,
        layout_desc: EPCommLayoutDesc,
        profile_stages: bool,
        comm_num_sm: Optional[int],
        dispatched_weights: Optional[torch.Tensor],
    ):
        """Inter-node standalone combine scaffold.

        Stages the FC2 output into local-world peer-visible buffers and
        dispatches the CuTeDSL inter-node combine op. The concrete CuTeDSL
        kernels are intentionally still scaffolded in
        ``kernels/cutedsl_combine_inter.py``.
        """
        cfg = self.overlap_context.config
        ctx = self.overlap_context
        if input.dtype != cfg.token_dtype:
            raise TypeError(f"combine_cutedsl input dtype {input.dtype} != "
                            f"config token_dtype {cfg.token_dtype}")
        if not input.is_contiguous():
            raise ValueError("combine_cutedsl requires a contiguous input")
        if input.ndim != 2 or input.shape[1] != cfg.hidden:
            raise ValueError(f"combine_cutedsl input shape must be (*, {cfg.hidden}); "
                             f"got {tuple(input.shape)}")
        if layout_desc.token_within_expert_offset is None:
            raise ValueError("layout_desc.token_within_expert_offset is required for internode combine")
        if layout_desc.num_tokens_per_rank is None:
            raise ValueError("layout_desc.num_tokens_per_rank is required for internode combine")
        if (layout_desc.topk_indices is None or layout_desc.token_topk_send_mask is None
                or layout_desc.token_dst_scatter_indices is None):
            raise ValueError("internode combine requires per-node topk_indices, "
                             "token_topk_send_mask, and token_dst_scatter_indices")
        meta_shape = (cfg.nnodes, cfg.max_m, cfg.topk)
        for tensor, name in (
            (layout_desc.topk_indices, "layout_desc.topk_indices"),
            (layout_desc.token_dst_scatter_indices, "layout_desc.token_dst_scatter_indices"),
        ):
            if tensor.dtype != cfg.offset_dtype:
                raise TypeError(f"{name} dtype {tensor.dtype} != config offset_dtype {cfg.offset_dtype}")
            if tuple(tensor.shape) != meta_shape:
                raise ValueError(f"{name} must have shape {meta_shape} for internode combine; "
                                 f"got {tuple(tensor.shape)}")
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise ValueError(f"{name} must be a contiguous CUDA tensor")
        if layout_desc.token_topk_send_mask.dtype != torch.int32:
            raise TypeError("layout_desc.token_topk_send_mask dtype "
                            f"{layout_desc.token_topk_send_mask.dtype} != torch.int32")
        if tuple(layout_desc.token_topk_send_mask.shape) != meta_shape:
            raise ValueError("layout_desc.token_topk_send_mask must have shape "
                             f"{meta_shape} for internode combine; got "
                             f"{tuple(layout_desc.token_topk_send_mask.shape)}")
        if not layout_desc.token_topk_send_mask.is_cuda or not layout_desc.token_topk_send_mask.is_contiguous():
            raise ValueError("layout_desc.token_topk_send_mask must be a contiguous CUDA tensor")

        if ctx.dispatch_output_buf is None or ctx.dispatch_output_ptrs is None:
            raise RuntimeError("inter-node combine input buffers are not initialized")
        if ctx.rdma_rail_send_buf is None or ctx.rdma_rail_send_win_handle == 0:
            raise RuntimeError("inter-node combine RDMA rail buffer is not initialized")
        if input.shape[0] > ctx.dispatch_output_buf.shape[0]:
            raise RuntimeError(f"combine input rows {input.shape[0]} exceed peer-visible "
                               f"buffer capacity {ctx.dispatch_output_buf.shape[0]}")

        peer_visible_input = ctx.dispatch_output_buf[:input.shape[0]]
        if peer_visible_input.data_ptr() != input.data_ptr():
            peer_visible_input.copy_(input)

        has_weight = dispatched_weights is not None
        combine_weight_ptrs = None
        output_weight = None
        if has_weight:
            if dispatched_weights.dtype != cfg.weight_dtype:
                raise TypeError(f"combine_cutedsl dispatched_weights dtype "
                                f"{dispatched_weights.dtype} != config weight_dtype "
                                f"{cfg.weight_dtype}")
            if dispatched_weights.ndim != 1 or dispatched_weights.shape[0] != input.shape[0]:
                raise ValueError("combine_cutedsl dispatched_weights must be 1D with "
                                 f"length input.shape[0]={input.shape[0]}; got "
                                 f"{tuple(dispatched_weights.shape)}")
            if not dispatched_weights.is_contiguous():
                raise ValueError("combine_cutedsl dispatched_weights must be contiguous")
            ctx.ensure_dispatch_group_gemm_output_weight()
            if ctx.dispatch_group_gemm_output_weight_buf is None or ctx.dispatch_group_gemm_output_weight_ptrs is None:
                raise RuntimeError("inter-node combine weight buffers are not initialized")
            if input.shape[0] > ctx.dispatch_group_gemm_output_weight_buf.shape[0]:
                raise RuntimeError(f"combine weight rows {input.shape[0]} exceed peer-visible "
                                   f"weight capacity {ctx.dispatch_group_gemm_output_weight_buf.shape[0]}")
            peer_visible_weights = ctx.dispatch_group_gemm_output_weight_buf[:input.shape[0]]
            if peer_visible_weights.data_ptr() != dispatched_weights.data_ptr():
                peer_visible_weights.copy_(dispatched_weights)
            combine_weight_ptrs = ctx.dispatch_group_gemm_output_weight_ptrs

        # Peers pull-read the staged FC2 output and optional dispatched weights
        # inside the inter-node combine partial kernel, so every rank must make
        # its staging copy visible before any rank starts the combine op.
        self.ep_group_barrier()

        events = _StageEvents(profile_stages)
        events.record("push_start")
        num_tokens = int(layout_desc.token_within_expert_offset.shape[0])
        output = torch.empty((num_tokens, cfg.hidden), dtype=cfg.token_dtype, device=input.device)
        if has_weight:
            output_weight = torch.empty((num_tokens, cfg.topk), dtype=cfg.weight_dtype, device=input.device)
        self._combine_inter_op.run(
            dev_comm_ptr=ctx.nccl_gin_dev_comm_ptr(),
            input_buf=peer_visible_input,
            combine_x_ptrs=ctx.dispatch_output_ptrs,
            combine_weight_ptrs=combine_weight_ptrs,
            rdma_rail_send_buf=ctx.rdma_rail_send_buf,
            rdma_rail_send_win_handle=ctx.rdma_rail_send_win_handle,
            num_tokens_per_rank=layout_desc.num_tokens_per_rank,
            output=output,
            local_topk_indices=layout_desc.topk_indices,
            local_topk_send_mask=layout_desc.token_topk_send_mask,
            local_token_dst_scatter_indices=layout_desc.token_dst_scatter_indices,
            output_weight=output_weight,
            max_slot_num_token=cfg.max_m,
            experts_per_rank=cfg.num_experts // cfg.world_size,
            num_sm=self._resolve_comm_num_sm(comm_num_sm),
        )
        events.record("push_end")
        self._reset_combine_signals_barrier()
        events.record("barrier_end")
        events.record("reduce_end")
        result = (output, output_weight) if has_weight else output
        return events.finalize(result)

    # ------------------------------------------------------------------
    # CuTeDSL standalone group-GEMM.
    # ------------------------------------------------------------------

    def group_gemm(self, A_padded: torch.Tensor, B: torch.Tensor, problem_sizes_m: torch.Tensor,
                   output: Optional[torch.Tensor] = None, *, gemm_num_sm: Optional[int] = None) -> torch.Tensor:
        """Standalone per-expert M-contiguous group-GEMM.

        Per-expert M must be a multiple of :attr:`cluster_tile_m`
        (caller pads via ``expert_alignment``). ``B`` shape
        ``(N, K, num_experts)``; ``n_out = B.shape[0]``.

        ``gemm_num_sm`` caps the cluster-persistent scheduler at
        ``floor(gemm_num_sm / cluster_size)`` resident clusters; it is
        baked into the compiled kernel and included in the op cache
        key so distinct values reuse / produce distinct artifacts.
        ``None`` -> full device SM count.
        """
        _validate_gemm_B(B, A_padded, int(problem_sizes_m.shape[0]), op_name="group_gemm")
        cfg = self.overlap_context.config
        if output is None:
            output = torch.empty((A_padded.shape[0], int(B.shape[0])), dtype=cfg.token_dtype, device=A_padded.device)
        num_sm = self._resolve_gemm_num_sm(gemm_num_sm)
        return self._group_gemm_op.run(
            A_padded=A_padded,
            B=B,
            problem_sizes_m=problem_sizes_m,
            output=output,
            expert_signals=self.overlap_context.expert_signals,
            num_sm=num_sm,
        )

    # ------------------------------------------------------------------
    # CuTeDSL fused group-GEMM + push combine.
    # ------------------------------------------------------------------

    def group_gemm_combine(
        self,
        A_padded: torch.Tensor,
        B: torch.Tensor,
        layout_desc: EPCommLayoutDesc,
        *,
        run_reduce: bool = True,
        gemm_num_sm: Optional[int] = None,
        dispatched_weights: Optional[torch.Tensor] = None,
    ):
        """Fused FC2 group-GEMM + push combine.

        ``B`` shape ``(N, K, num_experts)``; ``n_out = B.shape[0]``.
        Always returns the 2-tuple ``(output, combine_weights)``:
          * ``output`` is the dense ``(num_local_tokens, n_out)``
            reduce result, or ``None`` when ``run_reduce=False`` (the
            caller then owns the post-push NVL barrier and a
            subsequent :meth:`topk_reduce_only`).
          * ``combine_weights`` is the gathered ``(num_tokens, topk)``
            weight tensor when ``dispatched_weights`` is supplied, or
            ``None`` otherwise. When ``run_reduce=False`` and
            ``dispatched_weights`` is supplied, the returned
            ``combine_weights`` is a *view* of the symmetric staging
            buffer (caller still owes the post-push NVL barrier).

        ``gemm_num_sm`` controls the GEMM scheduler SM budget; see
        :meth:`group_gemm` for semantics. ``None`` -> full device.
        """
        cfg = self.overlap_context.config
        if cfg.nnodes != 1:
            return self._group_gemm_combine_inter(
                A_padded=A_padded,
                B=B,
                layout_desc=layout_desc,
                run_reduce=run_reduce,
                gemm_num_sm=gemm_num_sm,
                dispatched_weights=dispatched_weights,
            )
        if A_padded.dtype != cfg.token_dtype:
            raise TypeError(f"A_padded dtype {A_padded.dtype} != config token_dtype "
                            f"{cfg.token_dtype}")
        if B.dtype != cfg.token_dtype:
            raise TypeError(f"B dtype {B.dtype} != config token_dtype {cfg.token_dtype}")
        if not A_padded.is_contiguous():
            raise ValueError("group_gemm_combine requires A_padded to be contiguous")
        if layout_desc.token_src_rank_topk_and_indices is None:
            raise ValueError("layout_desc.token_src_rank_topk_and_indices is required")
        if layout_desc.recv_expert_counts is None:
            raise ValueError("layout_desc.recv_expert_counts is required")
        if layout_desc.topk_indices is None:
            raise ValueError("layout_desc.topk_indices is required")
        # B must be a K-major view (strides (K, 1, N*K)); see _validate_gemm_B.
        n_out = _validate_gemm_B(
            B,
            A_padded,
            int(layout_desc.recv_expert_counts.shape[0]),
            op_name="group_gemm_combine",
        )

        has_weight = dispatched_weights is not None
        weight_output_ptrs = None
        if has_weight:
            if dispatched_weights.dtype != cfg.weight_dtype:
                raise TypeError(f"group_gemm_combine dispatched_weights dtype "
                                f"{dispatched_weights.dtype} != config weight_dtype "
                                f"{cfg.weight_dtype}")
            if dispatched_weights.ndim != 1:
                raise ValueError("group_gemm_combine dispatched_weights must be 1D"
                                 f" (A_padded.shape[0],); got shape "
                                 f"{tuple(dispatched_weights.shape)}")
            if dispatched_weights.shape[0] != A_padded.shape[0]:
                raise ValueError("group_gemm_combine dispatched_weights.shape[0] != "
                                 f"A_padded.shape[0]: {dispatched_weights.shape[0]} "
                                 f"vs {A_padded.shape[0]}")
            if not dispatched_weights.is_contiguous():
                raise ValueError("group_gemm_combine dispatched_weights must be "
                                 "contiguous")
            # Per-rank symmetric (max_m, topk) destination buffer.
            self.overlap_context.ensure_group_gemm_combine_output_weight()
            weight_output_ptrs = (self.overlap_context.group_gemm_combine_output_weight_ptrs)

        self.overlap_context.ensure_group_gemm_combine_output(n_out)
        num_sm = self._resolve_gemm_num_sm(gemm_num_sm)
        self._group_gemm_combine_op.run(
            A_padded=A_padded,
            B=B,
            token_src_rank_topk_and_indices=layout_desc.token_src_rank_topk_and_indices,
            recv_expert_counts=layout_desc.recv_expert_counts,
            output_ptrs=self.overlap_context.group_gemm_combine_output_ptrs,
            num_sm=num_sm,
            topk=int(layout_desc.topk_indices.shape[1]),
            weight_dtype=cfg.weight_dtype,
            dispatched_weights=dispatched_weights,
            weight_output_ptrs=weight_output_ptrs,
        )

        # Post-push fence: peers must finish writing into our combine
        # staging before the local topk-reduce reads it back. When the
        # caller opts out of the reduce (``run_reduce=False``) the
        # barrier becomes the caller's responsibility.
        if not run_reduce:
            combine_weights = None
            if has_weight:
                num_tokens = layout_desc.topk_indices.shape[0]
                combine_weights = (self.overlap_context.group_gemm_combine_output_weight_buf[:num_tokens, :cfg.topk])
            return None, combine_weights

        self.ep_group_barrier()
        num_tokens = layout_desc.topk_indices.shape[0]
        output = torch.empty((num_tokens, n_out), dtype=cfg.token_dtype, device=A_padded.device)
        staging = self.overlap_context.group_gemm_combine_output_buf[:num_tokens * cfg.topk]
        self._topk_reduce_op.run(
            staging=staging,
            topk_indices=layout_desc.topk_indices,
            output=output,
            num_experts=cfg.num_experts,
        )
        combine_weights = None
        if has_weight:
            # Copy symmetric staging into an independent tensor, zeroing
            # drop-sentinel slots to match FlashComm CUDA semantics.
            combine_weights = masked_weight_copy(
                src=self.overlap_context.group_gemm_combine_output_weight_buf[:num_tokens, :cfg.topk],
                topk_indices=layout_desc.topk_indices,
                drop_sentinel=int(cfg.num_experts),
            )
        return output, combine_weights

    def _group_gemm_combine_inter(
        self,
        *,
        A_padded: torch.Tensor,
        B: torch.Tensor,
        layout_desc: EPCommLayoutDesc,
        run_reduce: bool,
        gemm_num_sm: Optional[int],
        dispatched_weights: Optional[torch.Tensor],
    ):
        """Inter-node FC2 group-GEMM + combine correctness path."""
        cfg = self.overlap_context.config
        ctx = self.overlap_context
        if A_padded.dtype != cfg.token_dtype:
            raise TypeError(f"A_padded dtype {A_padded.dtype} != config token_dtype "
                            f"{cfg.token_dtype}")
        if B.dtype != cfg.token_dtype:
            raise TypeError(f"B dtype {B.dtype} != config token_dtype {cfg.token_dtype}")
        if not A_padded.is_contiguous():
            raise ValueError("group_gemm_combine requires A_padded to be contiguous")
        if layout_desc.token_within_expert_offset is None:
            raise ValueError("layout_desc.token_within_expert_offset is required for internode group_gemm_combine")
        if layout_desc.recv_expert_counts is None:
            raise ValueError("layout_desc.recv_expert_counts is required")
        if layout_desc.num_tokens_per_rank is None:
            raise ValueError("layout_desc.num_tokens_per_rank is required for internode group_gemm_combine")
        if (layout_desc.topk_indices is None or layout_desc.token_topk_send_mask is None
                or layout_desc.token_dst_scatter_indices is None):
            raise ValueError("internode group_gemm_combine requires per-node topk_indices, "
                             "token_topk_send_mask, and token_dst_scatter_indices")
        if ctx.rdma_rail_send_buf is None or ctx.rdma_rail_send_win_handle == 0:
            raise RuntimeError("inter-node group_gemm_combine RDMA rail buffer is not initialized")
        if ctx.dispatch_output_buf is None or ctx.dispatch_output_ptrs is None:
            raise RuntimeError("inter-node group_gemm_combine FC2 staging buffers are not initialized")
        if ctx.expert_signal_state is None or ctx.expert_signal_state_ptrs is None:
            raise RuntimeError("inter-node group_gemm_combine signal workspace is not initialized")

        meta_shape = (cfg.nnodes, cfg.max_m, cfg.topk)
        for tensor, name in (
            (layout_desc.topk_indices, "layout_desc.topk_indices"),
            (layout_desc.token_dst_scatter_indices, "layout_desc.token_dst_scatter_indices"),
        ):
            if tensor.dtype != cfg.offset_dtype:
                raise TypeError(f"{name} dtype {tensor.dtype} != config offset_dtype {cfg.offset_dtype}")
            if tuple(tensor.shape) != meta_shape:
                raise ValueError(f"{name} must have shape {meta_shape} for internode group_gemm_combine; "
                                 f"got {tuple(tensor.shape)}")
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise ValueError(f"{name} must be a contiguous CUDA tensor")
        if layout_desc.token_topk_send_mask.dtype != torch.int32:
            raise TypeError("layout_desc.token_topk_send_mask dtype "
                            f"{layout_desc.token_topk_send_mask.dtype} != torch.int32")
        if tuple(layout_desc.token_topk_send_mask.shape) != meta_shape:
            raise ValueError("layout_desc.token_topk_send_mask must have shape "
                             f"{meta_shape} for internode group_gemm_combine; got "
                             f"{tuple(layout_desc.token_topk_send_mask.shape)}")
        if not layout_desc.token_topk_send_mask.is_cuda or not layout_desc.token_topk_send_mask.is_contiguous():
            raise ValueError("layout_desc.token_topk_send_mask must be a contiguous CUDA tensor")
        recv_topk_scatter_indices = layout_desc.recv_topk_scatter_indices
        if recv_topk_scatter_indices is not None:
            if recv_topk_scatter_indices.dtype != cfg.offset_dtype:
                raise TypeError("layout_desc.recv_topk_scatter_indices dtype "
                                f"{recv_topk_scatter_indices.dtype} != config offset_dtype {cfg.offset_dtype}")
            if not recv_topk_scatter_indices.is_cuda or not recv_topk_scatter_indices.is_contiguous():
                raise ValueError("layout_desc.recv_topk_scatter_indices must be a contiguous CUDA tensor")

        n_out = _validate_gemm_B(
            B,
            A_padded,
            int(layout_desc.recv_expert_counts.shape[0]),
            op_name="group_gemm_combine_inter",
        )
        if n_out != cfg.hidden:
            raise ValueError("inter-node group_gemm_combine currently expects FC2 output "
                             f"hidden {cfg.hidden}; got B N dim {n_out}")

        has_weight = dispatched_weights is not None
        if has_weight:
            if dispatched_weights.dtype != cfg.weight_dtype:
                raise TypeError(f"group_gemm_combine dispatched_weights dtype "
                                f"{dispatched_weights.dtype} != config weight_dtype "
                                f"{cfg.weight_dtype}")
            if dispatched_weights.ndim != 1 or dispatched_weights.shape[0] != A_padded.shape[0]:
                raise ValueError("group_gemm_combine dispatched_weights must be 1D with "
                                 f"length A_padded.shape[0]={A_padded.shape[0]}; got "
                                 f"{tuple(dispatched_weights.shape)}")
            if not dispatched_weights.is_contiguous():
                raise ValueError("group_gemm_combine dispatched_weights must be contiguous")

        num_sm = self._resolve_gemm_num_sm(gemm_num_sm)
        num_tokens = int(layout_desc.token_within_expert_offset.shape[0])
        output = torch.empty(
            (num_tokens, n_out),
            dtype=cfg.token_dtype,
            device=A_padded.device,
        )
        output_weight = None
        if has_weight:
            output_weight = torch.empty(
                (num_tokens, cfg.topk),
                dtype=cfg.weight_dtype,
                device=A_padded.device,
            )
        if A_padded.shape[0] > ctx.dispatch_output_buf.shape[0]:
            raise RuntimeError(f"group_gemm_combine FC2 rows {A_padded.shape[0]} exceed peer-visible "
                               f"staging capacity {ctx.dispatch_output_buf.shape[0]}")
        fc2_output = ctx.dispatch_output_buf[:A_padded.shape[0], :n_out]
        barrier_workspace = ctx.expert_signal_state.view(-1)
        ctx.reset_expert_signals()
        # The single fused kernel uses peer-visible signal slots for its
        # in-kernel local-world ready wait. Make the zeroed slots visible
        # before any rank can observe a stale ready flag from a previous call.
        self.ep_group_barrier()
        output, output_weight = self._group_gemm_combine_inter_op.run(
            dev_comm_ptr=ctx.nccl_gin_dev_comm_ptr(),
            A_padded=A_padded,
            B=B,
            recv_expert_counts=layout_desc.recv_expert_counts,
            num_tokens_per_rank=layout_desc.num_tokens_per_rank,
            local_topk_indices=layout_desc.topk_indices,
            local_topk_send_mask=layout_desc.token_topk_send_mask,
            local_token_dst_scatter_indices=layout_desc.token_dst_scatter_indices,
            recv_topk_scatter_indices=recv_topk_scatter_indices,
            fc2_output=fc2_output,
            combine_x_ptrs=ctx.dispatch_output_ptrs,
            barrier_workspace=barrier_workspace,
            barrier_workspace_ptrs=ctx.expert_signal_state_ptrs,
            output=output,
            output_weight=output_weight,
            rdma_rail_send_buf=ctx.rdma_rail_send_buf,
            rdma_rail_send_win_handle=ctx.rdma_rail_send_win_handle,
            max_slot_num_token=cfg.max_m,
            experts_per_rank=cfg.num_experts // cfg.world_size,
            num_sm=num_sm,
            topk=cfg.topk,
            weight_dtype=cfg.weight_dtype,
            dispatched_weights=dispatched_weights,
            run_reduce=run_reduce,
        )
        self._reset_combine_signals_barrier()
        combine_weights = output_weight if has_weight else None
        if not run_reduce:
            return None, combine_weights
        return output, combine_weights

    def topk_reduce_only(self, *, topk_indices: torch.Tensor, use_ggc_staging: bool = True,
                         num_tokens: Optional[int] = None) -> torch.Tensor:
        """Local topk-reduce of an already-staged combine buffer.

        Reads from the group-GEMM+combine staging buffer (``use_ggc_staging
        =True``) or the standalone combine staging buffer. Used by perf
        ablations to back out reduce time. ``num_tokens`` is required when
        the metadata's leading dimension is not the local token count, as in
        the inter-node ``(nnodes, max_m, topk)`` layout.
        """
        cfg = self.overlap_context.config
        ctx = self.overlap_context
        if use_ggc_staging and cfg.nnodes != 1 and num_tokens is None:
            raise ValueError("num_tokens is required for inter-node topk_reduce_only")
        if num_tokens is None:
            num_tokens = int(topk_indices.shape[0])
        else:
            num_tokens = int(num_tokens)
        if use_ggc_staging and cfg.nnodes != 1:
            if ctx.rdma_rail_send_buf is None:
                raise RuntimeError("inter-node RDMA rail buffer is not initialised")
            output = torch.empty((num_tokens, cfg.hidden), dtype=cfg.token_dtype, device="cuda")
            self._combine_inter_op.run_reduce_only(
                output=output,
                output_weight=None,
                rdma_rail_send_buf=ctx.rdma_rail_send_buf,
                rdma_rail_send_win_handle=ctx.rdma_rail_send_win_handle,
                max_slot_num_token=cfg.max_m,
                experts_per_rank=cfg.num_experts // cfg.world_size,
                topk=cfg.topk,
            )
            return output
        if use_ggc_staging:
            if ctx.group_gemm_combine_output_buf is None:
                raise RuntimeError("group_gemm_combine staging buffer is not initialised; "
                                   "call group_gemm_combine at least once")
            staging_buf = ctx.group_gemm_combine_output_buf
            n_out = int(ctx.group_gemm_combine_output_n_out)
        else:
            ctx.ensure_combine_output()
            staging_buf = ctx.combine_output_buf
            n_out = int(cfg.hidden)

        output = torch.empty((num_tokens, n_out), dtype=cfg.token_dtype, device="cuda")
        staging = staging_buf[:num_tokens * cfg.topk]
        self._topk_reduce_op.run(
            staging=staging,
            topk_indices=topk_indices,
            output=output,
            num_experts=cfg.num_experts,
        )
        return output

    # ------------------------------------------------------------------
    # CuTeDSL fused dispatch + group-GEMM.
    # ------------------------------------------------------------------

    def dispatch_group_gemm(
        self,
        input: torch.Tensor,
        topk_indices: torch.Tensor,
        B: torch.Tensor,
        layout_desc: Optional[EPCommLayoutDesc],
        *,
        dispatch_num_stages: int = 1,
        A_padded: Optional[torch.Tensor] = None,
        gemm_num_sm: Optional[int] = None,
        comm_num_sm: Optional[int] = None,
        topk_weights: Optional[torch.Tensor] = None,
    ):
        """Fused pull-dispatch + FC1 GEMM overlap kernel.

        ``B`` shape ``(N, K, num_experts)``; ``n_out = B.shape[0]``.
        Always returns the 3-tuple
        ``(output, dispatch_weights, layout_desc)``;
        ``dispatch_weights`` is ``None`` when ``topk_weights`` is not
        supplied. ``A_padded`` lets the caller reuse a pre-allocated
        dispatch staging tensor.

        ``gemm_num_sm`` controls the GEMM scheduler SM budget; see
        :meth:`group_gemm` for semantics. ``None`` -> full device.
        ``comm_num_sm`` controls the dispatch layout helpers;
        ``None`` -> full device.

        ``topk_weights`` (Optional, shape ``(num_tokens, topk)``,
        ``cfg.weight_dtype``, contiguous) enables a side-loaded weight
        gather co-issued by the dispatch producer warp. The destination
        is a fresh 1D tensor of length ``A_padded.shape[0]`` (one
        scalar per dispatched flat row). Padded rows (layout sentinel
        ``-1``) are left untouched; densify via
        ``dispatch_weights[:recv_count]`` for the valid prefix.
        """
        cfg = self.overlap_context.config
        if cfg.nnodes != 1:
            return self._dispatch_group_gemm_inter(
                input=input,
                topk_indices=topk_indices,
                B=B,
                layout_desc=layout_desc,
                dispatch_num_stages=dispatch_num_stages,
                A_padded=A_padded,
                gemm_num_sm=gemm_num_sm,
                comm_num_sm=comm_num_sm,
                topk_weights=topk_weights,
            )

        if layout_desc is None:
            layout_desc = EPCommLayoutDesc()
        else:
            layout_desc.check_layout_desc(
                num_tokens=topk_indices.shape[0],
                topk=topk_indices.shape[1],
                num_experts=cfg.num_experts,
                world_size=cfg.world_size,
                local_world_size=cfg.local_world_size,
                max_slot_num_token=cfg.max_m,
            )
        self._validate_dispatch_input(
            input,
            cfg,
            op_name="dispatch_group_gemm",
        )
        has_weight = topk_weights is not None
        if has_weight:
            self._validate_dispatch_weights(
                topk_weights,
                input,
                cfg,
                op_name="dispatch_group_gemm",
            )
        prepared_A, kernel_recv_count, layout_desc = (self._prepare_dispatch_buffers(
            input,
            topk_indices,
            layout_desc,
            topk_weights=topk_weights,
            comm_num_sm=comm_num_sm,
        ))
        if A_padded is None:
            A_padded = prepared_A
        else:
            if A_padded.shape != prepared_A.shape:
                raise ValueError("external A_padded.shape "
                                 f"{tuple(A_padded.shape)} != expected "
                                 f"{tuple(prepared_A.shape)}")
            if A_padded.dtype != prepared_A.dtype:
                raise TypeError(f"external A_padded dtype {A_padded.dtype} != "
                                f"expected {prepared_A.dtype}")

        n_out = _validate_gemm_B(
            B,
            A_padded,
            int(layout_desc.recv_expert_counts.shape[0]),
            op_name="dispatch_group_gemm",
        )
        self.overlap_context.reset_expert_signals()

        # Optional weight: input staging + ep_group_barrier were already
        # batched inside ``_prepare_dispatch_buffers``. All that's left
        # is to grab the symmetric ptr table and allocate a local 1D
        # output for the kernel's per-row 4 B writes.
        input_weight_ptrs = None
        output_weight = None
        if has_weight:
            input_weight_ptrs = (self.overlap_context.dispatch_input_weight_ptrs)
            output_weight = torch.empty(
                (A_padded.shape[0], ),
                dtype=topk_weights.dtype,
                device=A_padded.device,
            )

        output = torch.empty((A_padded.shape[0], n_out), dtype=cfg.token_dtype, device=A_padded.device)
        num_sm = self._resolve_gemm_num_sm(gemm_num_sm)
        self._dispatch_group_gemm_op.run(
            A_padded=A_padded,
            B=B,
            token_src_rank_topk_and_indices=layout_desc.token_src_rank_topk_and_indices,
            recv_count=kernel_recv_count,
            recv_expert_counts=layout_desc.recv_expert_counts,
            output=output,
            dispatch_input_ptrs=self.overlap_context.dispatch_input_ptrs,
            expert_signals=self.overlap_context.expert_signals,
            expert_signal_counters=self.overlap_context.expert_signal_counters,
            dispatch_num_stages=dispatch_num_stages,
            num_sm=num_sm,
            input_weight_ptrs=input_weight_ptrs,
            output_weight=output_weight,
            topk=int(topk_indices.shape[1]),
            weight_dtype=cfg.weight_dtype,
        )
        return output, output_weight, layout_desc

    def _dispatch_group_gemm_inter(
        self,
        *,
        input: torch.Tensor,
        topk_indices: torch.Tensor,
        B: torch.Tensor,
        layout_desc: Optional[EPCommLayoutDesc],
        dispatch_num_stages: int,
        A_padded: Optional[torch.Tensor],
        gemm_num_sm: Optional[int],
        comm_num_sm: Optional[int],
        topk_weights: Optional[torch.Tensor],
    ):
        """Inter-node dispatch+FC1 fused op."""
        ctx = self.overlap_context
        cfg = ctx.config
        if layout_desc is None:
            layout_desc = EPCommLayoutDesc()
        else:
            layout_desc.check_layout_desc(
                num_tokens=topk_indices.shape[0],
                topk=topk_indices.shape[1],
                num_experts=cfg.num_experts,
                world_size=cfg.world_size,
                local_world_size=cfg.local_world_size,
                max_slot_num_token=cfg.max_m,
            )
        self._validate_dispatch_input(
            input,
            cfg,
            op_name="dispatch_group_gemm_inter",
        )
        has_weight = topk_weights is not None
        if has_weight:
            self._validate_dispatch_weights(
                topk_weights,
                input,
                cfg,
                op_name="dispatch_group_gemm_inter",
            )

        dispatch_recv_token_count, kernel_recv_count = self._prepare_dispatch_layout(
            layout_desc,
            topk_indices,
            comm_num_sm=comm_num_sm,
        )

        if ctx.dispatch_output_buf is None or ctx.dispatch_output_ptrs is None:
            raise RuntimeError("inter-node fused dispatch output buffers are not initialized")
        if ctx.expert_signal_state_ptrs is None:
            raise RuntimeError("inter-node fused dispatch expert signal buffer must be peer-visible")
        if dispatch_recv_token_count > ctx.dispatch_output_buf.shape[0]:
            raise RuntimeError(f"dispatch recv token count {dispatch_recv_token_count} exceeds "
                               f"preallocated inter-node output capacity "
                               f"{ctx.dispatch_output_buf.shape[0]}")
        if has_weight:
            ctx.ensure_dispatch_group_gemm_output_weight()

        peer_visible_A = ctx.dispatch_output_buf[:dispatch_recv_token_count]
        if A_padded is None:
            A_padded = peer_visible_A
        else:
            expected_shape = tuple(peer_visible_A.shape)
            if tuple(A_padded.shape) != expected_shape:
                raise ValueError("external A_padded.shape "
                                 f"{tuple(A_padded.shape)} != expected "
                                 f"{expected_shape}")
            if A_padded.dtype != cfg.token_dtype:
                raise TypeError(f"external A_padded dtype {A_padded.dtype} != "
                                f"expected {cfg.token_dtype}")
            if A_padded.device != input.device:
                raise ValueError(f"external A_padded device {A_padded.device} != "
                                 f"input device {input.device}")
            if not A_padded.is_contiguous():
                raise ValueError("external A_padded must be contiguous")
            if A_padded.data_ptr() != peer_visible_A.data_ptr():
                raise ValueError("inter-node fused dispatch_group_gemm requires "
                                 "A_padded to alias the peer-visible "
                                 "overlap_context.dispatch_output_buf prefix")

        rdma_rail_send_views = ctx.rdma_rail_send_slot_views(
            input.shape[0],
            cfg.hidden,
            cfg.topk,
            max_slot_num_token=cfg.max_m,
        )
        rdma_rail_send_views["x"].copy_(input)
        rdma_rail_send_views["topk_indices"].copy_(topk_indices)
        if topk_weights is not None:
            rdma_rail_send_views["topk_weights"].copy_(topk_weights)
        if ctx.dispatch_topk_scatter_indices_buf is not None:
            ctx.dispatch_topk_scatter_indices_buf[:dispatch_recv_token_count].fill_(-1)
        ctx.reset_expert_signals()
        self.ep_group_barrier()

        meta_shape = (cfg.nnodes, cfg.max_m, cfg.topk)
        node_topk_indices = torch.empty(
            meta_shape,
            dtype=topk_indices.dtype,
            device=topk_indices.device,
        )
        node_topk_indices.fill_(cfg.num_experts)
        node_topk_send_mask = torch.empty(
            meta_shape,
            dtype=torch.int32,
            device=topk_indices.device,
        )
        node_topk_send_mask.zero_()
        node_token_dst_scatter_indices = torch.empty(
            meta_shape,
            dtype=cfg.offset_dtype,
            device=topk_indices.device,
        )
        node_token_dst_scatter_indices.fill_(-1)

        n_out = _validate_gemm_B(
            B,
            A_padded,
            int(layout_desc.recv_expert_counts.shape[0]),
            op_name="dispatch_group_gemm_inter",
        )
        output = torch.empty(
            (A_padded.shape[0], n_out),
            dtype=cfg.token_dtype,
            device=A_padded.device,
        )
        output_weight = None
        if has_weight:
            output_weight = ctx.dispatch_group_gemm_output_weight_buf[:A_padded.shape[0]]
        num_sm = self._resolve_gemm_num_sm(gemm_num_sm)
        self._dispatch_group_gemm_inter_op.run(
            dev_comm_ptr=ctx.nccl_gin_dev_comm_ptr(),
            A_padded=A_padded,
            B=B,
            recv_expert_counts=layout_desc.recv_expert_counts,
            output=output,
            output_weight=output_weight,
            num_tokens_per_rank=layout_desc.num_tokens_per_rank,
            recv_x_ptrs=ctx.dispatch_output_ptrs,
            recv_weight_ptrs=(ctx.dispatch_group_gemm_output_weight_ptrs if has_weight else None),
            recv_topk_scatter_indices_ptrs=ctx.dispatch_topk_scatter_indices_ptrs,
            node_topk_indices=node_topk_indices,
            node_topk_send_mask=node_topk_send_mask,
            node_token_dst_scatter_indices=node_token_dst_scatter_indices,
            full_splits=ctx.full_splits_buf,
            expert_signals=ctx.expert_signals,
            expert_signal_counters=ctx.expert_signal_counters,
            expert_signal_state_ptrs=ctx.expert_signal_state_ptrs,
            rdma_rail_send_buf=ctx.rdma_rail_send_buf,
            rdma_rail_send_win_handle=ctx.rdma_rail_send_win_handle,
            max_slot_num_token=cfg.max_m,
            max_recv_tokens=int(ctx.dispatch_output_buf.shape[0]),
            experts_per_rank=cfg.num_experts // cfg.world_size,
            local_world_size=cfg.local_world_size,
            dispatch_num_stages=dispatch_num_stages,
            num_sm=num_sm,
            topk=int(topk_indices.shape[1]),
            weight_dtype=cfg.weight_dtype,
            has_weight=has_weight,
        )
        self._reset_dispatch_signals_barrier()
        layout_desc.node_topk_indices = node_topk_indices
        layout_desc.node_topk_send_mask = node_topk_send_mask
        layout_desc.node_token_dst_scatter_indices = node_token_dst_scatter_indices
        layout_desc.topk_indices = node_topk_indices
        layout_desc.token_topk_send_mask = node_topk_send_mask
        layout_desc.token_dst_scatter_indices = node_token_dst_scatter_indices
        if ctx.dispatch_topk_scatter_indices_buf is not None:
            layout_desc.recv_topk_scatter_indices = (ctx.dispatch_topk_scatter_indices_buf[:dispatch_recv_token_count])
        return output, output_weight, layout_desc


def _validate_gemm_B(B: torch.Tensor, A_padded: torch.Tensor, experts_per_rank: int, *, op_name: str) -> int:
    """Validate M-contig group-GEMM B layout, return ``N`` (=output dim).

    Required shape: ``(N, K, num_experts)`` with ``K == A_padded.shape[1]``.
    Required strides: ``(K, 1, N*K)`` -- the natural ``.permute(1, 2, 0)``
    *view* of a row-major ``(E, N, K)`` weight. The kernel bakes a K-major
    TMA descriptor; a contiguous repack silently breaks the GEMM.
    """
    if B.ndim != 3:
        raise ValueError(f"{op_name} requires B to be 3D (N, K, num_experts); "
                         f"got shape {tuple(B.shape)}")
    n_out = int(B.shape[0])
    hidden_in = int(A_padded.shape[1])
    if int(B.shape[1]) != hidden_in:
        raise ValueError(f"{op_name}: B.shape[1] (K={B.shape[1]}) != A_padded.shape[1] "
                         f"({hidden_in})")
    if int(B.shape[2]) != experts_per_rank:
        raise ValueError(f"{op_name}: B.shape[2] (L={B.shape[2]}) != "
                         f"experts_per_rank ({experts_per_rank})")
    n, k, l = B.shape
    expected = (int(k), 1, int(n) * int(k))
    actual = tuple(int(s) for s in B.stride())
    if actual != expected:
        raise ValueError(f"{op_name}: B must be k-major with strides (K, 1, N*K)="
                         f"{expected}; got {actual}. Use ``B_raw.permute(1, 2, 0)`` on a "
                         f"row-major (E, N, K) weight (do NOT ``.contiguous()`` after).")
    return n_out


class _StageEvents:
    """Optional CUDA-event recorder used by ``combine_cutedsl(profile_stages=True)``."""

    __slots__ = ("_enabled", "_events", "_order")

    def __init__(self, enabled: bool):
        self._enabled = enabled
        self._events = {}
        self._order = []

    def record(self, name: str) -> None:
        if not self._enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._events[name] = event
        self._order.append(name)

    def finalize(self, output: torch.Tensor):
        if not self._enabled:
            return output
        self._events[self._order[-1]].synchronize()
        push_ms = self._events["push_start"].elapsed_time(self._events["push_end"])
        barrier_ms = self._events["push_end"].elapsed_time(self._events["barrier_end"])
        reduce_ms = self._events["barrier_end"].elapsed_time(self._events["reduce_end"])
        return output, {
            "push_ms": push_ms,
            "barrier_ms": barrier_ms,
            "reduce_ms": reduce_ms,
        }


__all__ = [
    "EPOverlapKernels",
]
