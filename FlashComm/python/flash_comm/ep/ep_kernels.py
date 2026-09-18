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

import dataclasses
import os
import time

import torch

import flash_comm._C.ep_chunk_plan as _chunk_plan
import flash_comm._C.ep_internode as _ep_inter
import flash_comm._C.ep_intranode as _ep
import flash_comm._C.quantization as _quantization

from .chunk_plan import EPChunkPlan
from .ep_context import EPContext
from .nccl_gin import (
    acquire_nccl_gin,
    release_nccl_gin,
)


@dataclasses.dataclass
class EPCommLayoutDesc:
    # only dependent on local topk_indices, can be computed in advance
    token_within_expert_offset: torch.Tensor | None = None  # [num_tokens, topk]
    expert_counts: torch.Tensor | None = None  # [num_experts + 1]

    # dispatch layout
    recv_base_offset: torch.Tensor | None = None  # [world_size, experts_per_rank, world_size]

    # ------------------------------------------------------------------
    # Send plan (SENDER-side, physical shape [num_token, topk]).
    #
    # These are pure outputs of compute_dispatch_layout: they depend only on the
    # local routing and are indexed by *sender-side* input-token rows. They are
    # the layout's persistent send plan and MUST NOT be overwritten by dispatch.
    #
    # - token_dst_scatter_indices[t, k]: slot index inside the *target rank's*
    #   dispatch receive buffer where local token t's k-th expert-choice lands.
    # - token_topk_send_mask[t, k]: whether local token t's k-th choice is
    #   physically sent (0 when a same-target earlier choice already covers it).
    # - topk_indices[t, k]: the local routing (intranode combine consumes it;
    #   internode combine uses node_topk_indices instead).
    #
    # For a chunk-plan descriptor, the metadata is step-local: only the
    # prefix [0, logical_token_range[1] - logical_token_range[0]) is valid,
    # and range kernels index that prefix relative to the step begin. Standard
    # descriptors have no logical range and consume the whole tensor. Keeping
    # this contract uniform avoids mixing step-local receive metadata with
    # full-input sender metadata in one descriptor.
    #
    # Intranode dispatch/combine consume these directly. Internode dispatch
    # copies token_dst_scatter_indices / token_topk_send_mask into the RDMA rail
    # source slot every call (see dispatch_internode), which is what the NIC
    # actually reads -- the layout kernel no longer stages the RDMA slot.
    #
    # recv_topk_scatter_indices is the RECEIVER-side counterpart ([num_recv_token,
    # topk]); it is filled post-dispatch and relates received rows back to the
    # original token ordering for postprocess/combine.
    token_dst_scatter_indices: torch.Tensor | None = None  # [num_token, topk] sender-side
    token_topk_send_mask: torch.Tensor | None = None  # [num_token, topk] sender-side
    topk_indices: torch.Tensor | None = None  # [num_token, topk] local routing (intranode combine)

    # ------------------------------------------------------------------
    # Per-source-node receive metadata (RECEIVER-side, [nnodes, max_tokens, topk]).
    #
    # Internode-only. These are OUTPUTS of dispatch_internode (the consumer warp
    # writes what it actually applied) and INPUTS to combine_internode. They are
    # a distinct index space from the send plan above -- do not conflate the two.
    node_topk_indices: torch.Tensor | None = None  # [nnodes, max_tokens, topk]
    node_topk_send_mask: torch.Tensor | None = None  # [nnodes, max_tokens, topk]
    node_token_dst_scatter_indices: torch.Tensor | None = None  # [nnodes, max_tokens, topk]

    recv_token_count_cpu: torch.Tensor | None = None  # [world_size] CPU snapshot (layout-private, unaligned)
    recv_token_count: torch.Tensor | None = None  # [world_size] device memory (unaligned)
    # Internode-only: per-rank source token counts written by compute_dispatch_layout.
    # This is layout data for the current dispatch, not persistent communication state.
    num_tokens_per_rank: torch.Tensor | None = None  # [world_size] device memory
    recv_aligned_token_count_cpu: torch.Tensor | None = None  # [world_size] pinned CPU (aligned, for buffer alloc)
    recv_aligned_token_count: torch.Tensor | None = None  # [world_size] device (aligned, for postprocess/combine)
    recv_expert_counts: torch.Tensor | None = None  # [experts_per_rank] per-expert actual token counts
    expert_alignment: int = 1
    num_tokens: int = -1
    recv_topk_scatter_indices: torch.Tensor | None = None  # [num_recv_token, topk] receiver-side
    # Optional receiver-view metadata filled by compute_dispatch_layout for
    # CuTeDSL pull dispatch / push combine overlap paths.
    token_src_rank_topk_and_indices: torch.Tensor | None = None  # [num_recv_token] int64
    # Internode-only sender-view metadata. Inter-node CuTeDSL dispatch may
    # replace token_topk_send_mask/token_dst_scatter_indices with receiver
    # node-view tensors, but cached layout reuse still needs the original
    # sender-space tensors to restage the RDMA source slot.
    internode_sender_topk_send_mask: torch.Tensor | None = None  # [num_tokens, topk]
    internode_sender_token_dst_scatter_indices: torch.Tensor | None = None  # [num_tokens, topk]
    # Chunk-plan-only active input rows [begin, end). Standard EP leaves this
    # unset and processes the whole input.
    logical_token_range: torch.Tensor | None = None

    def check_combine_required_inputs(self):
        # Intranode combine consumes the sender-side send plan directly.
        if self.token_topk_send_mask is None or self.token_dst_scatter_indices is None:
            raise ValueError("token_topk_send_mask and token_dst_scatter_indices must be provided")

    def check_internode_combine_required_inputs(self):
        # Internode combine consumes the per-source-node receive metadata produced
        # by the preceding dispatch_internode, not the sender-side send plan.
        if (self.node_topk_indices is None or self.node_topk_send_mask is None
                or self.node_token_dst_scatter_indices is None):
            raise ValueError(
                "node_topk_indices / node_topk_send_mask / node_token_dst_scatter_indices must be provided; "
                "run dispatch_internode before combine_internode")

    def need_recompute_token_within_expert_offset_and_expert_counts(self, topk_indices: torch.Tensor | None = None):
        if self.token_within_expert_offset is None or self.expert_counts is None:
            return True
        return bool(topk_indices is not None
                    and tuple(self.token_within_expert_offset.shape) != tuple(topk_indices.shape))

    def need_recompute_dispatch_layout(self, expert_alignment: int = 1, num_tokens: int = -1):
        if (self.recv_base_offset is None or self.token_dst_scatter_indices is None or self.token_topk_send_mask is None
                or self.recv_token_count_cpu is None or self.recv_token_count is None
                or self.recv_expert_counts is None):
            return True
        if self.expert_alignment != expert_alignment:
            return True
        if num_tokens >= 0 and self.num_tokens != num_tokens:
            return True
        return (expert_alignment > 1
                and (self.recv_aligned_token_count_cpu is None or self.recv_aligned_token_count is None))

    def check_layout_desc(
        self,
        num_tokens: int,
        topk: int,
        num_experts: int,
        world_size: int = 1,
        local_world_size: int | None = None,
        max_slot_num_token: int = -1,
    ):
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if num_experts <= 0 or num_experts % world_size != 0:
            raise ValueError(f"num_experts must be positive and divisible by world_size, got "
                             f"num_experts={num_experts}, world_size={world_size}")
        if local_world_size is None:
            local_world_size = world_size
        if local_world_size <= 0 or world_size % local_world_size != 0:
            raise ValueError(f"local_world_size must divide world_size, got world_size={world_size}, "
                             f"local_world_size={local_world_size}")
        nnodes = world_size // local_world_size
        layout_num_tokens = self.num_tokens if self.num_tokens >= 0 else num_tokens

        def internode_metadata_shape(name: str):
            if max_slot_num_token <= 0:
                raise ValueError(f"max_slot_num_token must be positive when validating rank-3 {name}, "
                                 f"got {max_slot_num_token}")
            return (nnodes, max_slot_num_token, topk)

        def check_shape(tensor: torch.Tensor | None, name: str, expected_shape):
            if tensor is not None and tuple(tensor.shape) != tuple(expected_shape):
                raise ValueError(f"{name} must have shape {list(expected_shape)}, got shape {tuple(tensor.shape)}")

        def check_node_metadata_shape(tensor: torch.Tensor | None, name: str):
            # Per-source-node receive metadata is internode-only and always rank-3.
            if tensor is None:
                return
            if nnodes <= 1:
                raise ValueError(f"{name} is internode-only metadata but nnodes={nnodes}; "
                                 f"got shape {tuple(tensor.shape)}")
            expected = internode_metadata_shape(name)
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape {list(expected)} for internode layout, "
                                 f"got shape {tuple(tensor.shape)}")

        def check_recv_topk_scatter_shape(tensor: torch.Tensor | None, name: str):
            if tensor is None:
                return
            if tensor.dim() == 2:
                if tensor.shape[1] != topk:
                    raise ValueError(f"{name} second dim must be topk={topk}, got shape {tuple(tensor.shape)}")
            elif tensor.dim() == 3:
                if nnodes <= 1:
                    raise ValueError(f"{name} must be rank-2 for intranode layout, got shape {tuple(tensor.shape)}")
                expected = internode_metadata_shape(name)
                if tuple(tensor.shape) != expected:
                    raise ValueError(f"{name} must have shape {list(expected)} for internode layout, "
                                     f"got shape {tuple(tensor.shape)}")
            else:
                raise ValueError(f"{name} must be rank-2 or rank-3, got shape {tuple(tensor.shape)}")

        has_token_offset = self.token_within_expert_offset is not None
        has_expert_counts = self.expert_counts is not None
        if has_expert_counts != has_token_offset:
            raise ValueError("token_within_expert_offset and expert_counts must both be None or both be non-None")

        experts_per_rank = num_experts // world_size
        check_shape(self.token_within_expert_offset, "token_within_expert_offset", (layout_num_tokens, topk))
        check_shape(self.expert_counts, "expert_counts", (num_experts + 1, ))
        check_shape(self.recv_base_offset, "recv_base_offset", (world_size, experts_per_rank, world_size))
        # Sender-side send plan: always [num_token, topk].
        check_shape(self.token_dst_scatter_indices, "token_dst_scatter_indices", (layout_num_tokens, topk))
        check_shape(self.token_topk_send_mask, "token_topk_send_mask", (layout_num_tokens, topk))
        check_shape(self.topk_indices, "topk_indices", (layout_num_tokens, topk))
        # Per-source-node receive metadata: internode-only, [nnodes, max_slot_num_token, topk].
        check_node_metadata_shape(self.node_topk_indices, "node_topk_indices")
        check_node_metadata_shape(self.node_topk_send_mask, "node_topk_send_mask")
        check_node_metadata_shape(self.node_token_dst_scatter_indices, "node_token_dst_scatter_indices")
        check_shape(self.recv_token_count_cpu, "recv_token_count_cpu", (world_size, ))
        check_shape(self.recv_token_count, "recv_token_count", (world_size, ))
        check_shape(self.recv_aligned_token_count_cpu, "recv_aligned_token_count_cpu", (world_size, ))
        check_shape(self.recv_aligned_token_count, "recv_aligned_token_count", (world_size, ))
        check_shape(self.recv_expert_counts, "recv_expert_counts", (experts_per_rank, ))
        check_shape(self.num_tokens_per_rank, "num_tokens_per_rank", (world_size, ))
        if self.internode_sender_topk_send_mask is not None:
            check_shape(self.internode_sender_topk_send_mask, "internode_sender_topk_send_mask",
                        (layout_num_tokens, topk))
        if self.internode_sender_token_dst_scatter_indices is not None:
            check_shape(self.internode_sender_token_dst_scatter_indices, "internode_sender_token_dst_scatter_indices",
                        (layout_num_tokens, topk))
        check_recv_topk_scatter_shape(self.recv_topk_scatter_indices, "recv_topk_scatter_indices")


class EPKernels:
    """
    FlashComm EP kernels.

    Internode EP holds a lease on the process-global NCCL GIN communicator and
    owns one RDMA rail buffer per EPKernels instance. Dispatch/combine calls
    must be serialized; overlapping calls would race on those protocol slots.
    """

    def __init__(self, max_m: int, hidden: int, topk: int, num_experts: int, local_world_size: int,
                 ep_group: torch.distributed.ProcessGroup, num_sm: int = 16, capacity: float = 1.2,
                 num_worst_tokens: int = -1, expert_alignment: int = 1, check_num_worst_tokens: bool = False):
        self.ep_group = ep_group
        self.num_sm = num_sm
        self.alignment = 1024
        self.coeff = capacity  # for output buffer reallocation
        self.cpu_default_val = -1
        cpu_poll_sleep_us = int(os.environ.get("FLASH_COMM_EP_CPU_POLL_SLEEP_US", "50"))
        assert cpu_poll_sleep_us >= 0, f"cpu_poll_sleep_us must be >= 0, got {cpu_poll_sleep_us}"
        self.cpu_poll_sleep_us = cpu_poll_sleep_us
        self.cpu_poll_sleep_s = cpu_poll_sleep_us / 1_000_000
        self.rank = ep_group.rank()
        self.world_size = ep_group.size()
        self.local_world_size = local_world_size
        self.num_worst_tokens = num_worst_tokens
        assert expert_alignment >= 1, f"expert_alignment must be >= 1, got {expert_alignment}"
        self.expert_alignment = expert_alignment
        self.check_num_worst_tokens = check_num_worst_tokens

        self.is_internode = self.world_size > local_world_size
        self._has_nccl_gin_lease = False
        if self.is_internode:
            self._init_internode_nccl_gin()

        self.ep_context = EPContext.create(
            max_m=max_m,
            hidden=hidden,
            topk=topk,
            num_experts=num_experts,
            group=ep_group,
            local_world_size=local_world_size,
            capacity_coeff=capacity,
            num_worst_tokens=num_worst_tokens,
        )
        torch.distributed.barrier(group=ep_group)

    def _init_internode_nccl_gin(self) -> None:
        self._dispatch_signal_range, self._combine_signal_range = acquire_nccl_gin(self.ep_group, self.local_world_size,
                                                                                   min_rail_barriers=self.num_sm,
                                                                                   require_gin=True)
        self._has_nccl_gin_lease = True

    def finalize(self) -> None:
        ctx = getattr(self, "ep_context", None)
        if self.is_internode:
            if ctx is not None:
                self.ep_context.release_internode_nccl_resources()
            if self._has_nccl_gin_lease:
                release_nccl_gin(self.ep_group)
            self._has_nccl_gin_lease = False
        # Coordinated, leak-free release of the symmetric buffers. Critical for the
        # torch_ipc backend so producer storage is not left in torch's CUDA IPC
        # limbo across EPKernels re-creation. Must be called collectively by every
        # rank -- this is why teardown is an explicit finalize() and not __del__.
        if ctx is not None:
            ctx.free_buffers()

    def _realloc_dispatch_output_buf(self, recv_token_count_cpu: torch.Tensor, recv_token_count: torch.Tensor):
        assert recv_token_count_cpu.is_cpu
        assert recv_token_count_cpu.dtype == torch.int32
        max_output_token_num = 0

        if self.num_worst_tokens > 0 and self.num_worst_tokens <= self.ep_context.dispatch_output_buf.shape[0]:
            if self.check_num_worst_tokens:
                torch._assert_async(recv_token_count[self.rank] <= self.num_worst_tokens,
                                    f"num_worst_tokens = {self.num_worst_tokens} is not valid")
            return self.num_worst_tokens, self.num_worst_tokens

        # for target_rank in range(self.ep_context.config.world_size):
        #     # slice and item operations of the tensor are too time-consuming (10us level), so here we read directly from ptr
        #     while ctypes.c_int32.from_address(base_ptr + target_rank * elem_size).value == self.cpu_default_val:
        #         pass
        #     cur_output_token_num = ctypes.c_int32.from_address(base_ptr + target_rank * elem_size).value
        #     max_output_token_num = max(max_output_token_num, cur_output_token_num)
        arr = recv_token_count_cpu.numpy()
        while int(arr.min()) == self.cpu_default_val:
            if self.cpu_poll_sleep_us > 0:
                time.sleep(self.cpu_poll_sleep_s)
        max_output_token_num = int(arr.max())
        cur_output_token_num = int(arr[self.rank])
        if max_output_token_num > self.ep_context.dispatch_output_buf.shape[0]:
            self.ep_group_barrier()
            torch.cuda.synchronize()

            alloc_token = int(
                (max_output_token_num + self.alignment - 1) // self.alignment * self.alignment * self.coeff)
            print(
                f"reallocate dispatch output buf from {self.ep_context.dispatch_output_buf.shape[0]} to {alloc_token}")
            self.ep_context.reallocate_buffers(alloc_token)
            self.ep_group_barrier()
            torch.cuda.synchronize()

        return cur_output_token_num, max_output_token_num

    def _prepare_intranode_dispatch(self, topk_indices: torch.Tensor, layout_desc: EPCommLayoutDesc,
                                    num_tokens: int) -> int:
        self.ep_group_barrier()
        if layout_desc.need_recompute_token_within_expert_offset_and_expert_counts(topk_indices):
            layout_desc.token_within_expert_offset, layout_desc.expert_counts = \
                self.compute_stable_local_token_within_expert_offset_and_expert_counts(topk_indices, self.num_sm)

        if layout_desc.need_recompute_dispatch_layout(self.expert_alignment, num_tokens):
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
                self.ep_context.full_splits_buf_ptrs,
                self.ep_context.nvl_barrier_buf_ptrs,
                self.ep_context.config.num_experts,
                self.ep_context.config.rank,
                self.ep_context.config.world_size,
                self.num_sm,
                token_src_rank_topk_and_indices_ptrs=None,
                expert_alignment=self.expert_alignment,
            )
            layout_desc.expert_alignment = self.expert_alignment
            layout_desc.num_tokens = num_tokens

        self.ep_group_barrier()
        if layout_desc.expert_alignment > 1:
            assert layout_desc.recv_aligned_token_count_cpu is not None, \
                "recv_aligned_token_count_cpu must be set when expert_alignment > 1"
            buf_count_cpu = layout_desc.recv_aligned_token_count_cpu
            buf_count_gpu = layout_desc.recv_aligned_token_count
        else:
            buf_count_cpu = layout_desc.recv_token_count_cpu
            buf_count_gpu = layout_desc.recv_token_count
        dispatch_recv_token_count, _ = self._realloc_dispatch_output_buf(buf_count_cpu, buf_count_gpu)
        return dispatch_recv_token_count

    def _finish_intranode_dispatch(self, topk_indices: torch.Tensor, topk_weights: torch.Tensor | None,
                                   layout_desc: EPCommLayoutDesc, dispatch_recv_token_count: int):
        self.ep_group_barrier()
        layout_desc.topk_indices = topk_indices
        layout_desc.recv_topk_scatter_indices = self.ep_context.dispatch_topk_scatter_indices_buf[:
                                                                                                  dispatch_recv_token_count]
        if topk_weights is None:
            return None
        return self.ep_context.dispatch_topk_weights_buf[:dispatch_recv_token_count]

    def dispatch_intranode(self, input: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor | None,
                           layout_desc: EPCommLayoutDesc):
        dispatch_recv_token_count = self._prepare_intranode_dispatch(topk_indices, layout_desc, input.shape[0])

        num_experts_per_rank = self.ep_context.config.num_experts // self.ep_context.config.world_size
        # dispatch
        # Note: if topk_weights is None, the kernel skips weight dispatch and the weight buffer contents are stale.
        _ep.dispatch_intranode(input, layout_desc.token_topk_send_mask, topk_weights, topk_indices,
                               layout_desc.token_dst_scatter_indices, self.ep_context.dispatch_output_buf_ptrs,
                               self.ep_context.dispatch_topk_weights_buf_ptrs,
                               self.ep_context.dispatch_topk_scatter_indices_buf_ptrs, self.rank, self.world_size,
                               num_experts_per_rank, self.num_sm)

        dispatch_weights = self._finish_intranode_dispatch(topk_indices, topk_weights, layout_desc,
                                                           dispatch_recv_token_count)
        return (self.ep_context.dispatch_output_buf[:dispatch_recv_token_count], dispatch_weights, layout_desc)

    def _dispatch_mxfp8_intranode(self, input: torch.Tensor, topk_indices: torch.Tensor,
                                  topk_weights: torch.Tensor | None, layout_desc: EPCommLayoutDesc, prequantized: bool):
        if input.ndim != 2 or not input.is_cuda or not input.is_contiguous():
            raise ValueError("MXFP8 dispatch input must be a contiguous CUDA 2D tensor")
        if topk_indices.ndim != 2 or topk_indices.shape[0] != input.shape[0]:
            raise ValueError("topk_indices must be 2D with the same number of rows as input")
        if topk_weights is not None and tuple(topk_weights.shape) != tuple(topk_indices.shape):
            raise ValueError("topk_weights must have the same shape as topk_indices")
        if input.dtype != torch.bfloat16:
            raise ValueError(f"MXFP8 dispatch requires BF16 input, got {input.dtype}")
        hidden = self.ep_context.config.hidden
        if hidden % 32 != 0:
            raise ValueError(f"MXFP8 hidden size must be divisible by 32, got {hidden}")

        packed_row_bytes = int(_quantization.mxfp8_packed_row_bytes(hidden))
        expected_width = packed_row_bytes // 2 if prequantized else hidden
        if input.shape[1] != expected_width:
            raise ValueError(f"MXFP8 dispatch input width mismatch, got {input.shape[1]}, expected {expected_width}")

        dispatch_recv_token_count = self._prepare_intranode_dispatch(topk_indices, layout_desc, input.shape[0])
        num_experts_per_rank = self.ep_context.config.num_experts // self.ep_context.config.world_size
        if prequantized:
            _ep.dispatch_mxfp8_prequantized_intranode(
                input,
                layout_desc.token_topk_send_mask,
                topk_weights,
                topk_indices,
                layout_desc.token_dst_scatter_indices,
                self.ep_context.dispatch_output_buf_ptrs,
                self.ep_context.dispatch_topk_weights_buf_ptrs,
                self.ep_context.dispatch_topk_scatter_indices_buf_ptrs,
                hidden,
                self.rank,
                self.world_size,
                num_experts_per_rank,
                self.num_sm,
            )
        else:
            _ep.dispatch_mxfp8_quant_fused_intranode(
                input,
                layout_desc.token_topk_send_mask,
                topk_weights,
                topk_indices,
                layout_desc.token_dst_scatter_indices,
                self.ep_context.dispatch_output_buf_ptrs,
                self.ep_context.dispatch_topk_weights_buf_ptrs,
                self.ep_context.dispatch_topk_scatter_indices_buf_ptrs,
                self.rank,
                self.world_size,
                num_experts_per_rank,
                self.num_sm,
            )

        dispatch_weights = self._finish_intranode_dispatch(topk_indices, topk_weights, layout_desc,
                                                           dispatch_recv_token_count)

        packed_storage = self.ep_context.dispatch_output_buf.view(torch.uint8).reshape(-1)
        packed_num_bytes = dispatch_recv_token_count * packed_row_bytes
        if packed_num_bytes > packed_storage.numel():
            raise RuntimeError("BF16 dispatch context is too small for the packed MXFP8 receive rows")
        packed_out = packed_storage[:packed_num_bytes].view(torch.bfloat16).reshape(dispatch_recv_token_count,
                                                                                    packed_row_bytes // 2)
        return packed_out, dispatch_weights, layout_desc

    def dispatch_mxfp8_intranode_postprocess(self, packed_out: torch.Tensor, dispatch_topk_weights: torch.Tensor | None,
                                             layout_desc: EPCommLayoutDesc, num_sm: int = 0, *,
                                             dispatch_weights_out: torch.Tensor | None = None):
        if layout_desc.recv_topk_scatter_indices is None:
            raise ValueError("layout_desc.recv_topk_scatter_indices is required for MXFP8 postprocess")
        hidden = self.ep_context.config.hidden
        topk = self.ep_context.config.topk
        packed_row_bytes = int(_quantization.mxfp8_packed_row_bytes(hidden))
        if tuple(packed_out.shape) != (layout_desc.recv_topk_scatter_indices.shape[0], packed_row_bytes // 2):
            raise ValueError(f"packed_out has invalid shape {tuple(packed_out.shape)}")

        buffer_size = packed_out.shape[0]
        dispatch_data = torch.empty((buffer_size, hidden), dtype=torch.float8_e4m3fn, device=packed_out.device)
        dispatch_scales = torch.empty((buffer_size, hidden // 32), dtype=torch.uint8, device=packed_out.device)
        recv_topk_scatter_indices = torch.empty((buffer_size, topk), dtype=torch.int32, device=packed_out.device)
        dispatch_weights_result = None
        if dispatch_topk_weights is not None:
            if dispatch_weights_out is None:
                dispatch_weights_result = torch.empty((buffer_size, ), dtype=dispatch_topk_weights.dtype,
                                                      device=packed_out.device)
            else:
                dispatch_weights_result = dispatch_weights_out
                if (tuple(dispatch_weights_result.shape) != (buffer_size, )
                        or dispatch_weights_result.dtype != dispatch_topk_weights.dtype
                        or dispatch_weights_result.device != dispatch_topk_weights.device
                        or not dispatch_weights_result.is_contiguous()):
                    raise ValueError(
                        "dispatch_weights_out must match the MXFP8 dispatch buffer shape, dtype, and device")
        elif dispatch_weights_out is not None:
            raise ValueError("dispatch_weights_out requires dispatch_topk_weights")
        if layout_desc.expert_alignment > 1:
            if layout_desc.recv_aligned_token_count is None:
                raise ValueError("expert_alignment > 1 requires aligned receive token counts")
            postprocess_token_count = layout_desc.recv_aligned_token_count
        else:
            if layout_desc.recv_token_count is None:
                raise ValueError("layout_desc.recv_token_count is required for MXFP8 postprocess")
            postprocess_token_count = layout_desc.recv_token_count

        _ep.dispatch_mxfp8_postprocess_unpack(
            packed_out,
            layout_desc.recv_topk_scatter_indices,
            dispatch_topk_weights,
            postprocess_token_count,
            dispatch_data,
            dispatch_scales,
            dispatch_weights_result,
            recv_topk_scatter_indices,
            hidden,
            topk,
            self.rank,
            self.world_size,
            num_sm,
        )

        layout_desc.recv_topk_scatter_indices = recv_topk_scatter_indices
        return dispatch_data, dispatch_scales, dispatch_weights_result, layout_desc

    def dispatch_intranode_postprocess(self, dispatch_out: torch.Tensor, dispatch_topk_weights: torch.Tensor | None,
                                       layout_desc: EPCommLayoutDesc, num_sm: int = 0,
                                       recv_topk_scatter_indices_out: torch.Tensor | None = None,
                                       dispatch_weights_out: torch.Tensor | None = None):
        if num_sm <= 0:
            num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count * 8

        topk = self.ep_context.config.topk
        hidden = self.ep_context.config.hidden
        if dispatch_topk_weights is not None:
            assert dispatch_out.shape[0] == dispatch_topk_weights.shape[0]
        assert dispatch_out.shape[0] == layout_desc.recv_topk_scatter_indices.shape[0]
        assert dispatch_out.shape[1] == hidden
        assert layout_desc.recv_topk_scatter_indices.shape[1] == topk
        if dispatch_topk_weights is not None:
            assert dispatch_topk_weights.shape[1] == topk
        assert layout_desc.recv_token_count is not None

        # postprocess:
        # 1. local dispatch(inplace)
        # 2. copy recv_topk_scatter_indices from symm tensor to torch tensor
        # 3. get dispatch weights
        buffer_size = dispatch_out.shape[0]
        if recv_topk_scatter_indices_out is None:
            recv_topk_scatter_indices = torch.empty(
                (buffer_size, topk),
                dtype=layout_desc.recv_topk_scatter_indices.dtype,
                device=layout_desc.recv_topk_scatter_indices.device,
            )
        else:
            recv_topk_scatter_indices = recv_topk_scatter_indices_out
            if (tuple(recv_topk_scatter_indices.shape) != (buffer_size, topk)
                    or recv_topk_scatter_indices.dtype != layout_desc.recv_topk_scatter_indices.dtype
                    or recv_topk_scatter_indices.device != layout_desc.recv_topk_scatter_indices.device
                    or not recv_topk_scatter_indices.is_contiguous()):
                raise ValueError(
                    "recv_topk_scatter_indices_out must match the dispatch buffer shape, dtype, and device")
        dispatch_weights_result = None
        if dispatch_topk_weights is not None:
            if dispatch_weights_out is None:
                dispatch_weights_result = torch.empty(
                    (buffer_size, ),
                    dtype=dispatch_topk_weights.dtype,
                    device=dispatch_topk_weights.device,
                )
            else:
                dispatch_weights_result = dispatch_weights_out
                if (tuple(dispatch_weights_result.shape) != (buffer_size, )
                        or dispatch_weights_result.dtype != dispatch_topk_weights.dtype
                        or dispatch_weights_result.device != dispatch_topk_weights.device
                        or not dispatch_weights_result.is_contiguous()):
                    raise ValueError("dispatch_weights_out must match the dispatch buffer shape, dtype, and device")
        elif dispatch_weights_out is not None:
            raise ValueError("dispatch_weights_out requires dispatch_topk_weights")
        self._buffer_in_range(layout_desc.recv_topk_scatter_indices, self.ep_context.dispatch_topk_scatter_indices_buf)
        if layout_desc.expert_alignment > 1:
            assert layout_desc.recv_aligned_token_count is not None, \
                "recv_aligned_token_count must be set when expert_alignment > 1"
            postprocess_token_count = layout_desc.recv_aligned_token_count
        else:
            postprocess_token_count = layout_desc.recv_token_count
        _ep.dispatch_postprocess(
            dispatch_out,
            layout_desc.recv_topk_scatter_indices,  # comm buffer
            dispatch_topk_weights,
            postprocess_token_count,
            dispatch_weights_result,
            recv_topk_scatter_indices,  # torch tensor
            hidden,
            topk,
            self.rank,
            self.world_size,
            num_sm,
        )

        # update layout desc
        layout_desc.recv_topk_scatter_indices = recv_topk_scatter_indices
        return dispatch_out, dispatch_weights_result, layout_desc

    def ep_group_barrier(self):
        if self.ep_context.config.nnodes == 1:
            assert self.ep_context.nvl_barrier_buf.dtype == torch.int32
            _ep.barrier_all_on_stream(self.ep_context.nvl_barrier_buf_ptrs, self.rank, self.world_size)
        else:
            _ep_inter.barrier_all_on_stream()

    def compute_stable_local_token_within_expert_offset_and_expert_counts(self, topk_indices: torch.Tensor, num_sm=-1):
        token_within_expert_offset, _block_cumsum_hist, expert_counts = _ep.compute_stable_local_token_within_expert_offset_and_expert_counts(
            topk_indices, self.ep_context.config.num_experts, num_sm)
        return token_within_expert_offset, expert_counts

    def prepare_chunk_layouts(self, topk_indices: torch.Tensor, token_within_expert_offset: torch.Tensor,
                              plan: EPChunkPlan) -> tuple[EPCommLayoutDesc, ...]:
        """Build every step layout with one local device kernel."""
        if topk_indices.dim() != 2:
            raise ValueError("topk_indices must be rank-2")
        num_tokens, topk = topk_indices.shape
        capacity = self.ep_context.dispatch_output_buf.shape[0]
        if self.num_worst_tokens <= 0:
            raise ValueError("range EP requires a fixed receive capacity")
        if plan.world_size != self.world_size:
            raise ValueError("chunk plan and EPKernels world size differ")
        if plan.recv_capacity_tokens != capacity:
            raise ValueError("chunk plan and EPKernels receive capacity differ")
        if plan.expert_alignment != self.expert_alignment:
            raise ValueError("chunk plan and EPKernels expert alignment differ")
        if plan.topk != topk or topk != self.ep_context.config.topk:
            raise ValueError("chunk plan and EPKernels topk differ")
        if plan.num_experts != self.ep_context.config.num_experts:
            raise ValueError("chunk plan and EPKernels expert count differ")
        if num_tokens > plan.max_num_tokens:
            raise ValueError("local token count exceeds the chunk plan specialization")
        if (plan.logical_token_ranges.device != topk_indices.device
                or plan.rank_chunk_prefix.device != topk_indices.device):
            raise ValueError("chunk plan and routing tensors must be on the same CUDA device")
        if (topk_indices.dtype != torch.int32 or token_within_expert_offset.dtype != torch.int32
                or not topk_indices.is_cuda or not token_within_expert_offset.is_cuda
                or not topk_indices.is_contiguous() or not token_within_expert_offset.is_contiguous()
                or topk_indices.shape != token_within_expert_offset.shape):
            raise ValueError("routing and token offsets must be matching contiguous CUDA int32 tensors")

        num_steps = plan.max_steps
        int_opts = {"dtype": torch.int32, "device": topk_indices.device}
        experts_per_rank = self.ep_context.config.num_experts // self.world_size
        recv_base_offset = torch.empty((num_steps, self.world_size, experts_per_rank, self.world_size), **int_opts)
        # Each step owns a step-local sender route plan.  The physical tensor
        # keeps the full input shape for descriptor compatibility, while only
        # [0, end - begin) is valid for that step's logical range.
        # Keep this O(Q*M*K) layout for consistency; revisit O(M*K) storage
        # only if extreme metadata memory becomes material.
        token_dst = torch.empty((num_steps, num_tokens, topk), **int_opts)
        send_mask = torch.empty_like(token_dst)
        recv_token_count = torch.empty((num_steps, self.world_size), **int_opts)
        recv_aligned_token_count = torch.empty_like(recv_token_count)
        recv_expert_counts = torch.empty((num_steps, experts_per_rank), **int_opts)
        num_tokens_per_rank = None
        node_topk_indices = None
        node_topk_send_mask = None
        node_token_dst_scatter_indices = None
        if self.is_internode:
            num_tokens_per_rank = torch.empty_like(recv_token_count)
            node_shape = (num_steps, self.ep_context.config.nnodes, self.ep_context.config.max_m, topk)
            node_topk_indices = torch.empty(node_shape, **int_opts)
            node_topk_send_mask = torch.empty_like(node_topk_indices)
            node_token_dst_scatter_indices = torch.empty_like(node_topk_indices)
        _chunk_plan.build_ep_chunk_layouts_out(
            topk_indices,
            token_within_expert_offset,
            plan.logical_token_ranges,
            plan.rank_chunk_prefix,
            plan.chunk_size,
            plan.expert_alignment,
            recv_base_offset,
            token_dst,
            send_mask,
            recv_token_count,
            recv_aligned_token_count,
            recv_expert_counts,
            num_tokens_per_rank,
        )
        if self.check_num_worst_tokens:
            capacity_count = (recv_aligned_token_count if self.expert_alignment > 1 else recv_token_count)
            torch._assert_async(
                torch.all(capacity_count[:, self.rank] <= self.num_worst_tokens),
                f"num_worst_tokens = {self.num_worst_tokens} is not valid",
            )
        return tuple(
            EPCommLayoutDesc(
                recv_base_offset=recv_base_offset[step],
                token_dst_scatter_indices=token_dst[step],
                token_topk_send_mask=send_mask[step],
                topk_indices=topk_indices,
                recv_token_count=recv_token_count[step],
                recv_aligned_token_count=recv_aligned_token_count[step],
                recv_expert_counts=recv_expert_counts[step],
                expert_alignment=self.expert_alignment,
                num_tokens=num_tokens,
                num_tokens_per_rank=(num_tokens_per_rank[step] if num_tokens_per_rank is not None else None),
                node_topk_indices=(node_topk_indices[step] if node_topk_indices is not None else None),
                node_topk_send_mask=(node_topk_send_mask[step] if node_topk_send_mask is not None else None),
                node_token_dst_scatter_indices=(
                    node_token_dst_scatter_indices[step] if node_token_dst_scatter_indices is not None else None),
                logical_token_range=plan.logical_token_ranges[step],
            ) for step in range(num_steps))

    def _intranode_range_barrier(self, logical_token_range: torch.Tensor) -> None:
        _ep.barrier_all_on_stream_range(
            self.ep_context.nvl_barrier_buf_ptrs,
            self.rank,
            self.world_size,
            logical_token_range,
        )

    def _range_group_barrier(self, logical_token_range: torch.Tensor) -> None:
        """Synchronize one active chunk step across the EP group.

        Chunk plans publish the same logical range on every rank.  Keeping the
        active check inside the backend avoids paying a barrier for the fixed
        trailing zero steps while preserving the same call sequence for the
        active steps on both intra- and inter-node EP.
        """
        if self.is_internode:
            _ep_inter.barrier_all_on_stream_if_active(logical_token_range)
        else:
            self._intranode_range_barrier(logical_token_range)

    def _validate_range_layout(self, topk_indices: torch.Tensor, layout: EPCommLayoutDesc) -> None:
        if (layout.logical_token_range is None or tuple(layout.logical_token_range.shape) != (2, )):
            raise ValueError("prepared range layout must contain logical_token_range[2]")
        if (layout.token_dst_scatter_indices is None or layout.token_topk_send_mask is None
                or layout.recv_token_count is None or layout.recv_aligned_token_count is None
                or layout.recv_expert_counts is None):
            raise ValueError("prepared range layout is incomplete")
        if layout.token_dst_scatter_indices.shape != topk_indices.shape:
            raise ValueError("prepared range layout routing shape mismatch")
        if layout.recv_token_count.numel() != self.world_size:
            raise ValueError("prepared range layout world-size mismatch")

    def _dispatch_range(self, input: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor | None,
                        layout: EPCommLayoutDesc, num_qps: int | None = None):
        self._validate_range_layout(topk_indices, layout)
        logical_token_range = layout.logical_token_range
        experts_per_rank = self.ep_context.config.num_experts // self.world_size
        if (self.is_internode
                and (layout.num_tokens_per_rank is None or layout.node_topk_indices is None
                     or layout.node_topk_send_mask is None or layout.node_token_dst_scatter_indices is None)):
            raise ValueError("prepared internode range layout is incomplete")
        # Retire consumers of the previous range's symmetric dispatch output
        # before this rank starts the next range dispatch.  The final dispatch
        # signal reset protects the GIN protocol, but it does not cover a
        # recompute path that skipped dispatch postprocess and still has a peer
        # reading the previous output buffer.
        self._range_group_barrier(logical_token_range)
        if self.is_internode:
            _ep_inter.stage_dispatch_internode_range(
                input,
                topk_indices,
                layout.token_topk_send_mask,
                layout.token_dst_scatter_indices,
                topk_weights,
                self.ep_context.rdma_rail_send_buf,
                self.ep_context.rdma_rail_send_win_handle,
                self.ep_context.config.max_m,
                logical_token_range,
            )
            self._range_group_barrier(logical_token_range)
            _ep_inter.dispatch_internode_range(
                self.ep_context.dispatch_output_buf_ptrs,
                self.ep_context.dispatch_topk_weights_buf_ptrs,
                self.ep_context.dispatch_topk_scatter_indices_buf_ptrs,
                self.ep_context.dispatch_output_buf.shape[0],
                self.ep_context.rdma_rail_send_buf,
                self.ep_context.rdma_rail_send_win_handle,
                layout.num_tokens_per_rank,
                layout.node_topk_indices,
                layout.node_topk_send_mask,
                layout.node_token_dst_scatter_indices,
                self.ep_context.config.max_m,
                input.shape[0],
                self.ep_context.config.hidden,
                topk_weights is not None,
                experts_per_rank,
                self.num_sm,
                num_qps,
                logical_token_range,
            )
            _ep_inter.reset_signals_barrier_all_on_stream_if_active(*self._dispatch_signal_range, logical_token_range)
        else:
            _ep.dispatch_intranode_range(
                input,
                layout.token_topk_send_mask,
                topk_weights,
                topk_indices,
                layout.token_dst_scatter_indices,
                logical_token_range,
                self.ep_context.dispatch_output_buf_ptrs,
                self.ep_context.dispatch_topk_weights_buf_ptrs,
                self.ep_context.dispatch_topk_scatter_indices_buf_ptrs,
                self.rank,
                self.world_size,
                experts_per_rank,
                self.num_sm,
            )
            self._range_group_barrier(logical_token_range)
        layout.topk_indices = topk_indices
        layout.recv_topk_scatter_indices = (self.ep_context.dispatch_topk_scatter_indices_buf)
        dispatch_weights = (self.ep_context.dispatch_topk_weights_buf if topk_weights is not None else None)
        return self.ep_context.dispatch_output_buf, dispatch_weights, layout

    def _combine_range(self, input_preprocessed: torch.Tensor, layout: EPCommLayoutDesc, output: torch.Tensor,
                       weight_preprocessed: torch.Tensor | None = None, output_weight: torch.Tensor | None = None,
                       num_qps: int | None = None):
        if layout.topk_indices is None:
            raise ValueError("range layout requires topk_indices from dispatch")
        self._validate_range_layout(layout.topk_indices, layout)
        if not self._validate_combine_input_buffer(input_preprocessed):
            raise ValueError("range combine input must use the EP combine buffer")
        if (weight_preprocessed is None) != (output_weight is None):
            raise ValueError("weight_preprocessed and output_weight must be both set or both None")
        logical_range = layout.logical_token_range
        if self.is_internode:
            self._range_group_barrier(logical_range)
            _ep_inter.combine_internode_range(
                self.ep_context.combine_input_buf_ptrs,
                self.ep_context.combine_topk_weights_buf_ptrs if weight_preprocessed is not None else None,
                self.ep_context.rdma_rail_send_buf,
                self.ep_context.rdma_rail_send_win_handle,
                layout.num_tokens_per_rank,
                output,
                layout.node_topk_indices,
                layout.node_topk_send_mask,
                layout.node_token_dst_scatter_indices,
                output_weight,
                self.ep_context.config.max_m,
                self.ep_context.config.num_experts // self.world_size,
                self.num_sm,
                num_qps,
                logical_range,
            )
            _ep_inter.reset_signals_barrier_all_on_stream_if_active(*self._combine_signal_range, logical_range)
            return output, output_weight
        self._range_group_barrier(logical_range)
        _ep.combine_intranode_range(
            self.ep_context.combine_input_buf_ptrs,
            layout.token_topk_send_mask,
            layout.topk_indices,
            layout.token_dst_scatter_indices,
            logical_range,
            output,
            self.rank,
            self.world_size,
            self.ep_context.config.num_experts // self.world_size,
            self.num_sm,
            self.ep_context.combine_topk_weights_buf_ptrs if weight_preprocessed is not None else None,
            output_weight,
        )
        self._range_group_barrier(logical_range)
        return output, output_weight

    def get_combine_buffer(self, num_recv_tokens: int, dtype: torch.dtype = None) -> torch.Tensor:
        max_capacity = self.ep_context.combine_input_buf.shape[0]
        if num_recv_tokens > max_capacity:
            raise ValueError(f"num_recv_tokens ({num_recv_tokens}) exceeds combine_buffer capacity ({max_capacity}). ")

        if num_recv_tokens < 0:
            raise ValueError(f"num_recv_tokens must be positive, got {num_recv_tokens}")

        combine_input_buf = self.ep_context.combine_input_buf[:num_recv_tokens]

        if dtype is not None and dtype != combine_input_buf.dtype:
            raise ValueError(f"dtype mismatch, got {dtype}, expected {combine_input_buf.dtype}")
        return combine_input_buf

    def _buffer_in_range(self, tensor: torch.Tensor, buf: torch.Tensor) -> bool:
        if tensor.numel() == 0:
            return True

        assert tensor.dtype == buf.dtype
        assert tensor.device == buf.device
        assert tensor.is_contiguous()

        tensor_start = tensor.data_ptr()
        tensor_end = tensor_start + tensor.numel() * tensor.element_size()
        buf_start = buf.data_ptr()
        buf_end = buf_start + buf.numel() * buf.element_size()
        return tensor_start >= buf_start and tensor_end <= buf_end

    def _validate_combine_input_buffer(self, tensor: torch.Tensor) -> bool:
        return self._buffer_in_range(tensor, self.ep_context.combine_input_buf)

    def _validate_combine_weight_buffer(self, tensor: torch.Tensor) -> bool:
        return self._buffer_in_range(tensor, self.ep_context.combine_topk_weights_buf)

    def combine_intranode_preprocess(self, input: torch.Tensor, layout_desc: EPCommLayoutDesc,
                                     weight: torch.Tensor | None = None, zero_copy: bool = False, num_sm: int = 0):
        if num_sm <= 0:
            num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count * 8

        assert layout_desc.recv_token_count is not None
        assert layout_desc.recv_topk_scatter_indices is not None
        assert input.shape[0] <= self.ep_context.combine_input_buf.shape[0]
        if zero_copy:
            assert self._validate_combine_input_buffer(input)
            combine_input_buf = input
        else:
            combine_input_buf = self.ep_context.combine_input_buf[:input.shape[0]]
            combine_input_buf.copy_(input)

        if weight is not None:
            assert len(weight.shape) == 1 and weight.shape[0] == input.shape[0]
            assert weight.dtype == self.ep_context.config.weight_dtype
            assert weight.is_contiguous()
            combine_input_weight_buf = self.ep_context.combine_topk_weights_buf[:input.shape[0]]
        else:
            combine_input_weight_buf = None
        if layout_desc.expert_alignment > 1:
            assert layout_desc.recv_aligned_token_count is not None, \
                "recv_aligned_token_count must be set when expert_alignment > 1"
            combine_token_count = layout_desc.recv_aligned_token_count
        else:
            combine_token_count = layout_desc.recv_token_count
        _ep.combine_preprocess_inplace(
            combine_input_buf,
            combine_token_count,
            layout_desc.recv_topk_scatter_indices,
            self.rank,
            self.world_size,
            num_sm,
            weight,
            combine_input_weight_buf,
        )
        return combine_input_buf, combine_input_weight_buf

    def combine_intranode(self, input_preprocessed: torch.Tensor, layout_desc: EPCommLayoutDesc,
                          weight_preprocessed: torch.Tensor | None = None, *, combine_out: torch.Tensor | None = None,
                          combine_out_weight: torch.Tensor | None = None):
        layout_desc.check_combine_required_inputs()
        assert self._validate_combine_input_buffer(input_preprocessed)
        has_weight = weight_preprocessed is not None
        if has_weight:
            assert self._validate_combine_weight_buffer(weight_preprocessed)

        # pull mode, wait ep ranks to finish preprocess
        self.ep_group_barrier()
        hidden = self.ep_context.config.hidden
        num_experts_per_rank = self.ep_context.config.num_experts // self.ep_context.config.world_size
        topk = self.ep_context.config.topk
        num_token = layout_desc.token_dst_scatter_indices.shape[0]
        if combine_out is None:
            combine_intranode_out_buf = torch.empty((num_token, hidden), dtype=input_preprocessed.dtype,
                                                    device=input_preprocessed.device)
        else:
            combine_intranode_out_buf = combine_out
            if (tuple(combine_out.shape) != (num_token, hidden) or combine_out.dtype != input_preprocessed.dtype
                    or combine_out.device != input_preprocessed.device or not combine_out.is_contiguous()):
                raise ValueError("combine_out must match the combine output shape, dtype, and device")
        if has_weight:
            combine_intranode_input_weight_ptrs = self.ep_context.combine_topk_weights_buf_ptrs
            if combine_out_weight is None:
                combine_intranode_out_weight_buf = torch.empty(
                    (num_token, topk),
                    dtype=self.ep_context.config.weight_dtype,
                    device=input_preprocessed.device,
                )
            else:
                combine_intranode_out_weight_buf = combine_out_weight
                if (tuple(combine_out_weight.shape) != (num_token, topk)
                        or combine_out_weight.dtype != self.ep_context.config.weight_dtype
                        or combine_out_weight.device != input_preprocessed.device
                        or not combine_out_weight.is_contiguous()):
                    raise ValueError("combine_out_weight must match the combine weight output shape, dtype, and device")
        else:
            if combine_out_weight is not None:
                raise ValueError("combine_out_weight requires weight_preprocessed")
            combine_intranode_input_weight_ptrs = None
            combine_intranode_out_weight_buf = None
        _ep.combine_intranode(
            self.ep_context.combine_input_buf_ptrs,
            layout_desc.token_topk_send_mask,
            layout_desc.topk_indices,
            layout_desc.token_dst_scatter_indices,
            combine_intranode_out_buf,
            self.rank,
            self.world_size,
            num_experts_per_rank,
            self.num_sm,
            combine_intranode_input_weight_ptrs,
            combine_intranode_out_weight_buf,
        )
        self.ep_group_barrier()
        return combine_intranode_out_buf, combine_intranode_out_weight_buf

    def dispatch_internode(self, input: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor | None,
                           layout_desc: EPCommLayoutDesc, num_qps: int | None = None):
        max_slot_num_token = self.ep_context.config.max_m
        if layout_desc.need_recompute_token_within_expert_offset_and_expert_counts(topk_indices):
            layout_desc.token_within_expert_offset, layout_desc.expert_counts = \
                self.compute_stable_local_token_within_expert_offset_and_expert_counts(topk_indices, self.num_sm)

        num_token = input.shape[0]
        rdma_rail_send_views = self.ep_context.rdma_rail_send_slot_views(
            num_token,
            self.ep_context.config.hidden,
            self.ep_context.config.topk,
            max_slot_num_token=max_slot_num_token,
        )
        recompute = layout_desc.need_recompute_dispatch_layout(self.expert_alignment, num_token)
        if recompute:
            layout_desc.num_tokens_per_rank = torch.empty((self.ep_context.config.world_size, ), dtype=torch.int32,
                                                          device=input.device)
            # compute_dispatch_layout is a pure layout function: it derives the
            # sender-side send plan (token_dst_scatter_indices / token_topk_send_mask)
            # and the recv counts, but does NOT stage the RDMA rail slot -- that is
            # done unconditionally below so a reused layout still refreshes the slot.
            # The recv-count buffers (incl. the pinned recv_token_count_cpu poll
            # target) are allocated fresh by compute_dispatch_layout and owned by
            # this layout; nothing is shared through ep_context, so a later dispatch
            # cannot clobber this layout's counts.
            (
                layout_desc.recv_base_offset,
                layout_desc.token_dst_scatter_indices,
                layout_desc.token_topk_send_mask,
                layout_desc.recv_token_count_cpu,
                layout_desc.recv_token_count,
                layout_desc.recv_aligned_token_count_cpu,
                layout_desc.recv_aligned_token_count,
                layout_desc.recv_expert_counts,
            ) = _ep_inter.compute_dispatch_layout(
                topk_indices,
                layout_desc.token_within_expert_offset,
                layout_desc.expert_counts,
                self.ep_context.full_splits_win_handle,
                layout_desc.num_tokens_per_rank,
                self.ep_context.config.num_experts,
                self.num_sm,
                expert_alignment=self.expert_alignment,
            )
            layout_desc.expert_alignment = self.expert_alignment
            layout_desc.num_tokens = num_token
        else:
            if layout_desc.num_tokens_per_rank is None:
                raise ValueError("layout_desc.num_tokens_per_rank is required for internode dispatch reuse")

        # Poll the layout-private recv-count buffer. On reuse it already holds the
        # final counts (poll returns immediately); on recompute it is the fresh
        # pinned buffer the layout kernel is writing.
        if layout_desc.expert_alignment > 1:
            buf_count_cpu = layout_desc.recv_aligned_token_count_cpu
            buf_count_gpu = layout_desc.recv_aligned_token_count
        else:
            buf_count_cpu = layout_desc.recv_token_count_cpu
            buf_count_gpu = layout_desc.recv_token_count
        dispatch_recv_token_count, _ = self._realloc_dispatch_output_buf(buf_count_cpu, buf_count_gpu)

        # Stage the RDMA rail source slot for THIS dispatch. x / topk_indices /
        # topk_weights were always staged here; the send plan (mask/scatter) must be
        # staged the same way so a reused layout does not read a previous dispatch's
        # stale slot metadata. Slot meta offsets are keyed by num_token, so staging
        # with the current views is correct for any token count.
        rdma_rail_send_views["x"].copy_(input)
        rdma_rail_send_views["topk_indices"].copy_(topk_indices)
        rdma_rail_send_views["topk_send_mask"].copy_(layout_desc.token_topk_send_mask)
        rdma_rail_send_views["token_dst_scatter"].copy_(layout_desc.token_dst_scatter_indices)
        if topk_weights is not None:
            rdma_rail_send_views["topk_weights"].copy_(topk_weights)

        num_experts_per_rank = self.ep_context.config.num_experts // self.ep_context.config.world_size
        max_recv_tokens = self.ep_context.dispatch_output_buf.shape[0]
        # Per-source-node receive metadata: dispatch OUTPUTS (consumed by combine).
        # Distinct from the sender-side send plan above; reallocated each dispatch.
        meta_shape = (self.ep_context.config.nnodes, max_slot_num_token, self.ep_context.config.topk)
        layout_desc.node_topk_indices = torch.empty(meta_shape, dtype=topk_indices.dtype, device=topk_indices.device)
        layout_desc.node_topk_send_mask = torch.empty(meta_shape, dtype=torch.int32, device=topk_indices.device)
        layout_desc.node_token_dst_scatter_indices = torch.empty(meta_shape, dtype=torch.int32,
                                                                 device=topk_indices.device)
        self.ep_group_barrier()
        _ep_inter.dispatch_internode(
            self.ep_context.dispatch_output_buf_ptrs,
            self.ep_context.dispatch_topk_weights_buf_ptrs,
            self.ep_context.dispatch_topk_scatter_indices_buf_ptrs,
            max_recv_tokens,
            self.ep_context.rdma_rail_send_buf,
            self.ep_context.rdma_rail_send_win_handle,
            layout_desc.num_tokens_per_rank,
            layout_desc.node_topk_indices,
            layout_desc.node_topk_send_mask,
            layout_desc.node_token_dst_scatter_indices,
            max_slot_num_token,
            num_token,
            self.ep_context.config.hidden,
            topk_weights is not None,
            num_experts_per_rank,
            self.num_sm,
            num_qps,
        )
        # Fused reset+barrier: dispatch signals are quiescent here and must be
        # reset before the next dispatch; plain barriers do not touch signals.
        _ep_inter.reset_signals_barrier_all_on_stream(*self._dispatch_signal_range)
        layout_desc.recv_topk_scatter_indices = self.ep_context.dispatch_topk_scatter_indices_buf[:
                                                                                                  dispatch_recv_token_count]
        dispatch_weights = None
        if topk_weights is not None:
            dispatch_weights = self.ep_context.dispatch_topk_weights_buf[:dispatch_recv_token_count]
        return (self.ep_context.dispatch_output_buf[:dispatch_recv_token_count], dispatch_weights, layout_desc)

    def combine_internode(self, input_preprocessed: torch.Tensor, layout_desc: EPCommLayoutDesc,
                          weight_preprocessed: torch.Tensor | None = None, num_qps: int | None = None, *,
                          combine_out: torch.Tensor | None = None, combine_out_weight: torch.Tensor | None = None):
        layout_desc.check_internode_combine_required_inputs()
        assert self._validate_combine_input_buffer(input_preprocessed)
        has_weight = weight_preprocessed is not None
        if has_weight:
            assert self._validate_combine_weight_buffer(weight_preprocessed)

        self.ep_group_barrier()
        hidden = self.ep_context.config.hidden
        topk = self.ep_context.config.topk
        num_experts_per_rank = self.ep_context.config.num_experts // self.ep_context.config.world_size
        if layout_desc.token_within_expert_offset is None:
            raise ValueError("layout_desc.token_within_expert_offset is required for internode combine")
        num_token = layout_desc.token_within_expert_offset.shape[0]
        if combine_out is None:
            combine_out_tensor = torch.empty((num_token, hidden), dtype=input_preprocessed.dtype,
                                             device=input_preprocessed.device)
        else:
            combine_out_tensor = combine_out
            if (tuple(combine_out.shape) != (num_token, hidden) or combine_out.dtype != input_preprocessed.dtype
                    or combine_out.device != input_preprocessed.device or not combine_out.is_contiguous()):
                raise ValueError("combine_out must match the combine output shape, dtype, and device")
        kernel_combine_weight = None
        weight_ptrs = None
        if layout_desc.num_tokens_per_rank is None:
            raise ValueError("layout_desc.num_tokens_per_rank is required for internode combine")
        if (layout_desc.node_topk_indices is None or layout_desc.node_topk_send_mask is None
                or layout_desc.node_token_dst_scatter_indices is None):
            raise ValueError("internode combine requires per-node "
                             "node_topk_indices/node_topk_send_mask/node_token_dst_scatter_indices")
        if has_weight:
            weight_ptrs = self.ep_context.combine_topk_weights_buf_ptrs
            if combine_out_weight is None:
                kernel_combine_weight = torch.empty((num_token, topk), dtype=self.ep_context.config.weight_dtype,
                                                    device=input_preprocessed.device)
            else:
                kernel_combine_weight = combine_out_weight
                if (tuple(combine_out_weight.shape) != (num_token, topk)
                        or combine_out_weight.dtype != self.ep_context.config.weight_dtype
                        or combine_out_weight.device != input_preprocessed.device
                        or not combine_out_weight.is_contiguous()):
                    raise ValueError("combine_out_weight must match the combine weight output shape, dtype, and device")
        elif combine_out_weight is not None:
            raise ValueError("combine_out_weight requires weight_preprocessed")
        _ep_inter.combine_internode(
            self.ep_context.combine_input_buf_ptrs,
            weight_ptrs,
            self.ep_context.rdma_rail_send_buf,
            self.ep_context.rdma_rail_send_win_handle,
            layout_desc.num_tokens_per_rank,
            combine_out_tensor,
            layout_desc.node_topk_indices,
            layout_desc.node_topk_send_mask,
            layout_desc.node_token_dst_scatter_indices,
            kernel_combine_weight,
            self.ep_context.config.max_m,
            num_experts_per_rank,
            self.num_sm,
            num_qps,
        )
        # Fused reset+barrier for the combine signal range (see dispatch).
        _ep_inter.reset_signals_barrier_all_on_stream(*self._combine_signal_range)
        return combine_out_tensor, kernel_combine_weight

    def dispatch(self, input: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor | None,
                 layout_desc: EPCommLayoutDesc = None, num_qps: int | None = None):
        # Contract: topk_indices must be in [0, num_experts], where num_experts
        # is the drop sentinel.  Negative values are not a supported sentinel.
        if layout_desc is None:
            layout_desc = EPCommLayoutDesc()
        else:
            layout_desc.check_layout_desc(num_tokens=topk_indices.shape[0], topk=topk_indices.shape[1],
                                          num_experts=self.ep_context.config.num_experts,
                                          world_size=self.ep_context.config.world_size,
                                          local_world_size=self.ep_context.config.local_world_size,
                                          max_slot_num_token=self.ep_context.config.max_m)

        if layout_desc.logical_token_range is not None:
            return self._dispatch_range(input, topk_indices, topk_weights, layout_desc, num_qps=num_qps)

        if self.ep_context.config.nnodes == 1:
            # num_qps only affects internode RDMA traffic; ignored intranode so
            # the same caller code runs on a single node.
            return self.dispatch_intranode(input, topk_indices, topk_weights, layout_desc)
        else:
            return self.dispatch_internode(input, topk_indices, topk_weights, layout_desc, num_qps=num_qps)

    def dispatch_mxfp8(self, input: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor | None,
                       layout_desc: EPCommLayoutDesc = None):
        """Quantize BF16 input inside intranode MXFP8 dispatch."""
        if self.ep_context.config.nnodes != 1:
            raise NotImplementedError("MXFP8 dispatch currently supports intranode EP only")
        if layout_desc is None:
            layout_desc = EPCommLayoutDesc()
        else:
            layout_desc.check_layout_desc(num_tokens=topk_indices.shape[0], topk=topk_indices.shape[1],
                                          num_experts=self.ep_context.config.num_experts,
                                          world_size=self.ep_context.config.world_size,
                                          local_world_size=self.ep_context.config.local_world_size,
                                          max_slot_num_token=self.ep_context.config.max_m)
        return self._dispatch_mxfp8_intranode(input, topk_indices, topk_weights, layout_desc, prequantized=False)

    def dispatch_mxfp8_prequantized(self, input: torch.Tensor, topk_indices: torch.Tensor,
                                    topk_weights: torch.Tensor | None, layout_desc: EPCommLayoutDesc = None):
        """Dispatch transport-packed MXFP8 input without quantization."""
        if self.ep_context.config.nnodes != 1:
            raise NotImplementedError("MXFP8 dispatch currently supports intranode EP only")
        if layout_desc is None:
            layout_desc = EPCommLayoutDesc()
        else:
            layout_desc.check_layout_desc(num_tokens=topk_indices.shape[0], topk=topk_indices.shape[1],
                                          num_experts=self.ep_context.config.num_experts,
                                          world_size=self.ep_context.config.world_size,
                                          local_world_size=self.ep_context.config.local_world_size,
                                          max_slot_num_token=self.ep_context.config.max_m)
        return self._dispatch_mxfp8_intranode(input, topk_indices, topk_weights, layout_desc, prequantized=True)

    def dispatch_postprocess(self, dispatch_out: torch.Tensor, dispatch_topk_weights: torch.Tensor | None,
                             layout_desc: EPCommLayoutDesc, num_sm: int = 0,
                             recv_topk_scatter_indices_out: torch.Tensor | None = None,
                             dispatch_weights_out: torch.Tensor | None = None):
        return self.dispatch_intranode_postprocess(
            dispatch_out,
            dispatch_topk_weights,
            layout_desc,
            num_sm,
            recv_topk_scatter_indices_out,
            dispatch_weights_out,
        )

    def dispatch_mxfp8_postprocess_unpack(self, packed_out: torch.Tensor, dispatch_topk_weights: torch.Tensor | None,
                                          layout_desc: EPCommLayoutDesc, num_sm: int = 0, *,
                                          dispatch_weights_out: torch.Tensor | None = None):
        return self.dispatch_mxfp8_intranode_postprocess(
            packed_out,
            dispatch_topk_weights,
            layout_desc,
            num_sm,
            dispatch_weights_out=dispatch_weights_out,
        )

    def combine_preprocess(self, input: torch.Tensor, layout_desc: EPCommLayoutDesc, weight: torch.Tensor | None = None,
                           zero_copy: bool = False, num_sm: int = 0):
        return self.combine_intranode_preprocess(input, layout_desc, weight=weight, zero_copy=zero_copy, num_sm=num_sm)

    def combine(self, input_preprocessed: torch.Tensor, layout_desc: EPCommLayoutDesc,
                weight_preprocessed: torch.Tensor | None = None, num_qps: int | None = None, *,
                output: torch.Tensor | None = None, output_weight: torch.Tensor | None = None,
                combine_out: torch.Tensor | None = None, combine_out_weight: torch.Tensor | None = None):
        if layout_desc.logical_token_range is not None:
            if output is not None and combine_out is not None and output is not combine_out:
                raise ValueError("output and combine_out must refer to the same tensor")
            if output_weight is not None and combine_out_weight is not None and output_weight is not combine_out_weight:
                raise ValueError("output_weight and combine_out_weight must refer to the same tensor")
            range_output = output if output is not None else combine_out
            range_output_weight = output_weight if output_weight is not None else combine_out_weight
            if range_output is None:
                raise ValueError("range combine requires the caller-owned full output tensor")
            return self._combine_range(input_preprocessed, layout_desc, range_output,
                                       weight_preprocessed=weight_preprocessed, output_weight=range_output_weight,
                                       num_qps=num_qps)
        if output is not None or output_weight is not None:
            raise ValueError("output/output_weight are only valid for range combine")
        if self.ep_context.config.nnodes == 1:
            # num_qps ignored intranode (see dispatch).
            return self.combine_intranode(input_preprocessed, layout_desc=layout_desc,
                                          weight_preprocessed=weight_preprocessed, combine_out=combine_out,
                                          combine_out_weight=combine_out_weight)
        else:
            return self.combine_internode(input_preprocessed, layout_desc=layout_desc,
                                          weight_preprocessed=weight_preprocessed, num_qps=num_qps,
                                          combine_out=combine_out, combine_out_weight=combine_out_weight)
