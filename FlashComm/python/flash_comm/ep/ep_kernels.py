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
import dataclasses
from typing import Optional
import flash_comm._C.ep_intranode as _ep

from .ep_context import EPContext


@dataclasses.dataclass
class EPCommLayoutDesc:
    # only dependent on local topk_indices, can be computed in advance
    token_within_expert_offset: Optional[torch.Tensor] = None  # [num_tokens, topk]
    expert_counts: Optional[torch.Tensor] = None  # [num_experts + 1]

    # dispatch layout
    recv_base_offset: Optional[torch.Tensor] = None  # [world_size, experts_per_rank, world_size]

    # token_dst_scatter_indices / recv_topk_scatter_indices both store "positions in the dispatch output buffer".
    #
    # - token_dst_scatter_indices: sender-token space mapping.
    #   Shape: [num_token, topk] (intranode).
    #   For each local input token t and its k-th expert choice:
    #     token_dst_scatter_indices[t, k] is the slot index inside the *target rank's* dispatch receive buffer
    #     where this (token, expert-choice) should be written during dispatch.
    #
    # - recv_topk_scatter_indices: receiver-token space mapping.
    #   Shape: [num_recv_token, topk] (intranode), where num_recv_token = recv_token_count[rank].
    #   For each token row r in *this rank's* dispatch receive buffer and its k-th lane:
    #     recv_topk_scatter_indices[r, k] stores the "scatter index" used by dispatch_postprocess/combine to
    #     relate received rows back to the original token ordering (and/or to compute dispatch_weights).
    #
    # In short:
    # - token_dst_scatter_indices is indexed by *sender-side* token rows (pre-dispatch).
    # - recv_topk_scatter_indices is indexed by *receiver-side* token rows (post-dispatch).
    # Both refer to offsets/slots in the dispatch receive buffers, but their index space (and thus shape) differs.
    token_dst_scatter_indices: Optional[
        torch.Tensor] = None  # intranode: [num_token, topk]; internode: [nnodes, max_tokens, topk]
    token_topk_send_mask: Optional[
        torch.Tensor] = None  # intranode: [num_token, topk]; internode: [nnodes, max_tokens, topk]
    topk_indices: Optional[torch.Tensor] = None  # intranode: [num_token, topk]; internode: [nnodes, max_tokens, topk]
    recv_token_count_cpu: Optional[torch.Tensor] = None  # [world_size] pinned CPU memory (unaligned)
    recv_token_count: Optional[torch.Tensor] = None  # [world_size] device memory (unaligned)
    recv_aligned_token_count_cpu: Optional[torch.Tensor] = None  # [world_size] pinned CPU (aligned, for buffer alloc)
    recv_aligned_token_count: Optional[torch.Tensor] = None  # [world_size] device (aligned, for postprocess/combine)
    recv_expert_counts: Optional[torch.Tensor] = None  # [experts_per_rank] per-expert actual token counts
    expert_alignment: int = 1
    recv_topk_scatter_indices: Optional[
        torch.Tensor] = None  # intranode: [num_recv_token, topk], internode: [nnodes, max_tokens, topk]

    def check_combine_required_inputs(self):
        if self.token_topk_send_mask is None or self.token_dst_scatter_indices is None:
            raise ValueError("token_topk_send_mask and token_dst_scatter_indices must be provided")

    def need_recompute_token_within_expert_offset_and_expert_counts(self):
        return self.token_within_expert_offset is None or self.expert_counts is None

    def need_recompute_dispatch_layout(self, expert_alignment: int = 1):
        if (self.recv_base_offset is None or self.token_dst_scatter_indices is None or self.token_topk_send_mask is None
                or self.recv_token_count_cpu is None or self.recv_token_count is None
                or self.recv_expert_counts is None):
            return True
        if self.expert_alignment != expert_alignment:
            return True
        if expert_alignment > 1:
            if self.recv_aligned_token_count_cpu is None or self.recv_aligned_token_count is None:
                return True
        return False

    def check_layout_desc(self, num_tokens: int, topk: int, num_experts: int, world_size: int = 1):

        def check_shape(tensor: Optional[torch.Tensor], name: str, expected_shape):
            if tensor is not None and tuple(tensor.shape) != tuple(expected_shape):
                raise ValueError(f"{name} must have shape {list(expected_shape)}, got shape {tuple(tensor.shape)}")

        has_token_offset = self.token_within_expert_offset is not None
        has_expert_counts = self.expert_counts is not None
        if has_expert_counts != has_token_offset:
            raise ValueError("token_within_expert_offset and expert_counts must both be None or both be non-None")

        # experts_per_rank = num_experts // world_size
        # check_shape(self.token_within_expert_offset, "token_within_expert_offset", (num_tokens, topk))
        # check_shape(self.expert_counts, "expert_counts", (num_experts + 1,))
        # check_shape(self.recv_base_offset, "recv_base_offset", (world_size, experts_per_rank, world_size))
        # check_shape(self.token_dst_scatter_indices, "token_dst_scatter_indices", (num_tokens, topk))
        # check_shape(self.token_topk_send_mask, "token_topk_send_mask", (num_tokens, topk))
        # check_shape(self.topk_indices, "topk_indices", (num_tokens, topk))
        # check_shape(self.recv_token_count_cpu, "recv_token_count_cpu", (world_size,))
        # check_shape(self.recv_token_count, "recv_token_count", (world_size,))
        # check_shape(self.recv_aligned_token_count_cpu, "recv_aligned_token_count_cpu", (world_size,))
        # check_shape(self.recv_aligned_token_count, "recv_aligned_token_count", (world_size,))
        # check_shape(self.recv_expert_counts, "recv_expert_counts", (experts_per_rank,))
        if self.recv_topk_scatter_indices is not None:
            if self.recv_topk_scatter_indices.dim() != 2 or self.recv_topk_scatter_indices.shape[1] != topk:
                raise ValueError("recv_topk_scatter_indices must have shape [num_recv_tokens, topk], "
                                 f"got shape {tuple(self.recv_topk_scatter_indices.shape)}")


class EPKernels:
    """
    FlashComm EP kernels, currently only support intranode dispatch and combine.
    """

    def __init__(self, max_m: int, hidden: int, topk: int, num_experts: int, local_world_size: int,
                 ep_group: torch.distributed.ProcessGroup, num_sm: int = 16, capacity: float = 1.2,
                 num_worst_tokens: int = -1, expert_alignment: int = 1, check_num_worst_tokens: bool = False):
        self.ep_context = EPContext.create(max_m=max_m, hidden=hidden, topk=topk, num_experts=num_experts,
                                           group=ep_group, local_world_size=local_world_size, capacity_coeff=capacity,
                                           num_worst_tokens=num_worst_tokens)
        self.ep_group = ep_group
        self.num_sm = num_sm
        self.alignment = 1024
        self.coeff = capacity  # for output buffer reallocation
        self.cpu_default_val = -1
        self.rank = ep_group.rank()
        self.world_size = ep_group.size()
        self.local_world_size = local_world_size
        self.num_worst_tokens = num_worst_tokens
        assert expert_alignment >= 1, f"expert_alignment must be >= 1, got {expert_alignment}"
        self.expert_alignment = expert_alignment
        self.check_num_worst_tokens = check_num_worst_tokens
        torch.distributed.barrier(group=ep_group)

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
            pass
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

    def dispatch_intranode(self, input: torch.Tensor, topk_indices: torch.Tensor, topk_weights: Optional[torch.Tensor],
                           layout_desc: EPCommLayoutDesc):
        self.ep_group_barrier()
        # recompute if not provided
        if layout_desc.need_recompute_token_within_expert_offset_and_expert_counts():
            layout_desc.token_within_expert_offset, layout_desc.expert_counts = \
                self.compute_stable_local_token_within_expert_offset_and_expert_counts(topk_indices, self.num_sm)

        if layout_desc.need_recompute_dispatch_layout(self.expert_alignment):
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
                self.ep_context.recv_token_count_cpu,
                expert_alignment=self.expert_alignment,
            )
            layout_desc.expert_alignment = self.expert_alignment

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

        num_experts_per_rank = self.ep_context.config.num_experts // self.ep_context.config.world_size
        # dispatch
        # Note: if topk_weights is None, the kernel skips weight dispatch and the weight buffer contents are stale.
        _ep.dispatch_intranode(input, layout_desc.token_topk_send_mask, topk_weights, topk_indices,
                               layout_desc.token_dst_scatter_indices, self.ep_context.dispatch_output_buf_ptrs,
                               self.ep_context.dispatch_topk_weights_buf_ptrs,
                               self.ep_context.dispatch_topk_scatter_indices_buf_ptrs, self.rank, self.world_size,
                               num_experts_per_rank, self.num_sm)

        # push mode, need barrier on ep group, wait ep ranks to finish dispatch
        self.ep_group_barrier()
        layout_desc.topk_indices = topk_indices
        layout_desc.recv_topk_scatter_indices = self.ep_context.dispatch_topk_scatter_indices_buf[:
                                                                                                  dispatch_recv_token_count]
        # directly return the symm tensors to avoid extra copy
        dispatch_weights = None
        if topk_weights is not None:
            dispatch_weights = self.ep_context.dispatch_topk_weights_buf[:dispatch_recv_token_count]
        return (self.ep_context.dispatch_output_buf[:dispatch_recv_token_count], dispatch_weights, layout_desc)

    def dispatch_intranode_postprocess(self, dispatch_out: torch.Tensor, dispatch_topk_weights: Optional[torch.Tensor],
                                       layout_desc: EPCommLayoutDesc, num_sm: int = 0):
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
        recv_topk_scatter_indices = torch.empty((buffer_size, topk), dtype=layout_desc.recv_topk_scatter_indices.dtype,
                                                device=layout_desc.recv_topk_scatter_indices.device)
        dispatch_weights = None
        if dispatch_topk_weights is not None:
            dispatch_weights = torch.empty((buffer_size, ), dtype=dispatch_topk_weights.dtype,
                                           device=dispatch_topk_weights.device)
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
            dispatch_weights,
            recv_topk_scatter_indices,  # torch tensor
            hidden,
            topk,
            self.rank,
            self.world_size,
            num_sm,
        )

        # update layout desc
        layout_desc.recv_topk_scatter_indices = recv_topk_scatter_indices
        return dispatch_out, dispatch_weights, layout_desc

    def ep_group_barrier(self):
        # torch.distributed.barrier(group=self.ep_group)
        if self.ep_context.config.nnodes == 1:
            assert self.ep_context.nvl_barrier_buf.dtype == torch.int32
            _ep.barrier_all_on_stream(self.ep_context.nvl_barrier_buf_ptrs, self.rank, self.world_size)
        else:
            torch.distributed.barrier(group=self.ep_group)

    def compute_stable_local_token_within_expert_offset_and_expert_counts(self, topk_indices: torch.Tensor, num_sm=-1):
        token_within_expert_offset, block_cumsum_hist, expert_counts = _ep.compute_stable_local_token_within_expert_offset_and_expert_counts(
            topk_indices, self.ep_context.config.num_experts, num_sm)
        return token_within_expert_offset, expert_counts

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
                                     weight: Optional[torch.Tensor] = None, zero_copy: bool = False, num_sm: int = 0):
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
                          weight_preprocessed: Optional[torch.Tensor] = None):
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
        combine_intranode_out_buf = torch.empty((layout_desc.token_dst_scatter_indices.shape[0], hidden),
                                                dtype=input_preprocessed.dtype, device=input_preprocessed.device)
        if has_weight:
            combine_intranode_input_weight_ptrs = self.ep_context.combine_topk_weights_buf_ptrs
            combine_intranode_out_weight_buf = torch.empty((layout_desc.token_dst_scatter_indices.shape[0], topk),
                                                           dtype=self.ep_context.config.weight_dtype,
                                                           device=input_preprocessed.device)
        else:
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

    def dispatch_internode(self, input: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor,
                           layout_desc: EPCommLayoutDesc):
        raise NotImplementedError("dispatch_internode is not implemented")

    def combine_internode(self, input: torch.Tensor, recv_topk_scatter_indices: torch.Tensor,
                          layout_desc: EPCommLayoutDesc):
        raise NotImplementedError("combine_internode is not implemented")

    def dispatch(self, input: torch.Tensor, topk_indices: torch.Tensor, topk_weights: Optional[torch.Tensor],
                 layout_desc: EPCommLayoutDesc = None):
        if layout_desc is None:
            layout_desc = EPCommLayoutDesc()
        else:
            layout_desc.check_layout_desc(num_tokens=topk_indices.shape[0], topk=topk_indices.shape[1],
                                          num_experts=self.ep_context.config.num_experts,
                                          world_size=self.ep_context.config.world_size)

        if self.ep_context.config.nnodes == 1:
            return self.dispatch_intranode(input, topk_indices, topk_weights, layout_desc)
        else:
            return self.dispatch_internode(input, topk_indices, topk_weights, layout_desc)

    def dispatch_postprocess(self, dispatch_out: torch.Tensor, dispatch_topk_weights: Optional[torch.Tensor],
                             layout_desc: EPCommLayoutDesc, num_sm: int = 0):
        if self.ep_context.config.nnodes == 1:
            return self.dispatch_intranode_postprocess(dispatch_out, dispatch_topk_weights, layout_desc, num_sm)
        else:
            raise NotImplementedError("dispatch_postprocess is not implemented for internode")

    def combine_preprocess(self, input: torch.Tensor, layout_desc: EPCommLayoutDesc,
                           weight: Optional[torch.Tensor] = None, zero_copy: bool = False, num_sm: int = 0):
        if self.ep_context.config.nnodes == 1:
            return self.combine_intranode_preprocess(input, layout_desc, weight=weight, zero_copy=zero_copy,
                                                     num_sm=num_sm)
        else:
            raise NotImplementedError("combine_preprocess is not implemented for internode")

    def combine(self, input_preprocessed: torch.Tensor, layout_desc: EPCommLayoutDesc,
                weight_preprocessed: Optional[torch.Tensor] = None):
        if self.ep_context.config.nnodes == 1:
            return self.combine_intranode(input_preprocessed, layout_desc=layout_desc,
                                          weight_preprocessed=weight_preprocessed)
        else:
            return self.combine_internode(input_preprocessed, layout_desc=layout_desc,
                                          weight_preprocessed=weight_preprocessed)
