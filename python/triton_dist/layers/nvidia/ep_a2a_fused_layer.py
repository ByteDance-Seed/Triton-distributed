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
import os
import triton
import dataclasses
import ctypes
import math
from typing import Optional

from triton_dist.kernels.nvidia.ep_a2a import (
    bincount,
    get_dispatch_send_reqs,
)
from triton_dist.kernels.nvidia.memory_ops import (
    copy_tensor,
    fill_tensor,
)
from triton_dist.kernels.nvidia.ep_all2all_fused import (
    mega_kernel_dispatch_token_moe_grouped_gemm,
    mega_kernel_moe_grouped_gemm_combine_token,
    mega_kernel_moe_grouped_gemm_combine_token_transposed_grouped_gemm,
    get_ag_splits_and_recv_offset_for_dispatch,
)
from triton_dist.kernels.nvidia.group_gemm import GROUP_GEMM_BLOCK_SIZE_M
from triton_dist.kernels.nvidia.common_ops import NVSHMEM_SIGNAL_DTYPE, nvshmem_barrier_all_on_stream, BarrierAllContext, barrier_all_on_stream
from triton_dist.tools.profiler import ProfilerBuffer, export_to_perfetto_trace
from triton_dist.utils import NVSHMEMLazyAllocator, nvshmem_free_lazy_tensor


@dataclasses.dataclass
class EPAllToAllLayoutDesc:
    num_dispatch_token_cur_rank: int
    recv_buf_offset_per_expert: torch.Tensor
    recv_buf_tokens_per_expert: torch.Tensor
    num_recv_tokens_per_rank: torch.Tensor
    num_input_tokens_per_rank: torch.Tensor
    send_reqs_for_nodes: torch.Tensor
    send_reqs_recv_tensor: torch.Tensor
    topk_indices_tensor: torch.Tensor
    non_drop_token_count_tensor: torch.Tensor  # this records the number of replica of input tokens
    token_dst_scatter_idx: torch.Tensor
    reversed_token_scatter_idx: torch.Tensor
    token_sort_indices: torch.Tensor
    skipped_token_mapping_indices: Optional[torch.Tensor] = None


class EpAll2AllFusedOp(torch.nn.Module):

    def __init__(self, ep_group, max_tokens: int, hidden: int, topk: int, rank: int, num_tot_experts: int,
                 local_world_size: int, world_size: int, dtype=torch.bfloat16, weight_dtype=torch.float32, num_sm=20,
                 sm_margin=0, duplicate_comm_buffer: int = 1, capacity=4.0, FWD_GEMM_BLOCK_SIZE_N=256,
                 need_reversed_token_scatter_idx=False, lazy: bool = False):
        super().__init__()
        self.offset_dtype = torch.int32
        self.ep_group = ep_group
        self.num_sm = num_sm
        self.sm_margin = sm_margin

        self.max_tokens = max_tokens
        self.topk = topk
        self.hidden = hidden
        self.dtype = dtype
        self.weight_dtype = weight_dtype

        assert num_tot_experts % world_size == 0
        self.num_tot_experts = num_tot_experts
        self.experts_per_rank = num_tot_experts // world_size

        self.local_world_size = local_world_size
        self.world_size = world_size
        self.rank = rank
        self.nnodes = self.world_size // self.local_world_size
        self.node_id = self.rank // self.local_world_size

        # Initialize lazy allocator for nvshmem tensors
        self._nvshmem_allocator = NVSHMEMLazyAllocator(lazy=lazy)
        self._lazy = lazy

        self.is_intra_node = (self.world_size == self.local_world_size)
        assert self.is_intra_node, "EPAll2AllFusedOp only supports intra-node mode"
        self.intra_node_dispatch_skipped_token_mapping_indices = self._nvshmem_allocator.create_tensor(
            "intra_node_dispatch_skipped_token_mapping_indices", [self.local_world_size * max_tokens * topk],
            NVSHMEM_SIGNAL_DTYPE, fill_value=-1)

        # for dispatch
        self.send_reqs_for_nodes = self._nvshmem_allocator.create_tensor("send_reqs_for_nodes",
                                                                         [self.nnodes, 2, max_tokens],
                                                                         self.offset_dtype, fill_value=-1)
        self.send_reqs_recv_bufs = self._nvshmem_allocator.create_tensor("send_reqs_recv_bufs",
                                                                         [self.nnodes, 2, max_tokens],
                                                                         self.offset_dtype, fill_value=-1)
        self.Alignment = 1024

        avg_tokens = max_tokens * topk

        # for flux-like multi comm buffers
        assert duplicate_comm_buffer >= 1
        self.duplicate_comm_buffer = duplicate_comm_buffer
        self.current_comm_buffer_id = 0

        self.capacity = capacity
        self.FWD_GEMM_BLOCK_SIZE_N = FWD_GEMM_BLOCK_SIZE_N
        self.need_reversed_token_scatter_idx = need_reversed_token_scatter_idx

        # for dispatch comm
        self.comm_buffers = [
            self._nvshmem_allocator.create_tensor(f"comm_buffer_{i}", [self.nnodes, max_tokens, hidden], dtype)
            for i in range(duplicate_comm_buffer)
        ]
        self.output_buffers = [
            self._nvshmem_allocator.create_tensor(f"output_buffer_{i}", [math.ceil(avg_tokens * self.capacity), hidden],
                                                  dtype) for i in range(duplicate_comm_buffer)
        ]
        self.weight_recv_buffers = [
            self._nvshmem_allocator.create_tensor(f"weight_recv_buffer_{i}", [math.ceil(avg_tokens * self.capacity)],
                                                  weight_dtype) for i in range(duplicate_comm_buffer)
        ]
        self.send_buf = self.comm_buffers[self.current_comm_buffer_id]
        self.output_buf = self.output_buffers[self.current_comm_buffer_id]
        self.weight_recv_buf = self.weight_recv_buffers[self.current_comm_buffer_id]

        # for combine comm
        self.combine_in_buf = self._nvshmem_allocator.create_tensor("combine_in_buf",
                                                                    [math.ceil(avg_tokens * self.capacity), hidden],
                                                                    dtype)
        self.combine_out_buf = self._nvshmem_allocator.create_tensor("combine_out_buf",
                                                                     [self.nnodes, max_tokens, hidden], dtype)
        self.combine_gate_in_buf = self._nvshmem_allocator.create_tensor("combine_gate_in_buf",
                                                                         [math.ceil(avg_tokens * self.capacity)],
                                                                         weight_dtype)
        self.combine_gate_out_buf = self._nvshmem_allocator.create_tensor("combine_gate_out_buf",
                                                                          [self.nnodes, max_tokens, topk], weight_dtype)

        self.signal_buf = self._nvshmem_allocator.create_tensor("signal_buf", [world_size], NVSHMEM_SIGNAL_DTYPE,
                                                                fill_value=0)
        self.topk_indices_buf = self._nvshmem_allocator.create_tensor("topk_indices_buf",
                                                                      [self.nnodes, max_tokens, topk],
                                                                      self.offset_dtype)
        self.weight_send_buf = self._nvshmem_allocator.create_tensor("weight_send_buf", [self.nnodes, max_tokens, topk],
                                                                     weight_dtype)
        self.counter = torch.zeros((self.nnodes, ), dtype=torch.int32).cuda()

        # dispatch preprocess, use push mode to reduce barrier_all
        # local_splits_buf[num_tot_experts] is used for drop token
        self.local_splits_buf = self._nvshmem_allocator.create_tensor("local_splits_buf", [self.num_tot_experts + 1],
                                                                      self.offset_dtype, fill_value=0)
        self.full_splits_buf = self._nvshmem_allocator.create_tensor("full_splits_buf",
                                                                     [world_size, num_tot_experts + 1],
                                                                     self.offset_dtype)
        self.num_input_tokens_per_rank_comm_buf = self._nvshmem_allocator.create_tensor(
            "num_input_tokens_per_rank_comm_buf", [world_size], self.offset_dtype)
        self.splits_signal_buf = self._nvshmem_allocator.create_tensor("splits_signal_buf", [world_size],
                                                                       NVSHMEM_SIGNAL_DTYPE, fill_value=0)
        self.expert_indices_signal_buf = self._nvshmem_allocator.create_tensor("expert_indices_signal_buf",
                                                                               [world_size], NVSHMEM_SIGNAL_DTYPE,
                                                                               fill_value=0)
        self.cpu_default_val = -1

        self.full_local_scatter_indices_buf = self._nvshmem_allocator.create_tensor("full_local_scatter_indices_buf",
                                                                                    [self.nnodes, max_tokens, topk],
                                                                                    self.offset_dtype)

        # for combine
        self.intra_node_reduce_buf = self._nvshmem_allocator.create_tensor("intra_node_reduce_buf",
                                                                           [self.nnodes, max_tokens, hidden], dtype)
        self.intra_node_gate_buf = self._nvshmem_allocator.create_tensor("intra_node_gate_buf",
                                                                         [self.nnodes, max_tokens, topk], weight_dtype)

        # for mega
        self.num_mega_dispatch_counters = max(
            self.num_tot_experts,
            triton.cdiv(self.max_tokens * self.topk * self.world_size, GROUP_GEMM_BLOCK_SIZE_M) +
            self.num_tot_experts // self.world_size)
        self.mega_dispatch_counter_buf = torch.zeros([self.num_mega_dispatch_counters], dtype=self.offset_dtype,
                                                     device="cuda")
        self.mega_dispatch_barrier_buf = self._nvshmem_allocator.create_tensor("mega_dispatch_barrier_buf",
                                                                               [self.num_mega_dispatch_counters],
                                                                               NVSHMEM_SIGNAL_DTYPE, fill_value=0)
        self.mega_token_rank_table_buf = torch.empty([self.max_tokens, self.local_world_size], dtype=self.offset_dtype,
                                                     device="cuda")
        self.mega_token_indirect_pos_buf = self._nvshmem_allocator.create_tensor(
            "mega_token_indirect_pos_buf", [self.max_tokens * self.topk * self.local_world_size], self.offset_dtype,
            fill_value=-1)
        self.mega_token_rank_table_buf.fill_(-1)

        self.mega_combine_counter_buf = torch.zeros([self.max_tokens * self.topk * local_world_size],
                                                    dtype=self.offset_dtype, device="cuda")
        self.mega_combine_barrier_buf = self._nvshmem_allocator.create_tensor(
            "mega_combine_barrier_buf",
            [self.max_tokens * self.topk * local_world_size, self.hidden // self.FWD_GEMM_BLOCK_SIZE_N],
            NVSHMEM_SIGNAL_DTYPE, fill_value=0)
        self.mega_combine_scatter_output_buf = self._nvshmem_allocator.create_tensor(
            "mega_combine_scatter_output_buf", [self.max_tokens * self.topk, self.hidden], dtype)
        self.mega_combine_scatter_output_barrier_buf = self._nvshmem_allocator.create_tensor(
            "mega_combine_scatter_output_barrier_buf", [self.max_tokens * self.topk * self.local_world_size],
            torch.int32)
        if self.need_reversed_token_scatter_idx:
            self.mega_reversed_token_scatter_idx_buf = self._nvshmem_allocator.create_tensor(
                "mega_reversed_token_scatter_idx_buf", [self.world_size * self.max_tokens * self.topk, 2],
                self.offset_dtype, fill_value=-1)
        else:
            self.mega_reversed_token_scatter_idx_buf = None
        self.MAX_SMS = max(torch.cuda.get_device_properties(0).multi_processor_count - self.sm_margin, 1)
        self._task_counter_buf = torch.zeros([self.MAX_SMS], dtype=torch.int32, device="cuda")
        self.barrier_all_workspace = self._nvshmem_allocator.create_tensor("barrier_all_workspace",
                                                                           [self.MAX_SMS, self.world_size], torch.int32,
                                                                           fill_value=0)
        self.barrier_all_ctx = BarrierAllContext(is_intra_node=(self.nnodes == 1))

        # If not lazy, sync immediately (backward compatible behavior)
        if not lazy:
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            torch.cuda.synchronize()

    # ==================== Nvshmem Memory Management APIs ====================

    def get_nvshmem_size(self) -> int:
        """
        Get the total nvshmem memory size in bytes.
        
        This can be called before sync() to query the total memory needed.
        """
        return self._nvshmem_allocator.get_total_nvshmem_size()

    def get_nvshmem_size_gb(self) -> float:
        """Get the total nvshmem memory size in GB."""
        return self._nvshmem_allocator.get_total_nvshmem_size_gb()

    def get_nvshmem_size_mb(self) -> float:
        """Get the total nvshmem memory size in MB."""
        return self._nvshmem_allocator.get_total_nvshmem_size_mb()

    def get_nvshmem_breakdown(self) -> dict:
        """
        Get a breakdown of nvshmem usage by buffer name.
        
        Returns:
            Dict mapping buffer name to size in bytes
        """
        return self._nvshmem_allocator.get_tensor_breakdown()

    def print_nvshmem_breakdown(self):
        """Print a human-readable breakdown of nvshmem memory usage."""
        self._nvshmem_allocator.print_memory_breakdown()

    def is_nvshmem_materialized(self) -> bool:
        """Check if nvshmem tensors have been allocated."""
        return self._nvshmem_allocator.is_materialized

    def sync(self):
        """
        Materialize all nvshmem tensors.
        
        This actually allocates the nvshmem memory for all pending tensors.
        Must be called before using any nvshmem buffers if lazy=True was set.
        """
        if self._nvshmem_allocator.is_materialized:
            return

        self._nvshmem_allocator.sync()

        # Run barrier after materialization
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    def materialize(self):
        """Alias for sync()."""
        self.sync()

    def finalize(self):
        nvshmem_free_lazy_tensor(self.num_input_tokens_per_rank_comm_buf)
        nvshmem_free_lazy_tensor(self.expert_indices_signal_buf)
        nvshmem_free_lazy_tensor(self.weight_send_buf)
        nvshmem_free_lazy_tensor(self.weight_recv_buf)
        nvshmem_free_lazy_tensor(self.send_reqs_for_nodes)
        nvshmem_free_lazy_tensor(self.send_reqs_recv_bufs)
        nvshmem_free_lazy_tensor(self.send_buf)
        nvshmem_free_lazy_tensor(self.combine_out_buf)
        nvshmem_free_lazy_tensor(self.output_buf)
        nvshmem_free_lazy_tensor(self.combine_in_buf)
        nvshmem_free_lazy_tensor(self.signal_buf)
        nvshmem_free_lazy_tensor(self.topk_indices_buf)
        nvshmem_free_lazy_tensor(self.local_splits_buf)
        nvshmem_free_lazy_tensor(self.full_splits_buf)
        nvshmem_free_lazy_tensor(self.splits_signal_buf)
        nvshmem_free_lazy_tensor(self.intra_node_reduce_buf)
        nvshmem_free_lazy_tensor(self.mega_dispatch_barrier_buf)
        nvshmem_free_lazy_tensor(self.mega_combine_barrier_buf)
        nvshmem_free_lazy_tensor(self.mega_combine_scatter_output_buf)
        nvshmem_free_lazy_tensor(self.intra_node_dispatch_skipped_token_mapping_indices)
        if self.need_reversed_token_scatter_idx:
            nvshmem_free_lazy_tensor(self.mega_reversed_token_scatter_idx_buf)
        nvshmem_free_lazy_tensor(self.barrier_all_workspace)

    def init_output_buffer(self, num_recv_tokens_per_rank, min_m: Optional[int] = None):
        # `num_recv_tokens_per_rank` is in the pin memory.
        # To avoid stream synchronization by polling on the cpu to reduce the gpu bubble.
        assert num_recv_tokens_per_rank.is_cpu
        assert num_recv_tokens_per_rank.dtype == torch.int32
        max_output_token_num = 0 if min_m is None else min_m
        base_ptr = num_recv_tokens_per_rank.data_ptr()
        elem_size = num_recv_tokens_per_rank.element_size()

        for target_rank in range(self.world_size):
            # slice and item operations of the tensor are too time-consuming (10us level), so here we read directly from ptr
            while ctypes.c_int32.from_address(base_ptr + target_rank * elem_size).value == self.cpu_default_val:
                pass
            cur_output_token_num = ctypes.c_int32.from_address(base_ptr + target_rank * elem_size).value
            max_output_token_num = max(max_output_token_num, cur_output_token_num)
        if max_output_token_num > self.output_buf.shape[0]:
            alloc_token = (max_output_token_num + self.Alignment - 1) // self.Alignment * self.Alignment
            print(
                f"trigger init_output_buffer, max_output_token_num: {max_output_token_num} is larger than the output_buf shape: {self.output_buf.shape[0]}"
            )
            nvshmem_size_required = (alloc_token * self.hidden * self.dtype.itemsize * self.duplicate_comm_buffer +
                                     alloc_token * self.weight_dtype.itemsize * self.duplicate_comm_buffer +
                                     alloc_token * self.hidden * self.dtype.itemsize +
                                     alloc_token * self.weight_dtype.itemsize)
            print(
                f"nvshmem_size_required: {nvshmem_size_required} bytes, allowing NVSHMEM_SYMMETRIC_SIZE is {os.environ.get('NVSHMEM_SYMMETRIC_SIZE', None)} bytes"
            )
            # delete original buffers
            for i in range(self.duplicate_comm_buffer):
                nvshmem_free_lazy_tensor(self.output_buffers[i])
                self.output_buffers[i] = None
                nvshmem_free_lazy_tensor(self.weight_recv_buffers[i])
                self.weight_recv_buffers[i] = None
            del self.output_buffers
            del self.weight_recv_buffers
            self.output_buf = None
            del self.output_buf
            self.weight_recv_buf = None
            del self.weight_recv_buf

            nvshmem_free_lazy_tensor(self.combine_in_buf)
            del self.combine_in_buf
            nvshmem_free_lazy_tensor(self.combine_gate_in_buf)
            del self.combine_gate_in_buf

            torch.distributed.barrier(self.ep_group)

            # create new buffers
            self.output_buffers = [
                self._nvshmem_allocator.create_tensor("output_buffers_{i}", [alloc_token, self.hidden], self.dtype)
                for i in range(self.duplicate_comm_buffer)
            ]
            self.weight_recv_buffers = [
                self._nvshmem_allocator.create_tensor("weight_recv_buffers_{i}", [alloc_token], self.weight_dtype)
                for i in range(self.duplicate_comm_buffer)
            ]
            self.output_buf = self.output_buffers[self.current_comm_buffer_id]
            self.weight_recv_buf = self.weight_recv_buffers[self.current_comm_buffer_id]
            self.combine_in_buf = self._nvshmem_allocator.create_tensor("combine_in_buf", [alloc_token, self.hidden],
                                                                        self.dtype)
            self.combine_gate_in_buf = self._nvshmem_allocator.create_tensor("combine_gate_in_buf", [
                alloc_token,
            ], self.weight_dtype)

            torch.distributed.barrier(self.ep_group)

        cur_output_token_num = ctypes.c_int32.from_address(base_ptr + self.rank * elem_size).value
        output_buf, weight_recv_buf = self.output_buf[:cur_output_token_num], self.weight_recv_buf[:
                                                                                                   cur_output_token_num]
        return output_buf, weight_recv_buf

    def preprocess(
        self,
        exp_indices: torch.Tensor,
        full_scatter_indices: Optional[torch.Tensor] = None,
        local_scatter_indices: Optional[torch.Tensor] = None,
    ):
        assert self.topk_indices_buf.dtype == self.send_reqs_for_nodes.dtype
        num_dispatch_token_cur_rank = exp_indices.shape[0]
        self.topk_indices_buf[self.node_id, :num_dispatch_token_cur_rank].copy_(exp_indices)
        if local_scatter_indices is not None:
            assert local_scatter_indices.shape[0] == num_dispatch_token_cur_rank
            self.full_local_scatter_indices_buf[self.node_id, :num_dispatch_token_cur_rank].copy_(local_scatter_indices)
            full_local_scatter_indices = self.full_local_scatter_indices_buf
        else:
            full_local_scatter_indices = None
        get_dispatch_send_reqs(exp_indices, self.send_reqs_for_nodes, self.experts_per_rank, self.local_world_size,
                               self.num_sm)

        # assume that the expert indices of the drop token is num_tot_experts,
        # it will be counted in the `local_splits_buf[num_tot_experts]`
        bincount(exp_indices.view(-1), length=self.local_splits_buf.shape[0], output=self.local_splits_buf,
                 num_sm=self.num_sm)
        (
            recv_buf_offset_per_expert,
            recv_buf_tokens_per_expert,
            num_recv_tokens_per_rank,
            num_input_tokens_per_rank,
            token_dst_scatter_idx,
            reversed_token_scatter_idx,
            send_reqs_for_nodes,
            send_reqs_recv_tensor,
            topk_indices_tensor,
            _,  # non_drop_token_count_tensor,
            token_sort_indices,
        ) = get_ag_splits_and_recv_offset_for_dispatch(
            exp_indices.shape[0],
            self.send_reqs_for_nodes,
            self.send_reqs_recv_bufs,
            self.topk_indices_buf,
            self.expert_indices_signal_buf,
            self.local_splits_buf,
            self.full_splits_buf,
            self.splits_signal_buf,
            self.mega_reversed_token_scatter_idx_buf,
            self.topk,
            self.local_world_size,
            self.world_size,
            self.max_tokens,
            self.experts_per_rank,
            full_global_scatter_indices=full_scatter_indices,
            full_local_scatter_indices=full_local_scatter_indices,
            cpu_default_val=self.cpu_default_val,
            offset_dtype=self.offset_dtype,
            num_sm=self.num_sm,
            need_reversed_token_scatter_idx=self.need_reversed_token_scatter_idx,
            need_non_drop_token_count_buf=False,
        )

        ep_a2a_layout_desc = EPAllToAllLayoutDesc(
            num_dispatch_token_cur_rank=num_dispatch_token_cur_rank,
            recv_buf_offset_per_expert=recv_buf_offset_per_expert,
            recv_buf_tokens_per_expert=recv_buf_tokens_per_expert,
            num_recv_tokens_per_rank=num_recv_tokens_per_rank,
            num_input_tokens_per_rank=num_input_tokens_per_rank,
            send_reqs_for_nodes=send_reqs_for_nodes,
            send_reqs_recv_tensor=send_reqs_recv_tensor,
            topk_indices_tensor=topk_indices_tensor,
            non_drop_token_count_tensor=None,  # non_drop_token_count_tensor,
            token_dst_scatter_idx=token_dst_scatter_idx,
            reversed_token_scatter_idx=reversed_token_scatter_idx,
            token_sort_indices=token_sort_indices,
        )

        return ep_a2a_layout_desc

    def dispatch_postprocess(self):
        self.expert_indices_signal_buf.fill_(0)
        self.local_splits_buf.fill_(0)
        self.signal_buf.zero_()
        self.splits_signal_buf.zero_()
        self.counter.zero_()
        self.send_reqs_for_nodes.fill_(-1)
        self.full_splits_buf.fill_(0)
        self.topk_indices_buf.fill_(-1)
        self.mega_dispatch_counter_buf.fill_(0)
        self.mega_dispatch_barrier_buf.fill_(0)
        self.mega_token_rank_table_buf.fill_(-1)
        self.mega_token_indirect_pos_buf.fill_(-1)
        if self.mega_reversed_token_scatter_idx_buf is not None:
            self.mega_reversed_token_scatter_idx_buf.fill_(-1)

    def combine_preprocess(self, M_recv=None):
        if M_recv is not None:
            s1 = slice(0, M_recv * self.topk * self.local_world_size)
        else:
            s1 = slice(None)
        self.mega_combine_counter_buf[s1].fill_(0)
        self.mega_combine_barrier_buf[s1].fill_(0)
        self.mega_combine_scatter_output_barrier_buf.fill_(-1)
        self.intra_node_dispatch_skipped_token_mapping_indices.fill_(-1)

    def ep_barrier_all(self):
        barrier_all_on_stream(self.barrier_all_ctx, torch.cuda.current_stream())

    def mega_preprocess_group_gemm(
        self,
        gemm_input_data: torch.Tensor,
        gemm_weight: torch.Tensor,
        gemm_expert_ids: torch.Tensor,
        gemm_split_size: torch.Tensor,
        gemm_split_size_cum: torch.Tensor,
        gemm_tile_num: torch.Tensor,
        gemm_tile_num_cum: torch.Tensor,
        gemm_input_reduce_last_dim: bool,
        gemm_weight_reduce_last_dim: bool,
    ):
        # for group gemm
        if not gemm_input_reduce_last_dim:
            assert not gemm_weight_reduce_last_dim
            gemm_K, gemm_M = gemm_input_data.shape
            gemm_G, gemm_K_, gemm_N = gemm_weight.shape
            gemm_input_stride1 = gemm_input_data.stride(0)
            gemm_input_stride0 = gemm_input_data.stride(1)
            gemm_weight_stride1 = gemm_weight.stride(2)
            gemm_weight_stride2 = gemm_weight.stride(1)
            gemm_weight_stride1 = gemm_weight.stride(2)
            gemm_weight_stride2 = gemm_weight.stride(1)
        elif not gemm_weight_reduce_last_dim:
            gemm_M, gemm_K = gemm_input_data.shape
            gemm_G, gemm_K_, gemm_N = gemm_weight.shape
            gemm_input_stride0 = gemm_input_data.stride(0)
            gemm_input_stride1 = gemm_input_data.stride(1)
            gemm_weight_stride1 = gemm_weight.stride(2)
            gemm_weight_stride2 = gemm_weight.stride(1)
        else:
            gemm_M, gemm_K = gemm_input_data.shape
            gemm_G, gemm_N, gemm_K_ = gemm_weight.shape
            gemm_input_stride0 = gemm_input_data.stride(0)
            gemm_input_stride1 = gemm_input_data.stride(1)
            gemm_weight_stride1 = gemm_weight.stride(1)
            gemm_weight_stride2 = gemm_weight.stride(2)

        assert gemm_K == gemm_K_

        M_grid = triton.cdiv(gemm_M, GROUP_GEMM_BLOCK_SIZE_M) + gemm_G
        assert gemm_expert_ids.shape[
            0] >= M_grid, f"expert_ids.shape[0] ({gemm_expert_ids.shape[0]}) must be >= M_grid ({M_grid})"
        assert gemm_split_size.shape[
            0] == gemm_G, f"split_size.shape[0] ({gemm_split_size.shape[0]}) must be == G ({gemm_G})"
        assert gemm_split_size_cum.shape[
            0] >= M_grid, f"split_size_cum.shape[0] ({gemm_split_size_cum.shape[0]}) must be >= M_grid ({M_grid})"
        assert gemm_tile_num.shape[
            0] >= M_grid, f"tile_num.shape[0] ({gemm_tile_num.shape[0]}) must be >= M_grid ({M_grid})"
        assert gemm_tile_num_cum.shape[
            0] >= M_grid, f"tile_num_cum.shape[0] ({gemm_tile_num_cum.shape[0]}) must be >= M_grid ({M_grid})"

        problem_shape = (gemm_G, gemm_M, gemm_N, gemm_K)
        input_strides = (gemm_input_stride0, gemm_input_stride1)
        weight_strides = (gemm_weight.stride(0), gemm_weight_stride1, gemm_weight_stride2)
        return problem_shape, input_strides, weight_strides, M_grid

    def mega_dispatch_group_gemm(
        self,
        # dispatch token
        input: torch.Tensor,
        exp_indices: torch.Tensor,
        ep_a2a_layout_desc: EPAllToAllLayoutDesc,

        # group gemm
        gemm_weight,
        gemm_expert_ids,
        gemm_split_size,
        gemm_split_size_cum,
        gemm_tile_num,
        gemm_tile_num_cum,
        gemm_num_tiles_total,
        gemm_expert_offs,

        # dispatch token
        weight=None,
        with_cpy_flag=True,
        comm_buffer_id: int = 0,
        optional_sm: Optional[int] = None,
        num_tail_sms: int = 0,

        # group gemm
        gemm_input_reduce_last_dim=True,
        gemm_weight_reduce_last_dim=True,
        gemm_output_data=None,
        gemm_BLOCK_SIZE_N: int = 256,
        gemm_BLOCK_SIZE_K: int = 64,
        gemm_GROUP_SIZE_M: int = 3,
        gemm_num_stages=3,

        # common
        use_block_wise_barrier=False,
        num_warps=16,
        enable_profiler=False,
        profile_file_name: str = "mega_dispatch_group_gemm",
    ):
        assert self.nnodes == 1, "Mega dispatch only support single node for now"
        assert input.is_contiguous()
        assert exp_indices.is_contiguous()
        assert input.dtype == self.dtype
        assert exp_indices.dtype == self.offset_dtype
        assert len(
            exp_indices.shape) == 2 and exp_indices.shape[0] == input.shape[0] and exp_indices.shape[1] == self.topk
        token_num, N = input.shape
        assert N == self.hidden

        NUM_DISPATCH_SM = optional_sm if optional_sm is not None else self.num_sm

        # for flux's multi buffers
        if comm_buffer_id != self.current_comm_buffer_id:  # switch buffers
            self.current_comm_buffer_id = comm_buffer_id
            self.send_buf = self.comm_buffers[comm_buffer_id]
            self.output_buf = self.output_buffers[comm_buffer_id]
            self.weight_recv_buf = self.weight_recv_buffers[comm_buffer_id]
        if with_cpy_flag:
            self.send_buf[self.node_id, :token_num].copy_(input)

        has_weight = (weight is not None)
        if has_weight:
            assert weight.shape[0] == token_num
            assert weight.shape[1] == self.topk
            assert weight.is_contiguous()
            assert weight.dtype == self.weight_dtype, f"weight.dtype: {weight.dtype}, self.weight_dtype: {self.weight_dtype}"
            self.weight_send_buf[self.node_id, :token_num].copy_(weight)

        assert ep_a2a_layout_desc is not None

        dispatch_output_buf, weight_recv_buf = self.init_output_buffer(ep_a2a_layout_desc.num_recv_tokens_per_rank)

        grid = lambda meta: (self.MAX_SMS, )
        token_dst_scatter_idx = ep_a2a_layout_desc.token_dst_scatter_idx
        if token_dst_scatter_idx is None:
            with_scatter_indices = False
            token_dst_scatter_idx = torch.empty((self.nnodes, self.max_tokens, self.topk), dtype=self.offset_dtype,
                                                device=ep_a2a_layout_desc.recv_buf_offset_per_expert.device)
        else:
            assert len(token_dst_scatter_idx.shape) == 3
            assert token_dst_scatter_idx.shape[0] == self.nnodes
            assert token_dst_scatter_idx.shape[1] == self.max_tokens
            assert token_dst_scatter_idx.shape[2] == self.topk
            assert token_dst_scatter_idx.dtype == self.offset_dtype
            assert token_dst_scatter_idx.is_contiguous()
            with_scatter_indices = True

        gemm_input_data = dispatch_output_buf
        dispatch_output_local = torch.empty_like(dispatch_output_buf)

        gemm_problem_shape, gemm_input_strides, gemm_weight_strides, gemm_M_grid = self.mega_preprocess_group_gemm(
            gemm_input_data,
            gemm_weight,
            gemm_expert_ids,
            gemm_split_size,
            gemm_split_size_cum,
            gemm_tile_num,
            gemm_tile_num_cum,
            gemm_input_reduce_last_dim,
            gemm_weight_reduce_last_dim,
        )

        if gemm_output_data is None:
            gemm_output_data = torch.empty([gemm_problem_shape[1], gemm_problem_shape[2]], dtype=gemm_input_data.dtype,
                                           device=gemm_input_data.device)
        else:
            gemm_output_M_, gemm_output_N_ = gemm_output_data.shape
            assert gemm_problem_shape[1] == gemm_output_M_
            assert gemm_problem_shape[2] == gemm_output_N_

        # for mega kernel
        # fill_tensor(self._task_counter_buf, 0, 1)
        self._task_counter_buf.zero_()

        if enable_profiler:
            tasks_names = [
                "dispatch_token_main", "dispatch_token_tail_notify", "group_gemm_wait", "group_gemm_preprocess",
                "group_gemm_main", "no_notify"
            ]
            max_buf_slots = self.MAX_SMS * (
                NUM_DISPATCH_SM +
                (len(tasks_names) + gemm_M_grid) * triton.cdiv(N, gemm_BLOCK_SIZE_N)) * len(tasks_names) * 100
            print("mega dispatch profile buffer slots: ", max_buf_slots)
            profiler_buffer = ProfilerBuffer(max_num_profile_slots=max_buf_slots, trace_file=profile_file_name,
                                             task_names=tasks_names).profiler_buffer
        else:
            profiler_buffer = torch.empty([1], dtype=torch.uint64, device=torch.cuda.current_device())

        assert self.mega_dispatch_barrier_buf.shape[
            0] >= gemm_M_grid, f"mega_dispatch_barrier_buf.shape[0] ({self.mega_dispatch_barrier_buf.shape[0]}) must be >= gemm_M_grid ({gemm_M_grid})"

        mega_kernel_dispatch_token_moe_grouped_gemm[grid](
            self._task_counter_buf,

            # dispatch token params
            ep_a2a_layout_desc.recv_buf_offset_per_expert,
            self.local_splits_buf,
            self.send_buf,  # recv token from other nodes
            self.output_buf,
            self.weight_send_buf,
            self.weight_recv_buf,
            ep_a2a_layout_desc.topk_indices_tensor,  # [nnodes, max_tokens, topk]
            token_dst_scatter_idx,  # [self.nnodes, self.max_tokens, self.topk]
            ep_a2a_layout_desc.num_input_tokens_per_rank,  # [world_size]
            ep_a2a_layout_desc.recv_buf_tokens_per_expert,  # [world_size, experts_per_rank]
            ep_a2a_layout_desc.token_sort_indices,  # [nndoes, max_tokens * topk]
            self.topk,
            self.hidden,
            self.experts_per_rank,
            has_weight,  # HAS_WEIGHT
            with_scatter_indices,  # WITH_SCATTER_INDICES

            #
            NUM_DISPATCH_SM,  # num_dispatch_sms

            # grouped gemm params
            gemm_input_data,
            gemm_weight,
            gemm_output_data,
            gemm_expert_ids,
            gemm_split_size,
            gemm_split_size_cum,
            gemm_tile_num,
            gemm_tile_num_cum,
            gemm_num_tiles_total,
            gemm_expert_offs,
            *gemm_problem_shape,
            *gemm_input_strides,
            *gemm_weight_strides,
            gemm_output_data.stride(0),
            gemm_output_data.stride(1),
            GROUP_GEMM_BLOCK_SIZE_M,
            gemm_BLOCK_SIZE_N,
            gemm_BLOCK_SIZE_K,
            gemm_GROUP_SIZE_M,

            # dispatch local output
            dispatch_output_local,

            #
            self.mega_dispatch_counter_buf,  # local buf [num_ranks, num_experts_per_rank] = [num_experts]
            self.mega_dispatch_barrier_buf,  # symm buf [num_experts_per_rank, num_ranks] = [num_experts]
            self.mega_token_rank_table_buf,  # local buf [max_tokens, local_world_size]
            self.mega_token_indirect_pos_buf,  # symm buf [max_tokens * topk * local_world_size]
            USE_BLOCK_WISE_BARRIER=use_block_wise_barrier,
            NUM_WARPS=num_warps,
            NUM_TAIL_SMS=num_tail_sms,
            num_warps=num_warps,
            num_stages=gemm_num_stages,
            profiler_buffer=profiler_buffer,
            ENABLE_PROFILING=enable_profiler,
        )

        if enable_profiler:
            os.makedirs("prof/mega", exist_ok=True)
            export_to_perfetto_trace(profiler_buffer, tasks_names, f"prof/mega/{profile_file_name}_rank_{self.rank}")

        if not with_scatter_indices:
            ep_a2a_layout_desc.token_dst_scatter_idx = token_dst_scatter_idx
        self.ep_barrier_all()
        self.dispatch_postprocess()
        self.ep_barrier_all()

        dispatch_res = dispatch_output_buf
        weight_res = weight_recv_buf

        if num_tail_sms <= 0:
            # one-stage dispatch does not do checkpoint, do it additionally
            copy_tensor(dispatch_output_local, dispatch_res, persistent=False)

        return dispatch_output_local, weight_res, ep_a2a_layout_desc, gemm_output_data

    def mega_group_gemm_combine(
        self,
        # group gemm
        gemm_input_data,
        gemm_weight,
        gemm_expert_ids,
        gemm_split_size,
        gemm_split_size_cum,
        gemm_tile_num,
        gemm_tile_num_cum,
        gemm_num_tiles_total,

        # combine token
        ep_a2a_layout_desc: EPAllToAllLayoutDesc,

        # group gemm
        gemm_input_reduce_last_dim=True,
        gemm_weight_reduce_last_dim=True,
        gemm_BLOCK_SIZE_N: int = 256,
        gemm_BLOCK_SIZE_K: int = 64,
        gemm_GROUP_SIZE_M: int = 3,
        gemm_num_stages=3,

        # combine token
        gate_input: Optional[torch.Tensor] = None,
        cp_flag: bool = True,
        combine_output: Optional[torch.Tensor] = None,
        output_gate: Optional[torch.Tensor] = None,
        optional_sm: Optional[int] = None,
        num_reduce_sms: int = 0,
        optional_signal_tensor: Optional[torch.Tensor] = None,
        # num_scatter_warps: int = 22,
        # num_reduce_warps: int = 10,
        num_warps: int = 32,
        combine_mode: str = "serial",

        # transposed group gemm params
        grad_output=None,
        orig_input=None,
        grad_weight=None,
        split_size_cum_per_expert=None,
        grad_BLOCK_SIZE_M=64,
        grad_BLOCK_SIZE_N=128,
        grad_BLOCK_SIZE_K=256,
        grad_GROUP_SIZE_M=3,
        enable_profiler: bool = False,
        profile_file_name: str = "mega_group_gemm_combine",
    ):
        assert self.nnodes == 1, "Mega dispatch only support single node for now"
        gemm_problem_shape, gemm_input_strides, gemm_weight_strides, gemm_M_grid = self.mega_preprocess_group_gemm(
            gemm_input_data,
            gemm_weight,
            gemm_expert_ids,
            gemm_split_size,
            gemm_split_size_cum,
            gemm_tile_num,
            gemm_tile_num_cum,
            gemm_input_reduce_last_dim,
            gemm_weight_reduce_last_dim,
        )
        gemm_G, gemm_M, gemm_N, gemm_K = gemm_problem_shape

        # gemm output data is in combine buf
        gemm_output_data = self.combine_in_buf[:gemm_M, :].view(gemm_M, gemm_N)

        # different combine mode has different profile tasks
        assert combine_mode in ["serial", "fuse_scatter"]

        COMBINE_SM = optional_sm if optional_sm is not None else self.num_sm
        MEGA_SMS = self.MAX_SMS
        M = ep_a2a_layout_desc.num_dispatch_token_cur_rank
        N = self.hidden

        assert M <= self.max_tokens, f"M ({M}) must be <= self.max_tokens ({self.max_tokens})"
        # need to clear this buffer due to final reduce sum
        # fill_tensor(self.combine_out_buf.view(self.max_tokens, self.hidden)[:M, :], 0, MEGA_SMS)
        self.combine_out_buf.view(self.max_tokens, self.hidden)[:M, :].zero_()
        if combine_mode == "fuse_scatter":
            # need to clear this buffer due to drop token
            # fill_tensor(self.mega_combine_scatter_output_buf[:M * self.topk], 0, MEGA_SMS)
            self.mega_combine_scatter_output_buf[:M * self.topk].zero_()
        has_gate = gate_input is not None
        if has_gate:
            fill_tensor(
                self.intra_node_gate_buf,
                0,
            )
            fill_tensor(self.combine_gate_out_buf, 0, MEGA_SMS)
        # no need to copy input data
        if cp_flag and has_gate:
            copy_tensor(self.combine_gate_in_buf[:gate_input.shape[0]], gate_input, MEGA_SMS)
        if optional_signal_tensor is not None:
            fill_tensor(optional_signal_tensor, 0, MEGA_SMS)

        self.combine_preprocess()

        grid = lambda meta: (MEGA_SMS, )
        # for mega kernel
        # fill_tensor(self._task_counter_buf, 0, 1)
        self._task_counter_buf.zero_()

        # if tranposed group gemm is needed, we need to create the grad weight tensor
        if grad_output is not None:
            with_grad = True
            assert orig_input is not None
            grad_M, grad_N = grad_output.shape
            grad_M_, grad_K = orig_input.shape
            assert grad_M == grad_M_
            assert split_size_cum_per_expert is not None
            assert split_size_cum_per_expert.shape[0] == gemm_G
            if grad_weight is None:
                grad_weight = torch.empty([gemm_G, grad_N, grad_K], dtype=grad_output.dtype, device=grad_output.device)
            else:
                G_, N_, K_ = grad_weight.shape
                assert gemm_G == G_
                assert grad_N == N_
                assert grad_K == K_
        else:
            with_grad = False

        # initialize the profiler buffer
        if enable_profiler:
            if combine_mode == "serial":
                tasks_names = ["group_gemm", "barrier_all", "combine_token"
                               ] + (["transposed_group_gemm"] if with_grad else [])
            else:
                tasks_names = [
                    "combine_scatter_token", "combine_topk_reduce", "no_wait", "group_gemm_preprocess",
                    "group_gemm_main", "group_gemm_tail_notify"
                ] + (["transposed_group_gemm_preprocess", "transposed_group_gemm_main"] if with_grad else [])
            max_buf_slots = self.MAX_SMS * (
                COMBINE_SM +
                (len(tasks_names) + gemm_M_grid) * triton.cdiv(N, gemm_BLOCK_SIZE_N)) * len(tasks_names) * 100
            print("mega combine profile buffer slots: ", max_buf_slots)
            profiler_buffer = ProfilerBuffer(max_num_profile_slots=max_buf_slots, trace_file=profile_file_name,
                                             task_names=tasks_names).profiler_buffer
        else:
            profiler_buffer = torch.empty([1], dtype=torch.uint64, device=torch.cuda.current_device())

        grid_barrier_workspace = torch.empty((1, ), dtype=torch.int32, device=torch.cuda.current_device())
        # fill_tensor(grid_barrier_workspace, 0, 1)
        grid_barrier_workspace.zero_()

        self.ep_barrier_all()

        if not with_grad:
            mega_kernel_moe_grouped_gemm_combine_token[grid](
                self._task_counter_buf,

                # grouped gemm params
                gemm_input_data,
                gemm_weight,
                gemm_output_data,
                gemm_expert_ids,
                gemm_split_size,
                gemm_split_size_cum,
                gemm_tile_num,
                gemm_tile_num_cum,
                gemm_num_tiles_total,
                gemm_M,
                gemm_N,
                gemm_K,
                *gemm_input_strides,
                *gemm_weight_strides,
                gemm_output_data.stride(0),
                gemm_output_data.stride(1),
                GROUP_GEMM_BLOCK_SIZE_M,
                gemm_BLOCK_SIZE_N,
                gemm_BLOCK_SIZE_K,
                gemm_GROUP_SIZE_M,

                # combine token params
                ep_a2a_layout_desc.num_input_tokens_per_rank,  # [world_size]
                ep_a2a_layout_desc.num_recv_tokens_per_rank,  # [world_size]
                self.combine_in_buf,  # symm buffer (recv token in dispatch stage)
                self.mega_combine_scatter_output_buf,  # symm buffer [max_tokens, topk, hidden]
                self.mega_combine_scatter_output_barrier_buf if num_reduce_sms > 0 else None,  # [max_tokens, topk, ]
                self.combine_out_buf,  #[max_tokens, hidden]
                self.combine_gate_in_buf,  # symm buffer [dynamic_num_of_tokens]
                self.combine_gate_out_buf,  # symm buffer [max_tokens, topk]
                ep_a2a_layout_desc.topk_indices_tensor,  # [max_tokens, topk]
                ep_a2a_layout_desc.token_dst_scatter_idx,  # [max_tokens, topk]
                ep_a2a_layout_desc.reversed_token_scatter_idx,  # [max_tokens, topk, 2]
                # ep_a2a_layout_desc.non_drop_token_count_tensor,  # [max_tokens, ]
                self.topk,
                self.hidden,
                self.experts_per_rank,
                has_gate,
                combine_mode == "fuse_scatter",  # USE_SCATTER_MODE

                #
                COMBINE_SM,
                num_reduce_sms,

                #
                self.mega_combine_counter_buf,
                self.mega_combine_barrier_buf,
                self.barrier_all_workspace,
                grid_barrier_workspace,
                num_warps,
                profiler_buffer,
                ENABLE_PROFILING=enable_profiler,

                # num_warps=(num_scatter_warps + num_reduce_warps),
                num_warps=num_warps,
                num_stages=gemm_num_stages,
            )
        else:  # with transposed group gemm
            mega_kernel_moe_grouped_gemm_combine_token_transposed_grouped_gemm[grid](
                self._task_counter_buf,

                # grouped gemm params
                gemm_input_data,
                gemm_weight,
                gemm_output_data,
                gemm_expert_ids,
                gemm_split_size,
                gemm_split_size_cum,
                gemm_tile_num,
                gemm_tile_num_cum,
                gemm_num_tiles_total,
                gemm_G,
                gemm_M,
                gemm_N,
                gemm_K,
                *gemm_input_strides,
                *gemm_weight_strides,
                gemm_output_data.stride(0),
                gemm_output_data.stride(1),
                GROUP_GEMM_BLOCK_SIZE_M,
                gemm_BLOCK_SIZE_N,
                gemm_BLOCK_SIZE_K,
                gemm_GROUP_SIZE_M,

                # combine token params
                ep_a2a_layout_desc.num_input_tokens_per_rank,  # [world_size]
                ep_a2a_layout_desc.num_recv_tokens_per_rank,  # [world_size]
                self.combine_in_buf,  # symm buffer (recv token in dispatch stage)
                self.mega_combine_scatter_output_buf,  # symm buffer [max_tokens, topk, hidden]
                self.mega_combine_scatter_output_barrier_buf if num_reduce_sms > 0 else None,  # [max_tokens, topk, ]
                self.combine_out_buf,  #[max_tokens, hidden]
                self.combine_gate_in_buf,  # symm buffer [dynamic_num_of_tokens]
                self.combine_gate_out_buf,  # symm buffer [max_tokens, topk]
                ep_a2a_layout_desc.topk_indices_tensor,  # [max_tokens, topk]
                ep_a2a_layout_desc.token_dst_scatter_idx,  # [max_tokens, topk]
                ep_a2a_layout_desc.reversed_token_scatter_idx,  # [max_tokens, topk, 2]
                # ep_a2a_layout_desc.non_drop_token_count_tensor,  # [max_tokens, ]
                self.topk,
                self.hidden,
                self.experts_per_rank,
                has_gate,
                combine_mode == "fuse_scatter",  # USE_SCATTER_MODE

                #
                COMBINE_SM,
                num_reduce_sms,

                # transposed group gemm params
                grad_output,
                orig_input,
                grad_weight,
                split_size_cum_per_expert,
                grad_N,
                grad_K,
                grad_output.stride(0),
                grad_output.stride(1),
                orig_input.stride(0),
                orig_input.stride(1),
                grad_weight.stride(0),
                grad_weight.stride(1),
                grad_weight.stride(2),
                grad_BLOCK_SIZE_M,
                grad_BLOCK_SIZE_N,
                grad_BLOCK_SIZE_K,
                grad_GROUP_SIZE_M,

                #
                self.mega_combine_counter_buf,
                self.mega_combine_barrier_buf,
                self.barrier_all_workspace,
                grid_barrier_workspace,
                num_warps,
                profiler_buffer,
                ENABLE_PROFILING=enable_profiler,
                num_warps=num_warps,
                num_stages=gemm_num_stages,
            )

        torch.cuda.current_stream().synchronize()
        self.ep_barrier_all()

        reduce_buf = self.combine_out_buf
        reduce_gate_buf = self.combine_gate_out_buf

        if enable_profiler:
            os.makedirs("prof/mega", exist_ok=True)
            export_to_perfetto_trace(profiler_buffer, tasks_names, f"prof/mega/{profile_file_name}_rank_{self.rank}")

        reduce_buf = reduce_buf.view(self.max_tokens, self.hidden)[:M]
        reduce_inter_node = torch.empty([M, N], dtype=reduce_buf.dtype, device=reduce_buf.device)
        copy_tensor(reduce_inter_node, reduce_buf, MEGA_SMS)
        if has_gate:
            reduce_gate_buf = reduce_gate_buf.view(self.max_tokens, self.topk)[:M]
            reduce_gate_inter_node = torch.empty([M, self.topk], dtype=reduce_gate_buf.dtype,
                                                 device=reduce_gate_buf.device)
            copy_tensor(reduce_gate_inter_node, reduce_gate_buf, MEGA_SMS)

        output_ret = None
        if combine_output is None:
            output_ret = reduce_inter_node
        else:
            copy_tensor(combine_output, reduce_inter_node, MEGA_SMS)
            output_ret = combine_output

        if has_gate:
            if output_gate is None:
                gate_ret = reduce_gate_inter_node
            else:
                copy_tensor(output_gate, reduce_gate_inter_node, MEGA_SMS)
                gate_ret = output_gate
            if grad_weight is None:
                return output_ret, gate_ret
            else:
                return output_ret, gate_ret, grad_weight
        else:
            if grad_weight is None:
                return output_ret
            else:
                return output_ret, grad_weight
