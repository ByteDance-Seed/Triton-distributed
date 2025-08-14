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
import ctypes
from typing import Union

from triton_dist.kernels.nvidia.ep_a2a import (
    kernel_combine_token,
    kernel_dispatch_token,
    bincount,
    get_dispatch_send_reqs_for_target_node,
    get_ag_splits_and_recv_offset_for_dispatch,
)
from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE, nvshmem_barrier_all_on_stream, nvshmem_free_tensor_sync, nvshmem_create_tensor


@dataclasses.dataclass
class EPAllToAllLayoutDesc:
    num_dispatch_token_cur_rank: int
    num_input_tokens_per_rank: torch.Tensor
    send_reqs_recv_tensor: torch.Tensor
    topk_indices_tensor: torch.Tensor
    token_dst_scatter_idx: torch.Tensor


class EPAll2AllLayer(torch.nn.Module):

    def __init__(
        self,
        ep_group,
        max_tokens: int,
        hidden: int,
        topk: int,
        rank: int,
        num_tot_experts: int,
        local_world_size: int,
        world_size: int,
        dtype=torch.bfloat16,
        weight_dtype=torch.float32,
        num_sm=20,
    ):
        super().__init__()
        self.offset_dtype = torch.int32
        self.ep_group = ep_group
        self.num_sm = num_sm

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

        # for dispatch
        self.send_reqs_for_nodes = nvshmem_create_tensor([self.nnodes, 2, max_tokens], self.offset_dtype)
        self.send_reqs_for_nodes.fill_(-1)
        self.send_reqs_recv_bufs = nvshmem_create_tensor([self.nnodes, 2, max_tokens], self.offset_dtype)
        self.send_reqs_recv_bufs.fill_(-1)
        self.Alignment = 1024

        avg_tokens = max_tokens * topk

        self.send_buf = nvshmem_create_tensor([self.nnodes, max_tokens, hidden], dtype)
        self.output_buf = nvshmem_create_tensor([avg_tokens * 2, hidden], dtype)
        self.weight_recv_buf = nvshmem_create_tensor([avg_tokens * 2], weight_dtype)
        self.signal_buf = nvshmem_create_tensor((world_size, ), NVSHMEM_SIGNAL_DTYPE)
        self.signal_buf.fill_(0)
        self.topk_indices_buf = nvshmem_create_tensor([self.nnodes, max_tokens, topk], self.offset_dtype)
        self.weight_send_buf = nvshmem_create_tensor([self.nnodes, max_tokens, topk], weight_dtype)
        self.counter = torch.zeros((self.nnodes, ), dtype=torch.int32).cuda()
        # dispatch preprocess, use push mode to reduce barrier_all
        # local_splits_buf[num_tot_experts] is used for drop token
        self.local_splits_buf = nvshmem_create_tensor([self.num_tot_experts + 1], self.offset_dtype)
        self.local_splits_buf.fill_(0)
        self.full_splits_buf = nvshmem_create_tensor([world_size, num_tot_experts + 1], self.offset_dtype)
        self.num_input_tokens_per_rank_comm_buf = nvshmem_create_tensor([
            world_size,
        ], self.offset_dtype)
        self.splits_signal_buf = nvshmem_create_tensor((world_size, ), NVSHMEM_SIGNAL_DTYPE)
        self.splits_signal_buf.fill_(0)
        self.expert_indices_signal_buf = nvshmem_create_tensor((world_size, ), NVSHMEM_SIGNAL_DTYPE)
        self.expert_indices_signal_buf.fill_(0)
        self.cpu_default_val = -1

        # for combine
        self.intra_node_reduce_buf = nvshmem_create_tensor([self.nnodes, max_tokens, hidden], dtype)

        nvshmem_barrier_all_on_stream()
        torch.cuda.synchronize()

    def finalize(self):
        nvshmem_free_tensor_sync(self.num_input_tokens_per_rank_comm_buf)
        nvshmem_free_tensor_sync(self.expert_indices_signal_buf)
        nvshmem_free_tensor_sync(self.weight_send_buf)
        nvshmem_free_tensor_sync(self.weight_recv_buf)
        nvshmem_free_tensor_sync(self.send_reqs_for_nodes)
        nvshmem_free_tensor_sync(self.send_reqs_recv_bufs)
        nvshmem_free_tensor_sync(self.send_buf)
        nvshmem_free_tensor_sync(self.output_buf)
        nvshmem_free_tensor_sync(self.signal_buf)
        nvshmem_free_tensor_sync(self.topk_indices_buf)
        nvshmem_free_tensor_sync(self.local_splits_buf)
        nvshmem_free_tensor_sync(self.full_splits_buf)
        nvshmem_free_tensor_sync(self.splits_signal_buf)
        nvshmem_free_tensor_sync(self.intra_node_reduce_buf)

    def splits_golden(self, exp_indices, num_experts):
        splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=num_experts).to(torch.int32)
        # drop token logic :only need the splits information for the non-dropped tokenes
        splits_gpu_cur_rank = splits_gpu_cur_rank[:num_experts]
        # following are the all2all

        ag_splits = torch.empty([self.world_size, num_experts], dtype=torch.int32, device=exp_indices.device)
        torch.distributed.all_gather_into_tensor(
            ag_splits,
            splits_gpu_cur_rank,
            group=self.ep_group,
        )
        return ag_splits

    def preprocess(self, input: torch.Tensor, exp_indices: torch.Tensor, full_scatter_indices: Union[torch.Tensor,
                                                                                                     None] = None):
        num_dispatch_token_cur_rank = exp_indices.shape[0]
        token_node_idx = exp_indices // (self.experts_per_rank * self.local_world_size)

        # TODO(zhengxuegui.0): use triton kernel to gen send requests. It takes 150us to generate a request for each node(4096 tokens top 8).
        for traget_node_id in range(self.nnodes):
            if traget_node_id == self.node_id:
                continue
            start_indices, end_indices = get_dispatch_send_reqs_for_target_node(token_node_idx, traget_node_id,
                                                                                index_type=self.offset_dtype)
            self.send_reqs_for_nodes[traget_node_id, 0, :start_indices.shape[0]].copy_(start_indices)
            self.send_reqs_for_nodes[traget_node_id, 1, :end_indices.shape[0]].copy_(end_indices)

        # assume that the expert indices of the drop token is num_tot_experts,
        # it will be counted in the `local_splits_buf[num_tot_experts]`
        _ = bincount(exp_indices.view(-1), length=self.local_splits_buf.shape[0], output=self.local_splits_buf,
                     num_sm=self.num_sm)
        recv_buf_offset_per_expert, num_recv_tokens_per_rank, num_input_tokens_per_rank, token_dst_scatter_idx, send_reqs_recv_tensor, topk_indices_tensor = get_ag_splits_and_recv_offset_for_dispatch(
            self.send_reqs_for_nodes, self.send_reqs_recv_bufs, exp_indices, self.topk_indices_buf,
            self.expert_indices_signal_buf, self.local_splits_buf, self.full_splits_buf, self.splits_signal_buf,
            self.topk, self.local_world_size, self.world_size, self.max_tokens, self.experts_per_rank,
            full_scatter_indices=full_scatter_indices, cpu_default_val=self.cpu_default_val,
            offset_dtype=self.offset_dtype, num_sm=self.num_sm)

        ep_a2a_layout_desc = EPAllToAllLayoutDesc(num_dispatch_token_cur_rank=num_dispatch_token_cur_rank,
                                                  num_input_tokens_per_rank=num_input_tokens_per_rank,
                                                  send_reqs_recv_tensor=send_reqs_recv_tensor,
                                                  topk_indices_tensor=topk_indices_tensor,
                                                  token_dst_scatter_idx=token_dst_scatter_idx)
        return recv_buf_offset_per_expert, num_recv_tokens_per_rank, ep_a2a_layout_desc

    def dispatch_postprocess(self):
        self.expert_indices_signal_buf.fill_(0)
        self.local_splits_buf.fill_(0)
        self.signal_buf.zero_()
        self.splits_signal_buf.zero_()
        self.counter.zero_()
        self.send_reqs_for_nodes.fill_(-1)
        self.full_splits_buf.fill_(0)
        self.topk_indices_buf.fill_(-1)

    def combine_postprocess(self):
        self.send_reqs_recv_bufs.fill_(0)

    def dispatch_token(self, recv_buf_offset_per_expert, ep_a2a_layout_desc: EPAllToAllLayoutDesc, has_weight=False):
        grid = lambda meta: (self.num_sm, )
        assert self.topk_indices_buf.dtype == self.send_reqs_for_nodes.dtype
        token_dst_scatter_idx = ep_a2a_layout_desc.token_dst_scatter_idx
        if token_dst_scatter_idx is None:
            with_scatter_indices = False
            token_dst_scatter_idx = torch.empty((self.nnodes, self.max_tokens, self.topk), dtype=self.offset_dtype,
                                                device=recv_buf_offset_per_expert.device)
        else:
            assert len(token_dst_scatter_idx.shape) == 3
            assert token_dst_scatter_idx.shape[0] == self.nnodes
            assert token_dst_scatter_idx.shape[1] == self.max_tokens
            assert token_dst_scatter_idx.shape[2] == self.topk
            assert token_dst_scatter_idx.dtype == self.offset_dtype
            assert token_dst_scatter_idx.is_contiguous()
            with_scatter_indices = True

        kernel_dispatch_token[grid](
            self.send_reqs_for_nodes,
            self.signal_buf,
            self.counter,
            recv_buf_offset_per_expert,
            self.send_buf,
            self.output_buf,
            self.weight_send_buf,
            self.weight_recv_buf,
            ep_a2a_layout_desc.topk_indices_tensor,
            token_dst_scatter_idx,
            ep_a2a_layout_desc.num_input_tokens_per_rank,
            self.max_tokens,
            self.topk,
            self.hidden,
            self.dtype.itemsize * self.hidden,
            self.experts_per_rank,
            self.local_world_size,
            HAS_WEIGHT=has_weight,
            WITH_SCATTER_INDICES=with_scatter_indices,
            num_warps=32,
        )

        if not with_scatter_indices:
            ep_a2a_layout_desc.token_dst_scatter_idx = token_dst_scatter_idx
        return ep_a2a_layout_desc

    def init_output_buffer(self, num_recv_tokens_per_rank):
        # `num_recv_tokens_per_rank` is in the pin memory.
        # To avoid stream synchronization by polling on the cpu to reduce the gpu bubble.
        assert num_recv_tokens_per_rank.is_cpu
        assert num_recv_tokens_per_rank.dtype == torch.int32
        max_output_token_num = 0
        base_ptr = num_recv_tokens_per_rank.data_ptr()
        elem_size = num_recv_tokens_per_rank.element_size()

        for target_rank in range(self.world_size):
            # slice and item operations of the tensor are too time-consuming (10us level), so here we read directly from ptr
            while ctypes.c_int32.from_address(base_ptr + target_rank * elem_size).value == self.cpu_default_val:
                pass
            cur_output_token_num = ctypes.c_int32.from_address(base_ptr + target_rank * elem_size).value
            max_output_token_num = max(max_output_token_num, cur_output_token_num)
        if max_output_token_num > self.output_buf.shape[0]:
            torch.distributed.barrier()
            alloc_token = (max_output_token_num + self.Alignment - 1) // self.Alignment * self.Alignment * 2
            self.output_buf = nvshmem_create_tensor([alloc_token, self.hidden], self.dtype)
            self.weight_recv_buf = nvshmem_create_tensor([
                alloc_token,
            ], self.weight_dtype)

        cur_output_token_num = ctypes.c_int32.from_address(base_ptr + self.rank * elem_size).value
        return self.output_buf[:cur_output_token_num], self.weight_recv_buf[:cur_output_token_num]

    def dispatch(self, input: torch.Tensor, exp_indices: torch.Tensor, weight=None, full_scatter_indices=None):
        assert input.is_contiguous()
        assert exp_indices.is_contiguous()
        assert input.dtype == self.dtype
        assert exp_indices.dtype == self.offset_dtype
        assert len(
            exp_indices.shape) == 2 and exp_indices.shape[0] == input.shape[0] and exp_indices.shape[1] == self.topk
        current_stream = torch.cuda.current_stream()
        token_num, N = input.shape
        assert N == self.hidden
        self.send_buf[self.node_id, :token_num].copy_(input)
        self.topk_indices_buf[self.node_id, :token_num].copy_(exp_indices)
        has_weight = (weight is not None)
        if has_weight:
            assert weight.shape[0] == token_num
            assert weight.shape[1] == self.topk
            assert weight.is_contiguous()
            assert weight.dtype == self.weight_dtype
            self.weight_send_buf[self.node_id, :token_num].copy_(weight)

        recv_buf_offset_per_expert, num_recv_tokens_per_rank, ep_a2a_layout_desc = self.preprocess(
            input, exp_indices, full_scatter_indices)

        output_buf, weight_recv_buf = self.init_output_buffer(num_recv_tokens_per_rank)
        # if full_scatter_indices is None, token_dst_scatter_idx is calc in dispatch
        ep_a2a_layout_desc = self.dispatch_token(recv_buf_offset_per_expert, ep_a2a_layout_desc, has_weight=has_weight)
        nvshmem_barrier_all_on_stream(current_stream)
        self.dispatch_postprocess()
        nvshmem_barrier_all_on_stream(current_stream)
        # This copy is redundant and is only kept for stress testing, we can remove it during integration.
        copy_out = torch.empty(output_buf.shape, dtype=output_buf.dtype, device=output_buf.device)
        copy_weight = None
        if has_weight:
            copy_weight = torch.empty(weight_recv_buf.shape, dtype=weight_recv_buf.dtype, device=weight.device)
            copy_weight.copy_(weight_recv_buf)
        copy_out.copy_(output_buf)
        return copy_out, copy_weight, ep_a2a_layout_desc

    def combine_token_intra_node_and_send(self, input: torch.Tensor, ep_a2a_layout_desc: EPAllToAllLayoutDesc):
        grid = lambda meta: (self.num_sm, )
        BLOCK_SIZE = 1 << self.hidden.bit_length()
        counter_workspace = torch.zeros((self.nnodes, ), dtype=torch.int32, device=torch.cuda.current_device())
        kernel_combine_token[grid](
            counter_workspace,
            ep_a2a_layout_desc.num_input_tokens_per_rank,
            ep_a2a_layout_desc.send_reqs_recv_tensor,
            self.intra_node_reduce_buf,
            input,
            self.send_buf,
            ep_a2a_layout_desc.topk_indices_tensor,
            ep_a2a_layout_desc.token_dst_scatter_idx,
            self.max_tokens,
            self.topk,
            self.hidden,
            input.dtype.itemsize * self.hidden,
            self.experts_per_rank,
            self.local_world_size,
            BLOCK_SIZE,
            num_warps=32,
        )
        return self.send_buf

    def combine(self, input, ep_a2a_layout_desc: EPAllToAllLayoutDesc):
        assert input.is_contiguous()
        assert input.dtype == self.dtype
        current_stream = torch.cuda.current_stream()
        self.send_buf.fill_(0)
        self.output_buf[:input.shape[0]].copy_(input)
        nvshmem_barrier_all_on_stream(current_stream)
        reduce_buf = self.combine_token_intra_node_and_send(self.output_buf, ep_a2a_layout_desc)
        nvshmem_barrier_all_on_stream(current_stream)
        reduce_inter_node = reduce_buf.reshape(self.nnodes, self.max_tokens, self.hidden).sum(dim=0)
        return reduce_inter_node[:ep_a2a_layout_desc.num_dispatch_token_cur_rank]
