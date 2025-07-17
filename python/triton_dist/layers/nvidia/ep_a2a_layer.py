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

import ctypes

from triton_dist.kernels.nvidia.ep_a2a import (
    kernel_combine_token,
    kernel_dispatch_token,
    bincount,
    get_dispatch_send_reqs_for_target_node,
    get_ag_splits_and_recv_offset_for_dispatch,
)
from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE, nvshmem_barrier_all_on_stream, nvshmem_free_tensor_sync, nvshmem_create_tensor


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
        self.signal_buf = nvshmem_create_tensor((world_size, ), NVSHMEM_SIGNAL_DTYPE)
        self.signal_buf.fill_(0)
        self.top_indices_buf = nvshmem_create_tensor([self.nnodes, max_tokens, topk], self.offset_dtype)
        self.counter = torch.zeros((self.nnodes, ), dtype=torch.int32).cuda()
        # dispatch preprocess, use push mode to reduce barrier_all
        self.local_splits_buf = nvshmem_create_tensor([self.num_tot_experts], self.offset_dtype)
        self.local_splits_buf.fill_(0)
        self.full_splits_buf = nvshmem_create_tensor([world_size, num_tot_experts], self.offset_dtype)
        self.splits_signal_buf = nvshmem_create_tensor((world_size, ), NVSHMEM_SIGNAL_DTYPE)
        self.splits_signal_buf.fill_(0)
        self.cpu_default_val = -1

        # for combine
        self.token_dst_scatter_idx = torch.empty((self.nnodes, self.max_tokens, self.topk),
                                                 dtype=self.offset_dtype).cuda()
        self.num_dispatch_token_cur_rank = None  # save in dispatch stage
        self.num_input_tokens_per_rank = None

        self.intra_node_reduce_buf = nvshmem_create_tensor([self.nnodes, max_tokens, hidden], dtype)

    def finalize(self):
        nvshmem_free_tensor_sync(self.send_reqs_for_nodes)
        nvshmem_free_tensor_sync(self.send_reqs_recv_bufs)
        nvshmem_free_tensor_sync(self.send_buf)
        nvshmem_free_tensor_sync(self.output_buf)
        nvshmem_free_tensor_sync(self.signal_buf)
        nvshmem_free_tensor_sync(self.top_indices_buf)
        nvshmem_free_tensor_sync(self.local_splits_buf)
        nvshmem_free_tensor_sync(self.full_splits_buf)
        nvshmem_free_tensor_sync(self.splits_signal_buf)
        nvshmem_free_tensor_sync(self.intra_node_reduce_buf)

    def preprocess(self, input: torch.Tensor, exp_indices: torch.Tensor):
        token_node_idx = exp_indices // (self.experts_per_rank * self.local_world_size)

        # TODO(zhengxuegui.0): use triton kernel to gen send requests. It takes 150us to generate a request for each node(4096 tokens top 8).
        for traget_node_id in range(self.nnodes):
            if traget_node_id == self.node_id:
                continue
            start_indices, end_indices = get_dispatch_send_reqs_for_target_node(token_node_idx, traget_node_id,
                                                                                index_type=self.offset_dtype)
            self.send_reqs_for_nodes[traget_node_id, 0, :start_indices.shape[0]].copy_(start_indices)
            self.send_reqs_for_nodes[traget_node_id, 1, :end_indices.shape[0]].copy_(end_indices)

        _ = bincount(exp_indices.view(-1), length=self.num_tot_experts, output=self.local_splits_buf,
                     num_sm=self.num_sm)
        recv_buf_offset_per_expert, num_recv_tokens_per_rank, num_input_tokens_per_rank = get_ag_splits_and_recv_offset_for_dispatch(
            self.local_splits_buf, self.full_splits_buf, self.splits_signal_buf, self.topk, self.world_size,
            self.experts_per_rank, cpu_default_val=self.cpu_default_val, offset_dtype=self.offset_dtype,
            num_sm=self.num_sm)

        return recv_buf_offset_per_expert, num_recv_tokens_per_rank, num_input_tokens_per_rank

    def dispatch_postprocess(self):
        self.local_splits_buf.fill_(0)
        self.signal_buf.zero_()
        self.splits_signal_buf.zero_()
        self.counter.zero_()
        self.send_reqs_for_nodes.fill_(-1)

    def combine_postprocess(self):
        self.send_reqs_recv_bufs.fill_(0)

    def dispatch_token(self, recv_buf_offset_per_expert, num_input_tokens_per_rank):
        grid = lambda meta: (self.num_sm * 2, )
        assert self.top_indices_buf.dtype == self.send_reqs_for_nodes.dtype
        kernel_dispatch_token[grid](
            self.send_reqs_for_nodes,
            self.send_reqs_recv_bufs,
            self.signal_buf,
            self.counter,
            recv_buf_offset_per_expert,
            self.send_buf,
            self.output_buf,
            self.signal_buf,
            self.top_indices_buf,
            self.token_dst_scatter_idx,
            num_input_tokens_per_rank,
            self.max_tokens,
            self.topk,
            self.hidden,
            self.dtype.itemsize * self.hidden,
            self.experts_per_rank,
            self.local_world_size,
            num_warps=32,
        )

    def init_output_buffer(self, num_recv_tokens_per_rank):
        # `num_recv_tokens_per_rank` is in the pin memory.
        # To avoid stream synchronization by polling on the cpu to reduce the gpu bubble.
        assert num_recv_tokens_per_rank.is_cpu and num_recv_tokens_per_rank.is_pinned()
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
        cur_output_token_num = ctypes.c_int32.from_address(base_ptr + self.rank * elem_size).value
        return self.output_buf[:cur_output_token_num]

    def dispatch(self, input: torch.Tensor, exp_indices: torch.Tensor):
        current_stream = torch.cuda.current_stream()
        token_num, N = input.shape
        self.num_dispatch_token_cur_rank = token_num
        assert N == self.hidden
        self.send_buf[self.node_id, :token_num].copy_(input)
        self.top_indices_buf[self.node_id, :token_num].copy_(exp_indices)
        recv_buf_offset_per_expert, num_recv_tokens_per_rank, num_input_tokens_per_rank = self.preprocess(
            input, exp_indices)

        output_buf = self.init_output_buffer(num_recv_tokens_per_rank)
        self.dispatch_token(recv_buf_offset_per_expert, num_input_tokens_per_rank)
        self.num_input_tokens_per_rank = num_input_tokens_per_rank
        nvshmem_barrier_all_on_stream(current_stream)
        self.dispatch_postprocess()
        nvshmem_barrier_all_on_stream(current_stream)
        # This copy is redundant and is only kept for stress testing, we can remove it during integration.
        copy_out = torch.empty(output_buf.shape, dtype=output_buf.dtype, device=output_buf.device)
        copy_out.copy_(output_buf)
        return copy_out

    def combine_token_intra_node_and_send(self, input: torch.Tensor):
        grid = lambda meta: (self.num_sm, )
        BLOCK_SIZE = 1 << self.hidden.bit_length()
        counter_workspace = torch.zeros((self.nnodes, ), dtype=torch.int32, device=torch.cuda.current_device())
        kernel_combine_token[grid](
            counter_workspace,
            self.num_input_tokens_per_rank,
            self.send_reqs_recv_bufs,
            self.intra_node_reduce_buf,
            input,
            self.send_buf,
            self.top_indices_buf,
            self.token_dst_scatter_idx,
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

    def combine(self, input):
        current_stream = torch.cuda.current_stream()
        self.send_buf.fill_(0)
        nvshmem_barrier_all_on_stream(current_stream)
        self.output_buf[:input.shape[0]].copy_(input)
        reduce_buf = self.combine_token_intra_node_and_send(self.output_buf)
        nvshmem_barrier_all_on_stream(current_stream)
        reduce_inter_node = reduce_buf.reshape(self.nnodes, self.max_tokens, self.hidden).sum(dim=0)
        return reduce_inter_node[:self.num_dispatch_token_cur_rank]
