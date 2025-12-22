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
from dataclasses import dataclass

from triton_dist.kernels.nvidia.ep_a2a import (
    ep_combine_token_inplace,
    ep_dispatch_token_inplace,
    bincount,
    get_dispatch_send_reqs,
    get_ag_splits_and_recv_offset_for_dispatch,
)
from triton_dist.kernels.nvidia.ep_a2a_intra_node import (
    kernel_combine_token_intra_node,
    kernel_dispatch_token_intra_node,
    get_ag_splits_and_recv_offset_for_dispatch_intra_node,
    kernel_skipped_token_local_dispatch_intra_node,
    kernel_skipped_token_inplace_local_combine_intra_node,
)

from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE, nvshmem_barrier_all_on_stream, nvshmem_free_tensor_sync, nvshmem_create_tensor
from triton_dist.kernels.nvidia.common_ops import BarrierAllContext, barrier_all_on_stream


@dataclasses.dataclass
class EPAllToAllLayoutDesc:
    num_dispatch_token_cur_rank: int
    num_input_tokens_per_rank: torch.Tensor
    send_reqs_recv_tensor: Union[torch.Tensor, None]  # used for inter-node
    topk_indices_tensor: torch.Tensor
    token_dst_scatter_idx: torch.Tensor
    skipped_token_mapping_indices: Union[torch.Tensor, None]
    skipped_token_topk_mapping_indices: Union[torch.Tensor, None]


@dataclass
class EPConfig:
    max_tokens: int
    hidden: int
    topk: int
    num_experts: int

    rank: int
    world_size: int
    local_world_size: int

    token_dtype: torch.dtype
    weight_dtype: torch.dtype
    offset_dtype: torch.dtype

    @property
    def num_experts_per_rank(self):
        return self.num_experts // self.world_size

    @property
    def is_intra_node(self):
        return self.world_size == self.local_world_size


@dataclass
class DispatchCombineContext:
    ep_config: EPConfig

    grid_sync_buf: torch.Tensor  # (world_size, )

    send_reqs_for_nodes_rdma: torch.Tensor  # (nnodes, 2, max_tokens), init with -1
    send_reqs_recv_bufs_rdma: torch.Tensor  # (nnodes, 2, max_tokens), init with -1
    # as dispatch input buf, reuse as combine output buffer
    token_send_buf_rdma: torch.Tensor  # (nnodes, max_tokens, hidden)

    # as dispatch output buf, reuse as combine input buffer
    dispatch_output_buf: torch.Tensor  # (dispatch_recv_tokens, hidden)
    weight_recv_buf: torch.Tensor  # (dispatch_recv_tokens, topk)
    topk_indices_buf_rdma: torch.Tensor  # (nnodes, max_tokens, topk)
    weight_send_buf_rdma: torch.Tensor  # (nnodes, max_tokens, topk)
    signal_buf: torch.Tensor  # (world_size,), init with 0

    local_splits_buf: torch.Tensor  # (num_experts + 1), init with 0
    full_splits_buf: torch.Tensor  # (world_size, num_tot_experts + 1)

    num_input_tokens_per_rank_comm_buf: torch.Tensor  # (world_size,)
    splits_signal_buf: torch.Tensor  # (world_size,), init with 0
    expert_indices_signal_buf: torch.Tensor  # (world_size,), init with 0

    # for combine
    intra_node_reduce_buf: torch.Tensor  # (nnodes, max_tokens, hidden)

    # for intra node optimize
    intra_node_dispatch_skipped_token_mapping_indices: torch.Tensor  # (local_world_size * max_tokens * topk)
    intra_node_dispatch_skipped_token_topk_mapping_indices: torch.Tensor  # (local_world_size * max_tokens * topk, topk)

    @staticmethod
    def create(ep_config: EPConfig, capacity: int = 2) -> "DispatchCombineContext":
        nnodes = ep_config.world_size // ep_config.local_world_size
        num_tot_experts = ep_config.num_experts
        max_tokens = ep_config.max_tokens

        grid_sync_buf = torch.zeros([ep_config.world_size], dtype=torch.int32, device=torch.cuda.current_device())

        send_reqs_for_nodes_rdma = nvshmem_create_tensor([nnodes, 2, max_tokens], ep_config.offset_dtype)
        send_reqs_for_nodes_rdma.fill_(-1)
        send_reqs_recv_bufs_rdma = nvshmem_create_tensor([nnodes, 2, max_tokens], ep_config.offset_dtype)
        send_reqs_recv_bufs_rdma.fill_(-1)
        token_send_buf_rdma = nvshmem_create_tensor([nnodes, max_tokens, ep_config.hidden], ep_config.token_dtype)

        init_cap = int(max_tokens * ep_config.topk * capacity)
        dispatch_output_buf = nvshmem_create_tensor([init_cap, ep_config.hidden], ep_config.token_dtype)
        weight_recv_buf = nvshmem_create_tensor([
            init_cap,
        ], ep_config.weight_dtype)

        topk_indices_buf_rdma = nvshmem_create_tensor([nnodes, max_tokens, ep_config.topk], ep_config.offset_dtype)
        weight_send_buf_rdma = nvshmem_create_tensor([nnodes, max_tokens, ep_config.topk], ep_config.weight_dtype)
        signal_buf = nvshmem_create_tensor([ep_config.world_size], NVSHMEM_SIGNAL_DTYPE)
        signal_buf.zero_()

        local_splits_buf = nvshmem_create_tensor([ep_config.num_experts + 1], ep_config.offset_dtype)
        local_splits_buf.zero_()
        full_splits_buf = nvshmem_create_tensor([ep_config.world_size, num_tot_experts + 1], ep_config.offset_dtype)

        num_input_tokens_per_rank_comm_buf = nvshmem_create_tensor([ep_config.world_size], ep_config.offset_dtype)
        splits_signal_buf = nvshmem_create_tensor([ep_config.world_size], NVSHMEM_SIGNAL_DTYPE)
        splits_signal_buf.zero_()
        expert_indices_signal_buf = nvshmem_create_tensor([ep_config.world_size], NVSHMEM_SIGNAL_DTYPE)
        expert_indices_signal_buf.zero_()

        intra_node_reduce_buf = nvshmem_create_tensor([nnodes, max_tokens, ep_config.hidden], ep_config.token_dtype)

        # for dispatch
        intra_node_dispatch_skipped_token_mapping_indices = nvshmem_create_tensor(
            [ep_config.local_world_size * max_tokens * ep_config.topk], NVSHMEM_SIGNAL_DTYPE)
        intra_node_dispatch_skipped_token_mapping_indices.fill_(-1)
        intra_node_dispatch_skipped_token_topk_mapping_indices = nvshmem_create_tensor(
            [ep_config.local_world_size * max_tokens * ep_config.topk, ep_config.topk], NVSHMEM_SIGNAL_DTYPE)
        intra_node_dispatch_skipped_token_topk_mapping_indices.fill_(-1)

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        return DispatchCombineContext(
            ep_config=ep_config,
            grid_sync_buf=grid_sync_buf,
            send_reqs_for_nodes_rdma=send_reqs_for_nodes_rdma,
            send_reqs_recv_bufs_rdma=send_reqs_recv_bufs_rdma,
            token_send_buf_rdma=token_send_buf_rdma,
            dispatch_output_buf=dispatch_output_buf,
            weight_recv_buf=weight_recv_buf,
            topk_indices_buf_rdma=topk_indices_buf_rdma,
            weight_send_buf_rdma=weight_send_buf_rdma,
            signal_buf=signal_buf,
            local_splits_buf=local_splits_buf,
            full_splits_buf=full_splits_buf,
            num_input_tokens_per_rank_comm_buf=num_input_tokens_per_rank_comm_buf,
            splits_signal_buf=splits_signal_buf,
            expert_indices_signal_buf=expert_indices_signal_buf,
            intra_node_reduce_buf=intra_node_reduce_buf,
            intra_node_dispatch_skipped_token_mapping_indices=intra_node_dispatch_skipped_token_mapping_indices,
            intra_node_dispatch_skipped_token_topk_mapping_indices=
            intra_node_dispatch_skipped_token_topk_mapping_indices,
        )

    def finalize(self):
        nvshmem_free_tensor_sync(self.send_reqs_for_nodes_rdma)
        nvshmem_free_tensor_sync(self.send_reqs_recv_bufs_rdma)
        nvshmem_free_tensor_sync(self.token_send_buf_rdma)
        nvshmem_free_tensor_sync(self.dispatch_output_buf)
        nvshmem_free_tensor_sync(self.weight_recv_buf)
        nvshmem_free_tensor_sync(self.topk_indices_buf_rdma)
        nvshmem_free_tensor_sync(self.weight_send_buf_rdma)
        nvshmem_free_tensor_sync(self.signal_buf)
        nvshmem_free_tensor_sync(self.local_splits_buf)
        nvshmem_free_tensor_sync(self.full_splits_buf)
        nvshmem_free_tensor_sync(self.num_input_tokens_per_rank_comm_buf)
        nvshmem_free_tensor_sync(self.splits_signal_buf)
        nvshmem_free_tensor_sync(self.expert_indices_signal_buf)
        nvshmem_free_tensor_sync(self.intra_node_reduce_buf)
        nvshmem_free_tensor_sync(self.intra_node_dispatch_skipped_token_mapping_indices)
        nvshmem_free_tensor_sync(self.intra_node_dispatch_skipped_token_topk_mapping_indices)

    def reallocate_dispatch_output_buf(self, dispatch_recv_tokens: int):
        if dispatch_recv_tokens <= self.dispatch_output_buf.shape[0]:
            return self.dispatch_output_buf, self.weight_recv_buf

        nvshmem_free_tensor_sync(self.dispatch_output_buf)
        nvshmem_free_tensor_sync(self.weight_recv_buf)

        torch.distributed.barrier()
        self.dispatch_output_buf = nvshmem_create_tensor([dispatch_recv_tokens, self.ep_config.hidden],
                                                         self.ep_config.token_dtype)
        self.weight_recv_buf = nvshmem_create_tensor([dispatch_recv_tokens, self.ep_config.topk],
                                                     self.ep_config.weight_dtype)
        return self.dispatch_output_buf, self.weight_recv_buf


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
        enable_local_combine: bool = False,
        use_aot: bool = False,
    ):
        super().__init__()
        self.offset_dtype = torch.int32
        self.ep_group = ep_group
        self.num_sm = num_sm

        self.use_aot = use_aot

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
        self.is_intra_node = (self.world_size == self.local_world_size)
        self.is_intra_node = False

        self.enable_local_combine = enable_local_combine and self.is_intra_node
        self.Alignment = 1024
        self.cpu_default_val = -1
        self.ep_config = EPConfig(
            max_tokens=max_tokens,
            hidden=hidden,
            topk=topk,
            num_experts=num_tot_experts,
            rank=rank,
            world_size=world_size,
            local_world_size=local_world_size,
            token_dtype=dtype,
            weight_dtype=weight_dtype,
            offset_dtype=torch.int32,
        )
        self.a2a_ctx = DispatchCombineContext.create(self.ep_config)
        self.intra_node_barrier_ctx = BarrierAllContext(is_intra_node=True)

        nvshmem_barrier_all_on_stream()

    def finalize(self):
        self.a2a_ctx.finalize()
        self.intra_node_barrier_ctx.finalize()

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

    def ep_barrier_all(self, stream: torch.cuda.Stream, intra_node_only: bool = False):
        if intra_node_only or self.is_intra_node:
            barrier_all_on_stream(self.intra_node_barrier_ctx, stream)
        else:
            nvshmem_barrier_all_on_stream(stream)

    def preprocess(self, input: torch.Tensor, exp_indices: torch.Tensor, full_scatter_indices: Union[torch.Tensor,
                                                                                                     None] = None):
        num_dispatch_token_cur_rank = exp_indices.shape[0]
        send_reqs_for_nodes_rdma = self.a2a_ctx.send_reqs_for_nodes_rdma
        send_reqs_recv_bufs_rdma = self.a2a_ctx.send_reqs_recv_bufs_rdma
        topk_indices_buf_rdma = self.a2a_ctx.topk_indices_buf_rdma
        full_splits_buf = self.a2a_ctx.full_splits_buf
        splits_signal_buf = self.a2a_ctx.splits_signal_buf
        local_splits_buf = self.a2a_ctx.local_splits_buf
        expert_indices_signal_buf = self.a2a_ctx.expert_indices_signal_buf
        # inter-node
        if not self.is_intra_node or self.use_aot:
            get_dispatch_send_reqs(exp_indices, send_reqs_for_nodes_rdma, self.experts_per_rank, self.local_world_size,
                                   self.num_sm, use_aot=self.use_aot)

            # assume that the expert indices of the drop token is num_tot_experts,
            # it will be counted in the `local_splits_buf[num_tot_experts]`
            _ = bincount(exp_indices.view(-1), length=local_splits_buf.shape[0], output=local_splits_buf,
                         num_sm=self.num_sm, use_aot=self.use_aot)
            recv_buf_offset_per_expert, num_recv_tokens_per_rank, num_input_tokens_per_rank, token_dst_scatter_idx, send_reqs_recv_tensor, topk_indices_tensor = get_ag_splits_and_recv_offset_for_dispatch(
                send_reqs_for_nodes_rdma, send_reqs_recv_bufs_rdma, exp_indices, topk_indices_buf_rdma,
                expert_indices_signal_buf, local_splits_buf, full_splits_buf, splits_signal_buf, self.topk,
                self.local_world_size, self.world_size, self.max_tokens, self.experts_per_rank,
                full_scatter_indices=full_scatter_indices, cpu_default_val=self.cpu_default_val,
                offset_dtype=self.ep_config.offset_dtype, num_sm=self.num_sm, use_aot=self.use_aot)

            ep_a2a_layout_desc = EPAllToAllLayoutDesc(
                num_dispatch_token_cur_rank=num_dispatch_token_cur_rank,
                num_input_tokens_per_rank=num_input_tokens_per_rank, send_reqs_recv_tensor=send_reqs_recv_tensor,
                topk_indices_tensor=topk_indices_tensor, token_dst_scatter_idx=token_dst_scatter_idx,
                skipped_token_mapping_indices=None, skipped_token_topk_mapping_indices=None)
        else:
            # intra-node
            _ = bincount(exp_indices.view(-1), length=local_splits_buf.shape[0], output=local_splits_buf,
                         num_sm=self.num_sm)
            recv_buf_offset_per_expert, num_recv_tokens_per_rank, num_input_tokens_per_rank, token_dst_scatter_idx = get_ag_splits_and_recv_offset_for_dispatch_intra_node(
                exp_indices, local_splits_buf, full_splits_buf, splits_signal_buf, self.topk, self.local_world_size,
                self.world_size, self.max_tokens, self.experts_per_rank, full_scatter_indices=full_scatter_indices,
                cpu_default_val=self.cpu_default_val, offset_dtype=self.offset_dtype, num_sm=self.num_sm)

            ep_a2a_layout_desc = EPAllToAllLayoutDesc(num_dispatch_token_cur_rank=num_dispatch_token_cur_rank,
                                                      num_input_tokens_per_rank=num_input_tokens_per_rank,
                                                      send_reqs_recv_tensor=None, topk_indices_tensor=exp_indices,
                                                      token_dst_scatter_idx=token_dst_scatter_idx,
                                                      skipped_token_mapping_indices=None,
                                                      skipped_token_topk_mapping_indices=None)
        return recv_buf_offset_per_expert, num_recv_tokens_per_rank, ep_a2a_layout_desc

    def dispatch_postprocess(self):
        self.a2a_ctx.expert_indices_signal_buf.fill_(0)
        self.a2a_ctx.local_splits_buf.fill_(0)
        self.a2a_ctx.signal_buf.zero_()
        self.a2a_ctx.splits_signal_buf.zero_()
        self.a2a_ctx.grid_sync_buf.zero_()
        self.a2a_ctx.send_reqs_for_nodes_rdma.fill_(-1)
        self.a2a_ctx.full_splits_buf.fill_(0)
        self.a2a_ctx.topk_indices_buf_rdma.fill_(-1)
        if self.is_intra_node:
            self.a2a_ctx.intra_node_dispatch_skipped_token_mapping_indices.fill_(-1)
            self.a2a_ctx.intra_node_dispatch_skipped_token_topk_mapping_indices.fill_(-1)

    def combine_postprocess(self):
        self.ep_a2a_ctx.send_reqs_recv_bufs_rdma.fill_(0)

    def dispatch_token(self, recv_buf_offset_per_expert, output_buf: torch.Tensor,
                       ep_a2a_layout_desc: EPAllToAllLayoutDesc, has_weight=False):
        grid = lambda meta: (self.num_sm, )
        assert self.a2a_ctx.topk_indices_buf_rdma.dtype == self.a2a_ctx.send_reqs_for_nodes_rdma.dtype
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

        if not self.is_intra_node or self.use_aot:
            ep_dispatch_token_inplace(
                self.a2a_ctx.send_reqs_for_nodes_rdma, self.a2a_ctx.signal_buf, recv_buf_offset_per_expert,
                self.a2a_ctx.token_send_buf_rdma, output_buf,  # output
                self.a2a_ctx.weight_send_buf_rdma, self.a2a_ctx.weight_recv_buf, ep_a2a_layout_desc.topk_indices_tensor,
                token_dst_scatter_idx, ep_a2a_layout_desc.num_input_tokens_per_rank, self.max_tokens, self.topk,
                self.hidden, bytes_per_token=self.dtype.itemsize * self.hidden, experts_per_rank=self.experts_per_rank,
                local_world_size=self.local_world_size, has_weight=has_weight,
                with_scatter_indices=with_scatter_indices, num_sms=self.num_sm, use_aot=self.use_aot)
        else:
            dispatch_recv_token_num = output_buf.shape[0]
            kernel_dispatch_token_intra_node[grid](
                dispatch_recv_token_num,
                self.a2a_ctx.intra_node_dispatch_skipped_token_mapping_indices,
                self.a2a_ctx.intra_node_dispatch_skipped_token_topk_mapping_indices,
                recv_buf_offset_per_expert,
                self.a2a_ctx.token_send_buf_rdma,
                output_buf,
                self.a2a_ctx.weight_send_buf_rdma,
                self.a2a_ctx.weight_recv_buf,
                ep_a2a_layout_desc.topk_indices_tensor,
                token_dst_scatter_idx,
                ep_a2a_layout_desc.num_input_tokens_per_rank,
                self.topk,
                self.hidden,
                self.dtype.itemsize * self.hidden,
                self.experts_per_rank,
                self.local_world_size,
                HAS_WEIGHT=has_weight,
                WITH_SCATTER_INDICES=with_scatter_indices,
                num_warps=32,
            )
            current_stream = torch.cuda.current_stream()
            self.ep_barrier_all(current_stream)
            intra_node_dispatch_skipped_token_mapping_indices_copy = torch.empty(
                (dispatch_recv_token_num, ), dtype=self.a2a_ctx.intra_node_dispatch_skipped_token_mapping_indices.dtype,
                device=self.a2a_ctx.intra_node_dispatch_skipped_token_mapping_indices.device)
            intra_node_dispatch_skipped_token_topk_mapping_indices_copy = torch.empty(
                (dispatch_recv_token_num, self.topk),
                dtype=self.a2a_ctx.intra_node_dispatch_skipped_token_topk_mapping_indices.dtype,
                device=self.a2a_ctx.intra_node_dispatch_skipped_token_topk_mapping_indices.device)

            kernel_skipped_token_local_dispatch_intra_node[grid](
                dispatch_recv_token_num,
                self.a2a_ctx.intra_node_dispatch_skipped_token_mapping_indices,
                self.a2a_ctx.intra_node_dispatch_skipped_token_topk_mapping_indices,
                intra_node_dispatch_skipped_token_mapping_indices_copy,
                intra_node_dispatch_skipped_token_topk_mapping_indices_copy,
                output_buf,
                self.hidden,
                self.dtype.itemsize * self.hidden,
                self.topk,
                ENABLE_LOCAL_COMBINE=self.enable_local_combine,
                num_warps=32,
            )

            ep_a2a_layout_desc.skipped_token_mapping_indices = intra_node_dispatch_skipped_token_mapping_indices_copy
            ep_a2a_layout_desc.skipped_token_topk_mapping_indices = intra_node_dispatch_skipped_token_topk_mapping_indices_copy

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
        if max_output_token_num > self.a2a_ctx.dispatch_output_buf.shape[0]:
            torch.distributed.barrier()
            alloc_token = (max_output_token_num + self.Alignment - 1) // self.Alignment * self.Alignment * 2
            self.a2a_ctx.reallocate_dispatch_output_buf(alloc_token)

        cur_output_token_num = ctypes.c_int32.from_address(base_ptr + self.rank * elem_size).value
        return self.a2a_ctx.dispatch_output_buf[:
                                                cur_output_token_num], self.a2a_ctx.weight_recv_buf[:
                                                                                                    cur_output_token_num]

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
        self.a2a_ctx.token_send_buf_rdma[self.node_id, :token_num].copy_(input)
        self.a2a_ctx.topk_indices_buf_rdma[self.node_id, :token_num].copy_(exp_indices)
        has_weight = (weight is not None)
        if has_weight:
            assert weight.shape[0] == token_num
            assert weight.shape[1] == self.topk
            assert weight.is_contiguous()
            assert weight.dtype == self.weight_dtype
            self.a2a_ctx.weight_send_buf_rdma[self.node_id, :token_num].copy_(weight)

        recv_buf_offset_per_expert, num_recv_tokens_per_rank, ep_a2a_layout_desc = self.preprocess(
            input, exp_indices, full_scatter_indices)

        output_buf, weight_recv_buf = self.init_output_buffer(num_recv_tokens_per_rank)
        # if full_scatter_indices is None, token_dst_scatter_idx is calc in dispatch
        ep_a2a_layout_desc = self.dispatch_token(recv_buf_offset_per_expert, output_buf, ep_a2a_layout_desc,
                                                 has_weight=has_weight)
        self.ep_barrier_all(current_stream)
        self.dispatch_postprocess()
        self.ep_barrier_all(current_stream)
        # This copy is redundant and is only kept for stress testing, we can remove it during integration.
        copy_out = torch.empty(output_buf.shape, dtype=output_buf.dtype, device=output_buf.device)
        copy_weight = None
        if has_weight:
            copy_weight = torch.empty(weight_recv_buf.shape, dtype=weight_recv_buf.dtype, device=weight_recv_buf.device)
            copy_weight.copy_(weight_recv_buf)
        copy_out.copy_(output_buf)
        return copy_out, copy_weight, ep_a2a_layout_desc

    def combine_token_intra_node_and_send(self, input: torch.Tensor, ep_a2a_layout_desc: EPAllToAllLayoutDesc):
        counter_workspace = torch.zeros((self.nnodes, ), dtype=torch.int32, device=torch.cuda.current_device())
        grid = lambda meta: (self.num_sm, )
        current_stream = torch.cuda.current_stream()
        # inter-node
        if not self.is_intra_node or self.use_aot:
            counter_workspace = torch.zeros((self.nnodes, ), dtype=torch.int32, device=torch.cuda.current_device())
            ep_combine_token_inplace(
                counter_workspace,
                ep_a2a_layout_desc.num_input_tokens_per_rank,
                ep_a2a_layout_desc.send_reqs_recv_tensor,
                self.a2a_ctx.intra_node_reduce_buf,
                input,
                self.a2a_ctx.token_send_buf_rdma,
                ep_a2a_layout_desc.topk_indices_tensor,
                ep_a2a_layout_desc.token_dst_scatter_idx,
                self.max_tokens,
                self.topk,
                self.hidden,
                bytes_per_token=input.dtype.itemsize * self.hidden,
                experts_per_rank=self.experts_per_rank,
                local_world_size=self.local_world_size,
                num_sms=self.num_sm,
                use_aot=self.use_aot,
            )
            return self.a2a_ctx.token_send_buf_rdma
        else:
            # intra-node
            if self.enable_local_combine:
                assert ep_a2a_layout_desc.skipped_token_mapping_indices.shape[0] == input.shape[0]
                assert ep_a2a_layout_desc.skipped_token_topk_mapping_indices.shape[0] == input.shape[0]
                kernel_skipped_token_inplace_local_combine_intra_node[grid](
                    input.shape[0],
                    ep_a2a_layout_desc.skipped_token_mapping_indices,
                    ep_a2a_layout_desc.skipped_token_topk_mapping_indices,
                    input,
                    self.hidden,
                    self.topk,
                    num_warps=32,
                )
                self.ep_barrier_all(current_stream)
            combine_intra_node_out_buf = torch.empty((ep_a2a_layout_desc.num_dispatch_token_cur_rank, self.hidden),
                                                     dtype=input.dtype, device=input.device)
            kernel_combine_token_intra_node[grid](
                ep_a2a_layout_desc.num_input_tokens_per_rank,
                input,
                combine_intra_node_out_buf,
                ep_a2a_layout_desc.topk_indices_tensor,
                ep_a2a_layout_desc.token_dst_scatter_idx,
                self.max_tokens,
                self.topk,
                self.hidden,
                input.dtype.itemsize * self.hidden,
                self.experts_per_rank,
                self.local_world_size,
                ENABLE_LOCAL_COMBINE=self.enable_local_combine,
                num_warps=32,
            )
            return combine_intra_node_out_buf

    def combine(self, input, ep_a2a_layout_desc: EPAllToAllLayoutDesc):
        assert input.is_contiguous()
        assert input.dtype == self.dtype
        current_stream = torch.cuda.current_stream()
        # self.send_buf.fill_(0)
        self.a2a_ctx.token_send_buf_rdma.fill_(0)
        # reuse dispatch_output_buf as combine input
        self.a2a_ctx.dispatch_output_buf[:input.shape[0]].copy_(input)
        combine_input = self.a2a_ctx.dispatch_output_buf[:input.shape[0]]
        self.ep_barrier_all(current_stream)
        reduce_buf = self.combine_token_intra_node_and_send(combine_input, ep_a2a_layout_desc)
        self.ep_barrier_all(current_stream)
        if not self.is_intra_node:
            reduce_inter_node = reduce_buf.reshape(self.nnodes, self.max_tokens, self.hidden).sum(dim=0)
            return reduce_inter_node[:ep_a2a_layout_desc.num_dispatch_token_cur_rank]
        else:
            return reduce_buf
