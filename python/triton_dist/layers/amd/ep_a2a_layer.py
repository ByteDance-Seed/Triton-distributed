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
AMD (HIP/ROCm) Expert Parallel All-to-All layer – intra-node only.

Framework skeleton mirroring the NVIDIA ep_a2a_layer. Uses mori_shmem for
barrier and symmetric tensor create/free (no rocshmem). Only intra-node path.
"""

import torch
import dataclasses
import ctypes
from typing import Union
from dataclasses import dataclass

from triton_dist.kernels.amd.ep_a2a import (
    bincount,
    kernel_combine_token_intra_node,
    kernel_dispatch_token_intra_node,
    get_ag_splits_and_recv_offset_for_dispatch_intra_node,
    kernel_skipped_token_local_dispatch_intra_node,
    kernel_skipped_token_inplace_local_combine_intra_node,
)

from triton_dist.utils import mori_shmem_barrier_all_on_stream

from triton_dist.utils import MORI_SHMEM_SIGNAL_DTYPE

# Symmetric tensor: use mori_shmem directly (same as test_mori_shmem_api.py).
import mori.shmem as mori_shmem
from mori.shmem import mori_shmem_create_tensor


def _mori_shmem_free_sync(tensor):
    """Sync, free symmetric memory, sync. Use _mori_ptr when present (mori tensor), else data_ptr()."""
    torch.cuda.synchronize()
    ptr = getattr(tensor, "_mori_ptr", tensor.data_ptr())
    mori_shmem.shmem_free(ptr)
    torch.cuda.synchronize()


@dataclasses.dataclass
class EPAllToAllLayoutDesc:
    num_dispatch_token_cur_rank: int
    num_input_tokens_per_rank: torch.Tensor
    send_reqs_recv_tensor: Union[torch.Tensor, None]
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
    """Intra-node only: no send_reqs_for_nodes_rdma / send_reqs_recv_bufs_rdma."""

    ep_config: EPConfig
    grid_sync_buf: torch.Tensor
    token_send_buf_rdma: torch.Tensor
    dispatch_output_buf: torch.Tensor
    weight_recv_buf: torch.Tensor
    topk_indices_buf_rdma: torch.Tensor
    weight_send_buf_rdma: torch.Tensor
    signal_buf: torch.Tensor
    local_splits_buf: torch.Tensor
    full_splits_buf: torch.Tensor
    num_input_tokens_per_rank_comm_buf: torch.Tensor
    splits_signal_buf: torch.Tensor
    expert_indices_signal_buf: torch.Tensor
    intra_node_reduce_buf: torch.Tensor
    intra_node_dispatch_skipped_token_mapping_indices: torch.Tensor
    intra_node_dispatch_skipped_token_topk_mapping_indices: torch.Tensor

    @staticmethod
    def create(ep_config: EPConfig, capacity: int = 2) -> "DispatchCombineContext":
        nnodes = ep_config.world_size // ep_config.local_world_size
        num_tot_experts = ep_config.num_experts
        max_tokens = ep_config.max_tokens
        device = torch.cuda.current_device()

        grid_sync_buf = torch.zeros([ep_config.world_size], dtype=torch.int32, device=device)
        token_send_buf_rdma = mori_shmem_create_tensor([nnodes, max_tokens, ep_config.hidden], ep_config.token_dtype)
        init_cap = int(max_tokens * ep_config.topk * capacity)
        dispatch_output_buf = mori_shmem_create_tensor([init_cap, ep_config.hidden], ep_config.token_dtype)
        weight_recv_buf = mori_shmem_create_tensor([init_cap], ep_config.weight_dtype)
        topk_indices_buf_rdma = mori_shmem_create_tensor([nnodes, max_tokens, ep_config.topk], ep_config.offset_dtype)
        weight_send_buf_rdma = mori_shmem_create_tensor([nnodes, max_tokens, ep_config.topk], ep_config.weight_dtype)
        signal_buf = mori_shmem_create_tensor([ep_config.world_size], MORI_SHMEM_SIGNAL_DTYPE)
        signal_buf.zero_()
        local_splits_buf = mori_shmem_create_tensor([ep_config.num_experts + 1], ep_config.offset_dtype)
        local_splits_buf.zero_()
        full_splits_buf = mori_shmem_create_tensor([ep_config.world_size, num_tot_experts + 1], ep_config.offset_dtype)
        num_input_tokens_per_rank_comm_buf = mori_shmem_create_tensor([ep_config.world_size], ep_config.offset_dtype)
        splits_signal_buf = mori_shmem_create_tensor([ep_config.world_size], MORI_SHMEM_SIGNAL_DTYPE)
        splits_signal_buf.zero_()
        expert_indices_signal_buf = mori_shmem_create_tensor([ep_config.world_size], MORI_SHMEM_SIGNAL_DTYPE)
        expert_indices_signal_buf.zero_()
        intra_node_reduce_buf = mori_shmem_create_tensor([nnodes, max_tokens, ep_config.hidden], ep_config.token_dtype)
        intra_node_dispatch_skipped_token_mapping_indices = mori_shmem_create_tensor(
            [ep_config.local_world_size * max_tokens * ep_config.topk],
            MORI_SHMEM_SIGNAL_DTYPE,
        )
        intra_node_dispatch_skipped_token_mapping_indices.fill_(-1)
        intra_node_dispatch_skipped_token_topk_mapping_indices = mori_shmem_create_tensor(
            [
                ep_config.local_world_size * max_tokens * ep_config.topk,
                ep_config.topk,
            ],
            MORI_SHMEM_SIGNAL_DTYPE,
        )
        intra_node_dispatch_skipped_token_topk_mapping_indices.fill_(-1)

        mori_shmem_barrier_all_on_stream(torch.cuda.current_stream())

        return DispatchCombineContext(
            ep_config=ep_config,
            grid_sync_buf=grid_sync_buf,
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
        _mori_shmem_free_sync(self.token_send_buf_rdma)
        _mori_shmem_free_sync(self.dispatch_output_buf)
        _mori_shmem_free_sync(self.weight_recv_buf)
        _mori_shmem_free_sync(self.topk_indices_buf_rdma)
        _mori_shmem_free_sync(self.weight_send_buf_rdma)
        _mori_shmem_free_sync(self.signal_buf)
        _mori_shmem_free_sync(self.local_splits_buf)
        _mori_shmem_free_sync(self.full_splits_buf)
        _mori_shmem_free_sync(self.num_input_tokens_per_rank_comm_buf)
        _mori_shmem_free_sync(self.splits_signal_buf)
        _mori_shmem_free_sync(self.expert_indices_signal_buf)
        _mori_shmem_free_sync(self.intra_node_reduce_buf)
        _mori_shmem_free_sync(self.intra_node_dispatch_skipped_token_mapping_indices)
        _mori_shmem_free_sync(self.intra_node_dispatch_skipped_token_topk_mapping_indices)

    def reallocate_dispatch_output_buf(self, dispatch_recv_tokens: int):
        if dispatch_recv_tokens <= self.dispatch_output_buf.shape[0]:
            return self.dispatch_output_buf, self.weight_recv_buf
        _mori_shmem_free_sync(self.dispatch_output_buf)
        _mori_shmem_free_sync(self.weight_recv_buf)
        torch.distributed.barrier()
        self.dispatch_output_buf = mori_shmem_create_tensor([dispatch_recv_tokens, self.ep_config.hidden],
                                                            self.ep_config.token_dtype)
        self.weight_recv_buf = mori_shmem_create_tensor([dispatch_recv_tokens, self.ep_config.topk],
                                                        self.ep_config.weight_dtype)
        return self.dispatch_output_buf, self.weight_recv_buf


class EPAll2AllLayer(torch.nn.Module):
    """Intra-node only. No inter-node / legacy kernel path."""

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
        dispatch_grid_size: int = 512,
        combine_grid_size: int = 304,
    ):
        super().__init__()
        self.offset_dtype = torch.int32
        self.ep_group = ep_group
        self.num_sm = num_sm
        self.dispatch_grid_size = dispatch_grid_size
        self.combine_grid_size = combine_grid_size
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
        self.is_intra_node = self.world_size == self.local_world_size
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
        self._needs_postprocess = False
        self.intra_node_barrier_ctx = None

        mori_shmem_barrier_all_on_stream()

    def finalize(self):
        self.a2a_ctx.finalize()
        if self.intra_node_barrier_ctx is not None:
            self.intra_node_barrier_ctx.finalize()

    def ep_barrier_all(self, stream: torch.cuda.Stream, intra_node_only: bool = False):
        mori_shmem_barrier_all_on_stream(stream)

    def preprocess(
        self,
        input: torch.Tensor,
        exp_indices: torch.Tensor,
        full_scatter_indices: Union[torch.Tensor, None] = None,
    ):
        num_dispatch_token_cur_rank = exp_indices.shape[0]
        local_splits_buf = self.a2a_ctx.local_splits_buf
        full_splits_buf = self.a2a_ctx.full_splits_buf
        splits_signal_buf = self.a2a_ctx.splits_signal_buf

        # GPU bincount — writes directly to SHMEM buffer via atomics
        exp_flat = exp_indices.contiguous().view(-1).to(torch.int32)
        bincount(
            exp_flat,
            local_splits_buf.shape[0],
            output=local_splits_buf,
            output_dtype=local_splits_buf.dtype,
            num_sm=self.num_sm,
        )
        # No synchronize needed — bincount and get_ag_splits run on same stream
        (
            recv_buf_offset_per_expert,
            num_recv_tokens_per_rank,
            num_input_tokens_per_rank,
            token_dst_scatter_idx,
        ) = get_ag_splits_and_recv_offset_for_dispatch_intra_node(
            exp_indices,
            local_splits_buf,
            full_splits_buf,
            splits_signal_buf,
            self.topk,
            self.local_world_size,
            self.world_size,
            self.max_tokens,
            self.experts_per_rank,
            full_scatter_indices=full_scatter_indices,
            cpu_default_val=self.cpu_default_val,
            offset_dtype=self.offset_dtype,
            num_sm=self.num_sm,
        )
        ep_a2a_layout_desc = EPAllToAllLayoutDesc(
            num_dispatch_token_cur_rank=num_dispatch_token_cur_rank,
            num_input_tokens_per_rank=num_input_tokens_per_rank,
            send_reqs_recv_tensor=None,
            topk_indices_tensor=exp_indices,
            token_dst_scatter_idx=token_dst_scatter_idx,
            skipped_token_mapping_indices=None,
            skipped_token_topk_mapping_indices=None,
        )
        return recv_buf_offset_per_expert, num_recv_tokens_per_rank, ep_a2a_layout_desc

    def dispatch_postprocess(self):
        self.a2a_ctx.expert_indices_signal_buf.fill_(0)
        self.a2a_ctx.local_splits_buf.fill_(0)
        self.a2a_ctx.signal_buf.zero_()
        self.a2a_ctx.splits_signal_buf.zero_()
        self.a2a_ctx.grid_sync_buf.zero_()
        self.a2a_ctx.full_splits_buf.fill_(0)
        self.a2a_ctx.topk_indices_buf_rdma.fill_(-1)
        self.a2a_ctx.intra_node_dispatch_skipped_token_mapping_indices.fill_(-1)
        self.a2a_ctx.intra_node_dispatch_skipped_token_topk_mapping_indices.fill_(-1)

    def dispatch_token(
        self,
        recv_buf_offset_per_expert,
        output_buf: torch.Tensor,
        ep_a2a_layout_desc: EPAllToAllLayoutDesc,
        has_weight=False,
    ):
        grid = lambda meta: (self.num_sm, )
        token_dst_scatter_idx = ep_a2a_layout_desc.token_dst_scatter_idx
        if token_dst_scatter_idx is None:
            with_scatter_indices = False
            token_dst_scatter_idx = torch.empty(
                (self.nnodes, self.max_tokens, self.topk),
                dtype=torch.uint64,
                device=recv_buf_offset_per_expert.device,
            )
        else:
            assert len(token_dst_scatter_idx.shape) == 3
            assert token_dst_scatter_idx.shape[0] == self.nnodes
            assert token_dst_scatter_idx.shape[1] == self.max_tokens
            assert token_dst_scatter_idx.shape[2] == self.topk
            assert token_dst_scatter_idx.dtype == self.offset_dtype
            assert token_dst_scatter_idx.is_contiguous()
            with_scatter_indices = True

        dispatch_recv_token_num = output_buf.shape[0]
        dispatch_grid = (min(dispatch_recv_token_num, self.dispatch_grid_size), )
        kernel_dispatch_token_intra_node[dispatch_grid](
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
            num_warps=16,
        )
        current_stream = torch.cuda.current_stream()
        self.ep_barrier_all(current_stream)

        intra_node_dispatch_skipped_token_mapping_indices_copy = torch.empty(
            (dispatch_recv_token_num, ),
            dtype=self.a2a_ctx.intra_node_dispatch_skipped_token_mapping_indices.dtype,
            device=self.a2a_ctx.intra_node_dispatch_skipped_token_mapping_indices.device,
        )
        intra_node_dispatch_skipped_token_topk_mapping_indices_copy = torch.empty(
            (dispatch_recv_token_num, self.topk),
            dtype=self.a2a_ctx.intra_node_dispatch_skipped_token_topk_mapping_indices.dtype,
            device=self.a2a_ctx.intra_node_dispatch_skipped_token_topk_mapping_indices.device,
        )
        copy_grid = (min(dispatch_recv_token_num, self.dispatch_grid_size), )
        kernel_skipped_token_local_dispatch_intra_node[copy_grid](
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
            num_warps=16,
        )
        ep_a2a_layout_desc.skipped_token_mapping_indices = (intra_node_dispatch_skipped_token_mapping_indices_copy)
        ep_a2a_layout_desc.skipped_token_topk_mapping_indices = (
            intra_node_dispatch_skipped_token_topk_mapping_indices_copy)

        if not with_scatter_indices:
            ep_a2a_layout_desc.token_dst_scatter_idx = token_dst_scatter_idx
        return ep_a2a_layout_desc

    def init_output_buffer(self, num_recv_tokens_per_rank):
        assert num_recv_tokens_per_rank.is_cpu
        assert num_recv_tokens_per_rank.dtype == torch.int32
        max_output_token_num = 0
        base_ptr = num_recv_tokens_per_rank.data_ptr()
        elem_size = num_recv_tokens_per_rank.element_size()
        for target_rank in range(self.world_size):
            while (ctypes.c_int32.from_address(base_ptr + target_rank * elem_size).value == self.cpu_default_val):
                pass
            cur_output_token_num = ctypes.c_int32.from_address(base_ptr + target_rank * elem_size).value
            max_output_token_num = max(max_output_token_num, cur_output_token_num)
        if max_output_token_num > self.a2a_ctx.dispatch_output_buf.shape[0]:
            torch.distributed.barrier()
            alloc_token = ((max_output_token_num + self.Alignment - 1) // self.Alignment * self.Alignment * 2)
            self.a2a_ctx.reallocate_dispatch_output_buf(alloc_token)
        cur_output_token_num = ctypes.c_int32.from_address(base_ptr + self.rank * elem_size).value
        return (
            self.a2a_ctx.dispatch_output_buf[:cur_output_token_num],
            self.a2a_ctx.weight_recv_buf[:cur_output_token_num],
        )

    def dispatch(
        self,
        input: torch.Tensor,
        exp_indices: torch.Tensor,
        weight=None,
        full_scatter_indices=None,
    ):
        assert input.is_contiguous() and exp_indices.is_contiguous()
        assert input.dtype == self.dtype and exp_indices.dtype == self.offset_dtype
        assert (len(exp_indices.shape) == 2 and exp_indices.shape[0] == input.shape[0]
                and exp_indices.shape[1] == self.topk)
        current_stream = torch.cuda.current_stream()
        if self._needs_postprocess:
            self.dispatch_postprocess()
            self.ep_barrier_all(current_stream)
        self._needs_postprocess = True
        token_num, N = input.shape
        assert N == self.hidden
        self.a2a_ctx.token_send_buf_rdma[self.node_id, :token_num].copy_(input)
        self.a2a_ctx.topk_indices_buf_rdma[self.node_id, :token_num].copy_(exp_indices)
        has_weight = weight is not None
        if has_weight:
            assert (weight.shape[0] == token_num and weight.shape[1] == self.topk and weight.is_contiguous()
                    and weight.dtype == self.weight_dtype)
            self.a2a_ctx.weight_send_buf_rdma[self.node_id, :token_num].copy_(weight)
        recv_buf_offset_per_expert, num_recv_tokens_per_rank, ep_a2a_layout_desc = (self.preprocess(
            input, exp_indices, full_scatter_indices))
        output_buf, weight_recv_buf = self.init_output_buffer(num_recv_tokens_per_rank)
        ep_a2a_layout_desc = self.dispatch_token(
            recv_buf_offset_per_expert,
            output_buf,
            ep_a2a_layout_desc,
            has_weight=has_weight,
        )
        copy_out = torch.empty(output_buf.shape, dtype=output_buf.dtype, device=output_buf.device)
        copy_out.copy_(output_buf)
        copy_weight = None
        if has_weight:
            copy_weight = torch.empty(
                weight_recv_buf.shape,
                dtype=weight_recv_buf.dtype,
                device=weight_recv_buf.device,
            )
            copy_weight.copy_(weight_recv_buf)
        return copy_out, copy_weight, ep_a2a_layout_desc

    def combine_token_intra_node_and_send(
        self,
        input: torch.Tensor,
        ep_a2a_layout_desc: EPAllToAllLayoutDesc,
    ):
        grid = lambda meta: (self.num_sm, )
        current_stream = torch.cuda.current_stream()
        if self.enable_local_combine:
            assert ep_a2a_layout_desc.skipped_token_mapping_indices is not None
            assert ep_a2a_layout_desc.skipped_token_mapping_indices.shape[0] == input.shape[0]
            # Warp-level pre-combine using simt_exec_region + ld()/st().
            precombine_grid = (min(input.shape[0], self.combine_grid_size), )
            kernel_skipped_token_inplace_local_combine_intra_node[precombine_grid](
                input.shape[0],
                ep_a2a_layout_desc.skipped_token_mapping_indices,
                ep_a2a_layout_desc.skipped_token_topk_mapping_indices,
                input,
                self.hidden,
                self.topk,
                BLOCK_H=1024,
            )
            self.ep_barrier_all(current_stream)
        combine_intra_node_out_buf = torch.empty(
            (ep_a2a_layout_desc.num_dispatch_token_cur_rank, self.hidden),
            dtype=input.dtype,
            device=input.device,
        )
        # Cast scatter indices from uint64 to int32
        scatter_idx_i32 = ep_a2a_layout_desc.token_dst_scatter_idx.to(torch.int32)
        combine_grid = (min(ep_a2a_layout_desc.num_dispatch_token_cur_rank, self.combine_grid_size), )
        kernel_combine_token_intra_node[combine_grid](
            ep_a2a_layout_desc.num_dispatch_token_cur_rank,
            input,
            combine_intra_node_out_buf,
            ep_a2a_layout_desc.topk_indices_tensor,
            scatter_idx_i32,
            self.max_tokens,
            self.topk,
            self.hidden,
            input.dtype.itemsize * self.hidden,
            self.experts_per_rank,
            self.local_world_size,
            ENABLE_LOCAL_COMBINE=self.enable_local_combine,
            num_warps=16,
        )
        return combine_intra_node_out_buf

    def combine(self, input, ep_a2a_layout_desc: EPAllToAllLayoutDesc):
        assert input.is_contiguous() and input.dtype == self.dtype
        current_stream = torch.cuda.current_stream()
        self.a2a_ctx.dispatch_output_buf[:input.shape[0]].copy_(input)
        combine_input = self.a2a_ctx.dispatch_output_buf[:input.shape[0]]
        self.ep_barrier_all(current_stream)

        reduce_buf = self.combine_token_intra_node_and_send(
            combine_input,
            ep_a2a_layout_desc,
        )
        self.ep_barrier_all(current_stream)
        return reduce_buf.reshape(-1, self.hidden)[:ep_a2a_layout_desc.num_dispatch_token_cur_rank]
