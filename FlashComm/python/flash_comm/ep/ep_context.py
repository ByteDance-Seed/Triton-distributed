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
import torch.distributed as dist
import dataclasses
from typing import Union
from flash_comm.buffer import SymmetricTensor


@dataclasses.dataclass
class EPConfig:
    max_m: int
    hidden: int
    topk: int
    num_experts: int

    rank: int
    world_size: int  # ep group size
    local_world_size: int  # shared domain size
    local_rank: int = None
    node_id: int = None
    nnodes: int = None

    token_dtype: torch.dtype = torch.bfloat16  # bfloat16
    weight_dtype: torch.dtype = torch.float32  # float32
    offset_dtype: torch.dtype = torch.int32  # int32

    def __post_init__(self):
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        self.nnodes = self.world_size // self.local_world_size


@dataclasses.dataclass
class EPContext:
    config: EPConfig
    group: dist.ProcessGroup  # ep group
    # barrier in nvlink domain
    nvl_barrier_buf: torch.Tensor  # init with 0
    nvl_barrier_buf_ptrs: torch.Tensor

    # *_ptrs only contains pointers within same shared memory domain
    full_splits_buf: torch.Tensor  # (world_size, num_tot_experts + 1)
    full_splits_buf_ptrs: torch.Tensor  # (local_world_size)

    dispatch_output_buf: torch.Tensor  # (dispatch_recv_tokens, hidden)
    dispatch_output_buf_ptrs: torch.Tensor  # (local_world_size)
    dispatch_topk_weights_buf: torch.Tensor  # (dispatch_recv_tokens, topk), original topk weights for each token
    dispatch_topk_weights_buf_ptrs: torch.Tensor  # (local_world_size)
    dispatch_topk_scatter_indices_buf: torch.Tensor  # (dispatch_recv_tokens, topk), original topk indices for each token, init with -1
    dispatch_topk_scatter_indices_buf_ptrs: torch.Tensor  # (local_world_size)

    # dispatch output buf can be reused as combine input buf
    combine_input_buf: torch.Tensor  # (dispatch_recv_tokens, hidden)
    combine_input_buf_ptrs: torch.Tensor  # (local_world_size)
    combine_topk_weights_buf: torch.Tensor  # (dispatch_recv_tokens, topk), original topk weights for each token
    combine_topk_weights_buf_ptrs: torch.Tensor  # (local_world_size)

    # recv token count
    recv_token_count_cpu: torch.Tensor  # (world_size), pinned CPU memory
    recv_token_count: torch.Tensor  # (world_size), device memory

    # for rdma p2p (optional for single node)
    rdma_token_send_buf: Union[torch.Tensor, None] = None  # (nnodes, max_tokens, hidden)
    rdma_token_dst_scatter_indices_buf: Union[torch.Tensor, None] = None  # (nnodes, max_tokens, topk)
    rdma_token_topk_send_mask_buf: Union[torch.Tensor, None] = None  # (nnodes, max_tokens, topk)
    rdma_topk_weights_buf: Union[torch.Tensor, None] = None  # (nnodes, max_tokens, topk)
    rdma_topk_indices_buf: Union[torch.Tensor, None] = None  # (nnodes, max_tokens, topk)

    def reallocate_buffers(self, num_alloc_tokens: int, combine_input_reuse: bool = True):
        if num_alloc_tokens <= self.dispatch_output_buf.shape[0]:
            return

        # delete old symmetric tensors
        del self._symm_tensors['dispatch_output']
        del self._symm_tensors['dispatch_topk_weights']
        del self._symm_tensors['dispatch_topk_scatter_indices']
        del self._symm_tensors['combine_input']
        del self._symm_tensors['combine_weight']

        dispatch_output_symm_tensor = SymmetricTensor(shape=(num_alloc_tokens, self.config.hidden),
                                                      dtype=self.config.token_dtype, group=self.group)
        dispatch_topk_weights_symm_tensor = SymmetricTensor(shape=(num_alloc_tokens, self.config.topk),
                                                            dtype=self.config.weight_dtype, group=self.group)
        dispatch_topk_scatter_indices_symm_tensor = SymmetricTensor(shape=(num_alloc_tokens, self.config.topk),
                                                                    dtype=self.config.offset_dtype, group=self.group)

        self.dispatch_output_buf = dispatch_output_symm_tensor.get_local_tensor()
        self.dispatch_output_buf_ptrs = dispatch_output_symm_tensor.ptrs
        self.dispatch_topk_weights_buf = dispatch_topk_weights_symm_tensor.get_local_tensor()
        self.dispatch_topk_weights_buf_ptrs = dispatch_topk_weights_symm_tensor.ptrs
        self.dispatch_topk_scatter_indices_buf = dispatch_topk_scatter_indices_symm_tensor.get_local_tensor()
        self.dispatch_topk_scatter_indices_buf_ptrs = dispatch_topk_scatter_indices_symm_tensor.ptrs
        self.dispatch_topk_scatter_indices_buf.fill_(-1)  # init with -1, indicating the invalid index

        if combine_input_reuse:
            combine_input_symm_tensor = dispatch_output_symm_tensor
            combine_weight_symm_tensor = dispatch_topk_weights_symm_tensor
        else:
            combine_input_symm_tensor = SymmetricTensor(shape=(num_alloc_tokens, self.config.hidden),
                                                        dtype=self.config.token_dtype, group=self.group)
            combine_weight_symm_tensor = SymmetricTensor(shape=(num_alloc_tokens, self.config.topk),
                                                         dtype=self.config.weight_dtype, group=self.group)

        self.combine_input_buf = combine_input_symm_tensor.get_local_tensor()
        self.combine_input_buf_ptrs = combine_input_symm_tensor.ptrs
        self.combine_topk_weights_buf = combine_weight_symm_tensor.get_local_tensor()
        self.combine_topk_weights_buf_ptrs = combine_weight_symm_tensor.ptrs

        # update references to SymmetricTensor objects
        self._symm_tensors.update({
            'dispatch_output': dispatch_output_symm_tensor,
            'dispatch_topk_weights': dispatch_topk_weights_symm_tensor,
            'dispatch_topk_scatter_indices': dispatch_topk_scatter_indices_symm_tensor,
            'combine_input': combine_input_symm_tensor,
            'combine_weight': combine_weight_symm_tensor,
        })

    # need to perform ep group barrier after initialization
    @staticmethod
    def create(max_m: int, hidden: int, topk: int, num_experts: int, group: dist.ProcessGroup, local_world_size: int,
               capacity_coeff: float = 1.2, num_worst_tokens: int = -1,
               combine_input_reuse: bool = True) -> "EPContext":
        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        config = EPConfig(max_m=max_m, hidden=hidden, topk=topk, num_experts=num_experts, rank=rank,
                          world_size=world_size, local_world_size=local_world_size)

        nvl_barrier_symm_tensor = SymmetricTensor(shape=(config.local_world_size, ), dtype=torch.int32, group=group)
        nvl_barrier_symm_tensor.get_local_tensor().fill_(0)
        full_splits_symm_tensor = SymmetricTensor(shape=(config.world_size, config.num_experts + 1),
                                                  dtype=config.offset_dtype, group=group)
        dispatch_recv_tokens = int((max_m * topk * capacity_coeff + 1023) / 1024 * 1024)
        recv_token_count_cpu = torch.empty((config.world_size, ), dtype=torch.int32, device="cpu", pin_memory=True)
        recv_token_count_cpu.fill_(-1)
        recv_token_count = torch.empty((config.world_size, ), dtype=torch.int32, device="cuda")

        if num_worst_tokens > 0:
            dispatch_recv_tokens = num_worst_tokens
        dispatch_output_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, hidden), dtype=config.token_dtype,
                                                      group=group)
        dispatch_topk_weights_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, topk),
                                                            dtype=config.weight_dtype, group=group)
        dispatch_topk_scatter_indices_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, topk),
                                                                    dtype=config.offset_dtype, group=group)
        dispatch_topk_scatter_indices_symm_tensor.get_local_tensor().fill_(
            -1)  # init with -1, indicating the invalid index

        # reuse dispatch output buf as combine input buf
        if combine_input_reuse:
            combine_input_symm_tensor = dispatch_output_symm_tensor
            combine_weight_symm_tensor = dispatch_topk_weights_symm_tensor
        else:
            combine_input_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, hidden), dtype=config.token_dtype,
                                                        group=group)
            combine_weight_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, topk), dtype=config.weight_dtype,
                                                         group=group)
        combine_topk_weights_buf = combine_weight_symm_tensor.get_local_tensor()
        combine_topk_weights_buf_ptrs = combine_weight_symm_tensor.ptrs

        assert config.nnodes == 1, "current only support single node"
        ctx = EPContext(config=config, group=group, nvl_barrier_buf=nvl_barrier_symm_tensor.get_local_tensor(),
                        nvl_barrier_buf_ptrs=nvl_barrier_symm_tensor.ptrs,
                        full_splits_buf=full_splits_symm_tensor.get_local_tensor(),
                        full_splits_buf_ptrs=full_splits_symm_tensor.ptrs,
                        dispatch_output_buf=dispatch_output_symm_tensor.get_local_tensor(),
                        dispatch_output_buf_ptrs=dispatch_output_symm_tensor.ptrs,
                        dispatch_topk_weights_buf=dispatch_topk_weights_symm_tensor.get_local_tensor(),
                        dispatch_topk_weights_buf_ptrs=dispatch_topk_weights_symm_tensor.ptrs,
                        dispatch_topk_scatter_indices_buf=dispatch_topk_scatter_indices_symm_tensor.get_local_tensor(),
                        dispatch_topk_scatter_indices_buf_ptrs=dispatch_topk_scatter_indices_symm_tensor.ptrs,
                        combine_input_buf=combine_input_symm_tensor.get_local_tensor(),
                        combine_input_buf_ptrs=combine_input_symm_tensor.ptrs,
                        combine_topk_weights_buf=combine_topk_weights_buf,
                        combine_topk_weights_buf_ptrs=combine_topk_weights_buf_ptrs,
                        recv_token_count_cpu=recv_token_count_cpu, recv_token_count=recv_token_count,
                        # rdma buffers
                        rdma_token_send_buf=None, rdma_token_dst_scatter_indices_buf=None,
                        rdma_token_topk_send_mask_buf=None, rdma_topk_weights_buf=None, rdma_topk_indices_buf=None)

        # keep references to SymmetricTensor objects to prevent garbage collection
        # the underlying memory would be freed if these objects are collected
        ctx._symm_tensors = {
            'nvl_barrier': nvl_barrier_symm_tensor,
            'full_splits': full_splits_symm_tensor,
            'dispatch_output': dispatch_output_symm_tensor,
            'dispatch_topk_weights': dispatch_topk_weights_symm_tensor,
            'dispatch_topk_scatter_indices': dispatch_topk_scatter_indices_symm_tensor,
            'combine_input': combine_input_symm_tensor,
            'combine_weight': combine_weight_symm_tensor,
        }
        return ctx
