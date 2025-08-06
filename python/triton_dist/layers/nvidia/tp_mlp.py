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
from torch import nn
import torch.distributed

from triton_dist.kernels.allreduce import AllReduceMethod
from triton_dist.kernels.nvidia.allgather_gemm import AllGatherGEMMTensorParallelContext, get_auto_all_gather_method, ag_gemm
from triton_dist.kernels.nvidia import create_gemm_rs_context, gemm_rs
from triton_dist.utils import nvshmem_barrier_all_on_stream
from triton_dist.kernels.nvidia.allreduce import (create_allreduce_ctx, all_reduce)


def shard_local(tensor: torch.Tensor, world_size: int, dim: int, local_rank: int):
    tensor_dim = tensor.shape[dim]
    tensor_slice = tensor_dim // world_size
    if tensor_dim % world_size != 0:
        raise ValueError(f"Tensor dimension {tensor_dim} is not divisible by world size {world_size}.")
    if local_rank < 0 or local_rank >= world_size:
        raise ValueError(f"Local rank {local_rank} is out of bounds for world size {world_size}.")
    if dim < 0 or dim >= tensor.dim():
        raise ValueError(f"Dimension {dim} is out of bounds for tensor with {tensor.dim()} dimensions.")
    if tensor_slice == 0:
        raise ValueError(f"Tensor slice size is zero for tensor dimension {tensor_dim} and world size {world_size}.")
    return tensor.split(tensor_slice, dim=dim)[local_rank].contiguous()


class TP_MLP:
    """
    Tensor Parallel MLP.
    This MLP uses a common TP strategy:
    1. First linear layer (gate/up) weights are column-parallel.
    2. Second linear layer (down) weights are row-parallel.
    """

    def __init__(self, rank=0, world_size=8, group=None):
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.act_fn = None
        self.gate_up_proj = None
        self.down_proj = None
        self.ag_ctx = None
        self.rs_ctx = None
        self.ar_ctx = None

    def _init_parameters(self, mlp: nn.Module, verbose=False):
        """
        Initializes and shards MLP parameters for Tensor Parallelism.
        mlp: A standard nn.Module MLP (e.g., from HuggingFace Transformers).
             Expected to have mlp.gate_proj, mlp.up_proj, mlp.down_proj.
        """
        gate_proj = shard_local(mlp.gate_proj.weight.detach(), self.world_size, 0, self.rank)
        up_proj = shard_local(mlp.up_proj.weight.detach(), self.world_size, 0, self.rank)
        self.gate_up_proj = torch.cat((gate_proj, up_proj),
                                      dim=0).to("cuda", non_blocking=True)  # [MLP_size * 2 // world_size, hidden_size]
        self.down_proj = shard_local(mlp.down_proj.weight.detach(), self.world_size, 1,
                                     self.rank).to("cuda", non_blocking=True)  # [hidden_size, MLP_size // world_size]

        self.act_fn = mlp.act_fn
        self.ag_N_per_rank = self.gate_up_proj.shape[0]
        self.K = self.gate_up_proj.shape[1]
        self.dtype = self.gate_up_proj.dtype

        assert mlp.gate_proj.bias is None, "We do not support bias for now."

        if verbose:
            print(
                f"[RANK {self.rank}] MLP initialized with parameters: gate_up_proj shape: {self.gate_up_proj.shape}, down_proj shape: {self.down_proj.shape}"
            )

    def _init_ctx(self, max_M, ag_intranode_stream, ag_internode_stream, BLOCK_M, BLOCK_N, BLOCK_K, stages):
        # TODO(houqi.1993) BLOCK_SIZE should not be part of arguments, but be determined on forward.
        """Initializes contexts for triton_dist AllGather-GEMM and GEMM-ReduceScatter operations."""
        self.ag_ctx = AllGatherGEMMTensorParallelContext(
            N_per_rank=self.ag_N_per_rank, K=self.K, tensor_dtype=self.dtype, rank=self.rank, num_ranks=self.world_size,
            num_local_ranks=self.world_size, max_M=max_M, ag_intranode_stream=ag_intranode_stream,
            ag_internode_stream=ag_internode_stream, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, stages=stages,
            all_gather_method=get_auto_all_gather_method(self.world_size, self.world_size))
        self.rs_ctx = create_gemm_rs_context(
            max_M=max_M,
            N=self.K,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=self.world_size,
            output_dtype=self.dtype,
            rs_stream=ag_intranode_stream,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            stages=stages,
        )
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    def finalize(self):
        if self.ag_ctx:
            self.ag_ctx.finailize()
        if self.rs_ctx:
            self.rs_ctx.finalize()
        if self.ar_ctx:
            self.ar_ctx.finalize()

    @torch.inference_mode()
    def torch_fwd(self, x):
        '''
        Reference PyTorch forward pass using Tensor Parallelism.
        Final output is AllReduced.
        x: input tensor, shape [batch_size * seq_len, hidden_size] or [batch_size, seq_len, hidden_size]
        '''
        out_fused = torch.nn.functional.linear(x, self.gate_up_proj)
        wg, w1 = torch.chunk(out_fused, 2, dim=-1)
        out = self.act_fn(wg) * w1
        out = torch.nn.functional.linear(out, self.down_proj)
        if self.world_size > 1:
            torch.distributed.all_reduce(out, torch.distributed.ReduceOp.SUM, group=self.group)
        return out

    @torch.inference_mode()
    def dist_triton_fwd(self, x: torch.Tensor, ag_gemm_persistent=False, gemm_rs_persistent=False, autotune=True):
        """
        triton_dist forward pass for TP.
        This version uses ag_gemm and gemm_rs.
        x: input tensor, shape [batch_size, seq_len, hidden_size] or [batch_size * seq_len, hidden_size]
        """
        # Reshape input if it's 3D (e.g., [batch, seq_len, hidden_dim])
        if len(x.size()) == 3:
            bsz, seq, d = x.size()
            x = x.view(-1, d)
            is_3d_input = True
        else:
            is_3d_input = False

        # ag + gemm
        out_fused = ag_gemm(x, self.gate_up_proj, ctx=self.ag_ctx, persistent=ag_gemm_persistent, autotune=autotune)
        wg, w1 = torch.chunk(out_fused, 2, dim=-1)
        out = self.act_fn(wg) * w1
        # gemm + rs
        out = gemm_rs(out, self.down_proj, self.rs_ctx, persistent=gemm_rs_persistent, fuse_scatter=True)

        if is_3d_input:
            out = out.view(bsz, seq, -1)
        return out

    def _init_AR_ctx(self, max_M, method: AllReduceMethod, dtype=torch.bfloat16):
        self.ar_method = method
        N = self.down_proj.shape[0]
        self.ar_ctx = create_allreduce_ctx(
            workspace_nbytes=max_M * N * dtype.itemsize, rank=self.rank, world_size=self.world_size,
            local_world_size=self.world_size,  # TODO(houqi.1993) does not support multiple nodes now.
        )

    @torch.inference_mode()
    def dist_triton_AR_fwd(self, x: torch.Tensor):
        """
        triton_dist AR forward pass for TP.
        This version uses gemm + gemm + AllReduce
        x: input tensor, shape [batch_size, seq_len, hidden_size] or [batch_size * seq_len, hidden_size]
        """
        out_fused = torch.nn.functional.linear(x, self.gate_up_proj)
        wg, w1 = torch.chunk(out_fused, 2, dim=-1)
        out = self.act_fn(wg) * w1
        out = torch.nn.functional.linear(out, self.down_proj).view_as(x)
        if self.world_size > 1:
            out_ar = torch.empty_like(out)
            assert self.ar_ctx is not None, "AllReduce context is not initialized."
            out = all_reduce(out.contiguous(), output=out_ar, method=self.ar_method, ctx=self.ar_ctx)
        return out.view_as(x)

    @torch.inference_mode()
    def fwd(self, x: torch.Tensor):
        raise NotImplementedError("Please use torch_fwd or dist_triton_fwd instead.")

    @torch.inference_mode()
    def torch_ag_gemm(self, x: torch.Tensor):
        """
        Reference PyTorch forward pass using AllGather-GEMM.
        """
        M_per_rank, K = x.shape
        M = M_per_rank * self.world_size
        ag_buffer = torch.empty([M, K], dtype=x.dtype, device="cuda")
        # ag
        torch.distributed.all_gather_into_tensor(ag_buffer, x, group=self.group)
        # gemm
        return torch.matmul(ag_buffer, self.gate_up_proj.T)

    @torch.inference_mode()
    def dist_triton_ag_gemm(self, x: torch.Tensor, persistent=True, autotune=False):
        """
        Triton Dist forward pass using AllGather-GEMM.
        This version uses ag_gemm.
        x: input tensor, shape [batch_size * seq_len, hidden_size]
        """
        assert self.ag_ctx is not None
        return ag_gemm(x, self.gate_up_proj, ctx=self.ag_ctx, persistent=persistent, autotune=autotune)

    @torch.inference_mode()
    def torch_gemm_rs(self, x: torch.Tensor):
        """
        Reference PyTorch forward pass using GEMM-ReduceScatter.
        """
        # x: [M, K]
        M, K = x.shape
        rs_buffer = torch.empty([M // self.world_size, self.down_proj.shape[0]], dtype=x.dtype, device="cuda")
        # gemm
        gemm_out = torch.matmul(x, self.down_proj.T)
        torch.distributed.reduce_scatter_tensor(rs_buffer, gemm_out, group=self.group)
        return rs_buffer

    @torch.inference_mode()
    def dist_triton_gemm_rs(self, x: torch.Tensor, persistent=False):
        """
        Triton Dist forward pass using GEMM-ReduceScatter.
        This version uses gemm_rs.
        x: input tensor, shape [batch_size * seq_len, hidden_size]
        """
        assert self.rs_ctx is not None
        return gemm_rs(x, self.down_proj, self.rs_ctx, persistent=persistent, fuse_scatter=True)
