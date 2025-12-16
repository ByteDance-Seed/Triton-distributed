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

from triton_dist.kernels.amd.all_gather_gemm import create_ag_gemm_intra_node_context, ag_gemm_intra_node
from triton_dist.kernels.amd.gemm_reduce_scatter import create_gemm_rs_intra_node_context, gemm_rs_intra_node


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

    def __init__(self, rank=0, world_size=8, group: torch.distributed.ProcessGroup = None):
        self.rank = rank
        self.world_size = world_size
        self.group: torch.distributed.ProcessGroup = group
        self.act_fn = None
        self.gate_up_proj: torch.Tensor = None
        self.down_proj: torch.Tensor = None

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
                                     self.rank).to("cuda", non_blocking=True)  # [MLP_size // world_size, hidden_size]

        self.act_fn = mlp.act_fn
        self.ag_N_per_rank = self.gate_up_proj.shape[0]
        self.K = self.gate_up_proj.shape[1]
        self.dtype = self.gate_up_proj.dtype

        assert mlp.gate_proj.bias is None, "We do not support bias for now."

        if verbose:
            print(
                f"[RANK {self.rank}] MLP initialized with parameters: gate_up_proj shape: {self.gate_up_proj.shape}, down_proj shape: {self.down_proj.shape}"
            )

    def _init_ctx(self, max_M, ag_intranode_stream, ag_internode_stream=None):
        """Initializes contexts for triton_dist AllGather-GEMM and GEMM-ReduceScatter operations."""
        self.ag_ctx = create_ag_gemm_intra_node_context(max_M=max_M, N=self.ag_N_per_rank, K=self.K, rank=self.rank,
                                                        num_ranks=self.world_size, input_dtype=self.dtype,
                                                        output_dtype=self.dtype, tp_group=self.group,
                                                        ag_streams=ag_intranode_stream, M_PER_CHUNK=256)
        self.rs_ctx = create_gemm_rs_intra_node_context(
            max_M=max_M,
            N=self.K,
            rank=self.rank,
            num_ranks=self.world_size,
            output_dtype=self.dtype,
            tp_group=self.group,
            fuse_scatter=True,
        )
        torch.cuda.synchronize()

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
    def dist_triton_fwd(self, x):
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
        out_fused = ag_gemm_intra_node(x, self.gate_up_proj, ctx=self.ag_ctx)
        wg, w1 = torch.chunk(out_fused, 2, dim=-1)
        out = self.act_fn(wg) * w1
        # gemm + rs
        out = gemm_rs_intra_node(out, self.down_proj, self.rs_ctx)

        if is_3d_input:
            out = out.view(bsz, seq, -1)
        return out

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

        if not hasattr(self, 'ag_buffer'):
            self.ag_buffer = torch.empty([M, K], dtype=x.dtype, device="cuda")

        # ag
        torch.distributed.all_gather_into_tensor(self.ag_buffer, x, group=self.group)

        # gemm
        golden = torch.matmul(self.ag_buffer, self.gate_up_proj.T)

        return golden

    @torch.inference_mode()
    def dist_triton_ag_gemm(self, x: torch.Tensor):
        """
        Triton Dist forward pass using AllGather-GEMM.
        This version uses ag_gemm.
        x: input tensor, shape [batch_size * seq_len, hidden_size]
        """
        out = ag_gemm_intra_node(x, self.gate_up_proj, ctx=self.ag_ctx)
        return out

    @torch.inference_mode()
    def torch_gemm_rs(self, x: torch.Tensor):
        """
        Reference PyTorch forward pass using GEMM-ReduceScatter.
        """
        # x: [M, K]
        M, K = x.shape
        if not hasattr(self, 'rs_buffer'):
            self.rs_buffer = torch.empty([M // self.world_size, self.down_proj.shape[0]], dtype=x.dtype, device="cuda")

        # gemm
        gemm_out = torch.matmul(x, self.down_proj.T)

        # rs
        torch.distributed.reduce_scatter_tensor(self.rs_buffer, gemm_out, group=self.group)

        return self.rs_buffer

    @torch.inference_mode()
    def dist_triton_gemm_rs(self, x: torch.Tensor):
        """
        Triton Dist forward pass using GEMM-ReduceScatter.
        This version uses gemm_rs.
        x: input tensor, shape [batch_size * seq_len, hidden_size]
        """
        out = gemm_rs_intra_node(x, self.down_proj, self.rs_ctx)
        return out
