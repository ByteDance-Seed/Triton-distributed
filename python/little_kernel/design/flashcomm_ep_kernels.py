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
FlashComm EP Kernels - Elegant Python host-side binding for LittleKernel.

This module mirrors FlashComm's EPKernels class but uses LittleKernel's
build() API instead of C++ extension modules. All CUDA compilation,
kernel argument marshalling, and launch are handled by LK's runtime.

Usage:
    ep = FlashCommEPKernels(
        max_m=4096, hidden=7168, topk=8,
        num_experts=384, num_ranks=8, rank=0,
    )
    # Compute phase (single GPU)
    offset, counts = ep.compute_token_offset(topk_indices)
    layout = ep.compute_dispatch_layout(topk_indices, offset, counts, ...)

    # Dispatch/combine phase (multi-GPU with symmetric memory)
    ep.dispatch(...)
    ep.combine(...)
"""
import os
import sys
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

# Import LK kernel definitions
from flashcomm_compute import (
    kernel_compute_offset,
    kernel_compute_dispatch_layout,
    NUM_WARPS,
    MAX_EXPERTS_PLUS_1,
    BLOCK_SIZE,
)


@dataclass
class EPCommLayoutDesc:
    """Layout descriptor produced by compute phase, consumed by dispatch/combine."""
    token_within_expert_offset: torch.Tensor  # [num_token, topk]
    expert_counts: torch.Tensor  # [num_experts + 1]
    recv_base_offset: Optional[torch.Tensor] = None
    token_dst_scatter_indices: Optional[torch.Tensor] = None
    token_topk_send_mask: Optional[torch.Tensor] = None
    recv_token_count: Optional[torch.Tensor] = None


class FlashCommEPKernels:
    """
    Host-side manager for FlashComm EP kernels built with LittleKernel.

    Handles kernel compilation (cached), memory allocation, and kernel launch
    using LK's elegant Python binding (kernel.build() + compiled_kernel()).
    """

    def __init__(
        self,
        max_m: int,
        hidden: int,
        topk: int,
        num_experts: int,
        num_ranks: int = 1,
        rank: int = 0,
        num_sm: int = 4,
        device: str = "cuda",
    ):
        self.max_m = max_m
        self.hidden = hidden
        self.topk = topk
        self.num_experts = num_experts
        self.num_ranks = num_ranks
        self.rank = rank
        self.num_sm = num_sm
        self.device = device
        self.experts_per_rank = num_experts // num_ranks

        # Compiled kernels (lazy build)
        self._compute_offset_kernel = None
        self._dispatch_layout_kernel = None

    # ------------------------------------------------------------------
    # Kernel builders (lazy, cached)
    # ------------------------------------------------------------------
    @property
    def compute_offset_kernel(self):
        if self._compute_offset_kernel is None:
            actual_experts_plus_1 = max(self.num_experts + 1, MAX_EXPERTS_PLUS_1)
            smem = NUM_WARPS * actual_experts_plus_1 * 4
            self._compute_offset_kernel = kernel_compute_offset.build(
                passes=PASSES["cuda"],
                codegen_func=codegen_cuda,
                grid=(self.num_sm, 1, 1),
                block=(BLOCK_SIZE, 1, 1),
                shared_mem_bytes=smem,
            )
        return self._compute_offset_kernel

    @property
    def dispatch_layout_kernel(self):
        if self._dispatch_layout_kernel is None:
            self._dispatch_layout_kernel = kernel_compute_dispatch_layout.build(
                passes=PASSES["cuda"],
                codegen_func=codegen_cuda,
                grid=(self.num_sm, 1, 1),
                block=(BLOCK_SIZE, 1, 1),
                shared_mem_bytes=NUM_WARPS * 4,
            )
        return self._dispatch_layout_kernel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_token_offset(self, topk_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute stable within-expert token offsets and expert counts.

        Parameters
        ----------
        topk_indices : torch.Tensor[int32], shape [num_token, topk]
            Expert indices per (token, topk-slot). Drop tokens use index = num_experts.

        Returns
        -------
        token_within_expert_offset : torch.Tensor[int32], shape [num_token, topk]
        expert_counts : torch.Tensor[int32], shape [num_experts + 1]
        """
        num_token = topk_indices.shape[0]
        total_elements = num_token * self.topk
        num_tiles = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

        block_cumsum_hist = torch.zeros(
            (num_tiles, self.num_experts + 1),
            device=self.device,
            dtype=torch.int32,
        )
        token_within_expert_offset = torch.zeros(
            (num_token, self.topk),
            device=self.device,
            dtype=torch.int32,
        )
        expert_counts = torch.zeros(
            (self.num_experts + 1, ),
            device=self.device,
            dtype=torch.int32,
        )

        self.compute_offset_kernel(
            topk_indices,
            num_token,
            self.topk,
            self.num_experts,
            block_cumsum_hist,
            token_within_expert_offset,
            expert_counts,
        )

        return token_within_expert_offset, expert_counts

    def compute_dispatch_layout(
        self,
        topk_indices: torch.Tensor,
        token_within_expert_offset: torch.Tensor,
        expert_counts: torch.Tensor,
        full_splits_ptrs: torch.Tensor,
        barrier_ptrs: torch.Tensor,
    ) -> EPCommLayoutDesc:
        """
        Compute dispatch layout: all-gather splits, cumsum -> scatter indices + send mask.

        Parameters
        ----------
        topk_indices : torch.Tensor[int32], shape [num_token, topk]
        token_within_expert_offset : torch.Tensor[int32], shape [num_token, topk]
        expert_counts : torch.Tensor[int32], shape [num_experts + 1]
        full_splits_ptrs : torch.Tensor[int64], shape [num_ranks]
            Device pointers to full_splits buffers (symmetric memory).
        barrier_ptrs : torch.Tensor[int64], shape [num_ranks]
            Device pointers to barrier buffers (symmetric memory).

        Returns
        -------
        EPCommLayoutDesc with recv_base_offset, scatter_indices, send_mask, recv_token_count
        """
        num_token = topk_indices.shape[0]

        recv_base_offset = torch.zeros(
            self.num_ranks * self.experts_per_rank * self.num_ranks,
            device=self.device,
            dtype=torch.int32,
        )
        token_dst_scatter_indices = torch.full(
            (num_token, self.topk),
            -1,
            device=self.device,
            dtype=torch.int32,
        )
        token_topk_send_mask = torch.zeros(
            (num_token, self.topk),
            device=self.device,
            dtype=torch.int32,
        )
        recv_token_count = torch.zeros(
            self.num_ranks,
            device=self.device,
            dtype=torch.int32,
        )

        self.dispatch_layout_kernel(
            topk_indices,
            token_within_expert_offset,
            expert_counts,  # used as local_splits
            full_splits_ptrs,
            barrier_ptrs,
            recv_base_offset,
            token_dst_scatter_indices,
            token_topk_send_mask,
            0,  # recv_token_count_cpu (NULL)
            recv_token_count,
            num_token,
            self.topk,
            self.num_experts,
            self.rank,
            self.num_ranks,
        )

        return EPCommLayoutDesc(
            token_within_expert_offset=token_within_expert_offset,
            expert_counts=expert_counts,
            recv_base_offset=recv_base_offset.view(self.num_ranks, self.experts_per_rank, self.num_ranks),
            token_dst_scatter_indices=token_dst_scatter_indices,
            token_topk_send_mask=token_topk_send_mask,
            recv_token_count=recv_token_count,
        )

    def synchronize(self):
        """Synchronize the CUDA device."""
        torch.cuda.synchronize()


# ============================================================================
# Quick self-test
# ============================================================================
if __name__ == "__main__":
    print("FlashCommEPKernels self-test")
    ep = FlashCommEPKernels(
        max_m=4096,
        hidden=5120,
        topk=8,
        num_experts=256,
        num_ranks=1,
        rank=0,
        num_sm=4,
    )

    torch.manual_seed(42)
    topk_indices = torch.randint(0, 257, (1024, 8), device="cuda", dtype=torch.int32)

    print("  Computing token offsets...")
    offset, counts = ep.compute_token_offset(topk_indices)
    ep.synchronize()
    print(f"  offset shape: {offset.shape}, counts shape: {counts.shape}")
    print(f"  total tokens (non-drop): {counts[:256].sum().item()}")

    # Dispatch layout (single rank)
    full_splits_buf = torch.zeros(257, device="cuda", dtype=torch.int32)
    full_splits_ptrs = torch.tensor([full_splits_buf.data_ptr()], dtype=torch.int64, device="cuda")
    barrier_buf = torch.zeros(1, device="cuda", dtype=torch.int32)
    barrier_ptrs = torch.tensor([barrier_buf.data_ptr()], dtype=torch.int64, device="cuda")

    print("  Computing dispatch layout...")
    layout = ep.compute_dispatch_layout(topk_indices, offset, counts, full_splits_ptrs, barrier_ptrs)
    ep.synchronize()
    print(f"  recv_token_count: {layout.recv_token_count.tolist()}")
    print(f"  send_mask sum: {layout.token_topk_send_mask.sum().item()}")
    print(f"  scatter valid: {(layout.token_dst_scatter_indices >= 0).sum().item()}")

    print("  All OK!")
