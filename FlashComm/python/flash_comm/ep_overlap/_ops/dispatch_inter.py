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

from ._base import CuTeDSLEPOverlapOpBase, cute_compile_options, mark_dynamic


class CuTeDSLDispatchInterOp(CuTeDSLEPOverlapOpBase):
    """CuTeDSL inter-node dispatch using NCCL GIN + local TMA.

    The concrete CuTeDSL kernel lives in
    ``kernels/cutedsl_dispatch_inter.py``.  The small scaffold guard below is
    kept so an accidental placeholder restore fails with a clear error.
    """

    def __init__(self, *, rank: int, world_size: int, local_world_size: int):
        super().__init__(rank=rank, world_size=world_size)
        self.local_world_size = int(local_world_size)

    def _compile(
        self,
        *,
        dtype: torch.dtype,
        hidden: int,
        topk: int,
        max_slot_num_token: int,
        max_recv_tokens: int,
        experts_per_rank: int,
    ):
        variant_args = (
            dtype,
            int(hidden),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(self.local_world_size),
            int(max_slot_num_token),
            int(max_recv_tokens),
            int(experts_per_rank),
        )

        def factory():
            return self._build(
                dtype=dtype,
                hidden=hidden,
                topk=topk,
                max_slot_num_token=max_slot_num_token,
                max_recv_tokens=max_recv_tokens,
                experts_per_rank=experts_per_rank,
            )

        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
        )

    def _build(
        self,
        *,
        dtype: torch.dtype,
        hidden: int,
        topk: int,
        max_slot_num_token: int,
        max_recv_tokens: int,
        experts_per_rank: int,
    ):
        import cutlass
        import cutlass.cute as cute

        from ..kernels.cutedsl_dispatch_inter import MoEDispatchInter

        self._assert_supported_dtype(dtype, "CuTeDSL inter-node dispatch")
        if getattr(MoEDispatchInter, "is_scaffold", False):
            raise NotImplementedError("cutedsl_dispatch_inter.py is scaffolded; implement "
                                      "MoEDispatchInter before running inter-node dispatch_cutedsl")

        dummy_num_tokens_per_rank = torch.empty((self.world_size, ), dtype=torch.int32, device="cuda")
        dummy_recv_x_ptrs = torch.empty((self.local_world_size, ), dtype=torch.int64, device="cuda")
        dummy_recv_scatter_ptrs = torch.empty((self.local_world_size, ), dtype=torch.int64, device="cuda")
        dummy_node_topk_indices = torch.empty(
            (self.world_size // self.local_world_size, max_slot_num_token, topk),
            dtype=torch.int32,
            device="cuda",
        )
        dummy_node_mask = torch.empty_like(dummy_node_topk_indices)
        dummy_node_scatter = torch.empty_like(dummy_node_topk_indices)
        dummy_output = torch.empty((1, hidden), dtype=dtype, device="cuda")

        return cute.compile(
            MoEDispatchInter(),
            cutlass.Int64(0),
            cutlass.Int64(0),
            mark_dynamic(dummy_num_tokens_per_rank, enable_tvm_ffi=True),
            mark_dynamic(dummy_recv_x_ptrs, enable_tvm_ffi=True),
            mark_dynamic(dummy_recv_scatter_ptrs, enable_tvm_ffi=True),
            mark_dynamic(dummy_node_topk_indices, enable_tvm_ffi=True),
            mark_dynamic(dummy_node_mask, enable_tvm_ffi=True),
            mark_dynamic(dummy_node_scatter, enable_tvm_ffi=True),
            mark_dynamic(dummy_output, enable_tvm_ffi=True),
            int(hidden),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(self.local_world_size),
            int(max_slot_num_token),
            int(max_recv_tokens),
            int(experts_per_rank),
            cutlass.Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=cute_compile_options(enable_tvm_ffi=True),
        )

    def run(
        self,
        *,
        dev_comm_ptr: int,
        rdma_rail_send_win_handle: int,
        num_tokens_per_rank: torch.Tensor,
        recv_x_ptrs: torch.Tensor,
        recv_topk_scatter_indices_ptrs: torch.Tensor,
        node_topk_indices: torch.Tensor,
        node_topk_send_mask: torch.Tensor,
        node_token_dst_scatter_indices: torch.Tensor,
        output_buf: torch.Tensor,
        max_slot_num_token: int,
        max_recv_tokens: int,
        experts_per_rank: int,
        num_sm: int,
    ) -> None:
        compiled = self._compile(
            dtype=output_buf.dtype,
            hidden=int(output_buf.shape[1]),
            topk=int(node_topk_indices.shape[2]),
            max_slot_num_token=int(max_slot_num_token),
            max_recv_tokens=int(max_recv_tokens),
            experts_per_rank=int(experts_per_rank),
        )
        compiled(int(dev_comm_ptr), int(rdma_rail_send_win_handle), num_tokens_per_rank, recv_x_ptrs,
                 recv_topk_scatter_indices_ptrs, node_topk_indices, node_topk_send_mask, node_token_dst_scatter_indices,
                 output_buf, int(num_sm))
