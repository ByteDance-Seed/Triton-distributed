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


class CuTeDSLDispatchOp(CuTeDSLEPOverlapOpBase):
    """CuTeDSL pull-mode intranode dispatch."""

    def __init__(self, *, rank: int, world_size: int, expert_alignment: int):
        super().__init__(rank=rank, world_size=world_size)
        self.expert_alignment = int(expert_alignment)

    def _compile(self, dtype: torch.dtype, hidden: int, experts_per_rank: int, expert_alignment: int,
                 enable_expert_signals: bool, dispatch_input_ptrs: torch.Tensor,
                 token_src_rank_topk_and_indices: torch.Tensor, expert_signals: torch.Tensor,
                 expert_signal_counters: torch.Tensor):
        variant_args = (
            dtype,
            int(hidden),
            int(self.rank),
            int(self.world_size),
            int(experts_per_rank),
            int(expert_alignment),
            bool(enable_expert_signals),
        )

        def factory():
            return self._build(
                dtype,
                hidden,
                experts_per_rank,
                expert_alignment,
                enable_expert_signals,
                dispatch_input_ptrs,
                token_src_rank_topk_and_indices,
                expert_signals,
                expert_signal_counters,
            )

        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
        )

    def _build(self, dtype: torch.dtype, hidden: int, experts_per_rank: int, expert_alignment: int,
               enable_expert_signals: bool, dispatch_input_ptrs: torch.Tensor,
               token_src_rank_topk_and_indices: torch.Tensor, expert_signals: torch.Tensor,
               expert_signal_counters: torch.Tensor):
        import cutlass
        import cutlass.cute as cute

        from ..kernels.cutedsl_dispatch import MoEDispatch

        self._assert_supported_dtype(dtype, "CuTeDSL dispatch")

        dummy_recv_count = torch.empty((self.world_size, ), dtype=torch.int32, device="cuda")
        dummy_recv_expert_counts = torch.empty((experts_per_rank, ), dtype=torch.int32, device="cuda")
        dummy_output_buf = torch.empty(
            (1, hidden),
            dtype=dtype,
            device="cuda",
        )

        return cute.compile(
            MoEDispatch(),
            mark_dynamic(dispatch_input_ptrs, enable_tvm_ffi=True),
            mark_dynamic(token_src_rank_topk_and_indices, enable_tvm_ffi=True),
            mark_dynamic(dummy_recv_count, enable_tvm_ffi=True),
            mark_dynamic(dummy_output_buf, enable_tvm_ffi=True),
            int(hidden),
            int(self.rank),
            int(self.world_size),
            cutlass.Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            mark_dynamic(dummy_recv_expert_counts, enable_tvm_ffi=True),
            mark_dynamic(expert_signals, enable_tvm_ffi=True),
            mark_dynamic(expert_signal_counters, enable_tvm_ffi=True),
            int(experts_per_rank),
            int(expert_alignment),
            int(enable_expert_signals),
            options=cute_compile_options(enable_tvm_ffi=True),
        )

    def run(self, *, dispatch_input_ptrs: torch.Tensor, token_src_rank_topk_and_indices: torch.Tensor,
            recv_count: torch.Tensor, output_buf: torch.Tensor, recv_expert_counts: torch.Tensor,
            expert_signals: torch.Tensor, expert_signal_counters: torch.Tensor, enable_expert_signals: bool,
            comm_num_sm: int) -> None:
        """Launch the compiled pull-dispatch kernel."""
        compiled = self._compile(
            dtype=output_buf.dtype,
            hidden=int(output_buf.shape[1]),
            experts_per_rank=int(recv_expert_counts.shape[0]),
            expert_alignment=int(self.expert_alignment),
            enable_expert_signals=enable_expert_signals,
            dispatch_input_ptrs=dispatch_input_ptrs,
            token_src_rank_topk_and_indices=token_src_rank_topk_and_indices,
            expert_signals=expert_signals,
            expert_signal_counters=expert_signal_counters,
        )
        compiled(
            dispatch_input_ptrs,
            token_src_rank_topk_and_indices,
            recv_count,
            output_buf,
            int(comm_num_sm),
            recv_expert_counts,
            expert_signals,
            expert_signal_counters,
        )
