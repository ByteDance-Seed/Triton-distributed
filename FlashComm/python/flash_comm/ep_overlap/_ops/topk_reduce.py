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

from ._base import (
    CuTeDSLEPOverlapOpBase,
    cute_compile_options,
    mark_compact_dynamic,
)


class CuTeDSLTopkReduceOp(CuTeDSLEPOverlapOpBase):
    """Cached topk-reduce launcher."""

    def _compile(self, dtype: torch.dtype, hidden_size: int, topk: int, num_experts: int):
        variant_args = (
            dtype,
            int(hidden_size),
            int(topk),
            int(num_experts),
        )

        def factory():
            return self._build(dtype, hidden_size, topk, num_experts)

        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
        )

    def _build(self, dtype: torch.dtype, hidden_size: int, topk: int, num_experts: int):
        import cutlass
        import cutlass.cute as cute
        from ..kernels.cutedsl_topk_reduce import MoETopkReduceBlockPerToken

        self._assert_supported_dtype(dtype, "CuTeDSL topk reduce")
        dummy_staging = torch.empty(
            (1, hidden_size),
            dtype=dtype,
            device="cuda",
        )
        dummy_topk = torch.empty(
            (1, topk),
            dtype=torch.int32,
            device="cuda",
        )
        dummy_output = torch.empty(
            (1, hidden_size),
            dtype=dtype,
            device="cuda",
        )
        return cute.compile(
            MoETopkReduceBlockPerToken(),
            mark_compact_dynamic(dummy_staging, assumed_align=16, enable_tvm_ffi=True),
            mark_compact_dynamic(dummy_topk, assumed_align=4, enable_tvm_ffi=True),
            mark_compact_dynamic(dummy_output, assumed_align=16, enable_tvm_ffi=True),
            int(hidden_size),
            int(topk),
            int(num_experts),
            cutlass.Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=cute_compile_options(enable_tvm_ffi=True),
        )

    def run(self, *, staging: torch.Tensor, topk_indices: torch.Tensor, output: torch.Tensor, num_experts: int) -> None:
        """Run topk reduce. ``hidden_size`` is read from ``output.shape[1]``."""
        hidden_size = int(output.shape[1])
        if staging.shape[1] != hidden_size:
            raise ValueError(f"staging hidden ({staging.shape[1]}) != output hidden "
                             f"({hidden_size})")
        compiled = self._compile(
            dtype=output.dtype,
            hidden_size=hidden_size,
            topk=int(topk_indices.shape[1]),
            num_experts=num_experts,
        )
        compiled(
            staging,
            topk_indices,
            output,
            int(output.shape[0]),
        )
