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


class _CuTeDSLCombinePushBase(CuTeDSLEPOverlapOpBase):
    """Shared scaffolding for the 1D and tile push variants."""

    def _dummy_push_input(self, dtype: torch.dtype, hidden_size: int, max_recv_tokens: int) -> torch.Tensor:
        return torch.empty((max_recv_tokens, hidden_size), dtype=dtype, device="cuda")

    def _dummy_push_recv_count(self) -> torch.Tensor:
        return torch.empty((self.world_size, ), dtype=torch.int32, device="cuda")


class CuTeDSLCombinePushOp(_CuTeDSLCombinePushBase):
    """1D push-mode combine."""

    def _compile(self, dtype: torch.dtype, hidden: int, topk: int, meta: torch.Tensor, output_ptrs: torch.Tensor):
        variant_args = (
            dtype,
            int(hidden),
            int(topk),
            int(self.rank),
            int(self.world_size),
        )

        def factory():
            return self._build(dtype, hidden, topk, meta, output_ptrs)

        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
        )

    def _build(self, dtype: torch.dtype, hidden: int, topk: int, meta: torch.Tensor, output_ptrs: torch.Tensor):
        import cutlass
        import cutlass.cute as cute

        from ..kernels.cutedsl_combine import MoECombinePush

        self._assert_supported_dtype(dtype, "CuTeDSL combine push")

        return cute.compile(
            MoECombinePush(),
            mark_dynamic(self._dummy_push_input(dtype, hidden, 1), enable_tvm_ffi=True),
            mark_dynamic(meta, enable_tvm_ffi=True),
            mark_dynamic(self._dummy_push_recv_count(), enable_tvm_ffi=True),
            mark_dynamic(output_ptrs, enable_tvm_ffi=True),
            int(hidden),
            int(topk),
            int(self.rank),
            int(self.world_size),
            cutlass.Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=cute_compile_options(enable_tvm_ffi=True),
        )

    def run(self, *, input_buf: torch.Tensor, meta: torch.Tensor, recv_count: torch.Tensor, output_ptrs: torch.Tensor,
            topk: int, comm_num_sm: int) -> None:
        compiled = self._compile(
            dtype=input_buf.dtype,
            hidden=int(input_buf.shape[1]),
            topk=topk,
            meta=meta,
            output_ptrs=output_ptrs,
        )
        compiled(
            input_buf,
            meta,
            recv_count,
            output_ptrs,
            int(comm_num_sm),
        )


class CuTeDSLCombineTilePushOp(_CuTeDSLCombinePushBase):
    """2D tile push-mode combine."""

    def _compile(self, dtype: torch.dtype, hidden: int, topk: int, tile_m: int, tile_n: int, meta: torch.Tensor,
                 output_ptrs: torch.Tensor):
        variant_args = (
            dtype,
            int(hidden),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(tile_m),
            int(tile_n),
        )

        def factory():
            return self._build(dtype, hidden, topk, tile_m, tile_n, meta, output_ptrs)

        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
        )

    def _build(self, dtype: torch.dtype, hidden: int, topk: int, tile_m: int, tile_n: int, meta: torch.Tensor,
               output_ptrs: torch.Tensor):
        import cutlass
        import cutlass.cute as cute

        from ..kernels.cutedsl_combine import MoECombineTilePush

        self._assert_supported_dtype(dtype, "CuTeDSL combine tile push")

        return cute.compile(
            MoECombineTilePush(),
            mark_dynamic(self._dummy_push_input(dtype, hidden, 1), enable_tvm_ffi=True),
            mark_dynamic(meta, enable_tvm_ffi=True),
            mark_dynamic(self._dummy_push_recv_count(), enable_tvm_ffi=True),
            mark_dynamic(output_ptrs, enable_tvm_ffi=True),
            int(hidden),
            int(topk),
            int(self.rank),
            int(self.world_size),
            cutlass.Int32(0),
            int(tile_m),
            int(tile_n),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=cute_compile_options(enable_tvm_ffi=True),
        )

    def run(self, *, input_buf: torch.Tensor, meta: torch.Tensor, recv_count: torch.Tensor, output_ptrs: torch.Tensor,
            tile_m: int, tile_n: int, topk: int, comm_num_sm: int) -> None:
        compiled = self._compile(
            dtype=input_buf.dtype,
            hidden=int(input_buf.shape[1]),
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            meta=meta,
            output_ptrs=output_ptrs,
        )
        compiled(
            input_buf,
            meta,
            recv_count,
            output_ptrs,
            int(comm_num_sm),
        )
