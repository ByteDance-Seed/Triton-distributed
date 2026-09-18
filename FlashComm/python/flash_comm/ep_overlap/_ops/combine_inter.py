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

from typing import Optional

import torch

from ._base import (
    CuTeDSLEPOverlapOpBase,
    cute_compile_options,
    mark_compact_dynamic,
    mark_dynamic,
)


class CuTeDSLCombineInterOp(CuTeDSLEPOverlapOpBase):
    """Standalone inter-node CuTeDSL combine scaffold.

    The intended implementation mirrors the CUDA protocol:
      1. build per-owner-node partials in RDMA rail combine slots;
      2. exchange remote-owner partials with NCCL GIN;
      3. reduce incoming per-node partials into dense local output.
    """

    def __init__(self, *, rank: int, world_size: int, local_world_size: int):
        super().__init__(rank=rank, world_size=world_size)
        self.local_world_size = int(local_world_size)

    def _dummy_input(self, dtype: torch.dtype, hidden: int) -> torch.Tensor:
        return torch.empty((1, hidden), dtype=dtype, device="cuda")

    def _dummy_num_tokens_per_rank(self) -> torch.Tensor:
        return torch.empty((self.world_size, ), dtype=torch.int32, device="cuda")

    def _dummy_ptrs(self) -> torch.Tensor:
        return torch.empty((self.local_world_size, ), dtype=torch.int64, device="cuda")

    def _dummy_weight(self, topk: int) -> torch.Tensor:
        return torch.empty((1, topk), dtype=torch.float32, device="cuda")

    def _dummy_meta(self, max_slot_num_token: int, topk: int) -> torch.Tensor:
        return torch.empty(
            (self.world_size // self.local_world_size, max_slot_num_token, topk),
            dtype=torch.int32,
            device="cuda",
        )

    @staticmethod
    def _ensure_kernel_implemented() -> None:
        from ..kernels.cutedsl_combine_inter import (
            MoECombineInterExchange,
            MoECombineInterPartial,
            MoECombineInterReduce,
        )

        if (getattr(MoECombineInterPartial, "is_scaffold", False)
                or getattr(MoECombineInterExchange, "is_scaffold", False)
                or getattr(MoECombineInterReduce, "is_scaffold", False)):
            raise NotImplementedError("cutedsl_combine_inter.py is scaffolded; implement "
                                      "MoECombineInterPartial, MoECombineInterExchange, and "
                                      "MoECombineInterReduce before running inter-node combine_cutedsl")

    def _compile_partial(
        self,
        *,
        dtype: torch.dtype,
        hidden: int,
        topk: int,
        max_slot_num_token: int,
        experts_per_rank: int,
        has_weight: bool,
    ):
        variant_args = (
            "partial",
            dtype,
            int(hidden),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(self.local_world_size),
            int(max_slot_num_token),
            int(experts_per_rank),
            bool(has_weight),
        )

        def factory():
            return self._build_partial(
                dtype=dtype,
                hidden=hidden,
                topk=topk,
                max_slot_num_token=max_slot_num_token,
                experts_per_rank=experts_per_rank,
                has_weight=has_weight,
            )

        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
        )

    def _build_partial(
        self,
        *,
        dtype: torch.dtype,
        hidden: int,
        topk: int,
        max_slot_num_token: int,
        experts_per_rank: int,
        has_weight: bool,
    ):
        import cutlass
        import cutlass.cute as cute

        from ..kernels.cutedsl_combine_inter import MoECombineInterPartial

        self._assert_supported_dtype(dtype, "CuTeDSL inter-node combine partial")
        if getattr(MoECombineInterPartial, "is_scaffold", False):
            self._ensure_kernel_implemented()
        dummy_meta = self._dummy_meta(max_slot_num_token, topk)
        return cute.compile(
            MoECombineInterPartial(),
            mark_dynamic(self._dummy_input(dtype, hidden), enable_tvm_ffi=True),
            mark_dynamic(self._dummy_ptrs(), enable_tvm_ffi=True),
            mark_dynamic(self._dummy_ptrs(), enable_tvm_ffi=True),
            cutlass.Int64(0),
            mark_dynamic(self._dummy_num_tokens_per_rank(), enable_tvm_ffi=True),
            mark_dynamic(dummy_meta, enable_tvm_ffi=True),
            mark_dynamic(dummy_meta, enable_tvm_ffi=True),
            mark_dynamic(dummy_meta, enable_tvm_ffi=True),
            int(hidden),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(self.local_world_size),
            int(max_slot_num_token),
            int(experts_per_rank),
            int(bool(has_weight)),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=cute_compile_options(enable_tvm_ffi=True),
        )

    def _compile_exchange(
        self,
        *,
        dtype: torch.dtype,
        hidden: int,
        topk: int,
        max_slot_num_token: int,
        has_weight: bool,
    ):
        variant_args = (
            "exchange",
            dtype,
            int(hidden),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(self.local_world_size),
            int(max_slot_num_token),
            bool(has_weight),
        )

        def factory():
            return self._build_exchange(
                dtype=dtype,
                hidden=hidden,
                topk=topk,
                max_slot_num_token=max_slot_num_token,
                has_weight=has_weight,
            )

        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
        )

    def _build_exchange(
        self,
        *,
        dtype: torch.dtype,
        hidden: int,
        topk: int,
        max_slot_num_token: int,
        has_weight: bool,
    ):
        import cutlass
        import cutlass.cute as cute

        from ..kernels.cutedsl_combine_inter import MoECombineInterExchange

        self._assert_supported_dtype(dtype, "CuTeDSL inter-node combine exchange")
        if getattr(MoECombineInterExchange, "is_scaffold", False):
            self._ensure_kernel_implemented()
        return cute.compile(
            MoECombineInterExchange(),
            cutlass.Int64(0),
            mark_dynamic(self._dummy_input(dtype, hidden), enable_tvm_ffi=True),
            cutlass.Int64(0),
            mark_dynamic(self._dummy_num_tokens_per_rank(), enable_tvm_ffi=True),
            int(hidden),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(self.local_world_size),
            int(max_slot_num_token),
            int(bool(has_weight)),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=cute_compile_options(enable_tvm_ffi=True),
        )

    def _compile_reduce(
        self,
        *,
        dtype: torch.dtype,
        hidden: int,
        topk: int,
        max_slot_num_token: int,
        experts_per_rank: int,
        has_weight: bool,
    ):
        variant_args = (
            "reduce",
            dtype,
            int(hidden),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(self.local_world_size),
            int(max_slot_num_token),
            int(experts_per_rank),
            bool(has_weight),
        )

        def factory():
            return self._build_reduce(
                dtype=dtype,
                hidden=hidden,
                topk=topk,
                max_slot_num_token=max_slot_num_token,
                experts_per_rank=experts_per_rank,
                has_weight=has_weight,
            )

        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
        )

    def _build_reduce(
        self,
        *,
        dtype: torch.dtype,
        hidden: int,
        topk: int,
        max_slot_num_token: int,
        experts_per_rank: int,
        has_weight: bool,
    ):
        import cutlass
        import cutlass.cute as cute

        from ..kernels.cutedsl_combine_inter import MoECombineInterReduce

        self._assert_supported_dtype(dtype, "CuTeDSL inter-node combine reduce")
        if getattr(MoECombineInterReduce, "is_scaffold", False):
            self._ensure_kernel_implemented()
        return cute.compile(
            MoECombineInterReduce(),
            mark_compact_dynamic(self._dummy_input(dtype, hidden), assumed_align=16, enable_tvm_ffi=True),
            mark_dynamic(self._dummy_weight(topk), enable_tvm_ffi=True),
            cutlass.Int64(0),
            int(hidden),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(self.local_world_size),
            int(max_slot_num_token),
            int(experts_per_rank),
            int(bool(has_weight)),
            cutlass.Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=cute_compile_options(enable_tvm_ffi=True),
        )

    def run(
        self,
        *,
        dev_comm_ptr: int,
        input_buf: torch.Tensor,
        combine_x_ptrs: torch.Tensor,
        combine_weight_ptrs: Optional[torch.Tensor],
        rdma_rail_send_buf: torch.Tensor,
        rdma_rail_send_win_handle: int,
        num_tokens_per_rank: torch.Tensor,
        output: torch.Tensor,
        local_topk_indices: torch.Tensor,
        local_topk_send_mask: torch.Tensor,
        local_token_dst_scatter_indices: torch.Tensor,
        output_weight: Optional[torch.Tensor],
        max_slot_num_token: int,
        experts_per_rank: int,
        num_sm: int,
    ) -> torch.Tensor:
        has_weight = combine_weight_ptrs is not None or output_weight is not None
        if has_weight and (combine_weight_ptrs is None or output_weight is None):
            raise ValueError("combine_weight_ptrs and output_weight must be both set or both None")
        if input_buf.dtype != output.dtype:
            raise TypeError(f"input_buf dtype {input_buf.dtype} must match output dtype {output.dtype}")
        if input_buf.ndim != 2 or output.ndim != 2:
            raise ValueError("input_buf and output must be rank-2 tensors")
        if input_buf.shape[1] != output.shape[1]:
            raise ValueError("input_buf and output hidden dimensions must match")
        if not input_buf.is_cuda or not input_buf.is_contiguous():
            raise ValueError("input_buf must be a contiguous CUDA tensor")
        if not output.is_cuda or not output.is_contiguous():
            raise ValueError("output must be a contiguous CUDA tensor")
        if combine_x_ptrs.dtype != torch.int64 or combine_x_ptrs.shape != (self.local_world_size, ):
            raise ValueError("combine_x_ptrs must be int64 with shape "
                             f"({self.local_world_size},); got "
                             f"{combine_x_ptrs.dtype} {tuple(combine_x_ptrs.shape)}")
        if num_tokens_per_rank.dtype != torch.int32 or num_tokens_per_rank.shape != (self.world_size, ):
            raise ValueError("num_tokens_per_rank must be int32 with shape "
                             f"({self.world_size},); got "
                             f"{num_tokens_per_rank.dtype} {tuple(num_tokens_per_rank.shape)}")
        if not rdma_rail_send_buf.is_cuda or not rdma_rail_send_buf.is_contiguous():
            raise ValueError("rdma_rail_send_buf must be a contiguous CUDA tensor")

        if local_topk_indices.ndim != 3:
            raise ValueError("local_topk_indices must be rank-3 "
                             f"(nnodes, max_slot_num_token, topk); got {tuple(local_topk_indices.shape)}")
        topk = int(local_topk_indices.shape[2])
        expected_meta_shape = (self.world_size // self.local_world_size, int(max_slot_num_token), topk)
        for tensor, name in (
            (local_topk_indices, "local_topk_indices"),
            (local_topk_send_mask, "local_topk_send_mask"),
            (local_token_dst_scatter_indices, "local_token_dst_scatter_indices"),
        ):
            if tensor.dtype != torch.int32:
                raise TypeError(f"{name} must be int32; got {tensor.dtype}")
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise ValueError(f"{name} must be a contiguous CUDA tensor")
            if tuple(tensor.shape) != expected_meta_shape:
                raise ValueError(f"{name} must have shape {expected_meta_shape}; "
                                 f"got {tuple(tensor.shape)}")

        if has_weight:
            if combine_weight_ptrs.dtype != torch.int64 or combine_weight_ptrs.shape != (self.local_world_size, ):
                raise ValueError("combine_weight_ptrs must be int64 with shape "
                                 f"({self.local_world_size},); got "
                                 f"{combine_weight_ptrs.dtype} {tuple(combine_weight_ptrs.shape)}")
            if output_weight.dtype != torch.float32 or output_weight.shape != (output.shape[0], topk):
                raise ValueError("output_weight must be float32 with shape "
                                 f"({output.shape[0]}, {topk}); got "
                                 f"{output_weight.dtype} {tuple(output_weight.shape)}")
            if not output_weight.is_cuda or not output_weight.is_contiguous():
                raise ValueError("output_weight must be a contiguous CUDA tensor")

        self._ensure_kernel_implemented()

        hidden = int(output.shape[1])
        partial = self._compile_partial(
            dtype=input_buf.dtype,
            hidden=hidden,
            topk=topk,
            max_slot_num_token=int(max_slot_num_token),
            experts_per_rank=int(experts_per_rank),
            has_weight=has_weight,
        )
        exchange = self._compile_exchange(
            dtype=input_buf.dtype,
            hidden=hidden,
            topk=topk,
            max_slot_num_token=int(max_slot_num_token),
            has_weight=has_weight,
        )
        reduce = self._compile_reduce(
            dtype=input_buf.dtype,
            hidden=hidden,
            topk=topk,
            max_slot_num_token=int(max_slot_num_token),
            experts_per_rank=int(experts_per_rank),
            has_weight=has_weight,
        )
        launch_weight_ptrs = combine_weight_ptrs
        launch_output_weight = output_weight
        if not has_weight:
            launch_weight_ptrs = torch.empty((self.local_world_size, ), dtype=torch.int64, device=input_buf.device)
            launch_output_weight = torch.empty((1, topk), dtype=torch.float32, device=input_buf.device)

        partial(
            input_buf,
            combine_x_ptrs,
            launch_weight_ptrs,
            int(rdma_rail_send_win_handle),
            num_tokens_per_rank,
            local_topk_indices,
            local_topk_send_mask,
            local_token_dst_scatter_indices,
        )
        exchange(int(dev_comm_ptr), input_buf, int(rdma_rail_send_win_handle), num_tokens_per_rank)
        reduce(
            output,
            launch_output_weight,
            int(rdma_rail_send_win_handle),
            int(output.shape[0]),
        )
        _ = rdma_rail_send_buf
        _ = num_sm
        return output

    def run_reduce_only(
        self,
        *,
        output: torch.Tensor,
        output_weight: Optional[torch.Tensor],
        rdma_rail_send_buf: torch.Tensor,
        rdma_rail_send_win_handle: int,
        max_slot_num_token: int,
        experts_per_rank: int,
        topk: int,
    ) -> torch.Tensor:
        """Reduce already-populated inter-node combine slots.

        This is used by the group-gemm-combine perf ``reduce_only``
        baseline. A preceding combine or fused group-gemm-combine call
        must have built and exchanged the RDMA rail partial slots.
        """
        if output.ndim != 2:
            raise ValueError(f"output must be rank-2; got {tuple(output.shape)}")
        if not output.is_cuda or not output.is_contiguous():
            raise ValueError("output must be a contiguous CUDA tensor")
        if not rdma_rail_send_buf.is_cuda or not rdma_rail_send_buf.is_contiguous():
            raise ValueError("rdma_rail_send_buf must be a contiguous CUDA tensor")
        has_weight = output_weight is not None
        if has_weight:
            if output_weight.dtype != torch.float32 or output_weight.shape != (output.shape[0], int(topk)):
                raise ValueError("output_weight must be float32 with shape "
                                 f"({output.shape[0]}, {int(topk)}); got "
                                 f"{output_weight.dtype} {tuple(output_weight.shape)}")
            if not output_weight.is_cuda or not output_weight.is_contiguous():
                raise ValueError("output_weight must be a contiguous CUDA tensor")

        self._ensure_kernel_implemented()
        reduce = self._compile_reduce(
            dtype=output.dtype,
            hidden=int(output.shape[1]),
            topk=int(topk),
            max_slot_num_token=int(max_slot_num_token),
            experts_per_rank=int(experts_per_rank),
            has_weight=has_weight,
        )
        launch_output_weight = output_weight
        if not has_weight:
            launch_output_weight = torch.empty((1, int(topk)), dtype=torch.float32, device=output.device)
        reduce(
            output,
            launch_output_weight,
            int(rdma_rail_send_win_handle),
            int(output.shape[0]),
        )
        _ = rdma_rail_send_buf
        return output
