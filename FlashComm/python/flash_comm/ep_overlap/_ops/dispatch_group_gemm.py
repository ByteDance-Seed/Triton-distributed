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
    GEMM_CLUSTER_TILE_M,
    cute_compile_options,
    make_moe_jit_dummies,
    mark_dynamic,
    resolve_nvcc_opt_level,
)


class CuTeDSLDispatchGroupGemmOp(CuTeDSLEPOverlapOpBase):
    """Fused pull dispatch + FC1 group-GEMM op."""

    def __init__(self, *, rank: int, world_size: int, expert_alignment: int):
        super().__init__(rank=rank, world_size=world_size)
        self.expert_alignment = int(expert_alignment)

    def _compile(self, dtype: torch.dtype, experts_per_rank: int, n_out: int, hidden_in: int, dispatch_num_stages: int,
                 num_sm: int, topk: int, weight_dtype: torch.dtype, has_weight: bool):
        variant_args = (
            dtype,
            int(experts_per_rank),
            int(n_out),
            int(hidden_in),
            int(self.rank),
            int(self.world_size),
            int(self.expert_alignment),
            int(dispatch_num_stages),
            int(num_sm),
            int(topk),
            weight_dtype,
            bool(has_weight),
        )

        def factory():
            return self._build(dtype, experts_per_rank, n_out, hidden_in, dispatch_num_stages, num_sm, topk,
                               weight_dtype, has_weight)

        opt_level = resolve_nvcc_opt_level()
        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
            compile_options=("--enable-tvm-ffi", f"--opt-level {opt_level}"),
        )

    def _build(self, dtype: torch.dtype, experts_per_rank: int, n_out: int, hidden_in: int, dispatch_num_stages: int,
               num_sm: int, topk: int, weight_dtype: torch.dtype, has_weight: bool):
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        from ..kernels.cutedsl_dispatch_group_gemm import (
            CuTeDSLDispatchGroupGemmKernel, )

        self._assert_supported_dtype(dtype, "CuTeDSL dispatch+groupgemm")

        mma_tiler = (256, 256)
        cluster = (2, 2)
        use_2cta = True
        ab_dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
        hardware_info = utils.HardwareInfo()
        max_active_clusters = min(
            hardware_info.get_max_active_clusters(cluster[0] * cluster[1]),
            max(1,
                int(num_sm) // (cluster[0] * cluster[1])),
        )

        # expert_alignment is supplied by the owner and may be the
        # alignment==1 default; GEMM still needs cluster-tile alignment.
        expert_alignment = max(
            int(self.expert_alignment),
            GEMM_CLUSTER_TILE_M,
        )

        # JIT only inspects dtype / rank / mode-1-stride-1 of these tensors;
        # values and dynamic-axis sizes are never read.
        dummy_A, dummy_B, dummy_C, dummy_psm = make_moe_jit_dummies(
            num_experts=experts_per_rank,
            n=n_out,
            hidden_in=hidden_in,
            ab_dtype=cutlass_torch.dtype(ab_dtype),
        )
        dummy_input_ptrs = torch.empty((self.world_size, ), dtype=torch.int64, device="cuda")
        dummy_tsri = torch.empty((1, ), dtype=torch.int64, device="cuda")
        dummy_recv_count = torch.empty((self.world_size, ), dtype=torch.int32, device="cuda")
        dummy_signals = torch.empty((experts_per_rank, ), dtype=torch.int32, device="cuda")
        dummy_counters = torch.empty((experts_per_rank, ), dtype=torch.int32, device="cuda")
        dummy_input_weight_ptrs = torch.empty(
            (self.world_size, ),
            dtype=torch.int64,
            device="cuda",
        )
        dummy_output_weight = torch.empty(
            (1, ),
            dtype=weight_dtype,
            device="cuda",
        )

        opt_level = resolve_nvcc_opt_level()
        kernel = CuTeDSLDispatchGroupGemmKernel(
            cutlass.Float32,
            use_2cta,
            mma_tiler,
            cluster,
            dispatch_num_stages=dispatch_num_stages,
        )
        return cute.compile(
            kernel,
            mark_dynamic(dummy_input_ptrs, enable_tvm_ffi=True),
            mark_dynamic(dummy_tsri, enable_tvm_ffi=True),
            mark_dynamic(dummy_recv_count, enable_tvm_ffi=True),
            mark_dynamic(dummy_A, enable_tvm_ffi=True),
            mark_dynamic(dummy_B, enable_tvm_ffi=True),
            mark_dynamic(dummy_C, enable_tvm_ffi=True),
            int(experts_per_rank),
            int(n_out),
            int(hidden_in),
            mark_dynamic(dummy_psm, assumed_align=16, enable_tvm_ffi=True),
            int(max_active_clusters),
            int(max_active_clusters),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            mark_dynamic(dummy_signals, enable_tvm_ffi=True),
            mark_dynamic(dummy_counters, enable_tvm_ffi=True),
            int(self.rank),
            int(self.world_size),
            int(expert_alignment),
            mark_dynamic(dummy_input_weight_ptrs, enable_tvm_ffi=True),
            mark_dynamic(dummy_output_weight, enable_tvm_ffi=True),
            int(topk),
            int(bool(has_weight)),
            options=cute_compile_options(enable_tvm_ffi=True, opt_level=opt_level),
        )

    def run(self, *, A_padded: torch.Tensor, B: torch.Tensor, token_src_rank_topk_and_indices: torch.Tensor,
            recv_count: torch.Tensor, recv_expert_counts: torch.Tensor, output: torch.Tensor,
            dispatch_input_ptrs: torch.Tensor, expert_signals: torch.Tensor, expert_signal_counters: torch.Tensor,
            dispatch_num_stages: int, num_sm: int, input_weight_ptrs: Optional[torch.Tensor] = None,
            output_weight: Optional[torch.Tensor] = None, topk: int, weight_dtype: torch.dtype) -> None:
        """Launch the compiled dispatch+groupgemm kernel.

        Caller must stage input into ``dispatch_input_buf`` and zero
        ``expert_signals`` / ``expert_signal_counters`` beforehand.

        Pass ``(input_weight_ptrs, output_weight, topk)`` together to
        enable the side-loaded weight dispatch path; ``output_weight``
        must be a 1D weight tensor sized to ``A_padded``.
        Leave them ``None`` for the weightless path (a transient empty
        scratch is fed to satisfy the JIT signature).
        """
        experts_per_rank = int(recv_expert_counts.shape[0])
        hidden_in = int(A_padded.shape[1])
        n_out = int(B.shape[0])

        if B.shape != (n_out, hidden_in, experts_per_rank):
            raise ValueError(f"B must be (N, K={hidden_in}, L={experts_per_rank}); "
                             f"got {tuple(B.shape)}")
        if output.shape != (A_padded.shape[0], n_out):
            raise ValueError(f"output must be (M={A_padded.shape[0]}, N={n_out}); "
                             f"got {tuple(output.shape)}")

        has_weight = output_weight is not None
        if has_weight:
            if input_weight_ptrs is None:
                raise ValueError("input_weight_ptrs must be supplied when "
                                 "output_weight is given")
            if int(topk) <= 0:
                raise ValueError(f"topk must be > 0 when has_weight=True; got {topk}")
            if output_weight.dtype != weight_dtype:
                raise TypeError(f"output_weight dtype must be {weight_dtype}; got {output_weight.dtype}")
            if output_weight.ndim != 1:
                raise ValueError("output_weight must be 1D; got shape "
                                 f"{tuple(output_weight.shape)}")
            if output_weight.shape[0] != A_padded.shape[0]:
                raise ValueError("output_weight first dim must equal A_padded.shape[0]"
                                 f" ({A_padded.shape[0]}); got {output_weight.shape[0]}")
            if input_weight_ptrs.dtype != torch.int64:
                raise TypeError("input_weight_ptrs must be int64; got "
                                f"{input_weight_ptrs.dtype}")
            if input_weight_ptrs.shape != (self.world_size, ):
                raise ValueError("input_weight_ptrs must be shape (world_size,)="
                                 f"({self.world_size},); got "
                                 f"{tuple(input_weight_ptrs.shape)}")
        compiled = self._compile(
            dtype=A_padded.dtype,
            experts_per_rank=experts_per_rank,
            n_out=n_out,
            hidden_in=hidden_in,
            dispatch_num_stages=dispatch_num_stages,
            num_sm=num_sm,
            topk=int(topk),
            weight_dtype=weight_dtype,
            has_weight=has_weight,
        )

        if has_weight:
            launch_input_weight_ptrs = input_weight_ptrs
            launch_output_weight = output_weight
        else:
            # ``has_weight=0`` constexpr DCEs every read of these args,
            # but the JIT signature still needs a tensor of the right
            # shape/dtype for ``mark_dynamic`` to extract its descriptor.
            launch_input_weight_ptrs = torch.empty(
                (self.world_size, ),
                dtype=torch.int64,
                device=A_padded.device,
            )
            launch_output_weight = torch.empty(
                (A_padded.shape[0], ),
                dtype=weight_dtype,
                device=A_padded.device,
            )

        compiled(
            dispatch_input_ptrs,
            token_src_rank_topk_and_indices,
            recv_count,
            A_padded,
            B,
            output,
            recv_expert_counts,
            expert_signals,
            expert_signal_counters,
            launch_input_weight_ptrs,
            launch_output_weight,
        )
