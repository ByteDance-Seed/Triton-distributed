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
    make_moe_jit_dummies,
    mark_dynamic,
    resolve_nvcc_opt_level,
)


class CuTeDSLGroupGemmCombineOp(CuTeDSLEPOverlapOpBase):
    """Fused FC2 group-GEMM + push-mode combine (no local reduce)."""

    def _compile(self, dtype: torch.dtype, experts_per_rank: int, n_out: int, hidden_in: int, topk: int, num_sm: int,
                 weight_dtype: torch.dtype, has_weight: bool):
        variant_args = (
            dtype,
            int(experts_per_rank),
            int(n_out),
            int(hidden_in),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(num_sm),
            weight_dtype,
            bool(has_weight),
        )

        def factory():
            return self._build(dtype, experts_per_rank, n_out, hidden_in, topk, num_sm, weight_dtype, has_weight)

        opt_level = resolve_nvcc_opt_level()
        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
            compile_options=("--enable-tvm-ffi", f"--opt-level {opt_level}"),
        )

    def _build(self, dtype: torch.dtype, experts_per_rank: int, n_out: int, hidden_in: int, topk: int, num_sm: int,
               weight_dtype: torch.dtype, has_weight: bool):
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        from ..kernels.cutedsl_group_gemm_combine import MegaMoEGroupGEMMCombine

        self._assert_supported_dtype(dtype, "MegaMoEGroupGEMMCombine")

        mma_tiler = (256, 256)
        cluster = (2, 2)
        use_2cta = True
        ab_dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
        c_dtype = ab_dtype

        hardware_info = utils.HardwareInfo()
        max_active_clusters = min(
            hardware_info.get_max_active_clusters(cluster[0] * cluster[1]),
            max(1,
                int(num_sm) // (cluster[0] * cluster[1])),
        )

        # JIT only inspects dtype / rank / mode-1-stride-1 of these tensors;
        # values and dynamic-axis sizes are never read.
        dummy_A, dummy_B, dummy_C, dummy_psm = make_moe_jit_dummies(
            num_experts=experts_per_rank,
            n=n_out,
            hidden_in=hidden_in,
            ab_dtype=cutlass_torch.dtype(ab_dtype),
            c_dtype=cutlass_torch.dtype(c_dtype),
        )
        dummy_output_ptrs = torch.empty(self.world_size, dtype=torch.int64, device="cuda")
        dummy_tsri = torch.empty((1, ), dtype=torch.int64, device="cuda")
        dummy_dispatched_weights = torch.empty(
            (1, ),
            dtype=weight_dtype,
            device="cuda",
        )
        dummy_weight_output_ptrs = torch.empty(
            self.world_size,
            dtype=torch.int64,
            device="cuda",
        )

        opt_level = resolve_nvcc_opt_level()
        kernel = MegaMoEGroupGEMMCombine(
            cutlass.Float32,
            use_2cta,
            mma_tiler,
            cluster,
        )
        return cute.compile(
            kernel,
            mark_dynamic(dummy_A, enable_tvm_ffi=True),
            mark_dynamic(dummy_B, enable_tvm_ffi=True),
            mark_dynamic(dummy_C, enable_tvm_ffi=True),
            mark_dynamic(dummy_output_ptrs, enable_tvm_ffi=True),
            mark_dynamic(dummy_tsri, enable_tvm_ffi=True),
            mark_dynamic(dummy_dispatched_weights, enable_tvm_ffi=True),
            mark_dynamic(dummy_weight_output_ptrs, enable_tvm_ffi=True),
            int(experts_per_rank),
            int(n_out),
            int(hidden_in),
            int(topk),
            int(self.rank),
            int(self.world_size),
            mark_dynamic(dummy_psm, assumed_align=16, enable_tvm_ffi=True),
            int(max_active_clusters),
            int(max_active_clusters),
            int(1 if has_weight else 0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=cute_compile_options(enable_tvm_ffi=True, opt_level=opt_level),
        )

    def run(self, *, A_padded: torch.Tensor, B: torch.Tensor, token_src_rank_topk_and_indices: torch.Tensor,
            recv_expert_counts: torch.Tensor, output_ptrs: torch.Tensor, num_sm: int, topk: int,
            weight_dtype: torch.dtype, dispatched_weights: Optional[torch.Tensor] = None,
            weight_output_ptrs: Optional[torch.Tensor] = None) -> None:
        """Launch the fused groupgemm+combine kernel.

        Note: we do NOT zero the staging buffer between calls. The
        topk-reduce kernel skips slots whose ``topk_indices`` carry the
        drop sentinel, so stale bytes never reach the accumulator (saves
        a full-bandwidth fill per call).

        Optional ``dispatched_weights`` (1D ``weight_dtype`` sized
        to ``A_padded.shape[0]``) and ``weight_output_ptrs`` enable the
        side-loaded weight push: the epilogue writes one scalar per
        row to the source rank's symmetric ``(max_m, topk)`` weight
        buffer. Both must be supplied together or both omitted.
        """
        experts_per_rank = int(recv_expert_counts.shape[0])
        hidden_in = int(A_padded.shape[1])
        n_out = int(B.shape[0])

        if B.shape != (n_out, hidden_in, experts_per_rank):
            raise ValueError(f"B must be (N, K={hidden_in}, L={experts_per_rank}); "
                             f"got {tuple(B.shape)}")

        has_weight = dispatched_weights is not None
        if has_weight != (weight_output_ptrs is not None):
            raise ValueError("dispatched_weights and weight_output_ptrs must be "
                             "supplied together")
        if has_weight:
            if dispatched_weights.dtype != weight_dtype:
                raise TypeError(f"dispatched_weights dtype must be {weight_dtype}; got {dispatched_weights.dtype}")
            if dispatched_weights.ndim != 1:
                raise ValueError("dispatched_weights must be 1D; got shape "
                                 f"{tuple(dispatched_weights.shape)}")
            if dispatched_weights.shape[0] != A_padded.shape[0]:
                raise ValueError("dispatched_weights.shape[0] != A_padded.shape[0]: "
                                 f"{dispatched_weights.shape[0]} vs {A_padded.shape[0]}")
            if not dispatched_weights.is_contiguous():
                raise ValueError("dispatched_weights must be contiguous")
            if weight_output_ptrs.dtype != torch.int64:
                raise TypeError("weight_output_ptrs must be int64; got "
                                f"{weight_output_ptrs.dtype}")

        compiled = self._compile(
            dtype=A_padded.dtype,
            experts_per_rank=experts_per_rank,
            n_out=n_out,
            hidden_in=hidden_in,
            topk=topk,
            num_sm=num_sm,
            weight_dtype=weight_dtype,
            has_weight=has_weight,
        )

        # ``c_scratch`` is unused (epilogue pushes go to peer GMEM) but
        # the kernel signature still requires a local C for layout
        # consistency with the standalone GEMM.
        c_scratch = torch.empty((A_padded.shape[0], n_out), dtype=A_padded.dtype, device=A_padded.device)

        if has_weight:
            launch_dispatched_weights = dispatched_weights
            launch_weight_output_ptrs = weight_output_ptrs
        else:
            # ``has_weight=0`` DCEs every read of these args; just give
            # the JIT a tensor of the right shape/dtype.
            launch_dispatched_weights = torch.empty(
                (A_padded.shape[0], ),
                dtype=weight_dtype,
                device=A_padded.device,
            )
            launch_weight_output_ptrs = output_ptrs

        compiled(
            A_padded,
            B,
            c_scratch,
            output_ptrs,
            token_src_rank_topk_and_indices,
            launch_dispatched_weights,
            launch_weight_output_ptrs,
            recv_expert_counts,
        )
