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
    make_moe_jit_dummies,
    mark_dynamic,
)


class CuTeDSLGroupGemmOp(CuTeDSLEPOverlapOpBase):
    """Per-expert M-contiguous group-GEMM."""

    def _compile(self, dtype: torch.dtype, experts_per_rank: int, n_out: int, hidden_in: int, num_sm: int):
        arch_major, arch_minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        variant_args = (
            dtype,
            int(experts_per_rank),
            int(n_out),
            int(hidden_in),
            int(num_sm),
            int(arch_major),
            int(arch_minor),
        )

        def factory():
            return self._build(dtype, experts_per_rank, n_out, hidden_in, num_sm, arch_major, arch_minor)

        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
        )

    def _build(self, dtype: torch.dtype, experts_per_rank: int, n_out: int, hidden_in: int, num_sm: int,
               arch_major: int, arch_minor: int):
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        self._assert_supported_dtype(dtype, "CuTeDSL group GEMM")

        if arch_major >= 10:
            from ..kernels.m_contiguous_cutedsl_group_gemm import (
                MoeGroupGemmKernelMContig, )

            mma_tiler = (256, 256)
            cluster = (2, 2)
            use_2cta = True
        elif arch_major == 9:
            from ..kernels.m_contiguous_cutedsl_group_gemm_sm90 import (
                MoeGroupGemmKernelMContig, )

            mma_tiler = (128, 256)
            # Keep cluster_tile_M equal to GEMM_CLUSTER_TILE_M (256), which
            # dispatch/test packing uses for per-expert padded prefixes.
            cluster = (2, 1)
            use_2cta = False
        else:
            raise RuntimeError(f"CuTeDSL group GEMM requires Hopper (sm90) or newer, got sm{arch_major}{arch_minor}")

        ab_dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
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
        )
        dummy_signals = torch.empty((experts_per_rank, ), dtype=torch.int32, device="cuda")
        kernel = MoeGroupGemmKernelMContig(cutlass.Float32, use_2cta, mma_tiler, cluster)
        return cute.compile(
            kernel,
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
            options=cute_compile_options(enable_tvm_ffi=True),
        )

    def run(self, *, A_padded: torch.Tensor, B: torch.Tensor, problem_sizes_m: torch.Tensor, output: torch.Tensor,
            expert_signals: torch.Tensor, num_sm: int) -> torch.Tensor:
        experts_per_rank = int(problem_sizes_m.shape[0])
        hidden_in = int(A_padded.shape[1])
        n_out = int(B.shape[0])

        compiled = self._compile(
            dtype=A_padded.dtype,
            experts_per_rank=experts_per_rank,
            n_out=n_out,
            hidden_in=hidden_in,
            num_sm=num_sm,
        )
        # Standalone GEMM compiles with wait_signals=False; signal tensor
        # is in the signature but DCE'd, so we skip the redundant fill.
        compiled(
            A_padded,
            B,
            output,
            problem_sizes_m,
            expert_signals,
        )
        return output
