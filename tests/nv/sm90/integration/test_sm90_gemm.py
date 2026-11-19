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
"""Integration tests for SM90 (Hopper) GEMM kernels built with LittleKernel.

Validates correctness of the SM90 GEMM kernel implementations
(gemm_v1 through gemm_v10) by building each kernel and running
a small matrix multiplication against a PyTorch reference.

Requires a Hopper-class (SM90+) GPU.
"""

import importlib

import pytest

from tests.nv.conftest import requires_sm90

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VARIANTS = (
    "sm90_manual_wgmma_swizzle",
    "sm90_tma_wgmma_single_stage",
    "sm90_warp_specialized_multistage",
    "sm90_large_tile_128x128",
    "sm90_cluster_multicast",
    "sm90_persistent_m64n256",
    "sm90_l2_swizzle_bidirectional_multicast",
    "sm90_tma_store_bf16_output",
    "sm90_async_warpgroup_scheduling",
    "sm90_deepgemm_full_optimizations",
)

# Aligned tiles are the shared contract of these benchmark implementations.
# Short K, pipeline wraparound, rectangular grids, and >132 persistent tiles.
SHAPES = ((256, 256, 128), (512, 768, 512), (4096, 2304, 192))


@pytest.fixture(scope="module", params=VARIANTS)
def gemm(request):
    module = importlib.import_module("benchmarks.nv.sm90.gemm." + request.param)
    return module, module.build_kernel()


# ---------------------------------------------------------------------------
# Tests -- one per SM90 GEMM variant
# ---------------------------------------------------------------------------


@requires_sm90
@pytest.mark.gpu
@pytest.mark.sm90
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_gemm(gemm, shape):
    """Every output element must match an independent FP32 Torch matmul."""
    import torch

    module, kernel = gemm
    m, n, k = shape
    torch.manual_seed(2026)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference = a.float() @ b.float()
        output = module.run_gemm(kernel, a, b, m, n, k)
        torch.cuda.synchronize()
        assert output.shape == (m, n)
        assert torch.isfinite(output).all()
        # BF16 epilogues round once; FP32 epilogues retain their accumulators.
        bf16 = output.dtype == torch.bfloat16
        torch.testing.assert_close(output, reference.to(output.dtype), rtol=1e-2 if bf16 else 2e-4,
                                   atol=2e-2 if bf16 else 1e-3)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
