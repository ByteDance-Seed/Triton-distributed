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
"""Integration tests for the HIP backend: build with hipcc, launch through the HIP module API.

Requires an AMD GPU with ROCm (the MXFP8 GEMM: gfx950).
"""

import pytest
import torch

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.amd.cdna4.codegen.emitter import codegen_hip
from little_kernel.core.passes import PASSES
from tests.amd.cdna4.conftest import requires_gfx950, requires_hip


@lk.ll_kernel(backend="hip", is_entry=True)
def axpby(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32], n: ll.int32,
          alpha: ll.float32, beta: ll.float32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        out[i] = alpha * x[i] + beta * y[i]


@requires_hip
def test_axpby():
    n = 100003
    kernel = axpby.build(PASSES["hip"], codegen_hip, grid=((n + 255) // 256, 1, 1), block=(256, 1, 1))
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    kernel(x, y, out, n, 2.0, -0.5)
    kernel.synchronize()
    torch.testing.assert_close(out, 2.0 * x - 0.5 * y)


@requires_gfx950
class TestMxfp8GemmGfx950:

    @pytest.fixture(scope="class")
    def kernel(self):
        from benchmarks.amd.cdna4 import mxfp8_gemm_gfx950
        return mxfp8_gemm_gfx950.build_kernel()

    @pytest.mark.parametrize("m, n, k", [(256, 256, 128), (512, 768, 1024), (1000, 1100, 2048)])
    def test_correctness(self, kernel, m, n, k):
        from benchmarks.amd.cdna4 import mxfp8_gemm_gfx950
        assert mxfp8_gemm_gfx950.check(kernel, m, n, k)


@requires_gfx950
class TestMxfp8GemmGfx950Persistent:

    @pytest.fixture(scope="class")
    def kernel(self):
        from benchmarks.amd.cdna4 import mxfp8_gemm_gfx950_persistent
        return mxfp8_gemm_gfx950_persistent.build_kernel()

    # partial tiles; a column tail (ldd > n); a partial last round of tiles (289 tiles)
    @pytest.mark.parametrize("m, n, k, ldd", [(256, 256, 256, None), (1000, 1096, 2048, None), (1000, 1100, 2048, 1104),
                                              (4352, 4352, 512, None)])
    def test_correctness(self, kernel, m, n, k, ldd):
        from benchmarks.amd.cdna4 import mxfp8_gemm_gfx950, mxfp8_gemm_gfx950_persistent
        assert mxfp8_gemm_gfx950.check(kernel, m, n, k, launch=mxfp8_gemm_gfx950_persistent.run, ldd=ldd)

    def test_bitwise_equal_to_data_parallel(self, kernel):
        from benchmarks.amd.cdna4 import mxfp8_gemm_gfx950, mxfp8_gemm_gfx950_persistent
        reference = mxfp8_gemm_gfx950.build_kernel()
        assert mxfp8_gemm_gfx950_persistent.same_as_data_parallel(kernel, reference, 4352, 4352, 1024)
