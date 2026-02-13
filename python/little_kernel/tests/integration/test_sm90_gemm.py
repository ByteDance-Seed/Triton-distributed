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

from little_kernel.tests.conftest import requires_sm90

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _correctness_check(module, M=512, N=512, K=512, atol=1e-1, rtol=1e-1):
    """Build a kernel from the given module and verify correctness."""
    kernel = module.build_kernel()
    ok = module.test(kernel, M, N, K)
    assert ok, f"{module.__name__} correctness check failed at ({M}, {N}, {K})"


# ---------------------------------------------------------------------------
# Tests -- one per SM90 GEMM variant
# ---------------------------------------------------------------------------


@requires_sm90
class TestSM90GEMM:

    def test_gemm_v1(self):
        from little_kernel.benchmark.gemm_sm90 import gemm_v1
        _correctness_check(gemm_v1)

    def test_gemm_v2(self):
        from little_kernel.benchmark.gemm_sm90 import gemm_v2
        _correctness_check(gemm_v2)

    def test_gemm_v3(self):
        from little_kernel.benchmark.gemm_sm90 import gemm_v3
        _correctness_check(gemm_v3)

    def test_gemm_v4(self):
        from little_kernel.benchmark.gemm_sm90 import gemm_v4
        _correctness_check(gemm_v4)

    def test_gemm_v5(self):
        from little_kernel.benchmark.gemm_sm90 import gemm_v5
        _correctness_check(gemm_v5)

    def test_gemm_v6(self):
        from little_kernel.benchmark.gemm_sm90 import gemm_v6
        _correctness_check(gemm_v6)

    def test_gemm_v7(self):
        from little_kernel.benchmark.gemm_sm90 import gemm_v7
        _correctness_check(gemm_v7)

    def test_gemm_v8(self):
        from little_kernel.benchmark.gemm_sm90 import gemm_v8
        _correctness_check(gemm_v8)

    def test_gemm_v9(self):
        from little_kernel.benchmark.gemm_sm90 import gemm_v9
        _correctness_check(gemm_v9)

    def test_gemm_v10(self):
        from little_kernel.benchmark.gemm_sm90 import gemm_v10
        _correctness_check(gemm_v10)
