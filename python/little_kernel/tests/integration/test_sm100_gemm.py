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
"""Integration tests for SM100 (Blackwell) GEMM kernels built with LittleKernel.

Validates correctness of the SM100 GEMM kernel implementations
(level1 through level9) by building each kernel and running
a small matrix multiplication against a PyTorch reference.

Requires a Blackwell-class (SM100+) GPU.
"""

from little_kernel.tests.conftest import requires_sm100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _correctness_check(module, M=512, N=512, K=512):
    """Build a kernel from the given module and verify correctness."""
    kernel = module.build_kernel()
    ok = module.test(kernel, M, N, K)
    assert ok, f"{module.__name__} correctness check failed at ({M}, {N}, {K})"


# ---------------------------------------------------------------------------
# Tests -- one per SM100 GEMM level
# ---------------------------------------------------------------------------


@requires_sm100
class TestSM100GEMM:

    def test_gemm_level1(self):
        from little_kernel.benchmark.gemm_sm100 import gemm_level1
        _correctness_check(gemm_level1)

    def test_gemm_level2(self):
        from little_kernel.benchmark.gemm_sm100 import gemm_level2
        _correctness_check(gemm_level2)

    def test_gemm_level3(self):
        from little_kernel.benchmark.gemm_sm100 import gemm_level3
        _correctness_check(gemm_level3)

    def test_gemm_level4(self):
        from little_kernel.benchmark.gemm_sm100 import gemm_level4
        _correctness_check(gemm_level4)

    def test_gemm_level5(self):
        from little_kernel.benchmark.gemm_sm100 import gemm_level5
        _correctness_check(gemm_level5)

    def test_gemm_level6(self):
        from little_kernel.benchmark.gemm_sm100 import gemm_level6
        _correctness_check(gemm_level6)

    def test_gemm_level7(self):
        from little_kernel.benchmark.gemm_sm100 import gemm_level7
        _correctness_check(gemm_level7)

    def test_gemm_level8(self):
        from little_kernel.benchmark.gemm_sm100 import gemm_level8
        _correctness_check(gemm_level8)

    def test_gemm_level9(self):
        from little_kernel.benchmark.gemm_sm100 import gemm_level9
        _correctness_check(gemm_level9)
