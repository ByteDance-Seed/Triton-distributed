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
"""Repeated persistent tiles must not overwrite a pending TMA store's buffer."""
import pytest

from tests.nv.conftest import requires_sm90


@pytest.fixture(scope='module')
def asynchronous_gemm():
    from benchmarks.nv.sm90.gemm import sm90_async_warpgroup_scheduling as gemm
    return gemm, gemm.build_kernel()


@requires_sm90
@pytest.mark.gpu
@pytest.mark.sm90
@pytest.mark.parametrize('shape', [(4096, 2304, 64), (4096, 2304, 192), (8192, 2304, 192)])
def test_persistent_output_buffer_reuse(asynchronous_gemm, shape):
    import torch

    gemm, kernel = asynchronous_gemm
    m, n, k = shape
    torch.manual_seed(2026)
    a = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(k, n, dtype=torch.bfloat16, device='cuda')
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference = (a.float() @ b.float()).to(torch.bfloat16)
        # More tiles than persistent clusters forces the SAME shared output
        # allocation to be reused. Repeated launches expose timing-sensitive
        # overwrites; checking only a mean/cosine would miss a few corrupt rows.
        assert ((m + gemm.BM * gemm.CLUSTER_SIZE - 1) // (gemm.BM * gemm.CLUSTER_SIZE) *
                ((n + gemm.BN - 1) // gemm.BN)) > 132
        for iteration in range(32):
            output = gemm.run_gemm(kernel, a, b, m, n, k)
            torch.cuda.synchronize()
            assert output.shape == (m, n)
            assert torch.isfinite(output).all(), f'nonfinite output at repetition {iteration}'
            torch.testing.assert_close(output, reference, rtol=1e-2, atol=2e-2,
                                       msg=lambda message: f'repetition {iteration}, shape {shape}: {message}')
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
