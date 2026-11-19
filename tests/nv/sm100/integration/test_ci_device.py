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
"""Compile fresh SM100 CUDA and check tail handling on each assigned device."""
import os

import pytest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES

pytestmark = [pytest.mark.gpu, pytest.mark.sm100]


@lk.ll_kernel(backend='cuda', is_entry=True)
def ci_blackwell_vector_add(a: ll.Tensor[ll.float32], b: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32],
                            n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        out[i] = a[i] + b[i]


@pytest.fixture(scope='module', params=range(int(os.environ.get('LK_CI_DEVICE_COUNT', '1'))))
def blackwell(request):
    torch = pytest.importorskip('torch')
    assert torch.cuda.is_available(), 'SM100 CI requires a usable CUDA device'
    count = int(os.environ.get('LK_CI_DEVICE_COUNT', '1'))
    assert 1 <= count <= 8 and torch.cuda.device_count() >= count
    assert torch.cuda.get_device_capability(request.param) == (10, 0), 'SM100 CI received the wrong GPU'
    with torch.cuda.device(request.param):
        yield torch


@pytest.fixture(scope='module')
def generated_kernel(blackwell, tmp_path_factory):
    return ci_blackwell_vector_add.build(PASSES['cuda'], codegen_cuda, grid=(1, 1, 1), block=(256, 1, 1),
                                         arch='sm_100a',
                                         cache_dir=str(tmp_path_factory.mktemp('sm100-generated-kernel')))


@pytest.mark.parametrize('n', [1, 31, 32, 257, 4097])
def test_generated_vector_add_on_blackwell(blackwell, generated_kernel, n):
    torch = blackwell
    a = torch.arange(n, device='cuda', dtype=torch.float32) * .25
    b = torch.arange(n, device='cuda', dtype=torch.float32) * -.125
    out = torch.full_like(a, -777)
    generated_kernel(a, b, out, n, grid=((n + 255) // 256, 1, 1))
    torch.cuda.synchronize()
    torch.testing.assert_close(out, a + b, rtol=0, atol=0)
