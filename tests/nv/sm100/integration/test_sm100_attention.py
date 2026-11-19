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
"""Execute the SM100 dataflow attention program against independent Torch SDPA."""
import pytest

from tests.nv.conftest import requires_sm100

pytestmark = [pytest.mark.gpu, pytest.mark.sm100, requires_sm100]


@pytest.fixture(scope='module')
def attention_kernel(tmp_path_factory):
    from benchmarks.nv.sm100.dataflow.bench_fa4_attention_sm100 import build_kernel
    return build_kernel(cache_dir=str(tmp_path_factory.mktemp('sm100-attention')))


@pytest.mark.parametrize('batch,seqlen', [(1, 128), (1, 257), (2, 527)])
def test_causal_attention_matches_torch(attention_kernel, batch, seqlen, monkeypatch):
    import torch
    from benchmarks.nv.sm100.dataflow.bench_fa4_attention_sm100 import check_correctness

    # Exercise real CUDA computation with poisoned allocator contents. Padding
    # must be initialized even when its attention probabilities are masked out.
    allocate = torch.empty

    def poisoned_empty(*args, **kwargs):
        value = allocate(*args, **kwargs)
        if (value.is_cuda and value.ndim == 4 and value.shape[0] == batch and value.shape[1] > seqlen):
            value.fill_(float('nan'))
        return value

    monkeypatch.setattr(torch, 'empty', poisoned_empty)
    check_correctness(attention_kernel, (batch, seqlen))
