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
"""Execute Hopper varlen FlashAttention against a pure Torch math reference.

The maintained program is causal BF16 with head dimension 128. No external
FlashAttention package or prebuilt reference CUDA library is used by this gate.
"""
import math

import pytest

from tests.nv.conftest import requires_sm90

pytestmark = [requires_sm90, pytest.mark.gpu, pytest.mark.sm90]

CASES = (((128, ), 4, 4),  # MHA, one Q/K tile
         ((256, ), 4, 4),  # MHA, multiple causal tiles
         ((512, 512), 8, 2),  # batched GQA, pipeline wraparound
         ((128, 256, 384), 4, 1),  # ragged MQA
         ((129, 257), 4, 1),  # non-tile-aligned tail masking
         ((256, 384), 6, 2),  # non-power-of-two GQA divisor
         )


@pytest.fixture(scope="module")
def attention(tmp_path_factory):
    import torch
    from benchmarks.nv.sm90.flashattention.level6 import flash_attn_varlen as program

    kernel = program.build_kernel(
        num_sm=torch.cuda.get_device_properties(0).multi_processor_count,
        cache_dir=str(tmp_path_factory.mktemp("flashattention")),
    )
    return program, kernel


@pytest.mark.parametrize("lengths,heads,kv_heads", CASES)
def test_flashattention(attention, lengths, heads, kv_heads):
    import torch

    program, kernel = attention
    torch.manual_seed(2026)
    total = sum(lengths)
    q = torch.randn(total, heads, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total, kv_heads, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    cu = torch.tensor([0, *lengths], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)

    # Materialize scores/softmax in FP32, independently of the DSL's scheduling,
    # WGMMA accumulator layout, TMA descriptors and persistent work queue.
    references = []
    start = 0
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for length in lengths:
            qi = q[start:start + length].float().transpose(0, 1)
            ki = k[start:start + length].float().transpose(0, 1).repeat_interleave(heads // kv_heads, 0)
            vi = v[start:start + length].float().transpose(0, 1).repeat_interleave(heads // kv_heads, 0)
            scores = (qi @ ki.transpose(-1, -2)) / math.sqrt(128)
            mask = torch.ones(length, length, dtype=torch.bool, device=q.device).triu(1)
            scores.masked_fill_(mask, -float("inf"))
            references.append((scores.softmax(-1) @ vi).transpose(0, 1))
            start += length
        reference = torch.cat(references).to(q.dtype)
        output = program.run_dsl_kernel(kernel, q, k, v, cu, cu, len(lengths), max(lengths), max(lengths))
        torch.cuda.synchronize()
        assert output.shape == reference.shape and output.dtype == reference.dtype
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=5e-3)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
