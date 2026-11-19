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
"""Regression checks for standalone benchmark validation and acceptance reporting."""

import pytest

from tests.nv.conftest import cuda_available
from benchmarks.nv.sm100.dataflow.bench_optional import optional_import
from tests.nv.sm100.dataflow.test_generated_kernels_sm100 import check_performance


def _require_torch():
    return pytest.importorskip("torch")


def test_optional_import_modes(monkeypatch, tmp_path):
    (tmp_path / "lk_test_reference.py").write_text("VALUE = 42\n")
    monkeypatch.setenv("LK_TEST_REFERENCE_PATH", str(tmp_path))
    assert optional_import("lk_test_reference", "LK_TEST_REFERENCE_PATH", mode="none") is None
    assert optional_import("lk_test_reference", "LK_TEST_REFERENCE_PATH", mode="required").VALUE == 42
    assert optional_import("lk_missing_reference", "LK_TEST_REFERENCE_PATH") is None
    with pytest.raises(RuntimeError, match="LK_TEST_REFERENCE_PATH"):
        optional_import("lk_missing_reference", "LK_TEST_REFERENCE_PATH", mode="required")


def test_gdn_chunk_reference_matches_token_recurrence():
    torch = _require_torch()
    from benchmarks.nv.sm100.dataflow.bench_gated_delta_net import torch_reference

    torch.manual_seed(7)
    lengths, heads, dim = (3, 67), 2, 8
    total = sum(lengths)
    q, k, v = [torch.randn(1, total, heads, dim) for _ in range(3)]
    k = torch.nn.functional.normalize(k, dim=-1)
    g = -torch.rand(1, total, heads)
    beta = torch.rand(1, total, heads)
    h0 = torch.randn(len(lengths), heads, dim, dim)
    cu = torch.tensor([0, 3, total])
    actual, final = torch_reference(q, k, v, g, beta, h0, cu)
    expected = torch.empty_like(v)
    for seq, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
        state = h0[seq].clone()
        for token in range(start, end):
            key = k[0, token]
            state *= g[0, token].exp()[:, None, None]
            residual = (v[0, token] - (state * key[:, None, :]).sum(-1)) * beta[0, token, :, None]
            state += residual[:, :, None] * key[:, None, :]
            expected[0, token] = (state * q[0, token, :, None, :]).sum(-1) / dim**0.5
        torch.testing.assert_close(final[seq], state, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_performance_gate_uses_tracked_shape_and_rejects_regression():
    assert check_performance("gemm", "role_v2: 9 ms\nrole_v2: 0.73 ms", 0.1)["passed"]
    assert not check_performance("gemm", "role_v2: 0.90 ms", 0.1)["passed"]
    with pytest.raises(ValueError, match="missing tracked"):
        check_performance("gemm", "correctness: pass", 0.1)


def test_gdn_inverse_and_state_decay_with_correlated_keys():
    if not cuda_available():
        pytest.skip("requires an SM100-class CUDA GPU")
    torch = _require_torch()
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100")
    from benchmarks.nv.sm100.dataflow.bench_gated_delta_net import Shape, _inputs
    from examples.nv.sm100.gated_delta_net import GDNVarlenV2

    inputs = _inputs(Shape("correlated_keys", (64, )))
    q, k, v, g, beta, h0, cu = inputs
    # Correlated keys expose the off-diagonal terms hidden by random unit keys.
    k.copy_(k[:, :1].expand_as(k))
    g.fill_(-0.03)
    beta.fill_(0.7)
    kernel = GDNVarlenV2.build()
    target = GDNVarlenV2.create(
        kernel,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        cu_seqlens=cu,
    )
    out, state = target.launch()
    ref_out, ref_state = torch_reference(*inputs)  # noqa: F821
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state, ref_state, rtol=2e-2, atol=3e-2)
