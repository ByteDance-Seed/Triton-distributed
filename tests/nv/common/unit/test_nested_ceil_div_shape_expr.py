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
"""Regression coverage for semantic key ir.shape_expr.tensor_numel_floor_div.

The export-manifest shape/launch expression language must accept ceil_div
over a tensor extent divided by a constant in either argument order,
including nested ceil_div calls such as the grid form
``ceil_div(ceil_div(x[0], 8), 256)`` (exactly ceil(N / 8 / 256) blocks).

``_shape_expr`` lowers expressions via a single recursive ``emit`` walk, so
a Call's arguments are arbitrary sub-expressions; these tests pin that
contract at the parser, spec-load and binding-render layers (CPU only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_spec, _render_binding, _shape_expr


def test_shape_expr_nested_ceil_div_recurses_in_either_argument():
    ranks = {"x": 1}
    assert (_shape_expr("ceil_div(ceil_div(x[0], 8), 256)", ranks) == "lk_ceil_div(lk_ceil_div(x.size(0), 8), 256)")
    # Flattened equivalent spelling (8 * 256 == 2048).
    assert (_shape_expr("ceil_div(x[0], 2048)", ranks) == "lk_ceil_div(x.size(0), 2048)")
    # Tensor extent need not be the first argument.
    assert (_shape_expr("ceil_div(8, ceil_div(x[0], 256))", ranks) == "lk_ceil_div(8, lk_ceil_div(x.size(0), 256))")


def test_shape_expr_nested_ceil_div_mixed_with_arithmetic():
    ranks = {"x": 1}
    assert (_shape_expr("ceil_div(ceil_div(x[0], 8) + 1, 256)",
                        ranks) == "lk_ceil_div((lk_ceil_div(x.size(0), 8) + 1), 256)")


@pytest.mark.parametrize("bad", [
    "ceil_div(ceil_div(x[0], 8))",  # arity still enforced when nested
    "ceil_div(ceil_div(x[0], 8), 256, 1)", "ceil_div(ceil_div(x[0], -8), 256)",  # negative constants stay rejected
])
def test_shape_expr_nested_ceil_div_still_validated(bad):
    with pytest.raises(ValueError):
        _shape_expr(bad, {"x": 1})


def _write_spec(tmp_path: Path, grid: list) -> Path:
    spec = {
        "kernel": "probe",
        "function": "probe",
        "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
        "output": {"name": "dst", "dtype": "float32", "shape": ["x[0]"]},
        "launch": {"grid": grid, "block": [256, 1, 1]},
        "scalars": [],
    }
    path = tmp_path / "export.json"
    path.write_text(json.dumps(spec))
    return path


def test_nested_ceil_div_grid_loads_and_renders(tmp_path):
    path = _write_spec(tmp_path, ["ceil_div(ceil_div(x[0], 8), 256)", 1, 1])
    loaded = _load_spec(path)
    assert loaded["launch"]["_grid_cpp"] == ["lk_ceil_div(lk_ceil_div(x.size(0), 8), 256)", "1", "1"]
    binding = _render_binding(loaded)
    assert "lk_ceil_div(lk_ceil_div(x.size(0), 8), 256)" in binding


def test_flat_ceil_div_grid_equivalent_spelling(tmp_path):
    path = _write_spec(tmp_path, ["ceil_div(x[0], 2048)", 1, 1])
    loaded = _load_spec(path)
    assert loaded["launch"]["_grid_cpp"] == ["lk_ceil_div(x.size(0), 2048)", "1", "1"]
