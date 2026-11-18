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
"""Numeric clamp literals in launch.grid shape expressions.

A frontend expresses a capped grid-stride launch, e.g. the teacher's
``min(ceil_div(B * N, 256), 2112)`` where 2112 is a plain integer cap, by
mixing tensor-dimension/scalar terms with bare integer literals inside
``min``/``max``/``ceil_div``. The shape-expression language must accept that
mixture (validation in ``_shape_expr``), and the host binding must evaluate
it with the int64 helpers already used for block caps: the clamped value is
range-checked against the CUDA grid limit before the ``unsigned`` launch.

These are validation/render-level tests (no GPU, no nvcc).
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations import vibe_cuda_eval


def _spec(grid):
    return {
        "kernel": "k",
        "inputs": [{"name": "x", "dtype": "float32", "rank": 2}],
        "outputs": [{"name": "y", "dtype": "float32", "shape": ["x[0] * x[1]"]}],
        "scalars": [],
        "workspaces": [],
        "launch": {"grid": grid, "block": [256, 1, 1]},
    }


def _constructor_spec(grid):
    # Mirrors task t-6febf6b9: batch/N are integer constructor scalars and the
    # teacher caps the flat grid at 2112 blocks (AGENT_MAX_BLOCKS).
    return {
        "kernel": "k",
        "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
        "outputs": [{"name": "y", "dtype": "float32", "shape": ["x[0]"]}],
        "constructor": [{"name": "B", "dtype": "int64"}, {"name": "N", "dtype": "int64"}],
        "scalars": [],
        "workspaces": [],
        "launch": {"grid": grid, "block": [256, 1, 1]},
    }


def _load(spec):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
        path = Path(f.name)
    try:
        return vibe_cuda_eval._load_spec(path)
    finally:
        path.unlink(missing_ok=True)


class LaunchGridNumericClampTests(unittest.TestCase):

    def test_shape_expr_folds_literal_inside_clamp(self):
        ranks = {"x": 2}
        scalars = frozenset({"B", "N"})
        self.assertEqual(vibe_cuda_eval._shape_expr("min(ceil_div(x[0] * x[1], 256), 2112)", ranks, scalars),
                         "lk_min(lk_ceil_div((x.size(0) * x.size(1)), 256), 2112)")
        self.assertEqual(vibe_cuda_eval._shape_expr("max(ceil_div(B * N, 256), 2112)", ranks, scalars),
                         "lk_max(lk_ceil_div((B * N), 256), 2112)")
        # The literal may sit on either side and nest inside ceil_div.
        self.assertEqual(vibe_cuda_eval._shape_expr("min(2112, ceil_div(x[0], 256))", ranks),
                         "lk_min(2112, lk_ceil_div(x.size(0), 256))")
        self.assertEqual(vibe_cuda_eval._shape_expr("ceil_div(min(ceil_div(x[0] * x[1], 256), 2112), 4)", ranks),
                         "lk_ceil_div(lk_min(lk_ceil_div((x.size(0) * x.size(1)), 256), 2112), 4)")

    def test_capped_grid_loads_and_renders_host_clamp(self):
        spec = _load(_spec(["min(ceil_div(x[0] * x[1], 256), 2112)", 1, 1]))
        self.assertEqual(spec["launch"]["_grid_cpp"],
                         ["lk_min(lk_ceil_div((x.size(0) * x.size(1)), 256), 2112)", "1", "1"])
        binding = vibe_cuda_eval._render_binding(spec)
        # Host-side int64 evaluation of the capped grid, checked against the
        # CUDA grid range before being cast to unsigned for the launcher.
        self.assertIn("const int64_t lk_grid_x = "
                      "lk_min(lk_ceil_div((x.size(0) * x.size(1)), 256), 2112);", binding)
        self.assertIn('TORCH_CHECK(lk_grid_x > 0 && lk_grid_x <= UINT32_MAX, '
                      '"invalid grid x");', binding)
        self.assertIn("static_cast<unsigned>(lk_grid_x)", binding)
        self.assertIn("static int64_t lk_min(int64_t a, int64_t b) { return a < b ? a : b; }", binding)
        self.assertIn("static int64_t lk_ceil_div(int64_t a, int64_t b) "
                      "{ return (a + b - 1) / b; }", binding)

    def test_literal_floor_grid_loads(self):
        spec = _load(_spec(["max(ceil_div(x[0] * x[1], 256), 7)", 1, 1]))
        self.assertEqual(spec["launch"]["_grid_cpp"][0], "lk_max(lk_ceil_div((x.size(0) * x.size(1)), 256), 7)")

    def test_card_expression_with_constructor_scalars(self):
        # Exact capability-card reproducer: min(ceil_div(B * N, 256), 2112),
        # B/N integer constructor scalars (launch_scalars path in _load_spec).
        spec = _load(_constructor_spec(["min(ceil_div(B * N, 256), 2112)", 1, 1]))
        self.assertEqual(spec["launch"]["_grid_cpp"], ["lk_min(lk_ceil_div((B * N), 256), 2112)", "1", "1"])
        binding = vibe_cuda_eval._render_binding(spec)
        self.assertIn("const int64_t lk_grid_x = lk_min(lk_ceil_div((B * N), 256), 2112);", binding)

    def test_noninteger_clamp_literals_are_rejected(self):
        # Grid expressions are integer host sizes: a float or negative cap
        # has no int64 translation and must fail at spec load.
        for bad in ("min(ceil_div(x[0] * x[1], 256), 2112.0)", "max(ceil_div(x[0] * x[1], 256), -1)"):
            with self.assertRaisesRegex(ValueError, "unsupported shape expression"):
                _load(_spec([bad, 1, 1]))

    def test_clamp_arity_is_validated(self):
        with self.assertRaisesRegex(ValueError, "min requires two positional arguments"):
            vibe_cuda_eval._shape_expr("min(ceil_div(x[0], 256), 2112, 1)", {"x": 1})


if __name__ == "__main__":
    unittest.main()
