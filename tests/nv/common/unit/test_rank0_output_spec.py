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

################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software.
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
"""
Spec-validation tests for rank-0 scalar outputs
(integrations/vibe_cuda_eval.py::_load_spec).

Contract under test:

- An output with "shape": [] is a 0-dim scalar tensor (numel == 1): the
  spec loader accepts it and the host wrapper allocates at::empty({}).
- A shape-[1] output keeps the rank-1 ABI: scalar and length-one stay
  distinct (scalar-vs-length-one negative ABI distinction).
- The relaxation is confined to outputs and inputs: rank-0 workspaces and
  TMA descriptors on rank-0 tensors still raise; rank-0 inputs are device
  scalars (deliberate feature, see test_export_spec_contracts.py).

All assertions are validation-level (no GPU, no nvcc).
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations import vibe_cuda_eval


def _spec(outputs, **extra):
    data = {
        "kernel": "k",
        "inputs": [{"name": "x", "dtype": "float64", "rank": 1}],
        "outputs": outputs,
        "scalars": [],
        "workspaces": extra.pop("workspaces", []),
        "launch": {"grid": ["1", "1", "1"], "block": [32, 1, 1]},
    }
    data.update(extra)
    return data


def _load(spec):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
        path = Path(f.name)
    try:
        return vibe_cuda_eval._load_spec(path)
    finally:
        path.unlink(missing_ok=True)


class TestRank0Output(unittest.TestCase):

    def test_rank0_output_accepted(self):
        spec = _load(_spec([{"name": "c", "dtype": "float64", "shape": []}]))
        out = spec["_outputs"][0]
        self.assertEqual(out["_shape_cpp"], [])

    def test_rank0_binding_emits_scalar_alloc(self):
        spec = _load(_spec([{"name": "c", "dtype": "float64", "shape": []}]))
        binding = vibe_cuda_eval._render_binding(spec)
        self.assertIn("torch::Tensor c = at::empty({}, "
                      "x.options().dtype(at::kDouble));", binding)

    def test_length_one_stays_rank1(self):
        spec = _load(_spec([{"name": "c", "dtype": "float64", "shape": [1]}]))
        out = spec["_outputs"][0]
        self.assertEqual(out["_shape_cpp"], ["1"])
        binding = vibe_cuda_eval._render_binding(spec)
        self.assertIn("torch::Tensor c = at::empty({1}, "
                      "x.options().dtype(at::kDouble));", binding)

    def test_scalar_and_length_one_differ(self):
        rank0 = vibe_cuda_eval._render_binding(_load(_spec([{"name": "c", "dtype": "float64", "shape": []}])))
        rank1 = vibe_cuda_eval._render_binding(_load(_spec([{"name": "c", "dtype": "float64", "shape": [1]}])))
        self.assertNotEqual(rank0, rank1)
        self.assertNotIn("at::empty({1}", rank0)
        self.assertNotIn("at::empty({}", rank1)

    def test_rank0_workspace_rejected(self):
        with self.assertRaises(ValueError):
            _load(
                _spec([{"name": "c", "dtype": "float64", "shape": []}],
                      workspaces=[{"name": "w", "dtype": "float64", "shape": []}]))

    def test_rank0_input_accepted_as_device_scalar(self):
        # Candidate-base contract (supersedes the v009-lineage rejection
        # this check arrived with): rank-0 inputs are a deliberate feature —
        # device scalar tensors (in-place counts, status words) whose
        # binding emits a dim()==0 check; subscripts on a rank-0 name are
        # rejected inside _shape_expr. Positive end-to-end coverage lives in
        # test_export_spec_contracts.py::test_rank0_input_and_alias_output.
        spec = _spec([{"name": "c", "dtype": "float64", "shape": []}])
        spec["inputs"] = [{"name": "x", "dtype": "float64", "rank": 0}]
        loaded = _load(spec)
        self.assertEqual(loaded["_ranks"]["x"], 0)

    def test_tma_on_rank0_rejected(self):
        with self.assertRaises(ValueError):
            _load(
                _spec([{"name": "c", "dtype": "float64", "shape": []}],
                      tma_descriptors=[{"name": "d", "tensor": "c", "rank": 2, "box": [1, 1], "swizzle": 0}]))

    def test_rank1_outputs_unchanged(self):
        # Golden invariant: rank>=1 programs see no behavior change.
        spec = _load(_spec([{"name": "c", "dtype": "float64", "shape": ["x[0]"]}]))
        self.assertEqual(spec["_outputs"][0]["_shape_cpp"], ["x.size(0)"])


if __name__ == "__main__":
    unittest.main()
