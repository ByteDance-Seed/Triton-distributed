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
"""Opt-in spec-level ``no_alias`` -> __restrict__ on the entry kernel.

Render-layer legality tests (no GPU): the flag must
  * be emitted only on pointer/tensor params of the __global__ entry kernel,
  * never change function bodies or device-helper signatures,
  * be rejected at spec load when it contradicts an alias_of output, and
  * leave flag-off renders byte-identical to accepted-v007 behavior.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_kernel, _load_spec, _render_cuda

SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda")
def _scale(x: ll.float32) -> ll.float32:
    return 2.0 * x


@lk.ll_kernel(backend="cuda", is_entry=True)
def tiny(a: ll.Tensor[ll.float32], b: ll.Tensor[ll.float32],
         out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        out[i] = _scale(a[i]) + b[i]
'''


def _spec(extra=None):
    spec = {
        "kernel": "tiny",
        "function": "tiny",
        "inputs": [
            {"name": "a", "dtype": "float32", "rank": 1},
            {"name": "b", "dtype": "float32", "rank": 1},
        ],
        "output": {"name": "out", "dtype": "float32", "shape": ["a[0]"]},
        "scalars": [{"name": "n", "expr": "a[0]"}],
        "launch": {"grid": ["ceil_div(n, 256)", 1, 1], "block": [256, 1, 1]},
    }
    spec.update(extra or {})
    return spec


class NoAliasRestrictTests(unittest.TestCase):

    def _render(self, directory, extra=None):
        root = Path(directory)
        spec_path = root / "export.json"
        spec_path.write_text(json.dumps(_spec(extra)))
        source = root / "kernel.py"
        source.write_text(SOURCE)
        spec = _load_spec(spec_path)
        kernel = _load_kernel(source, "tiny")
        return _render_cuda(spec, kernel)

    def test_restrict_only_on_entry_pointer_params(self):
        with tempfile.TemporaryDirectory() as directory:
            default_cuda = self._render(directory)
            alias_cuda = self._render(directory, {"no_alias": True})
        self.assertNotIn("__restrict__", default_cuda)
        for needle in ("float* __restrict__ a", "float* __restrict__ b", "float* __restrict__ out"):
            self.assertIn(needle, alias_cuda)
        self.assertNotIn("int32_t __restrict__", alias_cuda)
        helper = [ln for ln in alias_cuda.splitlines() if "_scale" in ln and "__device__" in ln]
        self.assertTrue(helper)
        self.assertNotIn("__restrict__", helper[0])
        # Bodies unchanged: with the annotation stripped, only signature
        # line wrapping may differ, never C tokens.
        squeezed = lambda s: "".join(s.split())
        self.assertEqual(squeezed(alias_cuda.replace(" __restrict__", "")), squeezed(default_cuda))

    def test_rejects_aliased_output_contradiction(self):
        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / "export.json"
            spec = _spec({"no_alias": True})
            spec["output"] = {"name": "out", "dtype": "float32", "alias_of": "a"}
            spec_path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "no_alias"):
                _load_spec(spec_path)

    def test_rejects_non_boolean_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / "export.json"
            spec_path.write_text(json.dumps(_spec({"no_alias": "yes"})))
            with self.assertRaisesRegex(ValueError, "no_alias must be a boolean"):
                _load_spec(spec_path)


if __name__ == "__main__":
    unittest.main()
