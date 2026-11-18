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
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
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
"""Regression coverage for dtype-generic 64-bit support (semantic key
``ir.float64.warp_shuffle_and_host_binding``):

* warp shuffle intrinsics are dtype-generic: an ``ll.float64`` value shuffles
  through the full ``lower()`` path to CUDA's native 64-bit
  ``__shfl_*_sync`` overload (CUDA shuffles ``double`` natively, so no
  lo/hi u32 split is required);
* the host ABI binding table carries ``float64``, so an f64 workspace /
  staging tensor is a legal export-manifest entry and renders a ``double``
  device pointer backed by an ``at::kDouble`` allocation.

These are generic compiler properties, not task-specific behavior.
"""

import json
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.language.builtin_base import EVAL_RETURN_TYPE_ATTR
import little_kernel.backends.nv.common.language.intrin.shuffle as shuffle
from little_kernel.backends.nv.common.integrations import vibe_cuda_eval

F64_BUTTERFLY_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def f64bf(x: ll.Tensor[ll.float64], out: ll.Tensor[ll.float64]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 32 + ll.threadIdx_x()
    d: ll.float64 = x[i]
    d = ll.shfl_xor_sync(4294967295, d, 1)
    out[i] = d
'''


class TestFloat64Shuffle(unittest.TestCase):

    def test_return_type_preserves_float64(self):
        for fn in (shuffle.shfl_sync, shuffle.shfl_xor_sync, shuffle.shfl_up_sync, shuffle.shfl_down_sync):
            with self.subTest(name=fn.__name__):
                self.assertEqual(getattr(fn, EVAL_RETURN_TYPE_ATTR)(ll.uint32, ll.float64, ll.int32), ll.float64)

    def test_native_64bit_shuffle_emitted(self):
        """A float64 operand lowers to the native double shuffle overload."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            task = tmp / "task"
            task.mkdir()
            (task / "model.py").write_text("class Model: pass\n")
            (task / "test_cases.py").write_text("test_cases = []\n")
            program = tmp / "program"
            program.mkdir()
            (program / "draft.py").write_text(F64_BUTTERFLY_SOURCE)
            (program / "export.json").write_text(
                json.dumps({
                    "kernel": "f64bf",
                    "function": "f64bf",
                    "inputs": [{"name": "x", "dtype": "float64", "rank": 1}],
                    "output": {"name": "out", "dtype": "float64", "shape": ["x[0]"]},
                    "scalars": [],
                    "launch": {"grid": ["ceil_div(x[0], 32)", 1, 1], "block": [32, 1, 1]},
                }))
            output = tmp / "output"
            lower(task, program, output, 0)
            kernels = sorted((output / "kernels").glob("*.cu"))
            self.assertEqual(len(kernels), 1, kernels)
            cuda = kernels[0].read_text()
        # The double-typed value goes straight to the native intrinsic:
        # no lo/hi-int reinterpret pair or inline-asm escape is needed.
        self.assertIn("__shfl_xor_sync(4294967295, d, 1)", cuda)
        self.assertNotIn("__double2loint", cuda)
        self.assertNotIn("__hiloint2double", cuda)
        self.assertNotIn("asm", cuda)


class TestFloat64WorkspaceBinding(unittest.TestCase):

    def test_dtype_table_has_float64(self):
        cpp_type, torch_dtype, _ = vibe_cuda_eval._DTYPES["float64"]
        self.assertEqual(cpp_type, "double")
        self.assertEqual(torch_dtype, "at::kDouble")

    def _spec(self, tmp: Path) -> Path:
        spec = tmp / "export.json"
        spec.write_text(
            json.dumps({
                "kernel": "f64ws",
                "function": "f64ws",
                "inputs": [{"name": "x", "dtype": "float64", "rank": 1}],
                "output": {"name": "out", "dtype": "float64", "shape": ["x[0]"]},
                "scalars": [],
                "workspaces": [{"name": "ws", "dtype": "float64", "shape": ["x[0]"], "zero": True}],
                "launch": {"grid": [1, 1, 1], "block": [32, 1, 1]},
            }))
        return spec

    def test_float64_workspace_loads_and_binds(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(Path(tmp))
            loaded = vibe_cuda_eval._load_spec(spec)
            self.assertEqual([w["dtype"] for w in loaded["workspaces"]], ["float64"])
            binding = vibe_cuda_eval._render_binding(loaded)
        # f64 staging tensor crosses the export boundary as at::kDouble.
        self.assertIn("at::kDouble", binding)
        self.assertIn('torch::Tensor ws = at::empty({x.size(0)}, '
                      'x.options().dtype(at::kDouble))', binding)


if __name__ == "__main__":
    unittest.main()
