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
"""`__maxnreg__` per-kernel register-budget codegen knob.

Render-layer legality tests (no GPU):
  * the module global `__LITTLE_KERNEL_CUDA_MAXNREG__` (int > 0) makes the
    entry kernel render with `__maxnreg__(N)` on its `__global__`
    signature, decoupling the ptxas register budget from the occupancy
    target of `__launch_bounds__`;
  * it is mutually exclusive with
    `__LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__` (CUDA forbids
    combining `__maxnreg__` with `__launch_bounds__`): both set raises
    ValueError at codegen time;
  * non-int, bool, and non-positive values raise ValueError;
  * absence renders no register-budget attribute (the v010 default).
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_kernel, _load_spec, _render_cuda

SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

{global_line}


@lk.ll_kernel(backend="cuda", is_entry=True)
def saxpy(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32],
          n: ll.int32) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if c < n:
        y[c] = x[c] + y[c]
'''


def _spec():
    return {
        "kernel": "saxpy",
        "function": "saxpy",
        "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
        "output": {"name": "y", "dtype": "float32", "shape": ["x[0]"]},
        "scalars": [{"name": "n", "expr": "x[0]"}],
        "launch": {"grid": ["ceil_div(n, 256)", 1, 1], "block": [256, 1, 1]},
    }


class MaxnregTests(unittest.TestCase):

    def _render(self, directory, global_line=""):
        root = Path(directory)
        (root / "export.json").write_text(json.dumps(_spec()))
        kernel_path = root / "kernel.py"
        kernel_path.write_text(SOURCE.format(global_line=global_line))
        spec = _load_spec(root / "export.json")
        kernel = _load_kernel(kernel_path, "saxpy")
        return _render_cuda(spec, kernel)

    def test_maxnreg_renders_on_entry_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, "__LITTLE_KERNEL_CUDA_MAXNREG__ = 168")
        self.assertIn('extern "C" __global__ __maxnreg__(168) void agent_saxpy', cuda)
        self.assertNotIn("__launch_bounds__", cuda)

    def test_absent_renders_no_attribute(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory)
        self.assertNotIn("__maxnreg__", cuda)
        self.assertNotIn("__launch_bounds__", cuda)

    def test_mutual_exclusion_with_launch_bounds_min_blocks(self):
        global_line = ("__LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__ = 2\n"
                       "__LITTLE_KERNEL_CUDA_MAXNREG__ = 128")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self._render(directory, global_line)

    def test_launch_bounds_alone_still_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, "__LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__ = 2")
        self.assertIn("__launch_bounds__(256, 2)", cuda)
        self.assertNotIn("__maxnreg__", cuda)

    def test_illegal_values_raise(self):
        for global_line in (
                "__LITTLE_KERNEL_CUDA_MAXNREG__ = 0",
                "__LITTLE_KERNEL_CUDA_MAXNREG__ = -8",
                "__LITTLE_KERNEL_CUDA_MAXNREG__ = True",
                "__LITTLE_KERNEL_CUDA_MAXNREG__ = 1.5",
                '__LITTLE_KERNEL_CUDA_MAXNREG__ = "128"',
        ):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self._render(directory, global_line)


if __name__ == "__main__":
    unittest.main()
