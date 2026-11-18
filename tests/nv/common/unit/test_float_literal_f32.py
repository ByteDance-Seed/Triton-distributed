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
"""Python float constants are emitted with fp32 spelling in CUDA source.

Legality basis: the IR types every Python float literal as float32
(little_kernel/core/expr.py, "Simplification: default float to float32").
An unsuffixed C++ floating literal is a double; under usual arithmetic
conversions it promotes the whole enclosing fp32 expression to FP64 (and,
on datacenter GPUs with FP64 at a fraction of FP32 throughput, turns
memory-roofline kernels compute-bound).  The codegen therefore spells
Python float constants with a trailing ``f`` so the emitted arithmetic
matches the declared IR dtype.  Mixed float64 expressions still promote
the fp32-valued constant exactly as the IR promotion rules describe;
frontends needing full double-precision constants spell them via typed
kernel-local ``ll.float64`` declarations.

These tests drive the real draft -> PASSES -> codegen -> export() renderer
(no nvcc) and assert source snapshots.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations import vibe_cuda_eval

KERNEL_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def act(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32],
        n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    if i < n:
        v: ll.float32 = x[i]
        y[i] = 0.5 * v * (1.0 + ll.erff(v * 1e-3 + 0.25))
'''

SPEC = {
    "kernel": "act",
    "function": "f",
    "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
    "outputs": [{"name": "y", "dtype": "float32", "shape": ["x[0]"]}],
    "scalars": [{"name": "n", "dtype": "int32", "expr": "x[0]"}],
    "kernel_args": ["x", "y", "n"],
    "forward_args": ["x"],
    "launch": {"grid": ["ceil_div(x[0], 128)", 1, 1], "block": [128, 1, 1]},
}


class FloatLiteralF32Tests(unittest.TestCase):

    def _export_cuda(self, source=KERNEL_SOURCE):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task").mkdir()
        (root / "agent" / "kernels").mkdir(parents=True)
        draft = root / "agent" / "draft.py"
        draft.write_text(source)
        spec_path = root / "agent" / "export.json"
        spec_path.write_text(json.dumps(SPEC))
        paths = vibe_cuda_eval.export(draft, spec_path, root, force=True)
        return paths[0].read_text()

    def test_plain_float_constants_are_f32_spelled(self):
        cuda = self._export_cuda()
        self.assertIn("0.5f", cuda)
        self.assertIn("1.0f", cuda)
        self.assertIn("0.25f", cuda)
        # No bare double spellings of the same constants may survive.
        for bare in ("0.5 ", "1.0 ", "0.25 "):
            self.assertNotIn(bare, cuda)
        self.assertNotIn("0.5 *", cuda)

    def test_scientific_notation_is_f32_spelled(self):
        cuda = self._export_cuda()
        self.assertIn("0.001f", cuda)

    def test_negative_constant_via_unary_minus(self):
        src = KERNEL_SOURCE.replace("y[i] = 0.5 * v * (1.0 + ll.erff(v * 1e-3 + 0.25))", "y[i] = -0.75 * v")
        cuda = self._export_cuda(source=src)
        self.assertIn("-0.75f", cuda)
        self.assertNotIn("-0.75 *", cuda.replace("-0.75f *", ""))

    def test_integer_constants_keep_plain_spelling(self):
        cuda = self._export_cuda()
        # block size / indices stay unsuffixed integer literals.
        self.assertIn("* 128 +", cuda)
        self.assertNotIn("128f", cuda)


if __name__ == "__main__":
    unittest.main()
