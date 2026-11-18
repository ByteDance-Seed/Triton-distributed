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
"""ll.unroll_dyn(range(...), count) emits a counted partial-unroll pragma.

ll.unroll is a full-unroll request that the target compiler ignores on
dynamic-trip-count loops (e.g. `for c in range(lane, chunks, 32)` with a
runtime `chunks`). ll.unroll_dyn lowers to `#pragma unroll <count>`, which
the target compiler honors on runtime-bound loops: the body is duplicated
<count> times with a residue epilogue, letting the backend batch independent
predicated streaming loads across the copies (software-pipelining-style
load front-loading) without hand-duplicating the frontend body.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower

SOURCE_DYN = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def dyn_reduce(a: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32],
               chunks: ll.int32) -> ll.void:
    lane: ll.int32 = ll.threadIdx_x() & 31
    acc: ll.int32 = 0
    for c in {ITER}:
        if c < chunks:
            w: ll.int32 = ll.int32(ll.gmem_ld_nc_u32(ll.ptr_byte_offset(a, c << 2)))
            acc = acc + w
    if lane == 0:
        out[0] = acc
'''

SPEC = {
    "kernel": "dyn_reduce",
    "function": "dynred",
    "inputs": [{"name": "a", "dtype": "int32", "rank": 1}],
    "output": {"name": "out", "dtype": "int32", "shape": ["1"]},
    "constructor": [],
    "scalars": [{"name": "chunks", "expr": "a[0]"}],
    "kernel_args": ["a", "out", "chunks"],
    "launch": {"grid": ["1", 1, 1], "block": [32, 1, 1]},
}


def _build(tmp_root, iter_expr):
    root = Path(tmp_root)
    task = root / "task"
    task.mkdir()
    (task / "model.py").write_text("class Model: pass\n")
    (task / "test_cases.py").write_text("test_cases = []\n")
    program = root / "program"
    program.mkdir()
    (program / "draft.py").write_text(SOURCE_DYN.replace("{ITER}", iter_expr))
    (program / "export.json").write_text(json.dumps(SPEC))
    output = root / "out"
    lower(task, program, output, 0)
    return (output / "kernels/agent_lk_dynred.cu").read_text()


class UnrollDynTests(unittest.TestCase):

    def test_dynamic_loop_emits_counted_pragma(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = _build(directory, "ll.unroll_dyn(range(lane, chunks, 32), 4)")
            self.assertEqual(cuda.count("#pragma unroll 4"), 1)
            self.assertNotIn("#pragma unroll\n", cuda)
            # The dynamic trip loop itself is untouched: same affine bounds
            # and stride in the emitted header, body semantics unchanged.
            self.assertIn("c < chunks", cuda)
            self.assertIn("c += 32", cuda)
            self.assertIn("acc = acc + w;", cuda)

    def test_keyword_count_form(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = _build(directory, "ll.unroll_dyn(range(lane, chunks, 32), count=2)")
            self.assertEqual(cuda.count("#pragma unroll 2"), 1)

    def test_plain_unroll_and_plain_loop_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = _build(directory, "ll.unroll(range(lane, chunks, 32))")
            self.assertIn("#pragma unroll", cuda)
            self.assertNotIn("#pragma unroll 4", cuda)
        with tempfile.TemporaryDirectory() as directory:
            cuda = _build(directory, "range(lane, chunks, 32)")
            self.assertNotIn("#pragma", cuda)

    def test_missing_count_is_a_compile_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as ctx:
                _build(directory, "ll.unroll_dyn(range(lane, chunks, 32))")
            self.assertIn("requires an unroll count", str(ctx.exception))

    def test_non_positive_count_is_a_compile_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as ctx:
                _build(directory, "ll.unroll_dyn(range(lane, chunks, 32), 0)")
            self.assertIn("positive integer", str(ctx.exception))

    def test_non_literal_count_is_a_compile_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as ctx:
                _build(directory, "ll.unroll_dyn(range(lane, chunks, 32), chunks)")
            self.assertIn("integer literal", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
