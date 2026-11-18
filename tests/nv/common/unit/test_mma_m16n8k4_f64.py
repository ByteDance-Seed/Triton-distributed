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
"""FP64 tensor-core MMA atom: ll.mma_m16n8k4_f64.

Companion to the bf16/f16 warp-level mma family (mma_m16n8k8 /
mma_m16n8k16 / mma_m16n16k16). PTX defines exactly one FP64 mma.sync
shape, ``mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64``, with
binary64 operands and a binary64 accumulator (sm_89/sm_90). The fragment
shapes (A=2, B=1, C/D=4 f64 registers per lane) and the row.col layout
are fixed by the PTX ISA, so a single generic intrinsic covers every
f64 GEMM tile built from this atom.

Legality contract under test:
  * the intrinsic is a real language builtin exported through
    ``little_kernel.language`` and registered in the intrin module;
  * codegen emits the single m16n8k4 row.col f64 PTX instruction with
    four "+d" accumulator lanes bound to c[0..4), two "d" A lanes
    (a[0], a[1]) and one "d" B lane (b[0]);
  * the operand dtype is whitelisted BEFORE asm text is composed: f64
    aliases normalize to "f64", everything else raises ValueError;
  * C++ helper text uses ``double (&c)[4]``/``const double (&a)[2]``/
    ``const double (&b)[1]`` so fragment arrays stay typed float64
    through type inference and codegen.
"""

import json
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR
import little_kernel.backends.nv.common.language.intrin.memory as memory_intrin

DRAFT = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def f64_mma(a: ll.Tensor[ll.float64], b: ll.Tensor[ll.float64],
            c: ll.Tensor[ll.float64]) -> ll.void:
    acc = ll.empty([4], dtype=ll.float64, scope="local")
    frag_a = ll.empty([2], dtype=ll.float64, scope="local")
    frag_b = ll.empty([1], dtype=ll.float64, scope="local")
    for i in ll.unroll(range(4)):
        acc[i] = 0.0
    for i in ll.unroll(range(2)):
        frag_a[i] = a[i]
    frag_b[0] = b[0]
    ll.mma_m16n8k4_f64(acc, frag_a, frag_b, acc, dtype="f64")
    for i in ll.unroll(range(4)):
        c[i] = acc[i]
'''

SPEC = {
    "kernel": "f64_mma",
    "function": "f64_mma",
    "inputs": [
        {"name": "a", "dtype": "float64", "rank": 1},
        {"name": "b", "dtype": "float64", "rank": 1},
    ],
    "output": {"name": "c", "dtype": "float64", "shape": ["a[0]"]},
    "equalities": [],
    "scalars": [],
    "launch": {"grid": [1, 1, 1], "block": [32, 1, 1], "shared_memory_bytes": 0},
}


def _lower(root: Path):
    task = root / "task"
    task.mkdir()
    (task / "model.py").write_text("class Model: pass\n")
    (task / "test_cases.py").write_text("test_cases = []\n")
    program = root / "program"
    program.mkdir()
    (program / "draft.py").write_text(DRAFT)
    (program / "export.json").write_text(json.dumps(SPEC))
    out = root / "output"
    lower(task, program, out, 0)
    return next(out.glob("kernels/*.cu")).read_text()


class TestMmaM16N8K4F64(unittest.TestCase):

    def test_exported_as_builtin(self):
        """The symbol is exported through little_kernel.language as a builtin."""
        self.assertTrue(getattr(ll.mma_m16n8k4_f64, BUILTIN_ATTR, False))
        self.assertTrue(hasattr(ll.mma_m16n8k4_f64, CODEGEN_FUNC_ATTR))
        self.assertTrue(hasattr(ll.mma_m16n8k4_f64, EVAL_RETURN_TYPE_ATTR))
        self.assertTrue(getattr(memory_intrin.mma_m16n8k4_f64, BUILTIN_ATTR, False))

    def test_ptx_dtype_accepts_f64_aliases(self):
        """f64/fp64/float64/double all normalize to the f64 PTX token."""
        for token in ("f64", "fp64", "float64", "double", "F64", "'f64'"):
            self.assertEqual(memory_intrin._mma_ptx_dtype(token), "f64")

    def test_ptx_dtype_rejects_unsupported_dtype(self):
        """Non-f64 tokens raise ValueError before any asm text is composed."""
        with self.assertRaises(ValueError):
            memory_intrin._mma_ptx_dtype("f32")
        with self.assertRaises(ValueError):
            memory_intrin._mma_ptx_dtype("int8")

    def test_codegen_rejects_non_f64_dtype(self):
        """The f64 mma atom never emits a 16-bit operand dtype opcode."""
        with self.assertRaises(ValueError):
            memory_intrin.codegen_mma_m16n8k4_f64("acc", "a", "b", "acc", dtype="f16")

    def test_codegen_emits_m16n8k4_f64_ptx(self):
        """Direct codegen: one FP64 mma.sync with typed binary64 operands."""
        b = memory_intrin.codegen_mma_m16n8k4_f64("acc", "frag_a", "frag_b", "acc")
        self.assertIn("mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64", b.body)
        # Four "+d" accumulator lanes bound to c[0..4).
        self.assertIn('"+d"(c[0])', b.body)
        self.assertIn('"+d"(c[3])', b.body)
        # Two "d" A lanes and one "d" B lane.
        self.assertIn('"d"(a[0])', b.body)
        self.assertIn('"d"(a[1])', b.body)
        self.assertIn('"d"(b[0])', b.body)
        # Fragment arrays stay binary64-typed at the C++ boundary.
        self.assertIn("double (&c)[4]", b.body)
        self.assertIn("const double (&a)[2]", b.body)
        self.assertIn("const double (&b)[1]", b.body)
        self.assertEqual("mma_m16n8k4_f64_fn(acc, frag_a, frag_b, acc)", b.return_val)

    def test_e2e_kernel_lowers_f64_mma(self):
        """End-to-end lowering of a 16x8x4 f64 fragment program emits the PTX."""
        with tempfile.TemporaryDirectory() as d:
            cu = _lower(Path(d))
        self.assertIn("mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64", cu)
        self.assertIn("mma_m16n8k4_f64_fn(", cu)
        # The FP64 accumulator stays binary64; no f32 mma opcode is emitted.
        self.assertNotIn(".bf16.bf16.f32", cu)
        self.assertNotIn(".f16.f16.f32", cu)


if __name__ == "__main__":
    unittest.main()
