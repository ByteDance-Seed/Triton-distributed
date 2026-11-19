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
"""Typed (dtype-preserving) 16-byte global vector loads: ll.ld_global_v4_f32 /
ll.ld_global_v4_s32.

Generic companion to the uint32-container ``ld_global_v4_u32`` family.  These
emit the same single ``ld.global.v4.b32`` transaction but bind the four 32-bit
lanes to registers of the destination array's element class (``.f32`` /
integer), so float and signed-int element arrays keep their type through type
inference and codegen instead of round-tripping through uint32.

Legality contract under test (mirrors the module docstring in
language/intrin/memory.py):
  * one 16-byte transaction per call, four 32-bit lanes, little-endian, landing
    in out[0..4) in memory order;
  * the same helper is #ifndef-guarded, so a kernel with many call sites emits
    exactly one definition (no C++ redefinition);
  * the copy is bit-exact and cache-policy-free, so these tests assert codegen
    shape only; alignment/in-bounds and the ragged-tail scalar fallback are the
    frontend's responsibility (identical to ld_global_v4_u32).
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
def typed_v4(x: ll.Tensor[ll.float32], acc: ll.Tensor[ll.int32],
             y: ll.Tensor[ll.float32]) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    wv = ll.empty([4], dtype=ll.float32, scope="local")
    va = ll.empty([4], dtype=ll.int32, scope="local")
    # two f32 calls (exercise the #ifndef guard across repeat use of one helper)
    ll.ld_global_v4_f32(x, tid * 4, wv)
    ll.ld_global_v4_f32(x, tid * 4 + 128, wv)
    ll.ld_global_v4_s32(acc, tid * 4, va)
    y[tid * 4] = wv[0] + ll.val_cast(va[0], ll.float32)
    y[tid * 4 + 1] = wv[1] + ll.val_cast(va[1], ll.float32)
    y[tid * 4 + 2] = wv[2] + ll.val_cast(va[2], ll.float32)
    y[tid * 4 + 3] = wv[3] + ll.val_cast(va[3], ll.float32)
'''

SPEC = {
    "kernel": "typed_v4",
    "function": "typed_v4",
    "inputs": [
        {"name": "x", "dtype": "float32", "rank": 1},
        {"name": "acc", "dtype": "int32", "rank": 1},
    ],
    "output": {"name": "y", "dtype": "float32", "shape": ["x[0]"]},
    "equalities": [],
    "scalars": [],
    "launch": {"grid": ["(x[0] + 1023) // 1024", 1, 1], "block": [256, 1, 1], "shared_memory_bytes": 0},
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


class TestTypedVectorLdGlobalV4(unittest.TestCase):

    def test_exported_as_builtins(self):
        """Both symbols are exported through little_kernel.language as builtins."""
        self.assertTrue(getattr(ll.ld_global_v4_f32, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.ld_global_v4_s32, BUILTIN_ATTR, False))
        self.assertTrue(hasattr(ll.ld_global_v4_f32, CODEGEN_FUNC_ATTR))
        self.assertTrue(hasattr(ll.ld_global_v4_s32, CODEGEN_FUNC_ATTR))
        self.assertTrue(hasattr(ll.ld_global_v4_f32, EVAL_RETURN_TYPE_ATTR))
        self.assertTrue(hasattr(ll.ld_global_v4_s32, EVAL_RETURN_TYPE_ATTR))

    def test_registered_in_intrin_module(self):
        self.assertTrue(getattr(memory_intrin.ld_global_v4_f32, BUILTIN_ATTR, False))
        self.assertTrue(getattr(memory_intrin.ld_global_v4_s32, BUILTIN_ATTR, False))

    def test_codegen_f32_emits_typed_v4_load(self):
        """Direct codegen: f32 form binds four .f32 registers to out[0..3]."""
        b = memory_intrin.codegen_ld_global_v4_f32("x", "off", "wv")
        self.assertIn("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];", b.body)
        self.assertIn('"=f"(out[0])', b.body)
        self.assertIn('"=f"(out[3])', b.body)
        self.assertIn("float (&out)[4]", b.body)
        self.assertIn("#ifndef LK_LD_GLOBAL_V4_F32_FN_DEFINED", b.body)
        self.assertEqual("ld_global_v4_f32_fn(x + off, wv)", b.return_val)

    def test_codegen_s32_emits_typed_v4_load(self):
        """Direct codegen: s32 form binds four integer registers to out[0..3]."""
        b = memory_intrin.codegen_ld_global_v4_s32("acc", "off", "va")
        self.assertIn("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];", b.body)
        self.assertIn('"=r"(out[0])', b.body)
        self.assertIn('"=r"(out[3])', b.body)
        self.assertIn("int32_t (&out)[4]", b.body)
        self.assertIn("#ifndef LK_LD_GLOBAL_V4_S32_FN_DEFINED", b.body)
        self.assertEqual("ld_global_v4_s32_fn(acc + off, va)", b.return_val)

    def test_e2e_kernel_uses_typed_vector_loads(self):
        """End-to-end lowering emits the guarded helpers and typed v4 PTX."""
        with tempfile.TemporaryDirectory() as d:
            cu = _lower(Path(d))
        # One 16-byte typed vector transaction per logical load; none of the
        # four lanes degrade to scalar ld.global.f32/s32 element loads.
        self.assertIn("ld.global.v4.b32", cu)
        self.assertIn("ld_global_v4_f32_fn(x +", cu)
        self.assertIn("ld_global_v4_s32_fn(acc +", cu)
        # #ifndef guard holds: each helper is defined exactly once even though
        # the f32 form is used twice in the kernel.
        self.assertEqual(cu.count("void ld_global_v4_f32_fn("), 1)
        self.assertEqual(cu.count("void ld_global_v4_s32_fn("), 1)
        self.assertEqual(cu.count("LK_LD_GLOBAL_V4_F32_FN_DEFINED"), 2)  # ifndef + define
        self.assertEqual(cu.count("LK_LD_GLOBAL_V4_S32_FN_DEFINED"), 2)


if __name__ == "__main__":
    unittest.main()
