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
"""Typed (dtype-preserving) 16-byte global vector stores: ll.st_global_v4_bf16.

Store-side completion of the typed v4 family shipped in cycle-12
(``ll.ld_global_v4_f32`` / ``ll.ld_global_v4_s32``).  ``st_global_v4_u32``
takes its payload as a local ``uint32[4]`` array and ``gmem_st_v4`` lowers
through a C++ ``uint4`` reinterpret; ``st_global_v4_bf16`` emits the same
single ``st.global.v4.b32`` transaction directly from four uint32 SCALAR
words (the shape produced by ``ll.pack_bf16_float``), while the
``template <typename T>`` helper preserves the destination tensor's element
type for pointer arithmetic (``elem_offset`` in elements of ``mem``'s own
dtype, any dtype — bf16 motivating: 8 bf16 elements per 16-byte vector).

Legality contract under test (mirrors the module docstring in
language/intrin/memory.py):
  * one 16-byte transaction per call, four 32-bit lanes, little-endian, taken
    from w0..w3 in memory order;
  * the same helper is #ifndef-guarded, so a kernel with many call sites emits
    exactly one definition (no C++ redefinition);
  * the store is bit-exact and cache-policy-free, so these tests assert codegen
    shape only; alignment/in-bounds and the ragged-tail scalar fallback are the
    frontend's responsibility (identical to st_global_v4_u32).
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
def typed_v4_store(x: ll.Tensor[ll.float32],
                   y: ll.Tensor[ll.bfloat16]) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    base: ll.int32 = tid * 8
    p01: ll.uint32 = ll.pack_bf16_float(x[base], x[base + 1])
    p23: ll.uint32 = ll.pack_bf16_float(x[base + 2], x[base + 3])
    p45: ll.uint32 = ll.pack_bf16_float(x[base + 4], x[base + 5])
    p67: ll.uint32 = ll.pack_bf16_float(x[base + 6], x[base + 7])
    # two call sites exercise the #ifndef guard across repeat use of one helper
    ll.st_global_v4_bf16(y, base, p01, p23, p45, p67)
    ll.st_global_v4_bf16(y, base + 1024, p67, p45, p23, p01)
'''

SPEC = {
    "kernel": "typed_v4_store",
    "function": "typed_v4_store",
    "inputs": [
        {"name": "x", "dtype": "float32", "rank": 1},
    ],
    "output": {"name": "y", "dtype": "bfloat16", "shape": ["x[0]"]},
    "equalities": [],
    "scalars": [],
    "launch": {"grid": ["(x[0] + 2047) // 2048", 1, 1], "block": [256, 1, 1], "shared_memory_bytes": 0},
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


class TestTypedVectorStGlobalV4(unittest.TestCase):

    def test_exported_as_builtin(self):
        """The symbol is exported through little_kernel.language as a builtin."""
        self.assertTrue(getattr(ll.st_global_v4_bf16, BUILTIN_ATTR, False))
        self.assertTrue(hasattr(ll.st_global_v4_bf16, CODEGEN_FUNC_ATTR))
        self.assertTrue(hasattr(ll.st_global_v4_bf16, EVAL_RETURN_TYPE_ATTR))

    def test_registered_in_intrin_module(self):
        self.assertTrue(getattr(memory_intrin.st_global_v4_bf16, BUILTIN_ATTR, False))

    def test_codegen_emits_typed_v4_store(self):
        """Direct codegen: PTX v4 store from four uint32 scalar words."""
        b = memory_intrin.codegen_st_global_v4_bf16("y", "off", "p01", "p23", "p45", "p67")
        self.assertIn("st.global.v4.b32 [%0], {%1, %2, %3, %4};", b.body)
        self.assertIn('"l"(ptr), "r"(w0), "r"(w1), "r"(w2), "r"(w3)', b.body)
        self.assertIn("template <typename T>", b.body)
        self.assertIn("uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3", b.body)
        self.assertIn('"memory"', b.body)
        self.assertIn("#ifndef LK_ST_GLOBAL_V4_BF16_FN_DEFINED", b.body)
        self.assertEqual("st_global_v4_bf16_fn(y + off, p01, p23, p45, p67)", b.return_val)
        self.assertIn("<cstdint>", b.includes)

    def test_e2e_kernel_uses_typed_vector_store(self):
        """End-to-end lowering emits the guarded helper and typed v4 PTX."""
        with tempfile.TemporaryDirectory() as d:
            cu = _lower(Path(d))
        # One 16-byte typed vector transaction per logical store; none of the
        # four words degrade to scalar st.global.b32 element stores.
        self.assertIn("st.global.v4.b32", cu)
        self.assertIn("st_global_v4_bf16_fn(y +", cu)
        # #ifndef guard holds: the helper is defined exactly once even though
        # it is called at two sites in the kernel.
        self.assertEqual(cu.count("void st_global_v4_bf16_fn("), 1)
        self.assertEqual(cu.count("LK_ST_GLOBAL_V4_BF16_FN_DEFINED"), 2)
        # Pointer arithmetic stays in the tensor's own dtype (bf16*), i.e. the
        # template instantiation keeps element-typed offset units.
        self.assertIn("__nv_bfloat16*", cu.replace("const ", ""))


if __name__ == "__main__":
    unittest.main()
