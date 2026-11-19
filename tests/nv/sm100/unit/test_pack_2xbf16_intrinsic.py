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
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""Unit tests for the pack_2xbf16 out-param BF16 pack intrinsic.

Legality invariants under test (mirrors the docstring in
language/intrin/cuda_asm.py):
  * codegen emits the single-instruction PTX ``cvt.rn.bf16x2.f32`` idiom with
    RNE on both lanes, hi bound to the high-half operand slot and lo to the
    low-half operand slot (PTX packs the first source into d.hi, the second
    into d.lo);
  * the intrinsic is void-typed with an out-param: the call expression writes
    through ``&out`` into an already-declared uint32 register, matching the
    drain style of the ``tmem_ld`` family (complements the value-returning
    ``pack_bf16_float``);
  * the helper is #ifndef-guarded, so a kernel with several call sites emits
    exactly one ``pack_2xbf16_fn`` definition (no C++ redefinition); and
  * the symbol is registered in both the sm90 and sm100 backend.json export
    maps, resolving to the shared nv-common cuda_asm module.
"""

import json
from pathlib import Path
import unittest

import little_kernel as lk
import little_kernel.language as ll
import little_kernel.backends.nv.common.language.intrin.cuda_asm as cuda_asm
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES
from little_kernel.language.builtin_base import (
    BUILTIN_ATTR,
    CODEGEN_FUNC_ATTR,
    EVAL_RETURN_TYPE_ATTR,
    Builtin,
)


class TestPack2xBF16Intrinsic(unittest.TestCase):
    """Tests for pack_2xbf16 (out-param single-instruction RNE pack)."""

    def test_is_builtin_with_void_return(self):
        """The symbol is a builtin declared with a void eval return type."""
        self.assertTrue(getattr(cuda_asm.pack_2xbf16, BUILTIN_ATTR, False))
        self.assertTrue(hasattr(cuda_asm.pack_2xbf16, CODEGEN_FUNC_ATTR))
        eval_return_type = getattr(cuda_asm.pack_2xbf16, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(eval_return_type(), ll.void)

    def test_exported_via_language_namespace(self):
        """backend.json registration surfaces the symbol as ll.pack_2xbf16."""
        self.assertTrue(getattr(ll.pack_2xbf16, BUILTIN_ATTR, False))
        self.assertIs(ll.pack_2xbf16, cuda_asm.pack_2xbf16)

    def test_direct_codegen_emits_cvt_rn_bf16x2(self):
        """Direct codegen: single-instruction RNE pack writing through &out."""
        codegen_func = getattr(cuda_asm.pack_2xbf16, CODEGEN_FUNC_ATTR)
        result = codegen_func("lo", "hi", "p")
        self.assertIsInstance(result, Builtin)
        self.assertIn("cvt.rn.bf16x2.f32 %0, %2, %1;", result.body)
        self.assertIn('"=r"(*out)', result.body)
        self.assertIn('"f"(lo), "f"(hi)', result.body)
        self.assertIn("float lo, float hi, uint32_t* out", result.body)
        self.assertIn("__device__ __forceinline__ void pack_2xbf16_fn", result.body)
        self.assertIn("#ifndef LK_PACK_2XBF16_FN_DEFINED", result.body)
        self.assertEqual(result.includes, [])
        # void builtin: the call expression is the full statement, writing the
        # packed word through the address of the named out register.
        self.assertEqual("pack_2xbf16_fn(lo, hi, &p)", result.return_val)

    def test_call_raises_at_trace_time(self):
        """Like every builtin stub, direct Python calls are not evaluable."""
        with self.assertRaises(RuntimeError):
            cuda_asm.pack_2xbf16(1.0, 2.0, 3)

    def test_backend_json_registers_both_nv_backends(self):
        """sm90 and sm100 both export the symbol from the shared cuda_asm module."""
        expected = ("little_kernel.backends.nv.common.language.intrin."
                    "cuda_asm:pack_2xbf16")
        intrin_dir = Path(cuda_asm.__file__).resolve().parents[3]
        for backend in ("sm90", "sm100"):
            data = json.loads((intrin_dir / backend / "backend.json").read_text())
            exports = data["language_exports"]
            self.assertIn("pack_2xbf16", exports, f"{backend}/backend.json missing pack_2xbf16")
            self.assertEqual(expected, exports["pack_2xbf16"])

    def test_e2e_kernel_emits_guarded_single_definition(self):
        """End-to-end compile: two call sites share one #ifndef-guarded helper,
        and the packed registers feed a vectorized st.shared.v4 epilogue."""

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def pack_epilogue(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.uint32]) -> ll.void:
            tid: ll.int32 = ll.threadIdx_x()
            p0: ll.uint32 = 0
            p1: ll.uint32 = 0
            base: ll.int32 = tid * 4
            # two call sites exercise the #ifndef guard across repeat use
            ll.pack_2xbf16(x[base], x[base + 1], p0)
            ll.pack_2xbf16(x[base + 2], x[base + 3], p1)
            y[tid * 2] = p0
            y[tid * 2 + 1] = p1

        code = pack_epilogue.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertIn("cvt.rn.bf16x2.f32", code)
        self.assertIn("pack_2xbf16_fn(", code)
        # The helper is defined exactly once although called at two sites.
        self.assertEqual(code.count("void pack_2xbf16_fn("), 1)
        self.assertEqual(code.count("LK_PACK_2XBF16_FN_DEFINED"), 2)
        self.assertEqual(code.count("pack_2xbf16_fn(x["), 2)


if __name__ == "__main__":
    unittest.main()
