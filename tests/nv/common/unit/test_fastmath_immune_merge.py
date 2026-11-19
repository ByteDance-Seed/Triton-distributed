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
"""CPU regression tests for the conflict-16 union merge (incoming 5f011c8b).

Both merged branches inserted fast-math-immune scalar builtins after ``erff``
in ``language/intrin/cuda_asm.py``:

- ours: the ``*_rn`` family (``expf_rn`` ... ``fmaf_rn``), the packed fp8
  SATFINITE converters, and the libm-exact ``exp_f32``/``log1p_f32``/``pow_f32``;
- theirs: ``exp_precise_f32``.

``exp_precise_f32`` and ``expf_rn`` lower to the same libdevice ``__nv_expf``
semantics.  The merge keeps both names verbatim so frontends authored against
either branch keep compiling unchanged.

Small-frontend / perturbed-input evidence (CPU codegen, no GPU):

    # frontend kernel body (perturbed operands, e.g. softmax logits x_hat):
    p = ll.exp_precise_f32(x_hat[i] + eps)      # theirs-authored frontend
    q = ll.expf_rn(x_hat[i] - x_max)            # ours-authored frontend

expected IR/codegen: the same ``Builtin`` IR node kind for both, one calling
the named helper ``exp_precise_f32_fn(...)`` (whose body binds
``__nv_expf``), the other an inline ``__nv_expf(...)`` call with a bare
extern-"C" prototype as its helper body.  Legality of co-emission in one
translation unit: both sides may emit the identical line
``extern "C" __device__ float __nv_expf(float);``; duplicate identical
extern-"C" declarations are legal C++, and the body-cache symbols
(``exp_precise_f32_fn`` vs. the inline declaration) cannot collide.
Fallback: none needed — the names are additive aliases, not overloads.
"""

import unittest

import little_kernel.language as ll
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR
import little_kernel.backends.nv.common.language.intrin.cuda_asm as cuda_asm

EXTERN_PROTO = 'extern "C" __device__ float __nv_expf(float);'


class TestMergedExpAliases(unittest.TestCase):

    def test_both_names_exported_from_language(self):
        """Union merge: both branch names are reachable from `little_kernel.language`."""
        self.assertIs(ll.exp_precise_f32, cuda_asm.exp_precise_f32)
        self.assertIs(ll.expf_rn, cuda_asm.expf_rn)
        for name in ("logf_rn", "log1pf_rn", "expm1f_rn", "sinf_rn", "sincosf_rn", "fadd_rn", "fsub_rn", "fmul_rn",
                     "fdiv_rn", "fmaf_rn"):
            self.assertTrue(hasattr(ll, name), name)

    def test_coemission_on_perturbed_inputs(self):
        """Small frontend with perturbed inputs: both lower to __nv_expf semantics.

        Simulates one frontend kernel body computing
        ``exp_precise_f32("x_hat_i + eps")`` and ``expf_rn("x_hat_i - x_max")``
        (perturbed operands of the same logits tensor, as in a numerically
        stabilized softmax) and checks the expected codegen of both.
        """
        theirs = getattr(cuda_asm.exp_precise_f32, CODEGEN_FUNC_ATTR)("x_hat_i + eps")
        ours = getattr(cuda_asm.expf_rn, CODEGEN_FUNC_ATTR)("x_hat_i - x_max")

        # theirs: named-helper emission, helper body binds libdevice __nv_expf.
        self.assertEqual(theirs.return_val, "exp_precise_f32_fn(x_hat_i + eps)")
        self.assertIn(EXTERN_PROTO, theirs.body)
        self.assertIn("exp_precise_f32_fn", theirs.body)

        # ours: inline libdevice call, body is the bare extern-"C" prototype.
        self.assertEqual(ours.return_val, "__nv_expf(x_hat_i - x_max)")
        self.assertEqual(ours.body.strip(), EXTERN_PROTO)

        # Legality invariant for one translation unit: both bodies declare the
        # *identical* extern-"C" prototype (duplicate identical extern-"C"
        # declarations are legal C++), and the helper symbol
        # ``exp_precise_f32_fn`` appears only in theirs' body, so the body
        # cache cannot produce a redefinition.
        self.assertIn(EXTERN_PROTO, theirs.body)
        self.assertIn(EXTERN_PROTO, ours.body)
        self.assertNotIn("exp_precise_f32_fn(", ours.body)

    def test_no_task_name_dispatch(self):
        """Registry discipline: codegen is generic, no task/tensor-name matching."""
        for fn in (cuda_asm.exp_precise_f32, cuda_asm.expf_rn):
            body = getattr(fn, CODEGEN_FUNC_ATTR)("v").body + getattr(fn, CODEGEN_FUNC_ATTR)("v").return_val
            for probe in ("softmax", "lstm", "task", "case-"):
                self.assertNotIn(probe, body)

    def test_eval_return_types_float32(self):
        for fn in (cuda_asm.exp_precise_f32, cuda_asm.expf_rn):
            rt = getattr(fn, EVAL_RETURN_TYPE_ATTR)
            self.assertIsNotNone(rt)
            self.assertIs(rt(ll.Tensor[ll.float32]), ll.float32)


if __name__ == "__main__":
    unittest.main()
