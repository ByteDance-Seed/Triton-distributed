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
"""CPU regression tests for the correctly-rounded fp64 division intrinsic
(semantic key ir.scalar.arithmetic.div_rn_f64).

``div_rn_f32`` already pinned IEEE round-to-nearest-even float32 division
(emitting ``__fdiv_rn``, immune to ``--use_fast_math``); float64 needs the
same guarantee because bitwise float32 threshold reproduction in reduction
kernels (``mean_t = float32(sum_f64 / (double)N)``) breaks if the fp64
division is approximated or rewritten as ``sum * (1.0/N)`` (reciprocal-
rounded, not bit-identical).

Coverage:
  * ``div_rn_f64`` is a registered builtin, exported from the ``ll`` facade,
    with ``eval_return_type=float64``;
  * its codegen emits ``__ddiv_rn(a, b)`` (the device intrinsic nvcc never
    rewrites under ``--use_fast_math``);
  * it renders through the CUDA emitter on the task reproducer shape;
  * plain fp64 ``/`` is left untouched (only the explicit intrinsic gets the
    RN spelling — no blanket operator rewriting).

CPU-only: drives type inference + the CUDA emitter directly, no nvcc/GPU.
Paired GPU validation (correct rounding + ``--use_fast_math`` build) is left
to the controller queue.
"""

import ast
import unittest

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import CudaEmitter
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR
from little_kernel.language.builtin_base import Builtin
import little_kernel.backends.nv.common.language.intrin.arith as arith


def _codegen(src: str) -> str:
    tree = ast.parse(src)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ctx = {"ll": ll}
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    emitter = CudaEmitter(ctx=ctx, all_types=inferencer.all_types)
    emitter.visit(tree)
    return emitter.get_code()


KERNEL = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def k(total: ll.Tensor[ll.float64], out: ll.Tensor[ll.float32], dn: ll.float64) -> ll.void:
    s: ll.float64 = total[0]
    q: ll.float64 = ll.div_rn_f64(s, dn)
    p: ll.float64 = s / dn
    out[0] = ll.val_cast(q, ll.float32)
'''


class DivRnF64RegistryTests(unittest.TestCase):

    def test_registered_and_exported(self):
        fn = arith.div_rn_f64
        self.assertTrue(getattr(fn, BUILTIN_ATTR, False))
        self.assertIs(ll.div_rn_f64, fn)

    def test_eval_return_type_float64(self):
        rt = getattr(arith.div_rn_f64, EVAL_RETURN_TYPE_ATTR)
        self.assertIsNotNone(rt)
        self.assertIs(rt(ll.float64, ll.float64), ll.float64)
        # RN f64 division cannot silently widen/narrow: f64 operands only.
        self.assertIs(rt(ll.float32, ll.float32), ll.float64)

    def test_codegen_emits_ddiv_rn(self):
        result = getattr(arith.div_rn_f64, CODEGEN_FUNC_ATTR)("a", "b")
        self.assertIsInstance(result, Builtin)
        self.assertEqual(result.return_val, "__ddiv_rn(a, b)")
        # Mirror the f32 twin: pure inline call, no helper body/includes.
        twin = getattr(arith.div_rn_f32, CODEGEN_FUNC_ATTR)("a", "b")
        self.assertEqual(result.body, twin.body)
        self.assertEqual(result.includes, twin.includes)

    def test_no_task_name_dispatch(self):
        """Registry discipline: codegen is generic, no task/tensor matching."""
        result = getattr(arith.div_rn_f64, CODEGEN_FUNC_ATTR)("v", "w")
        body = result.body + result.return_val
        for probe in ("task", "case-", "reduction", "mean"):
            self.assertNotIn(probe, body)


class DivRnF64CudaCodegenTests(unittest.TestCase):

    def test_intrinsic_renders_and_plain_div_untouched(self):
        cuda = _codegen(KERNEL)
        # The explicit intrinsic gets the IEEE RN device spelling.
        self.assertIn("__ddiv_rn(s, dn)", cuda)
        # Plain fp64 `/` is not rewritten to the intrinsic nor to a multiply
        # by a reciprocal: it keeps the ordinary C++ division operator.
        self.assertIn("s / dn", cuda)
        self.assertNotIn("1.0 / dn", cuda)
        self.assertNotIn("1.0/dn", cuda)
        # Exactly one RN division: the plain operator line stays ordinary.
        self.assertEqual(cuda.count("__ddiv_rn("), 1)


if __name__ == "__main__":
    unittest.main()
