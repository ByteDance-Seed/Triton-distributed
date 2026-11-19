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
"""Typed float literal codegen: float32-inferred literals emit `f` suffix.

Legacy behavior emitted str(value) (a C++ double literal), whose usual
arithmetic conversions silently promote surrounding fp32 expressions to FP64
(1:64 FP64 rate on Hopper). The emission is type-directed: only constants the
type inferencer recorded as 32-bit FloatType gain the suffix; float64-typed or
untyped (missing all_types entry) constants keep the legacy double literal.
"""

import ast
import unittest

import little_kernel.language as ll
from little_kernel.codegen.codegen_base import CppEmitter
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.core.type_system import float32, float64


def _codegen(src: str, ctx=None) -> str:
    tree = ast.parse(src)
    ctx = {'ll': ll} if ctx is None else ctx
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    emitter = CppEmitter(ctx=ctx, all_types=inferencer.all_types)
    emitter.visit(tree)
    return emitter.get_code()


class TestFloatLiteralCodegen(unittest.TestCase):

    def test_float32_expression_literals_get_f_suffix(self):
        code = _codegen('''
def probe(x: ll.float32) -> ll.float32:
    ee: ll.float32 = x * x
    sig: ll.float32 = 1.0 / (1.0 + ee)
    return sig
''')
        self.assertIn("1.0f / (1.0f + ee)", code)
        self.assertNotIn("1.0 / (1.0 + ee)", code)

    def test_assignment_and_mixed_literals(self):
        code = _codegen('''
def probe(x: ll.float32) -> ll.float32:
    z: ll.float32 = 0.0
    half: ll.float32 = 0.5 * x + 2.5e-08
    return z + half
''')
        self.assertIn("float z = 0.0f;", code)
        self.assertIn("0.5f * x + 2.5e-08f", code)

    def test_missing_type_info_keeps_double_literal(self):
        # Emitter without inference results must fall back to the legacy
        # double literal (conservative, code still compiles).
        emitter = CppEmitter(ctx={'ll': ll}, all_types={})
        self.assertEqual(emitter.visit_Constant(ast.Constant(value=0.25)), "0.25")

    def test_float64_typed_constant_keeps_double_literal(self):
        emitter = CppEmitter(ctx={'ll': ll}, all_types={})
        node = ast.Constant(value=0.1)
        emitter.all_types[node] = float64
        self.assertEqual(emitter.visit_Constant(node), "0.1")

    def test_float32_special_values_unchanged(self):
        # inf/nan spellings are unaffected by the suffix path.
        emitter = CppEmitter(ctx={'ll': ll}, all_types={})
        node = ast.Constant(value=float("inf"))
        emitter.all_types[node] = float32
        self.assertEqual(emitter.visit_Constant(node), "INFINITY")
        node = ast.Constant(value=float("nan"))
        emitter.all_types[node] = float32
        self.assertEqual(emitter.visit_Constant(node), "NAN")


if __name__ == "__main__":
    unittest.main()
