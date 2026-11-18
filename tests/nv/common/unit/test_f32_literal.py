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
"""Opt-in fp32 literal wrapper: ll.f32(x) pins a subexpression to float32,
emitting a C++ float literal (`1.5f`) for literal args or an explicit cast
for non-literal args, so no silent FP64 promotion can occur. The legacy
inference-directed literal path (test_float_literal_codegen.py) is
untouched: bare literals still rely on a recorded 32-bit FloatType."""

import ast
import re
import unittest

import little_kernel.language as ll
from little_kernel.codegen.codegen_base import CppEmitter
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.language.builtin_base import EVAL_RETURN_TYPE_ATTR


def _codegen(src: str, ctx=None) -> str:
    tree = ast.parse(src)
    ctx = {'ll': ll} if ctx is None else ctx
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    emitter = CppEmitter(ctx=ctx, all_types=inferencer.all_types)
    emitter.visit(tree)
    return emitter.get_code()


class TestF32LiteralWrapper(unittest.TestCase):

    def test_exported_and_registered_as_builtin(self):
        # Surviving merged implementation: ll.f32 is a host-callable marker
        # constructor (language/intrin/scalar_literal.py); codegen
        # intercepts the call by identity and typing uses the
        # EVAL_RETURN_TYPE_ATTR protocol (same contract builtin sets).
        self.assertTrue(hasattr(ll, 'f32'))
        self.assertTrue(hasattr(ll, 'f64'))
        self.assertTrue(hasattr(ll.f32, EVAL_RETURN_TYPE_ATTR))
        self.assertTrue(hasattr(ll.f64, EVAL_RETURN_TYPE_ATTR))

    def test_decimal_literals_emit_f_suffix(self):
        code = _codegen('''
def probe(x: ll.float32) -> ll.float32:
    a: ll.float32 = ll.f32(1.5) * x + ll.f32(2) - ll.f32(-0.75) + ll.f32(1e-3)
    return a''')
        self.assertIn('1.5f', code)
        self.assertIn('2.0f', code)  # "2f" would be ill-formed
        self.assertIn('-0.75f', code)
        self.assertIn('0.001f', code)  # 1e-3 arrives pre-formatted via str()
        self.assertNotIn('ff', code)  # never double-suffixed

    def test_wrapped_expression_typed_float32(self):
        code = _codegen('''
def probe(x: ll.float32) -> ll.float32:
    a: ll.float32 = ll.f32(0.25)
    b: ll.float32 = a + ll.f32(1.0)
    return b''')
        self.assertRegex(code, r'\bfloat\s+a\b')
        self.assertRegex(code, r'\bfloat\s+b\b')

    def test_already_suffixed_arg_passes_through(self):
        # When inference records the argument Constant as float32, visit_
        # Constant emits "1.5f" before codegen_f32 sees it; the wrapper must
        # not append a second suffix.
        code = _codegen('''
def probe(x: ll.float32) -> ll.float32:
    sig: ll.float32 = ll.f32(1.0) / (ll.f32(1.0) + x)
    return sig''')
        self.assertIn('1.0f', code)
        self.assertNotIn('1.0ff', code)

    def test_non_literal_expression_gets_explicit_cast(self):
        # Surviving merged implementation lowers non-constant arguments to
        # a narrowing static_cast (numerically identical to (float)(x)).
        code = _codegen('''
def probe(x: ll.float64) -> ll.float32:
    a: ll.float32 = ll.f32(x)
    return a''')
        self.assertIn('static_cast<float>(x)', code)
        self.assertRegex(code, r'\bfloat\s+a\b')

    def test_legacy_bare_literal_path_unchanged(self):
        # Emitter-level guard (mirrors test_float_literal_codegen.py):
        # missing type info and float64-typed constants keep the legacy
        # double literal; float32-typed keeps the inference-directed suffix.
        emitter = CppEmitter(ctx={'ll': ll}, all_types={})
        self.assertEqual(emitter.visit_Constant(ast.Constant(value=0.25)), "0.25")
        node = ast.Constant(value=0.1)
        emitter.all_types[node] = ll.float64
        self.assertEqual(emitter.visit_Constant(node), "0.1")
        node32 = ast.Constant(value=0.1)
        emitter.all_types[node32] = ll.float32
        self.assertEqual(emitter.visit_Constant(node32), "0.1f")

    def test_python_const_invocation_is_idempotent(self):
        # Constant folders may invoke the stub with Python constants; the
        # result compares equal to a plain float. Non-numeric arguments
        # raise TypeError (same legality contract as C++ user literals).
        self.assertEqual(ll.f32(1.5), 1.5)
        self.assertEqual(ll.f32(2), 2.0)
        with self.assertRaises(TypeError):
            ll.f32("x")


if __name__ == '__main__':
    unittest.main()
