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
# publish, distribute, sublicense, and/or sell copies of the Software.
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
"""
Golden text tests for the explicit-width float literal constructors
(ll.f32 / ll.f64) and the LK_WARN_FLOAT_LITERAL legacy-literal probe.

Contract under test (codegen/visitors/{call,expression}_codegen.py +
language/intrin/scalar_literal.py):

- ll.f32(const) emits an f-suffixed parenthesized C++ float literal;
  ll.f64(const) emits a parenthesized C++ double literal.
- Non-constant arguments lower to static_cast<float/double>(...).
- ctx-folded module-scope markers (LOG2E = ll.f32(...)) emit the pinned
  literal at every use site.
- Bare floats follow the candidate base's inference-directed rule: a
  float32-typed constant gains the f suffix, missing/float64 type info
  keeps the legacy double literal; the opt-in LK_WARN_FLOAT_LITERAL=1
  probe still warns on unpinned bare floats (and only on them).
- Legality: non-numeric/bool arguments raise TypeError; wrong arity raises
  ValueError (codegen) / TypeError (host call).

All assertions are on output text only (deterministic, no GPU).
"""

import ast
import os
import unittest
import warnings

import little_kernel.language as ll
from little_kernel.codegen.codegen_base import CppEmitter
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.language.intrin.scalar_literal import _F32Literal, _F64Literal, format_f32_constant, format_f64_constant
from little_kernel.language.builtin_base import EVAL_RETURN_TYPE_ATTR


def emit_expr(code, ctx=None, scope_vars=None):
    """Type-infer and emit a single expression, returning the C++ text."""
    full_ctx = {'ll': ll}
    if ctx:
        full_ctx.update(ctx)
    tree = ast.parse(code, mode='eval')
    inferencer = TypeInferencer(ctx=full_ctx, scope_vars=scope_vars or {})
    inferencer.visit(tree)
    emitter = CppEmitter(ctx=full_ctx, all_types=inferencer.all_types)
    return emitter.visit(tree)


class TestScalarLiteralConstructors(unittest.TestCase):
    """Host-side constructor contracts."""

    def test_f32_returns_float_subclass_with_same_value(self):
        v = ll.f32(1.5)
        self.assertIsInstance(v, float)
        self.assertIsInstance(v, _F32Literal)
        self.assertEqual(v, 1.5)

    def test_f64_returns_float_subclass_with_same_value(self):
        v = ll.f64(1.5)
        self.assertIsInstance(v, float)
        self.assertIsInstance(v, _F64Literal)
        self.assertEqual(v, 1.5)

    def test_marker_does_not_escape_host_arithmetic(self):
        # Marker is a codegen-only signal: host math must stay plain float.
        v = ll.f32(1.5) + ll.f32(2.5)
        self.assertEqual(type(v), float)

    def test_int_argument_is_accepted(self):
        self.assertEqual(ll.f32(3), 3.0)
        self.assertEqual(ll.f64(0), 0.0)

    def test_bool_argument_rejected(self):
        with self.assertRaises(TypeError):
            ll.f32(True)
        with self.assertRaises(TypeError):
            ll.f64(False)

    def test_non_numeric_argument_rejected(self):
        with self.assertRaises(TypeError):
            ll.f32("1.5")
        with self.assertRaises(TypeError):
            ll.f32(None)

    def test_eval_return_type_attributes(self):
        self.assertIs(getattr(ll.f32, EVAL_RETURN_TYPE_ATTR)(), ll.float32)
        self.assertIs(getattr(ll.f64, EVAL_RETURN_TYPE_ATTR)(), ll.float64)


class TestScalarLiteralFormatting(unittest.TestCase):
    """Emission helper contracts."""

    def test_f32_format(self):
        self.assertEqual(format_f32_constant(1.5), "(1.5f)")
        self.assertEqual(format_f32_constant(-87.0), "(-87.0f)")
        self.assertEqual(format_f32_constant(1.4426950408889634), "(1.4426950408889634f)")
        self.assertEqual(format_f32_constant(3.0), "(3.0f)")

    def test_f32_format_special_values(self):
        self.assertEqual(format_f32_constant(float("inf")), "((float)INFINITY)")
        self.assertEqual(format_f32_constant(float("-inf")), "(-(float)INFINITY)")
        self.assertEqual(format_f32_constant(float("nan")), "((float)NAN)")

    def test_f64_format_matches_legacy_numeric_text(self):
        # Legacy bare-float emission is str(value); f64 pins the same
        # double value, parenthesized for expression safety.
        for v in (1.5, -87.0, 1.4426950408889634, 3.0):
            self.assertEqual(format_f64_constant(v), f"({str(v)})")

    def test_f64_format_special_values(self):
        self.assertEqual(format_f64_constant(float("inf")), "(INFINITY)")
        self.assertEqual(format_f64_constant(float("-inf")), "(-INFINITY)")
        self.assertEqual(format_f64_constant(float("nan")), "(NAN)")


class TestScalarLiteralCodegen(unittest.TestCase):
    """Golden text tests through type inference + CppEmitter."""

    def test_inline_f32_constant(self):
        result = emit_expr("x * ll.f32(1.5)", scope_vars={'x': ll.float32})
        self.assertIn("(1.5f)", result)
        self.assertNotIn("1.5)", result)  # no double-literal residue

    def test_inline_f64_constant(self):
        result = emit_expr("x * ll.f64(1.5)", scope_vars={'x': ll.float64})
        self.assertIn("(1.5)", result)
        self.assertNotIn("1.5f", result)

    def test_f32_negative_constant_folds(self):
        # ll.f32(-87.0): ast.literal_eval folds the unary minus.
        result = emit_expr("ll.f32(-87.0)", scope_vars={})
        self.assertIn("(-87.0f)", result)

    def test_f32_binop_of_constants_is_static_cast_not_literal(self):
        # ast.literal_eval deliberately does not evaluate BinOps; a
        # constant *expression* takes the narrowing static_cast path.
        # Candidate-base contract: the inference-directed float32 suffix
        # also applies to the bare 0.0 inside the argument (float32-typed
        # constant), consistent with visit_Constant's default emission.
        result = emit_expr("ll.f32(0.0 - x)", scope_vars={'x': ll.float64})
        self.assertEqual(result, "static_cast<float>(0.0f - x)")

    def test_f32_int_constant(self):
        result = emit_expr("ll.f32(3)", scope_vars={})
        self.assertIn("(3.0f)", result)

    def test_f32_nonconstant_lowers_to_static_cast(self):
        result = emit_expr("ll.f32(x)", scope_vars={'x': ll.float64})
        self.assertEqual(result, "static_cast<float>(x)")

    def test_f64_nonconstant_lowers_to_static_cast(self):
        result = emit_expr("ll.f64(x)", scope_vars={'x': ll.float32})
        self.assertEqual(result, "static_cast<double>(x)")

    def test_f32_nested_in_expression(self):
        # The sigmoid-clamp shape from the requesting frontends:
        # scale / (1.0 + exp2f(0.0 - x * LOG2E)) with pinned fp32 literals.
        result = emit_expr("scale / (ll.f32(1.0) + ll.exp2f(ll.f32(0.0) - x * ll.f32("
                           "1.4426950408889634)))", scope_vars={'scale': ll.float32, 'x': ll.float32})
        self.assertIn("(1.0f)", result)
        self.assertIn("(0.0f)", result)
        self.assertIn("(1.4426950408889634f)", result)

    def test_ctxfolded_module_scope_marker(self):
        # LOG2E = ll.f32(...) at module scope: the use site folds the
        # marker out of ctx and emits the pinned literal.
        ctx = {'LOG2E': ll.f32(1.4426950408889634)}
        result = emit_expr("x * LOG2E", ctx=ctx, scope_vars={'x': ll.float32})
        self.assertIn("(1.4426950408889634f)", result)

    def test_ctxfolded_f64_marker(self):
        ctx = {'K': ll.f64(0.25)}
        result = emit_expr("x * K", ctx=ctx, scope_vars={'x': ll.float64})
        self.assertIn("(0.25)", result)
        self.assertNotIn("0.25f", result)

    def test_legacy_bare_float_unchanged(self):
        # Candidate-base contract (supersedes the v009-lineage
        # zero-default-change variant this check arrived with): a bare
        # float literal inferred as float32 is emitted with the f suffix,
        # keeping the expression on the fp32 pipe. Untyped contexts keep
        # the legacy double literal (see test_float_literal_codegen.py).
        result = emit_expr("x * 1.5", scope_vars={'x': ll.float32})
        self.assertIn("1.5f", result)
        self.assertNotIn("(1.5)", result)  # no f64-style parenthesization

    def test_ctxfolded_plain_float_unchanged(self):
        # Candidate-base contract: ctx-folded plain floats follow the same
        # inference-directed suffix rule as inline literals (see
        # visit_Name's ctx-folding branch in expression_codegen.py).
        ctx = {'K': 0.25}
        result = emit_expr("x * K", ctx=ctx, scope_vars={'x': ll.float32})
        self.assertEqual(result, "x * 0.25f")

    def test_f32_wrong_arity_rejected(self):
        with self.assertRaises(ValueError):
            emit_expr("ll.f32(1.0, 2.0)", scope_vars={})

    def test_f32_string_constant_rejected(self):
        # Codegen intercepts the call before the host-side TypeError can
        # fire, so it must reject non-numeric constants itself.
        with self.assertRaises(TypeError):
            emit_expr('ll.f32("1.5")', scope_vars={})

    def test_f32_none_constant_rejected(self):
        with self.assertRaises(TypeError):
            emit_expr("ll.f32(None)", scope_vars={})

    def test_f32_bool_constant_rejected(self):
        with self.assertRaises(TypeError):
            emit_expr("ll.f32(True)", scope_vars={})


class TestWarnBareFloatLiteralProbe(unittest.TestCase):
    """Opt-in LK_WARN_FLOAT_LITERAL=1 advisory probe."""

    def setUp(self):
        self._old = os.environ.get("LK_WARN_FLOAT_LITERAL")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("LK_WARN_FLOAT_LITERAL", None)
        else:
            os.environ["LK_WARN_FLOAT_LITERAL"] = self._old

    def test_probe_off_by_default(self):
        os.environ.pop("LK_WARN_FLOAT_LITERAL", None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            emit_expr("x * 1.5", scope_vars={'x': ll.float32})
        self.assertEqual(caught, [])

    def test_probe_warns_on_bare_float_constant(self):
        os.environ["LK_WARN_FLOAT_LITERAL"] = "1"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            emit_expr("x * 1.5", scope_vars={'x': ll.float32})
        self.assertTrue(any("float literal" in str(w.message) for w in caught))

    def test_probe_warns_on_ctxfolded_plain_float(self):
        os.environ["LK_WARN_FLOAT_LITERAL"] = "1"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            emit_expr("x * K", ctx={'K': 0.25}, scope_vars={'x': ll.float32})
        self.assertTrue(any("float literal" in str(w.message) for w in caught))

    def test_probe_silent_for_pinned_literals(self):
        os.environ["LK_WARN_FLOAT_LITERAL"] = "1"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            emit_expr("x * ll.f32(1.5) * ll.f64(2.5)", scope_vars={'x': ll.float32})
            emit_expr("x * LOG2E", ctx={'LOG2E': ll.f32(1.4426950408889634)}, scope_vars={'x': ll.float32})
        self.assertEqual(caught, [])

    def test_probe_silent_for_int_constants(self):
        os.environ["LK_WARN_FLOAT_LITERAL"] = "1"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            emit_expr("x * 3 + 1", scope_vars={'x': ll.float32})
        self.assertEqual(caught, [])


if __name__ == '__main__':
    unittest.main()
