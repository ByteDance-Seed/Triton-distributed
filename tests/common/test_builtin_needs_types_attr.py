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
"""CPU-only regression tests for the ``__codegen_needs_types__`` opt-in.

``CODEGEN_NEEDS_TYPES_ATTR`` marks a builtin whose codegen function must
receive the per-argument LLType list via the ``_arg_dtypes`` keyword (the
emitter reads the flag in ``call_codegen._handle_builtin_with_codegen``).
The class-decorator path (``_make_builtin_from_class``) always propagated the
flag from a ``BuiltinOp`` subclass, but the function-decorator path
(``builtin._builtin``) copied only the attributes listed in ``INTERNAL_ATTR``
— which omitted it — so a function that set ``__codegen_needs_types__ = True``
before ``@builtin`` silently lost the opt-in and fell back to untyped codegen.
These tests pin both paths to the same contract.  No backend or GPU is
required.
"""

import ast
import unittest

import little_kernel.language as ll
from little_kernel.codegen.codegen_base import CppEmitter
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.core.type_system import float32, float64, int32
from little_kernel.language.builtin_base import (
    BUILTIN_ATTR,
    CODEGEN_FUNC_ATTR,
    CODEGEN_NEEDS_TYPES_ATTR,
    INTERNAL_ATTR,
    Builtin,
    BuiltinOp,
    builtin,
)


def _emit(src: str, ctx) -> str:
    """Run the real type inferencer + C++ emitter on a parsed function."""
    tree = ast.parse(src)
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    emitter = CppEmitter(ctx=ctx, all_types=inferencer.all_types)
    emitter.visit(tree)
    return emitter.get_code()


class TestInternalAttrCoversNeedsTypes(unittest.TestCase):
    """The decorator copy-loop contract must include the opt-in attribute."""

    def test_needs_types_attr_is_in_internal_attr(self):
        self.assertIn(CODEGEN_NEEDS_TYPES_ATTR, INTERNAL_ATTR)


class TestFunctionDecoratorPath(unittest.TestCase):
    """A flag set on the plain function must survive ``@builtin``."""

    def _make_flagged_builtin(self, codegen):

        def convert(a):
            pass

        setattr(convert, CODEGEN_NEEDS_TYPES_ATTR, True)
        return builtin(eval_return_type=float32, codegen_func=codegen)(convert)

    def test_flag_preserved_after_decoration(self):

        def codegen(a, _arg_dtypes=None):
            return Builtin(body="", includes=[], return_val=str(a))

        wrapper = self._make_flagged_builtin(codegen)
        self.assertTrue(getattr(wrapper, BUILTIN_ATTR, False))
        self.assertTrue(getattr(wrapper, CODEGEN_NEEDS_TYPES_ATTR, False))

    def test_flag_absent_when_not_set(self):
        # Without the opt-in the wrapper must not claim the typed channel:
        # the emitter keys strictly on this attribute being truthy.
        def codegen(a):
            return Builtin(body="", includes=[], return_val=str(a))

        def plain(a):
            pass

        wrapper = builtin(eval_return_type=float32, codegen_func=codegen)(plain)
        self.assertFalse(getattr(wrapper, CODEGEN_NEEDS_TYPES_ATTR, False))


class TestClassDecoratorPath(unittest.TestCase):
    """The ``BuiltinOp`` path must keep honoring the same opt-in."""

    def test_class_flag_propagates_to_wrapper(self):

        @builtin
        class TypedOp(BuiltinOp):
            __codegen_needs_types__ = True

            @staticmethod
            def return_type(a):
                return float32

            @staticmethod
            def codegen(a, _arg_dtypes=None):
                return Builtin(body="", includes=[], return_val=str(a))

        self.assertTrue(getattr(TypedOp, CODEGEN_NEEDS_TYPES_ATTR, False))

    def test_class_without_flag_stays_untyped(self):

        @builtin
        class PlainOp(BuiltinOp):

            @staticmethod
            def return_type(a):
                return float32

            @staticmethod
            def codegen(a):
                return Builtin(body="", includes=[], return_val=str(a))

        self.assertFalse(getattr(PlainOp, CODEGEN_NEEDS_TYPES_ATTR, False))


class TestEmitterDeliversArgDtypes(unittest.TestCase):
    """Through the real emitter path, the codegen receives ``_arg_dtypes``."""

    def test_function_builtin_receives_arg_dtypes(self):
        captured = {}

        def codegen(a, _arg_dtypes=None):
            captured["dtypes"] = _arg_dtypes
            return Builtin(body="", includes=[], return_val=f"cvt({a})")

        def convert(a):
            pass

        setattr(convert, CODEGEN_NEEDS_TYPES_ATTR, True)
        convert = builtin(eval_return_type=float32, codegen_func=codegen)(convert)

        code = _emit(
            """
def probe(x: ll.float32) -> ll.float32:
    y: ll.float32 = convert(x)
    return y
""",
            {"ll": ll, "convert": convert},
        )
        self.assertIn("cvt(x)", code)
        self.assertEqual(captured.get("dtypes"), [float32])

    def test_unflagged_builtin_does_not_receive_arg_dtypes(self):
        calls = []

        def codegen(a, **kwargs):
            calls.append(kwargs)
            return Builtin(body="", includes=[], return_val=f"nop({a})")

        def nop(a):
            pass

        nop = builtin(eval_return_type=float32, codegen_func=codegen)(nop)

        code = _emit(
            """
def probe(x: ll.float32) -> ll.float32:
    y: ll.float32 = nop(x)
    return y
""",
            {"ll": ll, "nop": nop},
        )
        self.assertIn("nop(x)", code)
        self.assertEqual(calls, [{}])

    def test_dtypes_follow_positional_arguments(self):
        captured = {}

        def codegen(x, y, _arg_dtypes=None):
            captured["dtypes"] = _arg_dtypes
            return Builtin(body="", includes=[], return_val=f"mix({x}, {y})")

        def mix_op(x, y):
            pass

        def ret(x, y):
            return float32

        setattr(mix_op, CODEGEN_NEEDS_TYPES_ATTR, True)
        mix_op = builtin(eval_return_type=ret, codegen_func=codegen)(mix_op)

        code = _emit(
            """
def probe(a: ll.int32, b: ll.float64) -> ll.float32:
    r: ll.float32 = mix_op(a, b)
    return r
""",
            {"ll": ll, "mix_op": mix_op},
        )
        self.assertIn("mix(a, b)", code)
        self.assertEqual(captured.get("dtypes"), [int32, float64])


if __name__ == "__main__":
    unittest.main()
