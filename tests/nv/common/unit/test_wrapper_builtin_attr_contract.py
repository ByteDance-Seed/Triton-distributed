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
"""Regression tests for the BUILTIN_ATTR contract of the helper decorators.

Contract under test (language/builtin_base.py): ``const_func``,
``const_type_func`` and ``inline_func`` only ever *propagate* an existing
``BUILTIN_ATTR`` from an already-@builtin source; they never fabricate one for
a plain function. Pipeline stages branch on ``getattr(f, BUILTIN_ATTR, False)``
to choose between builtin-codegen and folded/inline call handling, so a
codegen-less wrapper must stay a non-builtin while a wrapped builtin keeps its
marker. CPU-only; no hardware required.
"""

import unittest

from little_kernel.language.builtin_base import (
    BUILTIN_ATTR,
    CONST_FUNC_ATTR,
    CONST_TYPE_FUNC_ATTR,
    INLINE_FUNC_ATTR,
    ORIGINAL_PY_FUNC_ATTR,
    builtin,
    Builtin,
    BuiltinOp,
    const_func,
    const_type_func,
    inline_func,
)
from little_kernel.core.type_system import int32
from little_kernel.language.intrin.dtype import sizeof, typeof
from little_kernel.backends.nv.common.atom.scheduler import get_tile_coord_along_m


def _plain(a, b):
    return a + b


class TestConstFuncContract(unittest.TestCase):
    """const_func: const marker set; builtin marker only propagated, never invented."""

    def test_plain_function_is_const_but_not_builtin(self):
        wrapped = const_func(_plain)
        self.assertTrue(getattr(wrapped, CONST_FUNC_ATTR, False))
        # The crux of the defect: a plain helper must NOT look like a builtin.
        self.assertFalse(getattr(wrapped, BUILTIN_ATTR, False))

    def test_plain_function_carries_no_codegen(self):
        # A codegen-less const helper must not be routed to builtin codegen.
        wrapped = const_func(_plain)
        from little_kernel.language.builtin_base import CODEGEN_FUNC_ATTR
        self.assertFalse(hasattr(wrapped, CODEGEN_FUNC_ATTR))

    def test_already_builtin_keeps_builtin_attr(self):
        # sizeof is @builtin(BuiltinOp) wrapped by const_func; marker preserved.
        self.assertTrue(getattr(sizeof, BUILTIN_ATTR, False))
        self.assertTrue(getattr(sizeof, CONST_FUNC_ATTR, False))

    def test_manual_builtin_then_const_func_preserves_builtin(self):

        @builtin(eval_return_type=int32, codegen_func=lambda x: Builtin(body="", includes=[], return_val=x))
        def myinc(x):
            return x + 1

        self.assertTrue(getattr(myinc, BUILTIN_ATTR, False))
        wrapped = const_func(myinc)
        self.assertTrue(getattr(wrapped, BUILTIN_ATTR, False))
        self.assertTrue(getattr(wrapped, CONST_FUNC_ATTR, False))

    def test_original_py_func_points_at_source(self):
        wrapped = const_func(_plain)
        self.assertIs(getattr(wrapped, ORIGINAL_PY_FUNC_ATTR), _plain)

    def test_callable_still_invokes_source(self):
        wrapped = const_func(_plain)
        self.assertEqual(wrapped(2, 3), 5)


class TestConstTypeFuncContract(unittest.TestCase):
    """const_type_func: type-const marker set; builtin marker propagated only."""

    def test_plain_function_is_typeconst_but_not_builtin(self):
        wrapped = const_type_func(_plain)
        self.assertTrue(getattr(wrapped, CONST_TYPE_FUNC_ATTR, False))
        self.assertFalse(getattr(wrapped, BUILTIN_ATTR, False))

    def test_already_builtin_keeps_builtin_attr(self):
        # typeof is @builtin(BuiltinOp) wrapped by const_type_func.
        self.assertTrue(getattr(typeof, BUILTIN_ATTR, False))
        self.assertTrue(getattr(typeof, CONST_TYPE_FUNC_ATTR, False))


class TestInlineFuncContract(unittest.TestCase):
    """inline_func: inline marker set; builtin marker propagated only."""

    def test_plain_function_is_inline_but_not_builtin(self):

        @inline_func
        def local(a, b):
            return a * b

        self.assertTrue(getattr(local, INLINE_FUNC_ATTR, False))
        self.assertFalse(getattr(local, BUILTIN_ATTR, False))

    def test_real_inline_helper_is_not_builtin(self):
        # get_tile_coord_along_m is a production @inline_func plain function.
        self.assertTrue(getattr(get_tile_coord_along_m, INLINE_FUNC_ATTR, False))
        self.assertFalse(getattr(get_tile_coord_along_m, BUILTIN_ATTR, False))


class TestCrossDecoratorAsymmetry(unittest.TestCase):
    """All three decorators must agree on the no-fabrication contract."""

    def test_none_of_the_decorators_fabricate_builtin(self):
        for dec, marker in (
            (const_func, CONST_FUNC_ATTR),
            (const_type_func, CONST_TYPE_FUNC_ATTR),
            (inline_func, INLINE_FUNC_ATTR),
        ):
            with self.subTest(decorator=dec.__name__):
                wrapped = dec(_plain)
                self.assertFalse(getattr(wrapped, BUILTIN_ATTR, False),
                                 f"{dec.__name__} must not fabricate BUILTIN_ATTR")
                self.assertTrue(getattr(wrapped, marker, False))

    def test_all_preserve_builtin_from_builtin_source(self):
        for dec in (const_func, const_type_func, inline_func):
            with self.subTest(decorator=dec.__name__):

                @builtin(eval_return_type=int32, codegen_func=lambda x: Builtin(body="", includes=[], return_val=x))
                def src(x):
                    return x

                self.assertTrue(getattr(dec(src), BUILTIN_ATTR, False))


if __name__ == "__main__":
    unittest.main()
