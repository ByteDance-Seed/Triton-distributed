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
# (the "Software"), to deal in the Software without restriction
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
"""Tests for type-directed float32 literal emission (f suffix).

LittleKernel's type inference types every Python float literal as float32
(ConstantTypeInferencer), but the C emitter used to print bare `str(value)`
literals - C *double* constants - which silently promote fp32 arithmetic to
FP64 under the usual arithmetic conversions. When a float constant node is
inferred as float32, codegen must emit an `f` suffix so the generated C
matches the compiler's own type system.
"""

import unittest
import ast

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(source: str) -> str:
    tree = ast.parse(source)
    return codegen_cuda(tree, ctx={"ll": ll}, emit_header=False)


class TestFloatLiteralSuffix(unittest.TestCase):

    def test_fp32_expression_gets_f_suffix(self):
        """Float constants in typed fp32 kernel expressions emit as C floats."""
        code = _emit("""
def kern(x: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        v: ll.float32 = x[i]
        x[i] = v * 0.125 + 8.0
""")
        self.assertIn("0.125f", code)
        self.assertIn("8.0f", code)
        self.assertNotIn("v * 0.125 + 8.0", code)

    def test_annotated_scalar_constant_gets_f_suffix(self):
        """Named ll.float32 constants are declared from float literals."""
        code = _emit("""
def kern(x: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    f8c: ll.float32 = 8.0
    i = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        x[i] = x[i] * f8c
""")
        self.assertIn("8.0f", code)

    def test_int_literals_untouched(self):
        """Integer constants are not suffixed."""
        code = _emit("""
def kern(x: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        x[i] = x[i] + 3
""")
        self.assertIn("+ 3", code)
        self.assertNotIn("3f", code)

    def test_special_values_untouched(self):
        """inf keeps macro emission; exponent-form literals stay valid C."""
        code = _emit("""
def kern(x: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    small: ll.float32 = 1e-05
    i = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        v: ll.float32 = x[i]
        x[i] = v + small
        if v > 1e1000:
            x[i] = v
""")
        self.assertIn("1e-05f", code)
        self.assertIn("INFINITY", code)
        self.assertNotIn("INFINITYf", code)


class TestRcpF32Export(unittest.TestCase):

    def test_rcp_f32_exported(self):
        """rcp_f32 (SFU reciprocal) is exported from little_kernel.language."""
        self.assertTrue(hasattr(ll, "rcp_f32"))


if __name__ == "__main__":
    unittest.main()
