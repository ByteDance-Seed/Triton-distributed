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
"""CPU-only regression tests for portable codegen operand parenthesization.

``codegen_cdiv`` and ``codegen_align_power_of_2`` in
``little_kernel/language/intrin/portable.py`` interpolate their operand text
into multiplicative/bitwise C contexts (``/ b`` and ``& ~(b - 1)``).  Bare
interpolation of a composite operand such as ``k + 1`` rebinds C precedence:
``((a + k + 1 - 1) / k + 1)`` parses as ``((a + k) / k) + 1`` instead of
``ceil(a / (k + 1))``.  These tests call the codegen functions directly (no
backend or GPU required) and assert the emitted text keeps each substituted
operand parenthesized, matching the convention already used by
``codegen_select``.
"""

import unittest

from little_kernel.language.intrin.portable import (
    codegen_align_power_of_2,
    codegen_cdiv,
)


class TestCodegenCdivParenthesization(unittest.TestCase):
    """``codegen_cdiv(a, b)`` must divide by ``(b)`` as a whole."""

    def test_plain_operands(self):
        result = codegen_cdiv("a", "b")
        self.assertEqual(result.return_val, "(((a) + (b) - 1) / (b))")

    def test_composite_divisor_stays_grouped(self):
        # Defect regression: b = "k + 1" previously emitted
        # "((a + k + 1 - 1) / k + 1)", which parses as ((a + k) / k) + 1.
        result = codegen_cdiv("a", "k + 1")
        self.assertEqual(result.return_val, "(((a) + (k + 1) - 1) / (k + 1))")

    def test_composite_dividend_stays_grouped(self):
        result = codegen_cdiv("x * 4", "2")
        self.assertEqual(result.return_val, "(((x * 4) + (2) - 1) / (2))")

    def test_both_operands_composite(self):
        result = codegen_cdiv("n - 1", "k + 1")
        self.assertEqual(result.return_val, "(((n - 1) + (k + 1) - 1) / (k + 1))")

    def test_divisor_not_bare_at_multiplicative_context(self):
        # Whatever the operand text is, the division must apply to a
        # parenthesized divisor: the emitted text ends with "/ (b))".
        result = codegen_cdiv("a", "k + 1")
        self.assertTrue(result.return_val.endswith("/ (k + 1))"))

    def test_no_body_or_includes(self):
        result = codegen_cdiv("a", "b")
        self.assertEqual(result.body, "")
        self.assertEqual(result.includes, [])


class TestCodegenAlignPowerOf2Parenthesization(unittest.TestCase):
    """``codegen_align_power_of_2(a, b)`` must mask with ``~((b) - 1)``."""

    def test_plain_operands(self):
        result = codegen_align_power_of_2("a", "b")
        self.assertEqual(result.return_val, "(((a) + (b) - 1) & ~((b) - 1))")

    def test_composite_alignment_stays_grouped(self):
        # Defect regression: b = "k + 1" previously emitted
        # "((a + k + 1 - 1) & ~(k + 1 - 1))", whose mask is ~k rather than
        # ~((k + 1) - 1).
        result = codegen_align_power_of_2("a", "k + 1")
        self.assertEqual(result.return_val, "(((a) + (k + 1) - 1) & ~((k + 1) - 1))")

    def test_composite_value_stays_grouped(self):
        result = codegen_align_power_of_2("x * 4", "16")
        self.assertEqual(result.return_val, "(((x * 4) + (16) - 1) & ~((16) - 1))")

    def test_both_operands_composite(self):
        result = codegen_align_power_of_2("n - 1", "k + 1")
        self.assertEqual(result.return_val, "(((n - 1) + (k + 1) - 1) & ~((k + 1) - 1))")

    def test_mask_not_bare_at_bitwise_context(self):
        # The mask must negate a parenthesized alignment: "~((b) - 1)".
        result = codegen_align_power_of_2("a", "k + 1")
        self.assertIn("& ~((k + 1) - 1)", result.return_val)

    def test_no_body_or_includes(self):
        result = codegen_align_power_of_2("a", "b")
        self.assertEqual(result.body, "")
        self.assertEqual(result.includes, [])


if __name__ == "__main__":
    unittest.main()
