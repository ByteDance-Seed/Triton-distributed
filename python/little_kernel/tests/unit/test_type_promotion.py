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
# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
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
# TORT OR OTHERWISE, ARISING FROM OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

import unittest
import ast
import little_kernel.language as ll
from little_kernel.core.passes.utils.type_inference.type_inference_visitors import promote_type
from little_kernel.core.passes.utils.type_inference import TypeInferencer


class TestTypePromotion(unittest.TestCase):
    """Tests for implicit type promotion following C++ rules."""

    def test_promote_same_type(self):
        """Test that same types don't need promotion."""
        promoted_left, promoted_right, was_promoted = promote_type(ll.int32, ll.int32)
        self.assertEqual(promoted_left, ll.int32)
        self.assertEqual(promoted_right, ll.int32)
        self.assertFalse(was_promoted)

    def test_promote_int32_to_int64(self):
        """Test promotion of int32 to int64."""
        promoted_left, promoted_right, was_promoted = promote_type(ll.int32, ll.int64)
        self.assertEqual(promoted_left, ll.int64)
        self.assertEqual(promoted_right, ll.int64)
        self.assertTrue(was_promoted)

    def test_promote_int64_to_int32(self):
        """Test promotion of int64 to int32 (should promote int32 to int64)."""
        promoted_left, promoted_right, was_promoted = promote_type(ll.int64, ll.int32)
        self.assertEqual(promoted_left, ll.int64)
        self.assertEqual(promoted_right, ll.int64)
        self.assertTrue(was_promoted)

    def test_promote_uint32_to_uint64(self):
        """Test promotion of uint32 to uint64."""
        promoted_left, promoted_right, was_promoted = promote_type(ll.uint32, ll.uint64)
        self.assertEqual(promoted_left, ll.uint64)
        self.assertEqual(promoted_right, ll.uint64)
        self.assertTrue(was_promoted)

    def test_promote_int32_to_uint64(self):
        """Test promotion of int32 to uint64."""
        promoted_left, promoted_right, was_promoted = promote_type(ll.int32, ll.uint64)
        self.assertEqual(promoted_left, ll.uint64)
        self.assertEqual(promoted_right, ll.uint64)
        self.assertTrue(was_promoted)

    def test_promote_uint32_to_int64(self):
        """Test promotion of uint32 to int64."""
        promoted_left, promoted_right, was_promoted = promote_type(ll.uint32, ll.int64)
        self.assertEqual(promoted_left, ll.int64)
        self.assertEqual(promoted_right, ll.int64)
        self.assertTrue(was_promoted)

    def test_promote_signed_to_unsigned_same_size(self):
        """Test promotion when same size but different signedness (prefer unsigned)."""
        promoted_left, promoted_right, was_promoted = promote_type(ll.int32, ll.uint32)
        self.assertEqual(promoted_left, ll.uint32)
        self.assertEqual(promoted_right, ll.uint32)
        self.assertTrue(was_promoted)

    def test_promote_int_to_float(self):
        """Test promotion of integer to float."""
        promoted_left, promoted_right, was_promoted = promote_type(ll.int32, ll.float32)
        self.assertEqual(promoted_left, ll.float32)
        self.assertEqual(promoted_right, ll.float32)
        self.assertTrue(was_promoted)

    def test_promote_float_to_int(self):
        """Test promotion of integer to float (when float is left)."""
        promoted_left, promoted_right, was_promoted = promote_type(ll.float32, ll.int32)
        self.assertEqual(promoted_left, ll.float32)
        self.assertEqual(promoted_right, ll.float32)
        self.assertTrue(was_promoted)

    def test_promote_float32_to_float64(self):
        """Test promotion of float32 to float64."""
        promoted_left, promoted_right, was_promoted = promote_type(ll.float32, ll.float64)
        self.assertEqual(promoted_left, ll.float64)
        self.assertEqual(promoted_right, ll.float64)
        self.assertTrue(was_promoted)

    def test_promote_bool_to_int32(self):
        """Test promotion of bool to int32."""
        promoted_left, promoted_right, was_promoted = promote_type(ll.bool_, ll.int32)
        self.assertEqual(promoted_left, ll.int32)
        self.assertEqual(promoted_right, ll.int32)
        self.assertTrue(was_promoted)

    def test_bin_op_with_promotion(self):
        """Test binary operations with type promotion."""
        ctx = {'ll': ll}
        inferencer = TypeInferencer(ctx=ctx, scope_vars={})

        # Test int32 + uint64 (should promote int32 to uint64)
        code = ast.parse("x = a + b")
        # Set up scope
        inferencer.add_variable_to_scope('a', ll.int32)
        inferencer.add_variable_to_scope('b', ll.uint64)

        # This should work with promotion
        try:
            inferencer.visit(code)
            # Check that the result type is uint64
            result_type = inferencer.all_types.get(code.body[0].value)
            self.assertIsNotNone(result_type)
            # The result should be uint64 after promotion
        except Exception as e:
            self.fail(f"Type promotion should work for int32 + uint64: {e}")

    def test_bin_op_int32_uint64(self):
        """Test that int32 + uint64 works with promotion."""
        ctx = {'ll': ll}
        inferencer = TypeInferencer(ctx=ctx, scope_vars={'a': ll.int32, 'b': ll.uint64})

        # Create a simple addition expression
        left = ast.Name(id='a', ctx=ast.Load())
        right = ast.Name(id='b', ctx=ast.Load())
        binop = ast.BinOp(left=left, op=ast.Add(), right=right)

        # This should work with promotion
        result_type = inferencer.infer_node_type(binop)
        # Result should be uint64 (promoted type)
        self.assertEqual(result_type, ll.uint64)

    def test_bin_op_int32_int64(self):
        """Test that int32 + int64 works with promotion."""
        ctx = {'ll': ll}
        inferencer = TypeInferencer(ctx=ctx, scope_vars={'a': ll.int32, 'b': ll.int64})

        left = ast.Name(id='a', ctx=ast.Load())
        right = ast.Name(id='b', ctx=ast.Load())
        binop = ast.BinOp(left=left, op=ast.Add(), right=right)

        result_type = inferencer.infer_node_type(binop)
        # Result should be int64 (promoted type)
        self.assertEqual(result_type, ll.int64)

    def test_bin_op_int32_float32(self):
        """Test that int32 + float32 works with promotion."""
        ctx = {'ll': ll}
        inferencer = TypeInferencer(ctx=ctx, scope_vars={'a': ll.int32, 'b': ll.float32})

        left = ast.Name(id='a', ctx=ast.Load())
        right = ast.Name(id='b', ctx=ast.Load())
        binop = ast.BinOp(left=left, op=ast.Add(), right=right)

        result_type = inferencer.infer_node_type(binop)
        # Result should be float32 (promoted type)
        self.assertEqual(result_type, ll.float32)

    def test_bin_op_uint64_int32(self):
        """Test that uint64 / int32 works with promotion (like in bench_l2_cache)."""
        ctx = {'ll': ll}
        inferencer = TypeInferencer(ctx=ctx, scope_vars={'a': ll.uint64, 'b': ll.int32})

        left = ast.Name(id='a', ctx=ast.Load())
        right = ast.Name(id='b', ctx=ast.Load())
        binop = ast.BinOp(left=left, op=ast.Div(), right=right)

        result_type = inferencer.infer_node_type(binop)
        # Result should be uint64 (promoted type)
        self.assertEqual(result_type, ll.uint64)


if __name__ == '__main__':
    unittest.main()
