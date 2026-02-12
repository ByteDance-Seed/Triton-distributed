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

import unittest
from enum import Enum
from little_kernel.core.expr import (
    Literal,
    Var,
    UnOp,
    UnOpKind,
    BinOp,
    BinOpKind,
    bin_op_kind_str,
    TensorAccess,
    StructAccess,
    CallExpr,
    AllocateExpr,
    AllocateTensorExpr,
    MemorySpace,
    AllocateMode,
)
from little_kernel.core.type_system import (Tensor, Struct, int32, float32, ptr, bool_, str_, void, TensorType)


class TestLLIR(unittest.TestCase):
    """Test base LLIR class functionality"""

    def test_equality(self):
        """Test __eq__ method for LLIR subclasses"""
        # Equal instances
        a = Literal(5, int32)
        b = Literal(5, int32)
        self.assertEqual(a, b)

        # Unequal instances (different value)
        c = Literal(6, int32)
        self.assertNotEqual(a, c)

        # Unequal instances (different type)
        d = Literal(5, float32)
        self.assertNotEqual(a, d)


class TestLiteral(unittest.TestCase):
    """Test Literal expression class"""

    def test_valid_initialization(self):
        """Test initialization with valid literal values"""
        # Integer literal
        lit_int = Literal(42, int32)
        self.assertEqual(lit_int.value, 42)
        self.assertEqual(lit_int.type, int32)

        # Float literal
        lit_float = Literal(3.14, float32)
        self.assertEqual(lit_float.value, 3.14)
        self.assertEqual(lit_float.type, float32)

        # Boolean literal
        lit_bool = Literal(True, bool_)
        self.assertTrue(lit_bool.value)
        self.assertEqual(lit_bool.type, bool_)

        # String literal
        lit_str = Literal("test", str_)
        self.assertEqual(lit_str.value, "test")
        self.assertEqual(lit_str.type, str_)

    def test_invalid_initialization(self):
        """Test initialization with invalid value types"""
        with self.assertRaises(AssertionError):
            Literal([1, 2, 3], int32)  # List is not a valid literal type


class TestVar(unittest.TestCase):
    """Test Var expression class"""

    def test_basic_initialization(self):
        """Test basic variable initialization"""
        var = Var("x", int32)
        self.assertEqual(var.name, "x")
        self.assertEqual(var.type, int32)
        self.assertFalse(var.is_shared)
        self.assertFalse(var.is_global)

    def test_special_variables(self):
        """Test shared and global variables"""
        shared_var = Var("s", float32, is_shared=True)
        self.assertTrue(shared_var.is_shared)
        self.assertFalse(shared_var.is_global)

        global_var = Var("g", ptr[int32], is_global=True)
        self.assertTrue(global_var.is_global)
        self.assertFalse(global_var.is_shared)


class TestUnOp(unittest.TestCase):
    """Test UnOp (unary operator) expression class"""

    def test_initialization(self):
        """Test UnOp initialization with valid operands"""
        operand = Literal(5, int32)
        unop = UnOp(UnOpKind.NEG, operand, int32)

        self.assertEqual(unop.op, UnOpKind.NEG)
        self.assertEqual(unop.operand, operand)
        self.assertEqual(unop.type, int32)

    def test_repr(self):
        """Test string representation"""
        operand = Literal(5, int32)
        unop = UnOp(UnOpKind.NEG, operand, int32)
        self.assertEqual(repr(unop), "-(int32(5))")


class TestBinOpKindStr(unittest.TestCase):
    """Test bin_op_kind_str helper function"""

    def test_operator_strings(self):
        """Verify string mapping for all BinOpKind enum values"""
        self.assertEqual(bin_op_kind_str(BinOpKind.ADD), "+")
        self.assertEqual(bin_op_kind_str(BinOpKind.SUB), "-")
        self.assertEqual(bin_op_kind_str(BinOpKind.MUL), "*")
        self.assertEqual(bin_op_kind_str(BinOpKind.DIV), "/")
        self.assertEqual(bin_op_kind_str(BinOpKind.LOGIC_AND), "&&")
        self.assertEqual(bin_op_kind_str(BinOpKind.EQ), "==")
        self.assertEqual(bin_op_kind_str(BinOpKind.PTR_ADD), "+")

    def test_unknown_operator(self):
        """Test error handling for unknown operators"""
        with self.assertRaises(ValueError):
            # Create a mock unknown operator (out of enum range)
            class MockUnknownOp(Enum):
                UNKNOWN = 999

            bin_op_kind_str(MockUnknownOp.UNKNOWN)


class TestBinOp(unittest.TestCase):
    """Test BinOp (binary operator) expression class"""

    def test_initialization(self):
        """Test BinOp initialization with valid operands"""
        left = Literal(2, int32)
        right = Literal(3, int32)
        binop = BinOp(BinOpKind.ADD, left, right, int32)

        self.assertEqual(binop.op, BinOpKind.ADD)
        self.assertEqual(binop.left, left)
        self.assertEqual(binop.right, right)
        self.assertEqual(binop.type, int32)

    def test_repr(self):
        """Test string representation"""
        left = Literal(2, int32)
        right = Literal(3, int32)
        binop = BinOp(BinOpKind.ADD, left, right, int32)
        self.assertEqual(repr(binop), "(int32(int32(2) + int32(3)))")


class TestExprOperatorOverloading(unittest.TestCase):
    """Test operator overloading in the Expr base class"""

    def setUp(self):
        """Common test fixtures: basic expressions"""
        self.a = Literal(2, int32)  # int32(2)
        self.b = Literal(3, int32)  # int32(3)
        self.p = Var("p", ptr[int32])  # ptr(int32) variable

    def test_arithmetic_operators(self):
        """Test arithmetic operators (+, -, *, /, %)"""
        # Addition
        add = self.a + self.b
        self.assertIsInstance(add, BinOp)
        self.assertEqual(add.op, BinOpKind.ADD)
        self.assertEqual(add.left, self.a)
        self.assertEqual(add.right, self.b)

        # Subtraction
        sub = self.a - self.b
        self.assertIsInstance(sub, BinOp)
        self.assertEqual(sub.op, BinOpKind.SUB)

        # Multiplication
        mul = self.a * self.b
        self.assertIsInstance(mul, BinOp)
        self.assertEqual(mul.op, BinOpKind.MUL)

    def test_pointer_arithmetic(self):
        """Test pointer-specific arithmetic (+, - with integers)"""
        # Pointer + int
        ptr_add = self.p + 5
        self.assertIsInstance(ptr_add, BinOp)
        self.assertEqual(ptr_add.op, BinOpKind.PTR_ADD)
        self.assertEqual(ptr_add.left, self.p)
        self.assertIsInstance(ptr_add.right, Literal)
        self.assertEqual(ptr_add.right.value, 5)

        # Pointer - int
        ptr_sub = self.p - 2
        self.assertIsInstance(ptr_sub, BinOp)
        self.assertEqual(ptr_sub.op, BinOpKind.PTR_SUB)

        # Error: pointer + non-int
        with self.assertRaises(ValueError):
            self.p + 3.14  # float is invalid for pointer arithmetic

    def test_comparison_operators(self):
        """Test comparison operators (==, !=, <, >, etc.)"""
        # Equality
        eq = self.a == self.b
        self.assertIsInstance(eq, BinOp)
        self.assertEqual(eq.op, BinOpKind.EQ)
        self.assertEqual(eq.type, bool_)

        # Inequality
        ne = self.a != self.b
        self.assertIsInstance(ne, BinOp)
        self.assertEqual(ne.op, BinOpKind.NE)

        # Less than
        lt = self.a < self.b
        self.assertIsInstance(lt, BinOp)
        self.assertEqual(lt.op, BinOpKind.LT)

    def test_logical_operators(self):
        """Test logical operators (&&, ||, !)"""
        c = Literal(True, bool_)
        d = Literal(False, bool_)

        # Logical AND
        land = c.logic_and(d)
        self.assertIsInstance(land, BinOp)
        self.assertEqual(land.op, BinOpKind.LOGIC_AND)

        # Logical OR
        lor = c.logic_or(d)
        self.assertIsInstance(lor, BinOp)
        self.assertEqual(lor.op, BinOpKind.LOGIC_OR)

        # Logical NOT
        lnot = c.logic_not()
        self.assertIsInstance(lnot, UnOp)
        self.assertEqual(lnot.op, UnOpKind.LOGIC_NOT)
        self.assertEqual(lnot.type, bool_)

    def test_unary_operators(self):
        """Test unary operators (- (negation), ~ (bit not))"""
        # Numeric negation
        neg = -self.a
        self.assertIsInstance(neg, UnOp)
        self.assertEqual(neg.op, UnOpKind.NEG)
        self.assertEqual(neg.type, int32)

        # Bitwise NOT
        bnot = ~self.a
        self.assertIsInstance(bnot, UnOp)
        self.assertEqual(bnot.op, UnOpKind.BIT_NOT)
        self.assertEqual(bnot.type, int32)

    def test_reverse_operators(self):
        """Test reverse operators (e.g., 5 + expr instead of expr + 5)"""
        # Reverse addition (int + Expr)
        rev_add = 5 + self.a
        self.assertIsInstance(rev_add, BinOp)
        self.assertEqual(rev_add.op, BinOpKind.ADD)
        self.assertIsInstance(rev_add.left, Literal)  # 5 converted to Literal
        self.assertEqual(rev_add.right, self.a)

        # Reverse multiplication (int * Expr)
        rev_mul = 10 * self.a
        self.assertIsInstance(rev_mul, BinOp)
        self.assertEqual(rev_mul.op, BinOpKind.MUL)

    def test_pointer_operations(self):
        """Test pointer dereference and address-of"""
        # Address-of
        addr_of = self.a.addr_of()
        self.assertIsInstance(addr_of, UnOp)
        self.assertEqual(addr_of.op, UnOpKind.ADDR_OF)
        self.assertEqual(addr_of.type, ptr[int32])  # &int32 -> ptr(int32)

        # Dereference
        deref = self.p.deref()
        self.assertIsInstance(deref, UnOp)
        self.assertEqual(deref.op, UnOpKind.DEREF)
        self.assertEqual(deref.type, int32)  # *ptr(int32) -> int32

        # Error: dereference non-pointer
        with self.assertRaises(TypeError):
            self.a.deref()  # a is int32, not a pointer


class TestTensorAccess(unittest.TestCase):
    """Test TensorAccess expression class"""

    def setUp(self):
        """Create a tensor variable for testing"""
        self.tensor = Var("tensor", Tensor[int32])  # 2D tensor of int32

    def test_valid_indexing(self):
        """Test valid tensor indexing (1D and 2D)"""
        # 1D index
        idx1 = Literal(0, int32)
        access1 = self.tensor[idx1]
        self.assertIsInstance(access1, TensorAccess)
        self.assertEqual(access1.tensor, self.tensor)
        self.assertEqual(access1.indices, [idx1])
        self.assertEqual(access1.type, int32)  # tensor element type

        # 2D index (tuple)
        idx2 = (Literal(1, int32), Literal(2, int32))
        access2 = self.tensor[idx2]
        self.assertEqual(access2.indices, list(idx2))

    def test_invalid_indexing(self):
        """Test indexing non-tensor types (should raise error)"""
        non_tensor = Var("x", int32)  # int32 is not a tensor
        with self.assertRaises(TypeError):
            non_tensor[0]  # Only tensors allow [] indexing


class TestStructAccess(unittest.TestCase):
    """Test StructAccess expression class"""

    def setUp(self):
        """Create a struct variable for testing"""
        self.struct_var = Var("foo", Struct[("x", int32), ("y", float32)])  # Variable of type "Foo"

    def test_valid_member_access(self):
        """Test accessing existing struct members"""
        # Access "x" (int32)
        access_x = self.struct_var.x
        self.assertIsInstance(access_x, StructAccess)
        self.assertEqual(access_x.struct, self.struct_var)
        self.assertEqual(access_x.member, "x")
        self.assertEqual(access_x.type, int32)

        # Access "y" (float32)
        access_y = self.struct_var.y
        self.assertEqual(access_y.member, "y")
        self.assertEqual(access_y.type, float32)

    def test_invalid_member_access(self):
        """Test accessing non-existent struct members (should raise error)"""
        with self.assertRaises(TypeError):
            self.struct_var.z  # "z" is not a member of "Foo"

    def test_non_struct_access(self):
        """Test accessing members of non-struct types (should raise error)"""
        non_struct = Var("x", int32)  # int32 is not a struct
        with self.assertRaises(TypeError):
            non_struct.x  # Only structs allow .member access


class TestCallExpr(unittest.TestCase):
    """Test CallExpr (function call expression with return value)"""

    def setUp(self):
        """Create common test expressions"""
        self.arg1 = Literal(2, int32)
        self.arg2 = Literal(3.14, float32)
        self.args = [self.arg1, self.arg2]

    def test_valid_initialization(self):
        """Test initialization with valid parameters"""
        # Device function call
        call = CallExpr(callee="add", args=self.args, return_type=int32, is_builtin=False)
        self.assertEqual(call.callee, "add")
        self.assertEqual(call.args, self.args)
        self.assertEqual(call.type, int32)  # return_type maps to Expr's type
        self.assertFalse(call.is_builtin)

        # Built-in function call
        builtin_call = CallExpr(callee="sqrtf", args=[self.arg2], return_type=float32, is_builtin=True)
        self.assertEqual(builtin_call.callee, "sqrtf")
        self.assertEqual(builtin_call.type, float32)
        self.assertTrue(builtin_call.is_builtin)

    def test_kernel_call_validation(self):
        """Test validation for kernel calls (must return void)"""
        # Valid kernel call (void return)
        valid_kernel = CallExpr(
            callee="my_kernel",
            args=self.args,
            return_type=void,
        )
        self.assertEqual(valid_kernel.type, void)

    def test_repr(self):
        """Test string representation"""
        call = CallExpr(
            callee="max",
            args=[self.arg1, self.arg2],
            return_type=float32,
        )
        self.assertIn("max([int32(2), float32(3.14)])", repr(call))


class TestAllocateExpr(unittest.TestCase):
    """Test AllocateExpr (base memory allocation expression)"""

    def setUp(self):
        """Create common test components"""
        self.elem_type = int32
        self.size = Literal(1024, int32)  # Allocate 1024 elements
        self.return_type = ptr[self.elem_type]  # Pointer to element type
        self.alignment = Literal(16, int32)  # 16-byte alignment

    def test_valid_initialization(self):
        """Test initialization with valid parameters"""
        alloc = AllocateExpr(elem_type=self.elem_type, size=self.size, return_type=self.return_type,
                             space=MemorySpace.GLOBAL, alignment=self.alignment)
        self.assertEqual(alloc.elem_type, self.elem_type)
        self.assertEqual(alloc.size, self.size)
        self.assertEqual(alloc.type, self.return_type)
        self.assertEqual(alloc.space, MemorySpace.GLOBAL)
        self.assertEqual(alloc.alignment, self.alignment)

    def test_type_validation(self):
        """Test validation of element type, size type, and return type"""
        # Invalid: size is not integer type
        float_size = Literal(5.5, float32)
        with self.assertRaises(TypeError):
            AllocateExpr(elem_type=self.elem_type, size=float_size,  # Size must be integer
                         return_type=self.return_type)

        # Invalid: return type is not pointer to elem_type
        wrong_return_type = ptr[float32]  # Should be ptr[int32]
        with self.assertRaises(TypeError):
            AllocateExpr(elem_type=self.elem_type, size=self.size, return_type=wrong_return_type)

    def test_repr(self):
        """Test string representation"""
        alloc = AllocateExpr(elem_type=float32, size=Literal(64, int32), return_type=ptr[float32],
                             space=MemorySpace.SHARED)
        self.assertIn("AllocateExpr(elem_type=float32, size=int32(64), space=MemorySpace.SHARED", repr(alloc))
        self.assertIn("-> float32_ptr", repr(alloc))


class TestAllocateTensorExpr(unittest.TestCase):
    """Test AllocateTensorExpr (tensor allocation expression mimicking PyTorch)"""

    def setUp(self):
        """Create common test components"""
        self.static_shape = [Literal(32, int32), Literal(64, int32)]  # 32x64
        self.dynamic_shape = [Var("batch_size", int32), Literal(10, int32)]  # variable x 10
        self.dtype = float32

    def test_valid_initialization(self):
        """Test initialization with valid parameters"""
        # Basic empty tensor on CUDA
        empty_tensor = AllocateTensorExpr(
            shape=self.static_shape,
            dtype=self.dtype,
            mode=AllocateMode.EMPTY,
        )
        self.assertEqual(empty_tensor.shape, self.static_shape)
        self.assertEqual(empty_tensor.dtype, self.dtype)
        self.assertEqual(empty_tensor.mode, AllocateMode.EMPTY)
        self.assertIsInstance(empty_tensor.type, TensorType)  # Return type is TensorType

        # Zeros tensor on CPU with grad
        zeros_tensor = AllocateTensorExpr(
            shape=self.dynamic_shape,
            dtype=int32,
            mode=AllocateMode.ZEROS,
        )
        self.assertEqual(zeros_tensor.mode, AllocateMode.ZEROS)

    def test_shape_validation(self):
        """Test validation of tensor shape (must be integer expressions)"""
        # Invalid: shape contains non-integer expression
        invalid_shape = [self.static_shape[0], Literal(3.14, float32)]  # float dimension
        with self.assertRaises(TypeError):
            AllocateTensorExpr(
                shape=invalid_shape,
                dtype=self.dtype,
            )

    def test_repr(self):
        """Test string representation"""
        tensor_alloc = AllocateTensorExpr(
            shape=self.dynamic_shape,
            dtype=int32,
            mode=AllocateMode.ONES,
        )
        self.assertIn("shape=(Var(name=batch_size", repr(tensor_alloc))


if __name__ == "__main__":
    unittest.main()
