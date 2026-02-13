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
from little_kernel.core.stmt import (BlockStmt, AssignStmt, IfStmt, ForStmt, WhileStmt, CallStmt, ReturnStmt, DeclStmt,
                                     SyncThreadsStmt, BreakStmt, ContinueStmt, AllocateStmt, AllocateTensorStmt)
from little_kernel.core.expr import (Literal, Var, UnOp, UnOpKind, AllocateTensorExpr, BinOp, BinOpKind)
from little_kernel.core.type_system import (int32, float32, bool_, ptr, Tensor, Struct, str_)
from little_kernel.core.ir_base import MemorySpace, AllocateMode


class TestStmtBase(unittest.TestCase):
    """Base class for Stmt tests with common fixtures"""

    def setUp(self):
        """Create reusable expressions and variables"""
        # Basic literals
        self.lit_int = Literal(5, int32)
        self.lit_float = Literal(3.14, float32)
        self.lit_bool_true = Literal(True, bool_)
        self.lit_bool_false = Literal(False, bool_)

        # Variables
        self.var_int = Var("x", int32)
        self.var_float = Var("y", float32)
        self.var_ptr_int = Var("p", ptr[int32])  # Pointer to int32
        self.var_tensor = Var("t", Tensor[int32])  # 2D tensor of int32
        self.var_struct = Var("s", Struct[("a", int32), ("b", float32)])

        # Tensor access (l-value)
        self.tensor_access = self.var_tensor[self.lit_int, self.lit_int]  # t[5,5]

        # Struct access (l-value)
        self.struct_access = self.var_struct.a  # s.a

        # Simple statements for blocks
        self.assign_stmt = AssignStmt(self.var_int, self.lit_int)
        self.call_stmt = CallStmt("printf", [Literal("Hello", str_)], is_builtin=True)


class TestBlockStmt(TestStmtBase):
    """Test BlockStmt (sequence of statements)"""

    def test_initialization(self):
        """Test creating a block with multiple statements"""
        stmts = [self.assign_stmt, self.call_stmt]
        block = BlockStmt(stmts)

        self.assertEqual(block.stmts, stmts)
        self.assertEqual(len(block.stmts), 2)

    def test_repr(self):
        """Test string representation"""
        block = BlockStmt([self.assign_stmt])
        self.assertIn("BlockStmt(", repr(block))
        self.assertIn("AssignStmt(lhs=Var(name=x", repr(block))


class TestAssignStmt(TestStmtBase):
    """Test AssignStmt (lhs = rhs)"""

    def test_valid_lhs_types(self):
        """Test assignment with valid l-value types (Var, TensorAccess, StructAccess)"""
        # Var as lhs
        assign_var = AssignStmt(self.var_int, self.lit_int)
        self.assertEqual(assign_var.lhs, self.var_int)
        self.assertEqual(assign_var.rhs, self.lit_int)

        # TensorAccess as lhs
        assign_tensor = AssignStmt(self.tensor_access, self.lit_int)
        self.assertEqual(assign_tensor.lhs, self.tensor_access)

        # StructAccess as lhs
        assign_struct = AssignStmt(self.struct_access, self.lit_int)
        self.assertEqual(assign_struct.lhs, self.struct_access)

    def test_invalid_lhs_type(self):
        """Test error when lhs is not a valid l-value"""
        invalid_lhs = self.lit_int  # Literal is not mutable
        with self.assertRaises(TypeError):
            AssignStmt(invalid_lhs, self.lit_int)


class TestIfStmt(TestStmtBase):
    """Test IfStmt (conditional execution)"""

    def setUp(self):
        super().setUp()
        self.then_block = BlockStmt([self.assign_stmt])
        self.else_block = BlockStmt([self.call_stmt])

    def test_valid_boolean_condition(self):
        """Test if statement with boolean condition"""
        if_stmt = IfStmt(self.lit_bool_true, self.then_block, self.else_block)

        self.assertEqual(if_stmt.cond, self.lit_bool_true)
        self.assertEqual(if_stmt.then_block, self.then_block)
        self.assertEqual(if_stmt.else_block, self.else_block)

    def test_condition_auto_cast(self):
        """Test automatic casting of non-boolean conditions to bool"""
        # Non-boolean condition (int32) should be cast to bool
        int_cond = self.lit_int  # 5 (non-zero → true)
        if_stmt = IfStmt(int_cond, self.then_block)

        self.assertIsInstance(if_stmt.cond, UnOp)
        self.assertEqual(if_stmt.cond.op, UnOpKind.CAST)
        self.assertEqual(if_stmt.cond.operand, int_cond)
        self.assertEqual(if_stmt.cond.type, bool_)

    def test_no_else_block(self):
        """Test if statement without else block"""
        if_stmt = IfStmt(self.lit_bool_false, self.then_block)
        self.assertIsNone(if_stmt.else_block)


class TestForStmt(TestStmtBase):
    """Test ForStmt (counted loop)"""

    def setUp(self):
        super().setUp()
        self.loop_var = Var("i", int32)
        self.init = Literal(0, int32)
        self.cond = BinOp(BinOpKind.LT, self.loop_var, Literal(10, int32), bool_)
        self.incr = BinOp(BinOpKind.ADD, self.loop_var, Literal(1, int32), int32)
        self.loop_body = BlockStmt([self.assign_stmt])

    def test_valid_initialization(self):
        """Test valid for loop with matching types"""
        for_stmt = ForStmt(loop_var=self.loop_var, init=self.init, cond=self.cond, incr=self.incr, body=self.loop_body)

        self.assertEqual(for_stmt.loop_var, self.loop_var)
        self.assertEqual(for_stmt.init, self.init)
        self.assertEqual(for_stmt.cond, self.cond)
        self.assertEqual(for_stmt.incr, self.incr)
        self.assertEqual(for_stmt.body, self.loop_body)

    def test_condition_auto_cast(self):
        """Test automatic casting of non-boolean loop condition"""
        int_cond = BinOp(BinOpKind.SUB, self.loop_var, Literal(10, int32), int32)  # i - 10 (int)
        for_stmt = ForStmt(self.loop_var, self.init, int_cond, self.incr, self.loop_body)

        self.assertIsInstance(for_stmt.cond, UnOp)
        self.assertEqual(for_stmt.cond.op, UnOpKind.CAST)
        self.assertEqual(for_stmt.cond.type, bool_)

    def test_type_mismatch_errors(self):
        """Test errors when init/incr types don't match loop_var"""
        # Init type mismatch (float vs int loop_var)
        bad_init = Literal(0.0, float32)
        with self.assertRaises(TypeError):
            ForStmt(self.loop_var, bad_init, self.cond, self.incr, self.loop_body)

        # Incr type mismatch (float vs int loop_var)
        bad_incr = BinOp(BinOpKind.ADD, self.loop_var, Literal(1.0, float32), float32)
        with self.assertRaises(TypeError):
            ForStmt(self.loop_var, self.init, self.cond, bad_incr, self.loop_body)


class TestWhileStmt(TestStmtBase):
    """Test WhileStmt (condition-based loop)"""

    def setUp(self):
        super().setUp()
        self.loop_body = BlockStmt([self.assign_stmt])

    def test_valid_boolean_condition(self):
        """Test while loop with boolean condition"""
        while_stmt = WhileStmt(self.lit_bool_true, self.loop_body)

        self.assertEqual(while_stmt.cond, self.lit_bool_true)
        self.assertEqual(while_stmt.body, self.loop_body)

    def test_condition_auto_cast(self):
        """Test automatic casting of non-boolean condition"""
        int_cond = Literal(0, int32)  # 0 → false
        while_stmt = WhileStmt(int_cond, self.loop_body)

        self.assertIsInstance(while_stmt.cond, UnOp)
        self.assertEqual(while_stmt.cond.op, UnOpKind.CAST)
        self.assertEqual(while_stmt.cond.type, bool_)


class TestCallStmt(TestStmtBase):
    """Test CallStmt (void function calls)"""

    def test_basic_call(self):
        """Test call to a regular function"""
        args = [self.lit_int, self.lit_float]
        call = CallStmt("print_values", args)

        self.assertEqual(call.callee, "print_values")
        self.assertEqual(call.args, args)
        self.assertFalse(call.is_builtin)

    def test_builtin_call(self):
        """Test call to a built-in function"""
        call = CallStmt("atomicAdd", [self.var_ptr_int, self.lit_int], is_builtin=True)

        self.assertTrue(call.is_builtin)
        self.assertIn("[builtin]", repr(call))


class TestReturnStmt(TestStmtBase):
    """Test ReturnStmt (function return)"""

    def test_return_with_value(self):
        """Test return statement with a value"""
        return_stmt = ReturnStmt(self.lit_int)
        self.assertEqual(return_stmt.value, self.lit_int)
        self.assertIn("value=int32(5)", repr(return_stmt))

    def test_return_void(self):
        """Test return statement without a value (void)"""
        return_stmt = ReturnStmt()
        self.assertIsNone(return_stmt.value)
        self.assertEqual(repr(return_stmt), "ReturnStmt()")


class TestDeclStmt(TestStmtBase):
    """Test DeclStmt (variable declaration)"""

    def test_declaration_without_init(self):
        """Test variable declaration without initialization"""
        decl = DeclStmt(self.var_int)
        self.assertEqual(decl.var, self.var_int)
        self.assertIsNone(decl.init)
        self.assertIn("int32 x", repr(decl))

    def test_declaration_with_init(self):
        """Test variable declaration with initialization"""
        decl = DeclStmt(self.var_float, self.lit_float)
        self.assertEqual(decl.init, self.lit_float)
        self.assertIn("float32 y = float32(3.14)", repr(decl))

    def test_init_type_mismatch(self):
        """Test error when initializer type doesn't match variable type"""
        bad_init = self.lit_int  # int32 init for float32 var
        with self.assertRaises(TypeError):
            DeclStmt(self.var_float, bad_init)


class TestSyncThreadsStmt(TestStmtBase):
    """Test SyncThreadsStmt (__syncthreads())"""

    def test_repr(self):
        """Test string representation"""
        sync_stmt = SyncThreadsStmt()
        self.assertEqual(repr(sync_stmt), "SyncThreadsStmt()")


class TestBreakContinueStmt(TestStmtBase):
    """Test BreakStmt and ContinueStmt"""

    def test_break_stmt(self):
        break_stmt = BreakStmt()
        self.assertEqual(repr(break_stmt), "BreakStmt()")

    def test_continue_stmt(self):
        continue_stmt = ContinueStmt()
        self.assertEqual(repr(continue_stmt), "ContinueStmt()")


class TestAllocateStmt(TestStmtBase):
    """Test AllocateStmt (memory allocation)"""

    def setUp(self):
        super().setUp()
        self.elem_type = int32
        self.size = Literal(1024, int32)  # 1024 elements
        self.alignment = Literal(16, int32)  # 16-byte alignment

    def test_valid_allocation(self):
        """Test valid memory allocation"""
        alloc_stmt = AllocateStmt(var=self.var_ptr_int,  # ptr[int32]
                                  size=self.size, elem_type=self.elem_type, space=MemorySpace.SHARED,
                                  alignment=self.alignment)

        self.assertEqual(alloc_stmt.var, self.var_ptr_int)
        self.assertEqual(alloc_stmt.size, self.size)
        self.assertEqual(alloc_stmt.elem_type, self.elem_type)
        self.assertEqual(alloc_stmt.space, MemorySpace.SHARED)
        self.assertEqual(alloc_stmt.alignment, self.alignment)

    def test_invalid_var_type(self):
        """Test error when var is not a pointer to elem_type"""
        # Var is not a pointer (int32 instead of ptr[int32])
        with self.assertRaises(TypeError):
            AllocateStmt(var=self.var_int,  # Invalid: not a pointer
                         size=self.size, elem_type=self.elem_type)

        # Var is a pointer but to wrong type (ptr[float32] instead of ptr[int32])
        var_ptr_float = Var("q", ptr[float32])
        with self.assertRaises(TypeError):
            AllocateStmt(var=var_ptr_float,  # Invalid: pointer to wrong type
                         size=self.size, elem_type=self.elem_type)

    def test_invalid_size_type(self):
        """Test error when size is not an integer"""
        float_size = Literal(1024.0, float32)  # Invalid: float size
        with self.assertRaises(TypeError):
            AllocateStmt(var=self.var_ptr_int, size=float_size, elem_type=self.elem_type)

    def test_invalid_alignment_type(self):
        """Test error when alignment is not an integer"""
        float_align = Literal(16.0, float32)  # Invalid: float alignment
        with self.assertRaises(TypeError):
            AllocateStmt(var=self.var_ptr_int, size=self.size, elem_type=self.elem_type, alignment=float_align)


class TestAllocateTensorStmt(TestStmtBase):
    """Test AllocateTensorStmt (tensor allocation)"""

    def setUp(self):
        super().setUp()
        self.tensor_shape = [Literal(32, int32), Literal(64, int32)]  # 32x64
        self.alloc_expr = AllocateTensorExpr(
            shape=self.tensor_shape,
            dtype=int32,
            mode=AllocateMode.ZEROS,
        )
        self.tensor_var = Var("tensor_var", self.alloc_expr.type)  # Matches tensor type

    def test_valid_allocation(self):
        """Test valid tensor allocation and assignment"""
        alloc_stmt = AllocateTensorStmt(var=self.tensor_var, alloc_expr=self.alloc_expr)

        self.assertEqual(alloc_stmt.var, self.tensor_var)
        self.assertEqual(alloc_stmt.alloc_expr, self.alloc_expr)

    def test_type_mismatch(self):
        """Test error when variable type doesn't match tensor type"""
        wrong_var = Var("wrong_var", int32)  # int32 != tensor type
        with self.assertRaises(TypeError):
            AllocateTensorStmt(var=wrong_var, alloc_expr=self.alloc_expr)


if __name__ == "__main__":
    unittest.main()
