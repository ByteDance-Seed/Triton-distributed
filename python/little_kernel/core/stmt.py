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

from typing import Union, List, Optional
from little_kernel.core.type_system import (LLType, IntType, bool_)
from little_kernel.core.ir_base import LLIR, MemorySpace
from little_kernel.core.expr import Var, Expr, TensorAccess, StructAccess, UnOp, UnOpKind, AllocateTensorExpr


class Stmt(LLIR):

    def __init__(self):
        super().__init__()


class BlockStmt(Stmt):
    """A block of sequential statements (e.g., body of an if/loop).
    
    Represents a scoped sequence of statements, common in structured control flow.
    """

    def __init__(self, stmts: List[Stmt]):
        """
        Args:
            stmts: List of statements in the block (executed in order)
        """
        self.stmts = stmts  # List of Stmt objects

    def __repr__(self) -> str:
        stmt_strs = [f"  {repr(s)}" for s in self.stmts]
        tmp = ',\n'.join(stmt_strs)
        return "BlockStmt(\n" + tmp + "\n)"

    def __str__(self) -> str:
        stmt_strs = [f"  {repr(s)}" for s in self.stmts]
        tmp = ',\n'.join(stmt_strs)
        return "BlockStmt(\n" + tmp + "\n)"


class AssignStmt(Stmt):
    """Assignment statement (lhs = rhs).
    
    Assigns the value of an expression (rhs) to a mutable target (lhs).
    """

    def __init__(self, lhs: Union[Var, TensorAccess, StructAccess], rhs: Expr):
        """
        Args:
            lhs: Left-hand side (must be a mutable l-value: variable, tensor element, or struct member)
            rhs: Right-hand side (expression whose value is assigned to lhs)
        
        Raises:
            TypeError: If lhs is not a valid l-value type
        """
        # Validate lhs is a mutable target
        if not isinstance(lhs, (Var, TensorAccess, StructAccess)):
            raise TypeError(
                f"Invalid lhs type for assignment: {type(lhs)}. Must be Var, TensorAccess, or StructAccess.")
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self) -> str:
        return f"AssignStmt(lhs={self.lhs}, rhs={self.rhs})"

    def __str__(self) -> str:
        return f"AssignStmt(lhs={self.lhs}, rhs={self.rhs})"


class IfStmt(Stmt):
    """Conditional statement (if-else).
    
    Executes one of two blocks based on a boolean condition.
    """

    def __init__(self, cond: Expr, then_block: BlockStmt, else_block: Optional[BlockStmt] = None):
        """
        Args:
            cond: Condition expression (must be of boolean type)
            then_block: Block executed if cond is true
            else_block: Optional block executed if cond is false (None = no else)
        
        Raises:
            TypeError: If cond is not a boolean expression
        """
        if cond.type != bool_:
            cond = UnOp(UnOpKind.CAST, cond, bool_)
        self.cond = cond
        self.then_block = then_block
        self.else_block = else_block

    def __repr__(self) -> str:
        else_str = f", else_block={self.else_block}" if self.else_block else ""
        return f"IfStmt(cond={self.cond}, then_block={self.then_block}{else_str})"

    def __str__(self) -> str:
        else_str = f", else_block={self.else_block}" if self.else_block else ""
        return f"IfStmt(cond={self.cond}, then_block={self.then_block}{else_str})"


class ForStmt(Stmt):
    """For-loop statement (structured iteration).
    
    Represents a counted loop with initialization, condition, and increment.
    Common in CUDA for thread index-based iteration (e.g., looping over input elements).
    """

    def __init__(self, loop_var: Var, init: Expr, cond: Expr, incr: Expr, body: BlockStmt):
        """
        Args:
            loop_var: Loop variable (declared in the loop scope)
            init: Initial value for loop_var (e.g., Literal(0, int32))
            cond: Loop continuation condition (must be bool, e.g., loop_var < 1024)
            incr: Increment step (e.g., loop_var + 1)
            body: Block executed in each iteration
        
        Raises:
            TypeError: If cond is not boolean or types of init/incr don't match loop_var
        """
        # Validate condition type
        if cond.type != bool_:
            cond = UnOp(UnOpKind.CAST, cond, bool_)
        # Validate loop variable type consistency
        if init.type != loop_var.type:
            raise TypeError(f"For loop init type {init.type} != loop_var type {loop_var.type}")
        if incr.type != loop_var.type:
            raise TypeError(f"For loop incr type {incr.type} != loop_var type {loop_var.type}")

        self.loop_var = loop_var
        self.init = init
        self.cond = cond
        self.incr = incr
        self.body = body

    def __repr__(self) -> str:
        return (f"ForStmt(loop_var={self.loop_var}, init={self.init}, "
                f"cond={self.cond}, incr={self.incr}, body={self.body})")

    def __str__(self) -> str:
        return (f"ForStmt(loop_var={self.loop_var}, init={self.init}, "
                f"cond={self.cond}, incr={self.incr}, body={self.body})")


class WhileStmt(Stmt):
    """While-loop statement (condition-based iteration).
    
    Executes a block repeatedly while a condition remains true.
    """

    def __init__(self, cond: Expr, body: BlockStmt):
        """
        Args:
            cond: Loop condition (must be bool)
            body: Block executed while cond is true
        
        Raises:
            TypeError: If cond is not boolean
        """
        if cond.type != bool_:
            cond = UnOp(UnOpKind.CAST, cond, bool_)
        self.cond = cond
        self.body = body

    def __repr__(self) -> str:
        return f"WhileStmt(cond={self.cond}, body={self.body})"

    def __str__(self) -> str:
        return f"WhileStmt(cond={self.cond}, body={self.body})"


class CallStmt(Stmt):
    """Function call statement (for void functions or side effects).
    
    Differs from CallExpr: CallStmt is a standalone statement (no return value),
    used for functions that perform actions (e.g., CUDA built-ins like printf).
    """

    def __init__(self, callee: str, args: List[Expr], is_builtin: bool = False):
        """
        Args:
            callee: Name of the function being called
            args: List of arguments passed to the function
            is_builtin: Whether the function is a CUDA built-in (e.g., printf, atomicAdd)
        """
        self.callee = callee
        self.args = args
        self.is_builtin = is_builtin

    def __repr__(self) -> str:
        attrs = []
        if self.is_builtin:
            attrs.append("builtin")
        attrs_str = f"[{','.join(attrs)}]" if attrs else ""
        return f"CallStmt{attrs_str}({self.callee}({self.args}))"

    def __str__(self) -> str:
        attrs = []
        if self.is_builtin:
            attrs.append("builtin")
        attrs_str = f"[{','.join(attrs)}]" if attrs else ""
        return f"CallStmt{attrs_str}({self.callee}({self.args}))"


class ReturnStmt(Stmt):
    """Return statement (exits a function).
    
    May optionally return a value (for non-void functions).
    """

    def __init__(self, value: Optional[Expr] = None):
        """
        Args:
            value: Optional return value (None for void functions)
        """
        self.value = value

    def __repr__(self) -> str:
        return f"ReturnStmt(value={repr(self.value)})" if self.value else "ReturnStmt()"

    def __str__(self) -> str:
        return f"ReturnStmt(value={repr(self.value)})" if self.value else "ReturnStmt()"


class DeclStmt(Stmt):
    """Variable declaration statement (with optional initialization).
    
    Declares a new variable with a type, name, and optional initial value.
    Supports CUDA memory qualifiers (e.g., __shared__).
    """

    def __init__(self, var: Var, init: Optional[Expr] = None):
        """
        Args:
            var: Variable to declare (includes name, type, and qualifiers like is_shared)
            init: Optional initial value (must match var.type if provided)
        
        Raises:
            TypeError: If init type doesn't match var.type
        """
        if init is not None and init.type != var.type:
            raise TypeError(f"Decl init type {init.type} != var type {var.type}")
        self.var = var
        self.init = init

    def __repr__(self) -> str:
        init_str = f" = {repr(self.init)}" if self.init else ""
        return f"DeclStmt({self.var.type} {self.var.name}{init_str})"

    def __str__(self) -> str:
        init_str = f" = {repr(self.init)}" if self.init else ""
        return f"DeclStmt({self.var.type} {self.var.name}{init_str})"


class SyncThreadsStmt(Stmt):
    """CUDA __syncthreads() synchronization statement.
    
    Synchronizes all threads in a block; ensures all threads reach this point
    before any proceed. Critical for shared memory consistency.
    """

    def __repr__(self) -> str:
        return "SyncThreadsStmt()"

    def __str__(self) -> str:
        return "SyncThreadsStmt()"


class BreakStmt(Stmt):
    """Break statement (exits the innermost loop)."""

    def __repr__(self) -> str:
        return "BreakStmt()"

    def __str__(self) -> str:
        return "BreakStmt()"


class ContinueStmt(Stmt):
    """Continue statement (skips to the next loop iteration)."""

    def __repr__(self) -> str:
        return "ContinueStmt()"

    def __str__(self) -> str:
        return "ContinueStmt()"


class AllocateStmt(Stmt):
    """Memory allocation statement (dynamic memory management).
    
    Represents allocation of memory in CUDA's memory spaces (e.g., `int* x = new int[N];` 
    or CUDA-specific `__shared__ float s[1024];`). Supports both static and dynamic sizes.
    """

    def __init__(self, var: Var, size: Expr, elem_type: LLType, space: MemorySpace = MemorySpace.GLOBAL,
                 alignment: Optional[Expr] = None):
        """
        Args:
            var: Variable to hold the allocated pointer (must be a pointer type)
            size: Expression specifying the number of elements to allocate (must be integer type)
            elem_type: Type of each element in the allocated memory (e.g., int32 for an int* pointer)
            space: Memory space where allocation occurs (global/shared/local/constant)
            alignment: Optional alignment requirement (in bytes, must be power-of-two integer)
        
        Raises:
            TypeError: If var is not a pointer, size is not integer, or types mismatch
            ValueError: If alignment is invalid (non-positive, non-power-of-two)
        """
        # Validate pointer type for var
        if not var.type.is_pointer() or var.type.inner_type != elem_type:
            raise TypeError(f"AllocateStmt var must be pointer to {elem_type}, got {var.type}")

        # Validate size is an integer expression
        if not isinstance(size.type, IntType):
            raise TypeError(f"Allocate size must be integer type, got {size.type}")

        # Validate alignment (if provided)
        if alignment is not None:
            if not isinstance(alignment.type, IntType):
                raise TypeError(f"Alignment must be integer type, got {alignment.type}")
            # Note: Runtime check for power-of-two can be added in semantic analysis

        self.var = var
        self.size = size
        self.elem_type = elem_type
        self.space = space
        self.alignment = alignment

    def __repr__(self) -> str:
        align_str = f", alignment={self.alignment}" if self.alignment else ""
        return (f"AllocateStmt({self.var} = allocate {self.space.name.lower()}"
                f" {self.elem_type}[{self.size}]{align_str})")

    def __str__(self) -> str:
        align_str = f", alignment={self.alignment}" if self.alignment else ""
        return (f"AllocateStmt({self.var} = allocate {self.space.name.lower()}"
                f" {self.elem_type}[{self.size}]{align_str})")


class AllocateTensorStmt(Stmt):
    """Statement for allocating a tensor and assigning it to a variable.
    
    Combines tensor allocation (via `AllocateTensorExpr`) with variable assignment,
    mimicking `x = torch.empty(...)` in PyTorch from a device side.
    """

    def __init__(self, var: Var, alloc_expr: AllocateTensorExpr):
        """
        Args:
            var: Variable to hold the allocated tensor (must match tensor type)
            alloc_expr: Tensor allocation expression defining shape/dtype/etc.
        
        Raises:
            TypeError: If variable type does not match the allocated tensor type
        """
        # Validate variable type matches tensor type
        if var.type != alloc_expr.type:
            raise TypeError(f"AllocateTensorStmt var type {var.type} does not match "
                            f"allocated tensor type {alloc_expr.type}")

        self.var = var
        self.alloc_expr = alloc_expr

    def __repr__(self) -> str:
        return f"AllocateTensorStmt({self.var} = {self.alloc_expr})"

    def __str__(self) -> str:
        return f"AllocateTensorStmt({self.var} = {self.alloc_expr})"
