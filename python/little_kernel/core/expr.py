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

from enum import Enum, auto
from typing import Union, List, Optional
from little_kernel.core.type_system import (LLType, IntType, Tensor, int32, float32, bool_, str_, ptr)
from little_kernel.core.ir_base import LLIR, AllocateMode, MemorySpace


class Expr(LLIR):
    """Base class for expressions: all expressions must contain type information"""

    def __init__(self, expr_type: LLType):
        self.type = expr_type  # Static type of the expression (determined during semantic analysis)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type})"

    # ------------------------------
    # Utility: Convert Python values to Literal
    # ------------------------------
    def _to_expr(self, value) -> "Expr":
        """Convert Python native values (int/float/bool) to Literal nodes"""
        if isinstance(value, Expr):
            return value
        if isinstance(value, int):
            return Literal(value, int32)  # Simplification: default int to int32
        if isinstance(value, float):
            return Literal(value, float32)  # Simplification: default float to float32
        if isinstance(value, bool):
            return Literal(value, bool_)
        if isinstance(value, str):
            return Literal(value, str_)
        raise TypeError(f"Unsupported operand type: {type(value)}")

    # ------------------------------
    # Binary operator overloading (arithmetic)
    # ------------------------------
    def __add__(self, other):
        other = self._to_expr(other)
        # Pointer + int -> PTR_ADD
        if self.type.is_pointer():
            if isinstance(other.type, IntType):
                return BinOp(BinOpKind.PTR_ADD, self, other, self.type)
            else:
                raise ValueError(f"Can't add pointer with value of {other.type} type.")
        # Regular addition
        return BinOp(BinOpKind.ADD, self, other, self.type)

    def __sub__(self, other):
        other = self._to_expr(other)
        # Pointer - int -> PTR_SUB
        if self.type.is_pointer():
            if isinstance(other.type, IntType):
                return BinOp(BinOpKind.PTR_SUB, self, other, self.type)
            else:
                raise ValueError(f"Can't sub pointer with value of {other.type} type.")
        # Regular subtraction
        return BinOp(BinOpKind.SUB, self, other, self.type)

    def __mul__(self, other):
        other = self._to_expr(other)
        return BinOp(BinOpKind.MUL, self, other, self.type)

    def __truediv__(self, other):
        other = self._to_expr(other)
        return BinOp(BinOpKind.DIV, self, other, self.type)

    def __mod__(self, other):
        other = self._to_expr(other)
        return BinOp(BinOpKind.MOD, self, other, self.type)

    # ------------------------------
    # Binary operator overloading (bitwise)
    # ------------------------------
    def __and__(self, other):
        other = self._to_expr(other)
        return BinOp(BinOpKind.BIT_AND, self, other, self.type)

    def __or__(self, other):
        other = self._to_expr(other)
        return BinOp(BinOpKind.BIT_OR, self, other, self.type)

    def __xor__(self, other):
        other = self._to_expr(other)
        return BinOp(BinOpKind.BIT_XOR, self, other, self.type)

    def __lshift__(self, other):
        other = self._to_expr(other)
        return BinOp(BinOpKind.SHL, self, other, self.type)

    def __rshift__(self, other):
        other = self._to_expr(other)
        return BinOp(BinOpKind.SHR, self, other, self.type)

    # ------------------------------
    # Binary operator overloading (comparison)
    # ------------------------------
    def __eq__(self, other):  # ==
        other = self._to_expr(other)
        return BinOp(BinOpKind.EQ, self, other, bool_)

    def __ne__(self, other):  # !=
        other = self._to_expr(other)
        return BinOp(BinOpKind.NE, self, other, bool_)

    def __lt__(self, other):  # <
        other = self._to_expr(other)
        return BinOp(BinOpKind.LT, self, other, bool_)

    def __gt__(self, other):  # >
        other = self._to_expr(other)
        return BinOp(BinOpKind.GT, self, other, bool_)

    def __le__(self, other):  # <=
        other = self._to_expr(other)
        return BinOp(BinOpKind.LE, self, other, bool_)

    def __ge__(self, other):  # >=
        other = self._to_expr(other)
        return BinOp(BinOpKind.GE, self, other, bool_)

    def logic_and(self, other):  # self && other (logical AND)
        other = self._to_expr(other)
        return BinOp(BinOpKind.LOGIC_AND, self, other, bool_)

    def logic_or(self, other):  # self || other (logical OR)
        other = self._to_expr(other)
        return BinOp(BinOpKind.LOGIC_OR, self, other, bool_)

    # ------------------------------
    # Reverse binary operators (when left operand is not an Expr)
    # ------------------------------
    def __radd__(self, other):  # other + self
        other = self._to_expr(other)
        return other + self

    def __rsub__(self, other):  # other - self
        other = self._to_expr(other)
        return other - self

    def __rmul__(self, other):  # other * self
        other = self._to_expr(other)
        return other * self

    def __rand__(self, other):  # other & self
        other = self._to_expr(other)
        return other & self

    def __ror__(self, other):  # other | self
        other = self._to_expr(other)
        return other | self

    # ------------------------------
    # Unary operator overloading
    # ------------------------------
    def __neg__(self):  # -self
        return UnOp(UnOpKind.NEG, self, self.type)

    def __invert__(self):  # ~self (bitwise NOT)
        return UnOp(UnOpKind.BIT_NOT, self, self.type)

    def logic_not(self):  # !self (logical NOT; Python has no __not__ magic method)
        return UnOp(UnOpKind.LOGIC_NOT, self, bool_)

    # ------------------------------
    # Memory access overloading
    # ------------------------------
    def __getitem__(self, index):  # self[index] (array access)
        """Support multi-dimensional indexing (e.g., a[i,j])"""
        if not self.type.is_tensor():
            raise TypeError(f"Only tensor type allows index via [], but get {self.type}")
        # Handle multi-dimensional indices (convert tuple to list)
        indices = [index] if not isinstance(index, tuple) else list(index)
        # Convert all indices to Expr
        expr_indices = [self._to_expr(idx) for idx in indices]

        return TensorAccess(self, expr_indices)

    def __getattr__(self, name):  # self.name (struct/vector member access)
        """Support struct member access (.) and pointer member access (->)"""
        assert isinstance(name, str), f"Member access name must be str, but get {type(name)}"
        if not self.type.is_struct():
            raise TypeError(f"Only struct type allows member access, but get {self.type}")
        if name not in dict(self.type.type_tuple).keys():
            raise TypeError(f"Struct {self.type} has no member {name}")

        return StructAccess(self, name)

    # ------------------------------
    # Pointer operations (dereference/address-of)
    # ------------------------------
    def deref(self):  # *self (dereference)
        if not self.type.is_pointer():
            raise TypeError(f"Cannot dereference non-pointer type {self.type}")
        return UnOp(UnOpKind.DEREF, self, self.type.inner_type)

    def addr_of(self):  # &self (address-of)
        return UnOp(UnOpKind.ADDR_OF, self, ptr[self.type])


# Literal


class Literal(Expr):
    """Literal expression (integer, float, boolean constants)"""

    def __init__(self, value: Union[int, float, bool, str], expr_type: LLType):
        super().__init__(expr_type)
        assert type(value) in [int, float, bool,
                               str], f"Literal value must be int, float, bool, or str, but got {type(value)}"
        self.value = value  # Literal value

    def __str__(self) -> str:
        return f"Literal(value={self.value}, type={self.type})"

    def __repr__(self) -> str:
        return f"{self.type}({self.value})"


# Variable


class Var(Expr):
    """Variable expression (access to local/global variables or parameters)"""

    def __init__(self, name: str, expr_type: LLType, defining_stmt=None, is_shared: bool = False,
                 is_global: bool = False):
        super().__init__(expr_type)
        self.name = name  # Variable name
        self.defining_stmt = defining_stmt
        self.is_shared = is_shared  # Whether it's a __shared__ variable
        self.is_global = is_global  # Whether it's a __device__ global variable

    def __repr__(self) -> str:
        return f"Var(name={self.name}, type={self.type}, shared={self.is_shared}, global={self.is_global})"

    def __str__(self) -> str:
        return f"Var(name={self.name}, type={self.type}, shared={self.is_shared}, global={self.is_global})"


# Unary


class UnOpKind(Enum):
    """Kinds of unary operators"""
    NEG = auto()  # - (numeric negation)
    BIT_NOT = auto()  # ~ (bitwise NOT)
    LOGIC_NOT = auto()  # ! (logical NOT)
    DEREF = auto()  # * (dereference)
    ADDR_OF = auto()  # & (address-of)
    CAST = auto()  # type cast (special case of unary op)


def un_op_kind_str(kind):
    if kind == UnOpKind.NEG:
        return "-"
    elif kind == UnOpKind.BIT_NOT:
        return "~"
    elif kind == UnOpKind.LOGIC_NOT:
        return "!"
    elif kind == UnOpKind.DEREF:
        return "*"
    elif kind == UnOpKind.ADDR_OF:
        return "&"
    elif kind == UnOpKind.CAST:
        return "cast"
    else:
        raise ValueError(f"Unknown unary operator kind {kind}")


class UnOp(Expr):
    """Unary operation expression (e.g., !a, *p)"""

    def __init__(self, op: UnOpKind, operand: Expr, expr_type: LLType):
        super().__init__(expr_type)
        self.op = op  # Operator kind
        self.operand = operand  # Operand

    def __str__(self) -> str:
        return f"UnOp(op={self.op}, operand={self.operand}, type={self.type})"

    def __repr__(self) -> str:
        if self.op == UnOpKind.CAST:
            return f"cast<{self.type}>({self.operand})"
        return f"{un_op_kind_str(self.op)}({repr(self.operand)})"


# Binary


class BinOpKind(Enum):
    """Kinds of binary operators"""
    # Arithmetic
    ADD = auto()  # +
    SUB = auto()  # -
    MUL = auto()  # *
    DIV = auto()  # /
    MOD = auto()  # %
    # Bitwise
    BIT_AND = auto()  # &
    BIT_OR = auto()  # |
    BIT_XOR = auto()  # ^
    SHL = auto()  # <<
    SHR = auto()  # >>
    # Logical
    LOGIC_AND = auto()  # &&
    LOGIC_OR = auto()  # ||
    # Comparison
    EQ = auto()  # ==
    NE = auto()  # !=
    LT = auto()  # <
    GT = auto()  # >
    LE = auto()  # <=
    GE = auto()  # >=
    # Pointer arithmetic (only pointer + int / int + pointer)
    PTR_ADD = auto()  # pointer + offset
    PTR_SUB = auto()  # pointer - offset


def bin_op_kind_str(kind):
    if kind == BinOpKind.ADD:
        return "+"
    elif kind == BinOpKind.SUB:
        return "-"
    elif kind == BinOpKind.MUL:
        return "*"
    elif kind == BinOpKind.DIV:
        return "/"
    elif kind == BinOpKind.MOD:
        return "%"
    elif kind == BinOpKind.BIT_AND:
        return "&"
    elif kind == BinOpKind.BIT_OR:
        return "|"
    elif kind == BinOpKind.BIT_XOR:
        return "^"
    elif kind == BinOpKind.SHL:
        return "<<"
    elif kind == BinOpKind.SHR:
        return ">>"
    elif kind == BinOpKind.LOGIC_AND:
        return "&&"
    elif kind == BinOpKind.LOGIC_OR:
        return "||"
    elif kind == BinOpKind.EQ:
        return "=="
    elif kind == BinOpKind.NE:
        return "!="
    elif kind == BinOpKind.LT:
        return "<"
    elif kind == BinOpKind.GT:
        return ">"
    elif kind == BinOpKind.LE:
        return "<="
    elif kind == BinOpKind.GE:
        return ">="
    elif kind == BinOpKind.PTR_ADD:
        return "+"
    elif kind == BinOpKind.PTR_SUB:
        return "-"
    else:
        raise ValueError(f"Unknown binary operator kind: {kind}")


class BinOp(Expr):
    """Binary operation expression (e.g., a + b, a && b)"""

    def __init__(self, op: BinOpKind, left: Expr, right: Expr, expr_type: LLType):
        super().__init__(expr_type)
        self.op = op  # Operator kind
        self.left = left  # Left operand
        self.right = right  # Right operand

    def __str__(self) -> str:
        return f"BinOp(op={self.op}, left={self.left}, right={self.right}, type={self.type})"

    def __repr__(self) -> str:
        return f"({self.type}({repr(self.left)} {bin_op_kind_str(self.op)} {repr(self.right)}))"


# Tensor Access


class TensorAccess(Expr):
    """Tensor access expression (e.g., a[i,j])"""

    def __init__(self, tensor: Expr, indices: List[Expr]):
        super().__init__(tensor.type.element_type)
        self.tensor = tensor  # Tensor expression
        self.indices = indices  # List of index expressions


# Struct Access


class StructAccess(Expr):
    """Struct access expression (e.g., a.member)"""

    def __init__(self, struct: Expr, member: str):
        super().__init__(dict(struct.type.type_tuple)[member])
        self.struct = struct  # Struct expression
        self.member = member  # Member name


# Call


class CallExpr(Expr):
    """Function call expression with a return value.
    
    Unlike CallStmt (which is a statement with no return value), CallExpr is an expression
    that evaluates to a value (e.g., `int x = add(a, b);` where `add(a, b)` is a CallExpr).
    Supports device/host functions, built-ins, and CUDA math functions.
    """

    def __init__(self, callee: str, args: List[Expr], return_type: LLType, is_builtin: bool = False):
        """
        Args:
            callee: Name of the function being called (e.g., "sinf", "my_device_func")
            args: List of argument expressions passed to the function
            return_type: Type of the value returned by this function call
            is_builtin: Whether the function is a CUDA built-in (e.g., "sqrtf", "atomicAdd")
        
        Raises:
            ValueError: If conflicting qualifiers (e.g., is_kernel and return_type is not void)
        """
        super().__init__(expr_type=return_type)  # Expr's type is the return type
        self.callee = callee
        self.args = args
        self.is_builtin = is_builtin

    def __repr__(self) -> str:
        attrs = []
        if self.is_builtin: attrs.append("builtin")
        attrs_str = f"[{','.join(attrs)}]" if attrs else ""
        return f"CallExpr{attrs_str}({self.callee}({self.args}) -> {self.type})"

    def __str__(self) -> str:
        attrs = []
        if self.is_builtin: attrs.append("builtin")
        attrs_str = f"[{','.join(attrs)}]" if attrs else ""
        return f"CallExpr{attrs_str}({self.callee}({self.args}) -> {self.type})"


# AllocateExpr


class AllocateExpr(Expr):
    """Base class for allocation expressions (returns a pointer/handle to allocated memory).
    
    Serves as a common parent for raw memory allocation (e.g., `malloc`-like) that returns
    a pointer. Specialized for tensors via `AllocateTensorExpr`.
    """

    def __init__(self, elem_type: LLType, size: Expr, return_type: LLType, space: MemorySpace = MemorySpace.GLOBAL,
                 alignment: Optional[Expr] = None):
        """
        Args:
            elem_type: Type of individual elements in the allocated block
            size: Number of elements to allocate (integer expression)
            return_type: Type of the result (typically a pointer to elem_type)
            space: Memory space (global/shared/local/constant)
            alignment: Optional byte alignment for the allocation
        
        Raises:
            TypeError: If size is not integer type or return_type is not a pointer
        """
        super().__init__(expr_type=return_type)

        # Validate size is integer
        if not isinstance(size.type, IntType):
            raise TypeError(f"Allocation size must be integer type, got {size.type}")

        # Validate return type is a pointer
        if not return_type.is_pointer() or return_type.inner_type != elem_type:
            raise TypeError(f"AllocateExpr return type must be pointer to {elem_type}, got {return_type}")

        self.elem_type = elem_type
        self.size = size
        self.space = space
        self.alignment = alignment

    def __repr__(self) -> str:
        align_str = f", alignment={self.alignment}" if self.alignment else ""
        return (f"AllocateExpr(elem_type={self.elem_type}, size={repr(self.size)}, "
                f"space={self.space}{align_str} -> {self.type})")

    def __str__(self) -> str:
        align_str = f", alignment={self.alignment}" if self.alignment else ""
        return (f"AllocateExpr(elem_type={self.elem_type}, size={self.size}, "
                f"space={self.space}{align_str} -> {self.type})")


# Allocate Tensor


class AllocateTensorExpr(Expr):
    """Expression for allocating a tensor (returns a tensor object).
    
    Mimics PyTorch's `torch.empty`, `torch.zeros`, etc. Supports multi-dimensional shapes,
    device placement, and initialization modes.
    """

    def __init__(self, shape: List[Expr], dtype: LLType, mode: AllocateMode = AllocateMode.EMPTY):
        """
        Args:
            shape: List of expressions defining tensor dimensions (e.g., [Literal(32), threadIdx.x + 1])
            dtype: Data type of tensor elements (e.g., float32, int32)
            mode: Initialization mode (empty/zeros/ones/rand)
        
        Raises:
            TypeError: If shape elements are not integers or dtype is invalid
        """
        # Tensor type is represented as a specialized type (assume LLType has a tensor constructor)
        tensor_type = Tensor[dtype]
        super().__init__(expr_type=tensor_type)

        # Validate shape: all dimensions must be integer expressions
        for dim in shape:
            if not isinstance(dim.type, IntType):
                raise TypeError(f"Tensor shape dimension must be integer, got {dim.type}")

        self.shape = shape
        self.dtype = dtype
        self.mode = mode

    def __repr__(self) -> str:
        shape_str = ", ".join(repr(dim) for dim in self.shape)
        return (f"AllocateTensorExpr(shape=({shape_str}), dtype={self.dtype}, "
                f"mode={self.mode.name.lower()})")

    def __str__(self) -> str:
        shape_str = ", ".join(repr(dim) for dim in self.shape)
        return (f"AllocateTensorExpr(shape=({shape_str}), dtype={self.dtype}, "
                f"mode={self.mode.name.lower()})")
