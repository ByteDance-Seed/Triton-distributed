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

from typing import Optional, List


# ==============================================================================
# Base LLType System
# ==============================================================================
class LLType:
    """Root base class for all types in the system.
    
    Ensures consistent interface for equality, hashing, and string representation.
    All types are immutable to guarantee stable hashing.
    """

    def is_scalar(self) -> bool:
        """Check if the type is a scalar"""
        return False

    def is_pointer(self) -> bool:
        """Check if the type is a pointer"""
        return False

    def is_const(self) -> bool:
        """Check if the type is a compile-time constant"""
        return False

    def is_grid_constant(self) -> bool:
        """Check if the type is a grid constant"""
        return False

    def is_tensor(self) -> bool:
        """Check if the type is a tensor (ND Array)"""
        return False

    def is_struct(self) -> bool:
        """Check if the type is a struct"""
        return False

    def is_tuple(self) -> bool:
        """Check if the type is a tuple"""
        return False

    def is_special_struct(self) -> bool:
        """Check if the type is a special struct (like Scheduler)"""
        return False

    def is_template(self) -> bool:
        """Check if the type is a template parameter"""
        return False

    def is_generic(self) -> bool:
        """Check if the type is a generic type"""
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash((type(self), tuple(sorted(self.__dict__.items()))))

    def __str__(self) -> str:
        return "LLType"

    def __repr__(self) -> str:
        return "LLType"


# ==============================================================================
# Scalar Types
# ==============================================================================
class ScalarType(LLType):
    """Base class for scalar types (atomic values with no internal structure)."""
    kind: str  # Categorizes scalar type (e.g., "int", "float", "str")

    def is_scalar(self):
        return True


# ------------------------------ Integer Types ------------------------------
class IntType(ScalarType):
    kind: str = "int"
    _cache: dict = {}

    def __new__(cls, bits: int, signed: bool = True, special=None):
        if special is not None:
            if special not in ("bool", "binary"):
                raise ValueError(f"Invalid special int type: {special!r}. Must be 'bool' or 'binary'")
            if bits != 1:
                raise ValueError(f"Special int types require 1 bit (got {bits})")
            key = (bits, signed, special)
        else:
            if bits not in (4, 8, 16, 32, 64, 128):
                raise ValueError(f"Invalid int bit width: {bits}. Must be 4,8,16,32,64,128")
            key = (bits, signed, None)

        if key not in cls._cache:
            cls._cache[key] = super().__new__(cls)
        return cls._cache[key]

    def __init__(self, bits: int, signed: bool = True, special=None) -> None:
        self.bits = bits
        self.signed = signed
        self.special = special

    def __str__(self) -> str:
        if self.special == "bool":
            return "bool"
        if self.special == "binary":
            return "binary"
        prefix = "u" if not self.signed else ""
        return f"{prefix}int{self.bits}"

    def __repr__(self) -> str:
        if self.special == "bool":
            return "bool"
        if self.special == "binary":
            return "binary"
        prefix = "u" if not self.signed else ""
        return f"{prefix}int{self.bits}"

    def __call__(self, value):
        """Allow type objects to be used as constructors: ll.uint32(0xFFFFFFFF)"""
        from little_kernel.language.intrin.dtype import val_cast
        return val_cast(value, self)


# ------------------------------ Floating-Point Types ------------------------------
class FloatType(ScalarType):
    kind: str = "float"
    _FORMAT_SPECS = {
        "fp4_e2m1": 4, "fp8_e5m2": 8, "fp8_e4m3": 8, "bfloat16": 16, "float16": 16, "float32": 32, "tfloat32": 32,
        "float64": 64
    }
    _cache = {}

    def __new__(cls, fmt: str) -> "FloatType":
        if fmt not in cls._FORMAT_SPECS:
            valid = sorted(cls._FORMAT_SPECS.keys())
            raise ValueError(f"Invalid float format: {fmt!r}. Must be one of {valid}")
        if fmt not in cls._cache:
            cls._cache[fmt] = super().__new__(cls)
        return cls._cache[fmt]

    def __init__(self, fmt: str) -> None:
        self.fmt = fmt
        self.bits = self._FORMAT_SPECS[fmt]

    def __str__(self) -> str:
        return self.fmt

    def __repr__(self) -> str:
        return self.fmt

    def __call__(self, value):
        """Allow type objects to be used as constructors: ll.float32(3.14)"""
        from little_kernel.language.intrin.dtype import val_cast
        return val_cast(value, self)


# ------------------------------ Void Types ---------------------------------
class VoidType(ScalarType):
    kind: str = "void"
    _instance: Optional["VoidType"] = None

    def __new__(cls) -> "VoidType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __str__(self) -> str:
        return "void"

    def __repr__(self) -> str:
        return "VoidType()"


# ------------------------------ String LLType ------------------------------
class StringType(ScalarType):
    kind: str = "str"
    _instance: Optional["StringType"] = None

    def __new__(cls) -> "StringType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __str__(self) -> str:
        return "str"

    def __repr__(self) -> str:
        return "StringType()"


# ==============================================================================
# Annotate Types (const, grid_const)
# ==============================================================================
class AnnotateType(LLType):
    """Base class for types that wrap other types (e.g., const, grid_const)."""

    def __init__(self, inner_type: LLType) -> None:
        self.inner_type = inner_type

    def __str__(self) -> str:
        return f"{self.__class__.__name__.lower()}[{self.inner_type}]"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(inner_type={self.inner_type!r})"


class Pointer(AnnotateType):
    """Annotate for pointer types (e.g., ptr[uint32])."""
    _instance = {}

    def __new__(cls, inner_type: LLType) -> "Pointer":
        if inner_type not in cls._instance:
            cls._instance[inner_type] = super().__new__(cls)
        return cls._instance[inner_type]

    def __init__(self, inner_type: LLType) -> None:
        super().__init__(inner_type)
        self.bits = 64

    def is_pointer(self):
        return True

    def __str__(self) -> str:
        return f"{self.inner_type}_ptr"

    def __repr__(self) -> str:
        return f"{self.inner_type}_ptr"


class Const(AnnotateType):
    """Annotate for compile-time constant types (e.g., const[uint32])."""
    _instance = {}

    def __new__(cls, inner_type: LLType) -> "Const":
        if inner_type not in cls._instance:
            cls._instance[inner_type] = super().__new__(cls)
        return cls._instance[inner_type]

    def __init__(self, inner_type: LLType) -> None:
        super().__init__(inner_type)

    def is_const(self):
        return True

    def __str__(self) -> str:
        return f"const[{self.inner_type}]"

    def __repr__(self) -> str:
        return f"const[{self.inner_type}]"


class GridConstant(Const):
    """Annotate for grid-scoped constant types (e.g., grid_const[TmaDescriptor])."""
    _instance = {}

    def __new__(cls, inner_type: LLType) -> "GridConstant":
        if inner_type not in cls._instance:
            cls._instance[inner_type] = super().__new__(cls, inner_type)
        return cls._instance[inner_type]

    def __init__(self, inner_type: LLType) -> None:
        super().__init__(inner_type)

    def is_grid_constant(self):
        return True

    def __str__(self) -> str:
        return f"grid_constant[{self.inner_type}]"

    def __repr__(self) -> str:
        return f"grid_constant[{self.inner_type}]"


class Template(AnnotateType):
    """Annotate for template parameter types (e.g., template[uint32])."""
    _instance = {}

    def __new__(cls, inner_type: LLType) -> "Template":
        if inner_type not in cls._instance:
            cls._instance[inner_type] = super().__new__(cls)
        return cls._instance[inner_type]

    def __init__(self, inner_type: LLType) -> None:
        super().__init__(inner_type)

    def is_template(self):
        return True

    def __str__(self) -> str:
        return f"template[{self.inner_type}]"

    def __repr__(self) -> str:
        return f"template[{self.inner_type}]"


class AnnotateTypeHelper(LLType):

    def __init__(self, inner_type: LLType):
        self.inner_type = inner_type

    def __getitem__(self, inner_type: LLType) -> LLType:
        return self.inner_type(inner_type)

    def is_generic(self):
        return True


class ConstTypeHelper(AnnotateTypeHelper):
    """
    Helper for const that supports both:
    - const[Type] syntax (via __getitem__)
    - x: const = 5 syntax (as a type annotation)
    """

    def __init__(self):
        super().__init__(Const)
        # Also act as ConstAnnotationType for type checking
        self._is_const_annotation = True

    def __eq__(self, other):
        # Allow comparison with ConstAnnotationType
        from .type_system import ConstAnnotationType, const_annotation
        if isinstance(other, ConstAnnotationType) or other == const_annotation:
            return True
        return super().__eq__(other)


# ==============================================================================
# Composite Types
# ==============================================================================
class TensorType(LLType):
    """Tensor type parameterized by its element scalar type (e.g., tensor[int32])."""
    _instance = {}

    def __new__(cls, element_type: LLType) -> "TensorType":
        if element_type not in cls._instance:
            cls._instance[element_type] = super().__new__(cls)
        return cls._instance[element_type]

    def __init__(self, element_type: LLType) -> None:
        if not isinstance(element_type, LLType):
            raise TypeError(f"Tensor element must be LLType (got {type(element_type).__name__})")
        self.element_type = element_type

    def is_tensor(self):
        return True

    def __str__(self) -> str:
        return f"Tensor[{self.element_type}]"

    def __repr__(self) -> str:
        return f"TensorType(element_type={self.element_type!r})"


class TensorTypeHelper(LLType):

    def __getitem__(self, inner_type: LLType) -> TensorType:
        if not isinstance(inner_type, LLType):
            raise TypeError(f"TensorTypeHelper requires a LLType instance (got {type(inner_type).__name__})")
        return TensorType(inner_type)

    def is_generic(self):
        return True


class StructType(LLType):
    """Struct type parameterized by a tuple of types."""
    _instance = {}

    def __new__(cls, type_list: List[tuple]) -> "TensorType":
        assert isinstance(type_list, (list, tuple))
        assert all(isinstance(t, tuple) and len(t) == 2 for t in type_list)
        assert all(isinstance(t[0], str) for t in type_list)
        assert all(isinstance(t[1], LLType) for t in type_list)
        type_tuple = tuple(type_list)
        if type_tuple not in cls._instance:
            cls._instance[type_tuple] = super().__new__(cls)
        return cls._instance[type_tuple]

    def __init__(self, type_list: list) -> None:
        assert isinstance(type_list, (list, tuple))
        assert all(isinstance(t, tuple) and len(t) == 2 for t in type_list)
        assert all(isinstance(t[0], str) for t in type_list)
        assert all(isinstance(t[1], LLType) for t in type_list)
        type_tuple = tuple(type_list)
        self.type_tuple = type_tuple

    def is_struct(self):
        return True

    def __str__(self) -> str:
        return f"Struct[{', '.join([str(x) for x in self.type_tuple])}]"

    def __repr__(self) -> str:
        return f"Struct[{', '.join([str(x[1]) for x in self.type_tuple])}]"


class StructTypeHelper(LLType):

    def __getitem__(self, type_list: list) -> StructType:
        return StructType(type_list)


class TupleType(LLType):
    """Tuple type parameterized by a list of element types (e.g., Tuple[bool, int32, int32])."""
    _instance = {}

    def __new__(cls, element_types: tuple) -> "TupleType":
        if not isinstance(element_types, tuple):
            element_types = tuple(element_types)
        if element_types not in cls._instance:
            cls._instance[element_types] = super().__new__(cls)
        return cls._instance[element_types]

    def __init__(self, element_types: tuple) -> None:
        if not isinstance(element_types, tuple):
            element_types = tuple(element_types)
        if not all(isinstance(t, LLType) for t in element_types):
            raise TypeError(f"Tuple elements must be LLType (got {[type(t).__name__ for t in element_types]})")
        self.element_types = element_types

    def is_tuple(self):
        return True

    def __str__(self) -> str:
        return f"Tuple[{', '.join([str(t) for t in self.element_types])}]"

    def __repr__(self) -> str:
        return f"TupleType(element_types={self.element_types!r})"

    def __getitem__(self, index: int) -> LLType:
        """Get element type at index."""
        return self.element_types[index]

    def __len__(self) -> int:
        """Get number of elements in tuple."""
        return len(self.element_types)


class TupleTypeHelper(LLType):
    """Helper class for creating tuple types: Tuple[type1, type2, ...]"""

    def __getitem__(self, element_types) -> TupleType:
        if isinstance(element_types, tuple):
            return TupleType(element_types)
        elif isinstance(element_types, list):
            return TupleType(tuple(element_types))
        else:
            # Single element tuple
            return TupleType((element_types, ))

    def is_generic(self):
        return True


class TmaDescriptorType(LLType):
    """TMA (Tensor Memory Accelerator) descriptor type (singleton)."""
    _instance: Optional["TmaDescriptorType"] = None

    def __new__(cls) -> "TmaDescriptorType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __str__(self) -> str:
        return "TmaDescriptor"

    def __repr__(self) -> str:
        return "TmaDescriptor"


class WgmmaType(LLType):
    "WGMMA (Tensor Core)."
    _instance: Optional["WgmmaType"] = None

    def __new__(cls) -> "WgmmaType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __str__(self) -> str:
        return "Wgmma"

    def __repr__(self) -> str:
        return "Wgmma"


# ==============================================================================
# Predefined LLType Instances
# ==============================================================================
# Integer types
bool_ = IntType(bits=1, signed=False, special="bool")
binary = IntType(bits=1, signed=True, special="binary")

int4 = IntType(bits=4, signed=True)
uint4 = IntType(bits=4, signed=False)
int8 = IntType(bits=8, signed=True)
uint8 = IntType(bits=8, signed=False)
int16 = IntType(bits=16, signed=True)
uint16 = IntType(bits=16, signed=False)
int32 = IntType(bits=32, signed=True)
uint32 = IntType(bits=32, signed=False)
int64 = IntType(bits=64, signed=True)
uint64 = IntType(bits=64, signed=False)
int128 = IntType(bits=128, signed=True)
uint128 = IntType(bits=128, signed=False)

# Float types
fp4_e2m1 = FloatType("fp4_e2m1")
fp8_e5m2 = FloatType("fp8_e5m2")
fp8_e4m3 = FloatType("fp8_e4m3")
bfloat16 = FloatType("bfloat16")
float16 = FloatType("float16")
float32 = FloatType("float32")
tfloat32 = FloatType("tfloat32")
float64 = FloatType("float64")

# Void type
void = VoidType()

# String type
str_ = StringType()

# Annotate types
ptr = AnnotateTypeHelper(Pointer)
const = ConstTypeHelper()  # Supports both const[Type] and x: const = 5
grid_constant = AnnotateTypeHelper(GridConstant)  # Expose the class to support grid_constant[LLType] syntax
template = AnnotateTypeHelper(Template)  # Expose the class to support template[LLType] syntax


# Const annotation type (for "x: const = 5" syntax)
# This is kept for backward compatibility and type checking
class ConstAnnotationType(LLType):
    """Special type marker for const annotations in AnnAssign (e.g., "x: const = 5")."""

    def __str__(self) -> str:
        return "const"

    def __repr__(self) -> str:
        return "const"


const_annotation = ConstAnnotationType()


# ==============================================================================
# Special Struct Type
# ==============================================================================
class SpecialStructType(LLType):
    """Type for special structs like Scheduler that need custom code generation."""
    _instance = {}

    def __new__(cls, struct_name: str) -> "SpecialStructType":
        if struct_name not in cls._instance:
            cls._instance[struct_name] = super().__new__(cls)
        return cls._instance[struct_name]

    def __init__(self, struct_name: str) -> None:
        self.struct_name = struct_name

    def is_special_struct(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"SpecialStruct[{self.struct_name}]"

    def __repr__(self) -> str:
        return f"SpecialStruct[{self.struct_name}]"  # Singleton instance for const annotations


# Tensor Types
Tensor = TensorTypeHelper()

# Struct Types
Struct = StructTypeHelper()

# Tuple Types
Tuple = TupleTypeHelper()

# TMADescriptor Types
TmaDescriptor = TmaDescriptorType()

# Wgmma Types
Wgmma = WgmmaType()

# Basic Type
type_ = LLType()


def get_dtype_size(dtype: LLType) -> int:
    """
    Calculate the size (in bytes) of a single element for the given LLType.
    Supports scalar types, structs, and other composite types.
    """
    # Handle scalar types (int/float/bool/etc.)
    if dtype.is_scalar():
        if hasattr(dtype, "bits"):
            return dtype.bits / 8  # Convert bits to bytes, but the calculation is float
        elif isinstance(dtype, StringType):
            raise NotImplementedError("StringType size calculation is not supported")
        elif isinstance(dtype, VoidType):
            raise ValueError("Cannot allocate memory for VoidType")
        else:
            raise NotImplementedError(f"Scalar type {type(dtype).__name__} size not implemented")

    # Handle struct types (sum of field sizes)
    elif dtype.is_struct():
        total_size = 0
        for _, field_type in dtype.type_tuple:
            total_size += get_dtype_size(field_type)
        return total_size

    # Handle tuple types (sum of element sizes)
    elif dtype.is_tuple():
        total_size = 0
        for element_type in dtype.element_types:
            total_size += get_dtype_size(element_type)
        return total_size

    elif dtype.is_tensor():
        # tensor dtype size is the pointer size
        return ptr[dtype].bits // 8

    # Handle other types (add more as needed)
    else:
        raise NotImplementedError(f"Size calculation for {type(dtype).__name__} not implemented")
