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

    Every concrete LLType is an *immutable value*: observable attributes are
    read-only, equality is structural and stable, and hashes are derived from
    the same frozen key so interned instances are safe to use as dict keys and
    in sets for the entire lifetime of the process. Composite members are
    always stored as tuples; mutability cannot leak in through a caller-supplied
    list.
    """

    # Frozen key used for structural equality and hashing. Subclasses set it
    # exactly once, in ``__new__`` (or ``__init__`` for non-interned types),
    # through ``object.__setattr__``; afterwards every attribute assignment is
    # rejected, so cached interned instances cannot be reinitialized or
    # corrupted by later constructor calls.
    _type_key: tuple

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

    def is_constexpr(self) -> bool:
        """Check if the type is a constexpr (compile-time specialization)"""
        return False

    __slots__ = ("_type_key", )

    def __setattr__(self, name, value):
        # Every observable field lives in a declared slot and is frozen after
        # construction; there is no instance __dict__ to bypass this guard.
        # Dunders used by Python internals (e.g. __class__ during unpickling)
        # remain settable.
        if name.startswith("__") and name.endswith("__"):
            object.__setattr__(self, name, value)
            return
        raise AttributeError(f"{type(self).__name__} instances are immutable; "
                             f"cannot set attribute {name!r}")

    def __delattr__(self, name):
        raise AttributeError(f"{type(self).__name__} instances are immutable; "
                             f"cannot delete attribute {name!r}")

    def _freeze(self, key: tuple) -> None:
        """Assign the immutable identity key exactly once."""
        if hasattr(self, "_type_key"):
            raise RuntimeError(f"{type(self).__name__} is already frozen")
        object.__setattr__(self, "_type_key", (type(self), ) + tuple(key))

    def __init__(self) -> None:
        # Non-interned leaf objects (e.g. the generic helpers and the base
        # LLType marker) still need a stable frozen key for hashing/equality.
        if not hasattr(self, "_type_key"):
            self._freeze(())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LLType):
            return NotImplemented
        return self._type_key == other._type_key

    def __hash__(self) -> int:
        return hash(self._type_key)

    def __copy__(self):
        # Immutable interned values are safe to share: copying must yield the
        # exact same object (baseline behavior), never a half-restored clone.
        return self

    def __deepcopy__(self, memo):
        # Register the shared instance so recursive deepcopy of containers
        # (e.g. ast nodes carrying a type as a constant value) terminates and
        # preserves identity.
        memo[id(self)] = self
        return self

    def __str__(self) -> str:
        return "LLType"

    def __repr__(self) -> str:
        return "LLType"


def _intern(cls, cache: dict, key, init):
    """Validate-once, intern-once construction helper.

    ``key`` is the fully normalized, validated, hashable interning key.
    ``init(new_object, key)`` populates a freshly created object exactly once;
    cached objects are returned untouched, so repeated constructor calls can
    never reinitialize or corrupt an existing interned instance.
    """
    existing = cache.get(key)
    if existing is not None:
        return existing
    obj = object.__new__(cls)
    init(obj, key)
    cache[key] = obj
    return obj


# ==============================================================================
# Scalar Types
# ==============================================================================
class ScalarType(LLType):
    """Base class for scalar types (atomic values with no internal structure)."""
    __slots__ = ()
    kind: str  # Categorizes scalar type (e.g., "int", "float", "str")

    def is_scalar(self):
        return True


# ------------------------------ Integer Types ------------------------------
class IntType(ScalarType):
    __slots__ = ("bits", "signed", "special")
    kind: str = "int"
    _cache: dict = {}

    @staticmethod
    def _validate_key(bits: int, signed: bool = True, special=None):
        if special is not None:
            if special not in ("bool", "binary"):
                raise ValueError(f"Invalid special int type: {special!r}. Must be 'bool' or 'binary'")
            if bits != 1:
                raise ValueError(f"Special int types require 1 bit (got {bits})")
        else:
            if bits not in (4, 8, 16, 32, 64, 128):
                raise ValueError(f"Invalid int bit width: {bits}. Must be 4,8,16,32,64,128")
        return bits, bool(signed), special

    def __new__(cls, bits: int, signed: bool = True, special=None):
        key = cls._validate_key(bits, signed, special)

        def init(obj, k):
            obj._freeze(k)
            object.__setattr__(obj, "bits", k[0])
            object.__setattr__(obj, "signed", k[1])
            object.__setattr__(obj, "special", k[2])

        return _intern(cls, cls._cache, key, init)

    def __init__(self, bits: int, signed: bool = True, special=None) -> None:
        # Validation and initialization happen once in __new__; this exists
        # only to preserve the public constructor signature. It must never
        # reinitialize a cached singleton.
        pass

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
    __slots__ = ("fmt", "bits")
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
        key = (fmt, )

        def init(obj, k):
            obj._freeze(k)
            object.__setattr__(obj, "fmt", k[0])
            object.__setattr__(obj, "bits", cls._FORMAT_SPECS[k[0]])

        return _intern(cls, cls._cache, key, init)

    def __init__(self, fmt: str) -> None:
        # Initialization happens once in __new__; cached instances are reused.
        pass

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
    __slots__ = ()
    kind: str = "void"
    _instance: Optional["VoidType"] = None

    def __new__(cls) -> "VoidType":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._freeze(())
            cls._instance = obj
        return cls._instance

    def __str__(self) -> str:
        return "void"

    def __repr__(self) -> str:
        return "VoidType()"


# ------------------------------ String LLType ------------------------------
class StringType(ScalarType):
    __slots__ = ()
    kind: str = "str"
    _instance: Optional["StringType"] = None

    def __new__(cls) -> "StringType":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._freeze(())
            cls._instance = obj
        return cls._instance

    def __str__(self) -> str:
        return "str"

    def __repr__(self) -> str:
        return "StringType()"


# ==============================================================================
# Fixed-width vector value types
# ==============================================================================
class VectorType(LLType):
    """Fixed-width vector register value type (e.g., u32x4 -> CUDA uint4).

    A VectorType is a first-class *value* type: it can annotate locals and
    builtin return values, but it has no internal structure addressable from
    the language except through explicit lane intrinsics (e.g.,
    ``vec_u32x4_lane``). Instances are immutable and cached per
    (element_type, lanes) pair so equality and hashing behave like the other
    predefined types.
    """
    __slots__ = ("element_type", "lanes")
    kind: str = "vector"
    _cache: dict = {}

    def __new__(cls, element_type: LLType, lanes: int) -> "VectorType":
        if not isinstance(element_type, LLType):
            raise TypeError(f"VectorType element must be an LLType, got {type(element_type).__name__}")
        if not isinstance(lanes, int) or isinstance(lanes, bool) or lanes < 2 or lanes > 16:
            raise ValueError(f"VectorType lanes must be an int in [2, 16], got {lanes!r}")
        key = (element_type, lanes)

        def init(obj, k):
            obj._freeze(k)
            object.__setattr__(obj, "element_type", k[0])
            object.__setattr__(obj, "lanes", k[1])

        return _intern(cls, cls._cache, key, init)

    def __init__(self, element_type: LLType, lanes: int) -> None:
        # Initialization happens once in __new__; cached instances are reused.
        pass

    @property
    def bits(self) -> int:
        """Total vector width in bits (element width must be bit-granular)."""
        elem_bits = getattr(self.element_type, "bits", None)
        if elem_bits is None:
            raise NotImplementedError(f"Vector element type {self.element_type!r} has no bit width")
        return elem_bits * self.lanes

    def is_scalar(self):
        # A vector register value is a single kernel-level value (one local),
        # but it is not an atomic scalar: keep it distinct so passes that
        # assume scalar arithmetic do not silently elementwise it.
        return False

    def __str__(self) -> str:
        return f"{self.element_type}x{self.lanes}"

    def __repr__(self) -> str:
        return f"VectorType(element_type={self.element_type!r}, lanes={self.lanes!r})"


# ==============================================================================
# Annotate Types (const, grid_const)
# ==============================================================================
class AnnotateType(LLType):
    """Base class for types that wrap other types (e.g., const, grid_const)."""
    __slots__ = ("inner_type", )

    def __init__(self, inner_type) -> None:
        # Public constructor preserved: AnnotateType(int32) produces a frozen
        # wrapper value. Built-in interned subclasses (Pointer/Const/...) bypass
        # this in their __new__ and keep their inert __init__ stubs; external
        # subclasses may call super().__init__(inner_type) directly.
        self._freeze((inner_type, ))
        object.__setattr__(self, "inner_type", inner_type)

    @staticmethod
    def _validate_inner(inner_type) -> None:
        """Wrapper payloads only need to be hashable so they can intern.

        Annotation wrappers historically accept arbitrary annotation objects
        (including plain Python classes such as enum types used purely as
        markers), so we deliberately do not require an LLType instance. An
        unhashable payload fails here instead of corrupting the intern pool.
        """
        try:
            hash(inner_type)
        except TypeError as exc:
            raise TypeError(f"Annotated inner type must be hashable, got {type(inner_type).__name__}") from exc

    def __str__(self) -> str:
        return f"{self.__class__.__name__.lower()}[{self.inner_type}]"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(inner_type={self.inner_type!r})"


def _make_annotated(cls, cache: dict, inner_type: LLType, extra_attrs=()):
    """Validate once, intern once and freeze a single-type annotation wrapper."""
    AnnotateType._validate_inner(inner_type)
    key = (inner_type, )

    def init(obj, k):
        obj._freeze(k + tuple(extra_attrs))
        object.__setattr__(obj, "inner_type", k[0])
        for attr, value in extra_attrs:
            object.__setattr__(obj, attr, value)

    existing = cache.get(key)
    if existing is not None and type(existing) is cls:
        return existing
    obj = object.__new__(cls)
    init(obj, key)
    cache[key] = obj
    return obj


class Pointer(AnnotateType):
    """Annotate for pointer types (e.g., ptr[uint32])."""
    __slots__ = ("bits", )
    _instance = {}

    def __new__(cls, inner_type: LLType) -> "Pointer":
        AnnotateType._validate_inner(inner_type)
        key = (inner_type, )

        def init(obj, k):
            obj._freeze(k)
            object.__setattr__(obj, "inner_type", k[0])
            object.__setattr__(obj, "bits", 64)

        return _intern(cls, cls._instance, key, init)

    def __init__(self, inner_type: LLType) -> None:
        # Initialization happens once in __new__; cached instances are reused.
        pass

    def is_pointer(self):
        return True

    def __str__(self) -> str:
        return f"{self.inner_type}_ptr"

    def __repr__(self) -> str:
        return f"{self.inner_type}_ptr"


class Const(AnnotateType):
    """Annotate for compile-time constant types (e.g., const[uint32])."""
    __slots__ = ()
    _instance = {}

    def __new__(cls, inner_type: LLType) -> "Const":
        return _make_annotated(cls, cls._instance, inner_type)

    def __init__(self, inner_type: LLType) -> None:
        pass

    def is_const(self):
        return True

    def __str__(self) -> str:
        return f"const[{self.inner_type}]"

    def __repr__(self) -> str:
        return f"const[{self.inner_type}]"


class GridConstant(Const):
    """Annotate for grid-scoped constant types (e.g., grid_const[TmaDescriptor])."""
    __slots__ = ()
    _instance = {}

    def __new__(cls, inner_type: LLType) -> "GridConstant":
        return _make_annotated(cls, cls._instance, inner_type)

    def __init__(self, inner_type: LLType) -> None:
        pass

    def is_grid_constant(self):
        return True

    def __str__(self) -> str:
        return f"grid_constant[{self.inner_type}]"

    def __repr__(self) -> str:
        return f"grid_constant[{self.inner_type}]"


class Template(AnnotateType):
    """Annotate for template parameter types (e.g., template[uint32])."""
    __slots__ = ()
    _instance = {}

    def __new__(cls, inner_type: LLType) -> "Template":
        return _make_annotated(cls, cls._instance, inner_type)

    def __init__(self, inner_type: LLType) -> None:
        pass

    def is_template(self):
        return True

    def __str__(self) -> str:
        return f"template[{self.inner_type}]"

    def __repr__(self) -> str:
        return f"template[{self.inner_type}]"


class Constexpr(AnnotateType):
    """Annotate for constexpr types -- values known before kernel launch.

    Modelled after Triton's tl.constexpr: these parameters are substituted
    with literal values at codegen time and are NOT emitted as C++ function
    parameters.
    """
    __slots__ = ()
    _instance: dict[LLType, "Constexpr"] = {}

    def __new__(cls, inner_type: LLType) -> "Constexpr":
        return _make_annotated(cls, cls._instance, inner_type)

    def __init__(self, inner_type: LLType) -> None:
        pass

    def is_constexpr(self) -> bool:
        return True

    def is_const(self):
        return True

    def __str__(self) -> str:
        return f"constexpr[{self.inner_type}]"

    def __repr__(self) -> str:
        return f"constexpr[{self.inner_type}]"


class AnnotateTypeHelper(LLType):
    __slots__ = ("inner_type", )

    def __init__(self, inner_type: LLType):
        self._freeze((inner_type, ))
        object.__setattr__(self, "inner_type", inner_type)

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
    __slots__ = ("_is_const_annotation", )

    def __init__(self):
        super().__init__(Const)
        # Also act as ConstAnnotationType for type checking
        object.__setattr__(self, "_is_const_annotation", True)

    def __eq__(self, other):
        # Allow comparison with ConstAnnotationType
        from little_kernel.core.type_system import ConstAnnotationType, const_annotation
        if isinstance(other, ConstAnnotationType) or other == const_annotation:
            return True
        return super().__eq__(other)

    def __hash__(self) -> int:
        # Equal objects must hash equally: compare equal to const_annotation,
        # so share its stable hash.
        from little_kernel.core.type_system import const_annotation
        return hash(const_annotation)


# ==============================================================================
# Composite Types
# ==============================================================================
def _normalize_element_types(element_types) -> tuple:
    """Normalize a composite member input exactly once into a validated tuple.

    Accepts tuples, lists and single-pass iterators/generators. The iterable is
    consumed exactly once, so callers such as ``TupleType(iter((int32,)))``
    cannot drain the same iterator a second time and corrupt a cached instance.
    """
    if isinstance(element_types, (str, bytes)):
        raise TypeError(f"Element types must be an iterable of LLType, got {type(element_types).__name__}")
    elements = tuple(element_types)
    if not all(isinstance(t, LLType) for t in elements):
        raise TypeError(f"Elements must be LLType instances (got {[type(t).__name__ for t in elements]})")
    return elements


class TensorType(LLType):
    """Tensor type parameterized by its element scalar type (e.g., tensor[int32])."""
    __slots__ = ("element_type", )
    _instance = {}

    def __new__(cls, element_type: LLType) -> "TensorType":
        if not isinstance(element_type, LLType):
            raise TypeError(f"Tensor element must be LLType (got {type(element_type).__name__})")
        key = (element_type, )

        def init(obj, k):
            obj._freeze(k)
            object.__setattr__(obj, "element_type", k[0])

        return _intern(cls, cls._instance, key, init)

    def __init__(self, element_type: LLType) -> None:
        # Initialization happens once in __new__; cached instances are reused.
        pass

    def is_tensor(self):
        return True

    def __str__(self) -> str:
        return f"Tensor[{self.element_type}]"

    def __repr__(self) -> str:
        return f"TensorType(element_type={self.element_type!r})"


class TensorTypeHelper(LLType):
    __slots__ = ()

    def __getitem__(self, inner_type: LLType) -> TensorType:
        if not isinstance(inner_type, LLType):
            raise TypeError(f"TensorTypeHelper requires a LLType instance (got {type(inner_type).__name__})")
        return TensorType(inner_type)

    def is_generic(self):
        return True


class StructType(LLType):
    """Struct type parameterized by a tuple of ``(name, LLType)`` fields."""
    __slots__ = ("type_tuple", )
    _instance = {}

    @staticmethod
    def _normalize_fields(type_list) -> tuple:
        """Normalize and validate the field list exactly once.

        Fields are stored as an immutable tuple of ``(str, LLType)`` pairs.
        """
        if not isinstance(type_list, (list, tuple)):
            raise TypeError(f"Struct field list must be a list or tuple (got {type(type_list).__name__})")
        fields = tuple(type_list)
        for field in fields:
            if not isinstance(field, tuple) or len(field) != 2:
                raise TypeError("Each struct field must be a (name, type) pair of length 2")
            name, field_type = field
            if not isinstance(name, str):
                raise TypeError(f"Struct field name must be str (got {type(name).__name__})")
            if not isinstance(field_type, LLType):
                raise TypeError(f"Struct field type must be LLType (got {type(field_type).__name__})")
        return fields

    def __new__(cls, type_list) -> "StructType":
        # Normalize/validate once before touching the intern pool: an invalid
        # input can never create or corrupt a pool entry.
        type_tuple = cls._normalize_fields(type_list)

        def init(obj, k):
            obj._freeze(k)
            object.__setattr__(obj, "type_tuple", k)

        return _intern(cls, cls._instance, type_tuple, init)

    def __init__(self, type_list) -> None:
        # Initialization happens once in __new__; cached instances are reused.
        pass

    def is_struct(self):
        return True

    def __str__(self) -> str:
        return f"Struct[{', '.join([str(x) for x in self.type_tuple])}]"

    def __repr__(self) -> str:
        return f"Struct[{', '.join([str(x[1]) for x in self.type_tuple])}]"


class StructTypeHelper(LLType):
    __slots__ = ()

    def __getitem__(self, type_list: list) -> StructType:
        return StructType(type_list)


class TupleType(LLType):
    """Tuple type parameterized by element types (e.g., Tuple[bool, int32, int32]).

    Accepts a tuple, list or any single-pass iterator of LLType instances.
    The input is normalized and validated exactly once, before interning;
    members are stored as an immutable tuple and instances are interned.
    """
    __slots__ = ("element_types", )
    _instance = {}

    def __new__(cls, element_types) -> "TupleType":
        # Normalize once: tuple() is called on a fresh local, never twice on a
        # caller-supplied iterator, and validation runs before pool insertion.
        elements = _normalize_element_types(element_types)

        def init(obj, k):
            obj._freeze(k)
            object.__setattr__(obj, "element_types", k)

        return _intern(cls, cls._instance, elements, init)

    def __init__(self, element_types) -> None:
        # Initialization happens once in __new__; cached instances are reused.
        pass

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
    __slots__ = ()

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

# Fixed-width vector register value types (CUDA native vector types)
u32x2 = VectorType(uint32, 2)  # maps to CUDA uint2; build via ll.VectorType for others
u32x4 = VectorType(uint32, 4)  # maps to CUDA uint4

# String type
str_ = StringType()

# Annotate types
ptr = AnnotateTypeHelper(Pointer)
const = ConstTypeHelper()  # Supports both const[Type] and x: const = 5
grid_constant = AnnotateTypeHelper(GridConstant)  # Expose the class to support grid_constant[LLType] syntax
template = AnnotateTypeHelper(Template)  # Expose the class to support template[LLType] syntax
constexpr = AnnotateTypeHelper(Constexpr)  # Expose the class to support constexpr[LLType] syntax


# Const annotation type (for "x: const = 5" syntax)
# This is kept for backward compatibility and type checking
class ConstAnnotationType(LLType):
    """Special type marker for const annotations in AnnAssign (e.g., "x: const = 5")."""
    __slots__ = ()

    def __init__(self):
        self._freeze(())

    def __eq__(self, other):
        # Symmetric counterpart to ConstTypeHelper.__eq__.
        from little_kernel.core.type_system import ConstTypeHelper
        if isinstance(other, ConstTypeHelper):
            return True
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()

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
    __slots__ = ("struct_name", )
    _instance = {}

    def __new__(cls, struct_name: str) -> "SpecialStructType":
        if not isinstance(struct_name, str) or not struct_name:
            raise TypeError(f"SpecialStructType name must be a non-empty str (got {struct_name!r})")
        key = (struct_name, )

        def init(obj, k):
            obj._freeze(k)
            object.__setattr__(obj, "struct_name", k[0])

        return _intern(cls, cls._instance, key, init)

    def __init__(self, struct_name: str) -> None:
        # Initialization happens once in __new__; cached instances are reused.
        pass

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

# Wgmma Types

# Register vector types (HIP backend: MFMA operands, packed loads and stores)
int32x2 = VectorType(int32, 2)
int32x4 = VectorType(int32, 4)
int32x8 = VectorType(int32, 8)
uint32x2 = VectorType(uint32, 2)
uint32x4 = VectorType(uint32, 4)
float32x2 = VectorType(float32, 2)
float32x4 = VectorType(float32, 4)
float32x16 = VectorType(float32, 16)

# AMDGPU buffer resource descriptor

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

    # Handle fixed-width vector register value types
    elif getattr(dtype, "kind", None) == "vector":
        return get_dtype_size(dtype.element_type) * dtype.lanes

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

    elif isinstance(dtype, VectorType):
        return get_dtype_size(dtype.element_type) * dtype.lanes

    # Handle other types (add more as needed)
    else:
        raise NotImplementedError(f"Size calculation for {type(dtype).__name__} not implemented")


__all__ = [
    "LLType",
    "ScalarType",
    "VectorType",
    "IntType",
    "FloatType",
    "VoidType",
    "StringType",
    "AnnotateType",
    "Pointer",
    "Const",
    "GridConstant",
    "Template",
    "Constexpr",
    "AnnotateTypeHelper",
    "ConstTypeHelper",
    "TensorType",
    "TensorTypeHelper",
    "StructType",
    "StructTypeHelper",
    "TupleType",
    "TupleTypeHelper",
    "VectorType",
    "bool_",
    "binary",
    "int4",
    "uint4",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "int128",
    "uint128",
    "fp4_e2m1",
    "fp8_e5m2",
    "fp8_e4m3",
    "bfloat16",
    "float16",
    "float32",
    "tfloat32",
    "float64",
    "void",
    "str_",
    "u32x2",
    "u32x4",
    "ptr",
    "const",
    "grid_constant",
    "template",
    "constexpr",
    "ConstAnnotationType",
    "const_annotation",
    "SpecialStructType",
    "Tensor",
    "Struct",
    "Tuple",
    "int32x2",
    "int32x4",
    "int32x8",
    "uint32x2",
    "uint32x4",
    "float32x2",
    "float32x4",
    "float32x16",
    "type_",
    "get_dtype_size",
]
