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
"""Typed kernel launch ABI (RF01).

The CUDA Driver launches kernels with a ``void**`` array where every entry
points to a C ABI value whose size, sign and layout are fixed by the *kernel
signature* -- never by the magnitude of the Python argument.  This module
provides:

* :class:`KernelParameter`: one immutable, ordered parameter specification
  (scalar LLType/string/explicit ``ctypes`` scalar, pointer, or structure);
* :class:`KernelSignature`: an immutable ordered sequence of parameters,
  constructible directly or via :func:`signature_from_annotations`;
* :func:`pack_arguments`: validates Python values against the signature and
  materializes real Driver argument buffers, returned as
  :class:`PackedArguments` whose ``pointers`` array is suitable for
  ``cuLaunchKernel`` and whose ``storage`` keeps every ctypes object alive.

Framework adaptation (NumPy arrays, Torch tensors, allocator handles) belongs
to the runtime layer: this module marshals raw addresses, ``ctypes`` values
and ``ctypes`` structures only.
"""

from __future__ import annotations

import ctypes
import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "SCALAR",
    "POINTER",
    "STRUCTURE",
    "KernelParameter",
    "KernelSignature",
    "PackedArguments",
    "pack_arguments",
    "signature_from_annotations",
    "ctypes_type_for_scalar",
    "register_struct_layout",
    "half_bits",
    "bfloat16_bits",
    "cuda_half",
    "cuda_bfloat16",
]

SCALAR = "scalar"
POINTER = "pointer"
STRUCTURE = "structure"

_ADDRESS_SPACE_GLOBAL = 0

# -----------------------------------------------------------------------------
# Exact 16-bit CUDA scalar storage (__half / __nv_bfloat16)
# -----------------------------------------------------------------------------


def half_bits(value: float) -> int:
    """Encode a Python float as IEEE-754 binary16 raw bits (round-to-nearest-even)."""
    import struct

    packed = struct.pack("<e", float(value))
    return struct.unpack("<H", packed)[0]


def bfloat16_bits(value: float) -> int:
    """Encode a Python float as raw BF16 bits.

    NaN and infinity are handled *before* round-to-nearest-even addition so
    their exponent/sign (and NaN payload/quiet bit) are preserved instead of
    collapsing to signed zero.  Finite values use round-to-nearest-even on the
    dropped low 16 mantissa bits.
    """
    import struct

    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    sign_exp = bits & 0x7F800000

    if sign_exp == 0x7F800000:
        result = (bits >> 16) & 0xFFFF
        mantissa = bits & 0x007FFFFF
        if mantissa != 0:
            # NaN: keep sign/payload but guarantee the BF16 quiet bit is set,
            # even when all payload lived in the discarded low 16 bits.
            result |= 0x0040
        return result

    lsb = (bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    return ((bits + rounding_bias) >> 16) & 0xFFFF


class cuda_half(ctypes.c_uint16):
    """2-byte storage for CUDA ``__half``: raw IEEE binary16 bits.

    The class itself only fixes the 16-bit storage; float encoding happens
    explicitly in :func:`pack_arguments` via :func:`half_bits`, so the host
    never performs an implicit float32 demotion.  Construct directly with a
    raw 16-bit bit pattern, or from a float through :func:`half_bits`.
    """


class cuda_bfloat16(ctypes.c_uint16):
    """2-byte storage for CUDA ``__nv_bfloat16``: raw BF16 bits."""


# -----------------------------------------------------------------------------
# Scalar LLType/string -> ctypes mapping
# -----------------------------------------------------------------------------

_SCALAR_NAME_TO_CTYPE = {
    "bool": ctypes.c_bool,
    "int8": ctypes.c_int8,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "uint8": ctypes.c_uint8,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
    "float16": cuda_half,
    "half": cuda_half,
    "bfloat16": cuda_bfloat16,
    # Python-level convenience aliases
    "int": ctypes.c_int64,
    "float": ctypes.c_double,
    "double": ctypes.c_double,
}

_SCALAR_RANGES = {
    ctypes.c_bool: (0, 1),
    ctypes.c_int8: (-(1 << 7), (1 << 7) - 1),
    ctypes.c_int16: (-(1 << 15), (1 << 15) - 1),
    ctypes.c_int32: (-(1 << 31), (1 << 31) - 1),
    ctypes.c_int64: (-(1 << 63), (1 << 63) - 1),
    ctypes.c_uint8: (0, (1 << 8) - 1),
    ctypes.c_uint16: (0, (1 << 16) - 1),
    ctypes.c_uint32: (0, (1 << 32) - 1),
    ctypes.c_uint64: (0, (1 << 64) - 1),
}


def ctypes_type_for_scalar(scalar_type: Any) -> type:
    """Return the exact host ``ctypes`` storage type for a scalar annotation.

    Accepts an ``LLType`` scalar instance (``IntType``/``FloatType``),
    a canonical scalar spelling (``"int64"``, ``"uint64"``, ``"bool"``,
    ``"float64"``, ``"float16"``, ``"bfloat16"`` ...) or an explicit
    ``ctypes`` scalar class (e.g. ``ctypes.c_double``).  CUDA ``__half`` and
    ``__nv_bfloat16`` have a real 16-bit by-value ABI and map to
    :class:`cuda_half`/:class:`cuda_bfloat16` raw-bit storage.  Device-only
    widths with no host scalar ABI (``int4``, FP4, FP8) are rejected rather
    than silently widened.
    """
    if isinstance(scalar_type, type) and issubclass(scalar_type, ctypes._SimpleCData):
        return scalar_type

    if isinstance(scalar_type, str):
        name = scalar_type.strip()
        if name not in _SCALAR_NAME_TO_CTYPE:
            raise ValueError(f"Unsupported scalar ABI type {scalar_type!r}; expected one of "
                             f"{sorted(_SCALAR_NAME_TO_CTYPE)} or an explicit ctypes scalar")
        return _SCALAR_NAME_TO_CTYPE[name]

    # LLType instances without importing the (mutable) type system here.
    if not getattr(scalar_type, "is_scalar", lambda: False)():
        raise TypeError(f"{scalar_type!r} is not a scalar kernel parameter type")

    special = getattr(scalar_type, "special", None)
    if special == "bool":
        return ctypes.c_bool

    bits = getattr(scalar_type, "bits", None)
    if isinstance(scalar_type, type):  # pragma: no cover - defensive
        raise TypeError(f"Unsupported scalar type class {scalar_type!r}")

    # Integer scalar: width and signedness come from the type, not the value.
    if hasattr(scalar_type, "signed"):
        if bits not in (8, 16, 32, 64):
            raise ValueError(f"Integer scalar width {bits} has no host kernel ABI storage "
                             f"for {scalar_type!r}")
        prefix = "u" if not scalar_type.signed else ""
        name = f"{prefix}int{bits}"
        if name not in _SCALAR_NAME_TO_CTYPE:  # pragma: no cover
            raise ValueError(f"Unsupported integer scalar {scalar_type!r}")
        return _SCALAR_NAME_TO_CTYPE[name]

    fmt = str(scalar_type)
    if fmt in ("float32", "float64", "float16", "bfloat16"):
        return _SCALAR_NAME_TO_CTYPE[fmt]
    raise ValueError(f"Scalar {scalar_type!r} has no native host ctypes representation "
                     "(device-only widths such as int4/FP4/FP8 must be passed via an "
                     "explicit ctypes layout adapter)")


def _is_scalar_lltype(t: Any) -> bool:
    return bool(getattr(t, "is_scalar", lambda: False)())


def _is_pointer_lltype(t: Any) -> bool:
    return bool(getattr(t, "is_pointer", lambda: False)())


def _is_tensor_lltype(t: Any) -> bool:
    return bool(getattr(t, "is_tensor", lambda: False)())


def _is_structure_lltype(t: Any) -> bool:
    return bool(
        getattr(t, "is_struct", lambda: False)() or getattr(t, "is_grid_constant", lambda: False)()
        or type(t).__name__ in ("TmaDescriptorType", "WgmmaType") or getattr(t, "is_special_struct", lambda: False)())


def _is_vector_lltype(t: Any) -> bool:
    return getattr(t, "kind", None) == "vector"


# -----------------------------------------------------------------------------
# Concrete structure layouts
# -----------------------------------------------------------------------------

# Registry of explicit frontend-struct -> ctypes layout adapters, keyed by
# LLType instance and by frontend type class name.  A signature parameter must
# always resolve to a concrete layout: unknown layouts are never silently
# accepted "size from value".
_STRUCT_LAYOUTS: dict = {}
_STRUCT_LAYOUTS_BY_NAME: dict = {}


def register_struct_layout(key: Any, structure_type: type) -> None:
    """Register an explicit ``ctypes.Structure`` layout for a frontend type.

    ``key`` is either:

    * an exact ``LLType`` **instance** (e.g. ``ll.Wgmma`` or a specific
      ``ll.Struct[...]``) -- registered for that value only, never for every
      instance of its class; or
    * a string frontend class name (``"WgmmaType"``) -- an explicit class-wide
      alias chosen deliberately by the caller.

    Exact-value registration never installs a class-name alias: distinct
    parameterizations (different ``Struct``/``VectorType``/special-struct
    instances) cannot collide through ``type(key).__name__``.
    """
    if not (isinstance(structure_type, type) and issubclass(structure_type, ctypes.Structure)):
        raise TypeError("structure_type must be a ctypes.Structure subclass, got "
                        f"{structure_type!r}")
    if isinstance(key, str):
        _STRUCT_LAYOUTS_BY_NAME[key] = structure_type
    else:
        _STRUCT_LAYOUTS[key] = structure_type


def _native_vector_alignment(element_bits: int, lanes: int) -> int:
    """CUDA native vector alignment: element size, capped at 16 bytes.

    These synthesized layouts are *top-level* argument byte storage; matching
    the native alignment (uint2=8, uint4=16) keeps their size/offset identical
    to the emitted CUDA vector type rather than the host struct default of 4.
    """
    return min(16, max(1, element_bits // 8 * lanes))


def _synthesize_structure(name: str, fields: Sequence) -> type:
    """Build a concrete ctypes layout from ordered (field_name, scalar) pairs.

    ctypes applies the platform C struct alignment/padding rules, so the
    generated layout matches the matching CUDA struct by-value ABI.  Fields
    must themselves have a host scalar ABI; nested vector/structure fields are
    rejected here with explicit-layout guidance rather than mispacked against
    the host's (different) struct alignment.
    """
    c_fields = []
    for field_name, field_type in fields:
        try:
            field_ctype = ctypes_type_for_scalar(field_type)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Cannot synthesize host layout for field {field_name!r} of "
                            f"{name!r}: {exc}. Register an explicit ctypes.Structure "
                            "adapter with register_struct_layout(); nested vector/struct "
                            "fields must not assume host ctypes alignment matches CUDA") from exc
        c_fields.append((str(field_name), field_ctype))
    return type(name, (ctypes.Structure, ), {"_fields_": c_fields})


def _synthesize_vector(name: str, element_type: Any, lanes: int) -> type:
    """Top-level native vector byte storage with CUDA vector alignment."""
    element_ctype = ctypes_type_for_scalar(element_type)
    alignment = _native_vector_alignment(int(getattr(element_type, "bits", 0)), lanes)
    return type(
        name,
        (ctypes.Structure, ),
        {
            "_align_": alignment,
            "_fields_": [(f"v{i}", element_ctype) for i in range(lanes)],
        },
    )


def _known_struct_layout(ann: Any):
    """Resolve a frontend structure/vector annotation to a concrete layout."""
    if ann in _STRUCT_LAYOUTS:
        return _STRUCT_LAYOUTS[ann]
    type_name = type(ann).__name__

    # Only *explicit* string-keyed registrations apply to a whole class.
    if type_name in _STRUCT_LAYOUTS_BY_NAME:
        return _STRUCT_LAYOUTS_BY_NAME[type_name]

    # Frontend Struct[(name, LLType), ...]: synthesize each distinct instance
    # independently -- never keyed by the shared "StructType" class name.
    if getattr(ann, "is_struct", lambda: False)():
        layout = _synthesize_structure(f"_LKStruct_{id(ann) & 0xFFFFFFFF:08x}", ann.type_tuple)
        _STRUCT_LAYOUTS[ann] = layout
        return layout

    # Fixed-width vector register values (uint2/uint4 and friends); every
    # (element_type, lanes) parameterization gets its own aligned layout.
    if _is_vector_lltype(ann):
        layout = _synthesize_vector(
            f"_LKVector_{type(ann).__name__}_{id(ann) & 0xFFFFFFFF:08x}",
            ann.element_type,
            int(ann.lanes),
        )
        _STRUCT_LAYOUTS[ann] = layout
        return layout

    raise TypeError(f"Frontend structure type {ann!r} has no concrete host layout. "
                    "Register an exact-instance adapter with register_struct_layout() "
                    "(or a deliberately string-keyed class alias), or pass the layout "
                    "through signature_from_annotations(..., context=...), instead of "
                    "relying on the argument value's size")


# -----------------------------------------------------------------------------
# Immutable parameter specification
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelParameter:
    """Specification of one kernel parameter.

    Attributes
    ----------
    name:
        Parameter name (used for keyword packing and diagnostics).
    kind:
        One of :data:`SCALAR`, :data:`POINTER`, :data:`STRUCTURE`.
    ll_type:
        Original frontend annotation (LLType/string), retained for metadata.
    c_type:
        Concrete ``ctypes`` storage class: a scalar ctype for ``scalar``,
        ``ctypes.c_void_p`` for ``pointer``, or a ``ctypes.Structure``
        subclass for ``structure`` when known statically (``None`` means any
        compatible structure instance is accepted at pack time).
    element_type:
        Pointee annotation for pointers (metadata only).
    const:
        Whether the pointer/scalar was annotated ``const``.
    address_space:
        Pointer address space (0 = global).
    grid_constant:
        Whether the parameter is a ``__grid_constant__`` structure parameter,
        which the Driver still receives as a pointer to the structure bytes.
    """

    name: str
    kind: str
    ll_type: Any = None
    c_type: Optional[type] = None
    element_type: Any = None
    const: bool = False
    address_space: int = _ADDRESS_SPACE_GLOBAL
    grid_constant: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("KernelParameter.name must be a non-empty str")
        if self.kind not in (SCALAR, POINTER, STRUCTURE):
            raise ValueError(f"KernelParameter {self.name!r}: kind must be one of "
                             f"{SCALAR!r}, {POINTER!r}, {STRUCTURE!r}, got {self.kind!r}")
        if self.kind == SCALAR and self.c_type is None:
            raise ValueError(f"Scalar parameter {self.name!r} requires a ctypes c_type")
        if self.kind == POINTER and self.c_type is not ctypes.c_void_p:
            raise ValueError(f"Pointer parameter {self.name!r} must use ctypes.c_void_p")
        if self.kind == STRUCTURE and self.c_type is not None:
            if not (isinstance(self.c_type, type) and issubclass(self.c_type, ctypes.Structure)):
                raise TypeError(f"Structure parameter {self.name!r} requires a "
                                f"ctypes.Structure subclass, got {self.c_type!r}")

    # -- constructors -------------------------------------------------------

    @classmethod
    def scalar(cls, name: str, scalar_type: Any, *, const: bool = False) -> "KernelParameter":
        """Create a by-value scalar parameter from its declared type."""
        return cls(
            name=name,
            kind=SCALAR,
            ll_type=scalar_type,
            c_type=ctypes_type_for_scalar(scalar_type),
            const=const,
        )

    @classmethod
    def pointer(
        cls,
        name: str,
        element_type: Any = None,
        *,
        const: bool = False,
        address_space: int = _ADDRESS_SPACE_GLOBAL,
    ) -> "KernelParameter":
        """Create a device pointer parameter (8-byte host address slot)."""
        return cls(
            name=name,
            kind=POINTER,
            ll_type=element_type,
            c_type=ctypes.c_void_p,
            element_type=element_type,
            const=const,
            address_space=int(address_space),
        )

    @classmethod
    def structure(
        cls,
        name: str,
        structure_type: Optional[type] = None,
        *,
        ll_type: Any = None,
        grid_constant: bool = True,
    ) -> "KernelParameter":
        """Create a by-value structure parameter.

        ``structure_type`` is an optional concrete ``ctypes.Structure``
        subclass; when omitted any compatible structure instance is accepted
        at pack time and its size is taken from the instance.
        """
        if structure_type is not None and not (isinstance(structure_type, type)
                                               and issubclass(structure_type, ctypes.Structure)):
            raise TypeError(f"structure_type for {name!r} must be a ctypes.Structure "
                            f"subclass, got {structure_type!r}")
        return cls(
            name=name,
            kind=STRUCTURE,
            ll_type=ll_type,
            c_type=structure_type,
            grid_constant=grid_constant,
        )

    @classmethod
    def from_annotation(cls, name: str, annotation: Any) -> "KernelParameter":
        """Create a parameter from a resolved frontend annotation."""
        return _parameter_from_annotation(name, annotation)

    # -- introspection ------------------------------------------------------

    @property
    def size(self) -> Optional[int]:
        """Argument buffer size in bytes; ``None`` for dynamic structures."""
        if self.c_type is None:
            return None
        return ctypes.sizeof(self.c_type)

    def __str__(self) -> str:
        target = self.c_type.__name__ if self.c_type is not None else self.ll_type
        return f"{self.name}: {self.kind}[{target}]"


# -----------------------------------------------------------------------------
# Immutable signature
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelSignature:
    """Ordered, immutable kernel parameter list."""

    parameters: Tuple[KernelParameter, ...] = field(default_factory=tuple)

    def __init__(self, parameters: Sequence[KernelParameter] = ()) -> None:
        params = tuple(parameters)
        if not all(isinstance(p, KernelParameter) for p in params):
            raise TypeError("KernelSignature parameters must be KernelParameter")
        names = [p.name for p in params]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate parameter names in signature: {names}")
        object.__setattr__(self, "parameters", params)
        object.__setattr__(self, "_names", tuple(names))

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def sizes(self) -> Tuple[Optional[int], ...]:
        return tuple(p.size for p in self.parameters)

    def parameter(self, name: str) -> KernelParameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(name)

    def __len__(self) -> int:
        return len(self.parameters)

    def __iter__(self):
        return iter(self.parameters)

    def __getitem__(self, index):
        return self.parameters[index]

    def __str__(self) -> str:
        return "KernelSignature(" + ", ".join(str(p) for p in self.parameters) + ")"


# -----------------------------------------------------------------------------
# Packing
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PackedArguments:
    """Materialized Driver argument buffers.

    ``pointers`` is a ``ctypes.c_void_p`` array suitable for casting to
    ``POINTER(c_void_p)`` (the Driver ``void**`` kernelParams).  ``storage``
    retains every ctypes value/structure the pointers address for the lifetime
    of this object -- callers must keep the :class:`PackedArguments` alive
    until ``cuLaunchKernel`` has returned.
    """

    signature: KernelSignature
    storage: Tuple[Any, ...]
    pointers: Any  # (ctypes.c_void_p * n) array

    @property
    def kernel_params(self):
        """``pointers`` cast to ``POINTER(c_void_p)`` for the Driver API."""
        if not self.storage:
            return None
        return ctypes.cast(self.pointers, ctypes.POINTER(ctypes.c_void_p))

    def argument_bytes(self) -> Tuple[bytes, ...]:
        """Copy of each argument's raw ABI bytes (deterministic checks)."""
        out = []
        for param, obj in zip(self.signature.parameters, self.storage):
            size = param.size if param.size is not None else ctypes.sizeof(obj)
            out.append(ctypes.string_at(ctypes.addressof(obj), size))
        return tuple(out)


def _checked_scalar(param: KernelParameter, value: Any) -> Any:
    c_type = param.c_type

    # Explicit ctypes scalar instances must match the declared storage exactly;
    # a c_float is never silently accepted where the kernel expects c_double.
    if isinstance(value, ctypes._SimpleCData):
        if type(value) is not c_type:
            raise TypeError(f"Parameter {param.name!r} expects {c_type.__name__}, got "
                            f"explicit {type(value).__name__}")
        return value

    if c_type is ctypes.c_bool:
        if isinstance(value, bool):
            return ctypes.c_bool(value)
        if hasattr(value, "dtype") and value.dtype == "bool":
            return ctypes.c_bool(bool(value))
        raise TypeError(f"Bool parameter {param.name!r} requires a Python bool "
                        f"(got {type(value).__name__})")

    if isinstance(value, bool):
        raise TypeError(f"Parameter {param.name!r} expects {c_type.__name__}, got bool")

    # 16-bit CUDA float storage encodes Python floats to exact raw bits.
    if isinstance(value, float) and not isinstance(value, int):
        if c_type is cuda_half:
            return c_type(half_bits(value))
        if c_type is cuda_bfloat16:
            return c_type(bfloat16_bits(value))
        if c_type not in (ctypes.c_float, ctypes.c_double):
            raise TypeError(f"Parameter {param.name!r} expects {c_type.__name__}, "
                            f"got float")
        return c_type(value)

    if isinstance(value, int):
        if c_type in (ctypes.c_float, ctypes.c_double):
            raise TypeError(f"Parameter {param.name!r} expects {c_type.__name__}, "
                            f"got int (pass an explicit float)")
        if c_type in (cuda_half, cuda_bfloat16):
            # Bare ints are accepted as explicit raw 16-bit bit patterns.
            if not 0 <= value <= 0xFFFF:
                raise OverflowError(f"Raw bits {value} for parameter {param.name!r} "
                                    f"({c_type.__name__}) must be in [0, 65535]")
            return c_type(value)
        lo, hi = _SCALAR_RANGES.get(c_type, (None, None))
        if lo is not None and not (lo <= value <= hi):
            raise OverflowError(f"Value {value} for parameter {param.name!r} "
                                f"({c_type.__name__}) is outside [{lo}, {hi}]")
        try:
            return c_type(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise OverflowError(f"Value {value} cannot be stored in {c_type.__name__} "
                                f"for parameter {param.name!r}") from exc

    if hasattr(value, "item"):
        # NumPy scalar: convert through its Python value with the same checks.
        return _checked_scalar(param, value.item())

    raise TypeError(f"Parameter {param.name!r} expects a value compatible with "
                    f"{c_type.__name__}, got {type(value).__name__}")


def _pointer_address(param: KernelParameter, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"Pointer parameter {param.name!r} cannot take a bool")
    if isinstance(value, int):
        address = value
    elif hasattr(value, "item") and not isinstance(value, ctypes.c_void_p):
        # NumPy integer address.
        address = int(value.item())
    elif isinstance(value, ctypes.c_void_p):
        address = value.value or 0
    elif isinstance(value, (ctypes._Pointer, ctypes.Array)):
        address = ctypes.cast(value, ctypes.c_void_p).value or 0
    else:
        # Any other ctypes pointer/array/structure: take its address. Raw
        # NumPy/Torch buffer adaptation lives in the runtime layer.
        try:
            address = ctypes.cast(value, ctypes.c_void_p).value
        except (ctypes.ArgumentError, TypeError) as exc:
            raise TypeError(f"Pointer parameter {param.name!r} expects an integer "
                            f"address or ctypes pointer, got {type(value).__name__}") from exc
        if address is None:
            address = 0
    if not (0 <= address <= (1 << 64) - 1):
        raise OverflowError(f"Pointer address for {param.name!r} must be an unsigned 64-bit "
                            f"value, got {address}")
    return int(address)


def _checked_structure(param: KernelParameter, value: Any) -> Any:
    if not isinstance(value, ctypes.Structure):
        raise TypeError(f"Structure parameter {param.name!r} requires a ctypes.Structure "
                        f"instance, got {type(value).__name__}")
    if param.c_type is not None:
        expected_size = ctypes.sizeof(param.c_type)
        if ctypes.sizeof(value) != expected_size:
            raise TypeError(f"Structure parameter {param.name!r} expects the declared "
                            f"{param.c_type.__name__} ABI ({expected_size} bytes), got "
                            f"{type(value).__name__} ({ctypes.sizeof(value)} bytes)")
    # Copy the bytes so later caller mutation cannot change the packed ABI
    # values; storage then owns the argument for the launch lifetime.
    owned = type(value)()
    ctypes.memmove(ctypes.addressof(owned), ctypes.addressof(value), ctypes.sizeof(value))
    return owned


def pack_arguments(
    signature: KernelSignature,
    arguments: Union[Sequence[Any], Mapping[str, Any]],
) -> PackedArguments:
    """Validate and materialize Driver argument buffers.

    ``arguments`` is either an ordered sequence matching the signature or a
    mapping keyed by parameter name.  Values are range/type-checked against
    the declared parameters -- a mismatch raises :class:`TypeError`,
    :class:`ValueError` or :class:`OverflowError` rather than being silently
    narrowed or reinterpreted.
    """
    if not isinstance(signature, KernelSignature):
        raise TypeError("signature must be a KernelSignature")

    params = signature.parameters
    if isinstance(arguments, Mapping):
        missing = [p.name for p in params if p.name not in arguments]
        unknown = sorted(set(arguments) - set(signature.names))
        if missing:
            raise TypeError(f"Missing kernel arguments: {missing}")
        if unknown:
            raise TypeError(f"Unknown kernel arguments: {unknown}")
        ordered = [arguments[p.name] for p in params]
    elif isinstance(arguments, (list, tuple)):
        if len(arguments) != len(params):
            raise TypeError(f"Expected {len(params)} arguments for kernel signature, "
                            f"got {len(arguments)}")
        ordered = list(arguments)
    else:
        raise TypeError("arguments must be a sequence or a name->value mapping, got "
                        f"{type(arguments).__name__}")

    storage = []
    for param, value in zip(params, ordered):
        if param.kind == SCALAR:
            storage.append(_checked_scalar(param, value))
        elif param.kind == POINTER:
            storage.append(ctypes.c_void_p(_pointer_address(param, value)))
        else:
            storage.append(_checked_structure(param, value))

    pointers = (ctypes.c_void_p * max(1, len(storage)))()
    for i, obj in enumerate(storage):
        pointers[i] = ctypes.addressof(obj)

    return PackedArguments(
        signature=signature,
        storage=tuple(storage),
        pointers=pointers,
    )


# -----------------------------------------------------------------------------
# Annotation resolution
# -----------------------------------------------------------------------------


def pack_contiguous_arguments(signature: KernelSignature, arguments) -> bytes:
    """Use the shared typed ABI checks, then lay values out at native alignment."""
    packed = pack_arguments(signature, arguments)
    buffer = bytearray()
    for parameter, storage, payload in zip(signature.parameters, packed.storage, packed.argument_bytes()):
        alignment = ctypes.alignment(storage)
        if parameter.c_type is not None:
            alignment = max(alignment, int(getattr(parameter.c_type, "_align_", 0)))
        buffer.extend(b"\0" * (-len(buffer) % alignment))
        buffer.extend(payload)
    return bytes(buffer)


def _unwrap_modifiers(annotation: Any):
    """Strip Const/GridConstant wrappers; return (inner, const, grid_const)."""
    const = bool(getattr(annotation, "is_const", lambda: False)())
    grid_constant = bool(getattr(annotation, "is_grid_constant", lambda: False)())
    inner = getattr(annotation, "inner_type", None)
    if inner is not None and type(annotation).__name__ in (
            "Const",
            "GridConstant",
    ):
        inner_const, inner_grid, inner_t = _unwrap_modifiers(inner)
        return const or inner_const, grid_constant or inner_grid, inner_t
    return const, grid_constant, annotation


def _parameter_from_annotation(name: str, annotation: Any) -> KernelParameter:
    const, grid_constant, ann = _unwrap_modifiers(annotation)

    # Explicit ctypes scalar class annotation.
    if isinstance(ann, type) and issubclass(ann, ctypes._SimpleCData):
        return KernelParameter.scalar(name, ann, const=const)

    # Explicit ctypes structure class annotation.
    if isinstance(ann, type) and issubclass(ann, ctypes.Structure):
        return KernelParameter.structure(name, ann, ll_type=annotation, grid_constant=grid_constant)

    if ann is int:
        return KernelParameter.scalar(name, ctypes.c_int64, const=const)
    if ann is float:
        return KernelParameter.scalar(name, ctypes.c_double, const=const)
    if ann is bool:
        return KernelParameter.scalar(name, ctypes.c_bool, const=const)

    if _is_tensor_lltype(ann):
        return KernelParameter.pointer(name, getattr(ann, "element_type", None), const=const)
    if _is_pointer_lltype(ann):
        return KernelParameter.pointer(name, getattr(ann, "inner_type", None), const=const)
    if _is_vector_lltype(ann):
        layout = _known_struct_layout(ann)
        return KernelParameter.structure(
            name,
            layout,
            ll_type=annotation,
            grid_constant=grid_constant,
        )
    if _is_structure_lltype(ann):
        layout = _known_struct_layout(ann)
        return KernelParameter.structure(
            name,
            layout,
            ll_type=annotation,
            grid_constant=grid_constant,
        )
    if _is_scalar_lltype(ann):
        return KernelParameter.scalar(name, ann, const=const)

    if isinstance(ann, str):
        return KernelParameter.scalar(name, ann, const=const)

    raise TypeError(f"Cannot derive a kernel ABI parameter for {name!r} from "
                    f"annotation {annotation!r}")


def _default_annotation_namespace() -> dict:
    ns: dict = {}
    try:
        import little_kernel.language as ll

        ns["ll"] = ll
        for exported in getattr(ll, "__all__", None) or vars(ll):
            value = getattr(ll, exported, None)
            if value is not None and exported not in ns:
                ns[exported] = value
        # The language module exposes ``bool_``; accept the natural ``bool``
        # spelling inside string annotations as well.
        ns.setdefault("bool", getattr(ll, "bool_", ctypes.c_bool))
    except ImportError:  # pragma: no cover - language is a core dependency
        pass
    ns["ctypes"] = ctypes
    return ns


def signature_from_annotations(
        py_func,
        constexpr_names: Sequence[str] = (),
        context: Optional[Mapping[str, Any]] = None,
) -> KernelSignature:
    """Build a :class:`KernelSignature` from a Python function's annotations.

    Resolves ``Tensor[T]``/``Pointer[T]`` to device pointers, ``Const``/
    ``grid_constant`` modifiers, scalar ``LLType`` instances, explicit
    ``ctypes`` scalars/structures and string annotations such as
    ``"ll.Tensor[ll.int64]"``.  Parameters named in ``constexpr_names`` (or
    annotated ``constexpr[...]``) are specialization-time values and removed
    from the launch signature.  ``context`` supplies extra globals for
    resolving string annotations (e.g. a concrete ``ctypes.Structure``).
    """
    if not callable(py_func):
        raise TypeError("py_func must be a callable with annotations")

    constexpr_set = set(constexpr_names)
    annotations = getattr(py_func, "__annotations__", {}) or {}
    ns = _default_annotation_namespace()
    if context:
        ns.update(dict(context))

    params = []
    for arg_name, arg in inspect.signature(py_func).parameters.items():
        if arg_name in constexpr_set:
            continue
        if arg.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if arg_name not in annotations:
            raise TypeError(f"Kernel function {py_func.__name__!r}: parameter "
                            f"{arg_name!r} has no type annotation")
        annotation = annotations[arg_name]

        if isinstance(annotation, str):
            try:
                annotation = eval(  # noqa: S307 - restricted namespace below
                    compile(annotation, f"<{py_func.__name__}:{arg_name}>", "eval"),
                    {"__builtins__": {}},
                    ns,
                )
            except Exception as exc:
                raise TypeError(f"Failed to resolve string annotation {annotation!r} "
                                f"for parameter {arg_name!r}: {exc}") from exc

        if getattr(annotation, "is_constexpr", lambda: False)():
            continue

        params.append(_parameter_from_annotation(arg_name, annotation))

    return KernelSignature(params)
