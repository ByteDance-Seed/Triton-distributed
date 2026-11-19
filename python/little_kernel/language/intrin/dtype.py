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
"""Target-independent type queries and casts using an emission policy."""
from little_kernel.language.builtin_base import const_func, const_type_func, builtin, Builtin, BuiltinOp
from little_kernel.core.type_system import LLType, TensorType, int32, uint32, int64, uint64, bool_, int8, uint8, float32, float16, float64, bfloat16, fp8_e4m3, fp8_e5m2


@builtin
class sizeof(BuiltinOp):
    """Get the size of a type in bytes."""

    @staticmethod
    def py_impl(dtype):
        """Python implementation for const_fold."""
        if isinstance(dtype, LLType):
            return dtype.bits // 8
        # For string type names, try to resolve
        type_sizes = {
            'float': 4,
            'double': 8,
            'int': 4,
            'long': 8,
            'int8_t': 1,
            'int16_t': 2,
            'int32_t': 4,
            'int64_t': 8,
            'uint8_t': 1,
            'uint16_t': 2,
            'uint32_t': 4,
            'uint64_t': 8,
            '__half': 2,
            '__nv_bfloat16': 2,
            'half': 2,
            'bfloat16': 2,
        }
        if isinstance(dtype, str) and dtype in type_sizes:
            return type_sizes[dtype]
        raise ValueError(f"Cannot determine size of {dtype}")

    @staticmethod
    def return_type(dtype):
        return int32

    @staticmethod
    def codegen(dtype):
        return Builtin(body="", includes=[], return_val=f"sizeof({dtype})")


sizeof = const_func(sizeof)


@builtin
class typeof(BuiltinOp):
    """Get the type of value. This is interpreted by const_fold pass."""

    @staticmethod
    def return_type(val_type):
        return val_type

    @staticmethod
    def codegen(val):
        return Builtin(body="", includes=[], return_val=f"decltype({val})")


typeof = const_type_func(typeof)


@builtin
class val_cast(BuiltinOp):
    """Cast a value to a specific type.
    
    Usage:
        val_cast(value, ll.uint32)  # Cast value to uint32
        ll.uint32(value)  # Same as above (via __call__)
    """

    @staticmethod
    def return_type(val_type, dtype):
        return dtype if isinstance(dtype, LLType) else val_type

    @staticmethod
    def codegen(val, dtype):
        # dtype can be a string (C++ type name) or an LLType object
        if isinstance(dtype, str):
            cpp_type = dtype
        else:
            # Convert LLType to C++ type name
            from little_kernel.codegen.policy import cpp_type as _lltype_to_cpp
            cpp_type = _lltype_to_cpp(dtype)
        return Builtin(body="", includes=[], return_val=f"static_cast<{cpp_type}>({val})")


@builtin
class to(BuiltinOp):
    """Convert a value to a specific type.
    
    Usage:
        value.to(ll.uint32)  # Convert value to uint32
    """

    @staticmethod
    def return_type(val_type, dtype):
        return dtype if isinstance(dtype, LLType) else val_type

    @staticmethod
    def codegen(val, dtype):
        # This is called as val.to(ll.uint32), so val is the first argument
        # and dtype is the second argument
        if isinstance(dtype, str):
            cpp_type = dtype
        else:
            from little_kernel.codegen.policy import cpp_type as _lltype_to_cpp
            cpp_type = _lltype_to_cpp(dtype)
        return Builtin(body="", includes=[], return_val=f"static_cast<{cpp_type}>({val})")


@builtin
class tensor_view(BuiltinOp):
    """Typed element view of an existing tensor or allocation.

    ``tensor_view(src, dtype)`` returns the same underlying storage
    reinterpreted as a ``Tensor[dtype]``: no data movement, no numeric
    conversion.  Unlike ``ptr_cast`` (whose return type is the target
    *scalar* type — it exists for raw pointer expressions such as TMA
    descriptors) the result here carries a genuine ``TensorType``, so
    dtype-keyed consumers — the typed ``x[i]`` subscript load/store path,
    ``gmem_ld_scalar`` cache-policy variants, byte staging helpers —
    accept it as their first argument.

    Usage:
        xin = ll.tensor_view(x, ll.uint16)    # uint8 kernel arg as uint16
        v: ll.uint16 = ll.gmem_ld_scalar(xin, i, 'nc')
        s8 = ll.tensor_view(s32, ll.uint8)    # uint32 shared array as bytes
        s8[i] = b

    The caller is responsible for the element-count reinterpretation
    (viewing ``n`` uint32 elements as uint8 exposes ``4*n`` elements);
    alignment follows from the source allocation/argument, which is at
    least element-aligned.  At emission the view becomes a
    ``reinterpret_cast<dtype*>(src)``; for a shared-memory array the
    source identifier decays to its base pointer, so downstream
    ``view[i]`` and ``view + off`` expressions type-check against C++
    exactly as they do for a native ``Tensor[dtype]`` argument.
    """

    @staticmethod
    def return_type(src_type, dtype):
        if not isinstance(src_type, TensorType):
            raise TypeError(f"tensor_view expects a Tensor source, got "
                            f"{type(src_type).__name__}: {src_type}")
        if not isinstance(dtype, LLType):
            raise TypeError(f"tensor_view expects an LLType element dtype, got "
                            f"{type(dtype).__name__}: {dtype}")
        return TensorType(dtype)

    @staticmethod
    def codegen(src, dtype):
        # dtype can be a C++ type-name string (the common case: call args
        # are visited to their C++ text before this runs) or an LLType
        # object; mirror val_cast/to for the conversion.
        if isinstance(dtype, str):
            cpp_type = dtype
        else:
            from little_kernel.codegen.policy import cpp_type as _lltype_to_cpp
            cpp_type = _lltype_to_cpp(dtype)
        return Builtin(body="", includes=[], return_val=f"reinterpret_cast<{cpp_type}*>({src})")


@builtin
class bitcast(BuiltinOp):
    """Bits-preserving reinterpretation of a scalar value to another dtype.

    Unlike ``val_cast``/``to`` (numeric conversion, ``static_cast``) and
    ``ptr_cast`` (pointer reinterpretation), ``bitcast`` keeps the source
    bit pattern unchanged and only changes how it is typed. Source and
    destination dtypes must have equal bit width; this is checked when
    both widths are statically known (see ``return_type``).

    This is the target-independent lowering of the Python ``x.view(dtype)``
    idiom on scalars, e.g. inserting an E8M0 biased-exponent byte into the
    FP32 exponent field of a power-of-two scale: ``bitcast(b << 23, f32)``
    is exactly ``ldexp(1.0, b - 127)`` with no SFU approximation.

    Usage:
        val = ll.bitcast(u32_value, ll.float32)   # uint32 bits -> float32

    Emission uses ``__builtin_bit_cast`` (GCC/Clang/NVCC/HIPCC C++17), which
    for scalar widths compiles to a register rename with no memory traffic.
    """

    @staticmethod
    def return_type(val_type, dtype=None, **kwargs):
        # `dtype` may be passed positionally (ll.bitcast(x, ll.float32)) — the
        # second argument — or via keyword.  When the target dtype resolves to
        # an LLType, validate equal bit width here (type-inference time) so a
        # mismatch is rejected before any C++ is emitted.
        if isinstance(dtype, LLType):
            src_bits = getattr(val_type, "bits", None)
            dst_bits = getattr(dtype, "bits", None)
            if src_bits is not None and dst_bits is not None and src_bits != dst_bits:
                raise TypeError(f"ll.bitcast requires equal bit width, got {val_type} "
                                f"({src_bits} bits) -> {dtype} ({dst_bits} bits)")
            return dtype
        return val_type

    @staticmethod
    def codegen(val, dtype):
        # `dtype` reaches codegen as visited C++ text (a type-name string) in
        # the normal emitter path; an LLType only when codegen is invoked
        # directly.  Width validation lives in return_type, which the type
        # inferencer always runs before emission.
        if isinstance(dtype, str):
            cpp_dst = dtype
        else:
            from little_kernel.codegen.policy import cpp_type as _lltype_to_cpp
            cpp_dst = _lltype_to_cpp(dtype)
        return Builtin(body="", includes=[], return_val=f"__builtin_bit_cast({cpp_dst}, ({val}))")


@builtin
class ptr_cast(BuiltinOp):
    """Reinterpret cast a pointer to a different type."""

    @staticmethod
    def return_type(val_type, dtype):
        return dtype

    @staticmethod
    def codegen(val, dtype):
        # dtype can be a string (C++ type name, the common case: call args
        # are visited to their C++ text before this runs) or an LLType object.
        if isinstance(dtype, str):
            cpp_type = dtype
        else:
            from little_kernel.codegen.registries.type_converter import TypeConverter
            cpp_type = TypeConverter().lltype_to_cpp(dtype)
        return Builtin(body="", includes=[], return_val=f"reinterpret_cast<{cpp_type}>({val})")


@builtin
class neg_infinity(BuiltinOp):
    """Return -INFINITY as float32.
    
    Usage:
        x = ll.neg_infinity()  # Returns -INFINITY (float)
    """

    @staticmethod
    def return_type():
        return float32

    @staticmethod
    def codegen():
        return Builtin(body="", includes=["<cmath>"], return_val="-INFINITY")


@builtin
class pos_infinity(BuiltinOp):
    """Return INFINITY as float32.
    
    Usage:
        x = ll.pos_infinity()  # Returns INFINITY (float)
    """

    @staticmethod
    def return_type():
        return float32

    @staticmethod
    def codegen():
        return Builtin(body="", includes=["<cmath>"], return_val="INFINITY")


def ll_type_to_torch_type(dtype):
    import torch
    if dtype == int32:
        return torch.int32
    elif dtype == uint32:
        return torch.uint32
    elif dtype == int64:
        return torch.int64
    elif dtype == uint64:
        # PyTorch does not fully support torch.uint64; use int64 as fallback
        return torch.int64
    elif dtype == bool_:
        return torch.bool
    elif dtype == int8:
        return torch.int8
    elif dtype == uint8:
        return torch.uint8
    elif dtype == float32:
        return torch.float32
    elif dtype == bfloat16:
        return torch.bfloat16
    elif dtype == float16:
        return torch.float16
    elif dtype == float64:
        return torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
