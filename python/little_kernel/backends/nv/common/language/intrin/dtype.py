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
from little_kernel.language.builtin_base import const_func, const_type_func, builtin, Builtin, BuiltinOp
from little_kernel.core.type_system import LLType, int32, uint32, int64, uint64, bool_, int8, uint8, float32, float16, float64, bfloat16, fp8_e4m3, fp8_e5m2

# Low-precision 8-bit float formats handled by the shared cast legalization
# hook (half_float_cast_code).
_FP8_TYPES = (fp8_e4m3, fp8_e5m2)
_FP8_CVT_ENUM = {fp8_e4m3: "__NV_E4M3", fp8_e5m2: "__NV_E5M2"}

# Apply const_type_func decorator to typeof after builtin


def _fp8_cast_code(val_cpp, src_type, dst_type):
    """Lowering for casts involving an 8-bit float (fp8_e4m3 / fp8_e5m2).

    Torch CUDA extension builds additionally define
    ``__CUDA_NO_FP8_CONVERSIONS__``: under that macro (mirroring the 16-bit
    case above) the converting constructors and conversion operators of
    ``__nv_fp8_e4m3``/``__nv_fp8_e5m2`` in <cuda_fp8.h> are deleted, so
    ``static_cast<__nv_fp8_e4m3>(f)`` and ``static_cast<float>(e)`` are both
    ill-formed.  The ``__nv_cvt_*`` intrinsics exist regardless of the
    macros, so any cast whose source or target is fp8 must go through them.

    Narrowing (any scalar -> fp8): the value is staged through ``float``
    (the same conversion sequence the disabled header constructors perform),
    then converted with ``__nv_cvt_float_to_fp8(x, __NV_SATFINITE, E4M3|E5M2)``.
    SATFINITE + RN matches the header converting constructor's device
    semantics (round-to-nearest-even, saturate to the max finite magnitude,
    NaN-preserving), so this is a legality fix, not a semantics change.

    Widening (fp8 -> any scalar): ``__nv_cvt_fp8_to_halfraw`` +
    ``__half2float`` widen exactly (every finite fp8 value is exactly
    representable in fp16/f32), followed by the usual C++ conversion to the
    destination type.

    fp8 -> fp8 of the other flavour widens to float then re-narrows — the
    same double conversion the CUDA header's cross-flavour constructor
    performs.  Returns None for same-type casts (bit copy, keep the
    historical ``static_cast``).
    """
    includes = ["<cuda_fp8.h>", "<cuda_fp16.h>"]
    # Source widening: fp8 -> (exactly float) -> destination.
    if src_type in _FP8_TYPES:
        # __nv_cvt_fp8_to_halfraw needs the raw storage byte.  Under the
        # no-conversion macros a numeric cast off the wrapper is ill-formed;
        # the public `__x` storage member is always available instead.
        raw = f"({val_cpp}).__x"
        from_f = (f"__half2float(__nv_cvt_fp8_to_halfraw({raw}, "
                  f"{_FP8_CVT_ENUM[src_type]}))")
        if dst_type == float32:
            return from_f, includes
        if dst_type == float16:
            return f"__float2half_rn({from_f})", includes
        if dst_type == bfloat16:
            return f"__float2bfloat16_rn({from_f})", ["<cuda_fp8.h>", "<cuda_fp16.h>", "<cuda_bf16.h>"]
        if dst_type in _FP8_TYPES:
            # Cross-flavour fp8 -> float -> fp8 (double narrowing matches the
            # header's cross constructor; RN+SATFINITE at both steps).  The
            # __nv_fp8_storage_t result initializes the destination wrapper
            # via its (always available) storage constructor.
            return (f"__nv_cvt_float_to_fp8({from_f}, __NV_SATFINITE, "
                    f"{_FP8_CVT_ENUM[dst_type]})", includes)
        if dst_type == float64:
            # Widen exactly fp8 -> half -> float -> double.
            return f"static_cast<double>({from_f})", includes
        # Integer and boolean targets: widen exactly to float, then the
        # ordinary C++ conversion (exactly the disabled conversion-operator
        # sequence int(fp8) == int(float(fp8))).
        from little_kernel.codegen.registries.type_converter import TypeConverter
        dst_cpp = TypeConverter().lltype_to_cpp(dst_type)
        return f"static_cast<{dst_cpp}>({from_f})", includes
    # Destination narrowing: any scalar -> fp8, staged through float.
    if dst_type in _FP8_TYPES:
        conv = "__nv_cvt_float_to_fp8"
        enum = _FP8_CVT_ENUM[dst_type]
        if src_type == float32:
            arg = val_cpp
        elif src_type == float16:
            arg = f"__half2float({val_cpp})"
        elif src_type == bfloat16:
            arg = f"__bfloat162float({val_cpp})"
            includes = ["<cuda_fp8.h>", "<cuda_fp16.h>", "<cuda_bf16.h>"]
        else:
            # float64, integer and boolean sources: the (macro-disabled)
            # CUDA header constructors also convert through float first.
            arg = f"static_cast<float>({val_cpp})"
        # __nv_cvt_float_to_fp8 yields __nv_fp8_storage_t; the fp8 wrapper's
        # storage-type constructor is NOT disabled by the no-conversion
        # macros, so this rvalue still initializes an __nv_fp8_* lvalue.
        return f"{conv}({arg}, __NV_SATFINITE, {enum})", includes
    return None


def half_float_cast_code(val_cpp, src_type, dst_type):
    """Type-aware lowering for casts whose *target* is a low-precision float
    (float16/bfloat16, and the 8-bit fp8_e4m3/fp8_e5m2 formats).

    Torch CUDA extension builds define ``__CUDA_NO_HALF_CONVERSIONS__`` and
    ``__CUDA_NO_BFLOAT16_CONVERSIONS__``, which delete every converting
    constructor/operator of ``__half``/``__nv_bfloat16``.  Under those macros
    ``static_cast<__nv_bfloat16>(f)`` (and the ``__half`` twin) is ill-formed,
    so a target-16-bit ``val_cast``/``to``/type-constructor must be lowered to
    the CUDA conversion intrinsics, which exist regardless of the macros.

    The symmetric direction fails for the same reason: with conversion
    operators deleted, ``static_cast<float>(__nv_bfloat16)`` is ill-formed
    too.  Casting a 16-bit float source to float32/float64 therefore lowers
    to ``__half2float``/``__bfloat162float`` (with a ``static_cast`` to widen
    to double), also a legality-only change.

    ``src_type``/``dst_type`` are ``LLType`` instances (or None when the
    source expression's type is unknown).  Returns ``(code, includes)`` when
    the cast can be lowered through intrinsics, else ``None`` so the caller
    falls back to ``static_cast`` (targets/sources not involving a
    low-precision float, unknown source types, and same-type casts keep the
    historical behaviour).

    Rounding: ``__float2half_rn``/``__float2bfloat16_rn`` are the same
    round-to-nearest-even conversions the CUDA header constructors perform on
    device, so this is a legality fix, not a semantics change.  Integer, bool
    and float64 sources stage through ``float`` explicitly — exactly the
    conversion sequence of the (disabled) header constructors.
    """
    if not isinstance(dst_type, LLType):
        return None
    if not isinstance(src_type, LLType) or src_type == dst_type:
        return None
    if src_type in _FP8_TYPES or dst_type in _FP8_TYPES:
        return _fp8_cast_code(val_cpp, src_type, dst_type)
    includes = ["<cuda_fp16.h>", "<cuda_bf16.h>"]
    # 16-bit float -> float32/float64: the widening intrinsics remain valid
    # under the no-conversion macros; static_cast does not.
    if src_type in (float16, bfloat16) and dst_type in (float32, float64):
        widen = "__half2float" if src_type == float16 else "__bfloat162float"
        if dst_type == float64:
            return (f"static_cast<double>({widen}({val_cpp}))", includes)
        return f"{widen}({val_cpp})", includes
    if dst_type not in (float16, bfloat16):
        return None
    if dst_type == float16:
        conv = "__float2half_rn"
    else:
        conv = "__float2bfloat16_rn"
    if src_type == float32:
        arg = val_cpp
    elif src_type == float16:
        arg = f"__half2float({val_cpp})"
    elif src_type == bfloat16:
        arg = f"__bfloat162float({val_cpp})"
    else:
        # float64, integer and boolean sources: the (macro-disabled) CUDA
        # header constructors also convert through float, preserve that.
        arg = f"static_cast<float>({val_cpp})"
    return f"{conv}({arg})", includes


# ==============================================================================
# Special float constants
# ==============================================================================
