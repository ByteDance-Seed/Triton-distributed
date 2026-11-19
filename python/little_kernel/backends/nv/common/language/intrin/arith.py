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
from little_kernel.language.builtin_base import builtin, Builtin, CODEGEN_NEEDS_TYPES_ATTR
from little_kernel.language.intrin.portable import cdiv, align_power_of_2, min_val, max_val, select
from little_kernel.core.compile import ll_kernel
from little_kernel.core.type_system import LLType, TensorType, int32, uint16, uint32, float32, float64, float16, bfloat16


def codegen_sum(values):
    body = r"""
template <typename T, int N>
__device__ __forceinline__ T register_array_sum(const T (&values)[N]) {
    T partial[N];
    #pragma unroll
    for (int i = 0; i < N; ++i) partial[i] = values[i];
    #pragma unroll
    for (int stride = 1; stride < N; stride *= 2) {
        #pragma unroll
        for (int i = 0; i + stride < N; i += 2 * stride) partial[i] += partial[i + stride];
    }
    return partial[0];
}
"""
    return Builtin(body=body, includes=[], return_val=f"register_array_sum({values})")


def _sum_return_type(values_type):
    if not isinstance(values_type, TensorType):
        raise TypeError("ll.sum expects a register array")
    return values_type.element_type


@builtin(eval_return_type=_sum_return_type, codegen_func=codegen_sum)
def sum(values):
    """Sum a local register array using an unrolled balanced tree."""
    raise RuntimeError("should not call sum in compilation")


@ll_kernel(backend="cuda", is_entry=False)
def align_up_u32(value: uint32, alignment: uint32) -> uint32:
    """Round a runtime uint32 value up to an arbitrary positive alignment."""
    return cdiv(value, alignment) * alignment


def codegen_ldg(ptr):
    return Builtin(body="", includes=[], return_val=f"__ldg({ptr})")


@builtin(eval_return_type=lambda ptr_type: ptr_type, codegen_func=codegen_ldg)
def ldg(ptr):
    """Cache-qualified global load (__ldg)."""
    raise RuntimeError("should not call ldg in compilation")


# ==============================================================================
# Type conversion intrinsics
# ==============================================================================


def codegen_bf16_to_float(val):
    return Builtin(body="", includes=[], return_val=f"__bfloat162float({val})")


@builtin(eval_return_type=float32, codegen_func=codegen_bf16_to_float)
def bf16_to_float(val):
    """Convert bfloat16 value to float32."""
    raise RuntimeError("should not call bf16_to_float in compilation")


# Exactly one module-level definition exists for each public intrinsic
# (see docs/adr/0003-comment-provenance.md). The surviving bf16x2 pair below
# is the canonical one: exact bf16->f32 lane upcasts and a round-to-nearest-
# even bf16x2 pack with the low lane occupying bits [0,16).


def codegen_half_to_float(val):
    return Builtin(body="", includes=[], return_val=f"__half2float({val})")


@builtin(eval_return_type=float32, codegen_func=codegen_half_to_float)
def half_to_float(val):
    """Convert CUDA half to float32 even when implicit half conversions are disabled."""
    raise RuntimeError("should not call half_to_float in compilation")


def codegen_float_to_half(val):
    return Builtin(body="", includes=[], return_val=f"__float2half_rn({val})")


@builtin(eval_return_type=float16, codegen_func=codegen_float_to_half)
def float_to_half(val):
    """Round float32 to CUDA half using nearest-even."""
    raise RuntimeError("should not call float_to_half in compilation")


def codegen_float_to_half_bits_u16(val):
    # The __half value exists only inside this helper: __half_as_ushort is a
    # raw storage reinterpret (available regardless of
    # __CUDA_NO_HALF_CONVERSIONS__) and nvcc folds the pair into one
    # cvt.rn.f16.f32 with a .u16 destination. The caller never names __half.
    body = ("\n__device__ __forceinline__ uint16_t "
            "lk_f32_to_f16_bits_u16_fn(float val) {\n"
            "    return __half_as_ushort(__float2half_rn(val));\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"lk_f32_to_f16_bits_u16_fn({val})")


@builtin(eval_return_type=uint16, codegen_func=codegen_float_to_half_bits_u16)
def float_to_half_bits_u16(val):
    """Convert float32 to its IEEE fp16 bit pattern as uint16 (round-to-nearest-even).

    Unlike ``float_to_half`` + ``val_cast(_, ll.uint16)`` — which lowers to
    the ill-formed ``static_cast<uint16_t>((__half)h)`` (under the torch
    extension's ``__CUDA_NO_HALF_CONVERSIONS__`` build, ``__half`` has no
    conversion to integral types) — this intrinsic returns the raw 16-bit
    storage word directly, so a frontend packing fp16 header bytes into a
    uint8 body never materializes or names ``__half`` in its own IR. Extract
    the byte lanes with ordinary uint16 masks (``bits & 0xFF`` /
    ``(bits >> 8) & 0xFF``).
    """
    raise RuntimeError("should not call float_to_half_bits_u16 in compilation")


def codegen_float_to_bf16(val):
    return Builtin(body="", includes=[], return_val=f"__float2bfloat16_rn({val})")


@builtin(eval_return_type=bfloat16, codegen_func=codegen_float_to_bf16)
def float_to_bf16(val):
    """Convert float32 value to bfloat16."""
    raise RuntimeError("should not call float_to_bf16 in compilation")


# ==============================================================================
# Typed float32 literal wrapper
# ==============================================================================
#
# Python float literals lower to C++ double literals whenever the type
# inferencer does not record the Constant node as 32-bit FloatType (the typed
# `<value>f` emission only fires on a recorded 32-bit type). A bare double
# literal silently promotes a surrounding fp32 expression to FP64 via the
# usual arithmetic conversions (1:64 FP64 rate on Hopper). Frontends unable
# to rely on context write scalar fast math in inline PTX instead.
# ll.f32(x) pins the subexpression to float32 deterministically: numeric
# literals are emitted as C++ float literals (`1.5f`, `2.0f`); non-literal
# expressions get an explicit `(float)(...)` cast. Opt-in and additive: the
# existing inference-directed literal path is untouched. Design principle:
# make the element/scalar dtype explicit at the IR boundary rather than
# relying on ambient C++ promotion rules.


def _is_decimal_numeric_literal(v: str) -> bool:
    """True for decimal C++ numeric literals, optional leading sign, optional
    fraction/exponent: 0, 2, -2, .5, 1.5, -2.5e-3, 1E6. Hex literals, macros
    (INFINITY/NAN) and identifiers return False."""
    s = v.lstrip("+-").lower()
    if not s or s.startswith("0x"):
        return False
    seen_digit = seen_dot = seen_exp = False
    for i, c in enumerate(s):
        if c.isdigit():
            seen_digit = True
        elif c == "." and not seen_dot and not seen_exp:
            seen_dot = True
        elif c == "e" and seen_digit and not seen_exp and s[i + 1:]:
            seen_exp = True
        elif c in "+-" and seen_exp and s[i - 1] == "e":
            pass
        else:
            return False
    return seen_digit


def codegen_f32(val):
    """Emit `val` pinned to float32: float literal for literal args, explicit
    cast otherwise."""
    v = str(val).strip()
    if v[-1:] in ("f", "F") and _is_decimal_numeric_literal(v[:-1]):
        # Argument Constant was itself recorded float32 by inference and was
        # already emitted with the f suffix — pass through, never "1.5ff".
        return Builtin(body="", includes=[], return_val=v)
    if _is_decimal_numeric_literal(v):
        if "." not in v and "e" not in v.lower():
            v += ".0"  # "2f" is ill-formed; "2.0f" is a float literal
        return Builtin(body="", includes=[], return_val=v + "f")
    # Non-literal expression (name, call, INFINITY/NAN macro, ...).
    return Builtin(body="", includes=[], return_val=f"(float)({v})")


@builtin(eval_return_type=float32, codegen_func=codegen_f32)
def f32(val):
    """Force `val` into float32 at codegen, emitting a `f`-suffixed literal
    (or explicit cast) so no silent FP64 promotion can occur. The wrapped
    expression is typed float32 for downstream inference."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    raise RuntimeError("should not call f32 in compilation")


# ==============================================================================
# Correctly-rounded scalar arithmetic (IEEE RN, immune to --use_fast_math)
# ==============================================================================
#
# With nvcc `--use_fast_math`, `/`, sqrt, etc. lower to approximate
# instructions (div.approx.f32) and mul/add may be fused. Kernels with a
# bitwise numerics contract (e.g. quantization quotients feeding an RNE
# pack) need the IEEE correctly-rounded op at specific points. These
# builtins map 1:1 onto CUDA's __f*_rn device intrinsics, which nvcc never
# rewrites: pure scalar ops, no aliasing/alignment contract.


def codegen_add_rn_f32(a, b):
    return Builtin(body="", includes=[], return_val=f"__fadd_rn({a}, {b})")


@builtin(eval_return_type=float32, codegen_func=codegen_add_rn_f32)
def add_rn_f32(a, b):
    """Correctly-rounded (RN) float32 addition; not fma-fusable."""
    raise RuntimeError("should not call add_rn_f32 in compilation")


def codegen_sub_rn_f32(a, b):
    return Builtin(body="", includes=[], return_val=f"__fsub_rn({a}, {b})")


@builtin(eval_return_type=float32, codegen_func=codegen_sub_rn_f32)
def sub_rn_f32(a, b):
    """Correctly-rounded (RN) float32 subtraction; not fma-fusable."""
    raise RuntimeError("should not call sub_rn_f32 in compilation")


def codegen_mul_rn_f32(a, b):
    return Builtin(body="", includes=[], return_val=f"__fmul_rn({a}, {b})")


@builtin(eval_return_type=float32, codegen_func=codegen_mul_rn_f32)
def mul_rn_f32(a, b):
    """Correctly-rounded (RN) float32 multiplication; not fma-fusable."""
    raise RuntimeError("should not call mul_rn_f32 in compilation")


def codegen_div_rn_f32(a, b):
    return Builtin(body="", includes=[], return_val=f"__fdiv_rn({a}, {b})")


@builtin(eval_return_type=float32, codegen_func=codegen_div_rn_f32)
def div_rn_f32(a, b):
    """Correctly-rounded (RN) float32 division. Survives --use_fast_math
    (never lowered to div.approx.f32). Required by kernels with a bitwise
    quotient contract."""
    raise RuntimeError("should not call div_rn_f32 in compilation")


def codegen_div_rn_f64(a, b):
    # __ddiv_rn is never rewritten by --use_fast_math (f64 has no
    # div.approx.f64 fast-math lowering, but nvcc may still contract
    # surrounding ops; the intrinsic pins the division itself to IEEE RN).
    return Builtin(body="", includes=[], return_val=f"__ddiv_rn({a}, {b})")


@builtin(eval_return_type=float64, codegen_func=codegen_div_rn_f64)
def div_rn_f64(a, b):
    """Correctly-rounded (round-to-nearest-even) IEEE float64 division.
    Survives --use_fast_math: never approximated, never contracted. Required
    when a bitwise fp64 quotient contract holds (e.g. mean_t =
    float32(sum_f64 / (double)N) reproducing float32 thresholds); rewriting
    as sum * (1.0/N) is reciprocal-rounded, not bit-identical."""
    raise RuntimeError("should not call div_rn_f64 in compilation")


def codegen_fma_rn_f32(a, b, c):
    return Builtin(body="", includes=[], return_val=f"__fmaf_rn({a}, {b}, {c})")


@builtin(eval_return_type=float32, codegen_func=codegen_fma_rn_f32)
def fma_rn_f32(a, b, c):
    """Float32 fused multiply-add with a single RN rounding."""
    raise RuntimeError("should not call fma_rn_f32 in compilation")


# ==============================================================================
# Native scalar fp16 arithmetic (single-round RN, no-half-operators safe)
# ==============================================================================
#
# `__half * __half` (and +, -) only compile via the operator overloads in
# <cuda_fp16.h>; the pinned task build passes
# -D__CUDA_NO_HALF_OPERATORS__ (host-side half operators disabled), under
# which nvcc reports "no operator \"*\" matches these operands: __half *
# __half". The explicit __hmul/__hadd/__hsub device functions are NOT
# removed by that macro (it only strips the C++ operator overloads), map
# to the native scalar fp16 op, and round exactly once round-to-nearest-
# even — there is no f32 round trip and no FMA contraction. These
# builtins are the typed fp16 scalar twins of the _rn_f32 family; the
# CUDA emitter also legalizes plain `a * b` / `a + b` / `a - b` on two
# fp16 scalars to these same intrinsics, so frontends need neither the
# operators nor hand-packed f16x2 bit patterns
# (f16x2_reinterpret_u32 -> mul_f16x2 -> reinterpret_f16x2_lo chains).


def codegen_add_rn_f16(a, b):
    return Builtin(body="", includes=["<cuda_fp16.h>"], return_val=f"__hadd({a}, {b})")


@builtin(eval_return_type=float16, codegen_func=codegen_add_rn_f16)
def add_rn_f16(a, b):
    """Native fp16 scalar addition with a single round-to-nearest-even
    rounding (__hadd). Compiles under -D__CUDA_NO_HALF_OPERATORS__ and is
    not fma-fusable."""
    raise RuntimeError("should not call add_rn_f16 in compilation")


def codegen_sub_rn_f16(a, b):
    return Builtin(body="", includes=["<cuda_fp16.h>"], return_val=f"__hsub({a}, {b})")


@builtin(eval_return_type=float16, codegen_func=codegen_sub_rn_f16)
def sub_rn_f16(a, b):
    """Native fp16 scalar subtraction with a single round-to-nearest-even
    rounding (__hsub). Compiles under -D__CUDA_NO_HALF_OPERATORS__ and is
    not fma-fusable."""
    raise RuntimeError("should not call sub_rn_f16 in compilation")


def codegen_mul_rn_f16(a, b):
    return Builtin(body="", includes=["<cuda_fp16.h>"], return_val=f"__hmul({a}, {b})")


@builtin(eval_return_type=float16, codegen_func=codegen_mul_rn_f16)
def mul_rn_f16(a, b):
    """Native fp16 scalar multiplication with a single round-to-nearest-
    even rounding (__hmul). Compiles under
    -D__CUDA_NO_HALF_OPERATORS__ and is not fma-fusable."""
    raise RuntimeError("should not call mul_rn_f16 in compilation")


# ==============================================================================
# SIMD intrinsics
# ==============================================================================


def codegen_bf16x2_lo_f32(bits):
    body = ("\n__device__ __forceinline__ float lk_bf16x2_lo_f32_fn(uint32_t bits) {\n"
            "    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&bits);\n"
            "    return __low2float(v);\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"lk_bf16x2_lo_f32_fn({bits})")


@builtin(eval_return_type=float32, codegen_func=codegen_bf16x2_lo_f32)
def bf16x2_lo_f32(bits):
    """Unpack the low bf16 lane of a packed bf16x2 uint32 pattern to float32.

    Exact upcast (no rounding); pairs with gmem_ld_bf16x2_u32 so two bf16
    elements arrive in one 32-bit transaction.
    """
    raise RuntimeError("should not call bf16x2_lo_f32 in compilation")


def codegen_bf16x2_hi_f32(bits):
    body = ("\n__device__ __forceinline__ float lk_bf16x2_hi_f32_fn(uint32_t bits) {\n"
            "    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&bits);\n"
            "    return __high2float(v);\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"lk_bf16x2_hi_f32_fn({bits})")


@builtin(eval_return_type=float32, codegen_func=codegen_bf16x2_hi_f32)
def bf16x2_hi_f32(bits):
    """Unpack the high bf16 lane of a packed bf16x2 uint32 pattern to float32.

    Exact upcast (no rounding); pairs with gmem_ld_bf16x2_u32.
    """
    raise RuntimeError("should not call bf16x2_hi_f32 in compilation")


def codegen_bf16x2_pack_u32(lo, hi):
    body = ("\n__device__ __forceinline__ uint32_t lk_bf16x2_pack_u32_fn(float lo, float hi) {\n"
            "    const __nv_bfloat162 v = __float22bfloat162_rn(make_float2(lo, hi));\n"
            "    return *reinterpret_cast<const uint32_t*>(&v);\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"lk_bf16x2_pack_u32_fn({lo}, {hi})")


@builtin(eval_return_type=uint32, codegen_func=codegen_bf16x2_pack_u32)
def bf16x2_pack_u32(lo, hi):
    """Pack two float32 values into one bf16x2 uint32 pattern (round-to-nearest).

    Rounding happens once here; feed the result to gmem_st_bf16x2_u32 so a
    kernel has a single bf16 rounding boundary.
    """
    raise RuntimeError("should not call bf16x2_pack_u32 in compilation")


# ==============================================================================
# FP8 (e4m3fn) conversion intrinsics — exact hardware conversion chain
# ==============================================================================


def codegen_cvt_e4m3x2_to_f16x2(bits):
    body = ("\n__device__ __forceinline__ uint32_t lk_cvt_e4m3x2_to_f16x2_fn(uint32_t bits) {\n"
            "    uint32_t d;\n"
            '    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"((unsigned short)(bits & 0xFFFFu)));\n'
            "    return d;\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"lk_cvt_e4m3x2_to_f16x2_fn({bits})")


@builtin(eval_return_type=uint32, codegen_func=codegen_cvt_e4m3x2_to_f16x2)
def cvt_e4m3x2_to_f16x2(bits):
    """Convert two packed e4m3fn codes to a packed f16x2 uint32 pattern.

    Lowers to the single PTX instruction `cvt.rn.f16x2.e4m3x2` (sm_89+):
    the LOW 16 bits of `bits` hold the two e4m3 codes (byte 0 -> low f16
    lane, byte 1 -> high f16 lane, matching little-endian byte order of a
    16-bit-aligned fp8 pair in memory); the result is the packed half2 bit
    pattern. This is the exact hardware conversion an fp8 dequant source
    contract names; use f16x2_lo_f32/f16x2_hi_f32 to continue the chain in
    fp32 (cvt.f32.f16 per lane). Hardware semantics cover every finite
    e4m3fn encoding exactly and the e4m3fn NaN encoding per the ISA, which
    an arithmetic software fallback would have to special-case. Requires a
    2-byte-aligned source pair; pairs gathered via gmem_ld_*_v4_u32 words
    are aligned by construction.
    """
    raise RuntimeError("should not call cvt_e4m3x2_to_f16x2 in compilation")


def codegen_e4m3_to_f32(code):
    body = ("\n__device__ __forceinline__ float lk_e4m3_to_f32_fn(uint32_t code) {\n"
            "    return __half2float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(code & 0xFFu), __NV_E4M3));\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_fp8.h>", "<cuda_fp16.h>"], return_val=f"lk_e4m3_to_f32_fn({code})")


@builtin(eval_return_type=float32, codegen_func=codegen_e4m3_to_f32)
def e4m3_to_f32(code):
    """Decode a single e4m3fn byte code (uint8/uint32) to float32, exactly.

    Widening chain: `__nv_cvt_fp8_to_halfraw(byte, __NV_E4M3)` +
    `__half2float`. Every finite e4m3fn value is exactly representable in
    f16/f32 and the e4m3fn NaN encoding maps per the ISA, so the result is
    bit-exact with the two-byte path cvt_e4m3x2_to_f16x2 + f16x2_lo_f32.

    The argument is the raw byte code zero-extended into a 32-bit register
    (e.g. an element of a `Tensor[ll.uint8]` holding fp8 payloads, or a byte
    extracted from a wider gmem word with shifts/masks); only the low 8 bits
    are consumed. This is the scalar entry point for fp8 tensors whose row
    stride makes 16-byte vector loads (and therefore the packed-pair decode
    chain) inapplicable: a plain per-element u8 load plus this intrinsic
    decodes a byte at ANY gmem alignment.

    Requires <cuda_fp8.h>/<cuda_fp16.h> device intrinsics only (no PTX
    inline asm), so it is legal on every arch that ships cuda_fp8.h; the
    emitted SASS lowers to the same hardware cvt sequence as the x2 form.
    """
    raise RuntimeError("should not call e4m3_to_f32 in compilation")


def codegen_e5m2_to_f32(code):
    body = ("\n__device__ __forceinline__ float lk_e5m2_to_f32_fn(uint32_t code) {\n"
            "    return __half2float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(code & 0xFFu), __NV_E5M2));\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_fp8.h>", "<cuda_fp16.h>"], return_val=f"lk_e5m2_to_f32_fn({code})")


@builtin(eval_return_type=float32, codegen_func=codegen_e5m2_to_f32)
def e5m2_to_f32(code):
    """Decode a single e5m2 byte code (uint8/uint32) to float32, exactly.

    Twin of e4m3_to_f32 for the __NV_E5M2 encoding; same widening chain
    (`__nv_cvt_fp8_to_halfraw` + `__half2float`), same bit-exactness
    guarantee, same any-alignment applicability for per-element u8 loads
    from raw-byte fp8 tensors.
    """
    raise RuntimeError("should not call e5m2_to_f32 in compilation")


def codegen_f16x2_lo_f32(bits):
    body = ("\n__device__ __forceinline__ float lk_f16x2_lo_f32_fn(uint32_t bits) {\n"
            "    const __half2 v = *reinterpret_cast<const __half2*>(&bits);\n"
            "    return __low2float(v);\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"lk_f16x2_lo_f32_fn({bits})")


@builtin(eval_return_type=float32, codegen_func=codegen_f16x2_lo_f32)
def f16x2_lo_f32(bits):
    """Unpack the low f16 lane of a packed f16x2 uint32 pattern to float32.

    Exact upcast (cvt.f32.f16, no rounding); mirror of bf16x2_lo_f32.
    """
    raise RuntimeError("should not call f16x2_lo_f32 in compilation")


def codegen_f16x2_hi_f32(bits):
    body = ("\n__device__ __forceinline__ float lk_f16x2_hi_f32_fn(uint32_t bits) {\n"
            "    const __half2 v = *reinterpret_cast<const __half2*>(&bits);\n"
            "    return __high2float(v);\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"lk_f16x2_hi_f32_fn({bits})")


@builtin(eval_return_type=float32, codegen_func=codegen_f16x2_hi_f32)
def f16x2_hi_f32(bits):
    """Unpack the high f16 lane of a packed f16x2 uint32 pattern to float32.

    Exact upcast (cvt.f32.f16, no rounding); mirror of bf16x2_hi_f32.
    """
    raise RuntimeError("should not call f16x2_hi_f32 in compilation")


def codegen_f16x2_pack_u32(lo, hi):
    body = ("\n__device__ __forceinline__ uint32_t lk_f16x2_pack_u32_fn(float lo, float hi) {\n"
            "    const __half2 v = __floats2half2_rn(lo, hi);\n"
            "    return *reinterpret_cast<const uint32_t*>(&v);\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"lk_f16x2_pack_u32_fn({lo}, {hi})")


@builtin(eval_return_type=uint32, codegen_func=codegen_f16x2_pack_u32)
def f16x2_pack_u32(lo, hi):
    """Pack two float32 values into one f16x2 uint32 pattern (round-to-nearest-even).

    Lowers to `__floats2half2_rn`, a single rounding boundary for a packed
    half2 store; mirror of bf16x2_pack_u32 and inverse of
    f16x2_lo_f32/f16x2_hi_f32. Lets a frontend keep fp16 tiles in shared or
    global memory (vector 32/64/128-bit lanes) while computing in fp32.
    """
    raise RuntimeError("should not call f16x2_pack_u32 in compilation")


def codegen_f16x2_reinterpret_u32(lo, hi):
    body = ("\n__device__ __forceinline__ uint32_t lk_f16x2_reinterpret_u32_fn(__half lo, __half hi) {\n"
            "    const __half2 v = __halves2half2(lo, hi);\n"
            "    return *reinterpret_cast<const uint32_t*>(&v);\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"lk_f16x2_reinterpret_u32_fn({lo}, {hi})")


@builtin(eval_return_type=uint32, codegen_func=codegen_f16x2_reinterpret_u32)
def f16x2_reinterpret_u32(lo, hi):
    """Bit-pack two fp16 values into one f16x2 uint32 pattern with NO numeric conversion.

    Unlike f16x2_pack_u32 (which rounds float32 inputs), the operands are
    fp16 scalars whose bit patterns move unchanged: `lo` lands in the low 16
    bits, `hi` in the high 16 bits, matching the little-endian layout of an
    in-memory fp16 pair (same lane order as f16x2_pack_u32's output and
    gmem_ld words read from fp16 tensors). Use this to stage fp16 tile bits
    (e.g. values read elementwise from an fp16 tensor) into a u32-viewed
    shared/global tile without an F2F.F32.F16 round trip, and to write exact
    test vectors. Inverse of reinterpret_f16x2_lo/reinterpret_f16x2_hi.
    """
    raise RuntimeError("should not call f16x2_reinterpret_u32 in compilation")


def codegen_reinterpret_f16x2_lo(bits):
    body = ("\n__device__ __forceinline__ __half lk_reinterpret_f16x2_lo_fn(uint32_t bits) {\n"
            "    const __half2 v = *reinterpret_cast<const __half2*>(&bits);\n"
            "    return __low2half(v);\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"lk_reinterpret_f16x2_lo_fn({bits})")


@builtin(eval_return_type=float16, codegen_func=codegen_reinterpret_f16x2_lo)
def reinterpret_f16x2_lo(bits):
    """Reinterpret the low 16 bits of an f16x2 uint32 pattern as an fp16 value.

    Pure bit move (no conversion); unlike f16x2_lo_f32 the result stays fp16,
    so it can feed f16x2_reinterpret_u32 or an fp16 tensor store bit-exactly
    (denormals, infinities and NaNs pass through unchanged). Inverse of
    f16x2_reinterpret_u32's low lane.
    """
    raise RuntimeError("should not call reinterpret_f16x2_lo in compilation")


def codegen_reinterpret_f16x2_hi(bits):
    body = ("\n__device__ __forceinline__ __half lk_reinterpret_f16x2_hi_fn(uint32_t bits) {\n"
            "    const __half2 v = *reinterpret_cast<const __half2*>(&bits);\n"
            "    return __high2half(v);\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"lk_reinterpret_f16x2_hi_fn({bits})")


@builtin(eval_return_type=float16, codegen_func=codegen_reinterpret_f16x2_hi)
def reinterpret_f16x2_hi(bits):
    """Reinterpret the high 16 bits of an f16x2 uint32 pattern as an fp16 value.

    Pure bit move (no conversion); mirror of reinterpret_f16x2_lo and inverse
    of f16x2_reinterpret_u32's high lane.
    """
    raise RuntimeError("should not call reinterpret_f16x2_hi in compilation")


def codegen_viaddmin_s32(a, b, c):
    """Generate __viaddmin_s32 intrinsic: min(a + b, c) in a single instruction."""
    return Builtin(body="", includes=[], return_val=f"__viaddmin_s32({a}, {b}, {c})")


# ==============================================================================
# Fused multiply-add (explicit, dtype-spelled)
# ==============================================================================

_FMA_TARGETS = {float32: "fmaf", float64: "fma"}


def _fma_return_type(a_type, b_type, c_type):
    dtypes = [t for t in (a_type, b_type, c_type) if isinstance(t, LLType)]
    if len(dtypes) != 3:
        raise TypeError("ll.fma expects three typed tensor/scalar expressions, not untyped "
                        f"constants: got {a_type!r}, {b_type!r}, {c_type!r}")
    if not all(t in _FMA_TARGETS for t in dtypes):
        raise TypeError("ll.fma requires float32 or float64 arguments (one rounding, "
                        f"rn); got {a_type!r}, {b_type!r}, {c_type!r}")
    if not (dtypes[0] == dtypes[1] == dtypes[2]):
        raise TypeError("ll.fma requires all arguments to have the same dtype; got "
                        f"{a_type!r}, {b_type!r}, {c_type!r}. Cast with ll.to first.")
    return dtypes[0]


def codegen_fma(a, b, c, _arg_dtypes=None):
    dtypes = list(_arg_dtypes or [])
    spelled = None
    if len(dtypes) == 3 and all(isinstance(t, LLType) for t in dtypes):
        for dtype, name in _FMA_TARGETS.items():
            if all(t == dtype for t in dtypes):
                spelled = name
                break
    if spelled is None:
        raise TypeError("ll.fma could not resolve target intrinsic from argument dtypes "
                        f"{dtypes}; expected all float32 or all float64")
    return Builtin(body="", includes=[], return_val=f"{spelled}({a}, {b}, {c})")


@builtin(eval_return_type=_fma_return_type, codegen_func=codegen_fma)
def fma(a, b, c):
    """Explicit fused multiply-add: a * b + c with a single rounding (round-
    to-nearest-even), unlike `a * b + c` which nvcc may or may not contract
    depending on flags and statement structure.

    All three arguments must share one dtype: float32 lowers to `fmaf`,
    float64 to `fma`.  Other dtypes (including float16/bfloat16 and ints)
    raise at type-inference time; there is no silent fallback to mul+add.

    Opt-in intrinsic: existing `a * b + c` expressions are unaffected.
    """
    raise RuntimeError("fma should never be called in compilation")


setattr(fma, CODEGEN_NEEDS_TYPES_ATTR, True)


@builtin(eval_return_type=int32, codegen_func=codegen_viaddmin_s32)
def viaddmin_s32(a, b, c):
    """CUDA __viaddmin_s32 intrinsic: computes min(a + b, c) in a single instruction.

    This is more efficient than separate add and min operations.
    Used in Flash Attention for causal mask computation.

    Note: this is *not* a dot-product accumulator -- the result is clamped by
    ``c``.  Exact int8->int32 dot products need ll.dp4a_s8u8_s32, whose
    a*w+acc accumulation has no min/clamp.
    """
    if isinstance(a, int) and isinstance(b, int) and isinstance(c, int):
        return min(a + b, c)
    raise RuntimeError("should not call viaddmin_s32 in compilation")


def _wrap_i32(x):
    """Wrap a python int into signed int32 range (two's complement)."""
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & 0x80000000 else x


def _dp4a_s8u8_fold(a, b, c):
    """Constant-folder matching CUDA __dp4a(int, unsigned, int).

    ``a`` packs four *signed* int8 lanes and ``b`` packs four *unsigned*
    int8 lanes, lane i living in bits 8*i..8*i+7 (little endian).  The
    result is the exact sum of the four lane products plus ``c``, wrapped
    to signed int32 -- with no min/clamp at any step.
    """
    a &= 0xFFFFFFFF
    b &= 0xFFFFFFFF
    dot = 0
    for i in range(4):
        sa = (a >> (8 * i)) & 0xFF
        if sa >= 0x80:
            sa -= 0x100
        ub = (b >> (8 * i)) & 0xFF
        dot += sa * ub
    return _wrap_i32(dot + c)


def codegen_dp4a_s8u8_s32(a, b, c):
    """Emit CUDA __dp4a: four signed-int8 x unsigned-int8 products + c.

    The explicit static casts pin the ``int __dp4a(int, unsigned int, int)``
    overload (PTX ``dp4a.s32.s8.u8``) regardless of whether the packed
    operands arrived as int32 or uint32 containers.
    """
    return Builtin(
        body="", includes=[], return_val=(f"__dp4a(static_cast<int>({a}), "
                                          f"static_cast<unsigned int>({b}), "
                                          f"static_cast<int>({c}))"))


@builtin(eval_return_type=int32, codegen_func=codegen_dp4a_s8u8_s32)
def dp4a_s8u8_s32(a, b, c):
    """Exact int8 dot-product accumulator: CUDA ``__dp4a`` (single instruction).

    ``a`` is four *signed* int8 lanes and ``b`` is four *unsigned* int8
    lanes packed into 32-bit words; returns
    ``sum(a_i * b_i for i in 0..3) + c`` as a signed int32 with no
    min/clamp or saturation.  This is the primitive for exact int8->int32
    dots (e.g. quantized GEMV/GEMM): accumulate it in a loop and every
    step is a true running sum.

    Signed x signed dots are expressible with the usual 0x80 sign-bias
    trick on the unsigned operand; that rewrite is the caller's job.

    Contrast ll.viaddmin_s32, which emits ``__viaddmin_s32`` and computes
    ``min(a + b, c)`` -- using it as an accumulator silently clamps each
    step by the running value and corrupts a dot product.
    """
    if isinstance(a, int) and isinstance(b, int) and isinstance(c, int):
        return _dp4a_s8u8_fold(a, b, c)
    raise RuntimeError("should not call dp4a_s8u8_s32 in compilation")
