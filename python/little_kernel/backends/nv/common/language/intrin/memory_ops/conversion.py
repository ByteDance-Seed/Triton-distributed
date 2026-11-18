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
"""
Numerical conversion intrinsics (bf16/f16/fp8 widening, rounding packing).

RF10 canonical owner module; see little_kernel.backends.nv.common.language.intrin.contracts
for the shared operand-unit/lane/alignment/cache-policy/effect inventory.
"""

from little_kernel.language.builtin_base import builtin, Builtin, CODEGEN_NEEDS_TYPES_ATTR
from little_kernel.core.type_system import LLType, void, Tensor, int8, uint8, uint16, uint32, uint64, int32, bfloat16, float16, float32, u32x2, u32x4
import little_kernel.backends.nv.common.language.intrin.contracts as contracts

_make_gmem_ld_out_lane_type_check = contracts.make_gmem_ld_out_lane_type_check
_make_gmem_st_in_lane_type_check = contracts.make_gmem_st_in_lane_type_check

# ==============================================================================


def _gen_gmem_ld_16bit(name, cpp_dtype, vec_ty, lanes, lane_unpack, load_policy=None):
    """Factory for 16-bit typed vector loads widened to fp32 out-registers.

    load_policy selects the load used to fetch the 16-byte vector before
    widening:
      - None (default): plain C++ dereference (LDG.E.128, .ca/.wb behavior);
      - "nc":           __ldg on the vector pointer (read-only/.nc path);
      - "cs":           __ldcs on the vector pointer (evict-first streaming).
    The widening of each loaded 16-bit lane to fp32 via lane_unpack is
    identical for every policy; only the cache-allocation hint differs, so all
    three forms are bit-equivalent and share the same alignment/in-bounds
    legality contract (16-byte aligned ptr, all lanes in-bounds)."""
    outs = ", ".join(f"float* f{i}" for i in range(lanes))
    unpack = "\n".join(f"    *f{2 * w} = {lane_unpack}(p[{w}].x); "
                       f"*f{2 * w + 1} = {lane_unpack}(p[{w}].y);" for w in range(lanes // 2))
    if load_policy is None:
        load_line = f"    const {vec_ty} v = *reinterpret_cast<const {vec_ty}*>(ptr);"
    elif load_policy == "nc":
        load_line = f"    const {vec_ty} v = __ldg(reinterpret_cast<const {vec_ty}*>(ptr));"
    elif load_policy == "cs":
        load_line = f"    const {vec_ty} v = __ldcs(reinterpret_cast<const {vec_ty}*>(ptr));"
    else:
        raise ValueError(f"_gen_gmem_ld_16bit: unknown load_policy {load_policy!r} "
                         "(expected None, \"nc\", or \"cs\")")
    body = (f"\n__device__ __forceinline__ void {name}_fn(\n"
            f"        const {cpp_dtype}* ptr, {outs}) {{\n"
            f"{load_line}\n"
            f"    const {cpp_dtype}2* p ="
            f" reinterpret_cast<const {cpp_dtype}2*>(&v);\n"
            f"{unpack}\n}}\n")

    def codegen(mem, index, *outs_args):
        addrs = ", ".join(f"&{a}" for a in outs_args)
        inc = ["<cuda_bf16.h>"] if cpp_dtype == "__nv_bfloat16" else ["<cuda_fp16.h>"]
        return Builtin(body=body, includes=inc, return_val=f"{name}_fn({mem} + {index}, {addrs})")

    return codegen


def _gen_gmem_st_16bit(name, cpp_dtype, vec_ty, lanes, lane_pack2):
    """Factory for fp32 -> 16-bit typed vector stores with single RNE rounding."""
    ins = ", ".join(f"float f{i}" for i in range(lanes))
    pack = "\n".join(f"    p[{w}] = {lane_pack2}(f{2 * w}, f{2 * w + 1});" for w in range(lanes // 2))
    body = (f"\n__device__ __forceinline__ void {name}_fn(\n"
            f"        {cpp_dtype}* ptr, {ins}) {{\n"
            f"    {vec_ty} v;\n"
            f"    {cpp_dtype}2* p = reinterpret_cast<{cpp_dtype}2*>(&v);\n"
            f"{pack}\n"
            f"    *reinterpret_cast<{vec_ty}*>(ptr) = v;\n}}\n")

    def codegen(mem, index, *ins_args):
        vals = ", ".join(str(a) for a in ins_args)
        inc = ["<cuda_bf16.h>"] if cpp_dtype == "__nv_bfloat16" else ["<cuda_fp16.h>"]
        return Builtin(body=body, includes=inc, return_val=f"{name}_fn({mem} + {index}, {vals})")

    return codegen


_bf16x8_ld_doc = ("Load 8 contiguous naturally-aligned (16 B) bf16 values from GMEM; "
                  "each lane is delivered as an FP32 register via exact __bfloat162float. "
                  "Caller guarantees (mem + index) is 16-byte aligned and 8 lanes in-bounds; "
                  "use scalar mem[i] for unaligned head/tail.")

codegen_gmem_ld_bf16x8_f32x8 = _gen_gmem_ld_16bit("gmem_ld_bf16x8_f32x8", "__nv_bfloat16", "uint4", 8,
                                                  "__bfloat162float")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_bf16x8_f32x8", float32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_bf16x8_f32x8)
def gmem_ld_bf16x8_f32x8(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """Load 8 aligned contiguous bf16 values into eight FP32 registers.

    The plain gmem_ld_bf16x8 name is owned by the uint64-pair bit-pattern
    contract matching its .cg/.streaming policy twins (pair with
    vec_bf16x8_lane); this widening FP32-out variant keeps the explicit
    _f32x8 suffix so both APIs (and their emitted helper names) stay
    distinct."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_ld_bf16x8_f32x8_nc = _gen_gmem_ld_16bit("gmem_ld_bf16x8_f32x8_nc", "__nv_bfloat16", "uint4", 8,
                                                     "__bfloat162float", load_policy="nc")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_bf16x8_f32x8_nc", float32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_bf16x8_f32x8_nc)
def gmem_ld_bf16x8_f32x8_nc(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """Read-only (.nc, __ldg) twin of gmem_ld_bf16x8_f32x8.

    Loads the same 16-byte aligned bf16x8 vector and widens each lane with
    __bfloat162float exactly as the plain form, but fetches it via __ldg so
    the load uses the non-coherent/read-only path (LDG.E.128.CONSTANT). Legal
    only when the whole covered region is never written through any aliasing
    view for the duration of the load's use (same read-only contract as
    gmem_ld_ro_f32x4 / ld_global_nc_v4_u32). Bit-level loaded values and all
    alignment/in-bounds preconditions are identical to the plain form."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_ld_bf16x8_f32x8_cs = _gen_gmem_ld_16bit("gmem_ld_bf16x8_f32x8_cs", "__nv_bfloat16", "uint4", 8,
                                                     "__bfloat162float", load_policy="cs")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_bf16x8_f32x8_cs", float32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_bf16x8_f32x8_cs)
def gmem_ld_bf16x8_f32x8_cs(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """Cache-streaming (.cs, __ldcs) twin of gmem_ld_bf16x8_f32x8.

    Loads the same 16-byte aligned bf16x8 vector and widens each lane with
    __bfloat162float exactly as the plain form, but fetches it via __ldcs so
    the line is marked evict-first for single-touch read streams that would
    otherwise pollute L1/L2. __ldcs is only a cache hint (ld.global.cs), so
    any address legal for the plain form is legal here; bit-level loaded
    values and all alignment/in-bounds preconditions are identical."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_ld_bf16x4 = _gen_gmem_ld_16bit("gmem_ld_bf16x4", "__nv_bfloat16", "uint2", 4, "__bfloat162float")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_bf16x4", float32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_bf16x4)
def gmem_ld_bf16x4(mem, index, f0, f1, f2, f3):
    """Load 4 aligned (8 B) contiguous bf16 values into four FP32 registers."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# HEAD-shifted 16-byte-aligned bf16x8 window load (GEMV idiom)
# ==============================================================================
#
# Generic contract (no task or shape assumptions):
#   * ``mem``   -- any LittleKernel bf16 Tensor (global scope); reinterpreted
#                  as raw bytes, then offset by ``index`` elements (C++ pointer
#                  arithmetic, same as gmem_ld_bf16x8_f32x8).
#   * ``index`` -- element offset of the lane's *aligned floor* window; the
#                  byte address ``mem + index`` MUST be 16-byte aligned
#                  (index a multiple of 8).  In the GEMV idiom each lane
#                  computes ``index = (head + lane*8) & ~7`` per-lane, so
#                  adjacent lanes issue 16-byte-aligned loads on adjacent or
#                  overlapping 16-byte blocks.
#   * ``head``  -- runtime shift in [0, 7]: the lane's *logical* window
#                  [index + head, index + head + 8).  Elements with
#                  ``head + k < 8`` come from the lane's own aligned load;
#                  the rest come from ``lane + 1``'s aligned block, fetched
#                  with one ``__shfl_down_sync(0xffffffff, word, 1)`` per
#                  32-bit word (four shuffles total cover the whole block).
#   * ``f0..f7``-- FP32 out registers; ``f_k`` receives
#                  ``bf16_to_float(mem[index + head + k])`` for k in 0..7.
#
# Legality preconditions (checked by the frontend author, not the compiler):
#   * 16-byte alignment of ``mem + index`` (as above);
#   * ``head`` uniform or per-lane in [0, 7] — a per-lane head is fine;
#   * warp convergence: all 32 lanes of the warp must execute the call with
#     the full mask 0xffffffff (the intrinsic is a warp collective, like
#     shfl_down_sync); divergence around the call is UB;
#   * warp-edge: the *last* participating lane (lane 31 of the mask) has no
#     lane+1 to source the tail words from; the caller must guarantee either
#     (a) lane 31's logical window is masked out / not consumed, or (b) the
#     ``index`` used by lane 31 is the address of one *extra* valid aligned
#     block that the caller knows is in-bounds, and the values delivered to
#     lane 31 for ``head + k >= 8`` are read from its own load (i.e. the
#     caller treats lane 31's shuffle result as "the extra aligned block").
#     Concretely this is the memory-bound GEMV pattern: one row per lane,
#     lane 31's row reads through ``[index, index+head+8)`` which is covered
#     by its own block plus the shuffle from a lane whose ``index`` equals
#     lane 31's ``index + 8``; the frontend achieves this by either loading
#     per-lane ``index = window_floor`` (so lane+1's block IS the extra
#     block when windows are stride-8) or by dedicating the call to a
#     sub-warp partition where the tail lane is dead.
#   * Out-of-bounds: the aligned load touches ``[index*2, index*2+16)`` bytes;
#     the shuffle is register-only.  Whether the whole warp's collective load
#     set stays in bounds is the caller's responsibility — identical to the
#     legality split for gmem_ld_bf16x8_f32x8.
#
# Numerical contract: bf16 -> fp32 widening is bit-exact (__bfloat162float);
# the shuffle moves raw 32-bit words with no conversion, so the eight
# outputs are value-identical to eight scalar
# ``bf16_to_float(mem[index+head+k])`` loads.
#
# Usage (frontend, reproducer shape):
#     # logical window [head + tid*8, head + tid*8 + 8)
#     base = (head + tid * 8) - ((head + tid * 8) % 8)   # aligned floor
#     shift = (head + tid * 8) % 8
#     ll.gmem_ld_bf16x8_shifted(vec, base, shift,
#                               s0, s1, s2, s3, s4, s5, s6, s7)


def _gen_gmem_ld_bf16x8_shifted(name, load_stmt):
    """Factory for HEAD-shifted aligned bf16x8 window loads.

    ``load_stmt`` is the C++ statement producing ``uint4 value`` from
    ``const void* ptr``; policy twins differ only in that one statement."""

    body = (f"""
#include <cuda_bf16.h>
__device__ __forceinline__ void {name}_fn(
        const __nv_bfloat16* ptr, int head,
        float* f0, float* f1, float* f2, float* f3,
        float* f4, float* f5, float* f6, float* f7) {{
    const uint4 value = {load_stmt};
    const uint32_t own_w[4] = {{value.x, value.y, value.z, value.w}};
    // Shuffle lane+1's four words (index r means lane+1's word r).  For the
    // in-bounds tail lanes the caller guarantees lane+1 loaded the block at
    // (ptr + 8), so nxt[i] pairs with own_w[i] of the next 16B block.
    uint32_t nxt[4];
    nxt[0] = __shfl_down_sync(0xffffffffu, value.x, 1);
    nxt[1] = __shfl_down_sync(0xffffffffu, value.y, 1);
    nxt[2] = __shfl_down_sync(0xffffffffu, value.z, 1);
    nxt[3] = __shfl_down_sync(0xffffffffu, value.w, 1);
    // bf16 lane j of the shifted window lives in word (head + j) / 2:
    // from own when (head + j) < 8, from lane+1's block otherwise.
    float* outs[8] = {{f0, f1, f2, f3, f4, f5, f6, f7}};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {{
        const int g = head + j;              // global element slot in [0,15]
        const uint32_t word = (g < 8) ? own_w[g >> 1] : nxt[(g - 8) >> 1];
        const __nv_bfloat162 pair =
            *reinterpret_cast<const __nv_bfloat162*>(&word);
        const float fp = (g & 1) ? __bfloat162float(pair.y)
                                 : __bfloat162float(pair.x);
        *outs[j] = fp;
    }}
}}
""")

    def codegen(mem, index, head, *outs_args):
        addrs = ", ".join(f"&{a}" for a in outs_args)
        return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"{name}_fn({mem} + {index}, {head}, {addrs})")

    return codegen


codegen_gmem_ld_bf16x8_shifted = _gen_gmem_ld_bf16x8_shifted("gmem_ld_bf16x8_shifted",
                                                             "*reinterpret_cast<const uint4*>(ptr)")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_bf16x8_shifted", float32, leading_operands=3),
         eval_return_type=void, codegen_func=codegen_gmem_ld_bf16x8_shifted)
def gmem_ld_bf16x8_shifted(mem, index, head, f0, f1, f2, f3, f4, f5, f6, f7):
    """Load the HEAD-shifted window [index+head, index+head+8) of a bf16 GMEM
    tensor into eight FP32 registers with one aligned 16-byte load plus one
    cross-lane shuffle round.

    ``mem + index`` must be 16-byte aligned; ``head`` in [0, 7]; all 32
    lanes of the warp must execute the call (full-mask shuffle).  ``f_j``
    receives ``bf16_to_float(mem[index + head + j])`` for j in 0..7.  Tail
    elements with ``head + j >= 8`` are sourced from lane+1's aligned block;
    the caller guarantees the warp-edge lane's tail is either masked-out or
    covered by an extra in-bounds aligned block.  See the family contract
    above for the full legality and usage rules."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_ld_bf16x8_shifted_cg = _gen_gmem_ld_bf16x8_shifted("gmem_ld_bf16x8_shifted_cg",
                                                                "__ldcg(reinterpret_cast<const uint4*>(ptr))")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_bf16x8_shifted_cg", float32, leading_operands=3),
         eval_return_type=void, codegen_func=codegen_gmem_ld_bf16x8_shifted_cg)
def gmem_ld_bf16x8_shifted_cg(mem, index, head, f0, f1, f2, f3, f4, f5, f6, f7):
    """L2-only (.cg) cache-policy twin of gmem_ld_bf16x8_shifted.  Bits moved
    and the shuffle/select recombination are identical; only the aligned
    load's cache behavior differs.  Same legality contract."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_ld_bf16x8_shifted_streaming = _gen_gmem_ld_bf16x8_shifted("gmem_ld_bf16x8_shifted_streaming",
                                                                       "__ldcs(reinterpret_cast<const uint4*>(ptr))")


@builtin(
    eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_bf16x8_shifted_streaming", float32, leading_operands=3),
    eval_return_type=void, codegen_func=codegen_gmem_ld_bf16x8_shifted_streaming)
def gmem_ld_bf16x8_shifted_streaming(mem, index, head, f0, f1, f2, f3, f4, f5, f6, f7):
    """Streaming (evict-first, .cs) cache-policy twin of
    gmem_ld_bf16x8_shifted; the vector is single-touch in the GEMV idiom.
    Bits moved and recombination identical to the default policy; same
    legality contract."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_ld_f16x8_f32x8 = _gen_gmem_ld_16bit("gmem_ld_f16x8_f32x8", "__half", "uint4", 8, "__half2float")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_f16x8_f32x8", float32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_f16x8_f32x8)
def gmem_ld_f16x8_f32x8(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """Load 8 aligned (16 B) contiguous fp16 values into eight FP32 registers.

    See gmem_ld_bf16x8_f32x8: the plain name is owned by the uint64-pair
    contract; this widening variant keeps the _f32x8 suffix."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_ld_f16x4 = _gen_gmem_ld_16bit("gmem_ld_f16x4", "__half", "uint2", 4, "__half2float")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_f16x4", float32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_f16x4)
def gmem_ld_f16x4(mem, index, f0, f1, f2, f3):
    """Load 4 aligned (8 B) contiguous fp16 values into four FP32 registers."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_st_f32_to_bf16x8 = _gen_gmem_st_16bit("gmem_st_f32_to_bf16x8", "__nv_bfloat16", "uint4", 8,
                                                   "__floats2bfloat162_rn")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_f32_to_bf16x8", float32), eval_return_type=void,
         codegen_func=codegen_gmem_st_f32_to_bf16x8)
def gmem_st_f32_to_bf16x8(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """Round 8 FP32 lanes to bf16 (single RNE) and store in one 16 B transaction."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_st_f32_to_bf16x4 = _gen_gmem_st_16bit("gmem_st_f32_to_bf16x4", "__nv_bfloat16", "uint2", 4,
                                                   "__floats2bfloat162_rn")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_f32_to_bf16x4", float32), eval_return_type=void,
         codegen_func=codegen_gmem_st_f32_to_bf16x4)
def gmem_st_f32_to_bf16x4(mem, index, f0, f1, f2, f3):
    """Round 4 FP32 lanes to bf16 (single RNE) and store in one 8 B transaction."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_st_f32_to_f16x8 = _gen_gmem_st_16bit("gmem_st_f32_to_f16x8", "__half", "uint4", 8, "__floats2half2_rn")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_f32_to_f16x8", float32), eval_return_type=void,
         codegen_func=codegen_gmem_st_f32_to_f16x8)
def gmem_st_f32_to_f16x8(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """Round 8 FP32 lanes to fp16 (single RNE) and store in one 16 B transaction."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_st_f32_to_f16x4 = _gen_gmem_st_16bit("gmem_st_f32_to_f16x4", "__half", "uint2", 4, "__floats2half2_rn")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_f32_to_f16x4", float32), eval_return_type=void,
         codegen_func=codegen_gmem_st_f32_to_f16x4)
def gmem_st_f32_to_f16x4(mem, index, f0, f1, f2, f3):
    """Round 4 FP32 lanes to fp16 (single RNE) and store in one 8 B transaction."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# FP8 conversions (E4M3/E5M2), round-to-nearest-even with finite saturation.
# Generic dtype-conversion builtins for quantized-output kernels.
# ==============================================================================


def codegen_cvt_f32x2_to_e4m3x2(f_lo, f_hi):
    body = """
__device__ __forceinline__ uint16_t cvt_f32x2_to_e4m3x2_fn(float f_lo, float f_hi) {
    uint16_t v;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(v) : "f"(f_hi), "f"(f_lo));
    return v;
}
"""
    return Builtin(body=body, includes=["<cstdint>"], return_val=f"cvt_f32x2_to_e4m3x2_fn({f_lo}, {f_hi})")


@builtin(eval_return_type=uint16, codegen_func=codegen_cvt_f32x2_to_e4m3x2)
def cvt_f32x2_to_e4m3x2(f_lo, f_hi):
    """Convert two FP32 values to two packed E4M3 bytes (RNE, satfinite).

    Returns a uint16 whose low byte is cvt(f_lo) and high byte is cvt(f_hi).
    Requires SM89+/PTX for cvt.rn.satfinite.e4m3x2.f32.
    """
    raise RuntimeError("should not be called in compilation")


def codegen_cvt_f32x2_to_e5m2x2(f_lo, f_hi):
    body = """
__device__ __forceinline__ uint16_t cvt_f32x2_to_e5m2x2_fn(float f_lo, float f_hi) {
    uint16_t v;
    asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
                 : "=h"(v) : "f"(f_hi), "f"(f_lo));
    return v;
}
"""
    return Builtin(body=body, includes=["<cstdint>"], return_val=f"cvt_f32x2_to_e5m2x2_fn({f_lo}, {f_hi})")


@builtin(eval_return_type=uint16, codegen_func=codegen_cvt_f32x2_to_e5m2x2)
def cvt_f32x2_to_e5m2x2(f_lo, f_hi):
    """Convert two FP32 values to two packed E5M2 bytes (RNE, satfinite).

    Returns a uint16 whose low byte is cvt(f_lo) and high byte is cvt(f_hi).
    """
    raise RuntimeError("should not be called in compilation")


def _gen_gmem_st_f32_to_fp8x16(name, cvt_ptx):
    """Factory: pack 16 FP32 lanes to 16 fp8 bytes, one 16 B vector store."""
    ins = ", ".join(f"float f{i}" for i in range(16))
    cvt = "\n".join((f"    {{\n        uint16_t p;\n"
                     f'        asm volatile("{cvt_ptx} %0, %1, %2;"\n'
                     f'                     : "=h"(p) : "f"(f{2 * w + 1}), "f"(f{2 * w}));\n'
                     f"        w[{w}] = p;\n    }}") for w in range(8))
    body = (f"\n__device__ __forceinline__ void {name}_fn(\n"
            f"        uint8_t* ptr, {ins}) {{\n"
            f"    uint4 v;\n"
            f"    uint16_t* w = reinterpret_cast<uint16_t*>(&v);\n"
            f"{cvt}\n"
            f"    *reinterpret_cast<uint4*>(ptr) = v;\n}}\n")

    def codegen(mem, byte_index, *vals):
        args = ", ".join(str(a) for a in vals)
        inc = ["<cstdint>"]
        return Builtin(body=body, includes=inc,
                       return_val=f"{name}_fn(reinterpret_cast<uint8_t*>({mem}) + ({byte_index}), {args})")

    return codegen


_FP8X16_DOC = ("Convert 16 FP32 lanes with PTX {ptx} (round-to-nearest-even, finite "
               "saturation) and store the 16 packed bytes in ONE 16-byte aligned vector "
               "transaction. `mem` is the byte-addressed output tensor (fp8 storage "
               "dtypes are exported as uint8 tensors); `byte_index` counts BYTES. "
               "Caller guarantees 16-byte alignment; partial/unaligned tails use scalar "
               "byte stores of cvt_f32x2_to_* results.")

codegen_gmem_st_f32_to_e4m3x16 = _gen_gmem_st_f32_to_fp8x16("gmem_st_f32_to_e4m3x16", "cvt.rn.satfinite.e4m3x2.f32")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_f32_to_e4m3x16)
def gmem_st_f32_to_e4m3x16(mem, byte_index, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15):
    """Round 16 FP32 lanes to 16 E4M3 bytes (RNE, satfinite), one 16 B store."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_st_f32_to_e5m2x16 = _gen_gmem_st_f32_to_fp8x16("gmem_st_f32_to_e5m2x16", "cvt.rn.satfinite.e5m2x2.f32")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_f32_to_e5m2x16)
def gmem_st_f32_to_e5m2x16(mem, byte_index, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15):
    """Round 16 FP32 lanes to 16 E5M2 bytes (RNE, satfinite), one 16 B store."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Typed quantizing FP8 stores at ARBITRARY byte offsets (strided layouts)
# ==============================================================================
#
# gmem_st_f32_to_e4m3x16 / gmem_st_f32_to_e5m2x16 emit ONE 16-byte vector
# transaction, so their byte_index addresses a run of 16 CONTIGUOUS fp8
# elements.  A non-contiguous output cannot be written that way.  The
# canonical case is a manifest-declared rank-2 [M, N] fp8 output with
# column-major stride (1, M): logical element (m, n) lives at byte offset
# m + n*M, so lanes stepping across columns land M bytes apart and no
# physical [N, M] row-major alias exists in the export ABI.
#
# These builtins close the gap with typed element stores whose target
# address is an arbitrary byte offset the FRONTEND computes from the
# logical strides (the builtins themselves stay layout- and task-generic):
#   * gmem_st_f32_to_{e4m3,e5m2}(mem, byte_index, f) rounds ONE FP32 lane
#     to one fp8 byte (RNE, satfinite) and scalar-stores it at
#     (uint8_t*)mem + byte_index.  A scalar st.global.u8 needs no
#     alignment beyond the byte itself, so this is also the ragged /
#     unaligned-tail fallback the x16 vector contract points at.
#   * gmem_st_f32_to_{e4m3,e5m2}x2(mem, off_lo, off_hi, f_lo, f_hi)
#     performs ONE PTX pair conversion and scalar-stores the two result
#     bytes at TWO INDEPENDENT byte offsets -- the explicit-offset
#     ("strided") form at the hardware's native convert width: a
#     column-major column pair is (m + n*M, m + (n+1)*M), a contiguous
#     pair is (k, k+1).  The offsets need not be adjacent or ordered, but
#     both target bytes must be in bounds; writing two distinct values
#     through the same offset is a caller error (use the singleton for an
#     odd one-element tail).
#
# PTX exposes only the pair convert, so the singleton converts (f, f) and
# keeps the low byte; byte extraction follows cvt_f32x2_to_*x2's packing
# (low byte = cvt(f_lo), high byte = cvt(f_hi)).


def _gen_gmem_st_f32_to_fp8_scalar(name, cvt_ptx):
    """Factory: one FP32 lane -> one fp8 byte at an arbitrary byte offset."""
    body = f"""
__device__ __forceinline__ void {name}_fn(uint8_t* ptr, float f) {{
    uint16_t p;
    asm volatile("{cvt_ptx} %0, %1, %2;"
                 : "=h"(p) : "f"(f), "f"(f));
    *ptr = (uint8_t)(p & 0xFFu);
}}
"""

    def codegen(mem, byte_index, f):
        return Builtin(body=body, includes=["<cstdint>"],
                       return_val=f"{name}_fn(reinterpret_cast<uint8_t*>({mem}) + ({byte_index}), {f})")

    return codegen


def _gen_gmem_st_f32_to_fp8x2_offsets(name, cvt_ptx):
    """Factory: one FP32 pair -> two fp8 bytes at two independent offsets."""
    body = f"""
__device__ __forceinline__ void {name}_fn(
        uint8_t* base, long long off_lo, long long off_hi,
        float f_lo, float f_hi) {{
    uint16_t p;
    asm volatile("{cvt_ptx} %0, %1, %2;"
                 : "=h"(p) : "f"(f_hi), "f"(f_lo));
    base[off_lo] = (uint8_t)(p & 0xFFu);
    base[off_hi] = (uint8_t)((p >> 8) & 0xFFu);
}}
"""

    def codegen(mem, off_lo, off_hi, f_lo, f_hi):
        return Builtin(
            body=body, includes=["<cstdint>"], return_val=f"{name}_fn(reinterpret_cast<uint8_t*>({mem}), ({off_lo}), "
            f"({off_hi}), {f_lo}, {f_hi})")

    return codegen


codegen_gmem_st_f32_to_e4m3 = _gen_gmem_st_f32_to_fp8_scalar("gmem_st_f32_to_e4m3", "cvt.rn.satfinite.e4m3x2.f32")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_f32_to_e4m3)
def gmem_st_f32_to_e4m3(mem, byte_index, f):
    """Round ONE FP32 lane to one E4M3 byte (RNE, satfinite) and scalar-store
    it at the ARBITRARY byte offset ``(uint8_t*)mem + byte_index``.

    Typed element-store twin of :func:`gmem_st_f32_to_e4m3x16` for
    non-contiguous layouts: ``byte_index`` (in bytes) is computed by the
    caller from logical strides, e.g. m + n*M for a rank-2 [M, N] output
    with column-major stride (1, M).  A scalar byte store has no alignment
    precondition beyond the byte itself, so this also covers ragged /
    unaligned tails of the 16-byte vector store."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_st_f32_to_e4m3x2 = _gen_gmem_st_f32_to_fp8x2_offsets("gmem_st_f32_to_e4m3x2",
                                                                  "cvt.rn.satfinite.e4m3x2.f32")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_f32_to_e4m3x2)
def gmem_st_f32_to_e4m3x2(mem, off_lo, off_hi, f_lo, f_hi):
    """Round TWO FP32 lanes to two E4M3 bytes (RNE, satfinite) with ONE pair
    conversion and scalar-store them at TWO INDEPENDENT byte offsets
    (``base + off_lo`` gets cvt(f_lo), ``base + off_hi`` gets cvt(f_hi)).

    Explicit-offset / strided form of :func:`gmem_st_f32_to_e4m3x16` at the
    PTX convert width: the offsets need not be adjacent -- for a rank-2
    [M, N] output with column-major stride (1, M) a pair of columns is
    addressed (m + n*M, m + (n+1)*M), while a contiguous pair passes
    (k, k+1).  Both target bytes must be in bounds and distinct; an odd
    one-element tail uses :func:`gmem_st_f32_to_e4m3`."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_st_f32_to_e5m2 = _gen_gmem_st_f32_to_fp8_scalar("gmem_st_f32_to_e5m2", "cvt.rn.satfinite.e5m2x2.f32")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_f32_to_e5m2)
def gmem_st_f32_to_e5m2(mem, byte_index, f):
    """Round ONE FP32 lane to one E5M2 byte (RNE, satfinite) and scalar-store
    it at the ARBITRARY byte offset ``(uint8_t*)mem + byte_index``.

    E5M2 twin of :func:`gmem_st_f32_to_e4m3`: same strided-addressing
    contract (byte offset computed by the caller from logical strides) and
    no alignment precondition beyond the byte itself."""
    raise RuntimeError("should not be called in compilation")


codegen_gmem_st_f32_to_e5m2x2 = _gen_gmem_st_f32_to_fp8x2_offsets("gmem_st_f32_to_e5m2x2",
                                                                  "cvt.rn.satfinite.e5m2x2.f32")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_f32_to_e5m2x2)
def gmem_st_f32_to_e5m2x2(mem, off_lo, off_hi, f_lo, f_hi):
    """Round TWO FP32 lanes to two E5M2 bytes (RNE, satfinite) with ONE pair
    conversion and scalar-store them at TWO INDEPENDENT byte offsets.

    E5M2 twin of :func:`gmem_st_f32_to_e4m3x2`: same explicit-offset /
    strided-addressing contract (e.g. column-major stride (1, M) offsets
    m + n*M and m + (n+1)*M); both target bytes must be in bounds and
    distinct."""
    raise RuntimeError("should not be called in compilation")


def codegen_e4m3x2_to_f32x2(packed, f_lo, f_hi):
    body = """
__device__ __forceinline__ void e4m3x2_to_f32x2_fn(uint16_t packed, float* f_lo, float* f_hi) {
    *f_lo = float(__half(__nv_cvt_fp8_to_halfraw(
        (__nv_fp8_storage_t)(packed & 0x00FF), __NV_E4M3)));
    *f_hi = float(__half(__nv_cvt_fp8_to_halfraw(
        (__nv_fp8_storage_t)(packed >> 8), __NV_E4M3)));
}
"""
    return Builtin(body=body, includes=["<cuda_fp8.h>", "<cuda_fp16.h>"],
                   return_val=f"e4m3x2_to_f32x2_fn({packed}, &{f_lo}, &{f_hi})")


@builtin(eval_return_type=void, codegen_func=codegen_e4m3x2_to_f32x2)
def e4m3x2_to_f32x2(packed, f_lo, f_hi):
    """Decode two packed E4M3 bytes (uint16 low/high byte) into two FP32 registers."""
    raise RuntimeError("should not be called in compilation")
