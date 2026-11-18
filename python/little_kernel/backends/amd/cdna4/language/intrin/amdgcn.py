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
AMDGCN intrinsics for the HIP backend (gfx9 CDNA, tested on gfx950).

Each intrinsic is one AMDGPU instruction (a clang ``__builtin_amdgcn_*`` builtin or one line of
inline assembly); scheduling stays in the kernel. LDS operands are a byte offset from an LDS base
array (``ll.empty(..., scope="shared")`` or dynamic shared memory).
"""

from little_kernel.language.builtin_base import builtin, Builtin
from little_kernel.backends.amd.cdna4.core.types import AmdBufferRsrc
from little_kernel.core.type_system import float32x4, int32, int32x4, int32x8, uint8, uint32x2, uint32x4, void

# ==============================================================================
# Buffer resources and global -> LDS DMA
# ==============================================================================


def codegen_make_buffer_rsrc(base, byte_offset, num_records):
    return Builtin(
        body="", includes=[],
        return_val=(f"__builtin_amdgcn_make_buffer_rsrc((void*)((const char*)({base}) + ({byte_offset})), "
                    f"(short)0, (int)({num_records}), 0x00020000)"))


@builtin(eval_return_type=AmdBufferRsrc, codegen_func=codegen_make_buffer_rsrc)
def amdgcn_make_buffer_rsrc(base, byte_offset, num_records):
    """Raw buffer descriptor at ``base + byte_offset`` (64-bit address) covering ``num_records`` bytes.

    Buffer loads past ``num_records`` (voffset + soffset + offset, checked per dword) return 0 and
    stores past it are dropped.
    """
    raise RuntimeError("amdgcn_make_buffer_rsrc should never be called in compilation")


_BUFFER_LOAD_LDS_SIZES = (4, 16)


def _require_buffer_load_lds_size(size_bytes):
    # size_bytes selects ``buffer_load_dword`` vs ``buffer_load_dwordx4`` and
    # reaches the clang builtin verbatim, so only the documented compile-time
    # constants are legal.  The emitter renders constant arguments to their C++
    # literal text before they reach this codegen, so a folded int constant
    # arrives as ``4``/``16`` (int or its string spelling); anything else is a
    # non-constant or out-of-contract value and is rejected here, where the
    # error names the intrinsic, instead of leaving hipcc to fail on an
    # invalid builtin immediate.
    if isinstance(size_bytes, int) and not isinstance(size_bytes, bool):
        candidates = (size_bytes, )
    elif isinstance(size_bytes, str):
        try:
            # Base-0 also accepts the hex spelling the emitter keeps for
            # literals written as 0x10 in kernel source.
            candidates = (int(size_bytes.strip(), 0), )
        except ValueError:
            candidates = ()
    else:
        candidates = ()
    if candidates and candidates[0] in _BUFFER_LOAD_LDS_SIZES:
        return candidates[0]
    raise ValueError(f"amdgcn_buffer_load_lds size_bytes must be a compile-time constant in "
                     f"{_BUFFER_LOAD_LDS_SIZES}, got {size_bytes!r}")


def codegen_buffer_load_lds(rsrc, lds_base, lds_byte_offset, size_bytes, voffset, soffset):
    size = _require_buffer_load_lds_size(size_bytes)
    return Builtin(
        body="", includes=[], return_val=(f"__builtin_amdgcn_raw_ptr_buffer_load_lds({rsrc}, "
                                          f"(lk_lds_ptr_t)((uint8_t*)({lds_base}) + ({lds_byte_offset})), {size}, "
                                          f"{voffset}, {soffset}, 0, 0)"))


@builtin(eval_return_type=void, codegen_func=codegen_buffer_load_lds)
def amdgcn_buffer_load_lds(rsrc, lds_base, lds_byte_offset, size_bytes, voffset, soffset):
    """``buffer_load_dword{,x4} ... lds``: every lane reads ``size_bytes`` (4 or 16, a constant) at
    voffset (per lane) + soffset (uniform); the wave writes them to LDS at the uniform address
    ``lds_base + lds_byte_offset`` + size_bytes * lane. Counted by vmcnt."""
    raise RuntimeError("amdgcn_buffer_load_lds should never be called in compilation")


# ==============================================================================
# LDS reads
# ==============================================================================


def codegen_lds_read_b128(lds_base, byte_offset):
    return Builtin(body="", includes=[],
                   return_val=f"(*(const lk_int32x4*)((const uint8_t*)({lds_base}) + ({byte_offset})))")


@builtin(eval_return_type=int32x4, codegen_func=codegen_lds_read_b128)
def amdgcn_lds_read_b128(lds_base, byte_offset):
    """``ds_read_b128``: 16 bytes per lane at a 16-byte aligned LDS address."""
    raise RuntimeError("amdgcn_lds_read_b128 should never be called in compilation")


def codegen_lds_read_u8(lds_base, byte_offset):
    return Builtin(body="", includes=[], return_val=f"(*((const uint8_t*)({lds_base}) + ({byte_offset})))")


@builtin(eval_return_type=uint8, codegen_func=codegen_lds_read_u8)
def amdgcn_lds_read_u8(lds_base, byte_offset):
    """``ds_read_u8``: one byte per lane."""
    raise RuntimeError("amdgcn_lds_read_u8 should never be called in compilation")


def codegen_lds_write_b128(lds_base, byte_offset, value):
    return Builtin(body="", includes=[],
                   return_val=f"(*(lk_uint32x4*)((uint8_t*)({lds_base}) + ({byte_offset})) = (lk_uint32x4)({value}))")


@builtin(eval_return_type=void, codegen_func=codegen_lds_write_b128)
def amdgcn_lds_write_b128(lds_base, byte_offset, value):
    """``ds_write_b128``: 16 bytes per lane (a 4 x 32-bit vector) at a 16-byte aligned LDS address."""
    raise RuntimeError("amdgcn_lds_write_b128 should never be called in compilation")


# ==============================================================================
# MFMA
# ==============================================================================


def codegen_concat_i32x4(lo, hi):
    return Builtin(body="", includes=[], return_val=f"__builtin_shufflevector({lo}, {hi}, 0, 1, 2, 3, 4, 5, 6, 7)")


@builtin(eval_return_type=int32x8, codegen_func=codegen_concat_i32x4)
def amdgcn_concat_i32x4(lo, hi):
    """Two 4 x int32 registers as one 8 x int32 operand (lo in elements 0-3)."""
    raise RuntimeError("amdgcn_concat_i32x4 should never be called in compilation")


def codegen_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, fmt_a, fmt_b, opsel_a, scale_a, opsel_b, scale_b):
    return Builtin(
        body="", includes=[],
        return_val=(f"__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4({a}, {b}, {c}, {fmt_a}, {fmt_b}, "
                    f"{opsel_a}, {scale_a}, {opsel_b}, {scale_b})"))


@builtin(eval_return_type=float32x4, codegen_func=codegen_mfma_scale_f32_16x16x128_f8f6f4)
def amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, fmt_a, fmt_b, opsel_a, scale_a, opsel_b, scale_b):
    """``v_mfma_scale_f32_16x16x128_f8f6f4`` (gfx950): D(16 x 16, fp32) = A(16 x 128) * B(16 x 128)^T + C.

    a, b: 8 x int32 per lane, lane l holds row l % 16, K bytes 16 (l / 16) .. + 15 (elements 0-3) and
    64 + 16 (l / 16) .. + 15 (elements 4-7). fmt_*: 0 = fp8 e4m3, 1 = bf8 e5m2, 2 = fp6, 3 = bf6,
    4 = fp4 (constants). scale_*: UE8M0 codes, the code of lane l's 32-wide K block in byte opsel_*.
    Result: element e of lane l is D[l % 16][4 (l / 16) + e] with rows from b and columns from a.
    """
    raise RuntimeError("amdgcn_mfma_scale_f32_16x16x128_f8f6f4 should never be called in compilation")


# ==============================================================================
# Waits, barriers, scheduling
# ==============================================================================


def _waitcnt_immediate(n, field, bits, builtin_name):
    """Parse a visited codegen operand as a compile-time integer and range-check it.

    The gfx9 ``s_waitcnt`` immediate is a hardware field, not an expression: the
    builtin must see an integer literal at compile time.  A runtime value or an
    out-of-range literal is a compile-time error, never a silently masked or
    clipped encoding (masking 64 as ``64 & 15`` would encode the wait for 0).
    """
    text = str(n).strip()
    while len(text) >= 2 and text[0] == "(" and text[-1] == ")":
        text = text[1:-1].strip()
    try:
        value = int(text, 0)
    except ValueError:
        raise ValueError(f"{builtin_name}: {field} must be an integer literal at compile time "
                         f"(got {n!r}); the s_waitcnt immediate is a hardware field, not a "
                         "runtime value") from None
    if not 0 <= value <= (1 << bits) - 1:
        raise ValueError(f"{builtin_name}: {field} must be in [0, {(1 << bits) - 1}], got {value}")
    return value


def codegen_s_waitcnt_vmcnt(n):
    # gfx9 encoding: vmcnt in bits 3:0 and 15:14, expcnt (7 = no wait) in 6:4, lgkmcnt (15 = no wait) in 11:8
    vmcnt = _waitcnt_immediate(n, "n", 6, "amdgcn_s_waitcnt_vmcnt")
    return Builtin(
        body="", includes=[],
        return_val=f"__builtin_amdgcn_s_waitcnt({(vmcnt & 15) | ((vmcnt >> 4) << 14) | (7 << 4) | (15 << 8)})")


@builtin(eval_return_type=void, codegen_func=codegen_s_waitcnt_vmcnt)
def amdgcn_s_waitcnt_vmcnt(n):
    """``s_waitcnt vmcnt(n)``: wait until at most n (an integer literal constant, 0..63) vector
    memory operations of this wave are outstanding (they complete in order)."""
    raise RuntimeError("amdgcn_s_waitcnt_vmcnt should never be called in compilation")


def codegen_s_waitcnt_lgkmcnt(n):
    lgkmcnt = _waitcnt_immediate(n, "n", 4, "amdgcn_s_waitcnt_lgkmcnt")
    return Builtin(body="", includes=[],
                   return_val=f"__builtin_amdgcn_s_waitcnt({15 | (3 << 14) | (7 << 4) | ((lgkmcnt & 15) << 8)})")


@builtin(eval_return_type=void, codegen_func=codegen_s_waitcnt_lgkmcnt)
def amdgcn_s_waitcnt_lgkmcnt(n):
    """``s_waitcnt lgkmcnt(n)``: wait until at most n (an integer literal constant, 0..15) LDS /
    scalar memory operations of this wave are outstanding."""
    raise RuntimeError("amdgcn_s_waitcnt_lgkmcnt should never be called in compilation")


def codegen_s_barrier():
    return Builtin(body="", includes=[], return_val="asm volatile(\"s_barrier\" ::: \"memory\")")


@builtin(eval_return_type=void, codegen_func=codegen_s_barrier)
def amdgcn_s_barrier():
    """``s_barrier`` without the memory waits that ``__syncthreads`` adds: wait for the work-group's
    waves only. Pair it with explicit waitcnts."""
    raise RuntimeError("amdgcn_s_barrier should never be called in compilation")


def codegen_sched_barrier(mask):
    return Builtin(body="", includes=[], return_val=f"__builtin_amdgcn_sched_barrier({mask})")


@builtin(eval_return_type=void, codegen_func=codegen_sched_barrier)
def amdgcn_sched_barrier(mask):
    """Compiler-only scheduling barrier; mask 0 lets no instruction cross it. Emits nothing."""
    raise RuntimeError("amdgcn_sched_barrier should never be called in compilation")


def codegen_readfirstlane(x):
    return Builtin(body="", includes=[], return_val=f"__builtin_amdgcn_readfirstlane({x})")


@builtin(eval_return_type=int32, codegen_func=codegen_readfirstlane)
def amdgcn_readfirstlane(x):
    """``v_readfirstlane_b32``: the value of the first active lane, as a uniform (SGPR) value."""
    raise RuntimeError("amdgcn_readfirstlane should never be called in compilation")


# ==============================================================================
# Conversions and stores
# ==============================================================================

_CVT_PK_BF16_BODY = """
typedef __bf16 lk_bf16x2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ uint32_t lk_cvt_pk_bf16_f32(float a, float b) {  // v_cvt_pk_bf16_f32, RNE
    return __builtin_bit_cast(uint32_t, __builtin_convertvector((lk_float32x2){a, b}, lk_bf16x2));
}
"""


def codegen_cvt_pk_bf16_f32x4(v):
    return Builtin(
        body=_CVT_PK_BF16_BODY, includes=[], return_val=(f"((lk_uint32x2){{lk_cvt_pk_bf16_f32(({v})[0], ({v})[1]), "
                                                         f"lk_cvt_pk_bf16_f32(({v})[2], ({v})[3])}})"))


@builtin(eval_return_type=uint32x2, codegen_func=codegen_cvt_pk_bf16_f32x4)
def amdgcn_cvt_pk_bf16_f32x4(v):
    """Four fp32 values to four bf16 (round to nearest even), packed two per uint32, in order."""
    raise RuntimeError("amdgcn_cvt_pk_bf16_f32x4 should never be called in compilation")


def codegen_cvt_pk_bf16_f32x8(lo, hi):
    return Builtin(
        body=_CVT_PK_BF16_BODY, includes=[],
        return_val=(f"((lk_uint32x4){{lk_cvt_pk_bf16_f32(({lo})[0], ({lo})[1]), "
                    f"lk_cvt_pk_bf16_f32(({lo})[2], ({lo})[3]), lk_cvt_pk_bf16_f32(({hi})[0], ({hi})[1]), "
                    f"lk_cvt_pk_bf16_f32(({hi})[2], ({hi})[3])}})"))


@builtin(eval_return_type=uint32x4, codegen_func=codegen_cvt_pk_bf16_f32x8)
def amdgcn_cvt_pk_bf16_f32x8(lo, hi):
    """Eight fp32 values (lo, then hi: two 4 x fp32 vectors) to eight bf16 (round to nearest even), packed
    two per uint32, in order."""
    raise RuntimeError("amdgcn_cvt_pk_bf16_f32x8 should never be called in compilation")


def codegen_global_store_b16(base, byte_offset, value):
    return Builtin(body="", includes=[],
                   return_val=f"(*(uint16_t*)((uint8_t*)({base}) + ({byte_offset})) = (uint16_t)({value}))")


@builtin(eval_return_type=void, codegen_func=codegen_global_store_b16)
def amdgcn_global_store_b16(base, byte_offset, value):
    """``global_store_short``: the low 16 bits of ``value`` at ``base + byte_offset`` (2-byte aligned)."""
    raise RuntimeError("amdgcn_global_store_b16 should never be called in compilation")


def codegen_global_store_b64(base, byte_offset, value):
    return Builtin(body="", includes=[],
                   return_val=f"(*(lk_uint32x2*)((uint8_t*)({base}) + ({byte_offset})) = ({value}))")


@builtin(eval_return_type=void, codegen_func=codegen_global_store_b64)
def amdgcn_global_store_b64(base, byte_offset, value):
    """``global_store_dwordx2``: 8 bytes per lane at ``base + byte_offset`` (8-byte aligned)."""
    raise RuntimeError("amdgcn_global_store_b64 should never be called in compilation")


def codegen_global_store_b128(base, byte_offset, value):
    return Builtin(body="", includes=[],
                   return_val=f"(*(lk_uint32x4*)((uint8_t*)({base}) + ({byte_offset})) = (lk_uint32x4)({value}))")


@builtin(eval_return_type=void, codegen_func=codegen_global_store_b128)
def amdgcn_global_store_b128(base, byte_offset, value):
    """``global_store_dwordx4``: 16 bytes per lane (a 4 x 32-bit vector) at ``base + byte_offset``
    (16-byte aligned)."""
    raise RuntimeError("amdgcn_global_store_b128 should never be called in compilation")


_BUFFER_STORE_B128_BODY = """
__device__ __forceinline__ void lk_buffer_store_b128(lk_uint32x4 v, __amdgpu_buffer_rsrc_t r, int voff, int soff) {
    __builtin_amdgcn_raw_buffer_store_b128(v, r, voff, soff, 0);
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_nop 0");
    __builtin_amdgcn_sched_barrier(0);
}
"""


def codegen_buffer_store_b128(value, rsrc, voffset, soffset):
    return Builtin(body=_BUFFER_STORE_B128_BODY, includes=[],
                   return_val=f"lk_buffer_store_b128((lk_uint32x4)({value}), {rsrc}, {voffset}, {soffset})")


@builtin(eval_return_type=void, codegen_func=codegen_buffer_store_b128)
def amdgcn_buffer_store_b128(value, rsrc, voffset, soffset):
    """``buffer_store_dwordx4``: 16 bytes per lane at voffset (per lane) + soffset (uniform) of the buffer
    ``rsrc``; stores past its size are dropped. Followed by ``s_nop 0``: the hardware reads the last 8
    bytes of the data a cycle late, and LLVM only keeps that cycle free when soffset is not an SGPR."""
    raise RuntimeError("amdgcn_buffer_store_b128 should never be called in compilation")
