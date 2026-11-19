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
Atomic operation intrinsics.
"""

from little_kernel.language.builtin_base import builtin, Builtin, CODEGEN_NEEDS_TYPES_ATTR
from little_kernel.core.type_system import int16, uint16, int32, int64, uint32, uint64, float32, void, LLType
from little_kernel.core.memory_effects import RW_GMEM_ATOMIC_SYNC, effect, operand_region

# Every intrinsic in this module is a hardware-atomic memory access; each
# declares its region effect via MEMORY_EFFECTS_ATTR (see
# core/memory_effects.py) so effect-driven analyses (dataflow mem-event
# collection, channel/pipeline validation, race and write-only-init checks)
# never see an atomic as side-effect free.  All scalar ops are RMW, and the
# fire-and-forget red.global vector reductions are RW as well: the stored
# value depends on prior memory contents.  Pointer-form ops touch the
# element at ``addr`` (region=None); (tensor, index) element ops and the
# (mem, index) vector reductions derive their region from the index operand.


def _atomic_effects_addr():
    return (effect(RW_GMEM_ATOMIC_SYNC, "addr"), )


def _atomic_effects_element():
    return (effect(RW_GMEM_ATOMIC_SYNC, "tensor", region=operand_region("index", 1)), )


# ==============================================================================
# Helpers for dynamic return type inference
# ==============================================================================


def _atomic_return_type_from_addr(addr: LLType) -> LLType:
    """Infer return type from pointer/tensor addr. Atomic ops return old value (same type as element)."""
    if addr is None:
        return int32
    if hasattr(addr, "is_pointer") and addr.is_pointer() and hasattr(addr, "inner_type"):
        return addr.inner_type
    if hasattr(addr, "is_tensor") and addr.is_tensor() and hasattr(addr, "element_type"):
        return addr.element_type
    return int32  # Fallback when type cannot be inferred


def _atomic_addr_ptr(addr: str, addr_dtype) -> str:
    """Normalize the address operand of a pointer-form atomic to address
    position, using the inferred LLType of the argument.

    The natural element form ``atomic_add(smem[i], v)`` passes an lvalue
    (tensor subscript, scalar local) whose inferred type is the *element*
    type; emitting the rendered expression directly would hand the CUDA
    atomic an rvalue slot (``atomicAdd(smem[i], v)`` — an int32_t, not an
    int32_t*), which NVCC rejects with "no instance of overloaded function
    atomicAdd matches the argument list".  The intrinsics are typed-address
    ops, so an element-typed operand is taken by address: ``&(smem[i])``.
    A pointer-typed operand (``smem + i``, a raw pointer local) is already
    in address position and passes through unchanged, as does an unknown
    type (no inference record — historical behavior is preserved verbatim).
    """
    if addr_dtype is not None and not (addr_dtype.is_pointer() or addr_dtype.is_tensor()):
        return f"&({addr})"
    return addr


# ==============================================================================
# atomicAdd
# ==============================================================================

# ==============================================================================
# Vector reduction atomics (red.global.add.v{2,4}.f32)
#
# Contiguous fp32 vector adds to GLOBAL memory, element-indexed like the
# gmem_ld/st_u32xN family: byte address = base + index * 4 must be width
# aligned (8 B for x2, 16 B for x4) and the destination must be a global
# tensor (red.global). Semantics equal two/four independent atomicAdds in
# lane order; no return value (fire-and-forget reduction). Legality is an
# explicit frontend contract — the compiler emits no scope/alignment check.
# Requires sm_90+ (PTX red.global.add.v2/v4.f32 vector reduction).


def codegen_atomic_add_f32x2(mem, index, v0, v1):
    body = """
__device__ __forceinline__ void atomic_add_f32x2_fn(float* ptr, float v0, float v1) {
    asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(ptr), "f"(v0), "f"(v1) : "memory");
}
"""
    return Builtin(body=body, includes=[], return_val=f"atomic_add_f32x2_fn({mem} + {index}, {v0}, {v1})")


@builtin(eval_return_type=void, codegen_func=codegen_atomic_add_f32x2,
         effects=(effect(RW_GMEM_ATOMIC_SYNC, "mem", region=operand_region("index", 2)), ))
def atomic_add_f32x2(mem, index, v0, v1):
    """Vector red.global.add.v2.f32 at mem + index (8-byte aligned, global).
    Args: (mem, element_index, v0, v1); lane order matches memory order."""
    raise RuntimeError("should not be called in compilation")


def codegen_atomic_add_f32x4(mem, index, v0, v1, v2, v3):
    body = """
__device__ __forceinline__ void atomic_add_f32x4_fn(float* ptr, float v0, float v1, float v2, float v3) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "f"(v0), "f"(v1), "f"(v2), "f"(v3) : "memory");
}
"""
    return Builtin(body=body, includes=[], return_val=f"atomic_add_f32x4_fn({mem} + {index}, {v0}, {v1}, {v2}, {v3})")


@builtin(eval_return_type=void, codegen_func=codegen_atomic_add_f32x4,
         effects=(effect(RW_GMEM_ATOMIC_SYNC, "mem", region=operand_region("index", 4)), ))
def atomic_add_f32x4(mem, index, v0, v1, v2, v3):
    """Vector red.global.add.v4.f32 at mem + index (16-byte aligned, global).
    Args: (mem, element_index, v0, v1, v2, v3); lane order matches memory order."""
    raise RuntimeError("should not be called in compilation")


def codegen_atomic_add(addr, val, _arg_dtypes=None):
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    return Builtin(body="", includes=[], return_val=f"atomicAdd({ptr}, {val})")


@builtin(eval_return_type=lambda addr, val: _atomic_return_type_from_addr(addr), codegen_func=codegen_atomic_add,
         effects=_atomic_effects_addr())
def atomic_add(addr, val):
    """atomicAdd on a pointer, or on an element lvalue (``tensor[i]`` /
    scalar local), whose address is taken automatically. Returns old value."""
    raise RuntimeError("should not call atomic_add in compilation")


setattr(atomic_add, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# 64-bit atomicAdd (CUDA has no signed/unsigned int64 atomicAdd overload on
# LP64 hosts where int64_t is `long`; emit the PTX u64 form, which is
# bit-identical for two's-complement signed add)
# ==============================================================================


def _atomic_add_i64_return_type(addr, _val):
    result_type = _atomic_return_type_from_addr(addr)
    if result_type not in (int64, uint64):
        raise TypeError("atomic_add_i64 requires an int64 or uint64 pointer/tensor element "
                        "type; use atomic_add for 32-bit and float element types")
    return result_type


def codegen_atomic_add_i64(addr, val, _arg_dtypes=None):
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    body = """
__device__ __forceinline__ int64_t lk_atomic_add_i64_fn(
        int64_t* ptr, int64_t value) {
    int64_t old;
    asm volatile("atom.add.u64 %0, [%1], %2;"
                 : "=l"(old) : "l"(ptr), "l"(value) : "memory");
    return old;
}

__device__ __forceinline__ uint64_t lk_atomic_add_i64_fn(
        uint64_t* ptr, uint64_t value) {
    uint64_t old;
    asm volatile("atom.add.u64 %0, [%1], %2;"
                 : "=l"(old) : "l"(ptr), "l"(value) : "memory");
    return old;
}
"""
    return Builtin(
        body=body,
        includes=["<cstdint>"],
        return_val=f"lk_atomic_add_i64_fn({ptr}, {val})",
    )


@builtin(
    eval_return_type=_atomic_add_i64_return_type,
    codegen_func=codegen_atomic_add_i64,
    effects=_atomic_effects_addr(),
)
def atomic_add_i64(addr, val):
    """64-bit global atomic add for int64/uint64 pointer or tensor element
    types. Returns the old value.

    CUDA's atomicAdd has no overload for `long*` (LP64 int64_t), so this
    intrinsic emits `atom.add.u64` on the generic address - bit-identical to a
    signed 64-bit add under two's complement. Device scope, relaxed order
    (same semantics as plain atomicAdd)."""
    raise RuntimeError("should not call atomic_add_i64 in compilation")


setattr(atomic_add_i64, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# atomicAdd with device-scope acquire-release ordering
# ==============================================================================


def _atomic_add_acq_rel_gpu_return_type(addr, _val):
    result_type = _atomic_return_type_from_addr(addr)
    if result_type not in (int32, uint32):
        raise TypeError("atomic_add_acq_rel_gpu only supports int32 and uint32 pointers")
    return result_type


def codegen_atomic_add_acq_rel_gpu(addr, val, _arg_dtypes=None):
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    body = """
__device__ __forceinline__ int32_t atomic_add_acq_rel_gpu_fn(
        int32_t* ptr, int32_t value) {
    int32_t old;
    asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], %2;"
                 : "=r"(old) : "l"(ptr), "r"(value) : "memory");
    return old;
}

__device__ __forceinline__ uint32_t atomic_add_acq_rel_gpu_fn(
        uint32_t* ptr, uint32_t value) {
    uint32_t old;
    asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;"
                 : "=r"(old) : "l"(ptr), "r"(value) : "memory");
    return old;
}
"""
    return Builtin(
        body=body,
        includes=["<cstdint>"],
        return_val=f"atomic_add_acq_rel_gpu_fn({ptr}, {val})",
    )


@builtin(
    eval_return_type=_atomic_add_acq_rel_gpu_return_type,
    codegen_func=codegen_atomic_add_acq_rel_gpu,
    effects=_atomic_effects_addr(),
)
def atomic_add_acq_rel_gpu(addr, val):
    """Acquire-release atomic add, dispatched from the pointer element type."""
    raise RuntimeError("should not call atomic_add_acq_rel_gpu in compilation")


setattr(atomic_add_acq_rel_gpu, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# 64-bit atomicCAS (CUDA has no atomicCAS overload for `long*` on LP64 hosts
# where int64_t/uint64_t alias `long`; the `unsigned long long*` overload is
# a distinct type, so `atomicCAS(&(ht[i]), ...)` on a Tensor[ll.uint64] shared
# or global element lvalue fails overload resolution at NVCC time.  Emit the
# PTX b64 form on the generic address, which is bit-identical for the two's
# -complement compare-exchange and covers CTA shared-memory slots as well as
# global tensors — the shared-element CAS a per-block last-wins hash table
# needs, packing (order tag << 32 | fp32 bits) into one CAS word.)
# ==============================================================================


def _atomic_cas_i64_return_type(addr, _compare, _val):
    # An element lvalue operand (``tensor[i]``, scalar local) infers as the
    # element type directly; a tensor/pointer operand resolves through
    # _atomic_return_type_from_addr.  Both are legalized in the result.
    if addr in (int64, uint64):
        return addr
    result_type = _atomic_return_type_from_addr(addr)
    if result_type not in (int64, uint64):
        raise TypeError("atomic_cas_i64 requires an int64 or uint64 pointer/tensor element "
                        "type; use atomic_cas for 32-bit element types")
    return result_type


def codegen_atomic_cas_i64(addr, compare, val, _arg_dtypes=None):
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    body = """
__device__ __forceinline__ int64_t lk_atomic_cas_i64_fn(
        int64_t* ptr, int64_t compare, int64_t value) {
    int64_t old;
    asm volatile("atom.cas.b64 %0, [%1], %2, %3;"
                 : "=l"(old) : "l"(ptr), "l"(compare), "l"(value) : "memory");
    return old;
}

__device__ __forceinline__ uint64_t lk_atomic_cas_i64_fn(
        uint64_t* ptr, uint64_t compare, uint64_t value) {
    uint64_t old;
    asm volatile("atom.cas.b64 %0, [%1], %2, %3;"
                 : "=l"(old) : "l"(ptr), "l"(compare), "l"(value) : "memory");
    return old;
}
"""
    return Builtin(
        body=body,
        includes=["<cstdint>"],
        return_val=f"lk_atomic_cas_i64_fn({ptr}, {compare}, {val})",
    )


@builtin(
    eval_return_type=_atomic_cas_i64_return_type,
    codegen_func=codegen_atomic_cas_i64,
    effects=_atomic_effects_addr(),
)
def atomic_cas_i64(addr, compare, val):
    """64-bit device-scope atomic compare-and-exchange on an int64/uint64
    pointer or element lvalue (``tensor[i]`` — shared or global — whose
    address is taken automatically).  Returns the old value.

    CUDA's atomicCAS has no overload for `long*` (LP64 int64_t/uint64_t are
    `long`, and the `unsigned long long*` overload does not match), so this
    intrinsic emits `atom.cas.b64` on the generic address, mirroring
    atomic_add_i64.  The generic-space form covers CTA shared-memory slots
    (`alloc_shared_memory` / dynamic-shared slices) without a separate cvta
    step, enabling parallel last-wins inserts on packed-u64 hash entries."""
    raise RuntimeError("should not call atomic_cas_i64 in compilation")


setattr(atomic_cas_i64, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# atomicCAS_system (cross-rank / system-wide)
# ==============================================================================


def codegen_atomic_cas_system(addr, compare, val, _arg_dtypes=None):
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    return Builtin(body="", includes=[], return_val=f"atomicCAS_system({ptr}, {compare}, {val})")


@builtin(eval_return_type=lambda addr, compare, val: _atomic_return_type_from_addr(addr),
         codegen_func=codegen_atomic_cas_system, effects=_atomic_effects_addr())
def atomic_cas_system(addr, compare, val):
    """System-scope atomicCAS. Returns old value."""
    raise RuntimeError("should not call atomic_cas_system in compilation")


setattr(atomic_cas_system, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# atomicCAS (device-scope)
# ==============================================================================


def _atomic_cas_element_type(addr: LLType) -> LLType:
    """Resolve the value type CAS operates on.

    An element lvalue operand (``tensor[i]``, scalar local) infers as the
    element type directly (its address is taken in codegen); a
    tensor/pointer operand resolves through _atomic_return_type_from_addr.
    """
    if addr is not None and addr in (int16, uint16, int32, uint32, int64, uint64, float32):
        return addr
    return _atomic_return_type_from_addr(addr)


def codegen_atomic_cas_i16(addr, compare, val, _arg_dtypes=None):
    # PTX atom.cas.b16 (sm_70+, so available on sm_90/sm_100) is the legal
    # form for signed 16-bit element lvalues: bit-identical two's-complement
    # compare-exchange, generic-address (shared or global).
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    body = """
__device__ __forceinline__ int16_t lk_atomic_cas_i16_fn(
        int16_t* ptr, int16_t compare, int16_t value) {
    int16_t old;
    asm volatile("atom.cas.b16 %0, [%1], %2, %3;"
                 : "=h"(old) : "l"(ptr), "h"(compare), "h"(value) : "memory");
    return old;
}
"""
    return Builtin(
        body=body,
        includes=["<cstdint>"],
        return_val=f"lk_atomic_cas_i16_fn({ptr}, {compare}, {val})",
    )


def codegen_atomic_cas_f32(addr, compare, val, _arg_dtypes=None):
    # PTX atom.cas.b32 (sm_20+, so available on sm_90/sm_100) is the legal
    # form for fp32 element lvalues: CUDA atomicCAS has no float* overload
    # (int*/unsigned int*/unsigned short int*/unsigned long long int* only)
    # and pointer types never convert implicitly, so a `Tensor[ll.float32]`
    # element lvalue would emit `atomicCAS(&(smem[0]))` and fail NVCC overload
    # resolution.  Bitcast through uint32 on the generic address; the
    # compare-exchange compares bit patterns, so fp32 bit equality is
    # preserved exactly.
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    body = """
__device__ __forceinline__ float lk_atomic_cas_f32_fn(
        float* ptr, float compare, float value) {
    uint32_t old_bits;
    uint32_t compare_bits = __float_as_uint(compare);
    uint32_t value_bits = __float_as_uint(value);
    asm volatile("atom.cas.b32 %0, [%1], %2, %3;"
                 : "=r"(old_bits)
                 : "l"(ptr), "r"(compare_bits), "r"(value_bits)
                 : "memory");
    return __uint_as_float(old_bits);
}
"""
    return Builtin(
        body=body,
        includes=["<cstdint>"],
        return_val=f"lk_atomic_cas_f32_fn({ptr}, {compare}, {val})",
    )


def codegen_atomic_cas(addr, compare, val, _arg_dtypes=None):
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    elem_ty = (_atomic_cas_element_type(_arg_dtypes[0])
               if _arg_dtypes and len(_arg_dtypes) > 0 and _arg_dtypes[0] is not None else int32)
    if elem_ty in (int64, uint64):
        # CUDA's atomicCAS has no overload for `long*` (LP64 int64_t/uint64_t
        # are `long`, distinct from the `unsigned long long*` overload), so a
        # 64-bit shared/global element lvalue (`&(ht[i])`) would fail NVCC
        # overload resolution.  Emit the PTX b64 form on the generic address
        # instead; bit-identical under two's complement and valid on both the
        # shared and global windows of the generic space.
        return codegen_atomic_cas_i64(addr, compare, val, _arg_dtypes)
    if elem_ty is int16:
        # CUDA's atomicCAS overload set is int*/unsigned int*/unsigned
        # short int*/unsigned long long int* — there is no signed short*
        # overload and pointer types never convert implicitly, so a
        # `Tensor[ll.int16]` element lvalue would emit
        # `atomicCAS(&(smem[0]))` and fail NVCC overload resolution.
        # Legalize to PTX atom.cas.b16, mirroring the 64-bit path.
        return codegen_atomic_cas_i16(addr, compare, val, _arg_dtypes)
    if elem_ty is float32:
        # CUDA's atomicCAS has no float* overload; legalize to PTX
        # atom.cas.b32 via bit-identical fp32 compare-exchange (bit pattern
        # equality is the natural semantics for fp32 CAS, and matches the
        # bitcast-based legalization used elsewhere in the stack).
        return codegen_atomic_cas_f32(addr, compare, val, _arg_dtypes)
    return Builtin(body="", includes=[], return_val=f"atomicCAS({ptr}, {compare}, {val})")


@builtin(eval_return_type=lambda addr, compare, val: _atomic_cas_element_type(addr), codegen_func=codegen_atomic_cas,
         effects=_atomic_effects_addr())
def atomic_cas(addr, compare, val):
    """Device-scope atomicCAS on a pointer or element lvalue. Returns old.

    int16/uint16/int32/uint32/int64/uint64/fp32 element types are
    supported.  Element types CUDA ``atomicCAS`` cannot overload are
    legalized to PTX ``atom.cas`` on the generic address (bit-identical
    two's-complement compare-exchange): int64/uint64 to ``atom.cas.b64``
    (NVCC has no LP64 ``long*`` overload), signed int16 to ``atom.cas.b16``
    (NVCC has no signed ``short*`` overload; the ``unsigned short*``
    overload used by uint16 lvalues is real and keeps the CUDA
    ``atomicCAS`` path), and fp32 to ``atom.cas.b32`` (NVCC has no
    ``float*`` overload; fp32 bit-pattern equality is preserved exactly)."""
    raise RuntimeError("should not call atomic_cas in compilation")


setattr(atomic_cas, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# atomicMin
# ==============================================================================


def codegen_atomic_min(addr, val, _arg_dtypes=None):
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    return Builtin(body="", includes=[], return_val=f"atomicMin({ptr}, {val})")


@builtin(eval_return_type=lambda addr, val: _atomic_return_type_from_addr(addr), codegen_func=codegen_atomic_min,
         effects=_atomic_effects_addr())
def atomic_min(addr, val):
    """atomicMin on a pointer or element lvalue. Returns old value."""
    raise RuntimeError("should not call atomic_min in compilation")


setattr(atomic_min, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# atomicMax
# ==============================================================================


def codegen_atomic_max(addr, val, _arg_dtypes=None):
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    return Builtin(body="", includes=[], return_val=f"atomicMax({ptr}, {val})")


@builtin(eval_return_type=lambda addr, val: _atomic_return_type_from_addr(addr), codegen_func=codegen_atomic_max,
         effects=_atomic_effects_addr())
def atomic_max(addr, val):
    """atomicMax on a pointer or element lvalue. Returns old value."""
    raise RuntimeError("should not call atomic_max in compilation")


setattr(atomic_max, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# atomicOr
# ==============================================================================


def codegen_atomic_or(addr, val, _arg_dtypes=None):
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    return Builtin(body="", includes=[], return_val=f"atomicOr({ptr}, {val})")


@builtin(eval_return_type=lambda addr, val: _atomic_return_type_from_addr(addr), codegen_func=codegen_atomic_or,
         effects=_atomic_effects_addr())
def atomic_or(addr, val):
    """atomicOr on a pointer or element lvalue. Returns old value."""
    raise RuntimeError("should not call atomic_or in compilation")


setattr(atomic_or, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# atomicAnd
# ==============================================================================


def codegen_atomic_and(addr, val, _arg_dtypes=None):
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    return Builtin(body="", includes=[], return_val=f"atomicAnd({ptr}, {val})")


@builtin(eval_return_type=lambda addr, val: _atomic_return_type_from_addr(addr), codegen_func=codegen_atomic_and,
         effects=_atomic_effects_addr())
def atomic_and(addr, val):
    """atomicAnd on a pointer or element lvalue. Returns old value."""
    raise RuntimeError("should not call atomic_and in compilation")


setattr(atomic_and, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# atomicSub / atomicXor (32-bit only)
#
# The CUDA C++ atomic API only overloads atomicSub and atomicXor for `int*`
# and `unsigned int*` (see device_atomic_functions.h: no float, no 64-bit
# overloads), so unlike the generic atomicAdd/atomicMin wrappers above the
# element type is gated at trace time (legality before emission). Both ops
# are relaxed device-scope RMW and return the old value. atomicXor covers
# parity/mask toggling in scatter epilogues; atomicSub covers signed
# decrement and (with unsigned wraparound) budget/lease counters.
# ==============================================================================


def _atomic_sub_xor_return_type(op_name):

    def _return_type(addr, _val):
        result_type = _atomic_return_type_from_addr(addr)
        if result_type not in (int32, uint32):
            raise TypeError(f"{op_name} requires an int32 or uint32 pointer/tensor "
                            f"element type; the CUDA {op_name.replace('_', '', 1)} API has "
                            "no float or 64-bit overloads")
        return result_type

    return _return_type


def codegen_atomic_sub(addr, val, _arg_dtypes=None):
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    return Builtin(body="", includes=[], return_val=f"atomicSub({ptr}, {val})")


@builtin(eval_return_type=_atomic_sub_xor_return_type("atomic_sub"), codegen_func=codegen_atomic_sub,
         effects=_atomic_effects_addr())
def atomic_sub(addr, val):
    """atomicSub on an int32/uint32 pointer or element lvalue. Returns old value."""
    raise RuntimeError("should not call atomic_sub in compilation")


setattr(atomic_sub, CODEGEN_NEEDS_TYPES_ATTR, True)


def codegen_atomic_xor(addr, val, _arg_dtypes=None):
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    return Builtin(body="", includes=[], return_val=f"atomicXor({ptr}, {val})")


@builtin(eval_return_type=_atomic_sub_xor_return_type("atomic_xor"), codegen_func=codegen_atomic_xor,
         effects=_atomic_effects_addr())
def atomic_xor(addr, val):
    """atomicXor on an int32/uint32 pointer or element lvalue. Returns old value."""
    raise RuntimeError("should not call atomic_xor in compilation")


setattr(atomic_xor, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# Element atomics on (tensor, index)
#
# Frontends cannot form the address `&tensor[i]` in kernel source (a tensor
# subscript evaluates to the element value), so these intrinsics take the
# tensor argument plus a separate element index and build the address inside
# the codegen fragment, following the same element-typed-pointer + element
# -offset contract as the ld_global_v4_u32/st_global_v4_u32 helpers in
# intrin/memory.py.
#
# Legality contract (caller's responsibility):
#   * `index` is an element offset (any integer type); the generated address
#     is `tensor + index` and must be in bounds.
#   * The tensor must reside in global memory and be element-typed exactly as
#     named (int32/uint32/uint64/float32); atomicity is per element access.
#   * All ops are relaxed device-scope. For cross-CTA producer/consumer
#     ordering combine with `ll.threadfence()` (intrin/barrier.py, release
#     ordering) on the producer and `ll.ld_acquire_gpu`-style acquire loads on
#     the consumer, as in persistent-kernel work queues and spin flags.
# ==============================================================================


def _codegen_atomic_add_global(tensor, index, val):
    return Builtin(body="", includes=[], return_val=f"atomicAdd(({tensor}) + ({index}), {val})")


@builtin(eval_return_type=int32, codegen_func=_codegen_atomic_add_global, effects=_atomic_effects_element())
def atomic_add_global_i32(tensor, index, val):
    """Relaxed device-scope atomicAdd on `tensor[index]` (int32). Returns old."""
    raise RuntimeError("should not call atomic_add_global_i32 in compilation")


@builtin(eval_return_type=uint32, codegen_func=_codegen_atomic_add_global, effects=_atomic_effects_element())
def atomic_add_global_u32(tensor, index, val):
    """Relaxed device-scope atomicAdd on `tensor[index]` (uint32). Returns old."""
    raise RuntimeError("should not call atomic_add_global_u32 in compilation")


def codegen_atomic_add_global_u64(tensor, index, val):
    return Builtin(body="", includes=[],
                   return_val=f"atomicAdd((unsigned long long*)(({tensor}) + ({index})), (unsigned long long)({val}))")


@builtin(eval_return_type=uint64, codegen_func=codegen_atomic_add_global_u64, effects=_atomic_effects_element())
def atomic_add_global_u64(tensor, index, val):
    """Relaxed device-scope atomicAdd on `tensor[index]` (uint64). Returns old."""
    raise RuntimeError("should not call atomic_add_global_u64 in compilation")


@builtin(eval_return_type=float32, codegen_func=_codegen_atomic_add_global, effects=_atomic_effects_element())
def atomic_add_global_f32(tensor, index, val):
    """Relaxed device-scope atomicAdd on `tensor[index]` (float32). Returns old."""
    raise RuntimeError("should not call atomic_add_global_f32 in compilation")


def codegen_atomic_cas_global_i32(tensor, index, compare, val):
    return Builtin(body="", includes=[], return_val=f"atomicCAS(({tensor}) + ({index}), {compare}, {val})")


@builtin(eval_return_type=int32, codegen_func=codegen_atomic_cas_global_i32, effects=_atomic_effects_element())
def atomic_cas_global_i32(tensor, index, compare, val):
    """Relaxed device-scope atomicCAS on `tensor[index]` (int32). Returns old."""
    raise RuntimeError("should not call atomic_cas_global_i32 in compilation")


def codegen_atomic_cas_global_u32(tensor, index, compare, val):
    return Builtin(body="", includes=[], return_val=f"atomicCAS(({tensor}) + ({index}), {compare}, {val})")


@builtin(eval_return_type=uint32, codegen_func=codegen_atomic_cas_global_u32, effects=_atomic_effects_element())
def atomic_cas_global_u32(tensor, index, compare, val):
    """Relaxed device-scope atomicCAS on `tensor[index]` (uint32). Returns old."""
    raise RuntimeError("should not call atomic_cas_global_u32 in compilation")


def codegen_atomic_exch_global_i32(tensor, index, val):
    return Builtin(body="", includes=[], return_val=f"atomicExch(({tensor}) + ({index}), {val})")


@builtin(eval_return_type=int32, codegen_func=codegen_atomic_exch_global_i32, effects=_atomic_effects_element())
def atomic_exch_global_i32(tensor, index, val):
    """Relaxed device-scope atomicExch on `tensor[index]` (int32). Returns old.

    Flag-publish primitive: combine with `ll.threadfence()` before the
    exchange so prior writes are visible when the flag becomes visible."""
    raise RuntimeError("should not call atomic_exch_global_i32 in compilation")


def codegen_atomic_exch_global_u32(tensor, index, val):
    return Builtin(body="", includes=[], return_val=f"atomicExch(({tensor}) + ({index}), {val})")


@builtin(eval_return_type=uint32, codegen_func=codegen_atomic_exch_global_u32, effects=_atomic_effects_element())
def atomic_exch_global_u32(tensor, index, val):
    """Relaxed device-scope atomicExch on `tensor[index]` (uint32). Returns old."""
    raise RuntimeError("should not call atomic_exch_global_u32 in compilation")


def _codegen_atomic_sub_global(tensor, index, val):
    return Builtin(body="", includes=[], return_val=f"atomicSub(({tensor}) + ({index}), {val})")


@builtin(eval_return_type=int32, codegen_func=_codegen_atomic_sub_global, effects=_atomic_effects_element())
def atomic_sub_global_i32(tensor, index, val):
    """Relaxed device-scope atomicSub on `tensor[index]` (int32). Returns old."""
    raise RuntimeError("should not call atomic_sub_global_i32 in compilation")


@builtin(eval_return_type=uint32, codegen_func=_codegen_atomic_sub_global, effects=_atomic_effects_element())
def atomic_sub_global_u32(tensor, index, val):
    """Relaxed device-scope atomicSub on `tensor[index]` (uint32). Returns old."""
    raise RuntimeError("should not call atomic_sub_global_u32 in compilation")


def _codegen_atomic_xor_global(tensor, index, val):
    return Builtin(body="", includes=[], return_val=f"atomicXor(({tensor}) + ({index}), {val})")


@builtin(eval_return_type=int32, codegen_func=_codegen_atomic_xor_global, effects=_atomic_effects_element())
def atomic_xor_global_i32(tensor, index, val):
    """Relaxed device-scope atomicXor on `tensor[index]` (int32). Returns old."""
    raise RuntimeError("should not call atomic_xor_global_i32 in compilation")


@builtin(eval_return_type=uint32, codegen_func=_codegen_atomic_xor_global, effects=_atomic_effects_element())
def atomic_xor_global_u32(tensor, index, val):
    """Relaxed device-scope atomicXor on `tensor[index]` (uint32). Returns old."""
    raise RuntimeError("should not call atomic_xor_global_u32 in compilation")


# ==============================================================================
# atomicInc (wrap-around ticket counter)
# ==============================================================================


def _atomic_inc_return_type(addr, _bound):
    result_type = _atomic_return_type_from_addr(addr)
    if result_type not in (int32, uint32):
        raise TypeError("atomic_inc only supports int32 and uint32 pointers")
    return uint32


def codegen_atomic_inc(addr, bound, _arg_dtypes=None):
    ptr = _atomic_addr_ptr(addr, _arg_dtypes[0] if _arg_dtypes else None)
    body = """
__device__ __forceinline__ uint32_t atomic_inc_fn(
        uint32_t* ptr, uint32_t bound) {
    return atomicInc(ptr, bound);
}

__device__ __forceinline__ uint32_t atomic_inc_fn(
        int32_t* ptr, uint32_t bound) {
    return atomicInc(reinterpret_cast<uint32_t*>(ptr), bound);
}
"""
    return Builtin(body=body, includes=["<cstdint>"], return_val=f"atomic_inc_fn({ptr}, {bound})")


@builtin(eval_return_type=_atomic_inc_return_type, codegen_func=codegen_atomic_inc, effects=_atomic_effects_addr())
def atomic_inc(addr, bound):
    """atomicInc on an int32 or uint32 pointer. Atomically computes
    old = *addr; *addr = (old >= bound) ? 0 : old + 1; and returns old
    (always uint32). Hardware ticket for single-launch turnstile
    epilogues (e.g. the last-arriving block of a grid-wide cross-CTA
    reduction sees the highest ticket and performs the final merge),
    paired with a device-scope fence (threadfence) before the ticket
    take.  Legality: the pointer element type must be int32 or uint32
    (counters commonly live in int32 workspaces); other element types
    raise TypeError.
    """
    raise RuntimeError("should not call atomic_inc in compilation")


setattr(atomic_inc, CODEGEN_NEEDS_TYPES_ATTR, True)
