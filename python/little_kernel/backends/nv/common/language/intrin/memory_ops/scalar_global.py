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
Scalar/element global-memory access and acquire/release ops.

RF10 canonical owner module; see little_kernel.backends.nv.common.language.intrin.contracts
for the shared operand-unit/lane/alignment/cache-policy/effect inventory.
"""

from little_kernel.language.builtin_base import builtin, Builtin, CODEGEN_NEEDS_TYPES_ATTR
from little_kernel.core.type_system import LLType, void, Tensor, int8, uint8, uint16, uint32, uint64, int32, bfloat16, float16, float32, u32x2, u32x4
import little_kernel.backends.nv.common.language.intrin.contracts as contracts

_normalize_cache_policy = contracts.normalize_cache_policy
_POLICY_ALIASES = contracts.POLICY_ALIASES


def codegen_ldcg(ptr, offset):
    return Builtin(body="", includes=["<device_functions.h>"], return_val=f"__ldcg({ptr} + {offset})")


def eval_return_type_ldcg(dtype, offset_dtype):
    """Evaluate return type for __ldcg function."""
    from little_kernel.core.type_system import TensorType
    if dtype is None:
        raise TypeError("__ldcg: cannot infer type of first argument (ptr)")
    if isinstance(dtype, TensorType):
        return dtype.element_type
    else:
        raise TypeError(f"__ldcg expects Tensor as first argument, got {type(dtype).__name__}: {dtype}")


@builtin(eval_return_type=eval_return_type_ldcg, codegen_func=codegen_ldcg)
def __ldcg(ptr: Tensor, offset: int):
    """Load a value from the given pointer with the given offset."""
    raise RuntimeError("ldcg should never be called in compilation")


def codegen_ldca(ptr, offset):
    return Builtin(body="", includes=["<device_functions.h>"], return_val=f"__ldca({ptr} + {offset})")


def eval_return_type_ldca(dtype, offset_dtype):
    """Evaluate return type for __ldca function."""
    from little_kernel.core.type_system import TensorType
    if dtype is None:
        raise TypeError("__ldca: cannot infer type of first argument (ptr)")
    if isinstance(dtype, TensorType):
        return dtype.element_type
    else:
        raise TypeError(f"__ldca expects Tensor as first argument, got {type(dtype).__name__}: {dtype}")


@builtin(eval_return_type=eval_return_type_ldca, codegen_func=codegen_ldca)
def __ldca(ptr: Tensor, offset: int):
    """Load a value from the given pointer with the given offset."""
    raise RuntimeError("ldca should never be called in compilation")


# ==============================================================================
# Ordered global memory
# ==============================================================================


def codegen_ld_acquire_gpu(ptr):
    body = """
__device__ __forceinline__ int32_t ld_acquire_gpu_fn(const int32_t* ptr) {
    int32_t value;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
                 : "=r"(value) : "l"(ptr) : "memory");
    return value;
}
"""
    return Builtin(body=body, includes=["<cstdint>"], return_val=f"ld_acquire_gpu_fn({ptr})")


@builtin(eval_return_type=int32, codegen_func=codegen_ld_acquire_gpu)
def ld_acquire_gpu(ptr):
    """Load an int32 from global memory with device-scope acquire ordering."""
    raise RuntimeError("ld_acquire_gpu should never be called in compilation")


def codegen_st_release_gpu(ptr, value):
    body = """
__device__ __forceinline__ void st_release_gpu_fn(int32_t* ptr, int32_t value) {
    asm volatile("st.release.gpu.global.b32 [%0], %1;"
                 :: "l"(ptr), "r"(value) : "memory");
}
"""
    return Builtin(body=body, includes=["<cstdint>"], return_val=f"st_release_gpu_fn({ptr}, {value})")


@builtin(eval_return_type=void, codegen_func=codegen_st_release_gpu)
def st_release_gpu(ptr, value):
    """Store an int32 to global memory with device-scope release ordering."""
    raise RuntimeError("st_release_gpu should never be called in compilation")


def codegen_ld_acquire_system(ptr):
    body = """
__device__ __forceinline__ int32_t ld_acquire_system_fn(const int32_t* ptr) {
    int32_t value;
    asm volatile("ld.acquire.sys.global.b32 %0, [%1];"
                 : "=r"(value) : "l"(ptr) : "memory");
    return value;
}
"""
    return Builtin(body=body, includes=["<cstdint>"], return_val=f"ld_acquire_system_fn({ptr})")


@builtin(eval_return_type=int32, codegen_func=codegen_ld_acquire_system)
def ld_acquire_system(ptr):
    """Load an int32 from global memory with system-scope acquire ordering."""
    raise RuntimeError("ld_acquire_system should never be called in compilation")


def codegen_st_release_system(ptr, value):
    body = """
__device__ __forceinline__ void st_release_system_fn(int32_t* ptr, int32_t value) {
    asm volatile("st.release.sys.global.b32 [%0], %1;"
                 :: "l"(ptr), "r"(value) : "memory");
}
"""
    return Builtin(body=body, includes=["<cstdint>"], return_val=f"st_release_system_fn({ptr}, {value})")


@builtin(eval_return_type=void, codegen_func=codegen_st_release_system)
def st_release_system(ptr, value):
    """Store an int32 to global memory with system-scope release ordering."""
    raise RuntimeError("st_release_system should never be called in compilation")


# Cache-policy name -> scalar load intrin applied to a typed pointer, or
# None for a plain subscript expression.
_GMEM_LD_SCALAR_INTRIN = {
    "wb": None,
    "ca": "__ldca",
    "cg": "__ldcg",
    "cs": "__ldcs",
    "lu": "__ldlu",
    "nc": "__ldg",
    "cv": "__ldcv",
}

# Cache-policy name -> scalar store intrin, or None for a plain subscript
# assignment.
_GMEM_ST_SCALAR_INTRIN = {
    "wb": None,
    "cg": "__stcg",
    "cs": "__stcs",
    "wt": "__stwt",
}


def _is_byte_int_element(dtype):
    """1-byte integer scalar element type (bool/int8/uint8) or False.

    These are the element dtypes whose C spelling (`bool`, `int8_t`,
    `uint8_t`) has no overload in the CUDA ``__ld*/__st*`` cache-operator
    family (bool/1-byte payloads are absent from the overload set), so a
    policy-selected scalar load/store on such a tensor must be legalized
    through an unsigned-char reinterpret.  Value semantics are bit-exact:
    it is purely a cache-policy spelling legalization.
    """
    from little_kernel.core.type_system import IntType
    return isinstance(dtype, IntType) and dtype.bits <= 8


def _gmem_scalar_byte_element(mem_dtype):
    """Element LLType of a Tensor LLType when it is a 1-byte int scalar."""
    from little_kernel.core.type_system import TensorType
    if isinstance(mem_dtype, TensorType) and _is_byte_int_element(mem_dtype.element_type):
        return mem_dtype.element_type
    return None


def codegen_gmem_ld_scalar(mem, index, policy=None, _arg_dtypes=None):
    policy = _normalize_cache_policy(policy, _GMEM_LD_SCALAR_INTRIN.keys(), "wb", "gmem_ld_scalar")
    intrin = _GMEM_LD_SCALAR_INTRIN[policy]
    if intrin is None:
        return Builtin(body="", includes=[], return_val=f"{mem}[{index}]")
    if _arg_dtypes and _gmem_scalar_byte_element(_arg_dtypes[0]) is not None:
        # 1-byte element dtype (bool/int8/uint8): the __ld* overload set has
        # no bool/1-byte entry; reinterpret the typed pointer as const
        # unsigned char* so the cache policy applies generically.  The
        # returned unsigned char converts implicitly back to the element
        # type (bool/int8/uint8), keeping value semantics bit-exact.
        return Builtin(body="", includes=[],
                       return_val=f"{intrin}(reinterpret_cast<const unsigned char*>({mem} + ({index})))")
    return Builtin(body="", includes=[], return_val=f"{intrin}({mem} + ({index}))")


def _eval_return_type_gmem_ld_scalar(mem_dtype, index_dtype, policy=None):
    """Return type: the element dtype of the tensor being loaded from.

    Runs in eval_return_type (type inference), never at string-emission
    time; the policy string itself is validated at codegen by
    _normalize_cache_policy, identical to the gmem v4/x2 families.
    """
    from little_kernel.core.type_system import TensorType
    if mem_dtype is None:
        raise TypeError("gmem_ld_scalar: cannot infer type of first argument (mem)")
    if isinstance(mem_dtype, TensorType):
        return mem_dtype.element_type
    raise TypeError(f"gmem_ld_scalar expects Tensor as first argument, got "
                    f"{type(mem_dtype).__name__}: {mem_dtype}")


@builtin(eval_return_type=_eval_return_type_gmem_ld_scalar, codegen_func=codegen_gmem_ld_scalar)
def gmem_ld_scalar(mem, index, policy=None):
    """Load one element from GMEM at ``mem[index]`` with an optional cache policy.

    policy (compile-time string literal): "wb" (default plain load), "ca",
    "cg", "cs", "lu", "nc" (__ldg), "cv". Bit-exact either way; the policy is
    a pure cache-allocation hint. "nc" carries the CUDA __ldg contract: the
    loaded bytes must not be written by any thread for the duration of the
    launch; the frontend asserts it, the compiler does not infer it.
    """
    raise RuntimeError("should not call gmem_ld_scalar in compilation")


def codegen_gmem_st_scalar(mem, index, value, policy=None, _arg_dtypes=None):
    policy = _normalize_cache_policy(policy, _GMEM_ST_SCALAR_INTRIN.keys(), "wb", "gmem_st_scalar")
    intrin = _GMEM_ST_SCALAR_INTRIN[policy]
    if intrin is None:
        return Builtin(body="", includes=[], return_val=f"{mem}[{index}] = {value}")
    if _arg_dtypes and _gmem_scalar_byte_element(_arg_dtypes[0]) is not None:
        # 1-byte element dtype (bool/int8/uint8): __stcs/__stcg/__stwt have
        # no bool/1-byte overload (NVCC: no instance of overloaded function
        # matches (bool*, bool)), so route the store through an unsigned
        # char* reinterpret with the payload cast to unsigned char.  This is
        # a pure cache-policy spelling legalization; the stored byte is
        # bit-identical.
        return Builtin(
            body="", includes=[],
            return_val=f"{intrin}(reinterpret_cast<unsigned char*>({mem} + ({index})), (unsigned char)({value}))")
    return Builtin(body="", includes=[], return_val=f"{intrin}({mem} + ({index}), {value})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_scalar)
def gmem_st_scalar(mem, index, value, policy=None):
    """Store one element to GMEM at ``mem[index]`` with an optional cache policy.

    policy (compile-time string literal): "wb" (default plain store), "cg",
    "cs" (evict-first, write-once outputs), "wt" (write-through). Bit-exact
    either way; the policy is a pure cache-allocation hint.  On 1-byte
    integer element dtypes (bool/int8/uint8) the policy path is legalized
    through an unsigned-char reinterpret because the CUDA __st* overload
    set has no 1-byte/bool entry; the stored byte is bit-identical.
    """
    raise RuntimeError("should not call gmem_st_scalar in compilation")


def codegen_gmem_ld_nc_u32(gmem_ptr):
    return Builtin(body="", includes=[], return_val=f"__ldg(reinterpret_cast<const uint32_t*>({gmem_ptr}))")


@builtin(eval_return_type=uint32, codegen_func=codegen_gmem_ld_nc_u32)
def gmem_ld_nc_u32(gmem_ptr):
    """Read-only (__ldg) scalar 32-bit load from a byte-addressed global pointer.

    void*-keyed (compose with ptr_byte_offset); returns the raw uint32 bits at
    ptr. Contract: ptr must be 4-byte aligned, in bounds, and the 4 bytes must
    not be written by any thread for the duration of the launch (the CUDA
    __ldg contract). Intended for gathered index/page-table reads; for typed
    element access prefer ldg or plain element loads.
    """
    raise RuntimeError("should not call gmem_ld_nc_u32 in compilation")


def codegen_gmem_ld_cs_u32(gmem_ptr):
    return Builtin(body="", includes=[], return_val=f"__ldcs(reinterpret_cast<const uint32_t*>({gmem_ptr}))")


@builtin(eval_return_type=uint32, codegen_func=codegen_gmem_ld_cs_u32)
def gmem_ld_cs_u32(gmem_ptr):
    """Streaming (evict-first, __ldcs) scalar 32-bit load from a byte-addressed global pointer.

    void*-keyed (compose with ptr_byte_offset); returns the raw uint32 bits at
    ptr. Contract: ptr must be 4-byte aligned, in bounds, and the 4 bytes must
    not be written by any thread for the duration of the launch -- the same
    legality contract as gmem_ld_nc_u32, differing only in the cache policy
    hint: __ldcs (ld.global.cs.b32) marks the line evict-first in L1/L2, for
    read-exactly-once data such as boolean keep-masks in bandwidth-bound
    elementwise kernels. The 128-bit streaming path already exists as
    gmem_ld_cs_v4_u32; this is its scalar sibling.
    """
    raise RuntimeError("should not call gmem_ld_cs_u32 in compilation")


def codegen_gmem_ld_cg_u32(gmem_ptr):
    return Builtin(body="", includes=[], return_val=f"__ldcg(reinterpret_cast<const uint32_t*>({gmem_ptr}))")


@builtin(eval_return_type=uint32, codegen_func=codegen_gmem_ld_cg_u32)
def gmem_ld_cg_u32(gmem_ptr):
    """Cache-global (L2-only, __ldcg) scalar 32-bit load from a byte-addressed global pointer.

    void*-keyed (compose with ptr_byte_offset); returns the raw uint32 bits at
    ptr. Same alignment/bounds/read-only-during-launch contract as
    gmem_ld_nc_u32; __ldcg (ld.global.cg.b32) bypasses L1 and caches only in
    L2, matching the vector gmem_ld_cg_* policy family.
    """
    raise RuntimeError("should not call gmem_ld_cg_u32 in compilation")


def codegen_gmem_ld_u8(gmem_ptr):
    return Builtin(body="", includes=[], return_val=f"(uint32_t)(*reinterpret_cast<const uint8_t*>({gmem_ptr}))")


@builtin(eval_return_type=uint32, codegen_func=codegen_gmem_ld_u8)
def gmem_ld_u8(gmem_ptr):
    """Zero-extending 8-bit load from a byte-addressed global pointer.

    void*-keyed (compose with ptr_byte_offset); returns the byte value
    zero-extended into a uint32 register (ld.global.u8). Intended for
    byte-indexed tables (e.g. uint8 code arrays) where word loads plus
    shifts would otherwise be needed; no alignment requirement beyond the
    byte itself.
    """
    raise RuntimeError("should not call gmem_ld_u8 in compilation")


def codegen_gmem_st_u8(gmem_ptr, value):
    return Builtin(body="", includes=[], return_val=f"*reinterpret_cast<uint8_t*>({gmem_ptr}) = (uint8_t)({value})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_u8)
def gmem_st_u8(gmem_ptr, value):
    """8-bit store to a byte-addressed global pointer.

    void*-keyed (compose with ptr_byte_offset); stores the low 8 bits of
    ``value`` (st.global.u8).  Store twin of :func:`gmem_ld_u8`: a
    gmem_ld_u8 -> gmem_st_u8 pair is a bit-exact one-byte move.  Intended
    for ragged tails and unaligned heads of dtype-agnostic byte moves
    (e.g. the sub-16-byte tail of a bit-exact fp8 index-table gather row,
    where gmem_ld_u8x16/gmem_st_u8x16's 16-byte alignment precondition does
    not hold) — byte stores need no alignment beyond the byte itself and,
    unlike a wider masked store, cannot clobber neighbouring bytes.
    """
    raise RuntimeError("should not call gmem_st_u8 in compilation")


def codegen_gmem_ld_u16(gmem_ptr):
    return Builtin(body="", includes=[], return_val=f"(uint32_t)(*reinterpret_cast<const uint16_t*>({gmem_ptr}))")


@builtin(eval_return_type=uint32, codegen_func=codegen_gmem_ld_u16)
def gmem_ld_u16(gmem_ptr):
    """Zero-extending 16-bit load from a byte-addressed global pointer.

    void*-keyed (compose with ptr_byte_offset); returns the 16-bit value
    zero-extended into a uint32 register (ld.global.u16).  Load twin of
    :func:`gmem_st_u16`: a gmem_ld_u16 -> gmem_st_u16 pair is a bit-exact
    two-byte move.  Legality precondition (frontend-author responsibility,
    same as PTX ld.global.u16 and a CUDA uint16_t dereference): the byte
    address MUST be 2-byte aligned; the compiler does not infer or promote
    alignment.  Intended for dtype-agnostic two-byte moves (e.g. packed
    bf16x2->fp8x2 rope pairs, or the 2-byte granule of a ragged tail) where
    two adjacent u8 accesses would be needed for lack of a typed u16 op.
    """
    raise RuntimeError("should not call gmem_ld_u16 in compilation")


def codegen_gmem_st_u16(gmem_ptr, value):
    return Builtin(body="", includes=[], return_val=f"*reinterpret_cast<uint16_t*>({gmem_ptr}) = (uint16_t)({value})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_u16)
def gmem_st_u16(gmem_ptr, value):
    """16-bit store to a byte-addressed global pointer.

    void*-keyed (compose with ptr_byte_offset); stores the low 16 bits of
    ``value`` (st.global.u16).  Store twin of :func:`gmem_ld_u16`: a
    gmem_ld_u16 -> gmem_st_u16 pair is a bit-exact two-byte move with NO
    numeric conversion.  Legality precondition (frontend-author
    responsibility, same as PTX st.global.u16 and a CUDA uint16_t
    dereference): the byte address MUST be 2-byte aligned and in bounds;
    the compiler does not infer or promote alignment, and passing a bad
    address is a fault identical to dereferencing a misaligned uint16_t*
    in CUDA C.  Intended for aligned 2-byte lanes of packed low-precision
    pairs (e.g. one bf16x2->fp8x2 rope pair per lane) that would otherwise
    require two adjacent gmem_st_u8 stores.
    """
    raise RuntimeError("should not call gmem_st_u16 in compilation")


def codegen_gmem_st_u32(gmem_ptr, value):
    return Builtin(body="", includes=[], return_val=f"*reinterpret_cast<uint32_t*>({gmem_ptr}) = (uint32_t)({value})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_u32)
def gmem_st_u32(gmem_ptr, value):
    """Scalar 32-bit store to a byte-addressed global pointer.

    void*-keyed (compose with ptr_byte_offset); stores the raw uint32 bits
    of ``value`` (st.global.u32).  Store twin of the gmem_ld_*_u32 scalar
    family: a gmem_ld_*_u32 -> gmem_st_u32 pair is a bit-exact 4-byte move
    with NO numeric conversion.  Contract: ptr must be 4-byte aligned and
    in bounds.  Intended for sub-vector tails of dtype-agnostic word moves
    (e.g. the tail words of a gathered int32 scale-factor row when the row
    width is not a multiple of 16 bytes, where gmem_st_v4_u32_ptr's
    16-byte alignment precondition does not hold); without it the only
    fallbacks were the typed element-indexed gmem_st_scalar (unavailable
    when addressing is byte-keyed) or a wider vector store (over-writes).
    """
    raise RuntimeError("should not call gmem_st_u32 in compilation")


def codegen_gmem_st_cg_u32(gmem_ptr, value):
    return Builtin(body="", includes=[],
                   return_val=f"__stcg(reinterpret_cast<uint32_t*>({gmem_ptr}), (uint32_t)({value}))")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_cg_u32)
def gmem_st_cg_u32(gmem_ptr, value):
    """Cache-global (L2-only, __stcg) scalar 32-bit store to a byte-addressed global pointer.

    void*-keyed (compose with ptr_byte_offset); bits stored are identical
    to :func:`gmem_st_u32` — only the cache behavior differs: __stcg
    (st.global.cg.b32) bypasses L1 and caches only in L2, the store-side
    twin of the gmem_ld_cg_u32 load policy.  Same 4-byte alignment/bounds
    contract as gmem_st_u32.
    """
    raise RuntimeError("should not call gmem_st_cg_u32 in compilation")


def codegen_gmem_st_cs_u32(gmem_ptr, value):
    return Builtin(body="", includes=[],
                   return_val=f"__stcs(reinterpret_cast<uint32_t*>({gmem_ptr}), (uint32_t)({value}))")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_cs_u32)
def gmem_st_cs_u32(gmem_ptr, value):
    """Streaming (evict-first, __stcs) scalar 32-bit store to a byte-addressed global pointer.

    void*-keyed (compose with ptr_byte_offset); bits stored are identical
    to :func:`gmem_st_u32` — only the cache behavior differs: __stcs
    (st.global.cs.b32) marks the line evict-first in L1/L2, the store-side
    twin of the gmem_ld_cs_u32 load policy, for write-once destinations
    (e.g. gathered output rows that are stored exactly once).  The 128-bit
    streaming store already exists as gmem_st_cs_v4_u32; this is its scalar
    sibling.  Same 4-byte alignment/bounds contract as gmem_st_u32.
    """
    raise RuntimeError("should not call gmem_st_cs_u32 in compilation")
