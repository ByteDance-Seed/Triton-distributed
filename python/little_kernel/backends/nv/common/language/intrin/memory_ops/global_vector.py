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
Packed/typed vector global-memory access families.

RF10 canonical owner module; see little_kernel.backends.nv.common.language.intrin.contracts
for the shared operand-unit/lane/alignment/cache-policy/effect inventory.
"""

from little_kernel.language.builtin_base import builtin, Builtin, CODEGEN_NEEDS_TYPES_ATTR
from little_kernel.core.type_system import LLType, void, Tensor, int8, uint8, uint16, uint32, uint64, int32, bfloat16, float16, float32, u32x2, u32x4
from little_kernel.core.memory_effects import MemorySpace as MS, r, w
import little_kernel.backends.nv.common.language.intrin.contracts as contracts
from little_kernel.backends.nv.common.language.intrin.memory_ops.scalar_global import gmem_ld_scalar, gmem_st_scalar

_make_gmem_ld_out_lane_type_check = contracts.make_gmem_ld_out_lane_type_check
_make_gmem_st_in_lane_type_check = contracts.make_gmem_st_in_lane_type_check
_normalize_cache_policy = contracts.normalize_cache_policy
_POLICY_ALIASES = contracts.POLICY_ALIASES


def codegen_load_128bit(gmem_ptr):
    """Generate 128-bit vectorized load from global memory."""
    body = ("\n__device__ __forceinline__ uint4 load_128bit_fn(const void* gmem_ptr) {\n"
            "    return *reinterpret_cast<const uint4*>(gmem_ptr);\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"load_128bit_fn({gmem_ptr})")


@builtin(eval_return_type=u32x4, codegen_func=codegen_load_128bit)
def load_128bit(gmem_ptr):
    """Load 128 bits (16 bytes) from global memory as a u32x4 vector value.

    Invariant: ``gmem_ptr`` must be 16-byte aligned; the frontend is
    responsible for proving alignment (e.g., by construction from an aligned
    base plus a multiple-of-4 uint32 offset) and for covering any unaligned
    tail with scalar accesses.
    """
    raise RuntimeError("should not be called in compilation")


def codegen_load_128bit_streaming(gmem_ptr):
    """Generate streaming 128-bit vectorized load (evict-first policy)."""
    body = ("\n__device__ __forceinline__ uint4 load_128bit_streaming_fn(void* gmem_ptr) {\n"
            "    return __ldcs(reinterpret_cast<const uint4*>(gmem_ptr));\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"load_128bit_streaming_fn({gmem_ptr})")


@builtin(eval_return_type=u32x4, codegen_func=codegen_load_128bit_streaming)
def load_128bit_streaming(gmem_ptr):
    """Load 128 bits (16 bytes) from global memory with streaming (evict-first)
    cache policy, as a u32x4 vector value. Same 16-byte alignment invariant as
    :func:`load_128bit`; use for compulsory single-touch traffic."""
    raise RuntimeError("should not be called in compilation")


def codegen_load_128bit_cg(gmem_ptr):
    """Generate L2-only (.cg) 128-bit vectorized load."""
    body = ("\n__device__ __forceinline__ uint4 load_128bit_cg_fn(void* gmem_ptr) {\n"
            "    return __ldcg(reinterpret_cast<const uint4*>(gmem_ptr));\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"load_128bit_cg_fn({gmem_ptr})")


@builtin(eval_return_type=u32x4, codegen_func=codegen_load_128bit_cg)
def load_128bit_cg(gmem_ptr):
    """Load 128 bits (16 bytes) from global memory with the L2-only (.cg)
    cache policy, as a u32x4 vector value. Same 16-byte alignment invariant
    as :func:`load_128bit`; use for read-once streams where polluting L1
    hurts the resident working set."""
    raise RuntimeError("should not be called in compilation")


def codegen_store_128bit_cg(gmem_ptr, data):
    """Generate L2-only (.cg) 128-bit vectorized store."""
    body = ("\n__device__ __forceinline__ void store_128bit_cg_fn(void* gmem_ptr, uint4 data) {\n"
            "    __stcg(reinterpret_cast<uint4*>(gmem_ptr), data);\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"store_128bit_cg_fn({gmem_ptr}, {data})")


@builtin(eval_return_type=void, codegen_func=codegen_store_128bit_cg)
def store_128bit_cg(gmem_ptr, data):
    """Store 128 bits (16 bytes) to global memory with the L2-only (.cg)
    cache policy (bypasses L1). Same operand invariants as
    :func:`store_128bit`."""
    raise RuntimeError("should not be called in compilation")


def codegen_store_128bit(gmem_ptr, data):
    """Generate 128-bit vectorized store to global memory."""
    body = ("\n__device__ __forceinline__ void store_128bit_fn(void* gmem_ptr, uint4 data) {\n"
            "    *reinterpret_cast<uint4*>(gmem_ptr) = data;\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"store_128bit_fn({gmem_ptr}, {data})")


@builtin(eval_return_type=void, codegen_func=codegen_store_128bit)
def store_128bit(gmem_ptr, data):
    """Store 128 bits (16 bytes) to global memory. ``data`` must be a u32x4
    vector value (e.g. from :func:`load_128bit` or :func:`make_u32x4`);
    ``gmem_ptr`` must be 16-byte aligned."""
    raise RuntimeError("should not be called in compilation")


def codegen_store_128bit_streaming(gmem_ptr, data):
    """Generate streaming 128-bit vectorized store (evict-first policy)."""
    body = ("\n__device__ __forceinline__ void store_128bit_streaming_fn(void* gmem_ptr, uint4 data) {\n"
            "    __stcs(reinterpret_cast<uint4*>(gmem_ptr), data);\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"store_128bit_streaming_fn({gmem_ptr}, {data})")


@builtin(eval_return_type=void, codegen_func=codegen_store_128bit_streaming)
def store_128bit_streaming(gmem_ptr, data):
    """Store 128 bits (16 bytes) to global memory with streaming (evict-first)
    cache policy. Same operand invariants as :func:`store_128bit`."""
    raise RuntimeError("should not be called in compilation")


_U32X4_LANE_FIELDS = ("x", "y", "z", "w")


def codegen_vec_u32x4_lane(vec, lane):
    """Extract one statically-selected uint32 lane from a u32x4 vector value."""
    lane_str = str(lane).strip().strip("()").strip()
    if lane_str not in ("0", "1", "2", "3"):
        raise ValueError("vec_u32x4_lane requires a compile-time literal lane in {0,1,2,3} "
                         f"(got {lane!r}); select lanes with unrolled code, not dynamic "
                         "indices, because a u32x4 register value has no addressable storage")
    field = _U32X4_LANE_FIELDS[int(lane_str)]
    return Builtin(body="", includes=[], return_val=f"(({vec}).{field})")


@builtin(eval_return_type=uint32, codegen_func=codegen_vec_u32x4_lane)
def vec_u32x4_lane(vec, lane):
    """Extract lane ``lane`` (a compile-time literal 0..3) of a u32x4 vector
    value as a uint32. Dynamic lane indices are illegal: a vector register
    value cannot be indexed at runtime."""
    raise RuntimeError("should not be called in compilation")


def codegen_make_u32x4(a, b, c, d):
    """Assemble four uint32 lanes into a u32x4 vector value."""
    return Builtin(body="", includes=[],
                   return_val=f"uint4{{(uint32_t)({a}), (uint32_t)({b}), (uint32_t)({c}), (uint32_t)({d})}}")


@builtin(eval_return_type=u32x4, codegen_func=codegen_make_u32x4)
def make_u32x4(a, b, c, d):
    """Assemble four uint32 values (lane order x, y, z, w) into a u32x4 vector
    value suitable for :func:`store_128bit`."""
    raise RuntimeError("should not be called in compilation")


def codegen_load_64bit(gmem_ptr):
    """Generate 64-bit vectorized load from global memory."""
    body = ("\n__device__ __forceinline__ uint2 load_64bit_fn(void* gmem_ptr) {\n"
            "    return *reinterpret_cast<uint2*>(gmem_ptr);\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"load_64bit_fn({gmem_ptr})")


@builtin(eval_return_type=u32x2, codegen_func=codegen_load_64bit)
def load_64bit(gmem_ptr):
    """Load 64 bits (8 bytes) from global memory as a u32x2 vector value.

    Invariant: ``gmem_ptr`` must be 8-byte aligned (e.g., 4 packed bf16/fp16
    lanes from an aligned base).
    """
    raise RuntimeError("should not be called in compilation")


def codegen_load_64bit_streaming(gmem_ptr):
    """Generate streaming 64-bit vectorized load (evict-first policy)."""
    body = ("\n__device__ __forceinline__ uint2 load_64bit_streaming_fn(void* gmem_ptr) {\n"
            "    return __ldcs(reinterpret_cast<const uint2*>(gmem_ptr));\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"load_64bit_streaming_fn({gmem_ptr})")


@builtin(eval_return_type=u32x2, codegen_func=codegen_load_64bit_streaming)
def load_64bit_streaming(gmem_ptr):
    """Load 64 bits (8 bytes) from global memory with streaming (evict-first)
    cache policy, as a u32x2 vector value. Same 8-byte alignment invariant as
    :func:`load_64bit`; use for compulsory single-touch traffic on 2-byte
    dtypes (bf16x4/fp16x4 per instruction)."""
    raise RuntimeError("should not be called in compilation")


def codegen_load_64bit_cg(gmem_ptr):
    """Generate L2-only (.cg) 64-bit vectorized load."""
    body = ("\n__device__ __forceinline__ uint2 load_64bit_cg_fn(void* gmem_ptr) {\n"
            "    return __ldcg(reinterpret_cast<const uint2*>(gmem_ptr));\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"load_64bit_cg_fn({gmem_ptr})")


@builtin(eval_return_type=u32x2, codegen_func=codegen_load_64bit_cg)
def load_64bit_cg(gmem_ptr):
    """Load 64 bits (8 bytes) from global memory with the L2-only (.cg)
    cache policy, as a u32x2 vector value. Same 8-byte alignment invariant
    as :func:`load_64bit`."""
    raise RuntimeError("should not be called in compilation")


def codegen_load_64bit_nc(gmem_ptr):
    """Generate read-only (__ldg, .nc) 64-bit vectorized load."""
    body = ("\n__device__ __forceinline__ uint2 load_64bit_nc_fn(void* gmem_ptr) {\n"
            "    return __ldg(reinterpret_cast<const uint2*>(gmem_ptr));\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"load_64bit_nc_fn({gmem_ptr})")


@builtin(eval_return_type=u32x2, codegen_func=codegen_load_64bit_nc)
def load_64bit_nc(gmem_ptr):
    """Load 64 bits (8 bytes) from global memory through the non-coherent
    read-only path (__ldg, ld.global.nc.v2.b32), as a u32x2 vector value.

    Same pointer contract as :func:`load_64bit` (generic pointer, 8-byte
    alignment invariant). Legal only when the addressed storage is not
    written by this kernel through any aliasing view for the duration of
    the load's use: the non-coherent read-only path may read a stale
    L1-resident copy otherwise. Bit-level result equals a plain load of
    the same address."""
    raise RuntimeError("should not be called in compilation")


def codegen_store_64bit_cg(gmem_ptr, data):
    """Generate L2-only (.cg) 64-bit vectorized store."""
    body = ("\n__device__ __forceinline__ void store_64bit_cg_fn(void* gmem_ptr, uint2 data) {\n"
            "    __stcg(reinterpret_cast<uint2*>(gmem_ptr), data);\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"store_64bit_cg_fn({gmem_ptr}, {data})")


@builtin(eval_return_type=void, codegen_func=codegen_store_64bit_cg)
def store_64bit_cg(gmem_ptr, data):
    """Store 64 bits (8 bytes) to global memory with the L2-only (.cg)
    cache policy (bypasses L1). Same operand invariants as
    :func:`store_64bit`."""
    raise RuntimeError("should not be called in compilation")


def codegen_store_64bit(gmem_ptr, data):
    """Generate 64-bit vectorized store to global memory."""
    body = ("\n__device__ __forceinline__ void store_64bit_fn(void* gmem_ptr, uint2 data) {\n"
            "    *reinterpret_cast<uint2*>(gmem_ptr) = data;\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"store_64bit_fn({gmem_ptr}, {data})")


@builtin(eval_return_type=void, codegen_func=codegen_store_64bit)
def store_64bit(gmem_ptr, data):
    """Store 64 bits (8 bytes) to global memory. ``data`` must be a u32x2
    vector value; ``gmem_ptr`` must be 8-byte aligned."""
    raise RuntimeError("should not be called in compilation")


def codegen_store_64bit_streaming(gmem_ptr, data):
    """Generate streaming 64-bit vectorized store (evict-first policy)."""
    body = ("\n__device__ __forceinline__ void store_64bit_streaming_fn(void* gmem_ptr, uint2 data) {\n"
            "    __stcs(reinterpret_cast<uint2*>(gmem_ptr), data);\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"store_64bit_streaming_fn({gmem_ptr}, {data})")


@builtin(eval_return_type=void, codegen_func=codegen_store_64bit_streaming)
def store_64bit_streaming(gmem_ptr, data):
    """Store 64 bits (8 bytes) to global memory with streaming (evict-first)
    cache policy. Same operand invariants as :func:`store_64bit`."""
    raise RuntimeError("should not be called in compilation")


def codegen_vec_u32x2_lane(vec, lane):
    """Extract one statically-selected uint32 lane from a u32x2 vector value."""
    lane_str = str(lane).strip().strip("()").strip()
    if lane_str not in ("0", "1"):
        raise ValueError("vec_u32x2_lane requires a compile-time literal lane in {0,1} "
                         f"(got {lane!r}); select lanes with unrolled code, not dynamic "
                         "indices, because a u32x2 register value has no addressable storage")
    field = ("x", "y")[int(lane_str)]
    return Builtin(body="", includes=[], return_val=f"(({vec}).{field})")


@builtin(eval_return_type=uint32, codegen_func=codegen_vec_u32x2_lane)
def vec_u32x2_lane(vec, lane):
    """Extract lane ``lane`` (a compile-time literal 0 or 1) of a u32x2 vector
    value as a uint32. Dynamic lane indices are illegal."""
    raise RuntimeError("should not be called in compilation")


def codegen_make_u32x2(a, b):
    """Assemble two uint32 lanes into a u32x2 vector value."""
    return Builtin(body="", includes=[], return_val=f"uint2{{(uint32_t)({a}), (uint32_t)({b})}}")


@builtin(eval_return_type=u32x2, codegen_func=codegen_make_u32x2)
def make_u32x2(a, b):
    """Assemble two uint32 values (lane order x, y) into a u32x2 vector value
    suitable for :func:`store_64bit`."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_f32x4(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_f32x4_fn(
        const void* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = *reinterpret_cast<const uint4*>(ptr);
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_f32x4_fn({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_f32x4", uint32),
         codegen_func=codegen_gmem_ld_f32x4, effects=(r(MS.GMEM, "mem"), ))
def gmem_ld_f32x4(mem, index, r0, r1, r2, r3):
    """Load one 16-byte vector from GMEM as four raw uint32 lanes.

    Bit-level and dtype-agnostic: ``mem + index`` is element-typed pointer
    arithmetic, so any tensor dtype may be passed; the address must be
    16-byte aligned (`index` a multiple of `16 / sizeof(element)`).
    """
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# 128-bit global vector access with explicit L1/L2 cache-policy control
# ==============================================================================
#
# Generic contract (no task or shape assumptions):
#   * ``mem``          -- any LittleKernel Tensor (global scope); reinterpreted
#                         as raw bytes, then offset by ``elem_offset`` elements
#                         of the tensor's own dtype (C++ pointer arithmetic).
#   * ``elem_offset``  -- element offset; (base_address + elem_offset *
#                         sizeof(dtype)) MUST be 16-byte aligned. This is a
#                         legality precondition of PTX ``ld/st.global.v4``;
#                         violating it is a fault (misaligned address), the
#                         same precondition CUDA ``uint4`` dereference has.
#   * ``out``/``vals`` -- a local-scope uint32 array of exactly 4 elements,
#                         e.g. ``ll.empty([4], dtype=ll.uint32, scope="local")``.
#                         The four 32-bit lanes map to bytes [0:16) in order.
#   * Cache policy mirroring CUDA __ldcs/__ldcg/__stcs:
#       - default  : ld.global.v4.u32 / st.global.v4.u32 (.ca/.wb behavior)
#       - "cs"     : evict-first streaming; use for data read/written once
#       - "cg"     : cache at L2 only, bypass L1 (loads only)
#     Bit-level results are identical to plain loads/stores; the policy only
#     changes cache allocation, so any policy is *numerically* legal for any
#     program and the choice is purely a performance hint.
#
# Usage (frontend):
#     frag = ll.empty([4], dtype=ll.uint32, scope="local")
#     ll.ld_global_cs_v4_u32(qkv, head * 128 + tid * 8, frag)   # bf16: 8 elems
#     ll.st_global_cs_v4_u32(qkv, head * 128 + tid * 8, frag)


def _gmem_vec_body(inst, is_load, lanes):
    """Emit one inline-asm helper for a lanes*4B global access with cache policy."""
    name = f"gmem_{inst.replace('.', '_')}_fn"
    regs = ", ".join(f"%{i}" for i in range(lanes))
    if is_load:
        body = ("\n__device__ __forceinline__ void " + name + f"(const void* ptr, uint32_t (&out)[{lanes}]) {{\n"
                f'    asm volatile("{inst} {{{regs}}}, [%{lanes}];"\n'
                '                 : ' + ", ".join(f'"=r"(out[{i}])' for i in range(lanes)) + '\n'
                '                 : "l"(ptr) : "memory");\n'
                "}\n")
    else:
        body = ("\n__device__ __forceinline__ void " + name + f"(void* ptr, const uint32_t (&vals)[{lanes}]) {{\n"
                f'    asm volatile("{inst} [%0], {{{", ".join(f"%{i+1}" for i in range(lanes))}}};"\n'
                '                 :: "l"(ptr), ' + ", ".join(f'"r"(vals[{i}])' for i in range(lanes)) + '\n'
                '                 : "memory");\n'
                "}\n")
    return name, body


def _gmem_v4_body(inst, is_load):
    """Emit one inline-asm helper for a 16B global access with cache policy."""
    return _gmem_vec_body(inst, is_load, 4)


def _make_gmem_vec(inst, is_load, lanes):

    def codegen(mem, elem_offset, words):
        name, body = _gmem_vec_body(inst, is_load, lanes)
        return Builtin(body=body, includes=["<cstdint>"], return_val=f"{name}({mem} + {elem_offset}, {words})")

    return codegen


def _make_gmem_v4(inst, is_load):
    return _make_gmem_vec(inst, is_load, 4)


@builtin(eval_return_type=void, codegen_func=_make_gmem_v4("ld.global.v4.u32", True))
def ld_global_v4_u32(mem, elem_offset, out):
    """16-byte aligned global load into out[0..4) (uint32), default cache."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_gmem_v4("ld.global.cs.v4.u32", True))
def ld_global_cs_v4_u32(mem, elem_offset, out):
    """16-byte aligned global load into out[0..4); evict-first streaming (.cs)."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_gmem_v4("ld.global.cg.v4.u32", True))
def ld_global_cg_v4_u32(mem, elem_offset, out):
    """16-byte aligned global load into out[0..4); L2-only, bypass L1 (.cg)."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_gmem_v4("ld.global.nc.v4.u32", True))
def ld_global_nc_v4_u32(mem, elem_offset, out):
    """16-byte aligned read-only global load into out[0..4); .nc (__ldg path).

    Legal only when mem refers to storage that is not written by this kernel
    through any aliasing view for the duration of the load's use (the PTX
    non-coherent/read-only path may read a stale L1-resident copy otherwise).
    Bit-level result equals a plain load of the same address."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_gmem_vec("ld.global.nc.v2.u32", True, 2))
def ld_global_nc_v2_u32(mem, elem_offset, out):
    """8-byte aligned read-only global load into out[0..2); .nc (__ldg path).

    Same read-only legality precondition as ld_global_nc_v4_u32; use for
    naturally aligned 8-byte pairs (e.g. two adjacent f32/u32 elements)."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_gmem_vec("st.global.v2.u32", False, 2))
def st_global_v2_u32(mem, elem_offset, vals):
    """8-byte aligned global store of vals[0..2) (uint32), default cache."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_gmem_vec("st.global.cs.v2.u32", False, 2))
def st_global_cs_v2_u32(mem, elem_offset, vals):
    """8-byte aligned global store of vals[0..2); evict-first streaming (.cs)."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_gmem_v4("ld.global.lu.v4.u32", True))
def ld_global_lu_v4_u32(mem, elem_offset, out):
    """16-byte aligned global load into out[0..4); last-use hint (.lu).

    Same signature and 16-byte alignment precondition as ld_global_v4_u32;
    the .lu cache policy marks the line evict-first after this use, for
    streams that are read exactly once and never reused."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_gmem_v4("st.global.v4.u32", False))
def st_global_v4_u32(mem, elem_offset, vals):
    """16-byte aligned global store of vals[0..4) (uint32), default cache."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_gmem_v4("st.global.cs.v4.u32", False))
def st_global_cs_v4_u32(mem, elem_offset, vals):
    """16-byte aligned global store of vals[0..4); evict-first streaming (.cs)."""
    raise RuntimeError("should not be called in compilation")


def _prefetch_l2_codegen(mem, elem_offset):
    """Emit one inline-asm helper for an L2 prefetch of one 128B line."""
    body = ("\n__device__ __forceinline__ void prefetch_l2_fn(const void* ptr) {\n"
            '    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));\n'
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"prefetch_l2_fn({mem} + {elem_offset})")


@builtin(eval_return_type=void, codegen_func=_prefetch_l2_codegen)
def prefetch_l2(mem, elem_offset):
    """Prefetch the 128-byte line holding mem[elem_offset] into L2.

    Offsets are in elements of mem's own dtype (C++ pointer arithmetic),
    exactly like the ld/st_global_v4_u32 family. No architectural effect;
    purely a latency retirement hint, so it is always safe to issue."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Typed (dtype-preserving) 16-byte global vector loads into a typed local array
# ==============================================================================
#
# Companion to the uint32-container ``ld_global_v4_u32`` family above.  That
# family always delivers raw uint32 bit patterns, which discards the element
# type of the destination: a program that loads FLOAT elements (needing .f32
# arithmetic downstream) into a ``float[4]`` local, or SIGNED-INT elements
# (needing .s32 semantics downstream, e.g. a later cvt.rn.f32.s32) into an
# ``int32[4]`` local, cannot express the load as one typed vector op and must
# hand-roll a ``simple_builtin`` asm block per dtype.  These two builtins emit
# the SAME single 16-byte transaction (ld.global.v4.b32) but bind the four
# 32-bit lanes to registers of the matching class (``.f32`` / integer), so the
# loaded values keep the destination array's element type all the way through
# type inference and codegen.  Generic: no task, shape, or layout assumption.
#
# Legality contract (identical to ld_global_v4_u32 / PTX ld.global.v4):
#   * ``mem``         -- any global Tensor of a 32-bit element dtype
#                        (float32 / int32 / uint32).  Reinterpreted as bytes and
#                        offset by ``elem_offset`` ELEMENTS of its own dtype
#                        (C++ pointer arithmetic), exactly like the u32 family.
#   * ``elem_offset`` -- element offset; the byte address
#                        (base + elem_offset*4) MUST be 16-byte aligned.  This is
#                        a hard PTX legality precondition (a misaligned or
#                        ragged-tail address is a fault); handle unaligned heads
#                        and ragged tails on a separate scalar path.
#   * ``out``         -- a local-scope typed array of exactly 4 elements, e.g.
#                        ``ll.empty([4], dtype=ll.float32, scope="local")`` for
#                        the f32 form or ``dtype=ll.int32`` for the s32 form;
#                        its element dtype MUST match the builtin used.  Lane k
#                        (little-endian bytes [4k,4k+4) of the 16-byte vector)
#                        lands in ``out[k]``.  The helpers are #ifndef-guarded
#                        so a kernel may call them at many sites safely.
# Lanes stay 4 (one 16-byte vector) regardless of dtype; 16-bit and 8-bit
# element tensors keep using the uint32-container family plus the pair codecs,
# exactly as the shipped frontends already do.
#
# Usage (frontend, replacing a hand-written simple_builtin):
#     wv = ll.empty([4], dtype=ll.float32, scope="local")
#     ll.ld_global_v4_f32(sw, c, wv)        # 4 f32 = one 16 B vector
#     va = ll.empty([4], dtype=ll.int32, scope="local")
#     ll.ld_global_v4_s32(acc, base, va)    # 4 s32 = one 16 B vector


def codegen_ld_global_v4_f32(mem, elem_offset, out):
    body = """
#ifndef LK_LD_GLOBAL_V4_F32_FN_DEFINED
#define LK_LD_GLOBAL_V4_F32_FN_DEFINED
__device__ __forceinline__ void ld_global_v4_f32_fn(const void* ptr, float (&out)[4]) {
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3])
                 : "l"(ptr) : "memory");
}
#endif
"""
    return Builtin(body=body, includes=["<cstdint>"], return_val=f"ld_global_v4_f32_fn({mem} + {elem_offset}, {out})")


@builtin(eval_return_type=void, codegen_func=codegen_ld_global_v4_f32)
def ld_global_v4_f32(mem, elem_offset, out):
    """16-byte aligned global load of 4 float32 lanes into ``out[0..4)``.

    Emits one ``ld.global.v4.b32`` whose four lanes land in a local
    ``float[4]`` array, preserving the float element type (default cache
    policy, plain write-back load).  ``elem_offset`` is in float-sized
    elements; see the module contract above for the 16-byte alignment and
    in-bounds legality preconditions and the scalar fallback for ragged tails.
    """
    raise RuntimeError("should not be called in compilation")


def codegen_ld_global_v4_s32(mem, elem_offset, out):
    body = """
#ifndef LK_LD_GLOBAL_V4_S32_FN_DEFINED
#define LK_LD_GLOBAL_V4_S32_FN_DEFINED
__device__ __forceinline__ void ld_global_v4_s32_fn(const void* ptr, int32_t (&out)[4]) {
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "l"(ptr) : "memory");
}
#endif
"""
    return Builtin(body=body, includes=["<cstdint>"], return_val=f"ld_global_v4_s32_fn({mem} + {elem_offset}, {out})")


@builtin(eval_return_type=void, codegen_func=codegen_ld_global_v4_s32)
def ld_global_v4_s32(mem, elem_offset, out):
    """16-byte aligned global load of 4 int32 lanes into ``out[0..4)``.

    Emits one ``ld.global.v4.b32`` whose four lanes land in a local
    ``int32[4]`` array, preserving the signed-int element type so downstream
    ops keep .s32 semantics (e.g. signed-to-float conversion).  Same offset
    units, alignment, and in-bounds legality contract as the f32 form.
    """
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Typed (dtype-preserving) 16-byte global vector store from four scalar words
# ==============================================================================
#
# Store-side completion of the typed v4 family above: the uint32-container
# ``st_global_v4_u32`` takes its payload as a local ``uint32[4]`` array, and
# ``gmem_st_v4`` lowers through a C++ ``uint4`` reinterpret; frontends whose
# packed words live in SCALAR registers (e.g. ``pack_bf16_float`` -> uint32)
# had to round-trip through a local array or hand-roll a per-frontend
# ``simple_builtin`` asm block.  This builtin emits the SAME single 16-byte
# transaction (st.global.v4.b32) directly from four scalar uint32 words while
# preserving the destination tensor's element type for pointer arithmetic:
# the helper is ``template <typename T>``, so ``elem_offset`` is in elements
# of ``mem``'s own dtype (C++ pointer arithmetic) for any tensor dtype —
# bfloat16 being the motivating use (8 bf16 elements per 16-byte vector);
# f16/f32/s32/f64 tensors are served identically.  Generic: no task, shape,
# or layout assumption.
#
# Legality contract (identical to st_global_v4_u32 / PTX st.global.v4):
#   * ``mem``         -- any global Tensor; reinterpreted as bytes and offset
#                        by ``elem_offset`` ELEMENTS of its own dtype.
#   * ``elem_offset`` -- element offset; the byte address
#                        (base + elem_offset * sizeof(dtype)) MUST be 16-byte
#                        aligned.  This is a hard PTX legality precondition
#                        (a misaligned or ragged-tail address is a fault);
#                        handle unaligned heads and ragged tails on a
#                        separate scalar path.
#   * ``w0..w3``      -- four uint32 SCALARS holding the four 32-bit lanes in
#                        little-endian memory order (e.g. packed bf16x2 pairs
#                        produced by ``ll.pack_bf16_float``); lane k lands at
#                        bytes [4k, 4k+4) of the 16-byte vector.  The helper is
#                        #ifndef-guarded so many call sites emit one definition.
#
# Usage (frontend, replacing a hand-written simple_builtin):
#     p01: ll.uint32 = ll.pack_bf16_float(a0, a1)
#     ...
#     ll.st_global_v4_bf16(y, base, p01, p23, p45, p67)   # 8 bf16 = one 16 B store


def codegen_st_global_v4_bf16(mem, elem_offset, w0, w1, w2, w3):
    body = """
#ifndef LK_ST_GLOBAL_V4_BF16_FN_DEFINED
#define LK_ST_GLOBAL_V4_BF16_FN_DEFINED
template <typename T>
__device__ __forceinline__ void st_global_v4_bf16_fn(
        T* ptr, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(w0), "r"(w1), "r"(w2), "r"(w3)
                 : "memory");
}
#endif
"""
    return Builtin(body=body, includes=["<cstdint>"], return_val=f"st_global_v4_bf16_fn({mem} + {elem_offset},"
                   f" {w0}, {w1}, {w2}, {w3})")


@builtin(eval_return_type=void, codegen_func=codegen_st_global_v4_bf16)
def st_global_v4_bf16(mem, elem_offset, w0, w1, w2, w3):
    """16-byte aligned global store of four uint32 scalar words, dtype-preserving.

    Emits one ``st.global.v4.b32`` (default cache policy, plain write-back
    store) of the raw little-endian bit patterns ``w0..w3`` at
    ``mem + elem_offset``, where ``elem_offset`` is in elements of ``mem``'s
    own dtype.  See the module contract above for the 16-byte alignment and
    in-bounds legality preconditions and the scalar fallback for ragged
    tails.  For a cache-policy-controlled variant or an array payload use
    ``gmem_st_v4`` / the ``st_global_*_v4_u32`` family.
    """
    raise RuntimeError("should not be called in compilation")


# Scalar global stores with cache policy + F32->F16-family RNE conversion
# ==============================================================================
#
# Generic contract (same element-pointer + element-offset convention as the
# v4 family above):
#   * ``mem``          -- a LittleKernel global Tensor whose element dtype is
#                         bf16; ``elem_offset`` is in elements of that dtype.
#   * ``value``        -- an f32 value, rounded ONCE to the destination dtype
#                         with round-to-nearest-even (``__float2bfloat16_rn``,
#                         identical semantics to ``ll.float_to_bf16``).
#   * Cache policy ``.cs`` (evict-first streaming, CUDA ``__stcs``): bit-level
#     result is identical to a plain store; the policy only changes cache
#     allocation, so it is numerically legal for any program.
#   * Legality: the address is 2-byte aligned by construction (bf16 element);
#     ``elem_offset`` must be in bounds.
#
# This is the scalar (single-element) member of the cache-policy store family;
# use it when the store stream is not 16-byte vectorizable (e.g. one element
# per lane group written from a single lane).


def codegen_st_global_cs_bf16_f32_rn(mem, elem_offset, value):
    body = """
__device__ __forceinline__ void st_global_cs_bf16_f32_rn_fn(__nv_bfloat16* p, float v) {
    asm volatile("st.global.cs.u16 [%0], %1;"
                 :
                 : "l"(p), "h"(__bfloat16_as_ushort(__float2bfloat16_rn(v)))
                 : "memory");
}
"""
    return Builtin(body=body, includes=["<cuda_bf16.h>"],
                   return_val=f"st_global_cs_bf16_f32_rn_fn(({mem}) + ({elem_offset}), {value})")


@builtin(eval_return_type=void, codegen_func=codegen_st_global_cs_bf16_f32_rn)
def st_global_cs_bf16_f32_rn(mem, elem_offset, value):
    """Store one f32 value to bf16 tensor[elem_offset] (RNE), evict-first (.cs)."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Global prefetch hints (L1 / L2)
# ==============================================================================
#
# Generic contract: ``prefetch.global.L1/L2 [addr]`` brings the 128-byte line
# containing ``addr`` into the named cache level. It issues no data movement
# into registers, has no alignment requirement, and is numerically inert --
# a pure latency hint for serial dependency chains.
# ``mem`` may be any global tensor; ``elem_offset`` is in elements of its own
# dtype (same pointer-arithmetic convention as the v4 family).


def _make_prefetch_global(level):

    def codegen(mem, elem_offset):
        name = f"prefetch_global_{level.lower()}_fn"
        body = ("\n__device__ __forceinline__ void " + name + "(const void* p) {\n"
                f'    asm volatile("prefetch.global.{level} [%0];" :: "l"(p) : "memory");\n'
                "}\n")
        return Builtin(body=body, includes=[], return_val=f"{name}((const void*)(({mem}) + ({elem_offset})))")

    return codegen


@builtin(eval_return_type=void, codegen_func=_make_prefetch_global("L1"))
def prefetch_global_l1(mem, elem_offset):
    """Prefetch the line at mem + elem_offset elements into L1 (latency hint)."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_prefetch_global("L2"))
def prefetch_global_l2(mem, elem_offset):
    """Prefetch the line at mem + elem_offset elements into L2 (latency hint)."""


# ==============================================================================
# 64-bit global access: v2.u32 register lanes and scalar u64
# ==============================================================================
#
# Generic contract (no task or shape assumptions): exactly the v4 family's
# contract narrowed to 8 bytes.
#   * ``mem``          -- any LittleKernel Tensor (global scope); offset by
#                         ``elem_offset`` elements of the tensor's own dtype.
#   * ``elem_offset``  -- element offset; (base_address + elem_offset *
#                         sizeof(dtype)) MUST be 8-byte aligned. This is a
#                         legality precondition of PTX ``ld/st.global.v2.u32``
#                         and of scalar 64-bit ``ld/st.global.u64``; violating
#                         it is a fault (misaligned address), the same
#                         precondition CUDA ``uint2``/``uint64_t`` dereference
#                         has. Existence proof in the pinned sources:
#                         flashinfer @ 5f8c9efd7feedaae580f2cc02733555de50f920f,
#                         include/flashinfer/comm/trtllm_mnnvl_allreduce.cuh
#                         L436 (ld.volatile.global.v2.u32).
#   * ``out``/``vals`` -- a local-scope uint32 array of exactly 2 elements;
#                         the two 32-bit lanes map to bytes [0:8) in order.
#   * Cache policy mirrors the v4 family: default (.ca/.wb), "cs" evict-first
#     streaming, "cg" L2-only (loads only). Policies are bit-exact; the
#     choice is purely a performance hint.
#   * Scalar 8-byte (u64) accesses are NOT duplicated here: default-policy
#     ``ld_global_u64`` / ``st_global_u64`` already exist in
#     intrin/cuda_asm.py; the evict-first ``st_global_cs_u64`` mirror lives
#     next to them there.
#
# Usage (frontend):
#     frag = ll.empty([2], dtype=ll.uint32, scope="local")
#     ll.ld_global_cs_v2_u32(x, off, frag)          # 8 bytes
#     ll.st_global_cs_v2_u32(y, off, frag)


def _gmem_v2_body(inst, is_load):
    """Emit one inline-asm helper for an 8B global access with cache policy."""
    name = f"gmem_{inst.replace('.', '_')}_fn"
    if is_load:
        body = ("\n__device__ __forceinline__ void " + name + "(const void* ptr, uint32_t (&out)[2]) {\n"
                f'    asm volatile("{inst} {{%0, %1}}, [%2];"\n'
                '                 : "=r"(out[0]), "=r"(out[1])\n'
                '                 : "l"(ptr) : "memory");\n'
                "}\n")
    else:
        body = ("\n__device__ __forceinline__ void " + name + "(void* ptr, const uint32_t (&vals)[2]) {\n"
                f'    asm volatile("{inst} [%0], {{%1, %2}};"\n'
                '                 :: "l"(ptr), "r"(vals[0]), "r"(vals[1])\n'
                '                 : "memory");\n'
                "}\n")
    return name, body


def _make_gmem_v2(inst, is_load):

    def codegen(mem, elem_offset, words):
        name, body = _gmem_v2_body(inst, is_load)
        return Builtin(body=body, includes=["<cstdint>"], return_val=f"{name}({mem} + {elem_offset}, {words})")

    return codegen


@builtin(eval_return_type=void, codegen_func=_make_gmem_v2("ld.global.v2.u32", True))
def ld_global_v2_u32(mem, elem_offset, out):
    """8-byte aligned global load into out[0..2) (uint32), default cache."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_gmem_v2("ld.global.cs.v2.u32", True))
def ld_global_cs_v2_u32(mem, elem_offset, out):
    """8-byte aligned global load into out[0..2); evict-first streaming (.cs)."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_gmem_v2("ld.global.cg.v2.u32", True))
def ld_global_cg_v2_u32(mem, elem_offset, out):
    """8-byte aligned global load into out[0..2); L2-only, bypass L1 (.cg)."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Packed BF16x2 bit-container -> FP32 widening (companion to the v4 family)
# ==============================================================================
#
# Generic companion conversion for words produced by ld_global_v4_u32-style
# bit-container loads on BF16 tensors: unpacks one packed __nv_bfloat162
# (little-endian: lo half = bits [0,16), hi half = bits [16,32)) into two
# exact FP32 values.  Widening BF16 -> FP32 is bit-exact, matching
# ll.bf16_to_float; this is the vector (2-way) form.  Pure register op, no
# memory access, no rounding.


def codegen_gmem_st_f32x4(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_f32x4_fn(
        void* ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    *reinterpret_cast<uint4*>(ptr) = uint4{r0, r1, r2, r3};
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_f32x4_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_f32x4", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_st_f32x4)
def gmem_st_f32x4(mem, index, r0, r1, r2, r3):
    """Store four raw uint32 lanes as one 16-byte vector to GMEM (bit-level, dtype-agnostic)."""
    raise RuntimeError("should not be called in compilation")


# Naming/lane contract of the packed-vs-numerical 16-byte families.  Each
# whole policy family (plain name and its .cg/.streaming twins) shares ONE
# signature, so a plain name and its cache-policy twins always agree:
#   * plain gmem_ld_{f16x8,bf16x8} = uint64-pair bit-pattern load matching
#     gmem_ld_{f16x8,bf16x8}_{cg,streaming} (pair with vec_*_lane);
#   * plain gmem_st_{f16x8,bf16x8} = RNE-pack 8 x FP32 store matching
#     gmem_st_{f16x8,bf16x8}_{cg,streaming};
#   * widening 8 x FP32-out loads use the explicit *_f32x8 suffix
#     (gmem_ld_{f16x8,bf16x8}_f32x8), and the sibling pack stores use the
#     gmem_st_f32_to_* spellings;
#   * raw 4 x uint32 bit-pattern ops keep the explicit *_u32x4 suffix.
# Every emitted device helper name is unique per public name, so any subset
# of the variants can coexist in one translation unit.
# ==============================================================================
# Typed 128-bit vectorized global memory access family
# ==============================================================================
#
# One 16-byte transaction per call, with typed element lanes exchanged through
# the caller's local registers (the callee writes through ``&arg`` at the call
# site, so the loaded values appear in the caller's own C++ locals).
#
# Legality contract (checked by the frontend author, not the compiler):
# ``mem + index`` is element-typed C++ pointer arithmetic, so ``index`` counts
# elements of the tensor dtype and must be a multiple of ``16 /
# sizeof(element)`` (4 for 32-bit dtypes, 8 for 16-bit dtypes) so the address
# is 16-byte aligned.  Frontends handle non-divisible tails with the usual
# scalar guarded path.  The ``_nc`` load variants route through the read-only
# data cache (``__ldg``) and require the loaded region to be read-only for the
# lifetime of the kernel launch.


def codegen_gmem_ld_v4f32(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_v4f32_fn(
        const void* ptr, float* r0, float* r1, float* r2, float* r3) {
    const float4 value = *reinterpret_cast<const float4*>(ptr);
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_v4f32_fn({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_v4f32", float32),
         codegen_func=codegen_gmem_ld_v4f32)
def gmem_ld_v4f32(mem, index, r0, r1, r2, r3):
    """Load one 16-byte vector as four FP32 lanes from GMEM into float locals."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_v4f32_nc(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_v4f32_nc_fn(
        const void* ptr, float* r0, float* r1, float* r2, float* r3) {
    const float4 value = __ldg(reinterpret_cast<const float4*>(ptr));
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_v4f32_nc_fn({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_v4f32_nc", float32),
         codegen_func=codegen_gmem_ld_v4f32_nc)
def gmem_ld_v4f32_nc(mem, index, r0, r1, r2, r3):
    """__ldg (read-only cache) variant of gmem_ld_v4f32."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_v4f32(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_v4f32_fn(
        void* ptr, float r0, float r1, float r2, float r3) {
    *reinterpret_cast<float4*>(ptr) = make_float4(r0, r1, r2, r3);
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_v4f32_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_v4f32", float32), eval_return_type=void,
         codegen_func=codegen_gmem_st_v4f32)
def gmem_st_v4f32(mem, index, r0, r1, r2, r3):
    """Store four FP32 lanes as one 16-byte vector to GMEM."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_v4f32_cg(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_v4f32_cg_fn(
        void* ptr, float r0, float r1, float r2, float r3) {
    __stcg(reinterpret_cast<float4*>(ptr), make_float4(r0, r1, r2, r3));
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_v4f32_cg_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_v4f32_cg", float32), eval_return_type=void,
         codegen_func=codegen_gmem_st_v4f32_cg)
def gmem_st_v4f32_cg(mem, index, r0, r1, r2, r3):
    """Cache-global (.cg, L2-only) variant of gmem_st_v4f32; value-identical, eviction hint only."""
    raise RuntimeError("should not be called in compilation")


def _gmem_v4s32_mem_element_lltype(_arg_dtypes):
    """Element LLType of the ``mem`` argument from the typed-codegen channel.

    The emitter hands each positional argument's declared type; a tensor
    operand arrives wrapped as ``Tensor[element]`` and is unwrapped here.
    Returns None when the channel is unavailable (direct codegen-func calls
    in unit tests, older emitter paths) or the operand is not statically
    typed, so callers fall back to the historical word-lane spelling.
    """
    if not _arg_dtypes:
        return None
    ll_type = _arg_dtypes[0]
    inner = getattr(ll_type, "element_type", None)
    if inner is not None:
        return inner
    return ll_type


def _gmem_ld_v4s32_body(helper_name, ldg, byte_lltype):
    """Build the 16-byte vector-load helper for gmem_ld_v4s32*.

    ``byte_lltype`` selects lane semantics for byte-width (int8/uint8)
    source streams:

    * int8  -> every output lane is one sign-extended int8 element
      (``r_i = (int32)(int8)mem[index+i]``);
    * uint8 -> every output lane is one zero-extended uint8 element;
    * anything else (int32 streams, unknown type) -> the historical
      spelling: lanes are the four raw int32 words of the 16-byte
      transaction.

    The 16-byte transaction itself is identical in every case (one
    ``int4``/``__ldg(int4*)`` load at the 16-byte-aligned address), so an
    int8 stream never reads four int32 elements (64 bytes).
    """
    load = ("__ldg(reinterpret_cast<const int4*>(ptr))" if ldg else "*reinterpret_cast<const int4*>(ptr)")
    if byte_lltype is int8:
        suffix, cpp_t = "i8sext", "int8_t"
    elif byte_lltype is uint8:
        suffix, cpp_t = "u8zext", "uint8_t"
    else:
        suffix, cpp_t = "", None
    fn_name = helper_name if not suffix else f"{helper_name[:-3]}_{suffix}_fn"
    if cpp_t is None:
        lane_extract = ("    *r0 = value.x; *r1 = value.y;"
                        " *r2 = value.z; *r3 = value.w;\n")
    else:
        # CUDA targets are little-endian: bytes 0..3 of the 16-byte chunk
        # live in value.x, lowest byte first.
        lane_extract = ("    *r0 = static_cast<int32_t>(static_cast"
                        f"<{cpp_t}>(value.x & 0xFF));\n"
                        "    *r1 = static_cast<int32_t>(static_cast"
                        f"<{cpp_t}>((value.x >> 8) & 0xFF));\n"
                        "    *r2 = static_cast<int32_t>(static_cast"
                        f"<{cpp_t}>((value.x >> 16) & 0xFF));\n"
                        "    *r3 = static_cast<int32_t>(static_cast"
                        f"<{cpp_t}>((value.x >> 24) & 0xFF));\n")
    body = ("\n__device__ __forceinline__ void " + fn_name + "(\n"
            "        const void* ptr, int32_t* r0, int32_t* r1,"
            " int32_t* r2, int32_t* r3) {\n"
            f"    const int4 value = {load};\n" + lane_extract + "}\n")
    return body, fn_name


def codegen_gmem_ld_v4s32(mem, index, r0, r1, r2, r3, _arg_dtypes=None):
    elem_lltype = _gmem_v4s32_mem_element_lltype(_arg_dtypes)
    byte_lltype = elem_lltype if getattr(elem_lltype, "bits", None) == 8 \
        and getattr(elem_lltype, "kind", None) == "int" else None
    body, fn_name = _gmem_ld_v4s32_body("gmem_ld_v4s32_fn", ldg=False, byte_lltype=byte_lltype)
    return Builtin(body=body, includes=[], return_val=f"{fn_name}({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_v4s32", int32),
         codegen_func=codegen_gmem_ld_v4s32)
def gmem_ld_v4s32(mem, index, r0, r1, r2, r3):
    """Load one 16-byte vector from GMEM into four int32 locals.

    ``index`` counts elements of ``mem``'s dtype (element-typed pointer
    arithmetic) and ``mem + index`` must be 16-byte aligned; exactly one
    16-byte transaction is issued, never 4 x int32 (64 bytes).

    Lane semantics depend on the source element dtype:

    * int32 streams: each lane is one raw int32 element, i.e. the four
      words of the 16-byte transaction.
    * int8 streams: each lane is the sign-extended int8 element
      ``mem[index + i]`` (i = 0..3) taken from the low word of the
      16-byte transaction.
    * uint8 streams: the analogous zero extension.

    Byte-stream callers therefore walk elements with ``index`` in int8
    units (a multiple of 16 for the alignment contract) and receive
    widened int32 lanes.
    """
    raise RuntimeError("should not be called in compilation")


setattr(gmem_ld_v4s32, CODEGEN_NEEDS_TYPES_ATTR, True)


def codegen_gmem_ld_v4s32_nc(mem, index, r0, r1, r2, r3, _arg_dtypes=None):
    elem_lltype = _gmem_v4s32_mem_element_lltype(_arg_dtypes)
    byte_lltype = elem_lltype if getattr(elem_lltype, "bits", None) == 8 \
        and getattr(elem_lltype, "kind", None) == "int" else None
    body, fn_name = _gmem_ld_v4s32_body("gmem_ld_v4s32_nc_fn", ldg=True, byte_lltype=byte_lltype)
    return Builtin(body=body, includes=[], return_val=f"{fn_name}({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_v4s32_nc", int32),
         codegen_func=codegen_gmem_ld_v4s32_nc)
def gmem_ld_v4s32_nc(mem, index, r0, r1, r2, r3):
    """__ldg (read-only cache) variant of gmem_ld_v4s32; identical lane
    semantics (sign-extended int8 / zero-extended uint8 / raw int32 words
    by source dtype)."""
    raise RuntimeError("should not be called in compilation")


setattr(gmem_ld_v4s32_nc, CODEGEN_NEEDS_TYPES_ATTR, True)


def codegen_gmem_st_v4s32(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_v4s32_fn(
        void* ptr, int32_t r0, int32_t r1, int32_t r2, int32_t r3) {
    *reinterpret_cast<int4*>(ptr) = make_int4(r0, r1, r2, r3);
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_v4s32_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_v4s32", int32), eval_return_type=void,
         codegen_func=codegen_gmem_st_v4s32)
def gmem_st_v4s32(mem, index, r0, r1, r2, r3):
    """Store four int32 lanes as one 16-byte vector to GMEM."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_v4s32_cg(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_v4s32_cg_fn(
        void* ptr, int32_t r0, int32_t r1, int32_t r2, int32_t r3) {
    __stcg(reinterpret_cast<int4*>(ptr), make_int4(r0, r1, r2, r3));
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_v4s32_cg_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_v4s32_cg", int32), eval_return_type=void,
         codegen_func=codegen_gmem_st_v4s32_cg)
def gmem_st_v4s32_cg(mem, index, r0, r1, r2, r3):
    """Cache-global (.cg, L2-only) variant of gmem_st_v4s32; value-identical, eviction hint only."""
    raise RuntimeError("should not be called in compilation")


# ------------------------------------------------------------------------------
# Typed 16-byte vector family with SIXTEEN int8 lanes
# ------------------------------------------------------------------------------
#
# Sibling of the v4s32 family above for one-byte SIGNED elements: one 16-byte
# transaction exchanges 16 int8-typed lane locals through the caller's own
# registers (callee writes through ``&arg``), exactly mirroring the four-lane
# v4s32 contract.  The opaque gmem_ld_u8x16 twin instead stages the same 16
# bytes as two uint64 bit patterns whose vec_u8x16_lane extractor zero-extends
# each byte to uint32 (0..255); int8 consumers would then need 16 explicit
# sign-extension unpacks and lose the typed-lane discipline.  These builtins
# deliver signed int8 lanes directly, so int8 GEMV/dp4a staging matches the
# teacher's 16-byte vector width with no bit-pattern unpacking.
#
# Legality contract (identical to gmem_ld_v4s32): ``index`` counts elements of
# ``mem``'s dtype and must be a multiple of 16 so ``mem + index`` is 16-byte
# aligned; non-divisible tails use the scalar guarded path.  The ``_nc`` load
# routes through the read-only data cache (__ldg) and requires the loaded bytes
# to be read-only for the launch; the ``_cs`` store is the streaming
# (evict-first, .cs / __stcs) write hint and is value-identical.

_V16S8_LANE_COUNT = 16


def _v16s8_ptr_params():
    return ", ".join(f"int8_t* r{i}" for i in range(_V16S8_LANE_COUNT))


def _v16s8_value_params():
    return ", ".join(f"int8_t r{i}" for i in range(_V16S8_LANE_COUNT))


def _v16s8_load_writes():
    return " ".join(f"*r{i} = b[{i}];" for i in range(_V16S8_LANE_COUNT))


def _v16s8_pack_writes():
    return " ".join(f"b[{i}] = r{i};" for i in range(_V16S8_LANE_COUNT))


def _codegen_gmem_ld_v16s8(policy, mem, index, *lanes):
    if len(lanes) != _V16S8_LANE_COUNT:
        raise ValueError(f"gmem_ld_v16s8 requires exactly {_V16S8_LANE_COUNT} output "
                         f"lanes, got {len(lanes)}")
    if policy == "nc":
        name = "gmem_ld_v16s8_nc_fn"
        load = "__ldg(reinterpret_cast<const uint4*>(ptr))"
    else:
        name = "gmem_ld_v16s8_fn"
        load = "*reinterpret_cast<const uint4*>(ptr)"
    out_args = ", ".join(f"&{lane}" for lane in lanes)
    body = f"""
__device__ __forceinline__ void {name}(
        const void* ptr, {_v16s8_ptr_params()}) {{
    const uint4 value = {load};
    const int8_t* b = reinterpret_cast<const int8_t*>(&value);
    {_v16s8_load_writes()}
}}
"""
    return Builtin(body=body, includes=[], return_val=f"{name}({mem} + {index}, {out_args})")


def codegen_gmem_ld_v16s8(mem, index, *lanes):
    return _codegen_gmem_ld_v16s8(None, mem, index, *lanes)


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_v16s8", int8),
         codegen_func=codegen_gmem_ld_v16s8)
def gmem_ld_v16s8(mem, index, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15):
    """Load one 16-byte vector as SIXTEEN int8 lanes from GMEM into int8 locals."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_v16s8_nc(mem, index, *lanes):
    return _codegen_gmem_ld_v16s8("nc", mem, index, *lanes)


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_v16s8_nc", int8),
         codegen_func=codegen_gmem_ld_v16s8_nc)
def gmem_ld_v16s8_nc(mem, index, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15):
    """__ldg (read-only cache, .nc) variant of gmem_ld_v16s8."""
    raise RuntimeError("should not be called in compilation")


def _codegen_gmem_st_v16s8(policy, mem, index, *lanes):
    if len(lanes) != _V16S8_LANE_COUNT:
        raise ValueError(f"gmem_st_v16s8 requires exactly {_V16S8_LANE_COUNT} value "
                         f"lanes, got {len(lanes)}")
    if policy == "cs":
        name = "gmem_st_v16s8_cs_fn"
        store = "__stcs(reinterpret_cast<uint4*>(ptr), value)"
    else:
        name = "gmem_st_v16s8_fn"
        store = "*reinterpret_cast<uint4*>(ptr) = value"
    val_args = ", ".join(str(lane) for lane in lanes)
    body = f"""
__device__ __forceinline__ void {name}(
        void* ptr, {_v16s8_value_params()}) {{
    uint4 value;
    int8_t* b = reinterpret_cast<int8_t*>(&value);
    {_v16s8_pack_writes()}
    {store};
}}
"""
    return Builtin(body=body, includes=[], return_val=f"{name}({mem} + {index}, {val_args})")


def codegen_gmem_st_v16s8(mem, index, *lanes):
    return _codegen_gmem_st_v16s8(None, mem, index, *lanes)


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_v16s8", int8),
         codegen_func=codegen_gmem_st_v16s8)
def gmem_st_v16s8(mem, index, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15):
    """Store SIXTEEN int8 lanes as one 16-byte vector to GMEM."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_v16s8_cs(mem, index, *lanes):
    return _codegen_gmem_st_v16s8("cs", mem, index, *lanes)


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_v16s8_cs", int8),
         codegen_func=codegen_gmem_st_v16s8_cs)
def gmem_st_v16s8_cs(mem, index, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15):
    """Streaming (.cs / __stcs, evict-first) variant of gmem_st_v16s8; value-identical, cache hint only."""
    raise RuntimeError("should not be called in compilation")


_GMEM_LD_V8_16BIT_BODY = """
__device__ __forceinline__ void gmem_ld_v8{tag}{nc_suffix}_fn(
        const void* ptr, float* r0, float* r1, float* r2, float* r3,
        float* r4, float* r5, float* r6, float* r7) {{
    const uint4 value = {load_expr};
    const {pair2}* packed = reinterpret_cast<const {pair2}*>(&value);
    const float2 f0 = {to_float2}(packed[0]);
    const float2 f1 = {to_float2}(packed[1]);
    const float2 f2 = {to_float2}(packed[2]);
    const float2 f3 = {to_float2}(packed[3]);
    *r0 = f0.x; *r1 = f0.y; *r2 = f1.x; *r3 = f1.y;
    *r4 = f2.x; *r5 = f2.y; *r6 = f3.x; *r7 = f3.y;
}}
"""

_GMEM_ST_V8_16BIT_BODY = """
__device__ __forceinline__ void gmem_st_v8{tag}_fn(
        void* ptr, float r0, float r1, float r2, float r3,
        float r4, float r5, float r6, float r7) {{
    uint4 value;
    {pair2}* packed = reinterpret_cast<{pair2}*>(&value);
    packed[0] = {from_float2}(r0, r1);
    packed[1] = {from_float2}(r2, r3);
    packed[2] = {from_float2}(r4, r5);
    packed[3] = {from_float2}(r6, r7);
    *reinterpret_cast<uint4*>(ptr) = value;
}}
"""

_GMEM_ST_V8_16BIT_CG_BODY = """
__device__ __forceinline__ void gmem_st_v8{tag}_cg_fn(
        void* ptr, float r0, float r1, float r2, float r3,
        float r4, float r5, float r6, float r7) {{
    uint4 value;
    {pair2}* packed = reinterpret_cast<{pair2}*>(&value);
    packed[0] = {from_float2}(r0, r1);
    packed[1] = {from_float2}(r2, r3);
    packed[2] = {from_float2}(r4, r5);
    packed[3] = {from_float2}(r6, r7);
    __stcg(reinterpret_cast<int4*>(ptr), *reinterpret_cast<const int4*>(&value));
}}
"""

_GMEM_16BIT_KINDS = {
    "f16": ("__half2", "__half22float2", "__floats2half2_rn", "<cuda_fp16.h>"),
    "bf16": ("__nv_bfloat162", "__bfloat1622float2", "__floats2bfloat162_rn", "<cuda_bf16.h>"),
}


def _make_gmem_v8_16bit(tag):
    pair2, to_float2, from_float2, include = _GMEM_16BIT_KINDS[tag]

    def ld_codegen(mem, index, r0, r1, r2, r3, r4, r5, r6, r7):
        body = _GMEM_LD_V8_16BIT_BODY.format(tag=tag, nc_suffix="", pair2=pair2, to_float2=to_float2,
                                             load_expr="*reinterpret_cast<const uint4*>(ptr)")
        return Builtin(
            body=body, includes=[include], return_val=(f"gmem_ld_v8{tag}_fn({mem} + {index}, "
                                                       f"&{r0}, &{r1}, &{r2}, &{r3}, &{r4}, &{r5}, &{r6}, &{r7})"))

    def ld_nc_codegen(mem, index, r0, r1, r2, r3, r4, r5, r6, r7):
        body = _GMEM_LD_V8_16BIT_BODY.format(tag=tag, nc_suffix="_nc", pair2=pair2, to_float2=to_float2,
                                             load_expr="__ldg(reinterpret_cast<const uint4*>(ptr))")
        return Builtin(
            body=body, includes=[include], return_val=(f"gmem_ld_v8{tag}_nc_fn({mem} + {index}, "
                                                       f"&{r0}, &{r1}, &{r2}, &{r3}, &{r4}, &{r5}, &{r6}, &{r7})"))

    def st_codegen(mem, index, r0, r1, r2, r3, r4, r5, r6, r7):
        body = _GMEM_ST_V8_16BIT_BODY.format(tag=tag, pair2=pair2, from_float2=from_float2)
        return Builtin(
            body=body, includes=[include], return_val=(f"gmem_st_v8{tag}_fn({mem} + {index}, "
                                                       f"{r0}, {r1}, {r2}, {r3}, {r4}, {r5}, {r6}, {r7})"))

    def st_cg_codegen(mem, index, r0, r1, r2, r3, r4, r5, r6, r7):
        body = _GMEM_ST_V8_16BIT_CG_BODY.format(tag=tag, pair2=pair2, from_float2=from_float2)
        return Builtin(
            body=body, includes=[include], return_val=(f"gmem_st_v8{tag}_cg_fn({mem} + {index}, "
                                                       f"{r0}, {r1}, {r2}, {r3}, {r4}, {r5}, {r6}, {r7})"))

    return ld_codegen, ld_nc_codegen, st_codegen, st_cg_codegen


_codegen_ld_v8f16, _codegen_ld_v8f16_nc, _codegen_st_v8f16, _codegen_st_v8f16_cg = _make_gmem_v8_16bit("f16")
_codegen_ld_v8bf16, _codegen_ld_v8bf16_nc, _codegen_st_v8bf16, _codegen_st_v8bf16_cg = _make_gmem_v8_16bit("bf16")


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_ld_out_lane_type_check("ld_v8f16", float32),
         codegen_func=_codegen_ld_v8f16)
def gmem_ld_v8f16(mem, index, r0, r1, r2, r3, r4, r5, r6, r7):
    """Load one 16-byte vector as eight half elements, widened to float locals."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_ld_out_lane_type_check("ld_v8f16_nc", float32),
         codegen_func=_codegen_ld_v8f16_nc)
def gmem_ld_v8f16_nc(mem, index, r0, r1, r2, r3, r4, r5, r6, r7):
    """__ldg (read-only cache) variant of gmem_ld_v8f16."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_v8f16", float32), eval_return_type=void,
         codegen_func=_codegen_st_v8f16)
def gmem_st_v8f16(mem, index, r0, r1, r2, r3, r4, r5, r6, r7):
    """Store eight float lanes narrowed to half as one 16-byte vector to GMEM."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_v8f16_cg", float32), eval_return_type=void,
         codegen_func=_codegen_st_v8f16_cg)
def gmem_st_v8f16_cg(mem, index, r0, r1, r2, r3, r4, r5, r6, r7):
    """Cache-global (.cg, L2-only) variant of gmem_st_v8f16; value-identical, eviction hint only."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_ld_out_lane_type_check("ld_v8bf16", float32),
         codegen_func=_codegen_ld_v8bf16)
def gmem_ld_v8bf16(mem, index, r0, r1, r2, r3, r4, r5, r6, r7):
    """Load one 16-byte vector as eight bf16 elements, widened to float locals."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_ld_out_lane_type_check("ld_v8bf16_nc", float32),
         codegen_func=_codegen_ld_v8bf16_nc)
def gmem_ld_v8bf16_nc(mem, index, r0, r1, r2, r3, r4, r5, r6, r7):
    """__ldg (read-only cache) variant of gmem_ld_v8bf16."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_v8bf16", float32), eval_return_type=void,
         codegen_func=_codegen_st_v8bf16)
def gmem_st_v8bf16(mem, index, r0, r1, r2, r3, r4, r5, r6, r7):
    """Store eight float lanes narrowed to bf16 as one 16-byte vector to GMEM."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_v8bf16_cg", float32), eval_return_type=void,
         codegen_func=_codegen_st_v8bf16_cg)
def gmem_st_v8bf16_cg(mem, index, r0, r1, r2, r3, r4, r5, r6, r7):
    """Cache-global (.cg, L2-only) variant of gmem_st_v8bf16; value-identical, eviction hint only."""


def codegen_gmem_ld_f32x2(mem, index, r0, r1):
    body = """
__device__ __forceinline__ void gmem_ld_f32x2_fn(
        const float* ptr, uint32_t* r0, uint32_t* r1) {
    const uint2 value = *reinterpret_cast<const uint2*>(ptr);
    *r0 = value.x; *r1 = value.y;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_f32x2_fn({mem} + {index}, &{r0}, &{r1})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_f32x2", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_f32x2)
def gmem_ld_f32x2(mem, index, r0, r1):
    """Load two aligned contiguous FP32 values from GMEM into two uint32 registers."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_f32x2(mem, index, r0, r1):
    body = """
__device__ __forceinline__ void gmem_st_f32x2_fn(
        float* ptr, uint32_t r0, uint32_t r1) {
    *reinterpret_cast<uint2*>(ptr) = uint2{r0, r1};
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_f32x2_fn({mem} + {index}, {r0}, {r1})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_f32x2", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_st_f32x2)
def gmem_st_f32x2(mem, index, r0, r1):
    """Store two uint32 register bit patterns as aligned contiguous FP32 values in GMEM."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_v2_f32(mem, index, r0, r1):
    body = """
__device__ __forceinline__ void gmem_ld_v2_f32_fn(
        const void* ptr, float* r0, float* r1) {
    const float2 value = *reinterpret_cast<const float2*>(ptr);
    *r0 = value.x; *r1 = value.y;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_v2_f32_fn({mem} + {index}, &{r0}, &{r1})")


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_v2_f32", float32),
         codegen_func=codegen_gmem_ld_v2_f32)
def gmem_ld_v2_f32(mem, index, r0, r1):
    """Load one 8-byte float2 vector as two FP32 lanes from GMEM into float locals.

    ``index`` counts elements of the tensor dtype and the caller guarantees
    ``mem + index`` is 8-byte aligned (even FP32 element index); frontends
    handle odd tails with the usual scalar guarded path.  Typed-lane twin of
    the raw bit-pattern ``gmem_ld_f32x2``."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_v2_f32(mem, index, r0, r1):
    body = """
__device__ __forceinline__ void gmem_st_v2_f32_fn(
        void* ptr, float r0, float r1) {
    *reinterpret_cast<float2*>(ptr) = make_float2(r0, r1);
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_v2_f32_fn({mem} + {index}, {r0}, {r1})")


@builtin(eval_return_type=void, eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_v2_f32", float32),
         codegen_func=codegen_gmem_st_v2_f32)
def gmem_st_v2_f32(mem, index, r0, r1):
    """Store two FP32 lanes as one 8-byte float2 vector to GMEM.

    ``index`` counts elements of the tensor dtype and the caller guarantees
    ``mem + index`` is 8-byte aligned (even FP32 element index); frontends
    handle odd tails with the usual scalar guarded path.  Typed-lane twin of
    the raw bit-pattern ``gmem_st_f32x2``, so FP32 float2 traffic no longer
    needs ``float_as_uint`` bitcasts at every call site."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_f16x8_u32x4(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_f16x8_u32x4_fn(
        const __half* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = *reinterpret_cast<const uint4*>(ptr);
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_f16x8_u32x4_fn({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_f16x8_u32x4", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_f16x8_u32x4)
def gmem_ld_f16x8_u32x4(mem, index, r0, r1, r2, r3):
    """Raw 4 x uint32 (bit-pattern) load of eight aligned contiguous FP16
    (or raw 16-bit) values from GMEM as four uint32 registers.

    The plain gmem_ld_f16x8 name is owned by the uint64-pair variant
    matching its .cg/.streaming twins; this raw load keeps the explicit
    suffix.  index is in 16-bit elements; the caller guarantees 16-byte
    alignment of mem + index (same contract as gmem_ld_f32x4); bit patterns
    are unconverted."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_f16x8_u32x4(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_f16x8_u32x4_fn(
        __half* ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    *reinterpret_cast<uint4*>(ptr) = uint4{r0, r1, r2, r3};
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_st_f16x8_u32x4_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_f16x8_u32x4", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_st_f16x8_u32x4)
def gmem_st_f16x8_u32x4(mem, index, r0, r1, r2, r3):
    """Store four uint32 register bit patterns as eight aligned contiguous FP16 values in GMEM.

    index is in 16-bit elements; the caller guarantees 16-byte alignment of
    mem + index (same contract as gmem_st_f32x4).
    Naming note: the plain gmem_st_f16x8 name is bound (below) to
    the RNE-pack 8 x FP32 variant matching its .cg/.streaming policy twins;
    this raw bit-pattern store keeps the explicit _u32x4 suffix, mirroring
    gmem_ld_f16x8_u32x4."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_bf16x8_u32x4(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_bf16x8_u32x4_fn(
        const __nv_bfloat16* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = *reinterpret_cast<const uint4*>(ptr);
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_bf16x8_u32x4_fn({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_bf16x8_u32x4", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_bf16x8_u32x4)
def gmem_ld_bf16x8_u32x4(mem, index, r0, r1, r2, r3):
    """Raw 4 x uint32 (bit-pattern) load of eight aligned contiguous BF16
    (or raw 16-bit) values from GMEM as four uint32 registers.

    The plain gmem_ld_bf16x8 name is owned by the uint64-pair variant
    matching its .cg/.streaming twins; this raw load keeps the explicit
    suffix.  index is in 16-bit elements; the caller guarantees 16-byte
    alignment of mem + index (same contract as gmem_ld_f32x4); bit patterns
    are unconverted."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_bf16x8_u32x4(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_bf16x8_u32x4_fn(
        __nv_bfloat16* ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    *reinterpret_cast<uint4*>(ptr) = uint4{r0, r1, r2, r3};
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_st_bf16x8_u32x4_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_bf16x8_u32x4", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_st_bf16x8_u32x4)
def gmem_st_bf16x8_u32x4(mem, index, r0, r1, r2, r3):
    """Store four uint32 register bit patterns as eight aligned contiguous BF16 values in GMEM.

    index is in 16-bit elements; the caller guarantees 16-byte alignment of
    mem + index (same contract as gmem_st_f32x4).
    Naming note: the plain gmem_st_bf16x8 name is bound (below) to
    the RNE-pack 8 x FP32 variant matching its .cg/.streaming policy twins;
    this raw bit-pattern store keeps the explicit _u32x4 suffix, mirroring
    gmem_ld_bf16x8_u32x4."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_ro_f32x4(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_ro_f32x4_fn(
        const float* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = __ldg(reinterpret_cast<const uint4*>(ptr));
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_ro_f32x4_fn({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_ro_f32x4", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_ro_f32x4)
def gmem_ld_ro_f32x4(mem, index, r0, r1, r2, r3):
    """Read-only (non-coherent texture path) variant of gmem_ld_f32x4.

    Lowers to __ldg -> LDG.E.128.CONSTANT-class SASS. Legal only when the
    loaded region is not written by any thread (incl. the caller) for the
    lifetime of the kernel; same 16-byte alignment contract as gmem_ld_f32x4."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_ro_f32x2(mem, index, r0, r1):
    body = """
__device__ __forceinline__ void gmem_ld_ro_f32x2_fn(
        const float* ptr, uint32_t* r0, uint32_t* r1) {
    const uint2 value = __ldg(reinterpret_cast<const uint2*>(ptr));
    *r0 = value.x; *r1 = value.y;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_ro_f32x2_fn({mem} + {index}, &{r0}, &{r1})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_ro_f32x2", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_ro_f32x2)
def gmem_ld_ro_f32x2(mem, index, r0, r1):
    """Read-only variant of gmem_ld_f32x2; see gmem_ld_ro_f32x4 for the legality contract."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_ro_f16x8(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_ro_f16x8_fn(
        const __half* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = __ldg(reinterpret_cast<const uint4*>(ptr));
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_ro_f16x8_fn({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_ro_f16x8", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_ro_f16x8)
def gmem_ld_ro_f16x8(mem, index, r0, r1, r2, r3):
    """Read-only variant of gmem_ld_f16x8_u32x4; see gmem_ld_ro_f32x4 for the legality contract."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_ro_bf16x8(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_ro_bf16x8_fn(
        const __nv_bfloat16* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = __ldg(reinterpret_cast<const uint4*>(ptr));
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_ro_bf16x8_fn({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_ro_bf16x8", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_ro_bf16x8)
def gmem_ld_ro_bf16x8(mem, index, r0, r1, r2, r3):
    """Read-only variant of gmem_ld_bf16x8_u32x4; see gmem_ld_ro_f32x4 for the legality contract."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_cs_bf16x8(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_cs_bf16x8_fn(
        const __nv_bfloat16* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = __ldcs(reinterpret_cast<const uint4*>(ptr));
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_cs_bf16x8_fn({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_cs_bf16x8", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_cs_bf16x8)
def gmem_ld_cs_bf16x8(mem, index, r0, r1, r2, r3):
    """Cache-streaming (evict-first) variant of gmem_ld_bf16x8_u32x4 via __ldcs.

    __ldcs is only a cache hint (ld.global.cs: data is not expected to be
    re-read), so any address legal for gmem_ld_bf16x8_u32x4 is legal here; the same
    16-byte alignment contract applies and bit patterns are unconverted."""

    raise RuntimeError("should not be called in compilation")


# ---------------------------------------------------------------------------
# Value-native typed 128-bit (x4) global-memory vector intrinsics.
#
# Unlike gmem_ld_f32x4 / gmem_st_f32x4 above, which shuffle raw uint32 bit
# patterns, these operate on the element dtype's native scalar registers
# (float / int / unsigned int), so a frontend can load four lanes, run element
# arithmetic or comparisons on them, and store the results without any manual
# bit-cast. Consecutive threads must own consecutive chunks (chunk base =
# 4 * chunk_id) so each LDG/STG.128 is a fully coalesced, full-sector
# wavefront.
#
# Generic contract (not task-specific):
#   * ``mem`` is a GMEM tensor of the matching element dtype (float32 -> float,
#     int32 -> int, uint32 -> unsigned int). ``index`` is the element offset of
#     the first of the four lanes.
#   * ``r0..r3`` are scalar registers of that same dtype.
#   * Alignment precondition: (base address of mem) must be 16-byte aligned
#     (true for torch allocations) and ``index`` must be a multiple of 4, so
#     ``mem + index`` is 16-byte aligned. Violating this is a hardware
#     misaligned-address fault, not a silent clamp; this is documented rather
#     than checked because the check would cost a runtime branch on the hot
#     path. Frontends handle the ``n % 4`` tail with a separate scalar loop.
#   * The plain variants emit a normal coalesced 128-bit LDG/STG.
#   * The ``_cs`` (cache-streaming, evict-first) variants use __ldcs/__stcs and
#     are for single-touch streams: write-once output canvases and read-once
#     inputs that would otherwise pollute L2.
# ---------------------------------------------------------------------------


def codegen_gmem_ld_f32x4_val(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_f32x4_val_fn(
        const float* ptr, float& r0, float& r1, float& r2, float& r3) {
    const float4 value = *reinterpret_cast<const float4*>(ptr);
    r0 = value.x; r1 = value.y; r2 = value.z; r3 = value.w;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_f32x4_val_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


# ==============================================================================
# 128-bit global vector access, float64 element dtype (two f64 lanes, typed)
# ==============================================================================
#
# Exact dtype-twin of gmem_ld_f32x4/gmem_st_f32x4 for 64-bit elements: one
# 16-byte double2 vector move, with the two lanes delivered/accepted as
# typed float64 registers (no manual uint32/uint64 bit assembly; pairs with
# f64_as_u64/u64_as_f64 cuda_asm bitcasts when raw bits are wanted).  Any
# f64-streaming kernel (or int64, after u64<->f64 bitcasts) reaches
# LDG.E.128/STG.E.128 without the generic byte-lane v4 family.


def codegen_gmem_ld_f64x2(mem, index, r0, r1):
    body = """
__device__ __forceinline__ void gmem_ld_f64x2_fn(
        const double* ptr, double* r0, double* r1) {
    const double2 value = *reinterpret_cast<const double2*>(ptr);
    *r0 = value.x; *r1 = value.y;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_f64x2_fn({mem} + {index}, &{r0}, &{r1})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_f64x2)
def gmem_ld_f64x2(mem, index, r0, r1):
    """Load two aligned contiguous FP64 values from GMEM into two float64 registers.

    Exact dtype-twin of ``gmem_ld_f32x4`` with one 16-byte ``double2`` move.
    ``mem`` must be a float64 tensor; ``index`` is in units of f64 elements.
    Legality precondition (frontend-author responsibility, identical wording
    to the f32x4 / v4_u32 families): the byte address of ``mem + index`` must
    be 16-byte aligned (equivalently, ``index`` even relative to the tensor
    base) and both elements in-bounds readable; ragged tails and unaligned
    offsets must go through a separate scalar path.  Violating alignment is
    a fault (misaligned address), the same precondition CUDA ``double2``
    dereference has.  Lane order matches memory order: ``r0 <- mem[index]``,
    ``r1 <- mem[index+1]``.
    """
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_f64x2(mem, index, v0, v1):
    body = """
__device__ __forceinline__ void gmem_st_f64x2_fn(
        double* ptr, double v0, double v1) {
    *reinterpret_cast<double2*>(ptr) = make_double2(v0, v1);
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_f64x2_fn({mem} + {index}, {v0}, {v1})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_f64x2)
def gmem_st_f64x2(mem, index, v0, v1):
    """Store two float64 registers as two aligned contiguous FP64 values in GMEM.

    Same dtype, index-unit, and 16-byte alignment contract as
    ``gmem_ld_f64x2``; exact dtype-twin of ``gmem_st_f32x4``.
    """
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_cs_f32x4(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_cs_f32x4_fn(
        const float* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = __ldcs(reinterpret_cast<const uint4*>(ptr));
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_cs_f32x4_fn({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_cs_f32x4", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_cs_f32x4)
def gmem_ld_cs_f32x4(mem, index, r0, r1, r2, r3):
    """Streaming (evict-first) 128-bit load of four aligned 32-bit lanes.

    Identical layout contract to gmem_ld_f32x4 (index counts float-sized
    elements, ptr must be 16-byte aligned at ptr + index) but lowers to
    __ldcs so the cache line is marked evict-first: use for data that is
    read exactly once. The returned r0..r3 are raw uint32 bit patterns.
    gmem_ld_f32x4_cs (below) is an alias spelling of this intrinsic.
    """
    raise RuntimeError("should not be called in compilation")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_f32x4_val", float32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_f32x4_val)
def gmem_ld_f32x4_val(mem, index, r0, r1, r2, r3):
    """Load four aligned contiguous float32 lanes into four float registers."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_f32x4_cs(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_f32x4_cs_fn(
        const float* ptr, float& r0, float& r1, float& r2, float& r3) {
    const float4 value = __ldcs(reinterpret_cast<const float4*>(ptr));
    r0 = value.x; r1 = value.y; r2 = value.z; r3 = value.w;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_f32x4_cs_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


def codegen_gmem_st_f32x4_val(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_f32x4_val_fn(
        float* ptr, float r0, float r1, float r2, float r3) {
    *reinterpret_cast<float4*>(ptr) = make_float4(r0, r1, r2, r3);
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_f32x4_val_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_f32x4_val", float32), eval_return_type=void,
         codegen_func=codegen_gmem_st_f32x4_val)
def gmem_st_f32x4_val(mem, index, r0, r1, r2, r3):
    """Store four float registers as aligned contiguous float32 lanes in GMEM."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_f32x4_cs(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_f32x4_cs_fn(
        float* ptr, float r0, float r1, float r2, float r3) {
    __stcs(reinterpret_cast<float4*>(ptr), make_float4(r0, r1, r2, r3));
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_f32x4_cs_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


def codegen_gmem_ld_i32x4(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_i32x4_fn(
        const int* ptr, int& r0, int& r1, int& r2, int& r3) {
    const int4 value = *reinterpret_cast<const int4*>(ptr);
    r0 = value.x; r1 = value.y; r2 = value.z; r3 = value.w;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_i32x4_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_i32x4)
def gmem_ld_i32x4(mem, index, r0, r1, r2, r3):
    """Load four aligned contiguous int32 lanes into four int registers."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_i32x4(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_i32x4_fn(
        int* ptr, int r0, int r1, int r2, int r3) {
    *reinterpret_cast<int4*>(ptr) = make_int4(r0, r1, r2, r3);
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_i32x4_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_i32x4)
def gmem_st_i32x4(mem, index, r0, r1, r2, r3):
    """Store four int registers as aligned contiguous int32 lanes in GMEM."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_i32x4_cs(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_i32x4_cs_fn(
        int* ptr, int r0, int r1, int r2, int r3) {
    __stcs(reinterpret_cast<int4*>(ptr), make_int4(r0, r1, r2, r3));
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_i32x4_cs_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_i32x4_cs)
def gmem_st_i32x4_cs(mem, index, r0, r1, r2, r3):
    """Streaming (evict-first) store of four int registers as int32 lanes."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_u32x4_cs(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_u32x4_cs_fn(
        unsigned int* ptr, unsigned int r0, unsigned int r1, unsigned int r2, unsigned int r3) {
    __stcs(reinterpret_cast<uint4*>(ptr), make_uint4(r0, r1, r2, r3));
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_u32x4_cs_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_u32x4_cs)
def gmem_st_u32x4_cs(mem, index, r0, r1, r2, r3):
    """Streaming (evict-first) store of four unsigned registers as uint32 lanes."""
    raise RuntimeError("should not be called in compilation")


# Cache-policy name -> load expression applied to a uint4 pointer.
_GMEM_LD_V4_EXPR = {
    "wb": "*reinterpret_cast<const uint4*>(ptr)",
    "ca": "__ldca(reinterpret_cast<const uint4*>(ptr))",
    "cg": "__ldcg(reinterpret_cast<const uint4*>(ptr))",
    "cs": "__ldcs(reinterpret_cast<const uint4*>(ptr))",
    "lu": "__ldlu(reinterpret_cast<const uint4*>(ptr))",
    "nc": "__ldg(reinterpret_cast<const uint4*>(ptr))",
    "cv": "__ldcv(reinterpret_cast<const uint4*>(ptr))",
}

# Cache-policy name -> store statement applied to a uint4 pointer/value.
_GMEM_ST_V4_STMT = {
    "wb": "*reinterpret_cast<uint4*>(ptr) = value;",
    "cg": "__stcg(reinterpret_cast<uint4*>(ptr), value);",
    "cs": "__stcs(reinterpret_cast<uint4*>(ptr), value);",
    "wt": "__stwt(reinterpret_cast<uint4*>(ptr), value);",
}

# Lane types that carry raw 32-bit patterns without value conversion.
_GMEM_V4_LANE_TYPES = (int32, uint32)


def _gmem_v4_legality(mem_type, lane_types, caller, lane_count=4):
    """Eval-time legality: tensor first arg + 32-bit integer lanes.

    Runs in eval_return_type (type inference), never at string-emission
    time: unsupported dtypes fail before any CUDA text exists.
    """
    if mem_type is None:
        raise TypeError(f"{caller}: cannot infer type of first argument (mem)")
    is_mem = (hasattr(mem_type, "is_tensor") and mem_type.is_tensor()) or \
             (hasattr(mem_type, "is_pointer") and mem_type.is_pointer())
    if not is_mem:
        raise TypeError(f"{caller} expects a Tensor (or pointer) as first argument, "
                        f"got {mem_type!r}")
    for i, lane in enumerate(lane_types):
        if lane is None:
            continue  # untyped integral literal: renders as-is
        if isinstance(lane, LLType) and lane not in _GMEM_V4_LANE_TYPES:
            raise TypeError(f"{caller}: lane r{i} must be int32 or uint32 (raw 32-bit "
                            f"bit pattern), got {lane!r}. Bit-cast float values first; "
                            "implicit C++ float->uint32 conversion would corrupt bits.")
    return void


def codegen_gmem_ld_v4(mem, index, r0, r1, r2, r3, policy=None):
    policy = _normalize_cache_policy(policy, _GMEM_LD_V4_EXPR.keys(), "wb", "gmem_ld_v4")
    suffix = "" if policy == "wb" else f"_{policy}"
    guard = f"LK_GMEM_LD_V4{suffix.upper()}_FN_DEFINED"
    body = f"""
#ifndef {guard}
#define {guard}
template <typename T>
__device__ __forceinline__ void gmem_ld_v4{suffix}_fn(
        const T* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {{
    const uint4 value = {_GMEM_LD_V4_EXPR[policy]};
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}}
#endif
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_v4{suffix}_fn({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(
    eval_return_type=lambda mem, index, r0, r1, r2, r3, policy=None: _gmem_v4_legality(
        mem, (r0, r1, r2, r3), "gmem_ld_v4"), codegen_func=codegen_gmem_ld_v4)
def gmem_ld_v4(mem, index, r0, r1, r2, r3, policy=None):
    """Load 16 contiguous bytes from GMEM at element ``mem[index]`` into four
    uint32 registers, for any element dtype.

    The pointee type is deduced from mem's C++ pointer type, so one
    intrinsic serves int32 (4 elements/dword), float32, bfloat16/float16
    (8 elements), int64 (2), int8 (16), etc.  ``index`` counts in units of
    mem's element dtype.  ``policy`` is an optional compile-time string
    literal (see the family doc block above); bit-identical either way.
    """
    raise RuntimeError("should not call gmem_ld_v4 in compilation")


def codegen_gmem_st_cs_f32x4(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_cs_f32x4_fn(
        float* ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    __stcs(reinterpret_cast<uint4*>(ptr), uint4{r0, r1, r2, r3});
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_cs_f32x4_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_cs_f32x4", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_st_cs_f32x4)
def gmem_st_cs_f32x4(mem, index, r0, r1, r2, r3):
    """Streaming (evict-first) 128-bit store of four uint32 register bit patterns.

    Same layout contract as gmem_st_f32x4 but lowers to __stcs, marking the
    line evict-first for write-once outputs. __stcs is only a cache hint, so
    any address legal for gmem_st_f32x4 is legal here. gmem_st_f32x4_cs
    (below) is an alias spelling of this intrinsic.
    """
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_v4(mem, index, r0, r1, r2, r3, policy=None):
    policy = _normalize_cache_policy(policy, _GMEM_ST_V4_STMT.keys(), "wb", "gmem_st_v4")
    suffix = "" if policy == "wb" else f"_{policy}"
    guard = f"LK_GMEM_ST_V4{suffix.upper()}_FN_DEFINED"
    body = f"""
#ifndef {guard}
#define {guard}
template <typename T>
__device__ __forceinline__ void gmem_st_v4{suffix}_fn(
        T* ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {{
    const uint4 value = make_uint4(r0, r1, r2, r3);
    {_GMEM_ST_V4_STMT[policy]}
}}
#endif
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_v4{suffix}_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(
    eval_return_type=lambda mem, index, r0, r1, r2, r3, policy=None: _gmem_v4_legality(
        mem, (r0, r1, r2, r3), "gmem_st_v4"), codegen_func=codegen_gmem_st_v4)
def gmem_st_v4(mem, index, r0, r1, r2, r3, policy=None):
    """Store four uint32 register bit patterns as 16 contiguous bytes in
    GMEM at element ``mem[index]``, for any element dtype.

    Same dtype deduction, index units, and alignment contract as
    ``gmem_ld_v4``.  ``policy``: "wb" (default write-back), "cg" L2-only,
    "cs" evict-first streaming, "wt" write-through (aliases
    "l2_only"/"evict_first"/"write_through").
    """
    raise RuntimeError("should not call gmem_st_v4 in compilation")


def codegen_gmem_st_v4_zero(mem, index, policy=None):
    policy = _normalize_cache_policy(policy, _GMEM_ST_V4_STMT.keys(), "wb", "gmem_st_v4_zero")
    suffix = "" if policy == "wb" else f"_{policy}"
    guard = f"LK_GMEM_ST_V4_ZERO{suffix.upper()}_FN_DEFINED"
    body = f"""
#ifndef {guard}
#define {guard}
template <typename T>
__device__ __forceinline__ void gmem_st_v4_zero{suffix}_fn(T* ptr) {{
    const uint4 value = make_uint4(0u, 0u, 0u, 0u);
    {_GMEM_ST_V4_STMT[policy]}
}}
#endif
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_v4_zero{suffix}_fn({mem} + {index})")


@builtin(eval_return_type=lambda mem, index, policy=None: _gmem_v4_legality(mem, (), "gmem_st_v4_zero"),
         codegen_func=codegen_gmem_st_v4_zero)
def gmem_st_v4_zero(mem, index, policy=None):
    """Zero-fill 16 contiguous bytes in GMEM at element ``mem[index]`` with
    one 128-bit store, for any element dtype, without materializing zero
    lanes in the frontend.  Same alignment/index/policy contract as
    ``gmem_st_v4``.
    """
    raise RuntimeError("should not call gmem_st_v4_zero in compilation")


# ==============================================================================
# Dtype-generic 128-bit global vector access over 64-bit elements (x2)
# ==============================================================================
#
# Value-native typed counterpart of the gmem_ld_v4/gmem_st_v4 family above
# for 8-byte element dtypes (float64 / int64 / uint64).  The v4 family
# shuffles raw uint32 lanes; there is no generic bit-cast in the language,
# so 64-bit-dtype frontends (complex128 interleaved as fp64 pairs, packed
# int64 stream pairs, ...) could not issue LDG/STG.128 at all.  These two
# intrinsics load/store two ELEMENT-TYPED registers, dtype deduced from
# mem's C++ pointer type, like the _val variants of the x4 family.
#
# Generic contract (no task or shape assumptions):
#   * ``mem``    -- any LittleKernel Tensor (global scope) whose element
#                   dtype is 64 bits; the helper is a C++ template on T.
#   * ``index``  -- element offset in units of mem's own dtype (C++ pointer
#                   arithmetic).
#   * Legality preconditions (frontend-author responsibility, same as the
#     x4 families): the byte address of ``mem + index`` MUST be 16-byte
#     aligned (index even for a 16-byte-aligned base) and all 16 bytes
#     in-bounds.  Ragged tails belong on a scalar path.  The compiler does
#     not infer alignment or bounds; a bad address is a misaligned-access
#     fault, identical to dereferencing ulonglong2* in CUDA C.
#   * Lanes lo/hi are element-typed scalars (float64/int64/uint64 matching
#     mem's dtype); the transfer is bit-exact (ulonglong2 carrier), so
#     every cache policy is numerically a pure performance hint.
#   * Cache policy mirrors the same aliases/normalization as gmem_ld_v4 /
#     gmem_st_v4: loads "wb"/"ca"/"cg"/"cs"/"lu"/"nc"/"cv", stores
#     "wb"/"cg"/"cs"/"wt".
#
# For interleaved complex128 (re at element 2*i, im at 2*i+1) with torch
# complex128 tensors, the base is 16-byte aligned and even indices keep
# every access 16-byte aligned.

# Cache-policy name -> load expression applied to a ulonglong2 pointer.
_GMEM_LD_X2_EXPR = {
    "wb": "*reinterpret_cast<const ulonglong2*>(ptr)",
    "ca": "__ldca(reinterpret_cast<const ulonglong2*>(ptr))",
    "cg": "__ldcg(reinterpret_cast<const ulonglong2*>(ptr))",
    "cs": "__ldcs(reinterpret_cast<const ulonglong2*>(ptr))",
    "lu": "__ldlu(reinterpret_cast<const ulonglong2*>(ptr))",
    "nc": "__ldg(reinterpret_cast<const ulonglong2*>(ptr))",
    "cv": "__ldcv(reinterpret_cast<const ulonglong2*>(ptr))",
}

# Cache-policy name -> store statement applied to a ulonglong2 pointer/value.
_GMEM_ST_X2_STMT = {
    "wb": "*reinterpret_cast<ulonglong2*>(ptr) = value;",
    "cg": "__stcg(reinterpret_cast<ulonglong2*>(ptr), value);",
    "cs": "__stcs(reinterpret_cast<ulonglong2*>(ptr), value);",
    "wt": "__stwt(reinterpret_cast<ulonglong2*>(ptr), value);",
}


def _gmem_x2_legality(mem_type, lane_types, caller):
    """Eval-time legality: tensor first arg, 64-bit element/lane dtypes.

    Runs in eval_return_type (type inference), before any CUDA text exists.
    """
    if mem_type is None:
        raise TypeError(f"{caller}: cannot infer type of first argument (mem)")
    is_mem = (hasattr(mem_type, "is_tensor") and mem_type.is_tensor()) or \
             (hasattr(mem_type, "is_pointer") and mem_type.is_pointer())
    if not is_mem:
        raise TypeError(f"{caller} expects a Tensor (or pointer) as first argument, "
                        f"got {mem_type!r}")
    from little_kernel.core.type_system import TensorType
    if isinstance(mem_type, TensorType):
        elem = mem_type.element_type
        if isinstance(elem, LLType) and elem.bits != 64:
            raise TypeError(f"{caller}: element dtype of mem must be 64-bit "
                            f"(float64/int64/uint64), got {elem!r}. gmem_ld_x2/"
                            f"gmem_st_x2 transfer two 64-bit lanes as one 128-bit "
                            f"access; use the gmem_ld_v4/gmem_st_v4 family for other "
                            f"element sizes.")
    for i, lane in enumerate(lane_types):
        if lane is None:
            continue  # untyped literal: renders as-is
        if isinstance(lane, LLType) and lane.bits != 64:
            raise TypeError(f"{caller}: lane {i} must be a 64-bit scalar "
                            f"(float64/int64/uint64) matching mem's dtype, got "
                            f"{lane!r}. The transfer is bit-exact; convert values "
                            f"explicitly instead of relying on implicit C++ casts.")
    return void


def codegen_gmem_ld_x2(mem, index, lo, hi, policy=None):
    policy = _normalize_cache_policy(policy, _GMEM_LD_X2_EXPR.keys(), "wb", "gmem_ld_x2")
    suffix = "" if policy == "wb" else f"_{policy}"
    guard = f"LK_GMEM_LD_X2{suffix.upper()}_FN_DEFINED"
    body = f"""
#ifndef {guard}
#define {guard}
template <typename T>
__device__ __forceinline__ void gmem_ld_x2{suffix}_fn(
        const T* ptr, T* lo, T* hi) {{
    static_assert(sizeof(T) == 8,
        "gmem_ld_x2 requires a 64-bit element dtype");
    const ulonglong2 value = {_GMEM_LD_X2_EXPR[policy]};
    *lo = *reinterpret_cast<const T*>(&value.x);
    *hi = *reinterpret_cast<const T*>(&value.y);
}}
#endif
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_x2{suffix}_fn({mem} + {index}, &{lo}, &{hi})")


@builtin(eval_return_type=lambda mem, index, lo, hi, policy=None: _gmem_x2_legality(mem, (lo, hi), "gmem_ld_x2"),
         codegen_func=codegen_gmem_ld_x2)
def gmem_ld_x2(mem, index, lo, hi, policy=None):
    """Load 16 contiguous bytes (two elements) from GMEM at ``mem[index]``
    into two element-typed registers, for any 64-bit element dtype.

    ``index`` counts in units of mem's element dtype and must be even for a
    16-byte-aligned base (alignment contract identical to gmem_ld_v4).
    ``policy`` is an optional compile-time string literal with the same
    spellings/aliases as gmem_ld_v4 (default "wb"); bit-identical either
    way.
    """
    raise RuntimeError("should not call gmem_ld_x2 in compilation")


def codegen_gmem_st_x2(mem, index, lo, hi, policy=None):
    policy = _normalize_cache_policy(policy, _GMEM_ST_X2_STMT.keys(), "wb", "gmem_st_x2")
    suffix = "" if policy == "wb" else f"_{policy}"
    guard = f"LK_GMEM_ST_X2{suffix.upper()}_FN_DEFINED"
    body = f"""
#ifndef {guard}
#define {guard}
template <typename T>
__device__ __forceinline__ void gmem_st_x2{suffix}_fn(
        T* ptr, T lo, T hi) {{
    static_assert(sizeof(T) == 8,
        "gmem_st_x2 requires a 64-bit element dtype");
    ulonglong2 value;
    value.x = *reinterpret_cast<const unsigned long long*>(&lo);
    value.y = *reinterpret_cast<const unsigned long long*>(&hi);
    {_GMEM_ST_X2_STMT[policy]}
}}
#endif
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_x2{suffix}_fn({mem} + {index}, {lo}, {hi})")


@builtin(eval_return_type=lambda mem, index, lo, hi, policy=None: _gmem_x2_legality(mem, (lo, hi), "gmem_st_x2"),
         codegen_func=codegen_gmem_st_x2)
def gmem_st_x2(mem, index, lo, hi, policy=None):
    """Store two element-typed registers as 16 contiguous bytes in GMEM at
    ``mem[index]``, for any 64-bit element dtype.

    Same dtype deduction, index units, and 16-byte alignment contract as
    ``gmem_ld_x2``.  ``policy``: "wb" (default), "cg", "cs", "wt" with the
    same aliases as gmem_st_v4.
    """
    raise RuntimeError("should not call gmem_st_x2 in compilation")


# The element-index and raw-address cp.async families share one canonical
# raw-address implementation (cp_async_cg_16_pred / cp_async_cg_16_zfill in
# async_copy); a second element-index spelling is intentionally not bound
# here so each public name has exactly one codegen/signature owner.
# ==============================================================================
# Packed bf16x2 global access as raw uint32 bit patterns
# ==============================================================================


def codegen_gmem_ld_bf16x2_u32(mem, index, bits=None):
    # Dual calling convention (same arity-split pattern as the cp.async
    # dual-arity intrinsics):
    #   3 args: (mem, index, bits) -- statement form assigning the loaded
    #           pair bit pattern into a pre-declared uint32 destination.
    #   2 args: (mem, index) -- expression form returning the uint32 value
    #           directly (the expression-form calling convention).
    # Both forms perform the SAME single 4-byte load at element offset
    # 2*index under the alignment contract, so the one-load/one-contract
    # invariant holds; ``bits is None`` unambiguously marks the expression
    # form because codegen arguments are pre-visited source strings.
    deref = f"*reinterpret_cast<const uint32_t*>(({mem}) + 2 * ({index}))"
    if bits is None:
        return Builtin(body="", includes=[], return_val=deref)
    return Builtin(body="", includes=[], return_val=f"({bits}) = {deref}")


@builtin(eval_return_type=lambda mem, index, bits=None: (void if bits is not None else uint32),
         codegen_func=codegen_gmem_ld_bf16x2_u32)
def gmem_ld_bf16x2_u32(mem, index, bits=None):
    """Load uint32 bit pattern from one packed bf16x2 pair in global memory.

    index counts PAIRS (element offset = 2 * index), mirroring
    gmem_st_bf16x2_u32; pair into bf16x2_to_float2 keeps rounding confined
    to the single conversion point. Contract: 4-byte aligned at element
    offset 2*index, 4 bytes in bounds.

    Same dual 2-arg/3-arg calling convention as the cp.async intrinsics'
    dual-arity split: pass a pre-declared uint32 destination as the third
    argument for statement form, or omit it to use the load as a
    uint32-valued expression.
    """
    raise RuntimeError("should not call gmem_ld_bf16x2_u32 in compilation")


def codegen_gmem_st_bf16x2_u32(mem, index, bits):
    return Builtin(body="", includes=[], return_val=f"*reinterpret_cast<uint32_t*>(({mem}) + 2 * ({index})) = ({bits})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_bf16x2_u32)
def gmem_st_bf16x2_u32(mem, index, bits):
    """Store uint32 bit pattern as one packed bf16x2 pair in global memory.

    index counts PAIRS (element offset = 2 * index); pair from
    bf16x2_pack_u32 keeps rounding confined to the single conversion point.
    """
    raise RuntimeError("should not call gmem_st_bf16x2_u32 in compilation")


def codegen_gmem_ld_v4_u32_ptr(gmem_ptr, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_v4_u32_ptr_fn(
        const void* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = *reinterpret_cast<const uint4*>(ptr);
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_v4_u32_ptr_fn({gmem_ptr}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_v4_u32_ptr)
def gmem_ld_v4_u32_ptr(gmem_ptr, r0, r1, r2, r3):
    """Load 16 bytes from a byte-addressed global pointer into four uint32 registers.

    Unlike gmem_ld_f32x4 this is keyed on a void* (compose with
    ptr_byte_offset), so it works for tensors of ANY dtype (bf16, fp8/uint8,
    int32, fp32) without a typed-pointer cast. Contract (frontend program's
    responsibility, matching the explicit-intrinsic legality split): ptr must
    be 16-byte aligned and the 16 bytes must be in bounds; r0..r3 receive raw
    32-bit lanes little-endian. Scalar element loads remain the fallback when
    alignment/contiguity is not provable.
    """
    raise RuntimeError("should not be called in compilation")


# Opt the scalar load/store codegens into the emitter's _arg_dtypes channel
# so the 1-byte-element legalization above can key on the tensor element
# LLType (typed intrinsic dispatch, no string/name matching).
setattr(gmem_ld_scalar, CODEGEN_NEEDS_TYPES_ATTR, True)
setattr(gmem_st_scalar, CODEGEN_NEEDS_TYPES_ATTR, True)


def codegen_gmem_ld_nc_v4_u32(gmem_ptr, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_nc_v4_u32_fn(
        const void* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = __ldg(reinterpret_cast<const uint4*>(ptr));
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_nc_v4_u32_fn({gmem_ptr}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_nc_v4_u32)
def gmem_ld_nc_v4_u32(gmem_ptr, r0, r1, r2, r3):
    """Read-only (non-coherent, __ldg) byte-addressed 128-bit global load.

    Same 16-byte alignment contract as gmem_ld_v4_u32 but lowers to __ldg
    (ld.global.nc.v4) for the read-only data path. Additional legality
    obligation on the program: the loaded bytes must not be written by any
    thread for the duration of the kernel launch (the CUDA __ldg contract).
    Use for gather/scatter copy sources (e.g. appended rows copied into a
    paged cache) where the source is provably read-only.
    """
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_v4_u32_ptr(gmem_ptr, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_v4_u32_ptr_fn(
        void* ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    *reinterpret_cast<uint4*>(ptr) = uint4{r0, r1, r2, r3};
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_v4_u32_ptr_fn({gmem_ptr}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_v4_u32_ptr)
def gmem_st_v4_u32_ptr(gmem_ptr, r0, r1, r2, r3):
    """Store four uint32 register bit patterns as 16 bytes at a byte-addressed global pointer.

    void*-keyed (compose with ptr_byte_offset), so it works for tensors of
    ANY dtype. Contract: ptr must be 16-byte aligned and the 16 bytes must be
    in bounds; scalar element stores remain the fallback otherwise.
    """
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_cs_v4_u32(gmem_ptr, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_cs_v4_u32_fn(
        const void* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = __ldcs(reinterpret_cast<const uint4*>(ptr));
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_cs_v4_u32_fn({gmem_ptr}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_cs_v4_u32)
def gmem_ld_cs_v4_u32(gmem_ptr, r0, r1, r2, r3):
    """Streaming (evict-first) byte-addressed 128-bit global load.

    Same layout/alignment contract as gmem_ld_nc_v4_u32 (byte-addressed
    void*, 16-byte alignment) but lowered to __ldcs (ld.global.cs.v4): use
    for read-once sources so the line is marked evict-first."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_cs_v4_u32(gmem_ptr, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_cs_v4_u32_fn(
        void* ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    __stcs(reinterpret_cast<uint4*>(ptr), uint4{r0, r1, r2, r3});
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_cs_v4_u32_fn({gmem_ptr}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_cs_v4_u32)
def gmem_st_cs_v4_u32(gmem_ptr, r0, r1, r2, r3):
    """Streaming (evict-first) byte-addressed 128-bit global store.

    Same layout/alignment contract as gmem_st_v4_u32 but lowers to __stcs
    (st.global.cs.v4): use for write-once destinations so the line is marked
    evict-first.
    """
    raise RuntimeError("should not be called in compilation")


# --- merge note (CPU global integrator): plain gmem_ld/st_v4_u32 keep the
# element-indexed (mem, index, ...) contract exercised by merged unit tests;
# the byte-addressed void* variants carry the _ptr suffix. ---


def codegen_gmem_ld_v4_u32(mem, index, r0, r1, r2, r3):
    body = r'''
__device__ __forceinline__ void gmem_ld_v4_u32_fn(
        const void* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(*r0), "=r"(*r1), "=r"(*r2), "=r"(*r3)
                 : "l"(ptr) : "memory");
}
'''
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_v4_u32_fn((const void*)({mem} + {index}), &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_v4_u32)
def gmem_ld_v4_u32(mem, index, r0, r1, r2, r3):
    """Load 16 contiguous bytes (four uint32 lanes) from GMEM via ld.global.v4.b32.

    Dtype-agnostic: `index` is an element index of `mem` (any element type,
    e.g. half/bf16/fp32/uint32); the typed element address is computed first
    and only then viewed as bytes. Legality contract (caller-owned, not
    compiler-checked): `mem + index` must be 16-byte aligned and the 16 bytes
    contiguous; unaligned rows/tails must use scalar loads instead."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_v4_u32(mem, index, v0, v1, v2, v3=None):
    # Dual calling convention, distinguished purely by arity:
    #   6 args: (mem, index, v0..v3) -- tensor + element index; address is
    #          (void*)(mem + index), index in units of mem's pointee dtype.
    #   5 args: (gmem_ptr, r0..r3)   -- raw pointer expression (e.g. from
    #          ptr_byte_offset), the raw-pointer convention; v3 is
    #          then None. Both conventions lower to the SAME single
    #          st.global.v4.b32 over the addressed 16 bytes; the 16-byte
    #          alignment + in-bounds legality contract is the caller's in
    #          both. v3 is never a real lane expression here because codegen
    #          arguments are visited source strings, so None is unambiguous.
    if v3 is None:
        ptr_expr = mem
        v0, v1, v2, v3 = index, v0, v1, v2
    else:
        ptr_expr = f"(void*)({mem} + {index})"
    body = r'''
__device__ __forceinline__ void gmem_st_v4_u32_fn(
        void* ptr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}
'''
    return Builtin(body=body, includes=[], return_val=f"gmem_st_v4_u32_fn({ptr_expr}, {v0}, {v1}, {v2}, {v3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_v4_u32)
def gmem_st_v4_u32(mem, index, v0, v1, v2, v3=None):
    """Store four uint32 lanes (16 contiguous bytes) to GMEM via st.global.v4.b32.

    Two equivalent calling conventions, distinguished purely by arity:
    ``(mem, index, v0..v3)`` addresses ``mem + index`` (element units,
    6 args); ``(gmem_ptr, r0..r3)`` stores through an already-formed byte
    pointer (5 args, e.g. ``ptr_byte_offset`` output). Both emit the same
    st.global.v4.b32. Same legality contract as gmem_ld_v4_u32: the
    addressed 16 bytes must be 16-byte aligned and contiguous; stores do
    no value conversion (bit patterns are preserved), so packing is the
    caller's job (e.g. pack_bf16)."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_v4_u32_pl(mem, index, r0, r1, r2, r3):
    body = r'''
__device__ __forceinline__ void gmem_ld_v4_u32_pl_fn(
        const void* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = *reinterpret_cast<const uint4*>(ptr);
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
'''
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_v4_u32_pl_fn((const void*)({mem} + {index}), &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_v4_u32_pl)
def gmem_ld_v4_u32_pl(mem, index, r0, r1, r2, r3):
    """Plain-C++ (non-barrier) form of gmem_ld_v4_u32.

    Same dtype-agnostic 16-byte vector load and the same caller-owned
    16-byte alignment/contiguity legality contract as gmem_ld_v4_u32, but
    lowered to a plain `*reinterpret_cast<const uint4*>` load instead of
    `asm volatile ... : "memory"`: ptxas may hoist, merge, or reorder it
    across other loads/stores exactly as for ordinary C++ dereferences.
    Loaded bit patterns are identical to gmem_ld_v4_u32 — this is an
    opt-in scheduling-relaxation hint only, no numeric change. Caller
    contract: do not use it where load/store ordering is relied on for
    synchronization."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_v4_u32_pl(mem, index, v0, v1, v2, v3):
    body = r'''
__device__ __forceinline__ void gmem_st_v4_u32_pl_fn(
        void* ptr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    *reinterpret_cast<uint4*>(ptr) = uint4{v0, v1, v2, v3};
}
'''
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_st_v4_u32_pl_fn((void*)({mem} + {index}), {v0}, {v1}, {v2}, {v3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_v4_u32_pl)
def gmem_st_v4_u32_pl(mem, index, v0, v1, v2, v3):
    """Plain-C++ (non-barrier) form of gmem_st_v4_u32.

    Same legality contract as gmem_st_v4_u32 and identical stored bit
    patterns, but lowered to a plain `*reinterpret_cast<uint4*>` store
    with no asm barrier/"memory" clobber, so ptxas may schedule across it
    like an ordinary C++ store. Opt-in scheduling-relaxation hint only."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_v4_u32_cs(mem, index, r0, r1, r2, r3):
    body = r'''
__device__ __forceinline__ void gmem_ld_v4_u32_cs_fn(
        const void* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(*r0), "=r"(*r1), "=r"(*r2), "=r"(*r3)
                 : "l"(ptr) : "memory");
}
'''
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_v4_u32_cs_fn((const void*)({mem} + {index}), &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_v4_u32_cs)
def gmem_ld_v4_u32_cs(mem, index, r0, r1, r2, r3):
    """gmem_ld_v4_u32 with the .cs (evict-first) cache policy for read-once streams.

    Same dtype-agnostic 16-byte vector load and the same caller-owned
    16-byte alignment/contiguity legality contract as gmem_ld_v4_u32; the
    only difference is the PTX cache op. Opt-in hint for pure-streaming
    read-once data to reduce L2 pressure — it does not change loaded
    values, so it is always semantics-preserving relative to
    gmem_ld_v4_u32."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_v4_u32_nc(mem, index, r0, r1, r2, r3):
    body = r'''
__device__ __forceinline__ void gmem_ld_v4_u32_nc_fn(
        const void* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(*r0), "=r"(*r1), "=r"(*r2), "=r"(*r3)
                 : "l"(ptr) : "memory");
}
'''
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_v4_u32_nc_fn((const void*)({mem} + {index}), &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_v4_u32_nc)
def gmem_ld_v4_u32_nc(mem, index, r0, r1, r2, r3):
    """gmem_ld_v4_u32 via the read-only/non-coherent path (ld.global.nc.v4.b32).

    Same dtype-agnostic 16-byte vector load, "l" 64-bit address constraint,
    "memory" clobber, and caller-owned 16-byte alignment/contiguity legality
    contract as gmem_ld_v4_u32. Additional caller contract: the loaded bytes
    must be read-only for the lifetime of the kernel launch (nc is the
    __restrict__/const read-only path). Values loaded equal those of
    gmem_ld_v4_u32 under that contract, so it is semantics-preserving; this is
    an opt-in aliasability hint only."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_v4_u32_cs(mem, index, v0, v1, v2, v3):
    body = r'''
__device__ __forceinline__ void gmem_st_v4_u32_cs_fn(
        void* ptr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}
'''
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_st_v4_u32_cs_fn((void*)({mem} + {index}), {v0}, {v1}, {v2}, {v3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_v4_u32_cs)
def gmem_st_v4_u32_cs(mem, index, v0, v1, v2, v3):
    """gmem_st_v4_u32 with the .cs (evict-first) cache policy for write-once streams.

    Same legality contract as gmem_st_v4_u32; opt-in for outputs that are
    written once and never re-read by the kernel, so the write-allocated
    lines are evicted first instead of thrashing L2 resident read data.
    Semantics-preserving relative to gmem_st_v4_u32."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_vld_f32x4(mem, index, f0, f1, f2, f3):
    body = """
__device__ __forceinline__ void gmem_vld_f32x4_fn(
        const float* ptr, float* f0, float* f1, float* f2, float* f3) {
    const float4 v = *reinterpret_cast<const float4*>(ptr);
    *f0 = v.x; *f1 = v.y; *f2 = v.z; *f3 = v.w;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_vld_f32x4_fn({mem} + {index}, &{f0}, &{f1}, &{f2}, &{f3})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_vld_f32x4", float32), eval_return_type=void,
         codegen_func=codegen_gmem_vld_f32x4)
def gmem_vld_f32x4(mem, index, f0, f1, f2, f3):
    """Load four aligned contiguous FP32 values from GMEM as FP32 registers
    (float-lane variant of gmem_ld_f32x4; no uint bit-pattern round-trip)."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_vst_f32x4(mem, index, f0, f1, f2, f3):
    body = """
__device__ __forceinline__ void gmem_vst_f32x4_fn(
        float* ptr, float f0, float f1, float f2, float f3) {
    *reinterpret_cast<float4*>(ptr) = float4{f0, f1, f2, f3};
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_vst_f32x4_fn({mem} + {index}, {f0}, {f1}, {f2}, {f3})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_vst_f32x4", float32), eval_return_type=void,
         codegen_func=codegen_gmem_vst_f32x4)
def gmem_vst_f32x4(mem, index, f0, f1, f2, f3):
    """Store four FP32 registers as aligned contiguous FP32 values in GMEM."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Raw 16-byte vector access (dtype-agnostic packed payloads, e.g. 16 x int8 /
# 16 x fp8 quantized bytes). `byte_index` counts BYTES relative to `mem`;
# caller guarantees 16-byte alignment and full in-bounds payload.
# ==============================================================================


def codegen_gmem_ld_u32x4(mem, byte_index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_u32x4_fn(
        const uint8_t* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 v = *reinterpret_cast<const uint4*>(ptr);
    *r0 = v.x; *r1 = v.y; *r2 = v.z; *r3 = v.w;
}
"""
    return Builtin(
        body=body, includes=["<cstdint>"], return_val=
        f"gmem_ld_u32x4_fn(reinterpret_cast<const uint8_t*>({mem}) + ({byte_index}), &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_u32x4)
def gmem_ld_u32x4(mem, byte_index, r0, r1, r2, r3):
    """Raw 16-byte aligned vector load as four uint32 registers (byte offset)."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_u32x4(mem, byte_index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_u32x4_fn(
        uint8_t* ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    *reinterpret_cast<uint4*>(ptr) = uint4{r0, r1, r2, r3};
}
"""
    return Builtin(
        body=body, includes=["<cstdint>"],
        return_val=f"gmem_st_u32x4_fn(reinterpret_cast<uint8_t*>({mem}) + ({byte_index}), {r0}, {r1}, {r2}, {r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_u32x4)
def gmem_st_u32x4(mem, byte_index, r0, r1, r2, r3):
    """Raw 16-byte aligned vector store of four uint32 registers (byte offset)."""
    raise RuntimeError("should not be called in compilation")


# Cache-policy name -> load expression applied to a uint4 pointer.
_GMEM_LD_POLICY_EXPR = {
    "wb": "*reinterpret_cast<const uint4*>(ptr)",  # plain / default
    "ca": "__ldca(reinterpret_cast<const uint4*>(ptr))",
    "cg": "__ldcg(reinterpret_cast<const uint4*>(ptr))",
    "cs": "__ldcs(reinterpret_cast<const uint4*>(ptr))",
    "lu": "__ldlu(reinterpret_cast<const uint4*>(ptr))",
    "nc": "__ldg(reinterpret_cast<const uint4*>(ptr))",
    "cv": "__ldcv(reinterpret_cast<const uint4*>(ptr))",
}

# Cache-policy name -> store statement applied to a uint4 pointer/value.
# (single canonical definition).


def codegen_gmem_ld_cs_f32x2(mem, index, r0, r1):
    body = """
__device__ __forceinline__ void gmem_ld_cs_f32x2_fn(
        const float* ptr, uint32_t* r0, uint32_t* r1) {
    const uint2 value = __ldcs(reinterpret_cast<const uint2*>(ptr));
    *r0 = value.x; *r1 = value.y;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_cs_f32x2_fn({mem} + {index}, &{r0}, &{r1})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_cs_f32x2", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_cs_f32x2)
def gmem_ld_cs_f32x2(mem, index, r0, r1):
    """Cache-streaming (evict-first) variant of gmem_ld_f32x2 via __ldcs.

    Same hint-only legality contract as gmem_ld_cs_f32x4 with the f32x2
    8-byte alignment contract. gmem_ld_f32x2_cs is an alias spelling."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_cs_f32x2(mem, index, r0, r1):
    body = """
__device__ __forceinline__ void gmem_st_cs_f32x2_fn(
        float* ptr, uint32_t r0, uint32_t r1) {
    __stcs(reinterpret_cast<uint2*>(ptr), uint2{r0, r1});
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_cs_f32x2_fn({mem} + {index}, {r0}, {r1})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_cs_f32x2", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_st_cs_f32x2)
def gmem_st_cs_f32x2(mem, index, r0, r1):
    """Cache-streaming (evict-first) variant of gmem_st_f32x2 via __stcs.

    Same hint-only legality contract as gmem_st_cs_f32x4 with the f32x2
    8-byte alignment contract. gmem_st_f32x2_cs is an alias spelling."""
    raise RuntimeError("should not be called in compilation")


# Request-spelled trailing-suffix aliases of the four f32 .cs intrinsics
# above; they share the canonical codegen functions, and the emitter dedups
# identical helper definitions by function name, so mixing spellings in one
# kernel is harmless.
@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_f32x4_cs", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_cs_f32x4)
def gmem_ld_f32x4_cs(mem, index, r0, r1, r2, r3):
    """Alias of gmem_ld_cs_f32x4."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_f32x4_cs", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_st_cs_f32x4)
def gmem_st_f32x4_cs(mem, index, r0, r1, r2, r3):
    """Alias of gmem_st_cs_f32x4."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_f32x2_cs", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_ld_cs_f32x2)
def gmem_ld_f32x2_cs(mem, index, r0, r1):
    """Alias of gmem_ld_cs_f32x2."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_f32x2_cs", uint32), eval_return_type=void,
         codegen_func=codegen_gmem_st_cs_f32x2)
def gmem_st_f32x2_cs(mem, index, r0, r1):
    """Alias of gmem_st_cs_f32x2."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Typed float-lane 128-bit global vector access.
# Four float32-register variants of the byte-addressed v4 global ld/st family,
# for programs streaming fp32 tiles without per-use uint<->float bitcasts.
# Same caller-owned legality contract (16B alignment, in-bounds) as the u32
# family; scalar element access remains the fallback when it does not hold.
# ==============================================================================


def codegen_gmem_ld_v4_f32(gmem_ptr, f0, f1, f2, f3):
    body = """
__device__ __forceinline__ void gmem_ld_v4_f32_fn(
        const void* ptr, float* f0, float* f1, float* f2, float* f3) {
    const float4 value = *reinterpret_cast<const float4*>(ptr);
    *f0 = value.x; *f1 = value.y; *f2 = value.z; *f3 = value.w;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_v4_f32_fn({gmem_ptr}, &{f0}, &{f1}, &{f2}, &{f3})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_v4_f32", float32, leading_operands=1),
         eval_return_type=void, codegen_func=codegen_gmem_ld_v4_f32)
def gmem_ld_v4_f32(gmem_ptr, f0, f1, f2, f3):
    """Load 16 bytes from a byte-addressed global pointer into four FLOAT32 registers.

    Typed float-lane counterpart of gmem_ld_v4_u32: same void*-keyed,
    byte-addressed contract (compose with ptr_byte_offset), but f0..f3 are
    real float32 registers (ld.global.v4.f32), so fp32 tile-streaming
    programs need no uint_as_float bitcast pair at every use site. Use the
    u32 form when the loaded lanes are NOT float (bf16x2 pairs, fp8 codes,
    index words). Contract (frontend program's responsibility, matching the
    explicit-intrinsic legality split): ptr must be 16-byte aligned and the
    16 bytes must be in bounds; scalar element loads remain the fallback
    when alignment/contiguity is not provable.
    """
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_cs_v4_f32(gmem_ptr, f0, f1, f2, f3):
    body = """
__device__ __forceinline__ void gmem_ld_cs_v4_f32_fn(
        const void* ptr, float* f0, float* f1, float* f2, float* f3) {
    const float4 value = __ldcs(reinterpret_cast<const float4*>(ptr));
    *f0 = value.x; *f1 = value.y; *f2 = value.z; *f3 = value.w;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_cs_v4_f32_fn({gmem_ptr}, &{f0}, &{f1}, &{f2}, &{f3})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_cs_v4_f32", float32, leading_operands=1),
         eval_return_type=void, codegen_func=codegen_gmem_ld_cs_v4_f32)
def gmem_ld_cs_v4_f32(gmem_ptr, f0, f1, f2, f3):
    """Streaming (evict-first) 128-bit global load into four FLOAT32 registers.

    Same layout/alignment contract as gmem_ld_v4_f32 but lowers to __ldcs
    (ld.global.cs.v4.f32): use for data read exactly once so it does not
    pollute L2.
    """
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_nc_v4_f32(gmem_ptr, f0, f1, f2, f3):
    body = """
__device__ __forceinline__ void gmem_ld_nc_v4_f32_fn(
        const void* ptr, float* f0, float* f1, float* f2, float* f3) {
    const float4 value = __ldg(reinterpret_cast<const float4*>(ptr));
    *f0 = value.x; *f1 = value.y; *f2 = value.z; *f3 = value.w;
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_ld_nc_v4_f32_fn({gmem_ptr}, &{f0}, &{f1}, &{f2}, &{f3})")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("gmem_ld_nc_v4_f32", float32, leading_operands=1),
         eval_return_type=void, codegen_func=codegen_gmem_ld_nc_v4_f32)
def gmem_ld_nc_v4_f32(gmem_ptr, f0, f1, f2, f3):
    """Read-only (non-coherent, __ldg) 128-bit global load into four FLOAT32 registers.

    Same 16-byte alignment contract as gmem_ld_v4_f32 but lowers to __ldg
    (ld.global.nc.v4.f32) for the read-only data path. Additional legality
    obligation on the program: the loaded bytes must not be written by any
    thread for the duration of the kernel launch (the CUDA __ldg contract).
    """
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_v4_f32(gmem_ptr, f0, f1, f2, f3):
    body = """
__device__ __forceinline__ void gmem_st_v4_f32_fn(
        void* ptr, float f0, float f1, float f2, float f3) {
    *reinterpret_cast<float4*>(ptr) = make_float4(f0, f1, f2, f3);
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_v4_f32_fn({gmem_ptr}, {f0}, {f1}, {f2}, {f3})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_v4_f32", float32, leading_operands=1),
         eval_return_type=void, codegen_func=codegen_gmem_st_v4_f32)
def gmem_st_v4_f32(gmem_ptr, f0, f1, f2, f3):
    """Store four FLOAT32 registers as 16 bytes at a byte-addressed global pointer.

    Typed float-lane counterpart of gmem_st_v4_u32 (st.global.v4.f32): no
    float_as_uint bitcast pair at the use site; use the u32 form when the
    lanes are not float. Contract: ptr must be 16-byte aligned and the 16
    bytes must be in bounds; scalar element stores remain the fallback
    otherwise.
    """
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_cs_v4_f32(gmem_ptr, f0, f1, f2, f3):
    body = """
__device__ __forceinline__ void gmem_st_cs_v4_f32_fn(
        void* ptr, float f0, float f1, float f2, float f3) {
    __stcs(reinterpret_cast<float4*>(ptr), make_float4(f0, f1, f2, f3));
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_cs_v4_f32_fn({gmem_ptr}, {f0}, {f1}, {f2}, {f3})")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("gmem_st_cs_v4_f32", float32, leading_operands=1),
         eval_return_type=void, codegen_func=codegen_gmem_st_cs_v4_f32)
def gmem_st_cs_v4_f32(gmem_ptr, f0, f1, f2, f3):
    """Streaming (evict-first) 128-bit global store of four FLOAT32 registers.

    Same layout/alignment contract as gmem_st_v4_f32 but lowers to __stcs
    (st.global.cs.v4.f32): use for write-once destinations so the line is
    marked evict-first.
    """
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Dtype-generic 128-bit (u32x4) and 64-bit (u32x2) vector access with
# explicit .cg/.cs/.wt cache-policy suffixes.  Addressing contract
# (element units): byte address = base + index * sizeof(element) must be
# 16B (u32x4) or 8B (u32x2) aligned.  NOTE the addressing difference
# against the plain names: gmem_ld_u32x4/gmem_st_u32x4 take a BYTE index,
# while the suffixed members of this family take an ELEMENT index, matching
# the policy-parameterized gmem_ld_v4/gmem_st_v4 convention.  Byte- vs
# element-index colliding plain spellings, a duplicate gmem_st_u32x4_cs, and
# a duplicate prefetch_l2 are intentionally not bound here so each public
# name has one owner; the canonical owners live earlier in this module.
# the incoming side of this hunk.
_GMEM_U32X4_LD_BODIES = {
    "_cg": "    const uint4 value = __ldcg(reinterpret_cast<const uint4*>(ptr));\n",
    "_cs": "    const uint4 value = __ldcs(reinterpret_cast<const uint4*>(ptr));\n",
}
_GMEM_U32X4_ST_BODIES = {
    "_wt": "    __stwt(reinterpret_cast<uint4*>(ptr), uint4{r0, r1, r2, r3});\n",
}


def _make_gmem_ld_u32x4(suffix):
    name = f"gmem_ld_u32x4{suffix}_fn"

    def codegen(mem, index, r0, r1, r2, r3):
        body = (f"\n__device__ __forceinline__ void {name}(\n"
                "        const void* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {\n" +
                _GMEM_U32X4_LD_BODIES[suffix] + "    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;\n"
                "}\n")
        return Builtin(body=body, includes=["<cuda_runtime.h>"],
                       return_val=f"{name}({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")

    return codegen


def _make_gmem_st_u32x4(suffix):
    name = f"gmem_st_u32x4{suffix}_fn"

    def codegen(mem, index, r0, r1, r2, r3):
        body = (f"\n__device__ __forceinline__ void {name}(\n"
                "        void* ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {\n" +
                _GMEM_U32X4_ST_BODIES[suffix] + "}\n")
        return Builtin(body=body, includes=["<cuda_runtime.h>"],
                       return_val=f"{name}({mem} + {index}, {r0}, {r1}, {r2}, {r3})")

    return codegen


_GMEM_LD_U32X4_DOC = ("Load 16 bytes from GMEM ({policy}) as four uint32 lanes. "
                      "Args: (mem, element_index, r0, r1, r2, r3); address "
                      "mem + element_index must be 16-byte aligned.")
_GMEM_ST_U32X4_DOC = ("Store four uint32 lanes as 16 bytes to GMEM ({policy}). "
                      "Args: (mem, element_index, r0, r1, r2, r3); address "
                      "mem + element_index must be 16-byte aligned.")


@builtin(eval_return_type=void, codegen_func=_make_gmem_ld_u32x4("_cg"))
def gmem_ld_u32x4_cg(mem, index, r0, r1, r2, r3):
    raise RuntimeError("should not be called in compilation")


gmem_ld_u32x4_cg.__doc__ = _GMEM_LD_U32X4_DOC.format(policy="cache-global L2")


@builtin(eval_return_type=void, codegen_func=_make_gmem_ld_u32x4("_cs"))
def gmem_ld_u32x4_cs(mem, index, r0, r1, r2, r3):
    raise RuntimeError("should not be called in compilation")


gmem_ld_u32x4_cs.__doc__ = _GMEM_LD_U32X4_DOC.format(policy="evict-first streaming")


@builtin(eval_return_type=void, codegen_func=_make_gmem_st_u32x4("_wt"))
def gmem_st_u32x4_wt(mem, index, r0, r1, r2, r3):
    raise RuntimeError("should not be called in compilation")


gmem_st_u32x4_wt.__doc__ = _GMEM_ST_U32X4_DOC.format(policy="write-through")

# Dtype-generic 64-bit (8-byte) GMEM vector load/store with explicit cache
# policy, scaled down from u32x4: mem + element_index must be 8-byte aligned.
_GMEM_U32X2_LD_BODIES = {
    "": "    const uint2 value = *reinterpret_cast<const uint2*>(ptr);\n",
    "_cg": "    const uint2 value = __ldcg(reinterpret_cast<const uint2*>(ptr));\n",
    "_cs": "    const uint2 value = __ldcs(reinterpret_cast<const uint2*>(ptr));\n",
    "_nc": "    const uint2 value = __ldg(reinterpret_cast<const uint2*>(ptr));\n",
}
_GMEM_U32X2_ST_BODIES = {
    "": "    *reinterpret_cast<uint2*>(ptr) = uint2{r0, r1};\n",
    "_cs": "    __stcs(reinterpret_cast<uint2*>(ptr), uint2{r0, r1});\n",
    "_wt": "    __stwt(reinterpret_cast<uint2*>(ptr), uint2{r0, r1});\n",
}


def _make_gmem_ld_u32x2(suffix):
    name = f"gmem_ld_u32x2{suffix}_fn"

    def codegen(mem, index, r0, r1):
        body = (f"\n__device__ __forceinline__ void {name}(\n"
                "        const void* ptr, uint32_t* r0, uint32_t* r1) {\n" + _GMEM_U32X2_LD_BODIES[suffix] +
                "    *r0 = value.x; *r1 = value.y;\n"
                "}\n")
        return Builtin(body=body, includes=["<cuda_runtime.h>"], return_val=f"{name}({mem} + {index}, &{r0}, &{r1})")

    return codegen


def _make_gmem_st_u32x2(suffix):
    name = f"gmem_st_u32x2{suffix}_fn"

    def codegen(mem, index, r0, r1):
        body = (f"\n__device__ __forceinline__ void {name}(\n"
                "        void* ptr, uint32_t r0, uint32_t r1) {\n" + _GMEM_U32X2_ST_BODIES[suffix] + "}\n")
        return Builtin(body=body, includes=["<cuda_runtime.h>"], return_val=f"{name}({mem} + {index}, {r0}, {r1})")

    return codegen


_GMEM_LD_U32X2_DOC = ("Load 8 bytes from GMEM ({policy}) as two uint32 lanes. "
                      "Args: (mem, element_index, r0, r1); address "
                      "mem + element_index must be 8-byte aligned.")
_GMEM_ST_U32X2_DOC = ("Store two uint32 lanes as 8 bytes to GMEM ({policy}). "
                      "Args: (mem, element_index, r0, r1); address "
                      "mem + element_index must be 8-byte aligned.")


@builtin(eval_return_type=void, codegen_func=_make_gmem_ld_u32x2(""))
def gmem_ld_u32x2(mem, index, r0, r1):
    raise RuntimeError("should not be called in compilation")


gmem_ld_u32x2.__doc__ = _GMEM_LD_U32X2_DOC.format(policy="default")


@builtin(eval_return_type=void, codegen_func=_make_gmem_ld_u32x2("_cg"))
def gmem_ld_u32x2_cg(mem, index, r0, r1):
    raise RuntimeError("should not be called in compilation")


gmem_ld_u32x2_cg.__doc__ = _GMEM_LD_U32X2_DOC.format(policy="cache-global L2")


@builtin(eval_return_type=void, codegen_func=_make_gmem_ld_u32x2("_cs"))
def gmem_ld_u32x2_cs(mem, index, r0, r1):
    raise RuntimeError("should not be called in compilation")


gmem_ld_u32x2_cs.__doc__ = _GMEM_LD_U32X2_DOC.format(policy="evict-first streaming")


@builtin(eval_return_type=void, codegen_func=_make_gmem_ld_u32x2("_nc"))
def gmem_ld_u32x2_nc(mem, index, r0, r1):
    raise RuntimeError("should not be called in compilation")


gmem_ld_u32x2_nc.__doc__ = _GMEM_LD_U32X2_DOC.format(
    policy="read-only (.nc / __ldg)") + ("  Read-only legality contract (same as gmem_ld_v4_u32_nc / "
                                         "gmem_ld_u8x16_nc): the loaded 8 bytes must not be written by any "
                                         "thread for the duration of the launch; the FRONTEND asserts this "
                                         "(e.g. a gathered read-only source table whose rows are only read "
                                         "while the outputs are disjoint fresh allocations).  This is the "
                                         "8-byte twin for pairs of int32/uint32 lanes — e.g. two packed "
                                         "scale-factor words per gathered entry — where the row width is "
                                         "8 bytes rather than 16: duplicate gathered rows make repeated "
                                         "reads of the same line common, which is exactly the pattern the "
                                         "read-only data cache path is for.  Bits moved are identical to "
                                         "gmem_ld_u32x2; only the cache path differs.")


@builtin(eval_return_type=void, codegen_func=_make_gmem_st_u32x2(""))
def gmem_st_u32x2(mem, index, r0, r1):
    raise RuntimeError("should not be called in compilation")


gmem_st_u32x2.__doc__ = _GMEM_ST_U32X2_DOC.format(policy="default")


@builtin(eval_return_type=void, codegen_func=_make_gmem_st_u32x2("_cs"))
def gmem_st_u32x2_cs(mem, index, r0, r1):
    raise RuntimeError("should not be called in compilation")


gmem_st_u32x2_cs.__doc__ = _GMEM_ST_U32X2_DOC.format(policy="evict-first streaming")


@builtin(eval_return_type=void, codegen_func=_make_gmem_st_u32x2("_wt"))
def gmem_st_u32x2_wt(mem, index, r0, r1):
    raise RuntimeError("should not be called in compilation")


gmem_st_u32x2_wt.__doc__ = _GMEM_ST_U32X2_DOC.format(policy="write-through")


# ==============================================================================
# uint32-lane .cg/.streaming (evict-first) variants of the gmem_ld/st_f32x4
# family, and uint64-pair bf16x8/f16x8 .cg/.streaming loads/stores with static
# lane extractors.  The plain gmem_ld/st_{bf16,f16}x8 names are owned by the
# widened/numerical contracts earlier in this module, so only the suffixed
# policy variants are defined here.
# ==============================================================================
def codegen_gmem_ld_f32x4_cg(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_f32x4_cg_fn(
        const float* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = __ldcg(reinterpret_cast<const uint4*>(ptr));
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_f32x4_cg_fn({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_f32x4_cg)
def gmem_ld_f32x4_cg(mem, index, r0, r1, r2, r3):
    """Load four aligned contiguous FP32 values from GMEM into four uint32
    registers with the L2-only (.cg) cache policy. Bits moved are identical
    to :func:`gmem_ld_f32x4` — only the cache behavior differs; the same
    16-byte alignment invariant applies."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_f32x4_streaming(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_ld_f32x4_streaming_fn(
        const float* ptr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    const uint4 value = __ldcs(reinterpret_cast<const uint4*>(ptr));
    *r0 = value.x; *r1 = value.y; *r2 = value.z; *r3 = value.w;
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_ld_f32x4_streaming_fn({mem} + {index}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_f32x4_streaming)
def gmem_ld_f32x4_streaming(mem, index, r0, r1, r2, r3):
    """Load four aligned contiguous FP32 values from GMEM into four uint32
    registers with the streaming (evict-first, .cs) cache policy. Bits moved
    are identical to :func:`gmem_ld_f32x4`; the same 16-byte alignment
    invariant applies."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_f32x4_cg(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_f32x4_cg_fn(
        float* ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    __stcg(reinterpret_cast<uint4*>(ptr), uint4{r0, r1, r2, r3});
}
"""
    return Builtin(body=body, includes=[], return_val=f"gmem_st_f32x4_cg_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_f32x4_cg)
def gmem_st_f32x4_cg(mem, index, r0, r1, r2, r3):
    """Store four uint32 register bit patterns as aligned contiguous FP32
    values in GMEM with the L2-only (.cg) cache policy. Same operand
    invariants as :func:`gmem_st_f32x4`."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_f32x4_streaming(mem, index, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void gmem_st_f32x4_streaming_fn(
        float* ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    __stcs(reinterpret_cast<uint4*>(ptr), uint4{r0, r1, r2, r3});
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"gmem_st_f32x4_streaming_fn({mem} + {index}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_f32x4_streaming)
def gmem_st_f32x4_streaming(mem, index, r0, r1, r2, r3):
    """Store four uint32 register bit patterns as aligned contiguous FP32
    values in GMEM with the streaming (evict-first, .cs) cache policy (no L2
    retention; good for write-once output streams). Same operand invariants
    as :func:`gmem_st_f32x4`."""
    raise RuntimeError("should not be called in compilation")


def _bf16x8_load_body(policy):
    if policy == "cg":
        load = "__ldcg(reinterpret_cast<const uint4*>(ptr))"
    elif policy == "streaming":
        load = "__ldcs(reinterpret_cast<const uint4*>(ptr))"
    else:
        load = "*reinterpret_cast<const uint4*>(ptr)"
    name = {
        "cg": "gmem_ld_bf16x8_cg_fn",
        "streaming": "gmem_ld_bf16x8_streaming_fn",
    }.get(policy, "gmem_ld_bf16x8_fn")
    body = f"""
__device__ __forceinline__ void {name}(
        const void* ptr, uint64_t* r0, uint64_t* r1) {{
    const uint4 value = {load};
    *r0 = (uint64_t)value.x | ((uint64_t)value.y << 32);
    *r1 = (uint64_t)value.z | ((uint64_t)value.w << 32);
}}
"""
    return name, body


def _codegen_gmem_ld_bf16x8(policy, mem, index, r0, r1):
    name, body = _bf16x8_load_body(policy)
    return Builtin(body=body, includes=[], return_val=f"{name}({mem} + {index}, &{r0}, &{r1})")


def codegen_gmem_ld_bf16x8(mem, index, r0, r1):
    return _codegen_gmem_ld_bf16x8(None, mem, index, r0, r1)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_bf16x8)
def gmem_ld_bf16x8(mem, index, r0, r1):
    """Load eight aligned contiguous BF16 values (16 bytes) from GMEM into
    two uint64 register bit patterns (r0 = elements 0-3, r1 = elements 4-7).

    Naming contract: the plain name carries this uint64-pair
    contract so the whole policy family (plain/.cg/.streaming) shares one
    signature; extract lanes with vec_bf16x8_lane. The widening 8 x FP32-out
    load remains available as gmem_ld_bf16x8_f32x8, the raw 4 x uint32
    bit-pattern load as gmem_ld_bf16x8_u32x4. index is in 16-bit elements;
    the caller guarantees 16-byte alignment of mem + index (same invariant
    as gmem_ld_f32x4)."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_bf16x8_cg(mem, index, r0, r1):
    return _codegen_gmem_ld_bf16x8("cg", mem, index, r0, r1)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_bf16x8_cg)
def gmem_ld_bf16x8_cg(mem, index, r0, r1):
    """Load eight aligned contiguous BF16 values (16 bytes) from GMEM into
    two uint64 register bit patterns with the L2-only (.cg) cache policy.
    Bits moved are identical to :func:`gmem_ld_bf16x8` — only the cache
    behavior differs; the same 16-byte alignment invariant applies."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_bf16x8_streaming(mem, index, r0, r1):
    return _codegen_gmem_ld_bf16x8("streaming", mem, index, r0, r1)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_bf16x8_streaming)
def gmem_ld_bf16x8_streaming(mem, index, r0, r1):
    """Load eight aligned contiguous BF16 values (16 bytes) from GMEM into
    two uint64 register bit patterns with the streaming (evict-first, .cs)
    cache policy. Bits moved are identical to :func:`gmem_ld_bf16x8`; the
    same 16-byte alignment invariant applies."""
    raise RuntimeError("should not be called in compilation")


def _bf16x8_store_body(policy):
    if policy == "cg":
        store = "__stcg(reinterpret_cast<uint4*>(ptr), value)"
    elif policy == "streaming":
        store = "__stcs(reinterpret_cast<uint4*>(ptr), value)"
    else:
        store = "*reinterpret_cast<uint4*>(ptr) = value"
    name = {
        "cg": "gmem_st_bf16x8_cg_fn",
        "streaming": "gmem_st_bf16x8_streaming_fn",
    }.get(policy, "gmem_st_bf16x8_fn")
    body = f"""
__device__ __forceinline__ void {name}(
        void* ptr, float f0, float f1, float f2, float f3,
        float f4, float f5, float f6, float f7) {{
    const __nv_bfloat162 w0 = __floats2bfloat162_rn(f0, f1);
    const __nv_bfloat162 w1 = __floats2bfloat162_rn(f2, f3);
    const __nv_bfloat162 w2 = __floats2bfloat162_rn(f4, f5);
    const __nv_bfloat162 w3 = __floats2bfloat162_rn(f6, f7);
    uint4 value;
    value.x = (uint32_t)__bfloat16_as_ushort(w0.x) | ((uint32_t)__bfloat16_as_ushort(w0.y) << 16);
    value.y = (uint32_t)__bfloat16_as_ushort(w1.x) | ((uint32_t)__bfloat16_as_ushort(w1.y) << 16);
    value.z = (uint32_t)__bfloat16_as_ushort(w2.x) | ((uint32_t)__bfloat16_as_ushort(w2.y) << 16);
    value.w = (uint32_t)__bfloat16_as_ushort(w3.x) | ((uint32_t)__bfloat16_as_ushort(w3.y) << 16);
    {store};
}}
"""
    return name, body


def _codegen_gmem_st_bf16x8(policy, mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    name, body = _bf16x8_store_body(policy)
    return Builtin(body=body, includes=[],
                   return_val=f"{name}({mem} + {index}, {f0}, {f1}, {f2}, {f3}, {f4}, {f5}, {f6}, {f7})")


def codegen_gmem_st_bf16x8(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    return _codegen_gmem_st_bf16x8(None, mem, index, f0, f1, f2, f3, f4, f5, f6, f7)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_bf16x8)
def gmem_st_bf16x8(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """RNE-pack eight fp32 values into eight BF16 and store them as one
    aligned 16-byte block in GMEM (default cache policy).

    Naming contract: the plain name carries this pack contract so
    the whole policy family (plain/.cg/.streaming) shares one signature.
    The raw 4 x uint32 bit-pattern store remains available as
    gmem_st_bf16x8_u32x4; the sibling widening spelling as
    gmem_st_f32_to_bf16x8. index is in 16-bit elements; the caller
    guarantees 16-byte alignment of mem + index (same invariant as
    gmem_st_f32x4)."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_bf16x8_cg(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    return _codegen_gmem_st_bf16x8("cg", mem, index, f0, f1, f2, f3, f4, f5, f6, f7)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_bf16x8_cg)
def gmem_st_bf16x8_cg(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """RNE-pack eight fp32 values into eight BF16 and store them as one
    aligned 16-byte block in GMEM with the L2-only (.cg) cache policy. Same
    operand invariants as :func:`gmem_st_bf16x8`."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_bf16x8_streaming(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    return _codegen_gmem_st_bf16x8("streaming", mem, index, f0, f1, f2, f3, f4, f5, f6, f7)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_bf16x8_streaming)
def gmem_st_bf16x8_streaming(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """RNE-pack eight fp32 values into eight BF16 and store them as one
    aligned 16-byte block in GMEM with the streaming (evict-first, .cs)
    cache policy (no L2 retention; good for write-once output streams).
    Same operand invariants as :func:`gmem_st_bf16x8`."""
    raise RuntimeError("should not be called in compilation")


def codegen_vec_bf16x8_lane(lo, hi, lane):
    """Extract one statically-selected BF16 lane of a bf16x8 load pair as fp32."""
    lane_str = str(lane).strip().strip("()").strip()
    if lane_str not in tuple(str(k) for k in range(8)):
        raise ValueError("vec_bf16x8_lane requires a compile-time literal lane in 0..7 "
                         f"(got {lane!r}); select lanes with unrolled code, not dynamic "
                         "indices, because a register pair has no addressable storage")
    body = """
__device__ __forceinline__ float vec_bf16x8_lane_fn(uint64_t lo, uint64_t hi, int lane) {
    const uint32_t word = (lane < 4) ? ((lane < 2) ? (uint32_t)lo : (uint32_t)(lo >> 32))
                                     : ((lane < 6) ? (uint32_t)hi : (uint32_t)(hi >> 32));
    const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&word);
    const float2 f = __bfloat1622float2(pair);
    return (lane & 1) ? f.y : f.x;
}
"""
    return Builtin(body=body, includes=[], return_val=f"vec_bf16x8_lane_fn({lo}, {hi}, {lane_str})")


@builtin(eval_return_type=float32, codegen_func=codegen_vec_bf16x8_lane)
def vec_bf16x8_lane(lo, hi, lane):
    """Extract lane ``lane`` (a compile-time literal in 0..7) of a bf16x8
    load pair as fp32: lanes 0-3 come from the ``lo`` register, lanes 4-7
    from ``hi``, matching :func:`gmem_ld_bf16x8`'s packing. Dynamic lane
    indices are illegal."""
    raise RuntimeError("should not be called in compilation")


def _f16x8_load_body(policy):
    if policy == "cg":
        load = "__ldcg(reinterpret_cast<const uint4*>(ptr))"
    elif policy == "streaming":
        load = "__ldcs(reinterpret_cast<const uint4*>(ptr))"
    else:
        load = "*reinterpret_cast<const uint4*>(ptr)"
    name = {
        "cg": "gmem_ld_f16x8_cg_fn",
        "streaming": "gmem_ld_f16x8_streaming_fn",
    }.get(policy, "gmem_ld_f16x8_fn")
    body = f"""
__device__ __forceinline__ void {name}(
        const void* ptr, uint64_t* r0, uint64_t* r1) {{
    const uint4 value = {load};
    *r0 = (uint64_t)value.x | ((uint64_t)value.y << 32);
    *r1 = (uint64_t)value.z | ((uint64_t)value.w << 32);
}}
"""
    return name, body


def _codegen_gmem_ld_f16x8(policy, mem, index, r0, r1):
    name, body = _f16x8_load_body(policy)
    return Builtin(body=body, includes=[], return_val=f"{name}({mem} + {index}, &{r0}, &{r1})")


def codegen_gmem_ld_f16x8(mem, index, r0, r1):
    return _codegen_gmem_ld_f16x8(None, mem, index, r0, r1)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_f16x8)
def gmem_ld_f16x8(mem, index, r0, r1):
    """Load eight aligned contiguous FP16 values (16 bytes) from GMEM into
    two uint64 register bit patterns (r0 = elements 0-3, r1 = elements 4-7).

    Naming contract: the plain name carries this uint64-pair
    contract so the whole policy family (plain/.cg/.streaming) shares one
    signature; extract lanes with vec_f16x8_lane. The widening 8 x FP32-out
    load remains available as gmem_ld_f16x8_f32x8, the raw 4 x uint32
    bit-pattern load as gmem_ld_f16x8_u32x4. index is in 16-bit elements;
    the caller guarantees 16-byte alignment of mem + index (same invariant
    as gmem_ld_f32x4)."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_f16x8_cg(mem, index, r0, r1):
    return _codegen_gmem_ld_f16x8("cg", mem, index, r0, r1)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_f16x8_cg)
def gmem_ld_f16x8_cg(mem, index, r0, r1):
    """Load eight aligned contiguous FP16 values (16 bytes) from GMEM into
    two uint64 register bit patterns with the L2-only (.cg) cache policy.
    Bits moved are identical to :func:`gmem_ld_f16x8` — only the cache
    behavior differs; the same 16-byte alignment invariant applies."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_f16x8_streaming(mem, index, r0, r1):
    return _codegen_gmem_ld_f16x8("streaming", mem, index, r0, r1)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_f16x8_streaming)
def gmem_ld_f16x8_streaming(mem, index, r0, r1):
    """Load eight aligned contiguous FP16 values (16 bytes) from GMEM into
    two uint64 register bit patterns with the streaming (evict-first, .cs)
    cache policy. Bits moved are identical to :func:`gmem_ld_f16x8`; the
    same 16-byte alignment invariant applies."""
    raise RuntimeError("should not be called in compilation")


def _f16x8_store_body(policy):
    if policy == "cg":
        store = "__stcg(reinterpret_cast<uint4*>(ptr), value)"
    elif policy == "streaming":
        store = "__stcs(reinterpret_cast<uint4*>(ptr), value)"
    else:
        store = "*reinterpret_cast<uint4*>(ptr) = value"
    name = {
        "cg": "gmem_st_f16x8_cg_fn",
        "streaming": "gmem_st_f16x8_streaming_fn",
    }.get(policy, "gmem_st_f16x8_fn")
    body = f"""
__device__ __forceinline__ void {name}(
        void* ptr, float f0, float f1, float f2, float f3,
        float f4, float f5, float f6, float f7) {{
    const __half2 w0 = __floats2half2_rn(f0, f1);
    const __half2 w1 = __floats2half2_rn(f2, f3);
    const __half2 w2 = __floats2half2_rn(f4, f5);
    const __half2 w3 = __floats2half2_rn(f6, f7);
    uint4 value;
    value.x = (uint32_t)__half_as_ushort(w0.x) | ((uint32_t)__half_as_ushort(w0.y) << 16);
    value.y = (uint32_t)__half_as_ushort(w1.x) | ((uint32_t)__half_as_ushort(w1.y) << 16);
    value.z = (uint32_t)__half_as_ushort(w2.x) | ((uint32_t)__half_as_ushort(w2.y) << 16);
    value.w = (uint32_t)__half_as_ushort(w3.x) | ((uint32_t)__half_as_ushort(w3.y) << 16);
    {store};
}}
"""
    return name, body


def _codegen_gmem_st_f16x8(policy, mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    name, body = _f16x8_store_body(policy)
    return Builtin(body=body, includes=[],
                   return_val=f"{name}({mem} + {index}, {f0}, {f1}, {f2}, {f3}, {f4}, {f5}, {f6}, {f7})")


def codegen_gmem_st_f16x8(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    return _codegen_gmem_st_f16x8(None, mem, index, f0, f1, f2, f3, f4, f5, f6, f7)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_f16x8)
def gmem_st_f16x8(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """RNE-pack eight fp32 values into eight FP16 and store them as one
    aligned 16-byte block in GMEM (default cache policy).

    Naming contract: the plain name carries this pack contract so
    the whole policy family (plain/.cg/.streaming) shares one signature.
    The raw 4 x uint32 bit-pattern store remains available as
    gmem_st_f16x8_u32x4; the sibling widening spelling as
    gmem_st_f32_to_f16x8. index is in 16-bit elements; the caller
    guarantees 16-byte alignment of mem + index (same invariant as
    gmem_st_f32x4)."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_f16x8_cg(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    return _codegen_gmem_st_f16x8("cg", mem, index, f0, f1, f2, f3, f4, f5, f6, f7)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_f16x8_cg)
def gmem_st_f16x8_cg(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """RNE-pack eight fp32 values into eight FP16 and store them as one
    aligned 16-byte block in GMEM with the L2-only (.cg) cache policy. Same
    operand invariants as :func:`gmem_st_f16x8`."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_f16x8_streaming(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    return _codegen_gmem_st_f16x8("streaming", mem, index, f0, f1, f2, f3, f4, f5, f6, f7)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_f16x8_streaming)
def gmem_st_f16x8_streaming(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """RNE-pack eight fp32 values into eight FP16 and store them as one
    aligned 16-byte block in GMEM with the streaming (evict-first, .cs)
    cache policy (no L2 retention; good for write-once output streams).
    Same operand invariants as :func:`gmem_st_f16x8`."""
    raise RuntimeError("should not be called in compilation")


def codegen_vec_f16x8_lane(lo, hi, lane):
    """Extract one statically-selected FP16 lane of a f16x8 load pair as fp32."""
    lane_str = str(lane).strip().strip("()").strip()
    if lane_str not in tuple(str(k) for k in range(8)):
        raise ValueError("vec_f16x8_lane requires a compile-time literal lane in 0..7 "
                         f"(got {lane!r}); select lanes with unrolled code, not dynamic "
                         "indices, because a register pair has no addressable storage")
    body = """
__device__ __forceinline__ float vec_f16x8_lane_fn(uint64_t lo, uint64_t hi, int lane) {
    const uint32_t word = (lane < 4) ? ((lane < 2) ? (uint32_t)lo : (uint32_t)(lo >> 32))
                                     : ((lane < 6) ? (uint32_t)hi : (uint32_t)(hi >> 32));
    const __half2 pair = *reinterpret_cast<const __half2*>(&word);
    const float2 f = __half22float2(pair);
    return (lane & 1) ? f.y : f.x;
}
"""
    return Builtin(body=body, includes=[], return_val=f"vec_f16x8_lane_fn({lo}, {hi}, {lane_str})")


@builtin(eval_return_type=float32, codegen_func=codegen_vec_f16x8_lane)
def vec_f16x8_lane(lo, hi, lane):
    """Extract lane ``lane`` (a compile-time literal in 0..7) of a f16x8
    load pair as fp32: lanes 0-3 come from the ``lo`` register, lanes 4-7
    from ``hi``, matching :func:`gmem_ld_f16x8`'s packing. Dynamic lane
    indices are illegal."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Opaque 8-bit (x16) 16-byte global vector moves: gmem_ld_u8x16 / gmem_st_u8x16
# ==============================================================================
#
# Completion of the typed 16-byte GMEM vector family for ONE-BYTE element
# dtypes.  The existing twins cover 32-bit (f32x4/s32x4/u32x4), 64-bit
# (f64x2), and 16-bit (bf16x8/f16x8) elements; an 8-bit-element tensor
# (fp8_e4m3 / fp8_e5m2 / uint8 / int8 payload storage, all transported as
# byte-addressed ``uint8_t*``) had NO load twin at all and only a
# converting STORE (gmem_st_f32_to_e4m3x16 packs 16 FP32 -> 16 fp8 bytes).
# Kernels that MOVE 8-bit payloads bit-exactly (e.g. an index-table gather
# of fp8 rows plus their packed scale-factor rows, where no arithmetic is
# performed on the payload bytes) therefore had to hand-roll asm or split
# the 16-byte vector into four u32 moves.  These builtins move 16 aligned
# contiguous ONE-BYTE elements (one 16-byte transaction) through two uint64
# register bit patterns with NO numeric conversion: bits in == bits out, so
# any 8-bit storage dtype (both fp8 encodings, signed and unsigned bytes)
# is served by the same opaque move.  Generic: no task, shape, or layout
# assumption; ``index`` is in elements (== bytes) of ``mem``'s dtype, so
# row/permute address arithmetic stays in the frontend.
#
# Legality contract (identical to gmem_ld_f32x4 / gmem_ld_bf16x8):
#   * ``mem``   -- any global Tensor of an 8-bit element dtype.
#   * ``index`` -- element offset (== byte offset for 1-byte dtypes); the
#                  address mem + index MUST be 16-byte aligned and the whole
#                  16-byte vector in-bounds.  This is a hard PTX legality
#                  precondition (a misaligned or ragged-tail address is a
#                  fault); handle unaligned heads and ragged tails on a
#                  separate scalar path.
#   * ``r0``/``r1`` -- uint64 SCALARS.  For loads, r0 receives bytes
#                  [0,8) of the vector (r0 bit 8k+j = byte k bit j) and r1
#                  bytes [8,16); stores consume the same packing, so
#                  ld -> st through the pair is a bit-exact copy.
#
# Load cache-policy twins: default (plain), _cg (L2-only), _streaming
# (.cs evict-first), and _nc (__ldg read-only path; frontend asserts the
# loaded bytes are unwritten for the launch, e.g. a gathered read-only
# source table).  Stores: default / _cg / _streaming (PTX has no st.nc).
#
# Usage (frontend, bit-exact 16 B row fragment move):
#     lo: ll.uint64 = 0
#     hi: ll.uint64 = 0
#     ll.gmem_ld_u8x16(storage, row * H + v, lo, hi)
#     ll.gmem_st_u8x16(out, out_row * H + v, lo, hi)


def _u8x16_load_body(policy):
    if policy == "cg":
        load = "__ldcg(reinterpret_cast<const uint4*>(ptr))"
    elif policy == "streaming":
        load = "__ldcs(reinterpret_cast<const uint4*>(ptr))"
    elif policy == "nc":
        load = "__ldg(reinterpret_cast<const uint4*>(ptr))"
    else:
        load = "*reinterpret_cast<const uint4*>(ptr)"
    name = {
        "cg": "gmem_ld_u8x16_cg_fn",
        "streaming": "gmem_ld_u8x16_streaming_fn",
        "nc": "gmem_ld_u8x16_nc_fn",
    }.get(policy, "gmem_ld_u8x16_fn")
    body = f"""
__device__ __forceinline__ void {name}(
        const void* ptr, uint64_t* r0, uint64_t* r1) {{
    const uint4 value = {load};
    *r0 = (uint64_t)value.x | ((uint64_t)value.y << 32);
    *r1 = (uint64_t)value.z | ((uint64_t)value.w << 32);
}}
"""
    return name, body


def _codegen_gmem_ld_u8x16(policy, mem, index, r0, r1):
    name, body = _u8x16_load_body(policy)
    return Builtin(body=body, includes=[], return_val=f"{name}({mem} + {index}, &{r0}, &{r1})")


def codegen_gmem_ld_u8x16(mem, index, r0, r1):
    return _codegen_gmem_ld_u8x16(None, mem, index, r0, r1)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_u8x16)
def gmem_ld_u8x16(mem, index, r0, r1):
    """Load sixteen aligned contiguous ONE-BYTE elements (16 bytes) from
    GMEM into two uint64 register bit patterns (r0 = elements 0-7, r1 =
    elements 8-15) with NO numeric conversion.

    Opaque-move twin of gmem_ld_bf16x8 / gmem_ld_f16x8 for 8-bit element
    dtypes (fp8_e4m3/fp8_e5m2/uint8/int8 storage): index is in elements
    (== bytes); the caller guarantees 16-byte alignment of mem + index
    (same invariant as gmem_ld_f32x4). Extract a raw byte with
    vec_u8x16_lane; pair with gmem_st_u8x16 for a bit-exact 16-byte move.
    """
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_u8x16_cg(mem, index, r0, r1):
    return _codegen_gmem_ld_u8x16("cg", mem, index, r0, r1)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_u8x16_cg)
def gmem_ld_u8x16_cg(mem, index, r0, r1):
    """Load sixteen aligned contiguous ONE-BYTE elements (16 bytes) from
    GMEM into two uint64 register bit patterns with the L2-only (.cg) cache
    policy.  Bits moved are identical to :func:`gmem_ld_u8x16` — only the
    cache behavior differs; the same 16-byte alignment invariant applies."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_u8x16_streaming(mem, index, r0, r1):
    return _codegen_gmem_ld_u8x16("streaming", mem, index, r0, r1)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_u8x16_streaming)
def gmem_ld_u8x16_streaming(mem, index, r0, r1):
    """Load sixteen aligned contiguous ONE-BYTE elements (16 bytes) from
    GMEM into two uint64 register bit patterns with the streaming
    (evict-first, .cs) cache policy.  Bits moved are identical to
    :func:`gmem_ld_u8x16`; the same 16-byte alignment invariant applies."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_ld_u8x16_nc(mem, index, r0, r1):
    return _codegen_gmem_ld_u8x16("nc", mem, index, r0, r1)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_ld_u8x16_nc)
def gmem_ld_u8x16_nc(mem, index, r0, r1):
    """Load sixteen aligned contiguous ONE-BYTE elements (16 bytes) from
    GMEM into two uint64 register bit patterns through the read-only data
    cache (.nc / ``__ldg``).  Bits moved are identical to
    :func:`gmem_ld_u8x16` — only the cache path differs; the same 16-byte
    alignment invariant applies.

    Read-only legality contract (same as gmem_ld_v4f32_nc /
    gmem_ld_v8bf16_nc): the loaded 16 bytes must not be written by any
    thread for the duration of the launch.  The FRONTEND asserts this
    (e.g. a gathered source table whose rows are only read while the
    outputs are disjoint fresh allocations); the compiler performs no
    aliasing analysis.  Duplicate gathered rows make repeated reads of the
    same line common, which is exactly the pattern the read-only path is
    for."""
    raise RuntimeError("should not be called in compilation")


def _u8x16_store_body(policy):
    if policy == "cg":
        store = "__stcg(reinterpret_cast<uint4*>(ptr), value)"
    elif policy == "streaming":
        store = "__stcs(reinterpret_cast<uint4*>(ptr), value)"
    else:
        store = "*reinterpret_cast<uint4*>(ptr) = value"
    name = {
        "cg": "gmem_st_u8x16_cg_fn",
        "streaming": "gmem_st_u8x16_streaming_fn",
    }.get(policy, "gmem_st_u8x16_fn")
    body = f"""
__device__ __forceinline__ void {name}(
        void* ptr, uint64_t r0, uint64_t r1) {{
    uint4 value;
    value.x = (uint32_t)r0;
    value.y = (uint32_t)(r0 >> 32);
    value.z = (uint32_t)r1;
    value.w = (uint32_t)(r1 >> 32);
    {store};
}}
"""
    return name, body


def _codegen_gmem_st_u8x16(policy, mem, index, r0, r1):
    name, body = _u8x16_store_body(policy)
    return Builtin(body=body, includes=[], return_val=f"{name}({mem} + {index}, {r0}, {r1})")


def codegen_gmem_st_u8x16(mem, index, r0, r1):
    return _codegen_gmem_st_u8x16(None, mem, index, r0, r1)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_u8x16)
def gmem_st_u8x16(mem, index, r0, r1):
    """Store two uint64 register bit patterns as sixteen aligned contiguous
    ONE-BYTE elements (one 16-byte block) in GMEM with NO numeric
    conversion (default cache policy).

    Opaque-move twin of :func:`gmem_ld_u8x16`: r0 supplies bytes [0,8) and
    r1 bytes [8,16) of the vector, so an ld -> st pair through the two
    registers is a bit-exact 16-byte copy of any 8-bit storage dtype
    (fp8_e4m3/fp8_e5m2/uint8/int8).  index is in elements (== bytes); the
    caller guarantees 16-byte alignment of mem + index (same invariant as
    gmem_st_f32x4).  For an FP32 -> fp8 quantizing store use
    gmem_st_f32_to_e4m3x16 / gmem_st_f32_to_e5m2x16 instead."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_u8x16_cg(mem, index, r0, r1):
    return _codegen_gmem_st_u8x16("cg", mem, index, r0, r1)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_u8x16_cg)
def gmem_st_u8x16_cg(mem, index, r0, r1):
    """Store two uint64 register bit patterns as sixteen aligned contiguous
    ONE-BYTE elements (one 16-byte block) in GMEM with the L2-only (.cg)
    cache policy.  Same operand invariants and packing as
    :func:`gmem_st_u8x16`."""
    raise RuntimeError("should not be called in compilation")


def codegen_gmem_st_u8x16_streaming(mem, index, r0, r1):
    return _codegen_gmem_st_u8x16("streaming", mem, index, r0, r1)


@builtin(eval_return_type=void, codegen_func=codegen_gmem_st_u8x16_streaming)
def gmem_st_u8x16_streaming(mem, index, r0, r1):
    """Store two uint64 register bit patterns as sixteen aligned contiguous
    ONE-BYTE elements (one 16-byte block) in GMEM with the streaming
    (evict-first, .cs) cache policy (no L2 retention; good for write-once
    output streams).  Same operand invariants and packing as
    :func:`gmem_st_u8x16`."""
    raise RuntimeError("should not be called in compilation")


def codegen_vec_u8x16_lane(lo, hi, lane):
    """Extract one statically-selected raw byte lane of a u8x16 load pair."""
    lane_str = str(lane).strip().strip("()").strip()
    if lane_str not in tuple(str(k) for k in range(16)):
        raise ValueError("vec_u8x16_lane requires a compile-time literal lane in 0..15 "
                         f"(got {lane!r}); select lanes with unrolled code, not dynamic "
                         "indices, because a register pair has no addressable storage")
    body = """
__device__ __forceinline__ uint32_t vec_u8x16_lane_fn(uint64_t lo, uint64_t hi, int lane) {
    const uint64_t reg = (lane < 8) ? lo : hi;
    return (uint32_t)((reg >> (8 * (lane & 7))) & 0xFFu);
}
"""
    return Builtin(body=body, includes=[], return_val=f"vec_u8x16_lane_fn({lo}, {hi}, {lane_str})")


@builtin(eval_return_type=uint32, codegen_func=codegen_vec_u8x16_lane)
def vec_u8x16_lane(lo, hi, lane):
    """Extract lane ``lane`` (a compile-time literal in 0..15) of a
    u8x16 load pair as the raw byte value (0..255) widened to uint32, with
    NO numeric conversion: lanes 0-7 come from the ``lo`` register, lanes
    8-15 from ``hi``, matching :func:`gmem_ld_u8x16`'s packing.  Converts
    of the raw byte (e.g. fp8 -> f32) remain the frontend's choice of
    existing cvt intrinsics.  Dynamic lane indices are illegal."""
    raise RuntimeError("should not be called in compilation")
