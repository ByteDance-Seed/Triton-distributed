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
Asynchronous global->shared copy intrinsics (cp.async) and waits.

RF10 canonical owner module; see little_kernel.backends.nv.common.language.intrin.contracts
for the shared operand-unit/lane/alignment/cache-policy/effect inventory.
"""

from little_kernel.language.builtin_base import builtin, Builtin, CODEGEN_NEEDS_TYPES_ATTR
from little_kernel.core.type_system import LLType, void, Tensor, int8, uint8, uint16, uint32, uint64, int32, bfloat16, float16, float32, u32x2, u32x4

# ==============================================================================
# Non-bulk cp.async (SM80 LDGSTS family) global -> shared staging
# ==============================================================================
#
# Generic contract (no task or shape assumptions):
#   * ``smem``         -- any LittleKernel shared tensor; ``gmem`` any global
#                         tensor.  Offsets are in elements of each tensor's own
#                         dtype (C++ pointer arithmetic), exactly like the
#                         ld/st_global_v4_u32 family above.
#   * Legality preconditions of PTX ``cp.async.ca/cg.shared.global [dst],
#     [src], 16`` (= cp-size 16, src-size 16, no zero-fill):
#       - byte addresses of (smem + smem_offset) and (gmem + gmem_offset)
#         MUST both be 16-byte aligned;
#       - smem destination must be in the CTA's shared window
#         (``__cvta_generic_to_shared`` truncates to the 32-bit shared
#         address; a non-shared generic pointer is a fault);
#       - ``cp.async.cg`` only supports cp-size 16; ``.ca`` here is fixed at
#         16 bytes as well.
#   * Completion model: the copy is asynchronous.  It becomes visible to the
#     issuing thread only after a matching ``cp_async_commit_group()`` and
#     ``cp_async_wait_group(n)`` covering it (or a fuller barrier such as
#     ``__syncthreads`` after the wait).  Reading the staged smem before the
#     group is retired is a data race, identical in kind to the TMA
#     expect_tx/wait discipline in intrin/tma.py.
#   * Cache policy: ``.ca`` allocates into L1 (operands reused soon),
#     ``.cg`` bypasses L1 (streamed, single-touch data).  Both are bit-exact;
#     the choice is a pure performance hint.
#
# Usage (frontend):
#     ll.cp_async_cg_16(s_x + off, x + goff)   # 16B: 8 bf16 / 4 f32
#     ll.cp_async_commit_group()
#     ll.cp_async_wait_group(0)              # all committed groups retired

# The cp.async copy intrinsics below use the raw-address form
# (smem_ptr, gmem_ptr).  Each public name is defined exactly once: the
# codegen callable and the public signature take the same argument list,
# and one name emits one stable helper body (frontend offsets are folded
# into the pointer expressions at the call site, which is plain C pointer
# arithmetic).
#
# Frontend programs use both the 2-arg pointer form and the 4-arg
# (smem, smem_offset, gmem, gmem_offset) element-offset form, so the four
# copy intrinsics accept both arities through _cp_async_dual -- same helper
# body either way.  cp_async_commit_group is the single group-commit owner;
# cp_async_wait_group is the per-immediate/template wait owner.


def _cp_async_code(policy, size):
    """Emit one named cp.async helper (dedup-safe: keyed on function name)."""
    fn = f"cp_async_{policy}_{size}_fn"
    body = (f"\n__device__ __forceinline__ void {fn}(void* smem_ptr, const void* gmem_ptr) {{\n"
            f"    const uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));\n"
            f'    asm volatile("cp.async.{policy}.shared.global [%0], [%1], {size};"\n'
            f'                 :: "r"(saddr), "l"(gmem_ptr) : "memory");\n'
            f"}}\n")
    return fn, body


def _cp_async_dual(fn, body, smem, a, gmem, b):
    """Shared arity dispatch for the cp.async copy intrinsics.

    Dual calling convention (same arity-split pattern as
    gmem_st_v4_u32's 5/6-arg split):
      2 args: (smem_ptr, gmem_ptr) -- already-formed pointers (e.g.
             ptr_byte_offset output).
      4 args: (smem, smem_offset, gmem, gmem_offset) -- element offsets
             folded into the pointers as plain C pointer arithmetic.
    Both conventions call the SAME policy+size-keyed helper, so the
    one-name/one-stable-body invariant of this module holds; codegen
    arguments are pre-visited source strings, so ``gmem is None``
    unambiguously marks the 2-arg form. Semantics of the two forms are
    identical when ``*_ptr == * + *_offset`` under C pointer arithmetic.
    """
    if gmem is None:
        return Builtin(body=body, includes=[], return_val=f"{fn}({smem}, {a})")
    return Builtin(body=body, includes=[], return_val=f"{fn}({smem} + {a}, {gmem} + {b})")


def codegen_cp_async_cg_16(smem, a, gmem=None, b=None):
    fn, body = _cp_async_code("cg", 16)
    return _cp_async_dual(fn, body, smem, a, gmem, b)


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_cg_16)
def cp_async_cg_16(smem, a, gmem=None, b=None):
    """Asynchronously copy 16 bytes from global to shared memory (cp.async.cg).

    Two equivalent conventions, selected purely by arity:
    ``cp_async_cg_16(smem_ptr, gmem_ptr)`` (raw pointers, 2 args) or
    ``cp_async_cg_16(smem, smem_offset, gmem, gmem_offset)`` (element
    offsets folded via C pointer arithmetic, 4 args).

    L2-only (bypasses L1) 16-byte per-thread async copy: the complementary
    staging primitive to TMA bulk copies for register-free multi-stage
    pipelines at sm_80+. Byte- and dtype-agnostic (pair with ptr_byte_offset
    and shared arrays from alloc_shared_memory / dynamic smem + slicing).
    Contract (frontend program's responsibility): sm_80+; the smem side
    identifies 16 in-bounds, 16-byte-aligned shared bytes; the gmem side
    identifies 16 in-bounds, 16-byte-aligned global bytes; no thread reads
    the staged smem until cp_async_commit_group + cp_async_wait_group and
    (for cross-thread visibility) a subsequent barrier complete. Completion
    is per-thread commit/wait-group based (this intrinsic carries no
    mbarrier).
    """
    raise RuntimeError("should not call cp_async_cg_16 in compilation")


def codegen_cp_async_ca_16(smem, a, gmem=None, b=None):
    fn, body = _cp_async_code("ca", 16)
    return _cp_async_dual(fn, body, smem, a, gmem, b)


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_ca_16)
def cp_async_ca_16(smem, a, gmem=None, b=None):
    """Asynchronously copy 16 bytes global->shared with .ca (cache at all levels).

    Same dual 2-arg/4-arg calling convention as cp_async_cg_16. Same
    contract as cp_async_cg_16 but allocates the source line in L1 as
    well: prefer when the staged source may be re-read or shares cache
    pressure with other hot data.
    """
    raise RuntimeError("should not call cp_async_ca_16 in compilation")


def codegen_cp_async_ca_8(smem, a, gmem=None, b=None):
    fn, body = _cp_async_code("ca", 8)
    return _cp_async_dual(fn, body, smem, a, gmem, b)


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_ca_8)
def cp_async_ca_8(smem, a, gmem=None, b=None):
    """Asynchronously copy 8 bytes global->shared (cp.async.ca, 8B granularity).

    Same dual 2-arg/4-arg calling convention as cp_async_cg_16. Same
    contract as cp_async_cg_16 with 8-byte size/alignment; cp.async
    sizes below 16 bytes only support the .ca policy.
    """
    raise RuntimeError("should not call cp_async_ca_8 in compilation")


def codegen_cp_async_ca_4(smem, a, gmem=None, b=None):
    fn, body = _cp_async_code("ca", 4)
    return _cp_async_dual(fn, body, smem, a, gmem, b)


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_ca_4)
def cp_async_ca_4(smem, a, gmem=None, b=None):
    """Asynchronously copy 4 bytes global->shared (cp.async.ca, 4B granularity).

    Same dual 2-arg/4-arg calling convention as cp_async_cg_16. Same
    contract as cp_async_cg_16 with 4-byte size/alignment; cp.async
    sizes below 16 bytes only support the .ca policy.
    """
    raise RuntimeError("should not call cp_async_ca_4 in compilation")


def codegen_cp_async_cg_16_pred(smem, smem_offset, gmem, gmem_offset, pred):
    body = ("\n__device__ __forceinline__ void cp_async_cg_16_pred_fn(\n"
            "        void* smem_ptr, const void* gmem_ptr, int32_t pred) {\n"
            "    const uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));\n"
            '    asm volatile("{\\n"\n'
            '        " .reg .pred p;\\n"\n'
            '        " setp.ne.b32 p, %2, 0;\\n"\n'
            '        " @p cp.async.cg.shared.global [%0], [%1], 16;\\n"\n'
            '        "}\\n"\n'
            '        :: "r"(saddr), "l"(gmem_ptr), "r"(pred) : "memory");\n'
            "}\n")
    return Builtin(body=body, includes=["<cstdint>"],
                   return_val=f"cp_async_cg_16_pred_fn({smem} + {smem_offset}, {gmem} + {gmem_offset}, {pred})")


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_cg_16_pred)
def cp_async_cg_16_pred(smem, smem_offset, gmem, gmem_offset, pred):
    """Predicated 16-byte cp.async: when pred==0 nothing is issued and the
    smem destination is left unchanged (stale bytes).

    Same alignment/offset/completion contract as cp_async_cg_16. Use for
    ragged-tail lanes where the consumer will not read the unstaged bytes
    (or where stale bytes were pre-initialized).
    """
    raise RuntimeError("should not be called in compilation")


def codegen_cp_async_cg_16_zfill(smem, a, gmem=None, b=None, src_size=None):
    body = ("\n__device__ __forceinline__ void cp_async_cg_16_zfill_fn("
            "void* smem_ptr, const void* gmem_ptr, uint32_t src_size) {\n"
            "    const uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));\n"
            '    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"\n'
            '                 :: "r"(saddr), "l"(gmem_ptr), "r"(src_size) : "memory");\n'
            "}\n")
    # Dual calling convention (same arity split as _cp_async_dual):
    #   3 args: (smem_ptr, gmem_ptr, src_size) -- already-formed pointers.
    #   5 args: (smem, smem_offset, gmem, gmem_offset, src_size) -- element
    #           offsets folded into the pointers as plain C pointer
    #           arithmetic, mirroring cp_async_cg_16_pred above.
    # Both conventions call the SAME src-size helper, so the
    # one-name/one-stable-body invariant of this module holds; codegen
    # arguments are pre-visited source strings, so ``src_size is None``
    # unambiguously marks the 3-arg form. Semantics of the two forms are
    # identical when ``*_ptr == * + *_offset`` under C pointer arithmetic.
    if src_size is None:
        return Builtin(body=body, includes=[], return_val=f"cp_async_cg_16_zfill_fn({smem}, {a}, {gmem})")
    return Builtin(body=body, includes=[],
                   return_val=f"cp_async_cg_16_zfill_fn({smem} + {a}, {gmem} + {b}, {src_size})")


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_cg_16_zfill)
def cp_async_cg_16_zfill(smem, a, gmem=None, b=None, src_size=None):
    """Async 16-byte global->shared copy with zero-filled tail (src-size form).

    Copies src_size bytes from gmem and fills the remaining 16 - src_size
    shared bytes with zeros: the generic ragged-edge staging form (tile tails
    at non-multiple-of-16 bounds need no scalar cleanup pass). Contract:
    cp_async_cg_16 contract plus 0 <= src_size <= 16 and the first src_size
    global bytes in bounds; the zero-filled tail makes no gmem access, so
    src_size = 0 is a pure smem-zeroing async op.

    Same dual 3-arg/5-arg calling convention as cp_async_cg_16's 2-arg/4-arg
    split: pass either already-formed pointers plus src_size, or base
    pointers plus element offsets plus src_size.
    """
    raise RuntimeError("should not call cp_async_cg_16_zfill in compilation")


# cp_async_commit_group / cp_async_wait_group are defined once, later in this
# module, alongside cp_async_16 (element-indexed cp.async family). That single
# definition accepts both Python-int immediates and runtime int32 counts, so
# it subsumes the switch-only form; keeping one definition avoids a silent
# re-import-time override.
def codegen_cp_async_wait_all():
    body = ("\n__device__ __forceinline__ void cp_async_wait_all_fn() {\n"
            '    asm volatile("cp.async.wait_all;\\n" ::: "memory");\n'
            "}\n")
    return Builtin(body=body, includes=[], return_val="cp_async_wait_all_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_wait_all)
def cp_async_wait_all():
    """Block this thread until all of its committed cp.async groups completed.

    Equivalent to cp_async_wait_group(0) but also waits on groups that were
    never committed (cp.async.wait_all covers both commit-groups and the
    generic async-group forms); use at pipeline teardown.
    """
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


def codegen_bf16x2_to_float2(bits, out, base):
    body = """
#include <cuda_bf16.h>
__device__ __forceinline__ void bf16x2_to_float2_fn(uint32_t bits, float* out) {
    const float2 pair = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits));
    out[0] = pair.x;
    out[1] = pair.y;
}
"""
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"bf16x2_to_float2_fn({bits}, {out} + {base})")


@builtin(eval_return_type=void, codegen_func=codegen_bf16x2_to_float2)
def bf16x2_to_float2(bits, out, base):
    """Unpack one uint32 bf16x2 bit container into local FP32 array
    elements out[base] (lo half) and out[base+1] (hi half)."""
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

# ==============================================================================
# Non-bulk cp.async (SM80 LDGSTS family) global -> shared staging
# ==============================================================================
#
# Generic contract (no task or shape assumptions):
#   * ``smem``         -- any LittleKernel shared tensor; ``gmem`` any global
#                         tensor.  Offsets are in elements of each tensor's own
#                         dtype (C++ pointer arithmetic), exactly like the
#                         ld/st_global_v4_u32 family above.
#   * Legality preconditions of PTX ``cp.async.ca/cg.shared.global [dst],
#     [src], 16`` (= cp-size 16, src-size 16, no zero-fill):
#       - byte addresses of (smem + smem_offset) and (gmem + gmem_offset)
#         MUST both be 16-byte aligned;
#       - smem destination must be in the CTA's shared window
#         (``__cvta_generic_to_shared`` truncates to the 32-bit shared
#         address; a non-shared generic pointer is a fault);
#       - ``cp.async.cg`` only supports cp-size 16; ``.ca`` here is fixed at
#         16 bytes as well.
#   * Completion model: the copy is asynchronous.  It becomes visible to the
#     issuing thread only after a matching ``cp_async_commit_group()`` and
#     ``cp_async_wait_group(n)`` covering it (or a fuller barrier such as
#     ``__syncthreads`` after the wait).  Reading the staged smem before the
#     group is retired is a data race, identical in kind to the TMA
#     expect_tx/wait discipline in intrin/tma.py.
#   * Cache policy: ``.ca`` allocates into L1 (operands reused soon),
#     ``.cg`` bypasses L1 (streamed, single-touch data).  Both are bit-exact;
#     the choice is a pure performance hint.
#
# Usage (frontend):
#     ll.cp_async_cg_16(s_x + off, x + goff)   # 16B: 8 bf16 / 4 f32
#     ll.cp_async_commit_group()
#     ll.cp_async_wait_group(0)              # all committed groups retired


def codegen_cp_async_commit_group():
    body = ("\n__device__ __forceinline__ void cp_async_commit_group_fn() {\n"
            '    asm volatile("cp.async.commit_group;\\n" ::: "memory");\n'
            "}\n")
    return Builtin(body=body, includes=[], return_val="cp_async_commit_group_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_commit_group)
def cp_async_commit_group():
    """Commit all prior uncommitted cp.async ops into a new group."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Non-bulk cp.async (per-thread async copy, SM80+)
# =============================================================================


def codegen_cp_async_16(smem_addr, gmem_ptr, goff):
    body = r'''
__device__ __forceinline__ void cp_async_16_fn(uint32_t smem_addr, const void* gptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(smem_addr), "l"(gptr) : "memory");
}
'''
    return Builtin(body=body, includes=[],
                   return_val=f"cp_async_16_fn({smem_addr}, (const void*)({gmem_ptr} + {goff}))")


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_16)
def cp_async_16(smem_addr, gmem_ptr, goff):
    """Issue one 16-byte non-bulk async copy from GMEM to SMEM (cp.async.cg).

    Args:
        smem_addr: uint32 shared-space address of the destination, e.g. from
            ll.cvta_generic_to_shared(smem_tensor + elem_offset).
        gmem_ptr: typed GMEM tensor/base pointer (any element dtype).
        goff: element index into gmem_ptr (same element-based addressing as
            gmem_ld_v4_u32).

    Legality contract (caller-owned, not compiler-checked): both addresses
    must be 16-byte aligned; the copy is asynchronous and unordered — the
    destination may only be read after a matching cp_async_commit_group /
    cp_async_wait_group sequence (and __syncthreads() when another thread
    consumes it). Semantics-equivalent to a plain 16-byte load+store pair
    once the wait has retired the group."""
    raise RuntimeError("should not be called in compilation")


# ONE shared helper block for every non-Python-int call form: the templated
# constexpr helper plus the runtime int overload, emitted atomically (see
# codegen_cp_async_wait_group docstring for why splitting them is unsafe).
_WAIT_GROUP_TEMPLATE_AND_SWITCH_BODY = """
template <int N>
__device__ __forceinline__ void cp_async_wait_group_fn() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__ void cp_async_wait_group_fn(int n) {
    switch (n) {
        case 0: asm volatile("cp.async.wait_group 0;\\n" ::: "memory"); break;
        case 1: asm volatile("cp.async.wait_group 1;\\n" ::: "memory"); break;
        case 2: asm volatile("cp.async.wait_group 2;\\n" ::: "memory"); break;
        case 3: asm volatile("cp.async.wait_group 3;\\n" ::: "memory"); break;
        case 4: asm volatile("cp.async.wait_group 4;\\n" ::: "memory"); break;
        case 5: asm volatile("cp.async.wait_group 5;\\n" ::: "memory"); break;
        case 6: asm volatile("cp.async.wait_group 6;\\n" ::: "memory"); break;
        case 7: asm volatile("cp.async.wait_group 7;\\n" ::: "memory"); break;
        default: asm volatile("cp.async.wait_group 0;\\n" ::: "memory"); break;
    }
}
"""


def codegen_cp_async_wait_group(n):
    """Emit one of three semantically identical forms, selected by how much
    is known about n at codegen time (all are cp.async.wait_group N):

    * Python-int n (direct API calls): per-value immediate helper
      cp_async_wait_group_<n>_fn with the literal baked into the PTX
      immediate (wgmma_wait_group precedent).  Outside [0, 7] raises
      ValueError at codegen instead of failing inside ptxas.
    * Numeric-string n (what the frontend machinery delivers for a literal,
      e.g. "0"): a C++ template helper cp_async_wait_group_fn<N>
      instantiated per distinct constant expression.
    * Any other string n (runtime int32 expression): cp_async_wait_group_fn(int)
      dispatching through a 0..7 switch because the PTX operand must be an
      immediate; out-of-range values clamp to 0 (wait-all), matching
      wgmma_wait_group.

    Both string-driven forms return ONE shared combined body containing the
    template helper AND the int overload (legal C++: template parameter vs
    function parameter).  The combined body is constant across all string
    arguments, so builtin-body dedup emits it exactly once per kernel; this
    matters because writeln_builtin additionally dedups by extracted helper
    name, and emitting the two same-named helpers in two separate bodies
    would silently drop the second one.
    """
    if isinstance(n, int):
        if not 0 <= n <= 7:
            raise ValueError(f"cp_async_wait_group immediate must be in [0, 7], got {n}")
        body = f"""
__device__ __forceinline__ void cp_async_wait_group_{n}_fn() {{
    asm volatile("cp.async.wait_group {n};\\n" ::: "memory");
}}
"""
        return Builtin(body=body, includes=[], return_val=f"cp_async_wait_group_{n}_fn()")
    try:
        n_val = int(str(n).strip())
    except (TypeError, ValueError):
        n_val = None
    if n_val is not None:
        if not 0 <= n_val <= 7:
            raise ValueError(f"cp_async_wait_group immediate must be in [0, 7], got {n_val}")
        return Builtin(body=_WAIT_GROUP_TEMPLATE_AND_SWITCH_BODY, includes=[],
                       return_val=f"cp_async_wait_group_fn<({n})>()")
    return Builtin(body=_WAIT_GROUP_TEMPLATE_AND_SWITCH_BODY, includes=[], return_val=f"cp_async_wait_group_fn({n})")


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_wait_group)
def cp_async_wait_group(n):
    """Wait until at most n committed cp.async groups of this thread are pending.

    n is a compile-time non-negative integer literal (PTX immediate operand,
    0..7) or a runtime int32 value.  Literals are emitted as a shared
    templated helper (frontend-delivered numeric strings) or a per-value
    immediate helper (direct Python-int calls); runtime values dispatch
    through a 0..7 switch (clamping to 0 = wait-all), matching the
    wgmma_wait_group precedent.  A non-literal value that is provably
    outside 0..7 still compiles (clamp); a literal outside 0..7 is a
    ValueError at codegen instead of failing inside ptxas."""
    raise RuntimeError("should not be called in compilation")
