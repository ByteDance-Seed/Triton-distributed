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
SM100 (Blackwell) UMMA/tcgen05 intrinsics.
Includes TMEM allocation/deallocation, UMMA (tcgen05.mma), tcgen05 commit/fences,
TMEM load, cta_group::1/2 TMA load, elect_one, and SM100 SMEM/instruction descriptors.
All implementations use standalone PTX (no CuTe/CUTLASS dependency).
"""

from little_kernel.language.builtin_base import builtin, Builtin
from little_kernel.core.memory_effects import effect, sync_effect, R_SMEM_TCGEN05_COMMIT, RW_TMEM_TCGEN05_COMMIT, W_TMEM_TCGEN05_COMMIT, RW_TMEM_TCGEN05_FENCE
from little_kernel.core.compile import ll_kernel
from little_kernel.core.type_system import void, uint32, uint64, float32
from little_kernel.backends.nv.common.language.intrin.cuda_asm import smem_st_128, smem_swizzle_128b_addr, pack_bf16, pack_bf16_float, uint_as_float

# ==============================================================================

# ==============================================================================
# TMEM allocation / deallocation
# ==============================================================================


def codegen_tmem_alloc_1sm(smem_addr_ptr, ncols):
    body = """
__device__ __forceinline__ void tmem_alloc_1sm_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}
"""
    return Builtin(body=body, includes=[], return_val=f"tmem_alloc_1sm_fn({smem_addr_ptr}, {ncols})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_alloc_1sm)
def tmem_alloc_1sm(smem_addr_ptr, ncols):
    """Allocate TMEM columns (cta_group::1). Result written to smem_addr_ptr."""
    raise RuntimeError("should not call tmem_alloc_1sm in compilation")


def codegen_tmem_dealloc_1sm(addr, ncols):
    body = """
__device__ __forceinline__ void tmem_dealloc_1sm_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}
"""
    return Builtin(body=body, includes=[], return_val=f"tmem_dealloc_1sm_fn({addr}, {ncols})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_dealloc_1sm)
def tmem_dealloc_1sm(addr, ncols):
    """Deallocate TMEM columns (cta_group::1)."""
    raise RuntimeError("should not call tmem_dealloc_1sm in compilation")


def codegen_tmem_relinquish_alloc_permit_1sm():
    body = """
__device__ __forceinline__ void tmem_relinquish_alloc_permit_1sm_fn() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="tmem_relinquish_alloc_permit_1sm_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_relinquish_alloc_permit_1sm)
def tmem_relinquish_alloc_permit_1sm():
    """Release the cta_group::1 TMEM allocation permit."""
    raise RuntimeError("should not call tmem_relinquish_alloc_permit_1sm in compilation")


def codegen_tmem_alloc_2sm(smem_addr_ptr, ncols):
    body = """
__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}
"""
    return Builtin(body=body, includes=[], return_val=f"tmem_alloc_fn({smem_addr_ptr}, {ncols})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_alloc_2sm)
def tmem_alloc_2sm(smem_addr_ptr, ncols):
    """Allocate TMEM columns (cta_group::2). Result written to smem_addr_ptr."""
    raise RuntimeError("should not call tmem_alloc_2sm in compilation")


def codegen_tmem_dealloc_2sm(addr, ncols):
    body = """
__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}
"""
    return Builtin(body=body, includes=[], return_val=f"tmem_dealloc_fn({addr}, {ncols})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_dealloc_2sm)
def tmem_dealloc_2sm(addr, ncols):
    """Deallocate TMEM columns (cta_group::2)."""
    raise RuntimeError("should not call tmem_dealloc_2sm in compilation")


# ==============================================================================
# TMEM load (tcgen05.ld)
# ==============================================================================


def codegen_tmem_ld_32dp_32bxn(col, *regs):
    repeat = len(regs)
    if repeat not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"unsupported tmem_ld_32dp_32bxn repeat {repeat}")
    args = ", ".join(f"&{reg}" for reg in regs)
    body = ""
    for rep in (1, 2, 4, 8, 16, 32):
        params = "".join(f", uint32_t* r{i}" for i in range(rep))
        outputs = ",".join(f"%{i}" for i in range(rep))
        constraints = ", ".join(f'"=r"(*r{i})' for i in range(rep))
        body += f"""
__device__ __forceinline__ void tmem_ld_32dp_32bxn_x{rep}_fn(uint32_t col{params}) {{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x{rep}.b32 {{{outputs}}}, [%{rep}];"
        : {constraints}
        : "r"(col));
}}
"""
    return Builtin(body=body, includes=[], return_val=f"tmem_ld_32dp_32bxn_x{repeat}_fn({col}, {args})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_ld_32dp_32bxn)
def tmem_ld_32dp_32bxn(col, *regs):
    """Load 32-dp TMEM values. The register count selects x1..x32 PTX."""
    raise RuntimeError("should not call tmem_ld_32dp_32bxn in compilation")


def codegen_tmem_ld_16dp_32bxn(col, *regs):
    repeat = len(regs)
    if repeat not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"unsupported tmem_ld_16dp_32bxn repeat {repeat}")
    args = ", ".join(f"&{reg}" for reg in regs)
    body = ""
    for rep in (1, 2, 4, 8, 16, 32):
        params = "".join(f", uint32_t* r{i}" for i in range(rep))
        outputs = ",".join(f"%{i}" for i in range(rep))
        constraints = ", ".join(f'"=r"(*r{i})' for i in range(rep))
        body += f"""
__device__ __forceinline__ void tmem_ld_16dp_32bxn_x{rep}_fn(uint32_t col{params}) {{
    asm volatile(
        "tcgen05.ld.sync.aligned.16x32bx2.x{rep}.b32 {{{outputs}}}, [%{rep}], {rep};"
        : {constraints}
        : "r"(col));
}}
"""
    return Builtin(body=body, includes=[], return_val=f"tmem_ld_16dp_32bxn_x{repeat}_fn({col}, {args})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_ld_16dp_32bxn)
def tmem_ld_16dp_32bxn(col, *regs):
    """Load the 16-dp x 32-bit-pair TMEM layout used by M=64 UMMA."""
    raise RuntimeError("should not call tmem_ld_16dp_32bxn in compilation")


def codegen_tmem_ld_16dp_256bx2(col, *regs):
    if len(regs) != 8:
        raise ValueError(f"tmem_ld_16dp_256bx2 requires 8 registers, got {len(regs)}")
    args = ", ".join(f"&{reg}" for reg in regs)
    params = "".join(f", uint32_t* r{i}" for i in range(8))
    outputs = ", ".join(f"%{i}" for i in range(8))
    constraints = ", ".join(f'"=r"(*r{i})' for i in range(8))
    body = f'''
__device__ __forceinline__ void tmem_ld_16dp_256bx2_fn(uint32_t col{params}) {{
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {{{outputs}}}, [%8];" : {constraints} : "r"(col));
}}
'''
    return Builtin(body=body, includes=[], return_val=f"tmem_ld_16dp_256bx2_fn({col}, {args})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_ld_16dp_256bx2)
def tmem_ld_16dp_256bx2(col, *regs):
    """Load one 16x256b.x2 TMEM fragment into eight registers."""
    raise RuntimeError("should not call tmem_ld_16dp_256bx2 in compilation")


def codegen_tmem_ld_16dp_256bx4(col, *regs):
    if len(regs) != 16:
        raise ValueError(f"tmem_ld_16dp_256bx4 requires 16 registers, got {len(regs)}")
    args = ", ".join(f"&{reg}" for reg in regs)
    params = "".join(f", uint32_t* r{i}" for i in range(16))
    outputs = ", ".join(f"%{i}" for i in range(16))
    constraints = ", ".join(f'"=r"(*r{i})' for i in range(16))
    body = f'''
__device__ __forceinline__ void tmem_ld_16dp_256bx4_fn(uint32_t col{params}) {{
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {{{outputs}}}, [%16];" : {constraints} : "r"(col));
}}
'''
    return Builtin(body=body, includes=[], return_val=f"tmem_ld_16dp_256bx4_fn({col}, {args})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_ld_16dp_256bx4)
def tmem_ld_16dp_256bx4(col, *regs):
    """Load one 16x256b.x4 TMEM fragment into sixteen registers."""
    raise RuntimeError("should not call tmem_ld_16dp_256bx4 in compilation")


def codegen_tmem_ld_fence():
    body = """
__device__ __forceinline__ void tmem_ld_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="tmem_ld_fence_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_ld_fence)
def tmem_ld_fence():
    """Fence for TMEM loads (tcgen05.wait::ld)."""
    raise RuntimeError("should not call tmem_ld_fence in compilation")


# ==============================================================================
# TMEM store (tcgen05.st)
# ==============================================================================


def codegen_tmem_st_32dp_32bxn(col, *regs):
    repeat = len(regs)
    if repeat not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"unsupported tmem_st_32dp_32bxn repeat {repeat}")
    reg_args = ", ".join(regs)
    body = ""
    for rep in (1, 2, 4, 8, 16, 32):
        params = "".join(f", uint32_t r{i}" for i in range(rep))
        operands = ", ".join(f"%{i + 1}" for i in range(rep))
        inputs = ", ".join(f'"r"(r{i})' for i in range(rep))
        body += f"""
__device__ __forceinline__ void tmem_st_32dp_32bxn_x{rep}_fn(uint32_t col{params}) {{
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x{rep}.b32 [%0], {{{operands}}};"
        :: "r"(col), {inputs} : "memory");
}}
"""
    return Builtin(
        body=body,
        includes=[],
        return_val=f"tmem_st_32dp_32bxn_x{repeat}_fn({col}, {reg_args})",
    )


@builtin(eval_return_type=void, codegen_func=codegen_tmem_st_32dp_32bxn)
def tmem_st_32dp_32bxn(col, *regs):
    """Store 32-dp TMEM values. The register count selects the PTX repetition."""
    raise RuntimeError("should not call tmem_st_32dp_32bxn in compilation")


def codegen_tmem_st_16dp_32bxn(col, *regs):
    repeat = len(regs)
    if repeat not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"unsupported tmem_st_16dp_32bxn repeat {repeat}")
    reg_args = ", ".join(regs)
    body = ""
    for rep in (1, 2, 4, 8, 16, 32):
        params = "".join(f", uint32_t r{i}" for i in range(rep))
        operands = ", ".join(f"%{i + 1}" for i in range(rep))
        inputs = ", ".join(f'"r"(r{i})' for i in range(rep))
        body += f"""
__device__ __forceinline__ void tmem_st_16dp_32bxn_x{rep}_fn(uint32_t col{params}) {{
    asm volatile(
        "tcgen05.st.sync.aligned.16x32bx2.x{rep}.b32 [%0], {rep}, {{{operands}}};"
        :: "r"(col), {inputs} : "memory");
}}
"""
    return Builtin(body=body, includes=[], return_val=f"tmem_st_16dp_32bxn_x{repeat}_fn({col}, {reg_args})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_st_16dp_32bxn)
def tmem_st_16dp_32bxn(col, *regs):
    """Store the 16-dp x 32-bit-pair TMEM layout used by M=64 UMMA."""
    raise RuntimeError("should not call tmem_st_16dp_32bxn in compilation")


def codegen_tmem_st_16dp_128bx2(col, *regs):
    if len(regs) != 4:
        raise ValueError(f"tmem_st_16dp_128bx2 requires 4 registers, got {len(regs)}")
    reg_args = ", ".join(regs)
    params = "".join(f", uint32_t r{i}" for i in range(4))
    operands = ", ".join(f"%{i + 1}" for i in range(4))
    inputs = ", ".join(f'"r"(r{i})' for i in range(4))
    body = f'''
__device__ __forceinline__ void tmem_st_16dp_128bx2_fn(uint32_t col{params}) {{
    asm volatile("tcgen05.st.sync.aligned.16x128b.x2.b32 [%0], {{{operands}}};"
                 :: "r"(col), {inputs} : "memory");
}}
'''
    return Builtin(body=body, includes=[], return_val=f"tmem_st_16dp_128bx2_fn({col}, {reg_args})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_st_16dp_128bx2)
def tmem_st_16dp_128bx2(col, *regs):
    """Store one PTX 16x128b.x2 TMEM fragment from four registers per thread."""
    raise RuntimeError("should not call tmem_st_16dp_128bx2 in compilation")


def codegen_tmem_st_fence():
    body = """
__device__ __forceinline__ void tmem_st_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="tmem_st_fence_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_st_fence)
def tmem_st_fence():
    """Fence for TMEM stores (tcgen05.wait::st)."""
    raise RuntimeError("should not call tmem_st_fence in compilation")


# ==============================================================================
# tcgen05 fences
# ==============================================================================


def codegen_tcgen05_fence_after():
    body = """
__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="tcgen05_fence_after_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_tcgen05_fence_after,
         effects=(sync_effect(RW_TMEM_TCGEN05_FENCE), ))
def tcgen05_fence_after():
    """tcgen05.fence::after_thread_sync — TMEM ordering boundary.

    Declared effect: ``sync_effect(RW_TMEM_TCGEN05_FENCE)`` — a THREADS-issued
    ordering boundary in TMEM space between asynchronous tcgen05 accesses
    (UMMA commits, tcgen05.ld/st) and surrounding thread synchronization.
    """
    raise RuntimeError("should not call tcgen05_fence_after in compilation")


def codegen_tcgen05_fence_before():
    body = """
__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="tcgen05_fence_before_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_tcgen05_fence_before,
         effects=(sync_effect(RW_TMEM_TCGEN05_FENCE), ))
def tcgen05_fence_before():
    """tcgen05.fence::before_thread_sync — TMEM ordering boundary.

    Declared effect: ``sync_effect(RW_TMEM_TCGEN05_FENCE)`` — the counterpart
    boundary required before CTA/cluster synchronization when a TMEM
    accumulator is about to be reused by a later asynchronous tcgen05 write.
    """
    raise RuntimeError("should not call tcgen05_fence_before in compilation")


# ==============================================================================
# SMEM -> TMEM copy (tcgen05.cp)
# ==============================================================================


def _codegen_tcgen05_cp_128x256b(tmem_dst, desc, cta_group):
    name = f"tcgen05_cp_128x256b_cg{cta_group}_fn"
    body = f"""
__device__ __forceinline__ void {name}(uint32_t tmem_dst, uint64_t desc) {{
    asm volatile("tcgen05.cp.cta_group::{cta_group}.128x256b [%0], %1;"
                 :: "r"(tmem_dst), "l"(desc) : "memory");
}}
"""
    return Builtin(body=body, includes=[], return_val=f"{name}({tmem_dst}, {desc})")


def codegen_tcgen05_cp_128x256b_1sm(tmem_dst, desc):
    return _codegen_tcgen05_cp_128x256b(tmem_dst, desc, 1)


@builtin(eval_return_type=void, codegen_func=codegen_tcgen05_cp_128x256b_1sm)
def tcgen05_cp_128x256b_1sm(tmem_dst, desc):
    """Copy one 128x256-bit SMEM tile to TMEM for cta_group::1."""
    raise RuntimeError("should not call tcgen05_cp_128x256b_1sm in compilation")


def codegen_tcgen05_cp_128x256b_2sm(tmem_dst, desc):
    return _codegen_tcgen05_cp_128x256b(tmem_dst, desc, 2)


@builtin(eval_return_type=void, codegen_func=codegen_tcgen05_cp_128x256b_2sm)
def tcgen05_cp_128x256b_2sm(tmem_dst, desc):
    """Copy one 128x256-bit SMEM tile to TMEM for cta_group::2."""
    raise RuntimeError("should not call tcgen05_cp_128x256b_2sm in compilation")


# ==============================================================================
# UMMA instruction (tcgen05.mma.cta_group::2.kind::f16)
# ==============================================================================


def codegen_umma_f16_1sm(tmem_c, desc_a, desc_b, idesc, accum):
    body = """
__device__ __forceinline__ void umma_f16_1sm_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\\n.reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\\n}\\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"umma_f16_1sm_fn({tmem_c}, {desc_a}, {desc_b}, {idesc}, {accum})")


@builtin(eval_return_type=void, codegen_func=codegen_umma_f16_1sm)
def umma_f16_ss_1sm(tmem_c, desc_a, desc_b, idesc, accum):
    """SM100 UMMA instruction: cta_group::1, kind::f16 (BF16×BF16→FP32).
    accum=0 clears accumulator, accum=1 accumulates."""
    raise RuntimeError("should not call umma_f16_ss_1sm in compilation")


def codegen_umma_f16_1sm_kblock(tmem_c, desc_a, desc_b, idesc, k, accum):
    body = """
template <int K>
__device__ __forceinline__ void umma_f16_1sm_kblock_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    static_assert(K > 0 && K % 16 == 0,
                  "BF16 UMMA K extent must be a positive multiple of 16");
    constexpr int KTiles = K / 16;
    asm volatile(
        "{\\n.reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\\n}\\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
#pragma unroll
    for (int k = 1; k < KTiles; ++k) {
        uint64_t cur_a = (desc_a & ~0x3FFFULL) | ((desc_a + k * 2) & 0x3FFFULL);
        uint64_t cur_b = (desc_b & ~0x3FFFULL) | ((desc_b + k * 2) & 0x3FFFULL);
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;"
            :: "r"(tmem_c), "l"(cur_a), "l"(cur_b), "r"(idesc));
    }
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"umma_f16_1sm_kblock_fn<{k}>({tmem_c}, {desc_a}, {desc_b}, {idesc}, {accum})")


@builtin(eval_return_type=void, codegen_func=codegen_umma_f16_1sm_kblock)
def umma_f16_ss_1sm_kblock(tmem_c, desc_a, desc_b, idesc, k, accum):
    """Issue a contiguous BF16 SS logical K extent; ``accum`` controls the first.

    The caller remains responsible for electing the issuing thread.
    """
    raise RuntimeError("should not call umma_f16_ss_1sm_kblock in compilation")


def codegen_umma_f16_ts_1sm(tmem_c, tmem_a, desc_b, idesc, accum):
    body = """
__device__ __forceinline__ void umma_f16_ts_1sm_fn(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\\n.reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\\n}\\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"umma_f16_ts_1sm_fn({tmem_c}, {tmem_a}, {desc_b}, {idesc}, {accum})")


@builtin(eval_return_type=void, codegen_func=codegen_umma_f16_ts_1sm)
def umma_f16_ts_1sm(tmem_c, tmem_a, desc_b, idesc, accum):
    """SM100 UMMA: cta_group::1, A from TMEM, B from SMEM, BF16×BF16→FP32."""
    raise RuntimeError("should not call umma_f16_ts_1sm in compilation")


def codegen_umma_f16_ts_2sm(tmem_c, tmem_a, desc_b, idesc, accum):
    body = """
__device__ __forceinline__ void umma_f16_ts_2sm_fn(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\\n.reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\\n}\\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"umma_f16_ts_2sm_fn({tmem_c}, {tmem_a}, {desc_b}, {idesc}, {accum})")


@builtin(eval_return_type=void, codegen_func=codegen_umma_f16_ts_2sm)
def umma_f16_ts_2sm(tmem_c, tmem_a, desc_b, idesc, accum):
    """SM100 UMMA: cta_group::2, A from TMEM, B from SMEM, BF16×BF16→FP32."""
    raise RuntimeError("should not call umma_f16_ts_2sm in compilation")


def _codegen_umma_f16_ts_1sm_kblock(tmem_c, tmem_a, desc_b, idesc, k, accum, desc_step16):
    body = """
template <int K, int DescStep16>
__device__ __forceinline__ void umma_f16_ts_1sm_kblock_fn(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    static_assert(K > 0 && K % 16 == 0,
                  "BF16 UMMA K extent must be a positive multiple of 16");
    constexpr int KTiles = K / 16;
    asm volatile(
        "{\\n.reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\\n}\\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
#pragma unroll
    for (int k = 1; k < KTiles; ++k) {
        uint64_t cur_b = (desc_b & ~0x3FFFULL) | ((desc_b + k * DescStep16) & 0x3FFFULL);
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, 1;"
            :: "r"(tmem_c), "r"(tmem_a + k * 8), "l"(cur_b), "r"(idesc));
    }
}
"""
    return Builtin(
        body=body, includes=[],
        return_val=f"umma_f16_ts_1sm_kblock_fn<{k}, {desc_step16}>({tmem_c}, {tmem_a}, {desc_b}, {idesc}, {accum})")


def codegen_umma_f16_ts_1sm_kblock(tmem_c, tmem_a, desc_b, idesc, k, accum):
    return _codegen_umma_f16_ts_1sm_kblock(tmem_c, tmem_a, desc_b, idesc, k, accum, 2)


@builtin(eval_return_type=void, codegen_func=codegen_umma_f16_ts_1sm_kblock)
def umma_f16_ts_1sm_kblock(tmem_c, tmem_a, desc_b, idesc, k, accum):
    """Issue a contiguous BF16 TS logical K extent; ``accum`` controls the first."""
    raise RuntimeError("should not call umma_f16_ts_1sm_kblock in compilation")


def codegen_umma_f16_ts_1sm_kblock_mn_sw128(tmem_c, tmem_a, desc_b, idesc, k, accum):
    # BF16 SW128 MN-major atom is [64, 8]; advancing K16 skips 64*16*2B.
    return _codegen_umma_f16_ts_1sm_kblock(tmem_c, tmem_a, desc_b, idesc, k, accum, 128)


@builtin(eval_return_type=void, codegen_func=codegen_umma_f16_ts_1sm_kblock_mn_sw128)
def umma_f16_ts_1sm_kblock_mn_sw128(tmem_c, tmem_a, desc_b, idesc, k, accum):
    """BF16 TS K-block with an SW128 MN-major SMEM B operand."""
    raise RuntimeError("should not call umma_f16_ts_1sm_kblock_mn_sw128 in compilation")


def codegen_umma_f16_2sm(tmem_c, desc_a, desc_b, idesc, accum):
    body = """
__device__ __forceinline__ void umma_f16_2sm_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\\n.reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\\n}\\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"umma_f16_2sm_fn({tmem_c}, {desc_a}, {desc_b}, {idesc}, {accum})")


@builtin(eval_return_type=void, codegen_func=codegen_umma_f16_2sm)
def umma_f16_ss_2sm(tmem_c, desc_a, desc_b, idesc, accum):
    """SM100 UMMA instruction: cta_group::2, kind::f16 (BF16×BF16→FP32).
    accum=0 clears accumulator, accum=1 accumulates."""
    raise RuntimeError("should not call umma_f16_ss_2sm in compilation")


def codegen_umma_f16_2sm_kblock(tmem_c, desc_a, desc_b, idesc, k, accum):
    body = """
template <int K>
__device__ __forceinline__ void umma_f16_2sm_kblock_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    static_assert(K > 0 && K % 16 == 0,
                  "BF16 UMMA K extent must be a positive multiple of 16");
    constexpr int KTiles = K / 16;
    asm volatile(
        "{\\n.reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\\n}\\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
#pragma unroll
    for (int k = 1; k < KTiles; ++k) {
        uint64_t cur_a = (desc_a & ~0x3FFFULL) | ((desc_a + k * 2) & 0x3FFFULL);
        uint64_t cur_b = (desc_b & ~0x3FFFULL) | ((desc_b + k * 2) & 0x3FFFULL);
        asm volatile(
            "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, 1;"
            :: "r"(tmem_c), "l"(cur_a), "l"(cur_b), "r"(idesc));
    }
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"umma_f16_2sm_kblock_fn<{k}>({tmem_c}, {desc_a}, {desc_b}, {idesc}, {accum})")


@builtin(eval_return_type=void, codegen_func=codegen_umma_f16_2sm_kblock)
def umma_f16_ss_2sm_kblock(tmem_c, desc_a, desc_b, idesc, k, accum):
    """Issue a contiguous two-CTA BF16 SS logical K extent; ``accum`` controls the first."""
    raise RuntimeError("should not call umma_f16_ss_2sm_kblock in compilation")


# ==============================================================================
# UMMA instruction (tcgen05.mma.cta_group::{1,2}.kind::tf32)
#
# A and B are read from shared memory as TF32-viewed f32 (instruction-descriptor
# type code 2), the TMEM accumulator is FP32 (idesc bit 4), and the K extent of a
# single PTX instruction is 8. ``accum=0`` clears the accumulator on the first
# tile; ``accum=1`` accumulates. The operands mirror the kind::f16 family.
# ==============================================================================


def codegen_encode_umma_tf32_idesc(m, n):
    body = """
__device__ __forceinline__ uint32_t encode_umma_tf32_idesc_fn(uint32_t m, uint32_t n) {
    // D = f32 (bit 4), A/B TF32 (type code 2 at bits 7..9 / 10..12), K-major,
    // N/8 at bits 17..22, M/16 at bits 24..28. K extent of one mma is 8.
    return (1u << 4) | (2u << 7) | (2u << 10) | ((n >> 3) << 17) | ((m >> 4) << 24);
}
"""
    return Builtin(body=body, includes=[], return_val=f"encode_umma_tf32_idesc_fn({m}, {n})")


@builtin(eval_return_type=uint32, codegen_func=codegen_encode_umma_tf32_idesc)
def encode_umma_tf32_idesc(m, n):
    """Encode the kind::tf32 UMMA instruction descriptor for an MxNxK8 tile."""
    raise RuntimeError("should not call encode_umma_tf32_idesc in compilation")


def codegen_umma_tf32_1sm(tmem_c, desc_a, desc_b, idesc, accum):
    body = """
__device__ __forceinline__ void umma_tf32_1sm_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\\n.reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p;\\n}\\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"umma_tf32_1sm_fn({tmem_c}, {desc_a}, {desc_b}, {idesc}, {accum})")


@builtin(
    eval_return_type=void,
    effects=(effect(RW_TMEM_TCGEN05_COMMIT, "tmem_c"), effect(R_SMEM_TCGEN05_COMMIT, "desc_a.mem"),
             effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")), codegen_func=codegen_umma_tf32_1sm, cta_group=1)
def umma_tf32_ss_1sm(tmem_c, desc_a, desc_b, idesc, accum):
    """SM100 UMMA: cta_group::1, kind::tf32 (TF32×TF32→FP32, K extent 8).
    accum=0 clears accumulator, accum=1 accumulates."""
    raise RuntimeError("should not call umma_tf32_ss_1sm in compilation")


def codegen_umma_tf32_2sm(tmem_c, desc_a, desc_b, idesc, accum):
    body = """
__device__ __forceinline__ void umma_tf32_2sm_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\\n.reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p;\\n}\\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"umma_tf32_2sm_fn({tmem_c}, {desc_a}, {desc_b}, {idesc}, {accum})")


@builtin(
    eval_return_type=void,
    effects=(effect(RW_TMEM_TCGEN05_COMMIT, "tmem_c"), effect(R_SMEM_TCGEN05_COMMIT, "desc_a.mem"),
             effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")), codegen_func=codegen_umma_tf32_2sm, cta_group=2)
def umma_tf32_ss_2sm(tmem_c, desc_a, desc_b, idesc, accum):
    """SM100 UMMA: cta_group::2, kind::tf32 (TF32×TF32→FP32, K extent 8).
    accum=0 clears accumulator, accum=1 accumulates."""
    raise RuntimeError("should not call umma_tf32_ss_2sm in compilation")


def _codegen_umma_tf32_ss_kblock(tmem_c, desc_a, desc_b, idesc, k, accum, cta_group):
    body = f"""
template <int K>
__device__ __forceinline__ void umma_tf32_{cta_group}sm_kblock_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {{
    static_assert(K > 0 && K % 8 == 0,
                  "TF32 UMMA K extent must be a positive multiple of 8");
    constexpr int KTiles = K / 8;
    asm volatile(
        "{{\\n.reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma.cta_group::{cta_group}.kind::tf32 [%0], %1, %2, %3, p;\\n}}\\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
#pragma unroll
    for (int k = 1; k < KTiles; ++k) {{
        // One K8 tf32 tile is 32 bytes = 2 x 16-byte descriptor units.
        uint64_t cur_a = (desc_a & ~0x3FFFULL) | ((desc_a + k * 2) & 0x3FFFULL);
        uint64_t cur_b = (desc_b & ~0x3FFFULL) | ((desc_b + k * 2) & 0x3FFFULL);
        asm volatile(
            "tcgen05.mma.cta_group::{cta_group}.kind::tf32 [%0], %1, %2, %3, 1;"
            :: "r"(tmem_c), "l"(cur_a), "l"(cur_b), "r"(idesc));
    }}
}}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"umma_tf32_{cta_group}sm_kblock_fn<{k}>({tmem_c}, {desc_a}, {desc_b}, {idesc}, {accum})")


def codegen_umma_tf32_1sm_kblock(tmem_c, desc_a, desc_b, idesc, k, accum):
    return _codegen_umma_tf32_ss_kblock(tmem_c, desc_a, desc_b, idesc, k, accum, 1)


@builtin(
    eval_return_type=void,
    effects=(effect(RW_TMEM_TCGEN05_COMMIT, "tmem_c"), effect(R_SMEM_TCGEN05_COMMIT, "desc_a.mem"),
             effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")), codegen_func=codegen_umma_tf32_1sm_kblock, cta_group=1)
def umma_tf32_ss_1sm_kblock(tmem_c, desc_a, desc_b, idesc, k, accum):
    """Issue a contiguous TF32 SS logical K extent; ``accum`` controls the first.

    The caller remains responsible for electing the issuing thread.
    """
    raise RuntimeError("should not call umma_tf32_ss_1sm_kblock in compilation")


def codegen_umma_tf32_2sm_kblock(tmem_c, desc_a, desc_b, idesc, k, accum):
    return _codegen_umma_tf32_ss_kblock(tmem_c, desc_a, desc_b, idesc, k, accum, 2)


@builtin(
    eval_return_type=void,
    effects=(effect(RW_TMEM_TCGEN05_COMMIT, "tmem_c"), effect(R_SMEM_TCGEN05_COMMIT, "desc_a.mem"),
             effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")), codegen_func=codegen_umma_tf32_2sm_kblock, cta_group=2)
def umma_tf32_ss_2sm_kblock(tmem_c, desc_a, desc_b, idesc, k, accum):
    """Issue a contiguous two-CTA TF32 SS logical K extent; ``accum`` controls the first."""
    raise RuntimeError("should not call umma_tf32_ss_2sm_kblock in compilation")


def codegen_umma_tf32_ts_1sm(tmem_c, tmem_a, desc_b, idesc, accum):
    body = """
__device__ __forceinline__ void umma_tf32_ts_1sm_fn(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\\n.reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p;\\n}\\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"umma_tf32_ts_1sm_fn({tmem_c}, {tmem_a}, {desc_b}, {idesc}, {accum})")


@builtin(eval_return_type=void, effects=(effect(RW_TMEM_TCGEN05_COMMIT,
                                                "tmem_c"), effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")),
         codegen_func=codegen_umma_tf32_ts_1sm, cta_group=1)
def umma_tf32_ts_1sm(tmem_c, tmem_a, desc_b, idesc, accum):
    """SM100 UMMA: cta_group::1, A from TMEM, B from SMEM, TF32×TF32→FP32."""
    raise RuntimeError("should not call umma_tf32_ts_1sm in compilation")


def codegen_umma_tf32_ts_2sm(tmem_c, tmem_a, desc_b, idesc, accum):
    body = """
__device__ __forceinline__ void umma_tf32_ts_2sm_fn(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\\n.reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma.cta_group::2.kind::tf32 [%0], [%1], %2, %3, p;\\n}\\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"umma_tf32_ts_2sm_fn({tmem_c}, {tmem_a}, {desc_b}, {idesc}, {accum})")


@builtin(eval_return_type=void, effects=(effect(RW_TMEM_TCGEN05_COMMIT,
                                                "tmem_c"), effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")),
         codegen_func=codegen_umma_tf32_ts_2sm, cta_group=2)
def umma_tf32_ts_2sm(tmem_c, tmem_a, desc_b, idesc, accum):
    """SM100 UMMA: cta_group::2, A from TMEM, B from SMEM, TF32×TF32→FP32."""
    raise RuntimeError("should not call umma_tf32_ts_2sm in compilation")


# ==============================================================================
# UMMA commit (tcgen05.commit.cta_group::2)
# ==============================================================================


def codegen_umma_commit_1sm(bar):
    body = """
__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cluster.b64"
        " [%0];"
        :: "r"(a) : "memory");
}
"""
    return Builtin(body=body, includes=[], return_val=f"umma_commit_1sm_fn({bar})")


@builtin(eval_return_type=void, codegen_func=codegen_umma_commit_1sm)
def umma_commit_1sm(bar):
    """UMMA commit with cta_group::1 barrier arrive."""
    raise RuntimeError("should not call umma_commit_1sm in compilation")


def codegen_umma_commit_2sm(bar):
    body = """
__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}
"""
    return Builtin(body=body, includes=[], return_val=f"umma_commit_2sm_fn({bar})")


@builtin(eval_return_type=void, codegen_func=codegen_umma_commit_2sm)
def umma_commit_2sm(bar):
    """UMMA commit with cta_group::2 multicast barrier arrive."""
    raise RuntimeError("should not call umma_commit_2sm in compilation")


# ==============================================================================

# ==============================================================================


def codegen_encode_umma_smem_desc(start_addr_16b, leading_bytes, stride_bytes, layout_type):
    body = """
__device__ __forceinline__ uint64_t encode_umma_smem_desc_fn(
    uint32_t start_addr_16b, uint32_t leading_bytes,
    uint32_t stride_bytes, uint32_t layout_type) {
    uint64_t d = 0;
    d |= (uint64_t)(start_addr_16b & 0x3FFF);
    d |= (uint64_t)((leading_bytes >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((stride_bytes >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)(layout_type & 0x7) << 61;
    return d;
}
"""
    return Builtin(
        body=body,
        includes=[],
        return_val=f"encode_umma_smem_desc_fn({start_addr_16b}, {leading_bytes}, {stride_bytes}, {layout_type})",
    )


@builtin(eval_return_type=uint64, codegen_func=codegen_encode_umma_smem_desc)
def encode_umma_smem_desc(start_addr_16b, leading_bytes, stride_bytes, layout_type):
    """Encode a planned SM100 UMMA SMEM descriptor."""
    raise RuntimeError("should not call encode_umma_smem_desc in compilation")


def codegen_make_umma_smem_desc(smem_ptr, offset_16b, leading_bytes, stride_bytes, layout_type):
    body = """
__device__ __forceinline__ uint64_t make_umma_smem_desc_fn(
    const void* smem_ptr, uint32_t offset_16b, uint32_t leading_bytes,
    uint32_t stride_bytes, uint32_t layout_type) {
    uint32_t start_addr_16b =
        (uint32_t)((__cvta_generic_to_shared(smem_ptr) >> 4) & 0x3FFF)
        + (offset_16b & 0x3FFF);
    uint64_t d = 0;
    d |= (uint64_t)(start_addr_16b & 0x3FFF);
    d |= (uint64_t)((leading_bytes >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((stride_bytes >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)(layout_type & 0x7) << 61;
    return d;
}
"""
    return Builtin(
        body=body,
        includes=[],
        return_val=f"make_umma_smem_desc_fn({smem_ptr}, {offset_16b}, {leading_bytes}, {stride_bytes}, {layout_type})",
    )


@builtin(eval_return_type=uint64, codegen_func=codegen_make_umma_smem_desc)
def make_umma_smem_desc(smem_ptr, offset_16b, leading_bytes, stride_bytes, layout_type):
    """Build an SM100 UMMA SMEM descriptor anchored at a shared-memory tile.

    This is the entry-kernel-callable counterpart of the dataflow intrinsic
    of the same name: the compiler-managed Flow graph derives swizzle, major
    mode, strides, and stage addresses from its layout plan, while a plain
    ``@lk.ll_kernel(is_entry=True)`` kernel states them explicitly::

        desc_a: ll.uint64 = ll.make_umma_smem_desc(
            s_a, ll.uint32(0), ll.uint32(128), ll.uint32(256), ll.uint32(4))

    ``smem_ptr`` is the shared-memory tile (a ``alloc_shared_memory`` /
    ``alloc_dynamic_shared_memory`` name, or any typed shared pointer). The
    result is a ``uint64`` hardware UMMA SMEM descriptor: the shared-window
    byte address is taken through ``__cvta_generic_to_shared``, converted to
    16-byte units, and ``offset_16b`` (also in 16-byte units) selects the
    logical sub-tile within it. ``leading_bytes`` and ``stride_bytes`` are
    the raw LBO/SBO byte distances (encoded internally as ``>>4`` fields),
    and ``layout_type`` is the hardware swizzle-mode encoding (0 = no
    swizzle / K-major interleave-free, 1 = 128B swizzle ... per the SM
    descriptor specification). Version 1 (SM100) is always set.

    Callers that already hold a ``uint32`` shared-window address (e.g. from
    ``cvta_generic_to_shared``) should instead compose
    ``encode_umma_smem_desc((addr >> 4) + offset_16b, ...)`` directly; this
    helper exists so the anchor can stay a typed pointer.
    """
    raise RuntimeError("should not call make_umma_smem_desc in compilation")


# ==============================================================================

# ==============================================================================

# ==============================================================================

# ==============================================================================

# ==============================================================================


@ll_kernel(backend="cuda", is_entry=False)
def tmem_ld_32dp_f32_to_smem_bf16_x8(src_col: uint32, dst_addr: uint32) -> void:
    """32-dp x8 TMEM ld, FP32-to-BF16 pack, and 128-bit SMEM st."""
    r0: uint32 = uint32(0)
    r1: uint32 = uint32(0)
    r2: uint32 = uint32(0)
    r3: uint32 = uint32(0)
    r4: uint32 = uint32(0)
    r5: uint32 = uint32(0)
    r6: uint32 = uint32(0)
    r7: uint32 = uint32(0)
    tmem_ld_32dp_32bxn(src_col, r0, r1, r2, r3, r4, r5, r6, r7)
    tmem_ld_fence()
    smem_st_128(
        dst_addr,
        pack_bf16(r0, r1),
        pack_bf16(r2, r3),
        pack_bf16(r4, r5),
        pack_bf16(r6, r7),
    )


@ll_kernel(backend="cuda", is_entry=False)
def tmem_ld_32dp_f32_to_smem_bf16_x32(
    src_col: uint32,
    dst_base: uint32,
    dst_row: uint32,
    dst_group_base: uint32,
    scale: float32,
) -> void:
    """32-dp x32 TMEM ld, scaled FP32-to-BF16 pack, and swizzled SMEM sts."""
    r0: uint32 = uint32(0)
    r1: uint32 = uint32(0)
    r2: uint32 = uint32(0)
    r3: uint32 = uint32(0)
    r4: uint32 = uint32(0)
    r5: uint32 = uint32(0)
    r6: uint32 = uint32(0)
    r7: uint32 = uint32(0)
    r8: uint32 = uint32(0)
    r9: uint32 = uint32(0)
    r10: uint32 = uint32(0)
    r11: uint32 = uint32(0)
    r12: uint32 = uint32(0)
    r13: uint32 = uint32(0)
    r14: uint32 = uint32(0)
    r15: uint32 = uint32(0)
    r16: uint32 = uint32(0)
    r17: uint32 = uint32(0)
    r18: uint32 = uint32(0)
    r19: uint32 = uint32(0)
    r20: uint32 = uint32(0)
    r21: uint32 = uint32(0)
    r22: uint32 = uint32(0)
    r23: uint32 = uint32(0)
    r24: uint32 = uint32(0)
    r25: uint32 = uint32(0)
    r26: uint32 = uint32(0)
    r27: uint32 = uint32(0)
    r28: uint32 = uint32(0)
    r29: uint32 = uint32(0)
    r30: uint32 = uint32(0)
    r31: uint32 = uint32(0)
    tmem_ld_32dp_32bxn(
        src_col,
        r0,
        r1,
        r2,
        r3,
        r4,
        r5,
        r6,
        r7,
        r8,
        r9,
        r10,
        r11,
        r12,
        r13,
        r14,
        r15,
        r16,
        r17,
        r18,
        r19,
        r20,
        r21,
        r22,
        r23,
        r24,
        r25,
        r26,
        r27,
        r28,
        r29,
        r30,
        r31,
    )
    tmem_ld_fence()
    addr0: uint32 = smem_swizzle_128b_addr(dst_base, dst_row, dst_group_base)
    addr1: uint32 = smem_swizzle_128b_addr(dst_base, dst_row, dst_group_base + uint32(1))
    addr2: uint32 = smem_swizzle_128b_addr(dst_base, dst_row, dst_group_base + uint32(2))
    addr3: uint32 = smem_swizzle_128b_addr(dst_base, dst_row, dst_group_base + uint32(3))
    smem_st_128(
        addr0,
        pack_bf16_float(uint_as_float(r0) * scale,
                        uint_as_float(r1) * scale),
        pack_bf16_float(uint_as_float(r2) * scale,
                        uint_as_float(r3) * scale),
        pack_bf16_float(uint_as_float(r4) * scale,
                        uint_as_float(r5) * scale),
        pack_bf16_float(uint_as_float(r6) * scale,
                        uint_as_float(r7) * scale),
    )
    smem_st_128(
        addr1,
        pack_bf16_float(uint_as_float(r8) * scale,
                        uint_as_float(r9) * scale),
        pack_bf16_float(uint_as_float(r10) * scale,
                        uint_as_float(r11) * scale),
        pack_bf16_float(uint_as_float(r12) * scale,
                        uint_as_float(r13) * scale),
        pack_bf16_float(uint_as_float(r14) * scale,
                        uint_as_float(r15) * scale),
    )
    smem_st_128(
        addr2,
        pack_bf16_float(uint_as_float(r16) * scale,
                        uint_as_float(r17) * scale),
        pack_bf16_float(uint_as_float(r18) * scale,
                        uint_as_float(r19) * scale),
        pack_bf16_float(uint_as_float(r20) * scale,
                        uint_as_float(r21) * scale),
        pack_bf16_float(uint_as_float(r22) * scale,
                        uint_as_float(r23) * scale),
    )
    smem_st_128(
        addr3,
        pack_bf16_float(uint_as_float(r24) * scale,
                        uint_as_float(r25) * scale),
        pack_bf16_float(uint_as_float(r26) * scale,
                        uint_as_float(r27) * scale),
        pack_bf16_float(uint_as_float(r28) * scale,
                        uint_as_float(r29) * scale),
        pack_bf16_float(uint_as_float(r30) * scale,
                        uint_as_float(r31) * scale),
    )


@ll_kernel(backend="cuda", is_entry=False)
def tmem_ld_16dp_f32_to_smem_bf16_x16(
    src_col: uint32,
    dst_base: uint32,
    dst_row: uint32,
    dst_group_base: uint32,
    dst_half: uint32,
    scale: float32,
) -> void:
    """16-dp x16 TMEM ld and two vectorized BF16 stores for an M=64 tile."""
    r0: uint32 = uint32(0)
    r1: uint32 = uint32(0)
    r2: uint32 = uint32(0)
    r3: uint32 = uint32(0)
    r4: uint32 = uint32(0)
    r5: uint32 = uint32(0)
    r6: uint32 = uint32(0)
    r7: uint32 = uint32(0)
    r8: uint32 = uint32(0)
    r9: uint32 = uint32(0)
    r10: uint32 = uint32(0)
    r11: uint32 = uint32(0)
    r12: uint32 = uint32(0)
    r13: uint32 = uint32(0)
    r14: uint32 = uint32(0)
    r15: uint32 = uint32(0)
    tmem_ld_16dp_32bxn(src_col, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)
    tmem_ld_fence()
    group: uint32 = dst_group_base + dst_half * uint32(2)
    addr0: uint32 = smem_swizzle_128b_addr(dst_base, dst_row, group)
    addr1: uint32 = smem_swizzle_128b_addr(dst_base, dst_row, group + uint32(1))
    smem_st_128(
        addr0,
        pack_bf16_float(uint_as_float(r0) * scale,
                        uint_as_float(r1) * scale),
        pack_bf16_float(uint_as_float(r2) * scale,
                        uint_as_float(r3) * scale),
        pack_bf16_float(uint_as_float(r4) * scale,
                        uint_as_float(r5) * scale),
        pack_bf16_float(uint_as_float(r6) * scale,
                        uint_as_float(r7) * scale),
    )
    smem_st_128(
        addr1,
        pack_bf16_float(uint_as_float(r8) * scale,
                        uint_as_float(r9) * scale),
        pack_bf16_float(uint_as_float(r10) * scale,
                        uint_as_float(r11) * scale),
        pack_bf16_float(uint_as_float(r12) * scale,
                        uint_as_float(r13) * scale),
        pack_bf16_float(uint_as_float(r14) * scale,
                        uint_as_float(r15) * scale),
    )


# ==============================================================================
# Blockscaled UMMA (tcgen05.mma.blockscaled) — NVFP4 / MXFP4 / MXFP8
# ==============================================================================

# a_format / b_format encodings (InstrDescriptorBlockScaled bits [7:10), [10:13))
FMT_E4M3 = 0
FMT_E5M2 = 1
FMT_E2M3 = 3
FMT_E3M2 = 4
FMT_E2M1 = 5

# scale_format encodings (bit [23])
SF_E4M3 = 0
SF_E8M0 = 1


def _codegen_typed_blockscaled_umma(
    tmem_c,
    desc_a,
    desc_b,
    umma_m,
    umma_n,
    accum,
    tsfa,
    tsfb,
    a_major,
    b_major,
    *,
    name,
    cta_group,
    kind,
    block_qual,
    a_format,
    b_format,
    scale_format,
):
    block = f".{block_qual}" if block_qual else ""
    m_constraint = "M == 128" if cta_group == 1 else ("M == 128 || M == 256" if kind != "mxf8f6f4" else "M == 256")
    n_constraint = "N >= 8 && N <= 256 && N % 8 == 0" if cta_group == 1 else "N >= 16 && N <= 256 && N % 16 == 0"
    body = f"""
template <int M, int N, int AMajor, int BMajor>
__device__ __forceinline__ void {name}_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t accum, uint32_t tsfa, uint32_t tsfb) {{
    static_assert({m_constraint}, "invalid UMMA M dimension");
    static_assert({n_constraint}, "invalid UMMA N dimension");
    static_assert(AMajor == 0 || AMajor == 1, "invalid A major");
    static_assert(BMajor == 0 || BMajor == 1, "invalid B major");
    static_assert(AMajor == 0 || {a_format} <= 1,
                  "A format only supports K-major");
    static_assert(BMajor == 0 || {b_format} <= 1,
                  "B format only supports K-major");
    constexpr uint32_t static_idesc =
        ({a_format}u << 7) | ({b_format}u << 10) |
        (uint32_t(AMajor) << 15) | (uint32_t(BMajor) << 16) |
        (uint32_t(N / 8) << 17) | ({scale_format}u << 23) |
        (uint32_t(M / 16) << 24);
    asm volatile("" : "+r"(tsfa), "+r"(tsfb));
    uint32_t idesc = static_idesc |
        (((tsfa >> 30) & 0x3u) << 29) |
        (((tsfb >> 30) & 0x3u) << 4);
    uint32_t sfa_addr = tsfa & 0x3fffffffu;
    uint32_t sfb_addr = tsfb & 0x3fffffffu;
    asm volatile(
        "{{\\n\\t"
        ".reg .pred p;\\n\\t"
        "setp.ne.b32 p, %4, 0;\\n\\t"
        "tcgen05.mma.cta_group::{cta_group}.kind::{kind}.block_scale{block}"
        " [%0], %1, %2, %3, [%5], [%6], p;\\n\\t"
        "}}\\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
          "r"(accum), "r"(sfa_addr), "r"(sfb_addr));
}}
"""
    return Builtin(
        body=body,
        includes=["<cuda.h>"],
        return_val=(f"{name}_fn<{umma_m}, {umma_n}, {a_major}, {b_major}>("
                    f"{tmem_c}, {desc_a}, {desc_b}, {accum}, {tsfa}, {tsfb})"),
    )


def _typed_blockscaled_umma(name, cta_group, a_format, b_format, *, nvfp4=False):
    kind = "mxf4nvf4" if nvfp4 else ("mxf4" if a_format == b_format == FMT_E2M1 else "mxf8f6f4")
    block_qual = "block16" if nvfp4 else ("block32" if kind == "mxf4" else "")
    scale_format = SF_E4M3 if nvfp4 else SF_E8M0
    if kind in {"mxf4", "mxf4nvf4"}:
        a_format = b_format = 1  # MXF4Format::E2M1, distinct from MXF8F6F4Format::E2M1.

    def codegen(tmem_c, desc_a, desc_b, umma_m, umma_n, accum, tsfa, tsfb, a_major=0, b_major=0):
        return _codegen_typed_blockscaled_umma(
            tmem_c,
            desc_a,
            desc_b,
            umma_m,
            umma_n,
            accum,
            tsfa,
            tsfb,
            a_major,
            b_major,
            name=name,
            cta_group=cta_group,
            kind=kind,
            block_qual=block_qual,
            a_format=a_format,
            b_format=b_format,
            scale_format=scale_format,
        )

    def op(tmem_c, desc_a, desc_b, umma_m, umma_n, accum, tsfa, tsfb, a_major=0, b_major=0):
        raise RuntimeError(f"should not call {name} in compilation")

    codegen.__name__ = f"codegen_{name}"
    op.__name__ = name
    op.__doc__ = (
        "Typed blockscaled SS UMMA; M/N/major are compile-time and scale-factor IDs come from the TMEM addresses.")
    return codegen, builtin(
        eval_return_type=void,
        effects=(
            effect(RW_TMEM_TCGEN05_COMMIT, "tmem_c"),
            effect(R_SMEM_TCGEN05_COMMIT, "desc_a.mem"),
            effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem"),
        ),
        codegen_func=codegen,
        cta_group=cta_group,
    )(op)


def _install_typed_blockscaled_umma(name, cta_group, a_format, b_format, *, nvfp4=False):
    codegen, op = _typed_blockscaled_umma(name, cta_group, a_format, b_format, nvfp4=nvfp4)
    globals()[codegen.__name__] = codegen
    globals()[name] = op


for _a_name, _b_name, _a_format, _b_format in (
    ("e2m1", "e2m1", FMT_E2M1, FMT_E2M1),
    ("e2m1", "e4m3", FMT_E2M1, FMT_E4M3),
    ("e4m3", "e2m1", FMT_E4M3, FMT_E2M1),
    ("e4m3", "e4m3", FMT_E4M3, FMT_E4M3),
    ("e3m2", "e4m3", FMT_E3M2, FMT_E4M3),
):
    for _cta_group in (1, 2):
        _name = f"umma_mxfp_{_a_name}_{_b_name}_ss_{_cta_group}sm"
        _install_typed_blockscaled_umma(_name, _cta_group, _a_format, _b_format)

for _cta_group in (1, 2):
    _install_typed_blockscaled_umma(f"umma_nvfp4_ss_{_cta_group}sm", _cta_group, FMT_E2M1, FMT_E2M1, nvfp4=True)

# ── SF load: SMEM → TMEM (tcgen05.cp.32x128b.warpx4) ─────────


def _codegen_tcgen05_cp_32x128b_warpx4(tmem_dst, desc, cta_group):
    name = f"tcgen05_cp_32x128b_warpx4_{cta_group}sm_fn"
    body = f"""
__device__ __forceinline__ void {name}(uint32_t tmem_dst, uint64_t desc) {{
    asm volatile("tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4 [%0], %1;"
                 :: "r"(tmem_dst), "l"(desc) : "memory");
}}
"""
    return Builtin(body=body, includes=[], return_val=f"{name}({tmem_dst}, {desc})")


def codegen_tcgen05_cp_32x128b_warpx4_1sm(tmem_dst, desc):
    return _codegen_tcgen05_cp_32x128b_warpx4(tmem_dst, desc, 1)


@builtin(eval_return_type=void, effects=(effect(W_TMEM_TCGEN05_COMMIT,
                                                "tmem_dst"), effect(R_SMEM_TCGEN05_COMMIT, "desc.mem")),
         codegen_func=codegen_tcgen05_cp_32x128b_warpx4_1sm)
def tcgen05_cp_32x128b_warpx4_1sm(tmem_dst, desc):
    """Load 32x128b SF tile from SMEM to TMEM (cta_group::1, 4-warp broadcast)."""
    raise RuntimeError("should not call tcgen05_cp_32x128b_warpx4_1sm in compilation")


def codegen_tcgen05_cp_32x128b_warpx4_2sm(tmem_dst, desc):
    return _codegen_tcgen05_cp_32x128b_warpx4(tmem_dst, desc, 2)


@builtin(eval_return_type=void, effects=(effect(W_TMEM_TCGEN05_COMMIT,
                                                "tmem_dst"), effect(R_SMEM_TCGEN05_COMMIT, "desc.mem")),
         codegen_func=codegen_tcgen05_cp_32x128b_warpx4_2sm)
def tcgen05_cp_32x128b_warpx4_2sm(tmem_dst, desc):
    """Load 32x128b SF tile from SMEM to TMEM (cta_group::2, 4-warp broadcast)."""
    raise RuntimeError("should not call tcgen05_cp_32x128b_warpx4_2sm in compilation")
