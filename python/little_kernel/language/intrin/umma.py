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
TMEM load, cta_group::2 TMA load, elect_one, and SM100 SMEM/instruction descriptors.
All implementations use standalone PTX (no CuTe/CUTLASS dependency).
"""

from ..builtin_base import builtin, Builtin
from little_kernel.core.type_system import void, uint32, uint64, float32

# ==============================================================================
# elect_one: elect a single thread from the warp
# ==============================================================================


def codegen_elect_one():
    body = """
__device__ __forceinline__ uint32_t elect_one_fn() {
    uint32_t pred;
    asm volatile(
        "{\\n.reg .pred p;\\n"
        "elect.sync _|p, 0xFFFFFFFF;\\n"
        "selp.b32 %0, 1, 0, p;\\n}\\n" : "=r"(pred));
    return pred;
}
"""
    return Builtin(body=body, includes=[], return_val="elect_one_fn()")


@builtin(eval_return_type=uint32, codegen_func=codegen_elect_one)
def elect_one():
    """Elect a single thread in the warp. Returns 1 for elected, 0 for others."""
    raise RuntimeError("should not call elect_one in compilation")


# ==============================================================================
# TMEM allocation / deallocation (cta_group::2)
# ==============================================================================


def codegen_tmem_alloc(smem_addr_ptr, ncols):
    body = """
__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}
"""
    return Builtin(body=body, includes=[], return_val=f"tmem_alloc_fn({smem_addr_ptr}, {ncols})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_alloc)
def tmem_alloc(smem_addr_ptr, ncols):
    """Allocate TMEM columns (cta_group::2). Result written to smem_addr_ptr."""
    raise RuntimeError("should not call tmem_alloc in compilation")


def codegen_tmem_dealloc(addr, ncols):
    body = """
__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}
"""
    return Builtin(body=body, includes=[], return_val=f"tmem_dealloc_fn({addr}, {ncols})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_dealloc)
def tmem_dealloc(addr, ncols):
    """Deallocate TMEM columns (cta_group::2)."""
    raise RuntimeError("should not call tmem_dealloc in compilation")


# ==============================================================================
# TMEM load (tcgen05.ld)
# ==============================================================================


def codegen_tmem_load_4x(col, r0, r1, r2, r3):
    body = """
__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}
"""
    return Builtin(body=body, includes=[], return_val=f"tmem_load_4x_fn({col}, &{r0}, &{r1}, &{r2}, &{r3})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_load_4x)
def tmem_load_4x(col, r0, r1, r2, r3):
    """Load 4×32b from TMEM column."""
    raise RuntimeError("should not call tmem_load_4x in compilation")


def codegen_tmem_load_8x(col, r0, r1, r2, r3, r4, r5, r6, r7):
    body = """
__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
                   "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"tmem_load_8x_fn({col}, &{r0}, &{r1}, &{r2}, &{r3}, &{r4}, &{r5}, &{r6}, &{r7})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_load_8x)
def tmem_load_8x(col, r0, r1, r2, r3, r4, r5, r6, r7):
    """Load 8×32b from TMEM column."""
    raise RuntimeError("should not call tmem_load_8x in compilation")


def codegen_tmem_load_fence():
    body = """
__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="tmem_load_fence_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_load_fence)
def tmem_load_fence():
    """Fence for TMEM loads (tcgen05.wait::ld)."""
    raise RuntimeError("should not call tmem_load_fence in compilation")


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


@builtin(eval_return_type=void, codegen_func=codegen_tcgen05_fence_after)
def tcgen05_fence_after():
    """tcgen05 fence after thread sync."""
    raise RuntimeError("should not call tcgen05_fence_after in compilation")


def codegen_tcgen05_fence_before():
    body = """
__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="tcgen05_fence_before_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_tcgen05_fence_before)
def tcgen05_fence_before():
    """tcgen05 fence before thread sync."""
    raise RuntimeError("should not call tcgen05_fence_before in compilation")


# ==============================================================================
# UMMA instruction (tcgen05.mma.cta_group::2.kind::f16)
# ==============================================================================


def codegen_umma_f16_cg2(tmem_c, desc_a, desc_b, idesc, accum):
    body = """
__device__ __forceinline__ void umma_f16_cg2_fn(
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
                   return_val=f"umma_f16_cg2_fn({tmem_c}, {desc_a}, {desc_b}, {idesc}, {accum})")


@builtin(eval_return_type=void, codegen_func=codegen_umma_f16_cg2)
def umma_f16_cg2(tmem_c, desc_a, desc_b, idesc, accum):
    """SM100 UMMA instruction: cta_group::2, kind::f16 (BF16×BF16→FP32).
    accum=0 clears accumulator, accum=1 accumulates."""
    raise RuntimeError("should not call umma_f16_cg2 in compilation")


# ==============================================================================
# UMMA commit (tcgen05.commit.cta_group::2)
# ==============================================================================


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
# cta_group::2 TMA load
# ==============================================================================


def codegen_tma_load_2d_cg2(desc, bar_ptr, smem_ptr, c0, c1):
    body = """
__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}
"""
    return Builtin(body=body, includes=["<cuda.h>"],
                   return_val=f"tma_load_2d_cg2_fn(&{desc}, {bar_ptr}, {smem_ptr}, {c0}, {c1})")


@builtin(eval_return_type=void, codegen_func=codegen_tma_load_2d_cg2)
def tma_load_2d_cg2(desc, bar_ptr, smem_ptr, c0, c1):
    """cta_group::2 TMA 2D load. Applies PEER_BIT_MASK to barrier address."""
    raise RuntimeError("should not call tma_load_2d_cg2 in compilation")


# ==============================================================================
# Cluster-scoped barrier operations (for SM100 2SM coordination)
# ==============================================================================


def codegen_mbarrier_arrive_expect_tx_cluster(bar_ptr, tx_bytes, target_cta):
    body = """
__device__ __forceinline__ void mbarrier_arrive_expect_tx_cluster_fn(uint64_t* bar, uint32_t tx, uint32_t target_cta) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(remote_a), "r"(tx));
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"mbarrier_arrive_expect_tx_cluster_fn({bar_ptr}, {tx_bytes}, {target_cta})")


@builtin(eval_return_type=void, codegen_func=codegen_mbarrier_arrive_expect_tx_cluster)
def mbarrier_arrive_expect_tx_cluster(bar_ptr, tx_bytes, target_cta):
    """Cluster-scoped arrive_expect_tx: signals a barrier in target_cta's shared memory."""
    raise RuntimeError("should not call mbarrier_arrive_expect_tx_cluster in compilation")


def codegen_mbarrier_arrive_cluster(bar_ptr, target_cta):
    body = """
__device__ __forceinline__ void mbarrier_arrive_cluster_fn(uint64_t* bar, uint32_t target_cta) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                 :: "r"(remote_a));
}
"""
    return Builtin(body=body, includes=[], return_val=f"mbarrier_arrive_cluster_fn({bar_ptr}, {target_cta})")


@builtin(eval_return_type=void, codegen_func=codegen_mbarrier_arrive_cluster)
def mbarrier_arrive_cluster(bar_ptr, target_cta):
    """Arrive at a barrier on a remote CTA in the cluster."""
    raise RuntimeError("should not call mbarrier_arrive_cluster in compilation")


# ==============================================================================
# SM100 SMEM descriptor (version=1)
# ==============================================================================


def codegen_make_smem_desc_sm100(smem_ptr, sbo):
    body = """
__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)2 << 61;   // layout_type = SWIZZLE_128B
    return d;
}
"""
    return Builtin(body=body, includes=[], return_val=f"make_smem_desc_sm100_fn({smem_ptr}, {sbo})")


@builtin(eval_return_type=uint64, codegen_func=codegen_make_smem_desc_sm100)
def make_smem_desc_sm100(smem_ptr, sbo):
    """Build SM100 UMMA SMEM descriptor (version=1, SWIZZLE_128B)."""
    raise RuntimeError("should not call make_smem_desc_sm100 in compilation")


# ==============================================================================
# SM100 instruction descriptor for tcgen05.mma
# ==============================================================================


def codegen_make_instr_desc(umma_m, umma_n):
    body = """
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    d |= ((N / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
    return d;
}
"""
    return Builtin(body=body, includes=[], return_val=f"make_instr_desc_fn({umma_m}, {umma_n})")


@builtin(eval_return_type=uint32, codegen_func=codegen_make_instr_desc)
def make_instr_desc(umma_m, umma_n):
    """Build SM100 UMMA instruction descriptor (BF16×BF16→FP32, K-major)."""
    raise RuntimeError("should not call make_instr_desc in compilation")


# ==============================================================================
# BF16 packing utility
# ==============================================================================


def codegen_pack_bf16(fp32_a, fp32_b):
    body = """
#include <cuda_bf16.h>
__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}
"""
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"pack_bf16_fn({fp32_a}, {fp32_b})")


@builtin(eval_return_type=uint32, codegen_func=codegen_pack_bf16)
def pack_bf16(fp32_a, fp32_b):
    """Pack two FP32 values (as uint32 bit patterns) into one uint32 of two BF16."""
    raise RuntimeError("should not call pack_bf16 in compilation")


# ==============================================================================
# 128-bit shared memory store
# ==============================================================================


def codegen_st_shared_128(addr, v0, v1, v2, v3):
    body = """
__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}
"""
    return Builtin(body=body, includes=[], return_val=f"st_shared_128_fn({addr}, {v0}, {v1}, {v2}, {v3})")


@builtin(eval_return_type=void, codegen_func=codegen_st_shared_128)
def st_shared_128(addr, v0, v1, v2, v3):
    """128-bit vectorized store to shared memory (st.shared.v4.b32)."""
    raise RuntimeError("should not call st_shared_128 in compilation")


# ==============================================================================
# TMA store with swizzle support (for SM100 epilogue)
# ==============================================================================


def codegen_tma_store_2d_sm100(desc, smem_ptr, c0, c1):
    body = """
__device__ __forceinline__ void tma_store_2d_sm100_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(sa) : "memory");
}
"""
    return Builtin(body=body, includes=["<cuda.h>"],
                   return_val=f"tma_store_2d_sm100_fn(&{desc}, {smem_ptr}, {c0}, {c1})")


@builtin(eval_return_type=void, codegen_func=codegen_tma_store_2d_sm100)
def tma_store_2d_sm100(desc, smem_ptr, c0, c1):
    """SM100 TMA 2D store from shared to global memory."""
    raise RuntimeError("should not call tma_store_2d_sm100 in compilation")


def codegen_tma_store_commit():
    body = """
__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="tma_store_commit_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_tma_store_commit)
def tma_store_commit():
    """TMA store commit group."""
    raise RuntimeError("should not call tma_store_commit in compilation")


# ==============================================================================
# uint_as_float: bitcast uint32 -> float32
# ==============================================================================


def codegen_uint_as_float(val):
    body = ""
    return Builtin(body=body, includes=[], return_val=f"__uint_as_float({val})")


@builtin(eval_return_type=float32, codegen_func=codegen_uint_as_float)
def uint_as_float(val):
    """Bitcast uint32 to float32 (__uint_as_float)."""
    raise RuntimeError("should not call uint_as_float in compilation")


# ==============================================================================
# TMEM epilogue: read TMEM columns, convert FP32->BF16, write to global memory
# ==============================================================================


def codegen_tmem_store_bf16_row(D, tid, M, N, m_base, n_base, bn):
    body = """
#include <cuda_bf16.h>
__device__ __forceinline__ void tmem_store_bf16_row_fn(
    __nv_bfloat16* D, uint32_t tid, uint32_t M, uint32_t N,
    uint32_t m_base, uint32_t n_base, uint32_t BN) {
    uint32_t m_idx = m_base + tid;
    if (m_idx >= M) return;
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        uint32_t nc = n_base + col;
        __nv_bfloat16* out = D + (uint64_t)m_idx * N + nc;
        if (nc     < N) out[0] = __float2bfloat16(f0);
        if (nc + 1 < N) out[1] = __float2bfloat16(f1);
        if (nc + 2 < N) out[2] = __float2bfloat16(f2);
        if (nc + 3 < N) out[3] = __float2bfloat16(f3);
    }
}
"""
    return Builtin(body=body, includes=["<cuda_bf16.h>"],
                   return_val=f"tmem_store_bf16_row_fn({D}, {tid}, {M}, {N}, {m_base}, {n_base}, {bn})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_store_bf16_row)
def tmem_store_bf16_row(D, tid, M, N, m_base, n_base, bn):
    """Read TMEM columns, convert FP32->BF16, store to global D.
    Each thread reads its own row (m_base + tid) for BN columns starting at n_base."""
    raise RuntimeError("should not call tmem_store_bf16_row in compilation")


# ==============================================================================
# tmem_epilogue_coalesced_4w: TMEM -> SMEM -> coalesced global writes
#
# REQUIRES blockDim.x == 128 (exactly 4 warps). Using other thread counts causes
# incorrect row mapping in Phase 2 and wrong output. The "4w" suffix enforces
# this constraint explicitly.
# ==============================================================================


def codegen_tmem_epilogue_coalesced_4w(D, smem_out, M, N, m_block, n_block, bm, bn):
    body = """
#include <cuda_bf16.h>
__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    // Phase 1: TMEM -> SMEM
    // Each of the 128 threads owns one row (tid 0..127 -> row 0..127). Load 4
    // FP32 cols at a time from TMEM, convert to BF16, write to smem_out[tid*BN+col].
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    // Phase 2: SMEM -> Global (coalesced vectorized 8-byte writes)
    // REQUIRES 4 warps: each step processes 4 rows (one per warp). row = step*4 + warp_id.
    // Each warp: 32 lanes write 4 BF16 (8 bytes) each, covering 32*4=128 cols per row.
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        if (global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
        }
    }
}
"""
    return Builtin(
        body=body, includes=["<cuda_bf16.h>"],
        return_val=f"tmem_epilogue_coalesced_4w_fn({D}, {smem_out}, {M}, {N}, {m_block}, {n_block}, {bm}, {bn})")


@builtin(eval_return_type=void, codegen_func=codegen_tmem_epilogue_coalesced_4w)
def tmem_epilogue_coalesced_4w(D, smem_out, M, N, m_block, n_block, bm, bn):
    """Coalesced epilogue: TMEM -> SMEM staging -> vectorized coalesced global writes.

    Phase 1: Each of 128 threads loads its row from TMEM (FP32), converts to BF16,
    writes to shared memory. Phase 2: Coalesced 8-byte stores to global D.

    REQUIRES blockDim.x == 128 (exactly 4 warps). The row mapping in Phase 2
    uses "row = step*4 + warp_id"; other thread counts will produce wrong output.

    Args:
        D: Global output pointer (BF16).
        smem_out: BM*BN BF16 buffer in shared memory (caller must allocate).
        M, N: Full matrix dimensions.
        m_block, n_block: Block start indices.
        bm, bn: Block tile sizes (BM, BN).
    """
    raise RuntimeError("should not call tmem_epilogue_coalesced_4w in compilation")
