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
"""One-CTA SM100 UMMA smoke test used by flash-attention bring-up.

This validates the local LittleKernel wrappers needed by the fixed SM100
attention kernel: cta_group::1 TMA load, cta_group::1 UMMA, TMEM load, and TMA
store. It is deliberately small and non-overlapped.
"""

import os

import torch

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.atom.scheduler import get_tile_coord_along_m
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor

BM, BN, BK = 128, 128, 64
NUM_STAGES = 1
NUM_THREADS = 256
NUM_EPI_THREADS = 128
UMMA_M, UMMA_N, UMMA_K = 128, 128, 16
GROUP_M = 8

SWIZZLE_BYTES = 128
STORE_BN = SWIZZLE_BYTES // 2
NUM_STORES = BN // STORE_BN
BANK_GROUP_BYTES = 16
ELEMS_PER_BG = BANK_GROUP_BYTES // 2

SMEM_A = BM * BK * 2
SMEM_B = BN * BK * 2
TX_BYTES = SMEM_A + SMEM_B
SMEM_CD = BM * SWIZZLE_BYTES
SMEM_TOTAL = SMEM_CD + NUM_STAGES * (SMEM_A + SMEM_B) + 1024
TMEM_COLS = BN

UMMA_SBO = 8 * BK * 2
DESC_HI = (((UMMA_SBO >> 4) & 0x3FFF) << 32) | (1 << 46) | (2 << 61)
IDESC = (1 << 4) | (1 << 7) | (1 << 10) | ((UMMA_N // 8) << 17) | ((UMMA_M // 16) << 24)

passes = PASSES["cuda"]


@lk.ll_kernel(backend="cuda", is_entry=True)
def umma_gemm_smoke_kernel(
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dD: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    M: ll.uint32,
    N: ll.uint32,
    K: ll.uint32,
    num_ctas: ll.uint32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")
    cd_smem = ll.empty([BM, STORE_BN], dtype=ll.bfloat16, scope="dynamic_shared")
    a_smem = ll.empty([BM, BK], dtype=ll.bfloat16, scope="dynamic_shared")
    b_smem = ll.empty([BN, BK], dtype=ll.bfloat16, scope="dynamic_shared")
    full_bar = ll.empty([1], dtype=ll.uint64, scope="dynamic_shared")
    empty_bar = ll.empty([1], dtype=ll.uint64, scope="dynamic_shared")
    tmem_full = ll.empty([1], dtype=ll.uint64, scope="dynamic_shared")
    tmem_empty = ll.empty([1], dtype=ll.uint64, scope="dynamic_shared")
    tmem_addr = ll.empty([1], dtype=ll.uint32, scope="dynamic_shared")

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid // 32
    cta_id: ll.uint32 = ll.blockIdx_x()

    num_m_blocks: ll.uint32 = (M + BM - 1) // BM
    num_n_blocks: ll.uint32 = (N + BN - 1) // BN
    num_tiles: ll.uint32 = num_m_blocks * num_n_blocks
    num_k_blocks: ll.uint32 = (K + BK - 1) // BK

    if warp_idx == 0:
        el0: ll.uint32 = ll.elect_one()
        if el0 == 1:
            ll.prefetch_tma_descriptor(dA)
            ll.prefetch_tma_descriptor(dB)
            ll.prefetch_tma_descriptor(dD)

    if warp_idx == 1:
        el1: ll.uint32 = ll.elect_one()
        if el1 == 1:
            ll.init_smem_barrier(full_bar, 1)
            ll.init_smem_barrier(empty_bar, 1)
            ll.init_smem_barrier(tmem_full, 1)
            ll.init_smem_barrier(tmem_empty, 1)
            ll.fence_smem_barrier_init()

    if warp_idx == 2:
        ll.tmem_alloc_1sm(tmem_addr, TMEM_COLS)
    ll.__syncthreads()

    base_a: ll.uint32 = ll.cvta_generic_to_shared(a_smem) >> 4
    base_b: ll.uint32 = ll.cvta_generic_to_shared(b_smem) >> 4
    cd_base: ll.uint32 = ll.cvta_generic_to_shared(cd_smem)

    if warp_idx == 0:
        el_tma: ll.uint32 = ll.elect_one()
        if el_tma == 1:
            tile_tma: ll.uint32 = cta_id
            phase: ll.uint32 = ll.uint32(0)
            while tile_tma < num_tiles:
                coord_m_tma, coord_n_tma = get_tile_coord_along_m(tile_tma, num_m_blocks)
                offset_m_tma: ll.int32 = ll.val_cast(coord_m_tma * BM, ll.int32)
                offset_n_tma: ll.int32 = ll.val_cast(coord_n_tma * BN, ll.int32)
                kb: ll.uint32 = ll.uint32(0)
                while kb < num_k_blocks:
                    ll.mbarrier_wait(empty_bar, phase ^ 1)
                    ll.mbarrier_arrive_and_expect_tx(full_bar, TX_BYTES)
                    offset_k_tma: ll.int32 = ll.val_cast(kb * BK, ll.int32)
                    ll.tma_load_2d_1sm(dA, full_bar, a_smem, offset_k_tma, offset_m_tma)
                    ll.tma_load_2d_1sm(dB, full_bar, b_smem, offset_k_tma, offset_n_tma)
                    phase = phase ^ 1
                    kb = kb + 1
                tile_tma = tile_tma + num_ctas

    if warp_idx == 1:
        el_mma: ll.uint32 = ll.elect_one()
        if el_mma == 1:
            tile_mma: ll.uint32 = cta_id
            phase_mma: ll.uint32 = ll.uint32(0)
            while tile_mma < num_tiles:
                ll.tcgen05_fence_after()
                kb: ll.uint32 = ll.uint32(0)
                while kb < num_k_blocks:
                    ll.mbarrier_wait(full_bar, phase_mma)
                    ll.tcgen05_fence_after()
                    for ki in ll.unroll(range(0, BK, UMMA_K)):
                        k_off = ki * 2 // 16
                        ad: ll.uint64 = (ll.uint64(base_a + k_off) & 0x3FFF) | DESC_HI
                        bd: ll.uint64 = (ll.uint64(base_b + k_off) & 0x3FFF) | DESC_HI
                        acc_flag: ll.uint32 = ll.uint32((kb > 0) | (ki > 0))
                        ll.umma_f16_ss_1sm(ll.uint32(0), ad, bd, IDESC, acc_flag)
                    ll.umma_commit_1sm(empty_bar)
                    if kb == num_k_blocks - 1:
                        ll.umma_commit_1sm(tmem_full)
                    phase_mma = phase_mma ^ 1
                    kb = kb + 1
                tile_mma = tile_mma + num_ctas

    if warp_idx >= 4:
        local_tid: ll.int32 = tid - 128
        epi_warp: ll.int32 = local_tid // 32
        tile_epi: ll.uint32 = cta_id
        while tile_epi < num_tiles:
            coord_m_epi, coord_n_epi = get_tile_coord_along_m(tile_epi, num_m_blocks)
            ll.mbarrier_wait(tmem_full, ll.uint32(0))
            ll.tcgen05_fence_after()
            for s in ll.unroll(range(NUM_STORES)):
                for i in ll.unroll(range(STORE_BN // ELEMS_PER_BG)):
                    tmem_col: ll.uint32 = ll.uint32(s * STORE_BN + i * ELEMS_PER_BG)
                    r0: ll.uint32 = ll.uint32(0)
                    r1: ll.uint32 = ll.uint32(0)
                    r2: ll.uint32 = ll.uint32(0)
                    r3: ll.uint32 = ll.uint32(0)
                    r4: ll.uint32 = ll.uint32(0)
                    r5: ll.uint32 = ll.uint32(0)
                    r6: ll.uint32 = ll.uint32(0)
                    r7: ll.uint32 = ll.uint32(0)
                    ll.tmem_ld_32dp_32bxn(tmem_col, r0, r1, r2, r3, r4, r5, r6, r7)
                    ll.tmem_ld_fence()
                    p0: ll.uint32 = ll.pack_bf16(r0, r1)
                    p1: ll.uint32 = ll.pack_bf16(r2, r3)
                    p2: ll.uint32 = ll.pack_bf16(r4, r5)
                    p3: ll.uint32 = ll.pack_bf16(r6, r7)
                    smem_addr: ll.uint32 = ll.smem_swizzle_128b_addr(cd_base, ll.uint32(local_tid), ll.uint32(i))
                    ll.smem_st_128(smem_addr, p0, p1, p2, p3)
                ll.named_barrier_sync(1, NUM_EPI_THREADS)
                if epi_warp == 0:
                    el_st: ll.uint32 = ll.elect_one()
                    if el_st == 1:
                        ll.fence_async_shared()
                        offset_n_epi: ll.int32 = ll.val_cast(coord_n_epi * BN + s * STORE_BN, ll.int32)
                        offset_m_epi: ll.int32 = ll.val_cast(coord_m_epi * BM, ll.int32)
                        ll.tma_store_2d(dD, cd_smem, offset_n_epi, offset_m_epi)
                        ll.tma_store_commit()
                        ll.tma_store_wait_group(0)
                ll.named_barrier_sync(1, NUM_EPI_THREADS)
            if local_tid == 0:
                ll.mbarrier_arrive(tmem_empty)
            tile_epi = tile_epi + num_ctas

    ll.__syncthreads()
    if warp_idx == 2:
        ll.tmem_relinquish_alloc_permit_1sm()
        ll.tmem_dealloc_1sm(ll.uint32(0), TMEM_COLS)


def build_kernel(cache_dir=None):
    if cache_dir is None:
        import tempfile
        cache_dir = os.path.join(tempfile.gettempdir(), 'lk_sm100_umma_gemm_smoke')
    return umma_gemm_smoke_kernel.build(
        passes,
        codegen_cuda,
        grid=(1, 1, 1),
        block=(NUM_THREADS, 1, 1),
        shared_mem_bytes=SMEM_TOTAL,
        arch="sm_100a",
        cache_dir=cache_dir,
        extra_nvcc_options=["-DNDEBUG", "--use_fast_math"],
    )


def run_gemm(kernel, a, b):
    m, k = a.shape
    n = b.shape[0]
    out = torch.zeros((m, n), device=a.device, dtype=torch.bfloat16)
    dA = create_tma_2d_descriptor(
        a,
        gmem_inner_dim=k,
        gmem_outer_dim=m,
        smem_inner_dim=BK,
        smem_outer_dim=BM,
        gmem_outer_stride=k,
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    dB = create_tma_2d_descriptor(
        b,
        gmem_inner_dim=k,
        gmem_outer_dim=n,
        smem_inner_dim=BK,
        smem_outer_dim=BN,
        gmem_outer_stride=k,
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    dD = create_tma_2d_descriptor(
        out,
        gmem_inner_dim=n,
        gmem_outer_dim=m,
        smem_inner_dim=STORE_BN,
        smem_outer_dim=BM,
        gmem_outer_stride=n,
        swizzle_mode=128,
        oob_fill=False,
        l2_promotion=3,
    )
    num_tiles = ((m + BM - 1) // BM) * ((n + BN - 1) // BN)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    nc = min(num_sms, num_tiles)
    kernel(dA, dB, dD, m, n, k, nc, grid=(nc, 1, 1))
    return out


def smoke_test():
    torch.manual_seed(0)
    a = torch.randn((BM, BK * 2), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((BN, BK * 2), device="cuda", dtype=torch.bfloat16)
    kernel = build_kernel()
    out = run_gemm(kernel, a, b)
    ref = a.float() @ b.float().t()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-1)
    print("SM100 cta_group::1 UMMA smoke PASS")


if __name__ == "__main__":
    smoke_test()
