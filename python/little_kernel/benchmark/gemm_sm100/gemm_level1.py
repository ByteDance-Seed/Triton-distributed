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
"""SM100 GEMM Level 1: Baseline 2SM UMMA Kernel.

D = A @ B^T  (A: M×K row-major BF16, B: N×K row-major BF16, D: M×N BF16)

Features:
- BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
- cta_group::2 UMMA (UMMA_M=256): two SMs cooperate
- 1-stage pipeline (no double buffering)
- TMA loads with B128 swizzle
- TMEM accumulator (FP32) -> BF16 output via tmem_store_bf16_row epilogue
- Cluster 2x1x1: each CTA loads half of B (64 cols)
"""
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda
from little_kernel.runtime.tma_descriptor import create_tma_2d_descriptor
import torch

BM, BN, BK = 128, 128, 64
CLUSTER_SIZE = 2
LOAD_N_PER_CTA = BN // CLUSTER_SIZE  # 64
UMMA_M, UMMA_N, UMMA_K = 256, BN, 16
NUM_THREADS = 128
TMEM_COLS = BN  # 128

SMEM_A = BM * BK * 2  # 16384
SMEM_B = LOAD_N_PER_CTA * BK * 2  # 8192
TX_BYTES = SMEM_A + SMEM_B

UMMA_SBO = 8 * BK * 2  # 1024

# SM100 SMEM descriptor high bits (compile-time constant)
# Bits [32:46): SBO >> 4
# Bits [46:48): version = 1 (SM100)
# Bits [61:64): layout_type = 2 (SWIZZLE_128B)
DESC_HI = (((UMMA_SBO >> 4) & 0x3FFF) << 32) | (1 << 46) | (2 << 61)

# UMMA instruction descriptor (compile-time constant)
IDESC = (1 << 4) | (1 << 7) | (1 << 10) | ((UMMA_N // 8) << 17) | ((UMMA_M // 16) << 24)

SMEM_TOTAL = 1024 + SMEM_A + SMEM_B + 128  # barriers+pad + data + extra

passes = PASSES["cuda"]


@lk.ll_kernel(backend="cuda", is_entry=True)
def gemm_level1_kernel(
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    D: ll.Tensor[ll.bfloat16],
    M: ll.uint32,
    N: ll.uint32,
    K: ll.uint32,
    num_n_blocks: ll.uint32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")
    # Data first (TMA requires 128-byte alignment)
    A_smem = ll.empty([BM, BK], dtype=ll.bfloat16, scope="dynamic_shared")
    B_smem = ll.empty([LOAD_N_PER_CTA, BK], dtype=ll.bfloat16, scope="dynamic_shared")
    # Barriers and metadata after data (only need 8-byte alignment)
    full_bar = ll.empty([1], dtype=ll.uint64, scope="dynamic_shared")
    empty_bar = ll.empty([1], dtype=ll.uint64, scope="dynamic_shared")
    tmem_bar = ll.empty([1], dtype=ll.uint64, scope="dynamic_shared")
    tmem_addr = ll.empty([1], dtype=ll.uint32, scope="dynamic_shared")
    _pad = ll.empty([120], dtype=ll.uint64, scope="dynamic_shared")

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid // 32
    cta: ll.uint32 = ll.cluster_rank()

    cluster_id: ll.uint32 = ll.blockIdx_x() // CLUSTER_SIZE
    m_block: ll.uint32 = cluster_id // num_n_blocks
    n_block: ll.uint32 = cluster_id % num_n_blocks

    # Prefetch TMA descriptors (warp 0)
    if warp_idx == 0:
        el0: ll.uint32 = ll.elect_one()
        if el0 == 1:
            ll.prefetch_tma_descriptor(dA)
            ll.prefetch_tma_descriptor(dB)

    # Init barriers (warp 1)
    if warp_idx == 1:
        el1: ll.uint32 = ll.elect_one()
        if el1 == 1:
            ll.init_smem_barrier(full_bar, 1)
            ll.init_smem_barrier(empty_bar, 1)
            ll.init_smem_barrier(tmem_bar, 1)
            ll.fence_smem_barrier_init()

    # TMEM alloc (warp 2)
    if warp_idx == 2:
        ll.tmem_alloc(tmem_addr, TMEM_COLS)

    ll.cluster_sync()

    # Build base SMEM descriptor addresses
    base_a: ll.uint32 = ll.cvta_generic_to_shared(A_smem) >> 4
    base_b: ll.uint32 = ll.cvta_generic_to_shared(B_smem) >> 4

    is_leader: ll.int32 = ll.val_cast(cta, ll.int32) == 0
    num_k_blocks: ll.uint32 = (K + BK - 1) // BK
    m_coord: ll.int32 = ll.val_cast(m_block * BM, ll.int32)
    n_coord: ll.int32 = ll.val_cast(n_block * BN + cta * LOAD_N_PER_CTA, ll.int32)

    # K loop
    phase: ll.uint32 = ll.uint32(0)
    first_umma: ll.uint32 = ll.uint32(1)
    kb: ll.uint32 = ll.uint32(0)
    while kb < num_k_blocks:
        # Wait empty barrier (skip on first iter via parity trick)
        ll.mbarrier_wait(empty_bar, phase ^ 1)

        # TMA loads (warp 0, elected thread)
        if warp_idx == 0:
            el_tma: ll.uint32 = ll.elect_one()
            if el_tma == 1:
                k_coord: ll.int32 = ll.val_cast(kb * BK, ll.int32)
                ll.tma_load_2d(dA, full_bar, A_smem, k_coord, m_coord)
                ll.tma_load_2d(dB, full_bar, B_smem, k_coord, n_coord)
                ll.mbarrier_arrive_and_expect_tx(full_bar, TX_BYTES)

        # Wait TMA
        ll.mbarrier_wait(full_bar, phase)

        # Cluster sync: ensure both CTAs have data
        ll.cluster_sync()

        # UMMA compute (leader CTA, warp 1, elected thread)
        if is_leader == 1:
            if warp_idx == 1:
                el_umma: ll.uint32 = ll.elect_one()
                if el_umma == 1:
                    ll.tcgen05_fence_after()
                    for ki in ll.unroll(range(0, BK, UMMA_K)):
                        k_off = ki * 2 // 16  # 0, 2, 4, 6
                        ad: ll.uint64 = (ll.uint64(base_a + k_off) & 0x3FFF) | DESC_HI
                        bd: ll.uint64 = (ll.uint64(base_b + k_off) & 0x3FFF) | DESC_HI
                        if ki == 0:
                            # Only first sub-K step: accum depends on first_umma
                            acc_flag: ll.uint32 = ll.uint32(1) - first_umma
                            ll.umma_f16_cg2(0, ad, bd, IDESC, acc_flag)
                        else:
                            ll.umma_f16_cg2(0, ad, bd, IDESC, ll.uint32(1))
                    # Signal: SMEM consumed
                    ll.umma_commit_2sm(empty_bar)
                    # Signal: TMEM result ready (last K tile)
                    if kb == num_k_blocks - 1:
                        ll.umma_commit_2sm(tmem_bar)

        first_umma = ll.uint32(0)
        phase = phase ^ 1
        kb = kb + 1

    # Epilogue: TMEM -> global D
    ll.mbarrier_wait(tmem_bar, 0)
    ll.tcgen05_fence_after()

    m_base: ll.uint32 = m_block * BM
    n_base: ll.uint32 = n_block * BN
    ll.tmem_store_bf16_row(D, ll.val_cast(tid, ll.uint32), M, N, m_base, n_base, BN)

    ll.cluster_sync()

    if warp_idx == 2:
        ll.tmem_dealloc(0, TMEM_COLS)


def build_kernel():
    return gemm_level1_kernel.build(passes, codegen_cuda, grid=(1, 1, 1), block=(NUM_THREADS, 1, 1),
                                    shared_mem_bytes=SMEM_TOTAL, arch="sm_100a", cluster_dim=(CLUSTER_SIZE, 1, 1))


def run_gemm(kernel, A, B, M, N, K):
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    dA = create_tma_2d_descriptor(A, gmem_inner_dim=K, gmem_outer_dim=M, smem_inner_dim=BK, smem_outer_dim=BM,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True,
                                  l2_promotion=0)  # L2_PROMOTION_NONE
    dB = create_tma_2d_descriptor(B, gmem_inner_dim=K, gmem_outer_dim=N, smem_inner_dim=BK,
                                  smem_outer_dim=LOAD_N_PER_CTA, gmem_outer_stride=K, swizzle_mode=128, oob_fill=True,
                                  l2_promotion=0)
    num_m_blocks = (M + BM - 1) // BM
    num_n_blocks = (N + BN - 1) // BN
    num_ctas = num_m_blocks * num_n_blocks * CLUSTER_SIZE
    kernel(dA, dB, D, M, N, K, num_n_blocks, grid=(num_ctas, 1, 1))
    return D


def test(kernel, M, N, K):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    D = run_gemm(kernel, A, B, M, N, K)
    torch.cuda.synchronize()
    D_ref = torch.matmul(A.float(), B.float().t())
    cos = torch.nn.functional.cosine_similarity(D.float().flatten().unsqueeze(0), D_ref.flatten().unsqueeze(0)).item()
    ok = cos > 0.98
    print("  {}x{}x{}: cos={:.6f} {}".format(M, N, K, cos, "PASS" if ok else "FAIL"))
    return ok


def bench(kernel, M, N, K, iters=20):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    dA = create_tma_2d_descriptor(A, gmem_inner_dim=K, gmem_outer_dim=M, smem_inner_dim=BK, smem_outer_dim=BM,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True, l2_promotion=0)
    dB = create_tma_2d_descriptor(B, gmem_inner_dim=K, gmem_outer_dim=N, smem_inner_dim=BK,
                                  smem_outer_dim=LOAD_N_PER_CTA, gmem_outer_stride=K, swizzle_mode=128, oob_fill=True,
                                  l2_promotion=0)
    num_m_blocks = (M + BM - 1) // BM
    num_n_blocks = (N + BN - 1) // BN
    num_ctas = num_m_blocks * num_n_blocks * CLUSTER_SIZE
    for _ in range(3):
        kernel(dA, dB, D, M, N, K, num_n_blocks, grid=(num_ctas, 1, 1))
    torch.cuda.synchronize()
    s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s_ev.record()
    for _ in range(iters):
        kernel(dA, dB, D, M, N, K, num_n_blocks, grid=(num_ctas, 1, 1))
    e_ev.record()
    torch.cuda.synchronize()
    ms = s_ev.elapsed_time(e_ev) / iters
    tflops = 2.0 * M * N * K / (ms * 1e9)
    print("  {}x{}x{}: {:.3f}ms, {:.1f} TFLOPS".format(M, N, K, ms, tflops))


if __name__ == "__main__":
    print("=" * 60)
    print("SM100 GEMM Level 1: Baseline 2SM UMMA")
    print("=" * 60)
    kernel = build_kernel()
    print("Kernel built OK\n")
    print("=== Correctness ===")
    all_pass = True
    for size in [512, 1024, 2048, 4096]:
        ok = test(kernel, size, size, size)
        all_pass = all_pass and ok
    print("\nAll tests {}!".format("PASSED" if all_pass else "FAILED"))
    print("\n=== Benchmark ===")
    for size in [1024, 2048, 4096, 8192]:
        bench(kernel, size, size, size)
