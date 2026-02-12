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
"""SM100 GEMM Level 4: SMEM-Staged Coalesced Epilogue.

D = A @ B^T  (A: M×K, B: N×K, D: M×N, all BF16 row-major)

Improvements over Level 3:
- Coalesced epilogue: TMEM -> SMEM staging -> vectorized global writes
  (32 threads/warp write same row at consecutive addresses -> fully coalesced)
- Inherits: true warp specialization, 8-stage pipeline, cta_group::2 TMA
"""
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda
from little_kernel.runtime.tma_descriptor import create_tma_2d_descriptor
import torch

BM, BN, BK = 128, 128, 64
CLUSTER_SIZE = 2
NUM_STAGES = 8
LOAD_N_PER_CTA = BN // CLUSTER_SIZE
UMMA_M, UMMA_N, UMMA_K = 256, BN, 16
NUM_THREADS = 128
TMEM_COLS = BN

SMEM_A = BM * BK * 2
SMEM_B = LOAD_N_PER_CTA * BK * 2
SMEM_STAGE = SMEM_A + SMEM_B
TX_BYTES = SMEM_A + SMEM_B
UMMA_SBO = 8 * BK * 2

DESC_HI = (((UMMA_SBO >> 4) & 0x3FFF) << 32) | (1 << 46) | (2 << 61)
IDESC = (1 << 4) | (1 << 7) | (1 << 10) | ((UMMA_N // 8) << 17) | ((UMMA_M // 16) << 24)
A_STRIDE_16 = SMEM_A // 16
B_STRIDE_16 = SMEM_B // 16

SMEM_TOTAL = 1024 + NUM_STAGES * SMEM_STAGE + 128

passes = PASSES["cuda"]


@lk.ll_kernel(backend="cuda", is_entry=True)
def gemm_level4_kernel(
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
    A_smem = [ll.empty([BM, BK], dtype=ll.bfloat16, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    B_smem = [ll.empty([LOAD_N_PER_CTA, BK], dtype=ll.bfloat16, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    # Barriers and metadata after data
    full_bars = [ll.empty([1], dtype=ll.uint64, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    empty_bars = [ll.empty([1], dtype=ll.uint64, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    tmem_bar = ll.empty([1], dtype=ll.uint64, scope="dynamic_shared")
    tmem_addr = ll.empty([1], dtype=ll.uint32, scope="dynamic_shared")
    _pad = ll.empty([108], dtype=ll.uint64, scope="dynamic_shared")

    # Epilogue SMEM staging buffer (reuses pipeline SMEM space at A_smem[0])
    # This is safe because K-loop is complete before epilogue starts

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid // 32
    cta: ll.uint32 = ll.cluster_rank()

    cluster_id: ll.uint32 = ll.blockIdx_x() // CLUSTER_SIZE
    m_block: ll.uint32 = cluster_id // num_n_blocks
    n_block: ll.uint32 = cluster_id % num_n_blocks

    if warp_idx == 0:
        el0: ll.uint32 = ll.elect_one()
        if el0 == 1:
            ll.prefetch_tma_descriptor(dA)
            ll.prefetch_tma_descriptor(dB)

    if warp_idx == 1:
        el1: ll.uint32 = ll.elect_one()
        if el1 == 1:
            for s in ll.unroll(range(NUM_STAGES)):
                ll.init_smem_barrier(full_bars[s], CLUSTER_SIZE)
                ll.init_smem_barrier(empty_bars[s], 1)
            ll.init_smem_barrier(tmem_bar, 1)
            ll.fence_smem_barrier_init()

    if warp_idx == 2:
        ll.tmem_alloc(tmem_addr, TMEM_COLS)

    ll.cluster_sync()

    base_a: ll.uint32 = ll.cvta_generic_to_shared(A_smem[0]) >> 4
    base_b: ll.uint32 = ll.cvta_generic_to_shared(B_smem[0]) >> 4

    num_k_blocks: ll.uint32 = (K + BK - 1) // BK
    m_coord: ll.int32 = ll.val_cast(m_block * BM, ll.int32)
    n_coord: ll.int32 = ll.val_cast(n_block * BN + cta * LOAD_N_PER_CTA, ll.int32)
    is_leader: ll.int32 = ll.val_cast(cta, ll.int32) == 0

    # TMA Warp
    if warp_idx == 0:
        el_tma: ll.uint32 = ll.elect_one()
        if el_tma == 1:
            tma_stage: ll.int32 = 0
            tma_phase: ll.uint32 = ll.uint32(0)
            tma_kb: ll.uint32 = ll.uint32(0)
            while tma_kb < num_k_blocks:
                ll.mbarrier_wait(empty_bars[tma_stage], tma_phase ^ 1)
                if is_leader == 1:
                    ll.mbarrier_arrive_and_expect_tx(full_bars[tma_stage], TX_BYTES)
                if is_leader == 0:
                    ll.mbarrier_arrive_expect_tx_cluster(full_bars[tma_stage], TX_BYTES, 0)
                tma_kc: ll.int32 = ll.val_cast(tma_kb * BK, ll.int32)
                ll.tma_load_2d_cg2(dA, full_bars[tma_stage], A_smem[tma_stage], tma_kc, m_coord)
                ll.tma_load_2d_cg2(dB, full_bars[tma_stage], B_smem[tma_stage], tma_kc, n_coord)
                tma_stage = tma_stage + 1
                if tma_stage == NUM_STAGES:
                    tma_stage = 0
                    tma_phase = tma_phase ^ 1
                tma_kb = tma_kb + 1

    # MMA Warp
    if warp_idx == 1:
        if is_leader == 1:
            el_mma: ll.uint32 = ll.elect_one()
            if el_mma == 1:
                mma_stage: ll.int32 = 0
                mma_phase: ll.uint32 = ll.uint32(0)
                first_umma: ll.uint32 = ll.uint32(1)
                mma_kb: ll.uint32 = ll.uint32(0)
                while mma_kb < num_k_blocks:
                    ll.mbarrier_wait(full_bars[mma_stage], mma_phase)
                    ll.tcgen05_fence_after()
                    a_s_off: ll.uint32 = ll.uint32(mma_stage) * A_STRIDE_16
                    b_s_off: ll.uint32 = ll.uint32(mma_stage) * B_STRIDE_16
                    for ki in ll.unroll(range(0, BK, UMMA_K)):
                        k_off = ki * 2 // 16
                        ad: ll.uint64 = (ll.uint64(base_a + a_s_off + k_off) & 0x3FFF) | DESC_HI
                        bd: ll.uint64 = (ll.uint64(base_b + b_s_off + k_off) & 0x3FFF) | DESC_HI
                        if ki == 0:
                            acc_flag: ll.uint32 = ll.uint32(1) - first_umma
                            ll.umma_f16_cg2(0, ad, bd, IDESC, acc_flag)
                        else:
                            ll.umma_f16_cg2(0, ad, bd, IDESC, ll.uint32(1))
                    ll.umma_commit_2sm(empty_bars[mma_stage])
                    if mma_kb == num_k_blocks - 1:
                        ll.umma_commit_2sm(tmem_bar)
                    first_umma = ll.uint32(0)
                    mma_stage = mma_stage + 1
                    if mma_stage == NUM_STAGES:
                        mma_stage = 0
                        mma_phase = mma_phase ^ 1
                    mma_kb = mma_kb + 1

    # Epilogue: coalesced TMEM -> SMEM -> Global (leader only)
    ll.mbarrier_wait(tmem_bar, 0)
    ll.tcgen05_fence_after()

    if is_leader == 1:
        # Reuse A_smem[0] as staging buffer (BM*BN bf16 = 32KB fits in first stage)
        ll.tmem_epilogue_coalesced_4w(D, A_smem[0], M, N, m_block, n_block, BM, BN)

    ll.cluster_sync()
    if warp_idx == 2:
        ll.tmem_dealloc(0, TMEM_COLS)


def build_kernel():
    return gemm_level4_kernel.build(passes, codegen_cuda, grid=(1, 1, 1), block=(NUM_THREADS, 1, 1),
                                    shared_mem_bytes=SMEM_TOTAL, arch="sm_100a", cluster_dim=(CLUSTER_SIZE, 1, 1))


def run_gemm(kernel, A, B, M, N, K):
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    dA = create_tma_2d_descriptor(A, gmem_inner_dim=K, gmem_outer_dim=M, smem_inner_dim=BK, smem_outer_dim=BM,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True, l2_promotion=0)
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
    print("SM100 GEMM Level 4: Coalesced Epilogue")
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
