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
"""SM90 GEMM Level 5: Cluster 2x1x1 + TMA Multicast for B."""
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda
from little_kernel.runtime.tma_descriptor import create_tma_2d_descriptor
import torch

BM, BN, BK = 128, 128, 64
WGMMA_K = 16
NUM_STAGES = 2
CLUSTER_SIZE = 2
WGMMA_STRIDE = 8 * BK * 2
DESC_HI = (((WGMMA_STRIDE >> 4) & 0x3FFF) << 32) | (1 << 62)
SMEM_A = BM * BK * 2
SMEM_B = BN * BK * 2
TX_BYTES = SMEM_A + SMEM_B
A_STAGE_STRIDE = SMEM_A // 16
B_STAGE_STRIDE = SMEM_B // 16
M_TILE_STRIDE = 64 * BK * 2 // 16
N_TILE_STRIDE = 64 * BK * 2 // 16
SMEM_TOTAL = NUM_STAGES * (SMEM_A + SMEM_B) + 2 * NUM_STAGES * 8 + 256
passes = PASSES["cuda"]


@lk.ll_kernel(backend="cuda", is_entry=True)
def gemm_v5_kernel(
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    C: ll.Tensor[ll.float32],
    M: ll.int32,
    N: ll.int32,
    K: ll.int32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")
    A_smem = [ll.empty([BM, BK], dtype=ll.bfloat16, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    B_smem = [ll.empty([BN, BK], dtype=ll.bfloat16, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    full_barriers = [ll.empty([1], dtype=ll.uint64, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    empty_barriers = [ll.empty([1], dtype=ll.uint64, scope="dynamic_shared") for _ in range(NUM_STAGES)]

    tid: ll.int32 = ll.threadIdx_x()
    wg: ll.int32 = tid // 128
    ltid: ll.int32 = tid % 128
    lane: ll.int32 = tid % 32
    cta: ll.uint32 = ll.cluster_rank()

    # Cluster-aware indexing
    n_tile: ll.int32 = ll.blockIdx_x() // CLUSTER_SIZE
    bn: ll.int32 = n_tile * BN
    cluster_m: ll.int32 = ll.blockIdx_y() * (BM * CLUSTER_SIZE)
    bm: ll.int32 = cluster_m + ll.val_cast(cta, ll.int32) * BM
    nk: ll.int32 = (K + BK - 1) // BK

    if tid == 0:
        for s in ll.unroll(range(NUM_STAGES)):
            ll.init_smem_barrier(full_barriers[s], 1)
            ll.init_smem_barrier(empty_barriers[s], CLUSTER_SIZE * 4)
        ll.fence_smem_barrier_init()
    ll.cluster_sync()

    if wg == 0:
        ll.warpgroup_reg_dealloc(48)
        if tid == 0:
            p_stage: ll.int32 = 0
            p_phase: ll.int32 = 0
            p_k: ll.int32 = 0
            while p_k < nk:
                if p_k >= NUM_STAGES:
                    ll.mbarrier_wait(empty_barriers[p_stage], p_phase ^ 1)
                # Each CTA loads its own A tiles
                for ma in ll.unroll(range(2)):
                    ll.tma_load_2d(dA, full_barriers[p_stage], A_smem[p_stage] + ma * 64 * BK, p_k * BK, bm + ma * 64)
                # CTA 0 multicasts B to both CTAs
                if cta == 0:
                    for nb in ll.unroll(range(2)):
                        ll.tma_load_multicast_2d(dB, full_barriers[p_stage], B_smem[p_stage] + nb * 64 * BK, p_k * BK,
                                                 bn + nb * 64, 3)
                ll.mbarrier_arrive_and_expect_tx(full_barriers[p_stage], TX_BYTES)
                p_stage = p_stage + 1
                if p_stage == NUM_STAGES:
                    p_stage = 0
                    p_phase = p_phase ^ 1
                p_k = p_k + 1
    else:
        ll.warpgroup_reg_alloc(232)
        ll.wgmma_init_4acc()
        base_desc_a: ll.uint32 = ll.cvta_generic_to_shared(A_smem[0]) >> 4
        base_desc_b: ll.uint32 = ll.cvta_generic_to_shared(B_smem[0]) >> 4

        c_stage: ll.int32 = 0
        c_phase: ll.int32 = 0
        c_k: ll.int32 = 0
        while c_k < nk:
            ll.mbarrier_wait(full_barriers[c_stage], c_phase)
            ll.__syncwarp()
            ll.wgmma_fence_4acc()
            ll.wgmma_fence()
            a_s_off: ll.uint32 = ll.uint32(c_stage) * A_STAGE_STRIDE
            b_s_off: ll.uint32 = ll.uint32(c_stage) * B_STAGE_STRIDE
            for ki in ll.unroll(range(0, BK, WGMMA_K)):
                k_off: ll.int32 = ki * 2 // 16
                da0: ll.uint64 = ll.uint64((base_desc_a + a_s_off + k_off) & 0x3FFF) | DESC_HI
                db0: ll.uint64 = ll.uint64((base_desc_b + b_s_off + k_off) & 0x3FFF) | DESC_HI
                da1: ll.uint64 = ll.uint64((base_desc_a + a_s_off + M_TILE_STRIDE + k_off) & 0x3FFF) | DESC_HI
                db1: ll.uint64 = ll.uint64((base_desc_b + b_s_off + N_TILE_STRIDE + k_off) & 0x3FFF) | DESC_HI
                ll.wgmma_compute_00(da0, db0)
                ll.wgmma_compute_01(da0, db1)
                ll.wgmma_compute_10(da1, db0)
                ll.wgmma_compute_11(da1, db1)
            ll.wgmma_commit()
            ll.wgmma_fence_4acc()
            ll.wgmma_wait()
            # Cross-CTA arrive on empty barriers
            if lane < CLUSTER_SIZE:
                ll.mbarrier_arrive_remote(empty_barriers[c_stage], lane)
            ll.__syncwarp()
            c_stage = c_stage + 1
            if c_stage == NUM_STAGES:
                c_stage = 0
                c_phase = c_phase ^ 1
            c_k = c_k + 1

        ll.store_4acc_to_global_f32(C, bm, bn, M, N, ltid)


def build_kernel():
    return gemm_v5_kernel.build(passes, codegen_cuda, grid=(1, 1, 1), block=(256, 1, 1), shared_mem_bytes=SMEM_TOTAL,
                                arch="sm_90a", cluster_dim=(CLUSTER_SIZE, 1, 1))


def run_gemm(kernel, A, B, M, N, K):
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    dA = create_tma_2d_descriptor(A, gmem_inner_dim=K, gmem_outer_dim=M, smem_inner_dim=BK, smem_outer_dim=64,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True)
    dB = create_tma_2d_descriptor(B.t().contiguous(), gmem_inner_dim=K, gmem_outer_dim=N, smem_inner_dim=BK,
                                  smem_outer_dim=64, gmem_outer_stride=K, swizzle_mode=128, oob_fill=True)
    num_n_tiles = (N + BN - 1) // BN
    num_m_clusters = (M + BM * CLUSTER_SIZE - 1) // (BM * CLUSTER_SIZE)
    gx = num_n_tiles * CLUSTER_SIZE
    gy = num_m_clusters
    kernel(dA, dB, C, M, N, K, grid=(gx, gy, 1))
    return C


def test(kernel, M, N, K):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    C = run_gemm(kernel, A, B, M, N, K)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A.float(), B.float())
    cos = torch.nn.functional.cosine_similarity(C.float().flatten().unsqueeze(0), C_ref.flatten().unsqueeze(0)).item()
    ok = cos > 0.98
    print("  {}x{}x{}: cos={:.6f} {}".format(M, N, K, cos, "PASS" if ok else "FAIL"))
    return ok


def bench(kernel, M, N, K, iters=20):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_t = torch.randn(K, N, dtype=torch.bfloat16, device="cuda").t().contiguous()
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    dA = create_tma_2d_descriptor(A, gmem_inner_dim=K, gmem_outer_dim=M, smem_inner_dim=BK, smem_outer_dim=64,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True)
    dB = create_tma_2d_descriptor(B_t, gmem_inner_dim=K, gmem_outer_dim=N, smem_inner_dim=BK, smem_outer_dim=64,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True)
    num_n_tiles = (N + BN - 1) // BN
    num_m_clusters = (M + BM * CLUSTER_SIZE - 1) // (BM * CLUSTER_SIZE)
    gx = num_n_tiles * CLUSTER_SIZE
    gy = num_m_clusters
    for _ in range(3):
        kernel(dA, dB, C, M, N, K, grid=(gx, gy, 1))
    torch.cuda.synchronize()
    s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s_ev.record()
    for _ in range(iters):
        kernel(dA, dB, C, M, N, K, grid=(gx, gy, 1))
    e_ev.record()
    torch.cuda.synchronize()
    ms = s_ev.elapsed_time(e_ev) / iters
    tflops = 2.0 * M * N * K / (ms * 1e9)
    print("  {}x{}x{}: {:.3f}ms, {:.1f} TFLOPS".format(M, N, K, ms, tflops))
    return ms, tflops


if __name__ == "__main__":
    print("=" * 60)
    print("SM90 GEMM Level 5: Cluster 2x1x1 + TMA Multicast")
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
