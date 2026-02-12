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
"""SM90 GEMM Level 9: Fully Async Warpgroup Scheduling (DeepGEMM Style).

Features:
- TMA and math warpgroups run independent tile schedulers
- Barriers initialized ONCE at kernel start, not per-tile
- No __syncthreads() between tiles - pure barrier-based sync
- Pipeline stages span across tiles (continuous phase tracking)
- Pipelined TMA store overlapped with next tile's compute
- BM=128, BN=256, BK=64, Cluster 2x1x1
"""
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda
from little_kernel.runtime.tma_descriptor import create_tma_2d_descriptor
import torch

BM, BN, BK = 128, 256, 64
WGMMA_K = 16
NUM_STAGES = 3
CLUSTER_SIZE = 2
TOTAL_THREADS = 384
NUM_TMA_REGS = 40
NUM_MATH_REGS = 224
GROUP_M = 8

SMEM_A = BM * BK * 2
SMEM_B = BN * BK * 2
SMEM_AB = SMEM_A + SMEM_B
SMEM_C = BM * BN * 2
TX_BYTES = SMEM_A + SMEM_B

WGMMA_SBO = 8 * BK * 2
DESC_HI = (3 << 14) | (((WGMMA_SBO >> 4) & 0x3FFF) << 32) | (1 << 62)

A_STAGE_STRIDE = SMEM_A // 16
B_STAGE_STRIDE = SMEM_B // 16
M_TILE_STRIDE = 64 * BK * 2 // 16

SMEM_TOTAL = NUM_STAGES * SMEM_AB + 128 + SMEM_C
passes = PASSES["cuda"]


@lk.ll_kernel(backend="cuda", is_entry=True)
def gemm_v9_kernel(
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dC: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    C: ll.Tensor[ll.bfloat16],
    M: ll.int32,
    N: ll.int32,
    K: ll.int32,
    num_tiles: ll.int32,
    num_clusters: ll.int32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")
    C_smem = ll.empty([BM, BN], dtype=ll.bfloat16, scope="dynamic_shared")
    A_smem = [ll.empty([BM, BK], dtype=ll.bfloat16, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    B_smem = [ll.empty([BN, BK], dtype=ll.bfloat16, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    full_barriers = [ll.empty([1], dtype=ll.uint64, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    empty_barriers = [ll.empty([1], dtype=ll.uint64, scope="dynamic_shared") for _ in range(NUM_STAGES)]

    tid: ll.int32 = ll.threadIdx_x()
    wg: ll.int32 = tid // 128
    ltid: ll.int32 = tid % 128
    lane: ll.int32 = tid % 32
    cta: ll.uint32 = ll.cluster_rank()

    num_n_tiles: ll.int32 = (N + BN - 1) // BN
    num_m_clusters: ll.int32 = (M + BM * CLUSTER_SIZE - 1) // (BM * CLUSTER_SIZE)
    nk: ll.int32 = (K + BK - 1) // BK
    cluster_id: ll.int32 = ll.blockIdx_x() // CLUSTER_SIZE
    b_half_off: ll.int32 = ll.val_cast(cta, ll.int32) * (BN // 2)

    # Barrier init ONCE
    if tid == 0:
        for s in ll.unroll(range(NUM_STAGES)):
            ll.init_smem_barrier(full_barriers[s], 1)
            ll.init_smem_barrier(empty_barriers[s], CLUSTER_SIZE * 8)
        ll.fence_smem_barrier_init()
    ll.cluster_sync()

    if wg == 0:
        ll.warpgroup_reg_dealloc(NUM_TMA_REGS)
        if tid == 0:
            p_stage: ll.int32 = 0
            p_phase: ll.int32 = 0
            p_giter: ll.int32 = 0
            p_tile: ll.int32 = cluster_id
            while p_tile < num_tiles:
                # L2 swizzle
                num_pid_in_group_p: ll.int32 = GROUP_M * num_n_tiles
                group_id_p: ll.int32 = p_tile // num_pid_in_group_p
                first_pid_m_p: ll.int32 = group_id_p * GROUP_M
                group_size_m_p: ll.int32 = ll.min_val(GROUP_M, num_m_clusters - first_pid_m_p)
                tile_m_p: ll.int32 = first_pid_m_p + (p_tile % group_size_m_p)
                tile_n_p: ll.int32 = (p_tile % num_pid_in_group_p) // group_size_m_p
                p_bn: ll.int32 = tile_n_p * BN
                p_bm: ll.int32 = tile_m_p * (BM * CLUSTER_SIZE) + ll.val_cast(cta, ll.int32) * BM

                pk: ll.int32 = 0
                while pk < nk:
                    if p_giter >= NUM_STAGES:
                        ll.mbarrier_wait(empty_barriers[p_stage], p_phase ^ 1)
                    ll.tma_load_2d(dA, full_barriers[p_stage], A_smem[p_stage], pk * BK, p_bm)
                    ll.tma_load_multicast_2d(dB, full_barriers[p_stage], B_smem[p_stage] + b_half_off * BK, pk * BK,
                                             p_bn + b_half_off, 3)
                    ll.mbarrier_arrive_and_expect_tx(full_barriers[p_stage], TX_BYTES)
                    p_stage = p_stage + 1
                    if p_stage == NUM_STAGES:
                        p_stage = 0
                        p_phase = p_phase ^ 1
                    p_giter = p_giter + 1
                    pk = pk + 1
                p_tile = p_tile + num_clusters
    else:
        ll.warpgroup_reg_alloc(NUM_MATH_REGS)
        a_m_off: ll.uint32 = ll.uint32(wg - 1) * M_TILE_STRIDE
        m_wg_off: ll.int32 = (wg - 1) * 64
        math_wg_idx: ll.int32 = wg - 1

        base_desc_a: ll.uint32 = ll.cvta_generic_to_shared(A_smem[0]) >> 4
        base_desc_b: ll.uint32 = ll.cvta_generic_to_shared(B_smem[0]) >> 4

        c_stage: ll.int32 = 0
        c_phase: ll.int32 = 0
        c_first: ll.int32 = 1
        c_tile: ll.int32 = cluster_id
        while c_tile < num_tiles:
            # L2 swizzle
            num_pid_in_group_c: ll.int32 = GROUP_M * num_n_tiles
            group_id_c: ll.int32 = c_tile // num_pid_in_group_c
            first_pid_m_c: ll.int32 = group_id_c * GROUP_M
            group_size_m_c: ll.int32 = ll.min_val(GROUP_M, num_m_clusters - first_pid_m_c)
            tile_m_c: ll.int32 = first_pid_m_c + (c_tile % group_size_m_c)
            tile_n_c: ll.int32 = (c_tile % num_pid_in_group_c) // group_size_m_c
            c_bn: ll.int32 = tile_n_c * BN
            c_bm: ll.int32 = tile_m_c * (BM * CLUSTER_SIZE) + ll.val_cast(cta, ll.int32) * BM

            ll.wgmma_init_accum()

            ck: ll.int32 = 0
            while ck < nk:
                ll.mbarrier_wait(full_barriers[c_stage], c_phase)
                ll.__syncwarp()
                ll.wgmma_fence()
                a_s_off: ll.uint32 = ll.uint32(c_stage) * A_STAGE_STRIDE
                b_s_off: ll.uint32 = ll.uint32(c_stage) * B_STAGE_STRIDE
                for ki in ll.unroll(range(0, BK, WGMMA_K)):
                    k_off: ll.int32 = ki * 2 // 16
                    da: ll.uint64 = ll.uint64((base_desc_a + a_s_off + a_m_off + k_off) & 0x3FFF) | DESC_HI
                    db: ll.uint64 = ll.uint64((base_desc_b + b_s_off + k_off) & 0x3FFF) | DESC_HI
                    ll.wgmma_compute(da, db)
                ll.wgmma_commit()
                ll.wgmma_wait()
                if lane < CLUSTER_SIZE:
                    ll.mbarrier_arrive_remote(empty_barriers[c_stage], lane)
                ll.__syncwarp()
                c_stage = c_stage + 1
                if c_stage == NUM_STAGES:
                    c_stage = 0
                    c_phase = c_phase ^ 1
                ck = ck + 1

            # Wait for previous TMA store if not first tile
            if c_first == 0:
                if lane == 0:
                    ll.tma_store_wait()
                ll.__syncwarp()
            c_first = 0

            # Store acc to SMEM
            ll.store_acc_to_smem_bf16_n256(C_smem, ltid, m_wg_off)

            # Sync math warpgroups
            ll.named_barrier_sync(0, 256)

            # TMA store
            ll.tma_store_fence()
            if ltid == 0:
                if math_wg_idx == 0:
                    ll.tma_store_2d(dC, C_smem, c_bn, c_bm)
                    ll.tma_store_2d(dC, C_smem + 64 * BN, c_bn, c_bm + 64)
                    ll.tma_store_arrive()

            # Sync after store issued
            ll.named_barrier_sync(0, 256)
            c_tile = c_tile + num_clusters

        # Final TMA store wait
        if lane == 0:
            ll.tma_store_wait()

    ll.cluster_sync()


def build_kernel():
    return gemm_v9_kernel.build(passes, codegen_cuda, grid=(1, 1, 1), block=(TOTAL_THREADS, 1, 1),
                                shared_mem_bytes=SMEM_TOTAL, arch="sm_90a", cluster_dim=(CLUSTER_SIZE, 1, 1))


def run_gemm(kernel, A, B, M, N, K):
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    dA = create_tma_2d_descriptor(A, gmem_inner_dim=K, gmem_outer_dim=M, smem_inner_dim=BK, smem_outer_dim=BM,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True)
    dB = create_tma_2d_descriptor(B.t().contiguous(), gmem_inner_dim=K, gmem_outer_dim=N, smem_inner_dim=BK,
                                  smem_outer_dim=BN // 2, gmem_outer_stride=K, swizzle_mode=128, oob_fill=True)
    dC = create_tma_2d_descriptor(C, gmem_inner_dim=N, gmem_outer_dim=M, smem_inner_dim=BN, smem_outer_dim=64,
                                  gmem_outer_stride=N, swizzle_mode=0, oob_fill=False)
    num_n_tiles = (N + BN - 1) // BN
    num_m_clusters = (M + BM * CLUSTER_SIZE - 1) // (BM * CLUSTER_SIZE)
    num_tiles = num_m_clusters * num_n_tiles
    num_clusters = min(132, num_tiles)
    gx = num_clusters * CLUSTER_SIZE
    kernel(dA, dB, dC, C, M, N, K, num_tiles, num_clusters, grid=(gx, 1, 1))
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
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    dA = create_tma_2d_descriptor(A, gmem_inner_dim=K, gmem_outer_dim=M, smem_inner_dim=BK, smem_outer_dim=BM,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True)
    dB = create_tma_2d_descriptor(B_t, gmem_inner_dim=K, gmem_outer_dim=N, smem_inner_dim=BK, smem_outer_dim=BN // 2,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True)
    dC = create_tma_2d_descriptor(C, gmem_inner_dim=N, gmem_outer_dim=M, smem_inner_dim=BN, smem_outer_dim=64,
                                  gmem_outer_stride=N, swizzle_mode=0, oob_fill=False)
    num_n_tiles = (N + BN - 1) // BN
    num_m_clusters = (M + BM * CLUSTER_SIZE - 1) // (BM * CLUSTER_SIZE)
    num_tiles = num_m_clusters * num_n_tiles
    num_clusters = min(132, num_tiles)
    gx = num_clusters * CLUSTER_SIZE
    for _ in range(3):
        kernel(dA, dB, dC, C, M, N, K, num_tiles, num_clusters, grid=(gx, 1, 1))
    torch.cuda.synchronize()
    s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s_ev.record()
    for _ in range(iters):
        kernel(dA, dB, dC, C, M, N, K, num_tiles, num_clusters, grid=(gx, 1, 1))
    e_ev.record()
    torch.cuda.synchronize()
    ms = s_ev.elapsed_time(e_ev) / iters
    tflops = 2.0 * M * N * K / (ms * 1e9)
    print("  {}x{}x{}: {:.3f}ms, {:.1f} TFLOPS".format(M, N, K, ms, tflops))
    return ms, tflops


if __name__ == "__main__":
    print("=" * 60)
    print("SM90 GEMM Level 9: Fully Async Warpgroup Scheduling")
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
