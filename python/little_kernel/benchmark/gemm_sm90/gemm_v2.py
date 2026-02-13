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
SM90 GEMM Level 2: TMA + WGMMA (single-stage, no warp specialization).

Port of MatmulTutorial/examples/matmul/this-sm90/matmul-v2.cu

Key features:
  - TMA 2D loads replace manual SMEM loads
  - TMA 128B swizzle matches WGMMA B128 layout directly
  - Mbarrier for TMA completion notification
  - Single-stage pipeline (no double buffering)

Tile: 64x64x64, 128 threads (1 warpgroup), no cluster.
D(M,K) = A(M,K) @ B(K,N) -> C(M,N) float32
  A is (M, K) row-major bf16
  B is (K, N) row-major bf16, stored as B_t(N, K) K-contiguous for TMA
"""

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda
from little_kernel.runtime.tma_descriptor import create_tma_2d_descriptor
import torch

BM, BN, BK = 64, 64, 64
WGMMA_K = 16
WGMMA_STRIDE = 8 * BK * 2  # 1024 bytes
DESC_HI = (((WGMMA_STRIDE >> 4) & 0x3FFF) << 32) | (1 << 62)
SMEM_AB = (BM * BK + BN * BK) * 2  # bf16
TX_BYTES = SMEM_AB

passes = PASSES["cuda"]


@lk.ll_kernel(backend="cuda", is_entry=True)
def gemm_v2_kernel(
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    C: ll.Tensor[ll.float32],
    M: ll.int32,
    N: ll.int32,
    K: ll.int32,
) -> ll.void:
    sA = ll.empty([BM, BK], dtype=ll.bfloat16, scope="shared")
    sB = ll.empty([BN, BK], dtype=ll.bfloat16, scope="shared")
    mbar = ll.empty([1], dtype=ll.uint64, scope="shared")

    tid: ll.int32 = ll.threadIdx_x()
    bm: ll.int32 = ll.blockIdx_y() * BM
    bn: ll.int32 = ll.blockIdx_x() * BN

    ll.wgmma_init_accum_64x64()

    base_desc_a: ll.uint32 = ll.cvta_generic_to_shared(sA) >> 4
    base_desc_b: ll.uint32 = ll.cvta_generic_to_shared(sB) >> 4

    if tid == 0:
        ll.init_smem_barrier(mbar, 1)
        ll.fence_smem_barrier_init()
    ll.__syncthreads()

    kb: ll.int32 = 0
    while kb < K:
        # TMA loads (only thread 0)
        if tid == 0:
            ll.mbarrier_arrive_and_expect_tx(mbar, TX_BYTES)
            ll.tma_load_2d(dA, mbar, sA, kb, bm)
            ll.tma_load_2d(dB, mbar, sB, kb, bn)

        # Wait for TMA completion
        phase: ll.int32 = (kb // BK) & 1
        ll.mbarrier_wait(mbar, phase)
        ll.__syncwarp()

        ll.wgmma_fence_acc64()
        ll.wgmma_fence()

        for ki in ll.unroll(range(0, BK, WGMMA_K)):
            da: ll.uint64 = ll.uint64((base_desc_a + ki * 2 // 16) & 0x3FFF) | DESC_HI
            db: ll.uint64 = ll.uint64((base_desc_b + ki * 2 // 16) & 0x3FFF) | DESC_HI
            ll.wgmma_compute_64x64(da, db)

        ll.wgmma_commit()
        ll.wgmma_fence_acc64()
        ll.wgmma_wait()

        kb = kb + BK

    ll.store_acc64_to_global_f32(C, bm, bn, M, N, tid)


def build_kernel():
    return gemm_v2_kernel.build(
        passes,
        codegen_cuda,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        shared_mem_bytes=SMEM_AB + 256,
        arch="sm_90a",
    )


def run_gemm(kernel, A, B, M, N, K):
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    # Create TMA descriptors: A(M, K) K-contiguous, B stored as B_t(N, K) K-contiguous
    dA = create_tma_2d_descriptor(A, gmem_inner_dim=K, gmem_outer_dim=M, smem_inner_dim=BK, smem_outer_dim=BM,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True)
    dB = create_tma_2d_descriptor(B.t().contiguous(), gmem_inner_dim=K, gmem_outer_dim=N, smem_inner_dim=BK,
                                  smem_outer_dim=BN, gmem_outer_stride=K, swizzle_mode=128, oob_fill=True)
    grid_x = (N + BN - 1) // BN
    grid_y = (M + BM - 1) // BM
    kernel(dA, dB, C, M, N, K, grid=(grid_x, grid_y, 1))
    return C


def test(kernel, M, N, K, label=""):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    C = run_gemm(kernel, A, B, M, N, K)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A.float(), B.float())
    cos = torch.nn.functional.cosine_similarity(C.float().flatten().unsqueeze(0), C_ref.flatten().unsqueeze(0)).item()
    ok = cos > 0.98
    s = "{}x{}x{}".format(M, N, K) if not label else label
    print("  {}: cos={:.6f} {}".format(s, cos, "PASS" if ok else "FAIL"))
    return ok


def bench(kernel, M, N, K, iters=20, label=""):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    B_t = B.t().contiguous()
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    dA = create_tma_2d_descriptor(A, gmem_inner_dim=K, gmem_outer_dim=M, smem_inner_dim=BK, smem_outer_dim=BM,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True)
    dB = create_tma_2d_descriptor(B_t, gmem_inner_dim=K, gmem_outer_dim=N, smem_inner_dim=BK, smem_outer_dim=BN,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True)
    gx, gy = (N + BN - 1) // BN, (M + BM - 1) // BM
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
    s = "{}x{}x{}".format(M, N, K) if not label else label
    print("  {}: {:.3f}ms, {:.1f} TFLOPS".format(s, ms, tflops))
    return ms, tflops


if __name__ == "__main__":
    print("=" * 60)
    print("SM90 GEMM Level 2: TMA + WGMMA")
    print("=" * 60)
    kernel = build_kernel()
    print("Kernel built OK\n")
    print("=== Correctness ===")
    all_pass = True
    for size in [512, 1024, 2048, 4096]:
        ok = test(kernel, size, size, size)
        all_pass = all_pass and ok
    for Mv, Nv, Kv in [(1000, 1000, 1024), (500, 600, 256)]:
        ok = test(kernel, Mv, Nv, Kv)
        all_pass = all_pass and ok
    print("\nAll tests {}!".format("PASSED" if all_pass else "FAILED"))
    print("\n=== Benchmark ===")
    for size in [1024, 2048, 4096, 8192]:
        bench(kernel, size, size, size)
