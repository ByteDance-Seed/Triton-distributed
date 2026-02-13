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
SM90 GEMM Level 1: Basic WGMMA with manual SMEM load + B128 swizzle.

Port of MatmulTutorial/examples/matmul/this-sm90/matmul-v1.cu

Key features:
  - Uses wgmma.mma_async.m64n64k16.f32.bf16.bf16
  - B128 swizzle layout for shared memory (layout_type=1)
  - Manual global-to-shared loads (no TMA)
  - fence.proxy.async for SMEM->tensor core visibility
  - Direct FP32 output

Tile: 64x64x64, 128 threads (1 warpgroup), no cluster.
"""

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda
import torch

# Compile-time constants
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
WGMMA_K = 16
WGMMA_STRIDE = 8 * BLOCK_K * 2  # 1024 bytes (8 rows * 64 elems * 2 bytes/bf16)

# WGMMA descriptor high bits:
#   bits [16:30) leading_byte_offset = 0
#   bits [32:46) stride_byte_offset = WGMMA_STRIDE >> 4 = 64
#   bits [62:64) layout_type = 1 (B128 swizzle)
DESC_HI = (((WGMMA_STRIDE >> 4) & 0x3FFF) << 32) | (1 << 62)

# No DESC_K_STRIDE constant needed; offset computed per ki in the unrolled loop

SMEM_SIZE = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * 2  # bf16


@lk.ll_kernel(backend="cuda", is_entry=True)
def gemm_v1_kernel(
    A: ll.Tensor[ll.bfloat16],
    B: ll.Tensor[ll.bfloat16],
    C: ll.Tensor[ll.float32],
    M: ll.int32,
    N: ll.int32,
    K: ll.int32,
) -> ll.void:
    # Shared memory for A and B tiles with 128B swizzle layout
    sA = ll.empty([BLOCK_M * BLOCK_K], dtype=ll.bfloat16, scope="shared")
    sB = ll.empty([BLOCK_N * BLOCK_K], dtype=ll.bfloat16, scope="shared")

    tid: ll.int32 = ll.threadIdx_x()
    bm: ll.int32 = ll.blockIdx_y() * BLOCK_M
    bn: ll.int32 = ll.blockIdx_x() * BLOCK_N

    acc64 = ll.empty([32], dtype=ll.float32, scope="local")
    ll.wgmma_init_accum_64x64(acc64, 32, ll.float32)

    # Get base SMEM addresses for descriptor construction
    base_desc_a: ll.uint32 = ll.cvta_generic_to_shared(sA) >> 4
    base_desc_b: ll.uint32 = ll.cvta_generic_to_shared(sB) >> 4

    k_base: ll.int32 = 0
    while k_base < K:
        # Load A[BLOCK_M, BLOCK_K] from global to shared with B128 swizzle
        # Swizzle formula: k' = k XOR ((m & 7) * 8)
        i: ll.int32 = tid
        while i < BLOCK_M * BLOCK_K:
            m_local: ll.int32 = i // BLOCK_K
            k_local: ll.int32 = i % BLOCK_K
            gm: ll.int32 = bm + m_local
            gk: ll.int32 = k_base + k_local
            # B128 swizzle
            k_swizzled: ll.int32 = k_local ^ ((m_local & 7) * 8)
            smem_idx: ll.int32 = m_local * BLOCK_K + k_swizzled
            if gm < M and gk < K:
                sA[smem_idx] = A[gm * K + gk]
            else:
                sA[smem_idx] = ll.val_cast(0, ll.bfloat16)
            i = i + 128

        # Load B[BLOCK_K, BLOCK_N] as B_t[BLOCK_N, BLOCK_K] with B128 swizzle
        j: ll.int32 = tid
        while j < BLOCK_K * BLOCK_N:
            kb: ll.int32 = j // BLOCK_N
            nb: ll.int32 = j % BLOCK_N
            gkb: ll.int32 = k_base + kb
            gnb: ll.int32 = bn + nb
            # B128 swizzle on transposed layout
            ks: ll.int32 = kb ^ ((nb & 7) * 8)
            si: ll.int32 = nb * BLOCK_K + ks
            if gkb < K and gnb < N:
                sB[si] = B[gkb * N + gnb]
            else:
                sB[si] = ll.val_cast(0, ll.bfloat16)
            j = j + 128

        ll.__syncthreads()
        # Critical: proxy fence required for generic->async proxy visibility
        ll.fence_proxy_async()
        ll.__syncwarp()

        # WGMMA compute loop: 4 iterations (BLOCK_K/WGMMA_K = 64/16 = 4)
        for ki in ll.unroll(range(0, BLOCK_K, WGMMA_K)):
            ll.wgmma_fence_acc64(acc64, 32)
            ll.wgmma_fence()

            # Construct SMEM descriptors for this K slice
            # ki is compile-time (from unrolled range): 0, 16, 32, 48
            # Descriptor offset = ki * sizeof(bf16) / 16 = ki * 2 / 16 = ki / 8
            da: ll.uint64 = ll.uint64((base_desc_a + ki * 2 // 16) & 0x3FFF) | DESC_HI
            db: ll.uint64 = ll.uint64((base_desc_b + ki * 2 // 16) & 0x3FFF) | DESC_HI

            ll.wgmma_compute_64x64(acc64, da, db)
            ll.wgmma_commit()
            ll.wgmma_fence_acc64(acc64, 32)
            ll.wgmma_wait()

        k_base = k_base + BLOCK_K

    # Store accumulator to global memory (float32 output)
    ll.store_acc64_to_global_f32(C, acc64, bm, bn, M, N, tid)


# =========================================================================
# Host-side helpers
# =========================================================================

passes = PASSES["cuda"]


def generate_cuda_code():
    """Generate CUDA code for inspection."""
    return gemm_v1_kernel.compile(passes, codegen_cuda)


def build_kernel():
    """Build a launchable kernel."""
    return gemm_v1_kernel.build(
        passes,
        codegen_cuda,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        shared_mem_bytes=SMEM_SIZE + 256,
        arch="sm_90a",
    )


def run_gemm(kernel, A, B, M, N, K):
    """Run the GEMM kernel: C = A @ B where A(M,K), B(K,N), C(M,N)."""
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    grid_x = (N + BLOCK_N - 1) // BLOCK_N
    grid_y = (M + BLOCK_M - 1) // BLOCK_M
    kernel(A, B, C, M, N, K, grid=(grid_x, grid_y, 1))
    return C


def test(kernel, M, N, K, label=""):
    """Test correctness."""
    torch.cuda.synchronize()
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    C = run_gemm(kernel, A, B, M, N, K)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A.float(), B.float())
    cos = torch.nn.functional.cosine_similarity(C.float().flatten().unsqueeze(0), C_ref.flatten().unsqueeze(0)).item()
    ok = cos > 0.98
    shape_str = "{}x{}x{}".format(M, N, K) if not label else label
    print("  {}: cos={:.6f} {}".format(shape_str, cos, "PASS" if ok else "FAIL"))
    return ok


def bench(kernel, M, N, K, iters=20, label=""):
    """Benchmark performance."""
    torch.cuda.synchronize()
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").contiguous()
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda").contiguous()
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    grid_x = (N + BLOCK_N - 1) // BLOCK_N
    grid_y = (M + BLOCK_M - 1) // BLOCK_M

    for _ in range(3):
        kernel(A, B, C, M, N, K, grid=(grid_x, grid_y, 1))
    torch.cuda.synchronize()

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        kernel(A, B, C, M, N, K, grid=(grid_x, grid_y, 1))
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    tflops = 2.0 * M * N * K / (ms * 1e9)
    shape_str = "{}x{}x{}".format(M, N, K) if not label else label
    print("  {}: {:.3f}ms, {:.1f} TFLOPS".format(shape_str, ms, tflops))
    return ms, tflops


if __name__ == "__main__":
    print("=" * 60)
    print("SM90 GEMM Level 1: Basic WGMMA + Manual SMEM Load")
    print("=" * 60)
    print("Tile: {}x{}x{}, Threads: 128".format(BLOCK_M, BLOCK_N, BLOCK_K))

    code = generate_cuda_code()
    print("\nGenerated CUDA ({} chars)".format(len(code)))

    print("\nBuilding kernel...")
    kernel = build_kernel()
    print("OK\n")

    print("=== Correctness ===")
    all_pass = True
    for size in [512, 1024, 2048, 4096]:
        ok = test(kernel, size, size, size)
        all_pass = all_pass and ok

    print("\n=== Correctness (unaligned) ===")
    for Mv, Nv, Kv in [(1000, 1000, 1024), (500, 600, 256)]:
        ok = test(kernel, Mv, Nv, Kv)
        all_pass = all_pass and ok

    print("\nAll tests {}!".format("PASSED" if all_pass else "FAILED"))

    print("\n=== Benchmark ===")
    for size in [1024, 2048, 4096, 8192]:
        bench(kernel, size, size, size)
