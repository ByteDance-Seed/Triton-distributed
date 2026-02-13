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
SM90 BF16 GEMM - Python DSL Implementation
Generates CUDA code equivalent to learn_source/sm90_bf16_gemm_short.cuh
Uses unified codegen and Python-side runtime for kernel launching.
"""
import little_kernel.language as ll
from little_kernel.core.compile import ll_kernel
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda
from little_kernel.runtime.tma_descriptor import create_tma_2d_descriptor
from functools import partial
import torch

# Compile-time constants
BM, BN, BK = 128, 256, 64
WGMMA_K = 16
NUM_STAGES = 3
CLUSTER_SIZE = 2
TOTAL_THREADS = 384
GROUP_M = 8
SWIZZLE_D_MODE = 128
TMA_D_BLOCK_N = SWIZZLE_D_MODE // 2  # 64
NUM_TMA_D_BLOCKS = BN // TMA_D_BLOCK_N  # 4
WGMMA_M_PER_WARP = 64 // 4  # 16
SMEM_A = BM * BK * 2
SMEM_B = BN * BK * 2
SMEM_AB = SMEM_A + SMEM_B
SMEM_D_SIZE = ((BM * BN * 2 + 1023) // 1024) * 1024
TX = SMEM_A + SMEM_B
WGMMA_SBO = 8 * BK * 2
# Total shared memory: D + stages*(A+B) + barriers (with padding)
SMEM_SIZE = SMEM_D_SIZE + NUM_STAGES * SMEM_AB + 128
DESC_K_STRIDE = WGMMA_K * 2 // 16
# With ll.empty(), A and B buffers are laid out separately (not interleaved per stage)
# so we need separate strides for A and B descriptor offsets
A_STAGE_STRIDE = SMEM_A // 16
B_STAGE_STRIDE = SMEM_B // 16
M_STRIDE = 64 * BK * 2 // 16
DESC_HI = ((3 << 14) | (((WGMMA_SBO >> 4) & 0x3FFF) << 32) | (1 << 62))

barrier_dtype = ll.uint64
backend = "cuda"


@ll_kernel(backend=backend, is_entry=True)
def gemm_kernel(
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
    # Shared memory via ll.empty() -- passes auto-compute offsets & total size
    ll.align_memory(1024, scope="dynamic_shared")
    D_smem = ll.empty([BM, BN], dtype=ll.bfloat16, scope="dynamic_shared")
    A_smem = [ll.empty([BM, BK], dtype=ll.bfloat16, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    B_smem = [ll.empty([BN, BK], dtype=ll.bfloat16, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    full_barriers = [ll.empty([1], dtype=barrier_dtype, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    empty_barriers = [ll.empty([1], dtype=barrier_dtype, scope="dynamic_shared") for _ in range(NUM_STAGES)]

    tid: ll.int32 = ll.threadIdx_x()
    wg: ll.int32 = tid // 128
    ltid: ll.int32 = tid % 128
    lane: ll.int32 = tid % 32
    cta: ll.uint32 = ll.cluster_rank()

    num_n_tiles: ll.int32 = ll.cdiv(N, BN)
    num_m_clusters: ll.int32 = ll.cdiv(M, BM * CLUSTER_SIZE)
    nk: ll.int32 = ll.cdiv(K, BK)
    cluster_id: ll.int32 = ll.blockIdx_x() // CLUSTER_SIZE

    # Barrier init
    if tid == 0:
        for s in ll.unroll(range(NUM_STAGES)):
            ll.init_smem_barrier(full_barriers[s], 1)
            ll.init_smem_barrier(empty_barriers[s], CLUSTER_SIZE * 8)
        ll.fence_smem_barrier_init()
    ll.cluster_sync()

    # TMA warpgroup
    if wg == 0:
        ll.warpgroup_reg_dealloc(40)
        if tid == 0:
            stage: ll.int32 = 0
            phase: ll.int32 = 0
            tile_idx: ll.int32 = cluster_id
            while tile_idx < num_tiles:
                num_pid_in_group: ll.int32 = GROUP_M * num_n_tiles
                group_id: ll.int32 = tile_idx // num_pid_in_group
                first_pid_m: ll.int32 = group_id * GROUP_M
                group_size_m: ll.int32 = ll.min_val(GROUP_M, num_m_clusters - first_pid_m)
                tile_m: ll.int32 = first_pid_m + (tile_idx % group_size_m)
                tile_n: ll.int32 = (tile_idx % num_pid_in_group) // group_size_m
                bn: ll.int32 = tile_n * BN
                bm: ll.int32 = tile_m * (BM * CLUSTER_SIZE) + cta * BM

                k: ll.int32 = 0
                while k < nk:
                    if stage == 0 and phase > 0:
                        ll.mbarrier_wait(empty_barriers[0], phase ^ 1)
                    elif tile_idx > cluster_id or k >= NUM_STAGES:
                        ll.mbarrier_wait(empty_barriers[stage], phase ^ 1)

                    full_ptr = full_barriers[stage]
                    ll.tma_load_2d(dA, full_ptr, A_smem[stage], k * BK, bm)

                    b_offset: ll.int32 = cta * (BN // 2)
                    ll.tma_load_multicast_2d(dB, full_ptr, B_smem[stage] + b_offset * BK, k * BK, bn + b_offset, 3)
                    ll.mbarrier_arrive_and_expect_tx(full_ptr, TX)

                    stage = stage + 1
                    if stage == NUM_STAGES:
                        stage = 0
                        phase = phase ^ 1
                    k = k + 1
                tile_idx = tile_idx + num_clusters

    # Math warpgroups
    else:
        ll.warpgroup_reg_alloc(224)
        math_wg: ll.int32 = wg - 1
        m_offset: ll.int32 = math_wg * 64
        warp_in_wg: ll.int32 = ltid // 32
        lane_idx: ll.int32 = ltid % 32

        stage: ll.int32 = 0
        phase: ll.int32 = 0
        first_tile: ll.int32 = 1

        acc = ll.zeros([128], dtype=ll.float32, scope="local")

        base_desc_a: ll.uint32 = ll.cvta_generic_to_shared(A_smem[0]) >> 4
        base_desc_b: ll.uint32 = ll.cvta_generic_to_shared(B_smem[0]) >> 4

        tile_idx: ll.int32 = cluster_id
        while tile_idx < num_tiles:
            num_pid_in_group: ll.int32 = GROUP_M * num_n_tiles
            group_id: ll.int32 = tile_idx // num_pid_in_group
            first_pid_m: ll.int32 = group_id * GROUP_M
            group_size_m: ll.int32 = ll.min_val(GROUP_M, num_m_clusters - first_pid_m)
            tile_m: ll.int32 = first_pid_m + (tile_idx % group_size_m)
            tile_n: ll.int32 = (tile_idx % num_pid_in_group) // group_size_m
            bn: ll.int32 = tile_n * BN
            bm: ll.int32 = tile_m * (BM * CLUSTER_SIZE) + cta * BM

            ll.wgmma_zero_accum(acc, 128, ll.float32)

            k: ll.int32 = 0
            while k < nk:
                ll.mbarrier_wait(full_barriers[stage], phase)

                ll.wgmma_fence()
                a_s_off: ll.uint32 = ll.uint32(stage) * A_STAGE_STRIDE
                b_s_off: ll.uint32 = ll.uint32(stage) * B_STAGE_STRIDE
                a_m_off: ll.uint32 = ll.uint32(math_wg) * M_STRIDE

                for ki in ll.unroll(range(BK // WGMMA_K)):
                    k_off: ll.uint32 = ll.uint32(ki) * DESC_K_STRIDE
                    da: ll.uint64 = ll.uint64((base_desc_a + a_s_off + a_m_off + k_off) & 0x3FFF) | DESC_HI
                    db: ll.uint64 = ll.uint64((base_desc_b + b_s_off + k_off) & 0x3FFF) | DESC_HI
                    ll.wgmma_compute(acc, da, db)

                ll.wgmma_commit()
                ll.wgmma_wait()

                if lane < CLUSTER_SIZE:
                    ll.mbarrier_arrive_remote(empty_barriers[stage], lane)

                stage = stage + 1
                if stage == NUM_STAGES:
                    stage = 0
                    phase = phase ^ 1
                k = k + 1

            # Wait for previous TMA stores
            if first_tile == 0:
                if math_wg == 0 and ltid == 0:
                    ll.tma_store_wait()
                ll.named_barrier_sync(0, 256)
            first_tile = 0

            # Store accumulators to SMEM with swizzle
            ll.store_accum_swizzle(acc, D_smem, warp_in_wg, lane_idx, m_offset)

            ll.tma_store_fence()
            ll.named_barrier_sync(0, 256)

            # Issue TMA stores
            if math_wg == 0 and ltid == 0:
                if bm < M and bn < N:
                    for t in ll.unroll(range(NUM_TMA_D_BLOCKS)):
                        col_offset: ll.int32 = t * TMA_D_BLOCK_N
                        if bn + col_offset < N:
                            ll.tma_store_2d(dC, D_smem + t * BM * (SWIZZLE_D_MODE // 2), bn + col_offset, bm)
                ll.tma_store_arrive()

            tile_idx = tile_idx + num_clusters

        if math_wg == 0 and ltid == 0:
            ll.tma_store_wait()

    ll.cluster_sync()


passes = PASSES[backend]


def generate_cuda_code():
    """Generate CUDA code using the unified codegen."""
    code = gemm_kernel.compile(
        passes,
        partial(codegen_cuda, num_threads=TOTAL_THREADS),
    )
    return code


def build_kernel(grid, num_tiles, num_clusters):
    """Build a launchable kernel using LittleKernel's Python binding.

    Parameters
    ----------
    grid : tuple
        Grid dimensions (num_blocks, 1, 1).
    num_tiles : int
        Total number of tiles (passed as kernel arg).
    num_clusters : int
        Total number of clusters (passed as kernel arg).

    Returns
    -------
    CompiledKernel
        A kernel object that can be called with kernel arguments.
    """
    return gemm_kernel.build(passes, partial(codegen_cuda, num_threads=TOTAL_THREADS), grid=grid,
                             block=(TOTAL_THREADS, 1, 1), shared_mem_bytes=SMEM_SIZE, arch="sm_90a",
                             cluster_dim=(CLUSTER_SIZE, 1, 1), include_paths=[],  # No CuTe/CUTLASS includes needed
                             )


# ---------------------------------------------------------------------------
# Host-side helpers  (test & benchmark)
# ---------------------------------------------------------------------------


def align_k(k):
    return ((k + BK - 1) // BK) * BK


def align_n(n):
    return ((n + 63) // 64) * 64


def compute_launch_params(M, N, K):
    K_aligned = align_k(K)
    N_stride = align_n(N)
    num_n_tiles = (N_stride + BN - 1) // BN
    num_m_clusters = (M + BM * CLUSTER_SIZE - 1) // (BM * CLUSTER_SIZE)
    num_tiles = num_m_clusters * num_n_tiles
    num_clusters = min(132, num_tiles)
    num_blocks = num_clusters * CLUSTER_SIZE
    return K_aligned, N_stride, num_tiles, num_clusters, num_blocks


def create_descs(A, B, C, K_aligned, N_stride):
    dA = create_tma_2d_descriptor(A, gmem_inner_dim=K_aligned, gmem_outer_dim=A.shape[0], smem_inner_dim=BK,
                                  smem_outer_dim=BM, gmem_outer_stride=A.stride(0), swizzle_mode=128, oob_fill=True)
    dB = create_tma_2d_descriptor(B, gmem_inner_dim=K_aligned, gmem_outer_dim=B.shape[0], smem_inner_dim=BK,
                                  smem_outer_dim=BN // 2, gmem_outer_stride=B.stride(0), swizzle_mode=128,
                                  oob_fill=True)
    dC = create_tma_2d_descriptor(C, gmem_inner_dim=N_stride, gmem_outer_dim=C.shape[0], smem_inner_dim=TMA_D_BLOCK_N,
                                  smem_outer_dim=BM, gmem_outer_stride=C.stride(0), swizzle_mode=128, oob_fill=False)
    return dA, dB, dC


def test(kernel, M, N, K, label=''):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    K_aligned, N_stride, num_tiles, num_clusters, num_blocks = compute_launch_params(M, N, K)
    A = torch.zeros(M, K_aligned, dtype=torch.bfloat16, device='cuda')
    B = torch.zeros(N, K_aligned, dtype=torch.bfloat16, device='cuda')
    A[:, :K] = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B[:, :K] = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    C = torch.zeros(M, N_stride, dtype=torch.bfloat16, device='cuda')
    A, B, C = A.contiguous(), B.contiguous(), C.contiguous()
    torch.cuda.synchronize()
    dA, dB, dC = create_descs(A, B, C, K_aligned, N_stride)
    kernel(dA, dB, dC, C, M, N_stride, K_aligned, num_tiles, num_clusters, grid=(num_blocks, 1, 1))
    torch.cuda.synchronize()
    C_valid = C[:, :N]
    ref = torch.matmul(A[:, :K].float(), B[:, :K].t().float())
    cos = torch.nn.functional.cosine_similarity(C_valid.float().flatten().unsqueeze(0),
                                                ref.flatten().unsqueeze(0)).item()
    ok = cos > 0.98
    shape_str = f"{M}x{N}x{K}" if label == "" else label
    print(f"  {shape_str}: cos={cos:.6f} {'PASS' if ok else 'FAIL'}")
    del A, B, C, ref
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return ok


def bench(kernel, M, N, K, iters=20, label=''):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    K_aligned, N_stride, num_tiles, num_clusters, num_blocks = compute_launch_params(M, N, K)
    A = torch.randn(M, K_aligned, dtype=torch.bfloat16, device='cuda').contiguous()
    B = torch.randn(N, K_aligned, dtype=torch.bfloat16, device='cuda').contiguous()
    C = torch.zeros(M, N_stride, dtype=torch.bfloat16, device='cuda').contiguous()
    torch.cuda.synchronize()
    dA, dB, dC = create_descs(A, B, C, K_aligned, N_stride)
    for _ in range(3):
        kernel(dA, dB, dC, C, M, N_stride, K_aligned, num_tiles, num_clusters, grid=(num_blocks, 1, 1))
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        kernel(dA, dB, dC, C, M, N_stride, K_aligned, num_tiles, num_clusters, grid=(num_blocks, 1, 1))
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    tflops = 2.0 * M * N * K / (ms * 1e9)
    shape_str = f"{M}x{N}x{K}" if label == "" else label
    print(f"  {shape_str}: {ms:.3f}ms, {tflops:.1f} TFLOPS")
    del A, B, C
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 60)
    print("SM90 BF16 GEMM - LittleKernel Python DSL")
    print("=" * 60)
    print(f"BM={BM}, BN={BN}, BK={BK}, Cluster={CLUSTER_SIZE}x1x1")
    print(f"Threads: {TOTAL_THREADS}, SMEM: {SMEM_SIZE} bytes\n")

    code = generate_cuda_code()
    print("Generated CUDA code:")
    print("=" * 80)
    print(code)
    print("=" * 80)

    print("\nBuilding kernel via LLKernel.build()...")
    kernel = build_kernel(grid=(1, 1, 1), num_tiles=1, num_clusters=1)
    print("Build OK\n")

    print("=== Correctness (aligned sizes) ===")
    all_pass = True
    for size in [1024, 2048, 4096, 8192]:
        ok = test(kernel, size, size, size, f"{size}x{size}")
        all_pass = all_pass and ok

    print("\n=== Correctness (unaligned M, N) ===")
    for Mv, Nv, Kv in [(1000, 1000, 1024), (2000, 3000, 1536), (4097, 4097, 4096), (7777, 7777, 7680),
                       (1234, 5678, 2048)]:
        ok = test(kernel, Mv, Nv, Kv)
        all_pass = all_pass and ok

    print(f"\nAll tests {'PASSED' if all_pass else 'FAILED'}!")

    print("\n=== Benchmark (aligned sizes) ===")
    for size in [1024, 2048, 4096, 8192]:
        bench(kernel, size, size, size, label=f"{size}x{size}")

    print("\n=== Benchmark (unaligned M, N) ===")
    for Mv, Nv, Kv in [(4097, 4097, 4096), (7777, 7777, 7680), (8000, 8192, 8192)]:
        bench(kernel, Mv, Nv, Kv)
