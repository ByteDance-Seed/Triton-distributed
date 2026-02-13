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
TMA (Tensor Memory Accelerator) Bandwidth & Latency Benchmark
==============================================================

Measures Hopper's TMA unit performance using LittleKernel intrinsics.

Benchmarks:
  1. TMA 1D Bulk Copy bandwidth: cp.async.bulk global->shared with various sizes.
  2. TMA 2D Load bandwidth:     cp.async.bulk.tensor.2d with descriptor-based loads.
  3. TMA vs Manual Copy:        Comparing TMA to manual shared memory copy.
  4. TMA Load Latency:          Single TMA load latency measurement.

TMA on Hopper:
  - Dedicated hardware unit for async data movement
  - Supports 1D bulk copy and 2D/3D/4D/5D tensor copy
  - Integrates with mbarrier for async completion notification
  - Max single transfer: 128 bytes per TMA instruction (1D), up to tile-sized (2D)

Usage:
  python bench_tma.py
"""

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda
from little_kernel.runtime.tma_descriptor import create_tma_2d_descriptor
import torch
import os

from little_kernel.benchmark.utils import (get_gpu_info, print_gpu_info, CSVWriter, get_results_dir, print_table,
                                           format_bandwidth)

# =============================================================================
# Kernel 1: TMA 1D Bulk Copy Bandwidth (Global -> Shared)
# =============================================================================
# Uses cp.async.bulk.shared::cluster.global (tma_copy_1d_g2s)
# Measures sustained bandwidth for various transfer sizes.

COPY_1D_ITERS = 500


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_tma_1d_bandwidth(
    src: ll.Tensor[ll.float32],
    elapsed: ll.Tensor[ll.uint64],
    load_bytes: ll.int32,
) -> ll.void:
    # Allocate max shared memory (we'll use load_bytes of it)
    smem_data = ll.empty([16384], dtype=ll.uint8, scope="shared")
    barrier = ll.empty([1], dtype=ll.uint64, scope="shared")

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid // 32

    # Initialize barrier (only thread 0)
    if tid == 0:
        ll.init_smem_barrier(barrier, 1)
        ll.fence_smem_barrier_init()
    ll.__syncthreads()

    phase: ll.int32 = 0

    # Warmup
    if warp_idx == 0 and ll.elect_one_sync():
        for _w in range(5):
            ll.tma_copy_1d_g2s(src, barrier, smem_data, load_bytes)
            ll.mbarrier_arrive_and_expect_tx(barrier, load_bytes)
    ll.mbarrier_wait(barrier, phase)
    phase = phase ^ 1
    ll.__syncthreads()

    # Measure
    start: ll.uint64 = ll.clock64()

    for _iter in range(COPY_1D_ITERS):
        if warp_idx == 0 and ll.elect_one_sync():
            ll.tma_copy_1d_g2s(src, barrier, smem_data, load_bytes)
            ll.mbarrier_arrive_and_expect_tx(barrier, load_bytes)
        ll.mbarrier_wait(barrier, phase)
        phase = phase ^ 1

    end: ll.uint64 = ll.clock64()

    if tid == 0:
        elapsed[0] = end - start


# =============================================================================
# Kernel 2: TMA 2D Load Bandwidth (Global -> Shared, descriptor-based)
# =============================================================================
# Uses cp.async.bulk.tensor.2d with CUtensorMap descriptor.

TMA_2D_ITERS = 200
TMA_2D_BLOCK_M = 128
TMA_2D_BLOCK_N = 64  # = 128B swizzle / 2B per bf16
TMA_2D_SMEM_BYTES = TMA_2D_BLOCK_M * TMA_2D_BLOCK_N * 2  # bf16


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_tma_2d_bandwidth(
    input_desc: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    elapsed: ll.Tensor[ll.uint64],
) -> ll.void:
    smem_data = ll.empty([TMA_2D_BLOCK_M * TMA_2D_BLOCK_N], dtype=ll.bfloat16, scope="shared")
    barrier = ll.empty([1], dtype=ll.uint64, scope="shared")

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = ll.__shfl_sync("0xffffffff", tid // 32, 0)

    if warp_idx == 0 and ll.elect_one_sync():
        ll.prefetch_tma_descriptor(input_desc)
        ll.init_smem_barrier(barrier, 1)
        ll.fence_smem_barrier_init()
    ll.__syncthreads()

    phase: ll.int32 = 0

    # Warmup
    if warp_idx == 1 and ll.elect_one_sync():
        ll.tma_load_2d(input_desc, barrier, smem_data, 0, 0)
        ll.mbarrier_arrive_and_expect_tx(barrier, TMA_2D_SMEM_BYTES)
    ll.mbarrier_wait(barrier, phase)
    phase = phase ^ 1
    ll.__syncthreads()

    start: ll.uint64 = ll.clock64()

    for _iter in range(TMA_2D_ITERS):
        if warp_idx == 1 and ll.elect_one_sync():
            ll.tma_load_2d(input_desc, barrier, smem_data, 0, 0)
            ll.mbarrier_arrive_and_expect_tx(barrier, TMA_2D_SMEM_BYTES)
        ll.mbarrier_wait(barrier, phase)
        phase = phase ^ 1

    end: ll.uint64 = ll.clock64()

    if tid == 0:
        elapsed[0] = end - start


# =============================================================================
# Kernel 3: Manual Copy Baseline (for comparison)
# =============================================================================

MANUAL_ITERS = 500


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_manual_copy(
    src: ll.Tensor[ll.float32],
    elapsed: ll.Tensor[ll.uint64],
    num_elems: ll.int32,
) -> ll.void:
    smem_data = ll.empty([16384], dtype=ll.float32, scope="shared")

    tid: ll.int32 = ll.threadIdx_x()

    ll.__syncthreads()

    start: ll.uint64 = ll.clock64()

    for _iter in range(MANUAL_ITERS):
        i: ll.int32 = tid
        while i < num_elems:
            smem_data[i] = src[i]
            i = i + ll.blockDim_x()
        ll.__syncthreads()

    end: ll.uint64 = ll.clock64()

    if tid == 0:
        elapsed[0] = end - start


# =============================================================================
# Kernel 4: TMA Single-Load Latency
# =============================================================================

LATENCY_ITERS = 1000


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_tma_latency(
    src: ll.Tensor[ll.float32],
    elapsed: ll.Tensor[ll.uint64],
    load_bytes: ll.int32,
) -> ll.void:
    smem_data = ll.empty([1024], dtype=ll.uint8, scope="shared")
    barrier = ll.empty([1], dtype=ll.uint64, scope="shared")

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid // 32

    if tid == 0:
        ll.init_smem_barrier(barrier, 1)
        ll.fence_smem_barrier_init()
    ll.__syncthreads()

    phase: ll.int32 = 0

    # Warmup
    if warp_idx == 0 and ll.elect_one_sync():
        ll.tma_copy_1d_g2s(src, barrier, smem_data, load_bytes)
        ll.mbarrier_arrive_and_expect_tx(barrier, load_bytes)
    ll.mbarrier_wait(barrier, phase)
    phase = phase ^ 1
    ll.__syncthreads()

    start: ll.uint64 = ll.clock64()

    for _iter in range(LATENCY_ITERS):
        if warp_idx == 0 and ll.elect_one_sync():
            ll.tma_copy_1d_g2s(src, barrier, smem_data, load_bytes)
            ll.mbarrier_arrive_and_expect_tx(barrier, load_bytes)
        ll.mbarrier_wait(barrier, phase)
        phase = phase ^ 1

    end: ll.uint64 = ll.clock64()

    if tid == 0:
        elapsed[0] = end - start


# =============================================================================
# Host benchmark functions
# =============================================================================


def run_tma_1d_bandwidth(gpu_info, csv_w):
    """Benchmark TMA 1D bulk copy with various transfer sizes."""
    arch = gpu_info["sm_arch"]
    clock_mhz = gpu_info["clock_mhz"]
    passes = PASSES["cuda"]
    BLOCK = 128

    rows = []
    # Transfer sizes: 128B to 16KB
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    for load_bytes in sizes:
        # Allocate source in global memory
        num_f32 = max(load_bytes // 4, 4096)
        src = torch.randn(num_f32, dtype=torch.float32, device="cuda")
        elapsed = torch.zeros(1, dtype=torch.int64, device="cuda")

        smem_size = 16384 + 128  # smem_data + barrier
        kernel = kernel_tma_1d_bandwidth.build(
            passes,
            codegen_cuda,
            grid=(1, 1, 1),
            block=(BLOCK, 1, 1),
            shared_mem_bytes=smem_size,
            arch=arch,
        )

        # Warmup + measure
        kernel(src, elapsed, load_bytes)
        torch.cuda.synchronize()
        kernel(src, elapsed, load_bytes)
        torch.cuda.synchronize()

        cyc = elapsed[0].item()
        cyc_per_load = cyc / COPY_1D_ITERS
        total_bytes = COPY_1D_ITERS * load_bytes
        seconds = cyc / (clock_mhz * 1e6)
        bw_gbs = total_bytes / seconds / 1e9

        csv_w.writerow([load_bytes, str(cyc), "{:.1f}".format(cyc_per_load), "{:.2f}".format(bw_gbs)])
        rows.append(["{} B".format(load_bytes), "{:.1f}".format(cyc_per_load), format_bandwidth(bw_gbs)])

        del src

    return rows


def run_tma_2d_bandwidth(gpu_info, csv_w):
    """Benchmark TMA 2D load with descriptor-based transfers."""
    arch = gpu_info["sm_arch"]
    clock_mhz = gpu_info["clock_mhz"]
    passes = PASSES["cuda"]
    BLOCK = 128

    rows = []
    # Test various tile sizes
    tile_configs = [
        (64, 32, "64x32"),
        (64, 64, "64x64"),
        (128, 64, "128x64"),
        (128, 128, "128x128"),
    ]

    for bm, bn, label in tile_configs:
        # For 128B swizzle, smem_inner_dim is auto-clamped
        M_global = max(bm * 4, 1024)
        N_global = max(bn * 4, 1024)

        src = torch.randn(M_global, N_global, dtype=torch.bfloat16, device="cuda")
        elapsed = torch.zeros(1, dtype=torch.int64, device="cuda")

        tensor_map = create_tma_2d_descriptor(
            src,
            gmem_inner_dim=N_global,
            gmem_outer_dim=M_global,
            smem_inner_dim=bn,
            smem_outer_dim=bm,
            gmem_outer_stride=N_global,
            swizzle_mode=128,
            oob_fill=True,
        )

        # Rebuild kernel constants for this tile size
        # For simplicity, use the fixed 128x64 kernel
        # and only run the full test for the default size
        if bm == TMA_2D_BLOCK_M and bn == TMA_2D_BLOCK_N:
            kernel = kernel_tma_2d_bandwidth.build(
                passes,
                codegen_cuda,
                grid=(1, 1, 1),
                block=(BLOCK, 1, 1),
                shared_mem_bytes=TMA_2D_SMEM_BYTES + 256,
                arch=arch,
            )

            kernel(tensor_map, elapsed)
            torch.cuda.synchronize()
            kernel(tensor_map, elapsed)
            torch.cuda.synchronize()

            cyc = elapsed[0].item()
            cyc_per_load = cyc / TMA_2D_ITERS
            tile_bytes = bm * bn * 2  # bf16
            total_bytes = TMA_2D_ITERS * tile_bytes
            seconds = cyc / (clock_mhz * 1e6)
            bw_gbs = total_bytes / seconds / 1e9

            csv_w.writerow([label, tile_bytes, str(cyc), "{:.1f}".format(cyc_per_load), "{:.2f}".format(bw_gbs)])
            rows.append([label, "{} B".format(tile_bytes), "{:.1f} cyc".format(cyc_per_load), format_bandwidth(bw_gbs)])

        del src

    return rows


def run_manual_vs_tma(gpu_info, csv_w):
    """Compare TMA 1D vs manual shared memory copy."""
    arch = gpu_info["sm_arch"]
    passes = PASSES["cuda"]
    BLOCK = 128

    rows = []
    test_sizes = [1024, 4096, 16384]  # bytes

    for size_bytes in test_sizes:
        num_f32 = size_bytes // 4
        src = torch.randn(max(num_f32, 4096), dtype=torch.float32, device="cuda")

        # TMA 1D
        elapsed_tma = torch.zeros(1, dtype=torch.int64, device="cuda")
        k_tma = kernel_tma_1d_bandwidth.build(
            passes,
            codegen_cuda,
            grid=(1, 1, 1),
            block=(BLOCK, 1, 1),
            shared_mem_bytes=16384 + 128,
            arch=arch,
        )
        k_tma(src, elapsed_tma, size_bytes)
        torch.cuda.synchronize()
        k_tma(src, elapsed_tma, size_bytes)
        torch.cuda.synchronize()
        tma_cyc = elapsed_tma[0].item() / COPY_1D_ITERS

        # Manual copy
        elapsed_man = torch.zeros(1, dtype=torch.int64, device="cuda")
        k_man = kernel_manual_copy.build(
            passes,
            codegen_cuda,
            grid=(1, 1, 1),
            block=(BLOCK, 1, 1),
            shared_mem_bytes=16384 * 4,
            arch=arch,
        )
        k_man(src, elapsed_man, num_f32)
        torch.cuda.synchronize()
        k_man(src, elapsed_man, num_f32)
        torch.cuda.synchronize()
        man_cyc = elapsed_man[0].item() / MANUAL_ITERS

        speedup = man_cyc / tma_cyc if tma_cyc > 0 else 0

        csv_w.writerow([size_bytes, "{:.1f}".format(tma_cyc), "{:.1f}".format(man_cyc), "{:.2f}x".format(speedup)])
        rows.append([
            "{} B".format(size_bytes), "{:.1f} cyc".format(tma_cyc), "{:.1f} cyc".format(man_cyc),
            "{:.2f}x".format(speedup)
        ])

        del src

    return rows


def run_tma_latency(gpu_info, csv_w):
    """Measure single TMA load latency for small transfers."""
    arch = gpu_info["sm_arch"]
    clock_mhz = gpu_info["clock_mhz"]
    passes = PASSES["cuda"]
    BLOCK = 128

    rows = []
    sizes = [128, 256, 512, 1024]

    for load_bytes in sizes:
        src = torch.randn(4096, dtype=torch.float32, device="cuda")
        elapsed = torch.zeros(1, dtype=torch.int64, device="cuda")

        kernel = kernel_tma_latency.build(
            passes,
            codegen_cuda,
            grid=(1, 1, 1),
            block=(BLOCK, 1, 1),
            shared_mem_bytes=1024 + 128,
            arch=arch,
        )

        kernel(src, elapsed, load_bytes)
        torch.cuda.synchronize()
        kernel(src, elapsed, load_bytes)
        torch.cuda.synchronize()

        cyc = elapsed[0].item()
        cyc_per_load = cyc / LATENCY_ITERS
        ns_per_load = cyc_per_load / (clock_mhz * 1e6) * 1e9

        csv_w.writerow([load_bytes, str(cyc), "{:.1f}".format(cyc_per_load), "{:.1f}".format(ns_per_load)])
        rows.append(["{} B".format(load_bytes), "{:.1f} cyc".format(cyc_per_load), "{:.1f} ns".format(ns_per_load)])

        del src

    return rows


# =============================================================================
# Main
# =============================================================================


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)

    print("\n[1/4] TMA 1D Bulk Copy Bandwidth")
    print("-" * 50)
    csv1 = CSVWriter(os.path.join(results_dir, "tma_1d_bandwidth.csv"),
                     ["load_bytes", "total_cycles", "cyc_per_load", "bandwidth_GBs"], gpu_info,
                     "TMA 1D Bulk Copy Bandwidth")
    rows1 = run_tma_1d_bandwidth(gpu_info, csv1)
    csv1.close()
    print_table(["Size", "Cyc/Load", "Bandwidth"], rows1, "TMA 1D Bulk Copy Bandwidth (Global -> Shared)")

    print("[2/4] TMA 2D Descriptor-Based Load Bandwidth")
    print("-" * 50)
    csv2 = CSVWriter(os.path.join(results_dir, "tma_2d_bandwidth.csv"),
                     ["tile_shape", "tile_bytes", "total_cycles", "cyc_per_load", "bandwidth_GBs"], gpu_info,
                     "TMA 2D Load Bandwidth")
    rows2 = run_tma_2d_bandwidth(gpu_info, csv2)
    csv2.close()
    if rows2:
        print_table(["Tile", "Size", "Cyc/Load", "Bandwidth"], rows2, "TMA 2D Load Bandwidth (128B Swizzle)")

    print("[3/4] TMA vs Manual Copy Comparison")
    print("-" * 50)
    csv3 = CSVWriter(os.path.join(results_dir, "tma_vs_manual.csv"),
                     ["size_bytes", "tma_cyc_per_load", "manual_cyc_per_load", "speedup"], gpu_info,
                     "TMA vs Manual Copy")
    rows3 = run_manual_vs_tma(gpu_info, csv3)
    csv3.close()
    print_table(["Size", "TMA", "Manual", "Speedup"], rows3, "TMA vs Manual Shared Memory Copy")

    print("[4/4] TMA Load Latency")
    print("-" * 50)
    csv4 = CSVWriter(os.path.join(results_dir, "tma_latency.csv"),
                     ["load_bytes", "total_cycles", "cyc_per_load", "ns_per_load"], gpu_info, "TMA Load Latency")
    rows4 = run_tma_latency(gpu_info, csv4)
    csv4.close()
    print_table(["Size", "Latency (cyc)", "Latency (ns)"], rows4, "TMA Single Load Latency")

    # Summary
    print("=" * 60)
    print("  TMA Benchmark Summary")
    print("=" * 60)
    if rows1:
        best = rows1[-1]
        print("  Best 1D bandwidth: {} ({})".format(best[2], best[0]))
    if rows4:
        fastest = rows4[0]
        print("  Min latency:       {} ({})".format(fastest[1], fastest[0]))
    if rows3:
        best_speedup = max(rows3, key=lambda r: float(r[3].rstrip('x')))
        print("  TMA vs Manual:     {} speedup ({})".format(best_speedup[3], best_speedup[0]))
    print("\n  Results saved to: {}/".format(results_dir))


if __name__ == "__main__":
    main()
