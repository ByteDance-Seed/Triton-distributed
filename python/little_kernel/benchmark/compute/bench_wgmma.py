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
WGMMA (Warp Group Matrix Multiply-Accumulate) Throughput & Latency Benchmark
=============================================================================

Measures Hopper Tensor Core performance using WGMMA m64n256k16 (BF16->F32).

Benchmarks:
  1. WGMMA throughput: Maximum sustained WGMMA instructions per cycle per SM.
  2. WGMMA latency:   Cycles for a single WGMMA (fence->compute->commit->wait).
  3. WGMMA pipeline:  Throughput with multiple WGMMAs batched per group.
  4. GEMM roofline:   cuBLAS reference for full GEMM comparison.

WGMMA m64n256k16 bf16:
  - 2 * 64 * 256 * 16 = 524,288 FMA FLOPs per instruction
  - Accumulates into 128 float32 registers per warp group

Usage:
  python bench_wgmma.py
"""

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda
import torch
import os

from little_kernel.benchmark.utils import (get_gpu_info, print_gpu_info, CSVWriter, get_results_dir, print_table)

# =============================================================================
# Constants
# =============================================================================
WGMMA_M = 64
WGMMA_N = 256
WGMMA_K = 16
FLOPS_PER_WGMMA = 2 * WGMMA_M * WGMMA_N * WGMMA_K  # 524,288

# SMEM sizes for A (64x16 bf16) and B (256x16 bf16)
SMEM_A_BYTES = WGMMA_M * WGMMA_K * 2  # 2048
SMEM_B_BYTES = WGMMA_N * WGMMA_K * 2  # 8192
SMEM_AB_BYTES = SMEM_A_BYTES + SMEM_B_BYTES  # 10240

# WGMMA descriptor high bits for 128B swizzle
WGMMA_SBO = 8 * WGMMA_K * 2  # 256 bytes, swizzle base offset
DESC_HI = ((3 << 14) | (((WGMMA_SBO >> 4) & 0x3FFF) << 32) | (1 << 62))

# =============================================================================
# Kernel 1: WGMMA Throughput (sustained throughput per SM)
# =============================================================================
THROUGHPUT_ITERS = 2000


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_wgmma_throughput(
    elapsed: ll.Tensor[ll.uint64],
    sm_id_out: ll.Tensor[ll.uint32],
) -> ll.void:
    A_smem = ll.empty([WGMMA_M, WGMMA_K], dtype=ll.bfloat16, scope="shared")
    B_smem = ll.empty([WGMMA_N, WGMMA_K], dtype=ll.bfloat16, scope="shared")

    tid: ll.int32 = ll.threadIdx_x()
    bid: ll.int32 = ll.blockIdx_x()

    # Zero-init SMEM
    total_elems: ll.int32 = SMEM_AB_BYTES // 2
    i: ll.int32 = tid
    while i < total_elems:
        smem_ptr = ll.ptr_cast(A_smem, ll.bfloat16)
        smem_ptr[i] = ll.val_cast(0, ll.bfloat16)
        i = i + ll.blockDim_x()
    ll.__syncthreads()

    # Init accumulator: declares float acc[128] in generated CUDA
    ll.wgmma_init_accum()

    # Build SMEM descriptors for A and B
    base_desc_a: ll.uint32 = ll.cvta_generic_to_shared(A_smem) >> 4
    base_desc_b: ll.uint32 = ll.cvta_generic_to_shared(B_smem) >> 4
    desc_a: ll.uint64 = ll.uint64(base_desc_a & 0x3FFF) | DESC_HI
    desc_b: ll.uint64 = ll.uint64(base_desc_b & 0x3FFF) | DESC_HI

    ll.__syncthreads()

    start: ll.uint64 = ll.clock64()

    for _iter in range(THROUGHPUT_ITERS):
        ll.wgmma_fence()
        ll.wgmma_compute(desc_a, desc_b)
        ll.wgmma_commit()
        ll.wgmma_wait()

    end: ll.uint64 = ll.clock64()

    if tid == 0:
        elapsed[bid] = end - start
        sm_id_out[bid] = ll.smid()


# =============================================================================
# Kernel 2: WGMMA Latency (serialized single instruction)
# =============================================================================
LATENCY_ITERS = 500


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_wgmma_latency(elapsed: ll.Tensor[ll.uint64], ) -> ll.void:
    A_smem = ll.empty([WGMMA_M, WGMMA_K], dtype=ll.bfloat16, scope="shared")
    B_smem = ll.empty([WGMMA_N, WGMMA_K], dtype=ll.bfloat16, scope="shared")

    tid: ll.int32 = ll.threadIdx_x()

    total_elems: ll.int32 = SMEM_AB_BYTES // 2
    i: ll.int32 = tid
    while i < total_elems:
        smem_ptr = ll.ptr_cast(A_smem, ll.bfloat16)
        smem_ptr[i] = ll.val_cast(0, ll.bfloat16)
        i = i + ll.blockDim_x()
    ll.__syncthreads()

    ll.wgmma_init_accum()

    base_desc_a: ll.uint32 = ll.cvta_generic_to_shared(A_smem) >> 4
    base_desc_b: ll.uint32 = ll.cvta_generic_to_shared(B_smem) >> 4
    desc_a: ll.uint64 = ll.uint64(base_desc_a & 0x3FFF) | DESC_HI
    desc_b: ll.uint64 = ll.uint64(base_desc_b & 0x3FFF) | DESC_HI

    ll.__syncthreads()

    # Warmup
    for _w in range(10):
        ll.wgmma_fence()
        ll.wgmma_compute(desc_a, desc_b)
        ll.wgmma_commit()
        ll.wgmma_wait()

    ll.__syncthreads()

    start: ll.uint64 = ll.clock64()

    for _iter in range(LATENCY_ITERS):
        ll.wgmma_fence()
        ll.wgmma_compute(desc_a, desc_b)
        ll.wgmma_commit()
        ll.wgmma_wait()

    end: ll.uint64 = ll.clock64()

    if tid == 0:
        elapsed[0] = end - start


# =============================================================================
# Kernel 3: WGMMA Pipeline (4x batched per fence/commit/wait group)
# =============================================================================
PIPELINE_ITERS = 1000


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_wgmma_pipeline(elapsed: ll.Tensor[ll.uint64], ) -> ll.void:
    A_smem = ll.empty([WGMMA_M, WGMMA_K], dtype=ll.bfloat16, scope="shared")
    B_smem = ll.empty([WGMMA_N, WGMMA_K], dtype=ll.bfloat16, scope="shared")

    tid: ll.int32 = ll.threadIdx_x()

    total_elems: ll.int32 = SMEM_AB_BYTES // 2
    i: ll.int32 = tid
    while i < total_elems:
        smem_ptr = ll.ptr_cast(A_smem, ll.bfloat16)
        smem_ptr[i] = ll.val_cast(0, ll.bfloat16)
        i = i + ll.blockDim_x()
    ll.__syncthreads()

    ll.wgmma_init_accum()

    base_desc_a: ll.uint32 = ll.cvta_generic_to_shared(A_smem) >> 4
    base_desc_b: ll.uint32 = ll.cvta_generic_to_shared(B_smem) >> 4
    desc_a: ll.uint64 = ll.uint64(base_desc_a & 0x3FFF) | DESC_HI
    desc_b: ll.uint64 = ll.uint64(base_desc_b & 0x3FFF) | DESC_HI

    ll.__syncthreads()

    start: ll.uint64 = ll.clock64()

    for _iter in range(PIPELINE_ITERS):
        ll.wgmma_fence()
        # 4x WGMMA per group to saturate tensor core pipeline
        for _d in ll.unroll(range(4)):
            ll.wgmma_compute(desc_a, desc_b)
        ll.wgmma_commit()
        ll.wgmma_wait()

    end: ll.uint64 = ll.clock64()

    if tid == 0:
        elapsed[0] = end - start


# =============================================================================
# Host benchmark functions
# =============================================================================


def run_throughput_bench(gpu_info, csv_w, num_sms):
    """Measure WGMMA throughput across different SM counts."""
    arch = gpu_info["sm_arch"]
    clock_mhz = gpu_info["clock_mhz"]
    passes = PASSES["cuda"]

    BLOCK = 128  # 1 warp group
    smem_size = SMEM_AB_BYTES + 256

    rows = []
    sm_counts = sorted(set([1, 2, 4, 8, 16, 32, 64, min(132, num_sms), num_sms]))
    sm_counts = [s for s in sm_counts if s <= num_sms]

    for sm_count in sm_counts:
        elapsed = torch.zeros(sm_count, dtype=torch.int64, device="cuda")
        sm_ids = torch.zeros(sm_count, dtype=torch.int32, device="cuda")

        kernel = kernel_wgmma_throughput.build(
            passes,
            codegen_cuda,
            grid=(sm_count, 1, 1),
            block=(BLOCK, 1, 1),
            shared_mem_bytes=smem_size,
            arch=arch,
        )
        # Warmup
        kernel(elapsed, sm_ids)
        torch.cuda.synchronize()
        # Measure
        kernel(elapsed, sm_ids)
        torch.cuda.synchronize()

        max_cyc = elapsed.max().item()
        avg_cyc = elapsed.float().mean().item()
        total_flops_per_sm = THROUGHPUT_ITERS * FLOPS_PER_WGMMA
        seconds_avg = avg_cyc / (clock_mhz * 1e6)
        tflops_per_sm = total_flops_per_sm / seconds_avg / 1e12
        tflops_total = tflops_per_sm * sm_count
        cyc_per_wgmma = avg_cyc / THROUGHPUT_ITERS

        csv_w.writerow([
            sm_count, "{:.0f}".format(avg_cyc),
            str(max_cyc), "{:.1f}".format(cyc_per_wgmma), "{:.2f}".format(tflops_per_sm), "{:.2f}".format(tflops_total)
        ])
        rows.append(
            [sm_count, "{:.1f}".format(cyc_per_wgmma), "{:.2f}".format(tflops_per_sm), "{:.2f}".format(tflops_total)])

    return rows


def run_latency_bench(gpu_info, csv_w):
    """Measure single WGMMA instruction latency."""
    arch = gpu_info["sm_arch"]
    clock_mhz = gpu_info["clock_mhz"]
    passes = PASSES["cuda"]

    BLOCK = 128
    smem_size = SMEM_AB_BYTES + 256

    elapsed = torch.zeros(1, dtype=torch.int64, device="cuda")
    kernel = kernel_wgmma_latency.build(
        passes,
        codegen_cuda,
        grid=(1, 1, 1),
        block=(BLOCK, 1, 1),
        shared_mem_bytes=smem_size,
        arch=arch,
    )
    kernel(elapsed)
    torch.cuda.synchronize()
    kernel(elapsed)
    torch.cuda.synchronize()

    cyc = elapsed[0].item()
    cyc_per_wgmma = cyc / LATENCY_ITERS
    ns_per_wgmma = cyc_per_wgmma / (clock_mhz * 1e6) * 1e9

    csv_w.writerow(["single_wgmma", str(cyc), "{:.1f}".format(cyc_per_wgmma), "{:.1f}".format(ns_per_wgmma)])
    return cyc_per_wgmma, ns_per_wgmma


def run_pipeline_bench(gpu_info, csv_w):
    """Measure WGMMA pipeline throughput with 4 WGMMAs per group."""
    arch = gpu_info["sm_arch"]
    clock_mhz = gpu_info["clock_mhz"]
    passes = PASSES["cuda"]

    BLOCK = 128
    smem_size = SMEM_AB_BYTES + 256
    PIPELINE_DEPTH = 4

    elapsed = torch.zeros(1, dtype=torch.int64, device="cuda")
    kernel = kernel_wgmma_pipeline.build(
        passes,
        codegen_cuda,
        grid=(1, 1, 1),
        block=(BLOCK, 1, 1),
        shared_mem_bytes=smem_size,
        arch=arch,
    )
    kernel(elapsed)
    torch.cuda.synchronize()
    kernel(elapsed)
    torch.cuda.synchronize()

    cyc = elapsed[0].item()
    total_wgmmas = PIPELINE_ITERS * PIPELINE_DEPTH
    cyc_per_wgmma = cyc / total_wgmmas
    total_flops = total_wgmmas * FLOPS_PER_WGMMA
    seconds = cyc / (clock_mhz * 1e6)
    tflops = total_flops / seconds / 1e12

    csv_w.writerow([PIPELINE_DEPTH, str(cyc), "{:.2f}".format(cyc_per_wgmma), "{:.2f}".format(tflops)])
    return PIPELINE_DEPTH, cyc_per_wgmma, tflops


def run_gemm_roofline(gpu_info, csv_w):
    """Full GEMM benchmark using cuBLAS for reference roofline."""
    rows = []
    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    for M, N, K in shapes:
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

        for _ in range(3):
            torch.matmul(A, B.t())
        torch.cuda.synchronize()

        iters = 20
        s_evt = torch.cuda.Event(enable_timing=True)
        e_evt = torch.cuda.Event(enable_timing=True)

        s_evt.record()
        for _ in range(iters):
            torch.matmul(A, B.t())
        e_evt.record()
        torch.cuda.synchronize()

        ms = s_evt.elapsed_time(e_evt) / iters
        tflops = 2.0 * M * N * K / (ms * 1e9)
        csv_w.writerow(["{}x{}x{}".format(M, N, K), "{:.3f}".format(ms), "{:.2f}".format(tflops)])
        rows.append(["{}x{}x{}".format(M, N, K), "{:.3f} ms".format(ms), "{:.2f}".format(tflops)])

        del A, B
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return rows


# =============================================================================
# Main
# =============================================================================


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    num_sms = gpu_info["sm_count"]

    print("\n[1/4] WGMMA Throughput vs SM Count")
    print("-" * 50)
    csv1 = CSVWriter(os.path.join(results_dir, "wgmma_throughput.csv"),
                     ["sm_count", "avg_cycles", "max_cycles", "cyc_per_wgmma", "TFLOPS_per_SM", "TFLOPS_total"],
                     gpu_info, "WGMMA Throughput (m64n256k16 BF16)")
    rows1 = run_throughput_bench(gpu_info, csv1, num_sms)
    csv1.close()
    print_table(["SMs", "Cyc/WGMMA", "TFLOPS/SM", "TFLOPS Total"], rows1, "WGMMA Throughput (BF16->F32)")

    print("[2/4] WGMMA Latency (single instruction)")
    print("-" * 50)
    csv2 = CSVWriter(os.path.join(results_dir, "wgmma_latency.csv"),
                     ["mode", "total_cycles", "cyc_per_wgmma", "ns_per_wgmma"], gpu_info, "WGMMA Latency")
    lat_cyc, lat_ns = run_latency_bench(gpu_info, csv2)
    csv2.close()
    print("  WGMMA m64n256k16 latency: {:.1f} cycles ({:.1f} ns)".format(lat_cyc, lat_ns))
    print()

    print("[3/4] WGMMA Pipeline (4x batched per group)")
    print("-" * 50)
    csv3 = CSVWriter(os.path.join(results_dir, "wgmma_pipeline.csv"),
                     ["depth", "total_cycles", "cyc_per_wgmma", "TFLOPS"], gpu_info, "WGMMA Pipeline")
    depth, pipe_cyc, pipe_tflops = run_pipeline_bench(gpu_info, csv3)
    csv3.close()
    print("  Pipeline depth={}: {:.2f} cyc/WGMMA, {:.2f} TFLOPS/SM".format(depth, pipe_cyc, pipe_tflops))
    print()

    print("[4/4] GEMM Roofline (cuBLAS reference)")
    print("-" * 50)
    csv4 = CSVWriter(os.path.join(results_dir, "gemm_roofline.csv"), ["shape", "time_ms", "TFLOPS"], gpu_info,
                     "GEMM Roofline (cuBLAS)")
    rows4 = run_gemm_roofline(gpu_info, csv4)
    csv4.close()
    print_table(["Shape", "Time", "TFLOPS"], rows4, "GEMM Roofline (cuBLAS BF16)")

    # Summary
    print("=" * 60)
    print("  WGMMA Benchmark Summary")
    print("=" * 60)
    print("  WGMMA m64n256k16 shape: {}x{}x{}".format(WGMMA_M, WGMMA_N, WGMMA_K))
    print("  FLOPs per instruction:  {:,}".format(FLOPS_PER_WGMMA))
    print("  Single WGMMA latency:   {:.1f} cycles ({:.1f} ns)".format(lat_cyc, lat_ns))
    print("  Pipeline throughput:    {:.2f} cyc/WGMMA ({:.2f} TFLOPS/SM)".format(pipe_cyc, pipe_tflops))
    if rows1:
        last_row = rows1[-1]
        print("  Full-chip ({} SMs):  {} TFLOPS".format(last_row[0], last_row[3]))
    print("\n  Results saved to: {}/".format(results_dir))


if __name__ == "__main__":
    main()
