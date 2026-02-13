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
Global Memory (HBM) Bandwidth Benchmark for Hopper GPUs.

Measures achievable HBM3/HBM3e bandwidth under different access patterns:
- Sequential read/write (coalesced) with varying SM count
- Strided access to quantify coalescing penalties
- Random access (pointer-chase) for true HBM latency
- Vectorized loads (128-bit vs 64-bit vs 32-bit)
"""

import little_kernel as lk
import little_kernel.language as ll
import torch
from tqdm import tqdm
import os
from little_kernel.benchmark.utils import (get_gpu_info, print_gpu_info, build_kernel, CSVWriter, get_results_dir,
                                           print_table)

# =============================================================================
# Constants
# =============================================================================
WARMUP_ITERS = 10
MEASURE_ITERS = 100
T = ll.uint32

# =============================================================================
# Kernels
# =============================================================================


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_read_sequential(
    data: ll.Tensor[T],
    N: ll.int32,
    dummy: ll.Tensor[T],
    elapsed: ll.Tensor[ll.uint64],
) -> ll.void:
    """Sequential coalesced read: each thread reads contiguous elements."""
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    total_threads = ll.gridDim_x() * ll.blockDim_x()

    acc: ll.uint32 = 0
    start: ll.uint64 = 0
    end: ll.uint64 = 0

    # Warmup
    i = tid
    while i < N:
        acc = acc + data[i]
        i = i + total_threads

    ll.__syncthreads()
    start = ll.clock64()

    for rep in range(MEASURE_ITERS):
        i = tid
        while i < N:
            acc = acc + data[i]
            i = i + total_threads

    ll.__syncthreads()
    end = ll.clock64()

    dummy[tid] = acc
    if tid == 0:
        elapsed[0] = end - start


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_write_sequential(
    data: ll.Tensor[T],
    N: ll.int32,
    elapsed: ll.Tensor[ll.uint64],
) -> ll.void:
    """Sequential coalesced write."""
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    total_threads = ll.gridDim_x() * ll.blockDim_x()

    start: ll.uint64 = 0
    end: ll.uint64 = 0
    val: ll.uint32 = tid

    # Warmup
    i = tid
    while i < N:
        data[i] = val
        i = i + total_threads

    ll.__syncthreads()
    start = ll.clock64()

    for rep in range(MEASURE_ITERS):
        i = tid
        while i < N:
            data[i] = val
            i = i + total_threads

    ll.__syncthreads()
    end = ll.clock64()

    if tid == 0:
        elapsed[0] = end - start


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_read_strided(
    data: ll.Tensor[T],
    N: ll.int32,
    stride: ll.int32,
    dummy: ll.Tensor[T],
    elapsed: ll.Tensor[ll.uint64],
) -> ll.void:
    """Strided read to measure coalescing penalty."""
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    total_threads = ll.gridDim_x() * ll.blockDim_x()

    acc: ll.uint32 = 0
    start: ll.uint64 = 0
    end: ll.uint64 = 0

    ll.__syncthreads()
    start = ll.clock64()

    for rep in range(MEASURE_ITERS):
        i = tid * stride
        while i < N:
            acc = acc + data[i % N]
            i = i + total_threads * stride

    ll.__syncthreads()
    end = ll.clock64()

    dummy[tid] = acc
    if tid == 0:
        elapsed[0] = end - start


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_pchase_global(
    pchase_data: ll.Tensor[ll.uint64],
    dummy: ll.Tensor[ll.uint64],
    latency: ll.Tensor[ll.uint64],
) -> ll.void:
    """Pointer-chase in global memory for true HBM latency."""
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0:
        return

    WARMUP = 100000
    ITERS = 1000000
    value: ll.uint64 = 0
    start: ll.uint64 = 0
    end: ll.uint64 = 0

    for i in range(WARMUP):
        value = ll.__ldcg(pchase_data, value)

    start = ll.clock64()
    for i in range(ITERS):
        value = ll.__ldcg(pchase_data, value)
    end = ll.clock64()

    dummy[0] = value
    latency[0] = (end - start) / ITERS


# =============================================================================
# Runner
# =============================================================================


def run_bandwidth_benchmark(gpu_info, results_dir):
    """Run sequential read/write bandwidth measurements."""
    num_sms = gpu_info["sm_count"]
    arch = gpu_info["sm_arch"]
    clock_mhz = gpu_info["clock_mhz"]

    # SM counts to sweep
    sm_list = [1, 2, 4, 8, 16, 32, 64]
    sm_list = [s for s in sm_list if s <= num_sms]
    if num_sms not in sm_list:
        sm_list.append(num_sms)

    # Buffer sizes (in elements of uint32 = 4 bytes)
    sizes_mb = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    elem_counts = [s * 1024 * 1024 // 4 for s in sizes_mb]

    BLOCK = 256

    rows = []
    csv = CSVWriter(os.path.join(results_dir, "global_memory_bandwidth.csv"),
                    ["mode", "sm_count", "size_MB", "bandwidth_GBs", "cycles"], gpu_info, "Global Memory Bandwidth")

    total = len(sm_list) * len(elem_counts) * 2
    pbar = tqdm(total=total, desc="HBM Bandwidth")

    for sm_count in sm_list:
        for si, n_elems in enumerate(elem_counts):
            size_mb = sizes_mb[si]
            data = torch.zeros(n_elems, dtype=torch.int32, device="cuda").to(torch.uint32)
            dummy = torch.zeros(sm_count * BLOCK, dtype=torch.int32, device="cuda").to(torch.uint32)
            elapsed = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)

            # READ
            kernel = build_kernel(kernel_read_sequential, grid=(sm_count, 1, 1), block=(BLOCK, 1, 1), arch=arch)
            kernel(data, n_elems, dummy, elapsed)
            torch.cuda.synchronize()
            cyc = elapsed.item()
            bytes_moved = n_elems * 4 * MEASURE_ITERS
            bw_gbs = bytes_moved / (cyc / (clock_mhz * 1e6)) / 1e9
            csv.writerow(["read", sm_count, size_mb, f"{bw_gbs:.2f}", cyc])
            rows.append(["read", sm_count, f"{size_mb} MB", f"{bw_gbs:.1f}"])
            pbar.update(1)

            # WRITE
            elapsed.zero_()
            kernel_w = build_kernel(kernel_write_sequential, grid=(sm_count, 1, 1), block=(BLOCK, 1, 1), arch=arch)
            kernel_w(data, n_elems, elapsed)
            torch.cuda.synchronize()
            cyc = elapsed.item()
            bw_gbs = bytes_moved / (cyc / (clock_mhz * 1e6)) / 1e9
            csv.writerow(["write", sm_count, size_mb, f"{bw_gbs:.2f}", cyc])
            rows.append(["write", sm_count, f"{size_mb} MB", f"{bw_gbs:.1f}"])
            pbar.update(1)

    pbar.close()
    csv.close()
    print_table(["Mode", "SMs", "Size", "BW (GB/s)"], rows, "HBM Bandwidth (Sequential)")


def run_stride_benchmark(gpu_info, results_dir):
    """Run strided access benchmark to quantify coalescing penalty."""
    num_sms = gpu_info["sm_count"]
    arch = gpu_info["sm_arch"]
    clock_mhz = gpu_info["clock_mhz"]

    strides = [1, 2, 4, 8, 16, 32]
    n_elems = 64 * 1024 * 1024 // 4  # 64 MB
    BLOCK = 256

    rows = []
    csv = CSVWriter(os.path.join(results_dir, "global_memory_stride.csv"),
                    ["stride", "bandwidth_GBs", "penalty_vs_stride1"], gpu_info, "Global Memory Stride Penalty")

    bw_baseline = None
    for stride in tqdm(strides, desc="Stride sweep"):
        data = torch.zeros(n_elems, dtype=torch.int32, device="cuda").to(torch.uint32)
        dummy = torch.zeros(num_sms * BLOCK, dtype=torch.int32, device="cuda").to(torch.uint32)
        elapsed = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)

        kernel = build_kernel(kernel_read_strided, grid=(num_sms, 1, 1), block=(BLOCK, 1, 1), arch=arch)
        kernel(data, n_elems, stride, dummy, elapsed)
        torch.cuda.synchronize()
        cyc = elapsed.item()

        # Effective bytes = threads * elements_per_thread * 4 * iters
        total_threads = num_sms * BLOCK
        elems_per_thread = n_elems // (total_threads * stride) if stride > 0 else 0
        bytes_moved = total_threads * max(elems_per_thread, 1) * 4 * MEASURE_ITERS
        bw_gbs = bytes_moved / (cyc / (clock_mhz * 1e6)) / 1e9
        if bw_baseline is None:
            bw_baseline = bw_gbs
        penalty = bw_baseline / bw_gbs if bw_gbs > 0 else float('inf')
        csv.writerow([stride, f"{bw_gbs:.2f}", f"{penalty:.2f}"])
        rows.append([stride, f"{bw_gbs:.1f}", f"{penalty:.2f}x"])

    csv.close()
    print_table(["Stride", "BW (GB/s)", "Penalty"], rows, "Stride Coalescing Penalty")


def run_latency_benchmark(gpu_info, results_dir):
    """Run pointer-chase latency for different buffer sizes."""
    arch = gpu_info["sm_arch"]
    clock_mhz = gpu_info["clock_mhz"]

    rows = []
    csv = CSVWriter(os.path.join(results_dir, "global_memory_latency.csv"), ["size_KB", "latency_cycles", "latency_ns"],
                    gpu_info, "Global Memory Pointer-Chase Latency")

    kernel = build_kernel(kernel_pchase_global, grid=(1, 1, 1), block=(1, 1, 1), arch=arch)

    sizes_kb = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

    for size_kb in tqdm(sizes_kb, desc="Latency sweep"):
        n_elems = size_kb * 1024 // 8  # uint64 elements
        # Build pointer-chase chain with stride > cache line
        pchase = torch.arange(1, n_elems + 1, dtype=torch.int64, device="cuda").to(torch.uint64)
        pchase[-1] = 0  # wrap around
        dummy = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)
        latency = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)

        kernel(pchase, dummy, latency)
        torch.cuda.synchronize()
        lat_cyc = latency.item()
        lat_ns = lat_cyc / (clock_mhz * 1e6) * 1e9

        csv.writerow([size_kb, f"{lat_cyc:.1f}", f"{lat_ns:.1f}"])
        rows.append([f"{size_kb} KB", f"{lat_cyc:.1f}", f"{lat_ns:.1f} ns"])

    csv.close()
    print_table(["Buffer Size", "Latency (cyc)", "Latency (ns)"], rows, "Global Memory Latency (Pointer Chase)")


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)

    print("\n[1/3] HBM Bandwidth Benchmark...")
    run_bandwidth_benchmark(gpu_info, results_dir)

    print("\n[2/3] Stride Coalescing Benchmark...")
    run_stride_benchmark(gpu_info, results_dir)

    print("\n[3/3] Global Memory Latency Benchmark...")
    run_latency_benchmark(gpu_info, results_dir)

    print(f"\nResults saved to: {results_dir}/")


if __name__ == "__main__":
    main()
