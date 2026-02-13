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
"""Memory Access Latency Benchmark for Hopper GPUs.
Unified latency staircase: SMEM < L1 < L2 < HBM.
"""
import little_kernel as lk
import little_kernel.language as ll
import torch
from tqdm import tqdm
import os
from little_kernel.benchmark.utils import get_gpu_info, print_gpu_info, build_kernel, CSVWriter, get_results_dir, print_table

ITERS = 1000000
SMEM_ELEMS = 1024


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_smem_latency(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.uint32]) -> ll.void:
    ll.alloc_shared_memory("smem", ll.uint32, SMEM_ELEMS, 128)
    tid = ll.threadIdx_x()
    i = tid
    while i < SMEM_ELEMS:
        smem[i] = (i + 1) % SMEM_ELEMS  # noqa: F821
        i = i + ll.blockDim_x()
    ll.__syncthreads()
    if tid > 0: return
    idx: ll.uint32 = 0
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        idx = smem[idx]  # noqa: F821
    e = ll.clock64()
    dummy[0] = idx
    elapsed[0] = (e - s) / ITERS


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_l1_latency(pchase: ll.Tensor[ll.uint64], elapsed: ll.Tensor[ll.uint64],
                      dummy: ll.Tensor[ll.uint64]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    idx: ll.uint64 = 0
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    for i in range(100000):
        idx = ll.__ldca(pchase, idx)
    s = ll.clock64()
    for i in range(ITERS):
        idx = ll.__ldca(pchase, idx)
    e = ll.clock64()
    dummy[0] = idx
    elapsed[0] = (e - s) / ITERS


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_l2_latency(pchase: ll.Tensor[ll.uint64], elapsed: ll.Tensor[ll.uint64],
                      dummy: ll.Tensor[ll.uint64]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    idx: ll.uint64 = 0
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    for i in range(100000):
        idx = ll.__ldcg(pchase, idx)
    s = ll.clock64()
    for i in range(ITERS):
        idx = ll.__ldcg(pchase, idx)
    e = ll.clock64()
    dummy[0] = idx
    elapsed[0] = (e - s) / ITERS


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    arch = gpu_info["sm_arch"]
    clock_mhz = gpu_info["clock_mhz"]
    rows = []
    csv_w = CSVWriter(os.path.join(results_dir, "mem_latency.csv"),
                      ["level", "size_KB", "latency_cycles", "latency_ns"], gpu_info, "Memory Latency")
    elapsed = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)
    dummy_u = torch.zeros(1, dtype=torch.int32, device="cuda").to(torch.uint32)
    kernel = build_kernel(kernel_smem_latency, grid=(1, 1, 1), block=(32, 1, 1), arch=arch)
    kernel(elapsed, dummy_u)
    torch.cuda.synchronize()
    lat = elapsed.item()
    lat_ns = lat / (clock_mhz * 1e6) * 1e9
    csv_w.writerow(["SMEM", SMEM_ELEMS * 4 // 1024, lat, "{:.1f}".format(lat_ns)])
    rows.append(["SMEM", "4 KB", "{:.1f} cyc".format(lat), "{:.1f} ns".format(lat_ns)])
    sizes_kb = [4, 16, 32, 128, 512, 2048, 8192, 32768, 131072]
    for size_kb in tqdm(sizes_kb, desc="Mem latency"):
        n = size_kb * 1024 // 8
        pchase = torch.arange(1, n + 1, dtype=torch.int64, device="cuda").to(torch.uint64)
        pchase[-1] = 0
        dummy_64 = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)
        elapsed.zero_()
        if size_kb <= 64:
            kern = kernel_l1_latency
            level = "L1"
        elif size_kb <= 4096:
            kern = kernel_l2_latency
            level = "L2"
        else:
            kern = kernel_l2_latency
            level = "HBM"
        kernel = build_kernel(kern, grid=(1, 1, 1), block=(1, 1, 1), arch=arch)
        kernel(pchase, elapsed, dummy_64)
        torch.cuda.synchronize()
        lat = elapsed.item()
        lat_ns = lat / (clock_mhz * 1e6) * 1e9
        csv_w.writerow([level, size_kb, lat, "{:.1f}".format(lat_ns)])
        rows.append([level, str(size_kb) + " KB", "{:.1f} cyc".format(lat), "{:.1f} ns".format(lat_ns)])
    csv_w.close()
    print_table(["Level", "Size", "Latency (cyc)", "Latency (ns)"], rows, "Memory Latency Staircase")
    print("Results saved to: " + results_dir + "/")


if __name__ == "__main__":
    main()
