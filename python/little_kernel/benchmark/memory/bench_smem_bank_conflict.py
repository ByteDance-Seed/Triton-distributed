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
Shared Memory Bank Conflict Benchmark for Hopper GPUs.

Quantifies bank conflict penalties for shared memory access:
- Zero-conflict (stride=1 across 32 banks)
- 2-way, 4-way, 8-way, 16-way, 32-way conflicts
- Broadcast (all threads read same address)
"""

import little_kernel as lk
import little_kernel.language as ll
import torch
from tqdm import tqdm
import os
from little_kernel.benchmark.utils import (get_gpu_info, print_gpu_info, build_kernel, CSVWriter, get_results_dir,
                                           print_table)

ITERS = 100000
SMEM_ELEMS = 4096  # uint32 elements = 16 KB


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_smem_bank_stride(
    stride: ll.int32,
    dummy: ll.Tensor[ll.uint32],
    elapsed: ll.Tensor[ll.uint64],
) -> ll.void:
    """Access shared memory with given stride to induce bank conflicts."""
    ll.alloc_shared_memory("smem", ll.uint32, SMEM_ELEMS, 128)

    tid = ll.threadIdx_x()
    lane = ll.get_lane_idx()

    i = tid
    while i < SMEM_ELEMS:
        smem[i] = i  # noqa: F821
        i = i + ll.blockDim_x()
    ll.__syncthreads()

    acc: ll.uint32 = 0
    start: ll.uint64 = 0
    end: ll.uint64 = 0

    idx = (lane * stride) % SMEM_ELEMS

    start = ll.clock64()
    for rep in range(ITERS):
        acc = acc + smem[idx]  # noqa: F821
    end = ll.clock64()

    ll.__syncthreads()
    dummy[tid] = acc
    if tid == 0:
        elapsed[0] = end - start


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_smem_broadcast(
    dummy: ll.Tensor[ll.uint32],
    elapsed: ll.Tensor[ll.uint64],
) -> ll.void:
    """All threads read the same shared memory address (broadcast)."""
    ll.alloc_shared_memory("smem", ll.uint32, SMEM_ELEMS, 128)

    tid = ll.threadIdx_x()

    i = tid
    while i < SMEM_ELEMS:
        smem[i] = i  # noqa: F821
        i = i + ll.blockDim_x()
    ll.__syncthreads()

    acc: ll.uint32 = 0
    start: ll.uint64 = 0
    end: ll.uint64 = 0

    start = ll.clock64()
    for rep in range(ITERS):
        acc = acc + smem[0]  # noqa: F821
    end = ll.clock64()

    ll.__syncthreads()
    dummy[tid] = acc
    if tid == 0:
        elapsed[0] = end - start


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    arch = gpu_info["sm_arch"]

    BLOCK = 256
    rows = []
    csv_w = CSVWriter(os.path.join(results_dir, "smem_bank_conflict.csv"),
                      ["pattern", "stride", "total_cycles", "cycles_per_access", "slowdown_vs_noconflict"], gpu_info,
                      "Shared Memory Bank Conflicts")

    dummy = torch.zeros(BLOCK, dtype=torch.int32, device="cuda").to(torch.uint32)
    elapsed = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)

    baseline_cyc = None

    strides = [1, 2, 4, 8, 16, 32]
    for stride in tqdm(strides, desc="Stride sweep"):
        elapsed.zero_()
        kernel = build_kernel(kernel_smem_bank_stride, grid=(1, 1, 1), block=(BLOCK, 1, 1), arch=arch)
        kernel(stride, dummy, elapsed)
        torch.cuda.synchronize()
        cyc = elapsed.item()
        cyc_per_access = cyc / ITERS

        if baseline_cyc is None:
            baseline_cyc = cyc_per_access
        slowdown = cyc_per_access / baseline_cyc if baseline_cyc > 0 else 0

        if stride == 1:
            pattern = "no-conflict"
        else:
            pattern = str(stride) + "-way"

        csv_w.writerow([pattern, stride, cyc, "{:.2f}".format(cyc_per_access), "{:.2f}".format(slowdown)])
        rows.append([pattern, stride, "{:.2f} cyc".format(cyc_per_access), "{:.2f}x".format(slowdown)])

    # Broadcast test
    elapsed.zero_()
    kernel_bc = build_kernel(kernel_smem_broadcast, grid=(1, 1, 1), block=(BLOCK, 1, 1), arch=arch)
    kernel_bc(dummy, elapsed)
    torch.cuda.synchronize()
    cyc = elapsed.item()
    cyc_per_access = cyc / ITERS
    slowdown = cyc_per_access / baseline_cyc if baseline_cyc and baseline_cyc > 0 else 0
    csv_w.writerow(["broadcast", 0, cyc, "{:.2f}".format(cyc_per_access), "{:.2f}".format(slowdown)])
    rows.append(["broadcast", "all", "{:.2f} cyc".format(cyc_per_access), "{:.2f}x".format(slowdown)])

    csv_w.close()
    print_table(["Pattern", "Stride", "Latency", "Slowdown"], rows, "Shared Memory Bank Conflict Analysis")
    print("Results saved to: " + results_dir + "/")


if __name__ == "__main__":
    main()
