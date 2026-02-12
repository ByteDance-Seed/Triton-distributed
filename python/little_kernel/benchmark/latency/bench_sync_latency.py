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
"""Synchronization Primitive Latency Benchmark for Hopper GPUs."""
import little_kernel as lk
import little_kernel.language as ll
import torch
from tqdm import tqdm
import os
from little_kernel.benchmark.utils import get_gpu_info, print_gpu_info, build_kernel, CSVWriter, get_results_dir, print_table

ITERS = 100000


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_syncthreads_lat(elapsed: ll.Tensor[ll.uint64]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    ll.__syncthreads()
    s = ll.clock64()
    for i in range(ITERS):
        ll.__syncthreads()
    e = ll.clock64()
    if tid == 0: elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_syncwarp_lat(elapsed: ll.Tensor[ll.uint64]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    ll.__syncwarp()
    s = ll.clock64()
    for i in range(ITERS):
        ll.__syncwarp()
    e = ll.clock64()
    if tid == 0: elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_threadfence_lat(elapsed: ll.Tensor[ll.uint64]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        ll.threadfence()
    e = ll.clock64()
    elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_atomic_add_lat(counter: ll.Tensor[ll.int32], elapsed: ll.Tensor[ll.uint64]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        ll.atomic_add(counter, 0, 1)
    e = ll.clock64()
    elapsed[0] = e - s


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    arch = gpu_info["sm_arch"]
    rows = []
    csv_w = CSVWriter(os.path.join(results_dir, "sync_latency.csv"),
                      ["primitive", "total_cycles", "iters", "cycles_per_op"], gpu_info, "Sync Latency")
    elapsed = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    BLOCK = 256
    tests = [
        ("__syncthreads", kernel_syncthreads_lat, (1, 1, 1), (BLOCK, 1, 1), [elapsed]),
        ("__syncwarp", kernel_syncwarp_lat, (1, 1, 1), (32, 1, 1), [elapsed]),
        ("threadfence", kernel_threadfence_lat, (1, 1, 1), (1, 1, 1), [elapsed]),
        ("atomicAdd", kernel_atomic_add_lat, (1, 1, 1), (1, 1, 1), [counter, elapsed]),
    ]
    for name, kern, grid, block, args in tqdm(tests, desc="Sync latency"):
        elapsed.zero_()
        counter.zero_()
        kernel = build_kernel(kern, grid=grid, block=block, arch=arch)
        kernel(*args)
        torch.cuda.synchronize()
        cyc = elapsed.item()
        cpo = cyc / ITERS
        csv_w.writerow([name, cyc, ITERS, "{:.2f}".format(cpo)])
        rows.append([name, "{:.2f} cyc".format(cpo)])
    csv_w.close()
    print_table(["Primitive", "Latency"], rows, "Synchronization Latency")
    print("Results saved to: " + results_dir + "/")


if __name__ == "__main__":
    main()
