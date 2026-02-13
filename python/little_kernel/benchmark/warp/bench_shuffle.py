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
"""Warp Shuffle Performance Benchmark for Hopper GPUs."""
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.language.intrin.cuda_asm import shfl_sync_down, shfl_sync_up, shfl_sync_bfly
import torch
from tqdm import tqdm
import os
from little_kernel.benchmark.utils import get_gpu_info, print_gpu_info, build_kernel, CSVWriter, get_results_dir, print_table

ITERS = 1000000


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_shfl_down_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.uint32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    val: ll.uint32 = tid
    mask: ll.uint32 = 4294967295
    delta: ll.uint32 = 1
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        val = shfl_sync_down(mask, val, delta)
    e = ll.clock64()
    dummy[tid] = val
    if tid == 0: elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_shfl_up_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.uint32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    val: ll.uint32 = tid
    mask: ll.uint32 = 4294967295
    delta: ll.uint32 = 1
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        val = shfl_sync_up(mask, val, delta)
    e = ll.clock64()
    dummy[tid] = val
    if tid == 0: elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_shfl_bfly_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.uint32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    val: ll.uint32 = tid
    mask: ll.uint32 = 4294967295
    delta: ll.uint32 = 1
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        val = shfl_sync_bfly(mask, val, delta)
    e = ll.clock64()
    dummy[tid] = val
    if tid == 0: elapsed[0] = e - s


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    arch = gpu_info["sm_arch"]
    rows = []
    csv_w = CSVWriter(os.path.join(results_dir, "shuffle_latency.csv"),
                      ["variant", "total_cycles", "iters", "cycles_per_op"], gpu_info, "Shuffle Latency")
    elapsed = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)
    dummy = torch.zeros(32, dtype=torch.int32, device="cuda").to(torch.uint32)
    tests = [("shfl_down", kernel_shfl_down_lat), ("shfl_up", kernel_shfl_up_lat), ("shfl_bfly", kernel_shfl_bfly_lat)]
    for name, kern in tqdm(tests, desc="Shuffle"):
        elapsed.zero_()
        kernel = build_kernel(kern, grid=(1, 1, 1), block=(32, 1, 1), arch=arch)
        kernel(elapsed, dummy)
        torch.cuda.synchronize()
        cyc = elapsed.item()
        cpo = cyc / ITERS
        csv_w.writerow([name, cyc, ITERS, "{:.3f}".format(cpo)])
        rows.append([name, "{:.3f} cyc".format(cpo)])
    csv_w.close()
    print_table(["Variant", "Latency"], rows, "Warp Shuffle Latency")
    print("Results saved to: " + results_dir + "/")


if __name__ == "__main__":
    main()
