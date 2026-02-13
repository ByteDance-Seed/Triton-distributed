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
"""Thread Block Cluster Benchmark for Hopper GPUs.
Measures cluster_sync latency.
"""
import little_kernel as lk
import little_kernel.language as ll
import torch
import os
from little_kernel.benchmark.utils import get_gpu_info, print_gpu_info, build_kernel, CSVWriter, get_results_dir, print_table

ITERS = 100000


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_cluster_sync_lat(elapsed: ll.Tensor[ll.uint64]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    ll.cluster_sync()
    s = ll.clock64()
    for i in range(ITERS):
        ll.cluster_sync()
    e = ll.clock64()
    if tid == 0: elapsed[0] = e - s


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    arch = gpu_info["sm_arch"]
    rows = []
    csv_w = CSVWriter(os.path.join(results_dir, "cluster_sync.csv"),
                      ["cluster_size", "total_cycles", "cycles_per_sync"], gpu_info, "Cluster Sync")
    elapsed = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)
    elapsed.zero_()
    kernel = build_kernel(kernel_cluster_sync_lat, grid=(1, 1, 1), block=(128, 1, 1), arch=arch)
    kernel(elapsed)
    torch.cuda.synchronize()
    cyc = elapsed.item()
    cpo = cyc / ITERS
    csv_w.writerow([1, cyc, "{:.2f}".format(cpo)])
    rows.append([1, "{:.2f} cyc".format(cpo)])
    csv_w.close()
    print_table(["Cluster Size", "Latency"], rows, "Cluster Sync Latency")
    print("Note: Multi-block clusters require cuLaunchKernelEx.")
    print("Results saved to: " + results_dir + "/")


if __name__ == "__main__":
    main()
