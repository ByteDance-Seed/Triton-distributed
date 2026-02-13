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
"""Occupancy Effects Benchmark for Hopper GPUs.
Measures throughput at different occupancy levels.
"""
import little_kernel as lk
import little_kernel.language as ll
import torch
from tqdm import tqdm
import os
from little_kernel.benchmark.utils import get_gpu_info, print_gpu_info, build_kernel, CSVWriter, get_results_dir, print_table

ITERS = 100000
OPS = 16


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_occupancy_fma(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.float32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    a: ll.float32 = 1.0001
    b: ll.float32 = 1.0001
    c0: ll.float32 = 0.0
    c1: ll.float32 = 0.1
    c2: ll.float32 = 0.2
    c3: ll.float32 = 0.3
    c4: ll.float32 = 0.4
    c5: ll.float32 = 0.5
    c6: ll.float32 = 0.6
    c7: ll.float32 = 0.7
    c8: ll.float32 = 0.8
    c9: ll.float32 = 0.9
    c10: ll.float32 = 1.0
    c11: ll.float32 = 1.1
    c12: ll.float32 = 1.2
    c13: ll.float32 = 1.3
    c14: ll.float32 = 1.4
    c15: ll.float32 = 1.5
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    ll.__syncthreads()
    s = ll.clock64()
    for i in range(ITERS):
        c0 = a * b + c0
        c1 = a * b + c1
        c2 = a * b + c2
        c3 = a * b + c3
        c4 = a * b + c4
        c5 = a * b + c5
        c6 = a * b + c6
        c7 = a * b + c7
        c8 = a * b + c8
        c9 = a * b + c9
        c10 = a * b + c10
        c11 = a * b + c11
        c12 = a * b + c12
        c13 = a * b + c13
        c14 = a * b + c14
        c15 = a * b + c15
    ll.__syncthreads()
    e = ll.clock64()
    dummy[tid] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15
    if tid == 0: elapsed[0] = e - s


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    arch = gpu_info["sm_arch"]
    clock_mhz = gpu_info["clock_mhz"]
    num_sms = gpu_info["sm_count"]
    rows = []
    csv_w = CSVWriter(os.path.join(results_dir, "occupancy.csv"),
                      ["threads_per_block", "blocks_per_SM", "total_cycles", "ops_per_cyc_per_SM", "GFLOPS"], gpu_info,
                      "Occupancy Effects")
    block_sizes = [32, 64, 128, 256, 512, 1024]
    for bs in tqdm(block_sizes, desc="Occupancy"):
        blocks_per_sm = min(2048 // bs, 32)
        total_blocks = num_sms * blocks_per_sm
        elapsed = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)
        dummy = torch.zeros(total_blocks * bs, dtype=torch.float32, device="cuda")
        kernel = build_kernel(kernel_occupancy_fma, grid=(total_blocks, 1, 1), block=(bs, 1, 1), arch=arch)
        kernel(elapsed, dummy)
        torch.cuda.synchronize()
        cyc = elapsed.item()
        total_ops = total_blocks * bs * ITERS * OPS * 2
        ops_sm = (ITERS * OPS * 2 * bs * blocks_per_sm) / cyc if cyc > 0 else 0
        gflops = total_ops / (cyc / (clock_mhz * 1e6)) / 1e9
        csv_w.writerow([bs, blocks_per_sm, cyc, "{:.2f}".format(ops_sm), "{:.1f}".format(gflops)])
        rows.append([bs, blocks_per_sm, "{:.1f}".format(ops_sm), "{:.1f}".format(gflops)])
    csv_w.close()
    print_table(["Block Size", "Blocks/SM", "Ops/Cyc/SM", "GFLOPS"], rows, "Occupancy vs Throughput")
    print("Results saved to: " + results_dir + "/")


if __name__ == "__main__":
    main()
