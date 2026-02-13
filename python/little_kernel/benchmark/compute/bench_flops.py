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
Arithmetic Throughput (FLOPS/IOPS) Benchmark for Hopper GPUs.
Measures peak achievable throughput per SM and per GPU.
"""

import little_kernel as lk
import little_kernel.language as ll
import torch
from tqdm import tqdm
import os
from little_kernel.benchmark.utils import (get_gpu_info, print_gpu_info, build_kernel, CSVWriter, get_results_dir,
                                           print_table, format_flops)

ITERS = 100000
OPS_PER_ITER = 16


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_fp32_fma_throughput(
    elapsed: ll.Tensor[ll.uint64],
    dummy: ll.Tensor[ll.float32],
) -> ll.void:
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
    start: ll.uint64 = 0
    end: ll.uint64 = 0
    ll.__syncthreads()
    start = ll.clock64()
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
    end = ll.clock64()
    dummy[tid] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15
    if tid == 0:
        elapsed[0] = end - start


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_fp32_add_throughput(
    elapsed: ll.Tensor[ll.uint64],
    dummy: ll.Tensor[ll.float32],
) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    inc: ll.float32 = 0.0001
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
    start: ll.uint64 = 0
    end: ll.uint64 = 0
    ll.__syncthreads()
    start = ll.clock64()
    for i in range(ITERS):
        c0 = c0 + inc
        c1 = c1 + inc
        c2 = c2 + inc
        c3 = c3 + inc
        c4 = c4 + inc
        c5 = c5 + inc
        c6 = c6 + inc
        c7 = c7 + inc
        c8 = c8 + inc
        c9 = c9 + inc
        c10 = c10 + inc
        c11 = c11 + inc
        c12 = c12 + inc
        c13 = c13 + inc
        c14 = c14 + inc
        c15 = c15 + inc
    ll.__syncthreads()
    end = ll.clock64()
    dummy[tid] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15
    if tid == 0:
        elapsed[0] = end - start


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_int32_add_throughput(
    elapsed: ll.Tensor[ll.uint64],
    dummy: ll.Tensor[ll.uint32],
    iters: ll.int32,
) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    inc: ll.uint32 = 1
    c0: ll.uint32 = 0
    c1: ll.uint32 = 1
    c2: ll.uint32 = 2
    c3: ll.uint32 = 3
    c4: ll.uint32 = 4
    c5: ll.uint32 = 5
    c6: ll.uint32 = 6
    c7: ll.uint32 = 7
    c8: ll.uint32 = 8
    c9: ll.uint32 = 9
    c10: ll.uint32 = 10
    c11: ll.uint32 = 11
    c12: ll.uint32 = 12
    c13: ll.uint32 = 13
    c14: ll.uint32 = 14
    c15: ll.uint32 = 15
    start: ll.uint64 = 0
    end: ll.uint64 = 0
    ll.__syncthreads()
    start = ll.clock64()
    for i in range(iters):
        c0 = c0 + inc
        c1 = c1 + inc
        c2 = c2 + inc
        c3 = c3 + inc
        c4 = c4 + inc
        c5 = c5 + inc
        c6 = c6 + inc
        c7 = c7 + inc
        c8 = c8 + inc
        c9 = c9 + inc
        c10 = c10 + inc
        c11 = c11 + inc
        c12 = c12 + inc
        c13 = c13 + inc
        c14 = c14 + inc
        c15 = c15 + inc
    ll.__syncthreads()
    end = ll.clock64()
    dummy[tid] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15
    if tid == 0:
        elapsed[0] = end - start


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    arch = gpu_info["sm_arch"]
    clock_mhz = gpu_info["clock_mhz"]
    num_sms = gpu_info["sm_count"]
    BLOCK = 256
    sm_list = [1, 2, 4, 8, 16, 32, 64]
    sm_list = [s for s in sm_list if s <= num_sms]
    if num_sms not in sm_list:
        sm_list.append(num_sms)
    benchmarks = [
        ("FP32_FMA", kernel_fp32_fma_throughput, torch.float32, 2),
        ("FP32_ADD", kernel_fp32_add_throughput, torch.float32, 1),
        ("INT32_ADD", kernel_int32_add_throughput, torch.int32, 1),
    ]
    rows = []
    csv_w = CSVWriter(
        os.path.join(results_dir, "compute_flops.csv"),
        ["instruction", "sm_count", "threads", "total_cycles", "ops_per_cycle_per_SM", "GFLOPS", "TFLOPS"], gpu_info,
        "Compute Throughput")
    total = len(benchmarks) * len(sm_list)
    pbar = tqdm(total=total, desc="Compute throughput")
    for bench_name, kern_func, dtype, flop_per_op in benchmarks:
        for sm_count in sm_list:
            elapsed = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)
            dummy = torch.zeros(sm_count * BLOCK, dtype=dtype, device="cuda")
            kernel = build_kernel(kern_func, grid=(sm_count, 1, 1), block=(BLOCK, 1, 1), arch=arch)
            if bench_name == "INT32_ADD":
                kernel(elapsed, dummy, ITERS)
            else:
                kernel(elapsed, dummy)
            torch.cuda.synchronize()
            cyc = elapsed.item()
            total_ops = sm_count * BLOCK * ITERS * OPS_PER_ITER * flop_per_op
            ops_per_cyc = (ITERS * OPS_PER_ITER * flop_per_op * BLOCK) / cyc if cyc > 0 else 0
            gflops = total_ops / (cyc / (clock_mhz * 1e6)) / 1e9
            tflops = gflops / 1000
            csv_w.writerow([
                bench_name, sm_count, sm_count * BLOCK, cyc, "{:.2f}".format(ops_per_cyc), "{:.1f}".format(gflops),
                "{:.3f}".format(tflops)
            ])
            if sm_count == num_sms:
                rows.append([bench_name, sm_count, "{:.1f}".format(ops_per_cyc), format_flops(tflops)])
            pbar.update(1)
    pbar.close()
    csv_w.close()
    print_table(["Instruction", "SMs", "Ops/Cyc/SM", "Peak"], rows, "Peak Compute Throughput (all SMs)")
    print("Results saved to: " + results_dir + "/")


if __name__ == "__main__":
    main()
