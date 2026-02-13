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
"""Arithmetic Instruction Latency Benchmark for Hopper GPUs.
Measures RAW dependency latency for FP32 FMA/ADD/MUL, INT32 ADD/MUL.
"""
import little_kernel as lk
import little_kernel.language as ll
import torch
from tqdm import tqdm
import os
from little_kernel.benchmark.utils import get_gpu_info, print_gpu_info, build_kernel, CSVWriter, get_results_dir, print_table

ITERS = 1000000


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_fp32_fma_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.float32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    a: ll.float32 = 1.0001
    b: ll.float32 = 1.0001
    c: ll.float32 = 0.0
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        c = a * b + c
    e = ll.clock64()
    dummy[0] = c
    elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_fp32_add_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.float32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    c: ll.float32 = 0.0
    inc: ll.float32 = 0.0001
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        c = c + inc
    e = ll.clock64()
    dummy[0] = c
    elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_fp32_mul_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.float32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    c: ll.float32 = 1.0001
    m: ll.float32 = 1.000001
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        c = c * m
    e = ll.clock64()
    dummy[0] = c
    elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_int32_add_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.uint32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    c: ll.uint32 = 0
    inc: ll.uint32 = 1
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        c = c + inc
    e = ll.clock64()
    dummy[0] = c
    elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_int32_mul_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.uint32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    c: ll.uint32 = 3
    m: ll.uint32 = 3
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        c = c * m
    e = ll.clock64()
    dummy[0] = c
    elapsed[0] = e - s


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    arch = gpu_info["sm_arch"]
    rows = []
    csv_w = CSVWriter(os.path.join(results_dir, "arith_latency.csv"),
                      ["instruction", "total_cycles", "iters", "cycles_per_op"], gpu_info, "Arithmetic Latency")
    elapsed = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)
    dummy_f = torch.zeros(1, dtype=torch.float32, device="cuda")
    dummy_i = torch.zeros(1, dtype=torch.int32, device="cuda").to(torch.uint32)
    tests = [("FP32_FMA", kernel_fp32_fma_lat, dummy_f), ("FP32_ADD", kernel_fp32_add_lat, dummy_f),
             ("FP32_MUL", kernel_fp32_mul_lat, dummy_f), ("INT32_ADD", kernel_int32_add_lat, dummy_i),
             ("INT32_MUL", kernel_int32_mul_lat, dummy_i)]
    for name, kern, dummy in tqdm(tests, desc="Arith latency"):
        elapsed.zero_()
        kernel = build_kernel(kern, grid=(1, 1, 1), block=(1, 1, 1), arch=arch)
        kernel(elapsed, dummy)
        torch.cuda.synchronize()
        cyc = elapsed.item()
        cpo = cyc / ITERS
        csv_w.writerow([name, cyc, ITERS, "{:.3f}".format(cpo)])
        rows.append([name, "{:.3f} cyc".format(cpo)])
    csv_w.close()
    print_table(["Instruction", "Latency"], rows, "Arithmetic Instruction Latency")
    print("Results saved to: " + results_dir + "/")


if __name__ == "__main__":
    main()
