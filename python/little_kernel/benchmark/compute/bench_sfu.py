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
"""SFU Throughput and Latency Benchmark for Hopper GPUs."""
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.language.intrin.cuda_asm import sin_f32, cos_f32, rsqrt_f32, exp2_f32, log2_f32, rcp_f32
import torch
from tqdm import tqdm
import os
from little_kernel.benchmark.utils import get_gpu_info, print_gpu_info, build_kernel, CSVWriter, get_results_dir, print_table

ITERS = 100000


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_sfu_sin_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.float32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    x: ll.float32 = 0.5
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        x = sin_f32(x)
    e = ll.clock64()
    dummy[0] = x
    elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_sfu_cos_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.float32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    x: ll.float32 = 0.5
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        x = cos_f32(x)
    e = ll.clock64()
    dummy[0] = x
    elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_sfu_rsqrt_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.float32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    x: ll.float32 = 2.0
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        x = rsqrt_f32(x)
    e = ll.clock64()
    dummy[0] = x
    elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_sfu_exp2_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.float32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    x: ll.float32 = 0.01
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        x = exp2_f32(x)
    e = ll.clock64()
    dummy[0] = x
    elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_sfu_log2_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.float32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    x: ll.float32 = 2.0
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        x = log2_f32(x)
    e = ll.clock64()
    dummy[0] = x
    elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_sfu_rcp_lat(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.float32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0: return
    x: ll.float32 = 2.0
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    s = ll.clock64()
    for i in range(ITERS):
        x = rcp_f32(x)
    e = ll.clock64()
    dummy[0] = x
    elapsed[0] = e - s


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    arch = gpu_info["sm_arch"]
    rows = []
    csv_w = CSVWriter(os.path.join(results_dir, "sfu_performance.csv"),
                      ["function", "mode", "total_cycles", "cycles_per_op"], gpu_info, "SFU")
    elapsed = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)
    dummy = torch.zeros(1, dtype=torch.float32, device="cuda")
    tests = [("sin", kernel_sfu_sin_lat), ("cos", kernel_sfu_cos_lat), ("rsqrt", kernel_sfu_rsqrt_lat),
             ("exp2", kernel_sfu_exp2_lat), ("log2", kernel_sfu_log2_lat), ("rcp", kernel_sfu_rcp_lat)]
    for name, kern in tqdm(tests, desc="SFU latency"):
        elapsed.zero_()
        dummy.zero_()
        kernel = build_kernel(kern, grid=(1, 1, 1), block=(1, 1, 1), arch=arch)
        kernel(elapsed, dummy)
        torch.cuda.synchronize()
        cyc = elapsed.item()
        cpo = cyc / ITERS
        csv_w.writerow([name, "latency", cyc, "{:.2f}".format(cpo)])
        rows.append([name, "latency", "{:.2f} cyc".format(cpo)])
    csv_w.close()
    print_table(["Function", "Mode", "Result"], rows, "SFU Performance")
    print("Results saved to: " + results_dir + "/")


if __name__ == "__main__":
    main()
