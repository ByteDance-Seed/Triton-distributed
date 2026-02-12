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
"""Instructions Per Cycle (IPC) Benchmark for Hopper GPUs.
Measures single-issue and dual-issue IPC for different instruction mixes.
"""
import little_kernel as lk
import little_kernel.language as ll
import torch
from tqdm import tqdm
import os
from little_kernel.benchmark.utils import get_gpu_info, print_gpu_info, build_kernel, CSVWriter, get_results_dir, print_table

ITERS = 100000


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_ipc_fp32_only(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.float32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    a: ll.float32 = 1.0001
    b: ll.float32 = 1.0001
    c0: ll.float32 = 0.0
    c1: ll.float32 = 0.1
    c2: ll.float32 = 0.2
    c3: ll.float32 = 0.3
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    ll.__syncthreads()
    s = ll.clock64()
    for i in range(ITERS):
        c0 = a * b + c0
        c1 = a * b + c1
        c2 = a * b + c2
        c3 = a * b + c3
    ll.__syncthreads()
    e = ll.clock64()
    dummy[tid] = c0 + c1 + c2 + c3
    if tid == 0: elapsed[0] = e - s


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_ipc_fp32_int32_mix(elapsed: ll.Tensor[ll.uint64], dummy: ll.Tensor[ll.float32]) -> ll.void:
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    a: ll.float32 = 1.0001
    b: ll.float32 = 1.0001
    fc0: ll.float32 = 0.0
    fc1: ll.float32 = 0.1
    ic0: ll.uint32 = 0
    ic1: ll.uint32 = 1
    inc: ll.uint32 = 1
    s: ll.uint64 = 0
    e: ll.uint64 = 0
    ll.__syncthreads()
    s = ll.clock64()
    for i in range(ITERS):
        fc0 = a * b + fc0
        ic0 = ic0 + inc
        fc1 = a * b + fc1
        ic1 = ic1 + inc
    ll.__syncthreads()
    e = ll.clock64()
    dummy[tid] = fc0 + fc1
    if tid == 0: elapsed[0] = e - s


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    arch = gpu_info["sm_arch"]
    rows = []
    csv_w = CSVWriter(os.path.join(results_dir, "ipc.csv"), ["mix", "total_cycles", "total_ops", "IPC"], gpu_info,
                      "IPC")
    elapsed = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)
    BLOCK = 256
    num_sms = gpu_info["sm_count"]
    tests = [("FP32_only", kernel_ipc_fp32_only, 4), ("FP32+INT32", kernel_ipc_fp32_int32_mix, 4)]
    for name, kern, ops_per_iter in tqdm(tests, desc="IPC"):
        elapsed.zero_()
        dummy = torch.zeros(num_sms * BLOCK, dtype=torch.float32, device="cuda")
        kernel = build_kernel(kern, grid=(num_sms, 1, 1), block=(BLOCK, 1, 1), arch=arch)
        kernel(elapsed, dummy)
        torch.cuda.synchronize()
        cyc = elapsed.item()
        total_ops = ITERS * ops_per_iter * BLOCK
        ipc = total_ops / cyc if cyc > 0 else 0
        csv_w.writerow([name, cyc, total_ops, "{:.2f}".format(ipc)])
        rows.append([name, "{:.2f}".format(ipc)])
    csv_w.close()
    print_table(["Mix", "IPC (ops/cyc/SM)"], rows, "Instructions Per Cycle")
    print("Results saved to: " + results_dir + "/")


if __name__ == "__main__":
    main()
