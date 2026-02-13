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
Register File Benchmark for Hopper GPUs.

Measures register file characteristics:
- Register read latency via dependent FMA chains (RAW hazard)
- Throughput degradation under register pressure
- Register count effects on occupancy and performance
"""

import little_kernel as lk
import little_kernel.language as ll
import torch
from tqdm import tqdm
import os
from little_kernel.benchmark.utils import (get_gpu_info, print_gpu_info, build_kernel, CSVWriter, get_results_dir,
                                           print_table)

# =============================================================================
# Constants
# =============================================================================
CHAIN_LENGTH = 10000
T = ll.float32

# =============================================================================
# Kernels
# =============================================================================


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_reg_latency_fma(
    elapsed: ll.Tensor[ll.uint64],
    dummy: ll.Tensor[ll.float32],
) -> ll.void:
    """Measure FMA register-to-register latency via serial dependency chain.
    
    Each FMA depends on the output of the previous one, so the latency
    is total_cycles / CHAIN_LENGTH.
    """
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0:
        return

    a: ll.float32 = 1.0001
    b: ll.float32 = 1.0001
    c: ll.float32 = 0.0
    start: ll.uint64 = 0
    end: ll.uint64 = 0

    start = ll.clock64()
    for i in range(CHAIN_LENGTH):
        c = a * b + c  # serial dependency: c -> c
    end = ll.clock64()

    dummy[0] = c
    elapsed[0] = end - start


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_reg_throughput_low_pressure(
    elapsed: ll.Tensor[ll.uint64],
    dummy: ll.Tensor[ll.float32],
) -> ll.void:
    """Measure FMA throughput with low register pressure (few live values).
    
    Independent FMA chains -- GPU can issue multiple per cycle.
    """
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0:
        return

    a0: ll.float32 = 1.0001
    b0: ll.float32 = 1.0001
    c0: ll.float32 = 0.0
    c1: ll.float32 = 0.0
    c2: ll.float32 = 0.0
    c3: ll.float32 = 0.0
    start: ll.uint64 = 0
    end: ll.uint64 = 0

    start = ll.clock64()
    for i in range(CHAIN_LENGTH):
        c0 = a0 * b0 + c0
        c1 = a0 * b0 + c1
        c2 = a0 * b0 + c2
        c3 = a0 * b0 + c3
    end = ll.clock64()

    dummy[0] = c0 + c1 + c2 + c3
    elapsed[0] = end - start


@lk.ll_kernel(backend="cuda", is_entry=True)
def kernel_reg_throughput_high_pressure(
    elapsed: ll.Tensor[ll.uint64],
    dummy: ll.Tensor[ll.float32],
) -> ll.void:
    """Measure FMA throughput with high register pressure (many live values).
    
    Many independent accumulators force high register usage.
    """
    tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid > 0:
        return

    a: ll.float32 = 1.0001
    b: ll.float32 = 1.0001
    c0: ll.float32 = 0.0
    c1: ll.float32 = 0.0
    c2: ll.float32 = 0.0
    c3: ll.float32 = 0.0
    c4: ll.float32 = 0.0
    c5: ll.float32 = 0.0
    c6: ll.float32 = 0.0
    c7: ll.float32 = 0.0
    c8: ll.float32 = 0.0
    c9: ll.float32 = 0.0
    c10: ll.float32 = 0.0
    c11: ll.float32 = 0.0
    c12: ll.float32 = 0.0
    c13: ll.float32 = 0.0
    c14: ll.float32 = 0.0
    c15: ll.float32 = 0.0
    start: ll.uint64 = 0
    end: ll.uint64 = 0

    start = ll.clock64()
    for i in range(CHAIN_LENGTH):
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
    end = ll.clock64()

    dummy[0] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15
    elapsed[0] = end - start


# =============================================================================
# Runner
# =============================================================================


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    arch = gpu_info["sm_arch"]

    rows = []
    csv = CSVWriter(os.path.join(results_dir, "register_file.csv"),
                    ["test", "total_cycles", "ops", "cycles_per_op", "ops_per_cycle"], gpu_info, "Register File")

    elapsed = torch.zeros(1, dtype=torch.int64, device="cuda").to(torch.uint64)
    dummy = torch.zeros(1, dtype=torch.float32, device="cuda")

    tests = [
        ("FMA serial (latency)", kernel_reg_latency_fma, CHAIN_LENGTH),
        ("FMA 4-wide (low reg)", kernel_reg_throughput_low_pressure, CHAIN_LENGTH * 4),
        ("FMA 16-wide (high reg)", kernel_reg_throughput_high_pressure, CHAIN_LENGTH * 16),
    ]

    for name, kern, total_ops in tqdm(tests, desc="Register benchmarks"):
        elapsed.zero_()
        dummy.zero_()
        kernel = build_kernel(kern, grid=(1, 1, 1), block=(1, 1, 1), arch=arch)
        kernel(elapsed, dummy)
        torch.cuda.synchronize()
        cyc = elapsed.item()
        cyc_per_op = cyc / total_ops
        ops_per_cyc = total_ops / cyc if cyc > 0 else 0.0
        csv.writerow([name, cyc, total_ops, f"{cyc_per_op:.3f}", f"{ops_per_cyc:.3f}"])
        rows.append([name, f"{cyc_per_op:.3f} cyc", f"{ops_per_cyc:.3f} ops/cyc"])

    csv.close()
    print_table(["Test", "Latency/Op", "Throughput"], rows, "Register File Characterization")
    print(f"\nResults saved to: {results_dir}/")


if __name__ == "__main__":
    main()
