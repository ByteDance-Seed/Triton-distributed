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

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda
import torch
import csv
from tqdm import tqdm

WarmUps = 1000_000
Iterations = 10_000_000
T = ll.uint64


@lk.ll_kernel(backend="cuda", is_entry=True)
def bench_l1_cache_latency_kernel(
    pchase_data: ll.Tensor[T],
    dummy: ll.Tensor[T],
    latency: ll.Tensor[T],
) -> ll.void:
    global_tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if global_tid > 0:
        return
    start_clk: ll.uint64 = 0
    end_clk: ll.uint64 = 0
    value: T = 0

    for i in range(WarmUps):
        value = ll.__ldca(pchase_data, value)

    start_clk = ll.clock64()
    for i in range(Iterations):
        value = ll.__ldca(pchase_data, value)
    dummy[0] = value
    end_clk = ll.clock64()
    latency[0] = (end_clk - start_clk) / Iterations


@lk.ll_kernel(backend="cuda", is_entry=True)
def bench_l1_cache_throughput_kernel(
    pchase_data: ll.Tensor[T],
    N: ll.int32,
    dummy: ll.Tensor[T],
    throughput: ll.Tensor[ll.float32],
) -> ll.void:
    global_tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    value: T = pchase_data[global_tid % N]
    start_clk: ll.uint64 = 0
    end_clk: ll.uint64 = 0

    for i in range(WarmUps):
        value = ll.__ldca(pchase_data, value)

    start_clk = ll.clock64()
    for i in range(Iterations):
        value = ll.__ldca(pchase_data, value)
    dummy[0] = value
    end_clk = ll.clock64()
    throughput[global_tid] = 1.0 * Iterations * ll.sizeof(T) / (end_clk - start_clk)  # bytes/cycle


if __name__ == "__main__":
    passes = PASSES["cuda"]
    code = bench_l1_cache_latency_kernel.compile(passes, codegen_cuda)
    print(code)

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    sm_arch = torch.cuda.get_device_capability()
    sm_arch = f"sm_{sm_arch[0]}{sm_arch[1]}"

    gpu_name = torch.cuda.get_device_name().replace(" ", "_")
    print(f"GPU: {gpu_name}")

    import subprocess
    import re

    def get_gpu_frequency():
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,nounits,noheader", "-i", "0"],
                                    capture_output=True, text=True, check=True)
            freq_str = result.stdout.strip()
            freq_mhz = float(re.findall(r"\d+", freq_str)[0])
            return freq_mhz
        except Exception as e:
            print("Failed to get GPU frequency:", e)
            return None

    gpu_freq = get_gpu_frequency()
    print(f"GPU SM Frequency: {gpu_freq} MHz")

    grid = (1, 1, 1)
    block = (1, 1, 1)
    smem_size = 0
    kernel = bench_l1_cache_latency_kernel.build(
        passes,
        codegen_cuda,
        grid=grid,
        block=block,
        shared_mem_bytes=smem_size,
        arch=sm_arch,  # Hopper architecture
        verbose=True,
    )

    # benchmark latency
    latency_file = f"l1_cache_latency_{gpu_name}.csv"
    with open(latency_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["array_size(KB)", "latency(cycle)", "latency(ns)"])
        f.flush()  # Write header immediately

        for i in tqdm(range(20), desc="Latency benchmark"):
            array_size = 1024 * (2**i)
            pchase_data = torch.arange(1, array_size + 1, dtype=torch.int32, device="cuda").to(torch.uint64)
            dummy = torch.zeros(1, dtype=torch.uint64, device="cuda")
            pchase_data[-1].fill_(0)
            latency = torch.zeros(1, dtype=torch.uint64, device="cuda")
            kernel(pchase_data, dummy, latency)
            result = [array_size * 8 / 1024, latency.item(), latency.item() / (gpu_freq * 1e6) * 1e9]
            writer.writerow(result)
            f.flush()  # Write immediately after each result
            tqdm.write(f"Array size: {result[0]:.1f} KB, Latency: {result[1]:.2f} cycles, {result[2]:.2f} ns")

    # benchmark throughput
    throughput_file = f"l1_cache_throughput_{gpu_name}.csv"
    sm_range = [2**i if 2**i <= num_sms else -1 for i in range(9)]
    if -1 in sm_range:
        sm_range.remove(-1)
    if num_sms not in sm_range:
        sm_range.append(num_sms)
    total_iterations = len(sm_range) * 20

    with open(throughput_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["sm", "array_size(KB)", "throughput(GB/s)"])
        f.flush()  # Write header immediately

        pbar = tqdm(total=total_iterations, desc="Throughput benchmark")
        for sm in sm_range:
            grid = (sm, 1, 1)
            block = (1024, 1, 1)
            smem_size = 0
            kernel = bench_l1_cache_throughput_kernel.build(
                passes,
                codegen_cuda,
                grid=grid,
                block=block,
                shared_mem_bytes=smem_size,
                arch=sm_arch,  # Hopper architecture
                verbose=True,
            )

            for i in range(20):
                array_size = 1024 * (2**i)
                pchase_data = torch.arange(1, array_size + 1, dtype=torch.int32, device="cuda").to(torch.uint64)
                dummy = torch.zeros(1, dtype=torch.uint64, device="cuda")
                pchase_data[-1].fill_(0)
                throughput = torch.zeros(sm * 1024, dtype=torch.float32, device="cuda")
                kernel(pchase_data, array_size, dummy, throughput)
                result = [
                    sm, array_size * 8 / 1024,
                    throughput.float().mean().item() * 1024 * sm / 1e9 * gpu_freq * 1e6
                ]
                writer.writerow(result)
                f.flush()  # Write immediately after each result
                pbar.update(1)
                pbar.set_postfix({"SM": sm, "Size(KB)": f"{result[1]:.1f}", "Throughput(GB/s)": f"{result[2]:.4f}"})

        pbar.close()
