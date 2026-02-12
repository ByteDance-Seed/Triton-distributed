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
import subprocess
import re

WarmUps = 1000_000
Iterations = 10_000_000
ArraySize = 1024
T = ll.uint64


@ll.simple_builtin
def load_shared(ptr: ll.Tensor[ll.uint64], offset: int) -> ll.uint64:
    ret: ll.uint64 = 0
    ll.asm_volatile("ld.shared.u64 %0, [%1];", ["=l", "l"], [ret, ptr + offset])
    return ret


def bench_shared_cache_latency_kernel(
    dummy: ll.Tensor[T],
    latency: ll.Tensor[T],
) -> ll.void:
    buffer = ll.empty([ArraySize], dtype=ll.uint64, scope="shared")
    tid = ll.threadIdx_x()
    for i in range(tid, ArraySize, ll.blockDim_x()):
        buffer[i] = (i + 1) % ArraySize
    ll.__syncthreads()

    global_tid = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if global_tid > 0:
        return
    start_clk: ll.uint64 = 0
    end_clk: ll.uint64 = 0
    value: T = 0

    for i in range(WarmUps):
        value = load_shared(buffer, value)

    start_clk = ll.clock64()
    for i in range(Iterations):
        value = load_shared(buffer, value)
    dummy[0] = value
    end_clk = ll.clock64()
    latency[0] = (end_clk - start_clk) / Iterations


def bench_shared_cache_throughput_kernel(
    dummy: ll.Tensor[T],
    throughput: ll.Tensor[ll.float32],
) -> ll.void:
    buffer = ll.empty([ArraySize], dtype=ll.uint64, scope="shared")
    tid = ll.threadIdx_x()
    for i in range(tid, ArraySize, ll.blockDim_x()):
        buffer[i] = (i + 1) % ArraySize
    ll.__syncthreads()

    tid = ll.threadIdx_x()
    start_clk: ll.uint64 = 0
    end_clk: ll.uint64 = 0
    value: T = load_shared(buffer, tid % ArraySize)

    for i in range(WarmUps):
        value = load_shared(buffer, value)

    start_clk = ll.clock64()
    for i in range(Iterations):
        value = load_shared(buffer, value)
    dummy[0] = value
    end_clk = ll.clock64()
    throughput[tid] = 1.0 * Iterations * ll.blockDim_x() * 8 / (end_clk - start_clk)  # bytes/cycle


if __name__ == "__main__":
    passes = PASSES["cuda"]

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    sm_arch = torch.cuda.get_device_capability()
    sm_arch = f"sm_{sm_arch[0]}{sm_arch[1]}"

    gpu_name = torch.cuda.get_device_name().replace(" ", "_")
    print(f"GPU: {gpu_name}")

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

    # bench latency
    latency_file = f"shared_memory_latency_{gpu_name}.csv"
    with open(latency_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["array_size(KB)", "latency(cycle)", "latency(ns)"])
        f.flush()  # Write header immediately

        for i in tqdm(range(13), desc="Shared memory latency benchmark"):
            ArraySize = 2**i
            grid = (1, 1, 1)
            block = (32, 1, 1)
            smem_size = ArraySize * 8
            # template params changes, so reconstruct ll_kernel every time
            ll_bench_kernel = lk.ll_kernel(backend="cuda", is_entry=True)(bench_shared_cache_latency_kernel)
            kernel = ll_bench_kernel.build(
                passes,
                codegen_cuda,
                grid=grid,
                block=block,
                shared_mem_bytes=smem_size,
                arch=sm_arch,
                verbose=True,
            )

            dummy = torch.zeros(1, dtype=torch.uint64, device="cuda")
            latency = torch.zeros(1, dtype=torch.uint64, device="cuda")
            kernel(dummy, latency)
            latency_cycle = latency.item()
            latency_ns = latency_cycle / (gpu_freq * 1e6) * 1e9 if gpu_freq else None
            result = [ArraySize * 8 / 1024, latency_cycle]
            if latency_ns is not None:
                result.append(latency_ns)
            else:
                result.append(None)
            writer.writerow(result)
            f.flush()  # Write immediately after each result
            if latency_ns is not None:
                tqdm.write(f"Array size: {result[0]:.1f} KB, Latency: {result[1]:.2f} cycles, {result[2]:.2f} ns")
            else:
                tqdm.write(f"Array size: {result[0]:.1f} KB, Latency: {result[1]:.2f} cycles")

    # bench throughput
    throughput_file = f"shared_memory_throughput_{gpu_name}.csv"
    block_sizes = [32, 64, 128, 256, 512, 1024]
    array_sizes = [2**i for i in range(13)]
    total_iterations = len(block_sizes) * len(array_sizes)

    with open(throughput_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["block_size", "array_size(KB)", "throughput(GB/s)"])
        f.flush()  # Write header immediately

        pbar = tqdm(total=total_iterations, desc="Shared memory throughput benchmark")
        for block_size in block_sizes:
            for i in range(13):
                ArraySize = 2**i
                grid = (1, 1, 1)
                block = (block_size, 1, 1)
                smem_size = ll.align_power_of_2(ArraySize * 8, 1024)

                # template params changes, so reconstruct ll_kernel every time
                ll_bench_kernel = lk.ll_kernel(backend="cuda", is_entry=True)(bench_shared_cache_throughput_kernel)
                kernel = ll_bench_kernel.build(
                    passes,
                    codegen_cuda,
                    grid=grid,
                    block=block,
                    shared_mem_bytes=smem_size,
                    arch=sm_arch,
                    verbose=True,
                )

                print(block_size, ArraySize, smem_size)
                print(kernel.cuda_code)

                dummy = torch.zeros(1, dtype=torch.uint64, device="cuda")
                throughput = torch.zeros(block_size, dtype=torch.float32, device="cuda")
                kernel(dummy, throughput)
                throughput_bytes_per_cycle = throughput.float().mean().item()
                throughput_gb_per_s = num_sms * throughput_bytes_per_cycle * gpu_freq * 1e6 / 1e9 if gpu_freq else None

                result = [block_size, ArraySize * 8 / 1024]
                if throughput_gb_per_s is not None:
                    result.append(throughput_gb_per_s)
                else:
                    result.append(throughput_bytes_per_cycle)
                writer.writerow(result)
                f.flush()  # Write immediately after each result
                pbar.update(1)
                pbar.set_postfix({
                    "Block":
                    block_size, "Size(KB)":
                    f"{result[1]:.1f}", "Throughput":
                    f"{result[2]:.4f}" + (" GB/s" if throughput_gb_per_s is not None else " bytes/cycle")
                })

        pbar.close()
