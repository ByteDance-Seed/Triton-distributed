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
Shared benchmark utilities for the Hopper GPU Microbenchmark Suite.

Provides common helpers for GPU info detection, timing, CSV output,
parameter sweeps, and result formatting.
"""

import os
import csv
import subprocess
import re
import datetime
import torch
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

# =============================================================================
# GPU Information
# =============================================================================


def get_gpu_info(device_id=0):
    """Retrieve comprehensive GPU information.

    Returns:
        dict with keys: name, sm_count, sm_arch, clock_mhz, l2_size_bytes,
        memory_bytes, memory_bandwidth_gbs, compute_capability.
    """
    props = torch.cuda.get_device_properties(device_id)
    cap = torch.cuda.get_device_capability(device_id)
    sm_arch = f"sm_{cap[0]}{cap[1]}"

    clock_mhz = _get_gpu_clock_mhz(device_id)

    return {
        "name": props.name, "name_safe": props.name.replace(" ", "_"), "sm_count": props.multi_processor_count,
        "sm_arch": sm_arch, "compute_capability": f"{cap[0]}.{cap[1]}", "clock_mhz": clock_mhz, "l2_size_bytes":
        getattr(props, "L2_cache_size", getattr(props, "l2_cache_size", 0)), "memory_bytes": props.total_memory,
        "memory_bandwidth_gbs": None,  # not directly available from props
    }


def _get_gpu_clock_mhz(device_id=0):
    """Query current SM clock frequency via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,nounits,noheader", "-i",
             str(device_id)], capture_output=True, text=True, check=True)
        freq_str = result.stdout.strip()
        return float(re.findall(r"\d+", freq_str)[0])
    except Exception:
        # Fallback: use a typical H100 frequency
        return 1980.0


def print_gpu_info(info):
    """Pretty-print GPU information."""
    print("=" * 60)
    print(f"  GPU: {info['name']}")
    print(f"  SM Count: {info['sm_count']}")
    print(f"  SM Arch: {info['sm_arch']} (CC {info['compute_capability']})")
    print(f"  SM Clock: {info['clock_mhz']} MHz")
    print(f"  L2 Cache: {info['l2_size_bytes'] / 1024 / 1024:.1f} MB")
    print(f"  Memory: {info['memory_bytes'] / 1024**3:.1f} GB")
    print("=" * 60)


# =============================================================================
# Kernel Build Helpers
# =============================================================================


def build_kernel(kernel_func, grid, block, shared_mem_bytes=0, arch=None, verbose=False):
    """Build a LittleKernel kernel with standard passes."""
    passes = PASSES["cuda"]
    if arch is None:
        cap = torch.cuda.get_device_capability()
        arch = f"sm_{cap[0]}{cap[1]}"
    return kernel_func.build(
        passes,
        codegen_cuda,
        grid=grid,
        block=block,
        shared_mem_bytes=shared_mem_bytes,
        arch=arch,
        verbose=verbose,
    )


def compile_kernel(kernel_func):
    """Compile a LittleKernel kernel to CUDA source (for inspection)."""
    passes = PASSES["cuda"]
    return kernel_func.compile(passes, codegen_cuda)


# =============================================================================
# Timing Helpers
# =============================================================================


def cycles_to_ns(cycles, clock_mhz):
    """Convert GPU clock cycles to nanoseconds."""
    return cycles / (clock_mhz * 1e6) * 1e9


def cycles_to_us(cycles, clock_mhz):
    """Convert GPU clock cycles to microseconds."""
    return cycles / (clock_mhz * 1e6) * 1e6


# =============================================================================
# CSV Output
# =============================================================================


class CSVWriter:
    """Unified CSV result writer with metadata header."""

    def __init__(self, filename, headers, gpu_info=None, benchmark_name=None):
        self.filename = filename
        self.headers = headers
        self.f = open(filename, "w", newline="")
        self.writer = csv.writer(self.f)

        # Write metadata as comments
        if gpu_info:
            self.f.write(f"# GPU: {gpu_info['name']}\n")
            self.f.write(f"# SM Count: {gpu_info['sm_count']}\n")
            self.f.write(f"# SM Arch: {gpu_info['sm_arch']}\n")
            self.f.write(f"# Clock: {gpu_info['clock_mhz']} MHz\n")
        if benchmark_name:
            self.f.write(f"# Benchmark: {benchmark_name}\n")
        self.f.write(f"# Date: {datetime.datetime.now().isoformat()}\n")

        self.writer.writerow(headers)
        self.f.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# Results Directory
# =============================================================================


def get_results_dir(gpu_info=None):
    """Get or create results directory."""
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    if gpu_info:
        dir_name = f"results/{gpu_info['name_safe']}_{date_str}"
    else:
        dir_name = f"results/{date_str}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


# =============================================================================
# Table Pretty Printing
# =============================================================================


def print_table(headers, rows, title=None):
    """Pretty-print a results table to console."""
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    # Print header
    header_line = " | ".join(f"{h:>{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    print("-+-".join("-" * w for w in col_widths))

    # Print rows
    for row in rows:
        line = " | ".join(f"{str(v):>{w}}" for v, w in zip(row, col_widths))
        print(line)
    print()


def format_bandwidth(gbs):
    """Format bandwidth value for display."""
    if gbs >= 1000:
        return f"{gbs:.0f} GB/s"
    elif gbs >= 1:
        return f"{gbs:.1f} GB/s"
    else:
        return f"{gbs * 1000:.1f} MB/s"


def format_flops(tflops):
    """Format FLOPS value for display."""
    if tflops >= 1:
        return f"{tflops:.2f} TFLOPS"
    elif tflops >= 0.001:
        return f"{tflops * 1000:.1f} GFLOPS"
    else:
        return f"{tflops * 1e6:.1f} MFLOPS"


def format_latency(cycles):
    """Format latency in cycles for display."""
    if cycles >= 1000:
        return f"{cycles:.0f} cyc"
    elif cycles >= 1:
        return f"{cycles:.1f} cyc"
    else:
        return f"{cycles:.3f} cyc"
