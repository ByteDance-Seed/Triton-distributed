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
"""Full Suite Runner for Hopper GPU Microbenchmark Suite.
Runs all benchmarks with selective execution and progress tracking.
"""
import argparse
import importlib
import time

from little_kernel.benchmark.utils import get_gpu_info, print_gpu_info, get_results_dir

BENCHMARKS = {
    "memory": {
        "l1_cache": "little_kernel.benchmark.memory.bench_l1_cache",
        "l2_cache": "little_kernel.benchmark.memory.bench_l2_cache",
        "shared_memory": "little_kernel.benchmark.memory.bench_shared_memory",
        "global_memory": "little_kernel.benchmark.memory.bench_global_memory",
        "register_file": "little_kernel.benchmark.memory.bench_register_file",
        "smem_bank_conflict": "little_kernel.benchmark.memory.bench_smem_bank_conflict",
    },
    "compute": {
        "flops": "little_kernel.benchmark.compute.bench_flops",
        "sfu": "little_kernel.benchmark.compute.bench_sfu",
        "wgmma": "little_kernel.benchmark.compute.bench_wgmma",
    },
    "latency": {
        "arith_latency": "little_kernel.benchmark.latency.bench_arith_latency",
        "mem_latency": "little_kernel.benchmark.latency.bench_mem_latency",
        "sync_latency": "little_kernel.benchmark.latency.bench_sync_latency",
    },
    "warp": {
        "shuffle": "little_kernel.benchmark.warp.bench_shuffle",
        "vote": "little_kernel.benchmark.warp.bench_vote",
    },
    "sm90": {
        "tma": "little_kernel.benchmark.sm90.bench_tma",
        "cluster": "little_kernel.benchmark.sm90.bench_cluster",
        "async_overlap": "little_kernel.benchmark.sm90.bench_async_overlap",
    },
    "sm": {
        "occupancy": "little_kernel.benchmark.sm.bench_occupancy",
        "ipc": "little_kernel.benchmark.sm.bench_ipc",
    },
}


def run_benchmark(module_path, bench_name):
    """Run a single benchmark by importing and calling its main()."""
    print("\n" + "=" * 70)
    print("  Running: {}".format(bench_name))
    print("=" * 70)
    try:
        mod = importlib.import_module(module_path)
        start = time.time()
        mod.main()
        elapsed = time.time() - start
        print("  Completed {} in {:.1f}s".format(bench_name, elapsed))
        return True
    except Exception as e:
        print("  FAILED {}: {}".format(bench_name, str(e)))
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Hopper GPU Microbenchmark Suite Runner")
    parser.add_argument("--category", type=str, default=None,
                        help="Run only benchmarks in this category (memory, compute, latency, warp, sm90, sm)")
    parser.add_argument("--bench", type=str, default=None,
                        help="Run only this specific benchmark (e.g., flops, shuffle)")
    parser.add_argument("--list", action="store_true", help="List all available benchmarks")
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for cat, benches in BENCHMARKS.items():
            print("  Category: {}".format(cat))
            for name in benches:
                print("    - {}".format(name))
        return

    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    print("Results directory: {}".format(results_dir))

    # Determine which benchmarks to run
    to_run = []
    for cat, benches in BENCHMARKS.items():
        if args.category and cat != args.category:
            continue
        for name, module_path in benches.items():
            if args.bench and name != args.bench:
                continue
            to_run.append((name, module_path))

    if not to_run:
        print("No benchmarks matched the given filters.")
        return

    print("\nWill run {} benchmark(s):".format(len(to_run)))
    for name, _ in to_run:
        print("  - {}".format(name))

    # Run benchmarks
    passed = 0
    failed = 0
    start_all = time.time()
    for name, module_path in to_run:
        ok = run_benchmark(module_path, name)
        if ok:
            passed += 1
        else:
            failed += 1

    total_time = time.time() - start_all
    print("\n" + "=" * 70)
    print("  Suite Complete: {}/{} passed, {} failed, {:.0f}s total".format(passed, passed + failed, failed,
                                                                            total_time))
    print("  Results: {}".format(results_dir))
    print("=" * 70)


if __name__ == "__main__":
    main()
