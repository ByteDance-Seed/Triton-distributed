#!/usr/bin/env python3
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
"""Run all SM100 GEMM tutorial kernels (levels 1-9) for correctness and performance.

Runnable benchmark launcher (not a pytest module): ``run_level`` takes the CLI
level number and maps it to the descriptive technique module under
``benchmarks.nv.common.gemm``.  The stable performance id for every level is
``gemm-sm100-level<N>``; the tutorial technique distinction is preserved by
the module table below.
"""
import importlib
import sys
import traceback

# Level N -> descriptive technique module (distinct tutorial techniques preserved).
LEVEL_MODULES = {
    1: "benchmarks.nv.sm100.gemm.sm100_baseline_2sm_umma",
    2: "benchmarks.nv.sm100.gemm.sm100_twostage_pipeline_cg2_tma",
    3: "benchmarks.nv.sm100.gemm.sm100_warp_specialized_8stage",
    4: "benchmarks.nv.sm100.gemm.sm100_smem_staged_coalesced_epilogue",
    5: "benchmarks.nv.sm100.gemm.sm100_persistent_swizzle_scheduler",
    6: "benchmarks.nv.sm100.gemm.sm100_three_role_warp_specialized_tma_store",
    7: "benchmarks.nv.sm100.gemm.sm100_dual_tile_2sm",
    8: "benchmarks.nv.sm100.gemm.sm100_swizzled_cd_deep_pipeline",
    9: "benchmarks.nv.sm100.gemm.sm100_deepgemm_matching_architecture",
}


def run_level(level_num):
    """Test a single level kernel."""
    mod_name = LEVEL_MODULES[level_num]
    print(f"\n{'='*60}")
    print(f"  SM100 GEMM Level {level_num}")
    print(f"{'='*60}")
    try:
        mod = importlib.import_module(mod_name)
        print("Building kernel...")
        kernel = mod.build_kernel()
        print("Kernel built OK\n")

        print("=== Correctness ===")
        all_pass = True
        for size in [512, 1024, 2048]:
            ok = mod.test(kernel, size, size, size)
            all_pass = all_pass and ok

        print(f"\nAll tests {'PASSED' if all_pass else 'FAILED'}!")

        if all_pass:
            print("\n=== Benchmark ===")
            for size in [1024, 2048, 4096]:
                mod.bench(kernel, size, size, size)

        return all_pass
    except Exception as e:
        print(f"FAILED with exception: {e}")
        traceback.print_exc()
        return False


def main():
    levels = list(range(1, 10))  # level 1-9
    if len(sys.argv) > 1:
        levels = [int(x) for x in sys.argv[1:]]

    invalid = [level for level in levels if level not in LEVEL_MODULES]
    if invalid:
        print(f"Unknown level(s): {invalid}; valid levels are 1-9")
        return 2

    results = {}
    for level in levels:
        try:
            ok = run_level(level)
            results[level] = "PASS" if ok else "FAIL"
        except Exception as e:
            results[level] = f"ERROR: {e}"

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for level, status in sorted(results.items()):
        print(f"  Level {level}: {status}")

    failed = [level for level, status in results.items() if status != "PASS"]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
