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
"""Test all SM100 GEMM kernels (level 1-9) for correctness and performance."""
import sys
import traceback


def test_level(level_num):
    """Test a single level kernel."""
    mod_name = f"gemm_level{level_num}"
    print(f"\n{'='*60}")
    print(f"  SM100 GEMM Level {level_num}")
    print(f"{'='*60}")
    try:
        mod = __import__(f"little_kernel.benchmark.gemm_sm100.{mod_name}", fromlist=[mod_name])
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

    results = {}
    for level in levels:
        try:
            ok = test_level(level)
            results[level] = "PASS" if ok else "FAIL"
        except Exception as e:
            results[level] = f"ERROR: {e}"

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for level, status in sorted(results.items()):
        print(f"  Level {level}: {status}")


if __name__ == "__main__":
    main()
