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
"""Async Compute/Memory Overlap Benchmark for Hopper GPUs.
Framework for WGMMA + TMA pipeline overlap efficiency measurement.
"""
import os
from little_kernel.benchmark.utils import get_gpu_info, print_gpu_info, CSVWriter, get_results_dir


def main():
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    results_dir = get_results_dir(gpu_info)
    csv_w = CSVWriter(os.path.join(results_dir, "async_overlap.csv"),
                      ["pipeline_depth", "compute_ratio", "efficiency_pct"], gpu_info, "Async Overlap")
    print("Async overlap benchmark requires full TMA+WGMMA pipeline setup.")
    print("Framework ready for multi-stage buffering measurements.")
    csv_w.writerow(["N/A", "N/A", "framework_only"])
    csv_w.close()
    print("Results saved to: " + results_dir + "/")


if __name__ == "__main__":
    main()
