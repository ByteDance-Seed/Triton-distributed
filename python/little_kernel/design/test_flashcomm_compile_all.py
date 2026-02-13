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
Test that all FlashComm kernels compile (codegen + nvcc) successfully.
Usage:
  CUDA_VISIBLE_DEVICES=3 python design/test_flashcomm_compile_all.py
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda


def try_build(name, build_fn):
    t0 = time.time()
    try:
        build_fn()
        dt = time.time() - t0
        print(f"  PASS: {name} ({dt:.1f}s)")
        return True
    except Exception:
        dt = time.time() - t0
        print(f"  FAIL: {name} ({dt:.1f}s)")
        traceback.print_exc()
        return False


results = []

# 1. barrier
print("\n--- flashcomm_barrier ---")
from flashcomm_barrier import build_kernel as build_barrier  # noqa: E402

results.append(("barrier", try_build("kernel_barrier_all_on_stream", build_barrier)))

# 2. compute: compute_offset
print("\n--- flashcomm_compute: compute_offset ---")
from flashcomm_compute import kernel_compute_offset, NUM_WARPS, BLOCK_SIZE, MAX_EXPERTS_PLUS_1  # noqa: E402


def build_compute_offset():
    return kernel_compute_offset.build(passes=PASSES["cuda"], codegen_func=codegen_cuda, grid=(4, 1, 1),
                                       block=(BLOCK_SIZE, 1, 1), shared_mem_bytes=NUM_WARPS * MAX_EXPERTS_PLUS_1 * 4,
                                       verbose=False)


results.append(("compute_offset", try_build("kernel_compute_offset", build_compute_offset)))

# 3. compute: dispatch_layout
print("\n--- flashcomm_compute: dispatch_layout ---")
from flashcomm_compute import kernel_compute_dispatch_layout  # noqa: E402


def build_dispatch_layout():
    return kernel_compute_dispatch_layout.build(passes=PASSES["cuda"], codegen_func=codegen_cuda, grid=(4, 1, 1),
                                                block=(BLOCK_SIZE, 1, 1), shared_mem_bytes=NUM_WARPS * 4, verbose=False)


results.append(("dispatch_layout", try_build("kernel_compute_dispatch_layout", build_dispatch_layout)))

# 4. dispatch: basic
print("\n--- flashcomm_dispatch: basic ---")
from flashcomm_dispatch import build_kernel as build_dispatch  # noqa: E402

results.append(("dispatch_basic", try_build("kernel_dispatch_intranode", lambda: build_dispatch("basic"))))

# 5. dispatch: v1
print("\n--- flashcomm_dispatch: v1 ---")
results.append(("dispatch_v1", try_build("kernel_dispatch_intranode_v1", lambda: build_dispatch("v1"))))

# 6. dispatch_chunk
print("\n--- flashcomm_dispatch_chunk ---")
from flashcomm_dispatch_chunk import build_kernel as build_dispatch_chunk  # noqa: E402

results.append(("dispatch_chunk", try_build("kernel_dispatch_intranode_chunk", build_dispatch_chunk)))

# 7. combine: preprocess
print("\n--- flashcomm_combine: preprocess ---")
from flashcomm_combine import build_kernel as build_combine  # noqa: E402

results.append(("combine_preprocess", try_build("kernel_combine_preprocess_inplace",
                                                lambda: build_combine("preprocess"))))

# 8. combine: basic
print("\n--- flashcomm_combine: basic ---")
results.append(("combine_basic", try_build("kernel_combine_intranode", lambda: build_combine("combine"))))

# 9. combine: v1
print("\n--- flashcomm_combine: v1 ---")
results.append(("combine_v1", try_build("kernel_combine_intranode_v1", lambda: build_combine("v1"))))

# 10. combine: v2
print("\n--- flashcomm_combine: v2 ---")
results.append(("combine_v2", try_build("kernel_combine_intranode_v2", lambda: build_combine("v2"))))

# 11. postprocess
print("\n--- flashcomm_postprocess ---")
from flashcomm_postprocess import build_kernel as build_postprocess  # noqa: E402

results.append(("postprocess", try_build("kernel_dispatch_postprocess_tma", build_postprocess)))

# Summary
print("\n" + "=" * 60)
print("Compilation Summary:")
total_pass = sum(1 for _, ok in results if ok)
for name, ok in results:
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")
print(f"\n  {total_pass}/{len(results)} kernels compiled successfully")
print("=" * 60)
if total_pass < len(results):
    sys.exit(1)
