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
"""stmatrix typed-store typed-IR CPU regression tests (no GPU required).

Covers the shared sm90/sm100 capability ``ir.device.stmatrix_x4_bf16_tile``:

* the typed ``stmatrix_x1``/``stmatrix_x2``/``stmatrix_x4`` store family --
  a shared-memory pointer plus 1/2/4 uint32 register operands -- is
  resolvable through the ``ll.*`` language namespace and lowers to
  ``stmatrix.sync.aligned.xN.m8n8.shared.b16`` inline PTX via CPU codegen;
* the emitted helpers convert the generic pointer with
  ``__cvta_generic_to_shared`` so converted bf16 P/dS-class tiles land in
  the swizzled shared layout that ldmatrix/wgmma consumers require;
* the x1 form (single 8x8 b16 fragment, one register per lane) previously
  failed at attribute lookup during ``lower()`` because only
  ``stmatrix_x1_reg`` (ptr+index+array form) was exported -- the typed
  scalar-register form completes the family next to x2/x4;
* helper bodies dedupe by name across repeated call sites.

When ``nvcc`` is on PATH, the emitted sources are additionally compiled
with ``-gencode arch=compute_90a,code=sm_90a`` as a pure CPU-side nvcc
acceptance gate (ptxas legality of the inline PTX); no device is used.
"""

import os
import re
import shutil
import subprocess
import tempfile
import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _compile(kernel):
    return kernel.compile(PASSES["cuda"], codegen_cuda)


@lk.ll_kernel(backend='cuda', is_entry=True)
def _stmatrix_family_repro() -> ll.void:
    # One warp stages three fragments (8x8, 8x16, 16x16 b16) into smem.
    smem = ll.empty([4096], dtype=ll.uint8, scope="dynamic_shared")
    r: ll.uint32 = 0
    p = ll.smem_bf16_ptr(smem, 0)
    ll.stmatrix_x1(p, r)
    ll.stmatrix_x2(p, r, r)
    ll.stmatrix_x4(p, r, r, r, r)


@lk.ll_kernel(backend='cuda', is_entry=True)
def _stmatrix_dedup_repro() -> ll.void:
    smem = ll.empty([4096], dtype=ll.uint8, scope="dynamic_shared")
    a: ll.uint32 = 0
    b: ll.uint32 = 1
    p0 = ll.smem_bf16_ptr(smem, 0)
    p1 = ll.smem_bf16_ptr(smem, 256)
    ll.stmatrix_x1(p0, a)
    ll.stmatrix_x1(p1, b)


class TestStmatrixTypedStore(unittest.TestCase):
    """CPU regressions for the typed stmatrix x1/x2/x4 store family."""

    def test_stmatrix_family_resolvable_on_ll(self):
        # Attribute lookup used to fail during lower() for the x1 form.
        for name in ("stmatrix_x1", "stmatrix_x2", "stmatrix_x4"):
            self.assertTrue(hasattr(ll, name), f"language symbol {name} missing")

    def test_stmatrix_family_lowering(self):
        src = _compile(_stmatrix_family_repro)
        # One dedicated helper per fragment width, each emitting the matching
        # stmatrix.sync.aligned.xN.m8n8.shared.b16 PTX.
        for n in ("x1", "x2", "x4"):
            self.assertIn(f"stmatrix.sync.aligned.{n}.m8n8.shared.b16", src)
            self.assertIn(f"stmatrix_{n}_fn(", src)
        # Generic pointers are converted to shared-state-space byte addresses.
        self.assertIn("__cvta_generic_to_shared", src)
        # All three helpers take a pointer plus N uint32 register operands.
        self.assertIsNotNone(re.search(r"void stmatrix_x1_fn\(void\* smem_ptr, uint32_t r0\)", src))
        self.assertIsNotNone(re.search(r"void stmatrix_x2_fn\(void\* smem_ptr, uint32_t r0, uint32_t r1\)", src))
        self.assertIsNotNone(
            re.search(r"void stmatrix_x4_fn\(void\* smem_ptr, uint32_t r0, uint32_t r1,"
                      r" uint32_t r2, uint32_t r3\)", src))

    def test_stmatrix_x1_dedup_by_name(self):
        src = _compile(_stmatrix_dedup_repro)
        # Two call sites, one helper definition.
        self.assertEqual(src.count("__device__ __forceinline__ void stmatrix_x1_fn("), 1)

    def test_nvcc_sm90a_stmatrix_acceptance(self):
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc not on PATH; CPU codegen gate only")
        src = _compile(_stmatrix_family_repro)
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "stmatrix_family.cu")
            obj = os.path.join(td, "stmatrix_family.o")
            with open(f, "w") as fh:
                fh.write(src)
            proc = subprocess.run([nvcc, "-c", "-gencode", "arch=compute_90a,code=sm_90a", "-std=c++17", f, "-o", obj],
                                  capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, f"nvcc sm_90a rejected stmatrix family:\n"
                             f"{proc.stderr}")
            self.assertTrue(os.path.exists(obj))


if __name__ == "__main__":
    unittest.main()
