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
"""Regression: typed FP32 WGMMA accumulator register-fragment arrays.

A WGMMA accumulator is a fixed-size per-thread FP32 fragment (N//2 registers
for m64nNk16). The frontend-visible typed constructor for it is
``ll.empty([N // 2], ll.float32, scope="local")``: a fixed-size local array
accessed with compile-time-constant indices is scalarized by nvcc into
physical float registers, and the WGMMA builtins bind the elements as the
instruction's ``"+f"`` register operand list. There is deliberately no
separate ``scope="register"`` — requesting it is rejected with an actionable
diagnostic pointing at the local path instead of an opaque KeyError.

These are CPU lowering/codegen tests; the optional nvcc/ptxas assembly check
runs only when a CUDA toolkit is installed and needs no GPU.
"""

import os
import re
import shutil
import subprocess
import tempfile
import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES


def _compile(func, num_threads=128):
    kernel = lk.ll_kernel(backend="cuda", is_entry=True)(func)
    return kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=num_threads)


def _constraint_indices(cuda, helper_name):
    """Return the sorted ``c[i]`` constraint indices inside one helper body."""
    start = cuda.index(helper_name)
    body = cuda[start:start + 20000]
    end = body.find("}\n")
    return sorted(int(m) for m in re.findall(r'\"\+f\"\(c\[(\d+)\]\)', body[:end]))


class TestWgmmaLocalAccumulatorRegisterPath(unittest.TestCase):
    """scope="local" is the typed register-fragment-array path."""

    def test_generic_n64_emits_32_float_register_list(self):

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            acc: ll.Tensor[ll.float32] = ll.empty([32], ll.float32, scope="local")
            for i in ll.unroll(range(32)):
                acc[i] = ll.float32(0)
            ll.wgmma_mma_async_ss(acc, ll.uint64(0), ll.uint64(0), 64, dtype="bf16", scale_d=0)
            for i in ll.unroll(range(32)):
                y[i] = acc[i]

        cuda = _compile(k)
        # Flat register-array declaration, no pointer-backed local allocator.
        self.assertIn("float acc[32];", cuda)
        # The builtin takes the whole array and binds every element as a
        # read-write float register constraint in register order.
        helper = "wgmma_mma_ss_n64_bf16"
        self.assertIn(f"void {helper}", cuda)
        self.assertRegex(cuda, r"void %s[^\(]*\(float\* c" % helper)
        self.assertEqual(_constraint_indices(cuda, helper), list(range(32)))
        self.assertIn("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16", cuda)
        # Whole-array passing and element-wise reads both survive.
        self.assertIn(f"{helper}", cuda)
        self.assertRegex(cuda, r"_fn\(acc, ")
        # Unrolled epilogue reads bind compile-time-constant element indices.
        self.assertRegex(cuda, r"int32_t i = 0;\s*y\[i\] = acc")
        self.assertRegex(cuda, r"int32_t i = 31;\s*y\[i\] = acc")

    def test_generic_n128_emits_64_float_register_list(self):

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            acc: ll.Tensor[ll.float32] = ll.empty([64], ll.float32, scope="local")
            ll.wgmma_mma_async_ss(acc, ll.uint64(0), ll.uint64(0), 128, dtype="bf16", scale_d=1)
            for i in ll.unroll(range(64)):
                y[i] = acc[i]

        cuda = _compile(k)
        self.assertIn("float acc[64];", cuda)
        helper = "wgmma_mma_ss_n128_bf16"
        self.assertEqual(_constraint_indices(cuda, helper), list(range(64)))
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16", cuda)

    def test_named_m64n128_ss_variant_same_path(self):

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            acc: ll.Tensor[ll.float32] = ll.empty([64], ll.float32, scope="local")
            ll.wgmma_m64n128k16_ss(acc, ll.uint64(0), ll.uint64(0), 1)
            for i in ll.unroll(range(64)):
                y[i] = acc[i]

        cuda = _compile(k)
        self.assertIn("float acc[64];", cuda)
        # The named fixed-shape builtin emits the same 64-"+f" binding.
        self.assertEqual(sorted(int(m) for m in re.findall(r'\"\+f\"\(c\[(\d+)\]\)', cuda)), list(range(64)))
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16", cuda)

    def test_register_scope_rejected_with_actionable_message(self):

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            acc: ll.Tensor[ll.float32] = ll.empty([32], ll.float32, scope="register")
            y[0] = acc[0]

        with self.assertRaises(Exception) as ctx:
            _compile(k)
        message = str(ctx.exception)
        self.assertNotIsInstance(ctx.exception, KeyError)
        self.assertIn("scope", message)
        self.assertIn("local", message)


@unittest.skipUnless(shutil.which("nvcc"), "nvcc not available")
class TestWgmmaLocalAccumulatorPtxas(unittest.TestCase):
    """Assemble the emitted CUDA for sm_90a (no GPU required).

    The accumulator must bind a real float-register operand list rather than
    a .local-backed address: nvcc scalarizes the fixed-size local array.
    """

    def test_m64n64_assembles_with_register_list(self):

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            acc: ll.Tensor[ll.float32] = ll.empty([32], ll.float32, scope="local")
            ll.wgmma_fence()
            ll.wgmma_mma_async_ss(acc, ll.uint64(0), ll.uint64(0), 64, dtype="bf16", scale_d=1)
            ll.wgmma_commit()
            ll.wgmma_wait()
            for i in ll.unroll(range(32)):
                y[i] = acc[i]

        cuda = _compile(k)
        nvcc = shutil.which("nvcc")
        with tempfile.TemporaryDirectory() as tmp:
            cu = os.path.join(tmp, "k.cu")
            ptx = os.path.join(tmp, "k.ptx")
            with open(cu, "w") as f:
                f.write(cuda)
            proc = subprocess.run([nvcc, "-ptx", "-arch=sm_90a", cu, "-o", ptx], capture_output=True, text=True,
                                  timeout=300)
            self.assertEqual(proc.returncode, 0, f"nvcc -ptx failed:\n{proc.stderr}")
            text = re.sub(r"\s+", "", open(ptx).read())
            self.assertIn("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16", text)
            # The accumulator operand is a list of 32 .f32 registers, e.g.
            # {%f1,...,%f32} — never a local-memory address.
            m = re.search(r"bf16\.bf16\{([^}]*)\}", text)
            self.assertIsNotNone(m, text[:400])
            self.assertEqual(len(re.findall(r"%f\d+", m.group(1))), 32)


if __name__ == "__main__":
    unittest.main()
