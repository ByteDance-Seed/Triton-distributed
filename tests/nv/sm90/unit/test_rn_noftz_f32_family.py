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

# CPU regression tests for the ``ll.{add,sub,mul,div}_rn_noftz_f32`` family.
#
# Semantic key: ``ir.nv.ptx.rn_float_ops.no_ftz``.  Bare round-to-nearest-even
# fp32 arithmetic (``add.rn.f32`` / ``sub.rn.f32`` / ``mul.rn.f32`` /
# ``div.rn.f32``) emitted via inline asm with NO ``.ftz`` suffix and no
# source-level fma contraction, immune to a ``--use_fast_math`` CUDA build.
# The existing ``__fadd_rn``/``__fmul_rn``/... device intrinsics pin the
# rounding mode but still inherit the module-wide flush-to-zero flag, so
# under ``--use_fast_math`` their PTX is ``add.rn.ftz.f32`` etc. and denormal
# results flush to zero.  These intrinsics emit the bare non-ftz PTX opcodes
# through inline assembly, which the compiler cannot tag with ``.ftz`` and
# cannot contract into an fma, so IEEE binary32 denormal (subnormal) outputs
# are preserved exactly.  These tests are entirely CPU-based (type +
# CUDA source lowering); paired GPU validation is left to the queue.

import unittest

import little_kernel as lk
import little_kernel.language as ll

from little_kernel.backends.nv.common.language.intrin.cuda_asm import (
    add_rn_noftz_f32,
    div_rn_noftz_f32,
    mul_rn_noftz_f32,
    sub_rn_noftz_f32,
)


def _compile(fn):
    from little_kernel.core.passes import PASSES
    from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
    return fn.compile(PASSES["cuda"], codegen_cuda, num_threads=128)


class TestRnNoftzF32Lowering(unittest.TestCase):
    """End-to-end CUDA source lowering of the bare non-ftz rn family."""

    def test_mul_emits_bare_mul_rn_f32(self):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def k(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32]) -> ll.void:
            a: ll.float32 = x[0]
            b: ll.float32 = x[1]
            y[0] = ll.mul_rn_noftz_f32(a, b)

        src = _compile(k)
        self.assertIn("mul.rn.f32", src)
        # The whole point: no flush-to-zero suffix and no fma contraction.
        self.assertNotIn("mul.rn.ftz.f32", src)
        self.assertNotIn("fma.rn.f32", src)

    def test_add_emits_bare_add_rn_f32(self):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def k(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32]) -> ll.void:
            a: ll.float32 = x[0]
            b: ll.float32 = x[1]
            y[0] = ll.add_rn_noftz_f32(a, b)

        src = _compile(k)
        self.assertIn("add.rn.f32", src)
        self.assertNotIn("add.rn.ftz.f32", src)

    def test_sub_emits_bare_sub_rn_f32(self):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def k(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32]) -> ll.void:
            a: ll.float32 = x[0]
            b: ll.float32 = x[1]
            y[0] = ll.sub_rn_noftz_f32(a, b)

        src = _compile(k)
        self.assertIn("sub.rn.f32", src)
        self.assertNotIn("sub.rn.ftz.f32", src)

    def test_div_emits_bare_div_rn_f32(self):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def k(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32]) -> ll.void:
            a: ll.float32 = x[0]
            b: ll.float32 = x[1]
            y[0] = ll.div_rn_noftz_f32(a, b)

        src = _compile(k)
        self.assertIn("div.rn.f32", src)
        self.assertNotIn("div.rn.ftz.f32", src)
        # --use_fast_math must not be able to approximate this division.
        self.assertNotIn("div.approx.f32", src)

    def test_mul_add_pair_does_not_contract(self):
        # A mul feeding an add is exactly the shape nvcc contracts into
        # fma.rn.f32 under fast math.  Routing both through the noftz
        # intrinsics must keep them as two separate non-ftz rn ops.
        @lk.ll_kernel(backend="cuda", is_entry=True)
        def k(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32]) -> ll.void:
            a: ll.float32 = x[0]
            b: ll.float32 = x[1]
            c: ll.float32 = x[2]
            p: ll.float32 = ll.mul_rn_noftz_f32(a, b)
            y[0] = ll.add_rn_noftz_f32(p, c)

        src = _compile(k)
        self.assertIn("mul.rn.f32", src)
        self.assertIn("add.rn.f32", src)
        self.assertNotIn("fma.rn.f32", src)
        self.assertNotIn(".ftz", src)


class TestRnNoftzF32DirectBuiltins(unittest.TestCase):
    """The plain-function forms also emit bare non-ftz rn opcodes."""

    def _compile_direct(self, op, name):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def k(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32]) -> ll.void:
            a: ll.float32 = x[0]
            b: ll.float32 = x[1]
            y[0] = op(a, b)

        src = _compile(k)
        self.assertIn(name, src)
        self.assertNotIn(name.replace(".f32", ".ftz.f32"), src)

    def test_add_direct(self):
        self._compile_direct(add_rn_noftz_f32, "add.rn.f32")

    def test_sub_direct(self):
        self._compile_direct(sub_rn_noftz_f32, "sub.rn.f32")

    def test_mul_direct(self):
        self._compile_direct(mul_rn_noftz_f32, "mul.rn.f32")

    def test_div_direct(self):
        self._compile_direct(div_rn_noftz_f32, "div.rn.f32")


if __name__ == "__main__":
    unittest.main()
