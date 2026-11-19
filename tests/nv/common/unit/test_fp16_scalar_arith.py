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
"""Typed native scalar fp16 arithmetic (semantic key
ir.fp16.scalar_native_arith).

Two fp16 scalars combined with the C++ operators `a * b` / `a + b` /
`a - b` lower to `__half * __half`, which nvcc rejects under the pinned
-D__CUDA_NO_HALF_OPERATORS__ host flags (the macro removes the __half
operator overloads; the explicit __hmul/__hadd/__hsub device functions
survive it and perform a single round-to-nearest-even native fp16 op).

Coverage:
  * the add_rn_f16/sub_rn_f16/mul_rn_f16 builtins are registered and
    exported and codegen to __hadd/__hsub/__hmul with <cuda_fp16.h>;
  * the CUDA emitter legalizes plain fp16 scalar +,-,* BinOps to those
    intrinsics (the reproducer: `w = a * b`), while float32 operators
    and the generic CppEmitter path are untouched.

CPU-only: drives type inference + the emitters directly, no nvcc/GPU.
Paired GPU validation (numerics + compile under the no-half-operators
flags) is left to the controller queue.
"""

import ast
import unittest

import little_kernel.language as ll
from little_kernel.codegen.codegen_base import CppEmitter
from little_kernel.backends.nv.common.codegen.emitter import CudaEmitter
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR
from little_kernel.language.builtin_base import Builtin
import little_kernel.backends.nv.common.language.intrin.arith as arith


def _codegen(src: str, emitter_cls) -> str:
    tree = ast.parse(src)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ctx = {"ll": ll}
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    emitter = emitter_cls(ctx=ctx, all_types=inferencer.all_types)
    emitter.visit(tree)
    return emitter.get_code()


KERNEL_FP16_OPS = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def k(a: ll.float16, b: ll.float16) -> ll.float16:
    s: ll.float16 = a + b
    d: ll.float16 = a - b
    p: ll.float16 = a * b
    return p
'''

KERNEL_FP32_OPS = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def kf(a: ll.float32, b: ll.float32) -> ll.float32:
    return a * b
'''


class Fp16ScalarArithRegistryTests(unittest.TestCase):

    def test_registered_and_exported(self):
        for name in ("add_rn_f16", "sub_rn_f16", "mul_rn_f16"):
            fn = getattr(arith, name)
            self.assertTrue(getattr(fn, BUILTIN_ATTR, False), name)
            self.assertIs(getattr(ll, name), fn, name)
            self.assertIs(getattr(fn, EVAL_RETURN_TYPE_ATTR)(None, None), ll.float16, name)

    def _check(self, name, call):
        result = getattr(getattr(arith, name), CODEGEN_FUNC_ATTR)("a", "b")
        self.assertIsInstance(result, Builtin)
        self.assertEqual(result.return_val, call)
        self.assertIn("<cuda_fp16.h>", result.includes)

    def test_codegen_intrinsics(self):
        self._check("add_rn_f16", "__hadd(a, b)")
        self._check("sub_rn_f16", "__hsub(a, b)")
        self._check("mul_rn_f16", "__hmul(a, b)")

    def test_explicit_intrinsic_renders_through_cuda_codegen(self):
        src = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def ke(a: ll.float16, b: ll.float16) -> ll.float16:
    return ll.mul_rn_f16(ll.add_rn_f16(a, b), ll.sub_rn_f16(a, b))
'''
        cuda = _codegen(src, CudaEmitter)
        self.assertIn("__hmul(__hadd(a, b), __hsub(a, b))", cuda)


class Fp16ScalarArithLegalizationTests(unittest.TestCase):

    def test_plain_fp16_ops_legalize_to_native_intrinsics(self):
        # Reproducer (sha256 830c2586...): `w = a * b` with both operands
        # ll.float16 must NOT emit the (compiled-out) __half operator.
        cuda = _codegen(KERNEL_FP16_OPS, CudaEmitter)
        self.assertIn("__hadd(a, b)", cuda)
        self.assertIn("__hsub(a, b)", cuda)
        self.assertIn("__hmul(a, b)", cuda)
        for line in cuda.splitlines():
            if "__half" in line and ("=" in line or "return" in line):
                self.assertNotIn(" a * b", line)
                self.assertNotIn(" a + b", line)
                self.assertNotIn(" a - b", line)
        self.assertIn("#include <cuda_fp16.h>", cuda)

    def test_fp32_operators_untouched(self):
        cuda = _codegen(KERNEL_FP32_OPS, CudaEmitter)
        self.assertIn("return a * b;", cuda)
        self.assertNotIn("__fmul", cuda)
        self.assertNotIn("__hmul", cuda)

    def test_generic_emitter_does_not_legalize(self):
        # The legalization is a CUDA-backend choice; the generic C++
        # emitter keeps the ambient operator spelling.
        class GenericHalfEmitter(CppEmitter):

            def _lltype_to_cpp(self, dtype):
                if dtype == ll.float16:
                    return "UserHalf"
                return super()._lltype_to_cpp(dtype)

        cpp = _codegen(KERNEL_FP16_OPS, GenericHalfEmitter)
        self.assertIn("a * b", cpp)
        self.assertNotIn("__hmul", cpp)


if __name__ == "__main__":
    unittest.main()
