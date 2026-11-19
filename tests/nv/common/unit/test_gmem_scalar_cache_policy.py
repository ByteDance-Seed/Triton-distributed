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
"""Render-level contracts for the gmem_ld_scalar / gmem_st_scalar
cache-policy argument.

The stock dtype-generic scalar global-access intrinsics accept an optional
compile-time string literal ``policy`` selecting the CUDA cache operator
(loads: wb/ca/cg/cs/lu/nc/cv; stores: wb/cg/cs/wt).  Unlike the v4
intrinsics these emit inline expressions (no #ifndef-guarded helper): a
plain subscript for the default policy, otherwise the CUDA
``__ld*/__st*`` intrinsic on the typed pointer.  The default (no policy)
path must stay bit-identical to the historical subscript so existing
frontends are unaffected.  Illegal policy names must fail at compile time,
not emit invalid C++; a non-Tensor first argument must fail at type
inference.
"""

import unittest
import ast

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

from little_kernel.core.passes.utils.type_inference.type_inference_core import TypeInferenceError


def _emit(source: str) -> str:
    tree = ast.parse(source)
    return codegen_cuda(tree, ctx={"ll": ll}, emit_header=False)


_KERN_DEFAULT = """
def kern(x: ll.Tensor[ll.int32], y: ll.Tensor[ll.int32]) -> ll.void:
    i = ll.threadIdx_x()
    v: ll.int32 = ll.gmem_ld_scalar(x, i)
    ll.gmem_st_scalar(y, i, v)
"""

_KERN_KW = """
def kern(x: ll.Tensor[ll.int32], y: ll.Tensor[ll.int32]) -> ll.void:
    i = ll.threadIdx_x()
    v: ll.int32 = ll.gmem_ld_scalar(x, i, policy="%s")
    ll.gmem_st_scalar(y, i, v, policy="%s")
"""

_KERN_MIX = """
def kern(x: ll.Tensor[ll.int32], y: ll.Tensor[ll.int32],
         p: ll.Tensor[ll.int64]) -> ll.void:
    i = ll.threadIdx_x()
    v: ll.int32 = ll.gmem_ld_scalar(x, i, policy="nc")
    q: ll.int64 = ll.gmem_ld_scalar(p, i, policy="lu")
    ll.gmem_st_scalar(y, i, v, policy="cs")
    ll.gmem_st_scalar(p, i, q, policy="wt")
"""

_KERN_POS = """
def kern(x: ll.Tensor[ll.int32]) -> ll.void:
    i = ll.threadIdx_x()
    v: ll.int32 = ll.gmem_ld_scalar(x, i, "cg")
"""

_LOAD_INTRIN = {
    "ca": "__ldca",
    "cg": "__ldcg",
    "cs": "__ldcs",
    "lu": "__ldlu",
    "nc": "__ldg",
    "cv": "__ldcv",
}

_STORE_INTRIN = {
    "cg": "__stcg",
    "cs": "__stcs",
    "wt": "__stwt",
}


class TestGmemScalarCachePolicy(unittest.TestCase):

    def test_default_is_plain_subscript(self):
        """No policy argument -> historical plain subscript load/store."""
        code = _emit(_KERN_DEFAULT)
        self.assertIn("= x[i];", code)
        self.assertIn("y[i] = v;", code)
        # No cache intrinsic leaked into the default path.
        for intrin in list(_LOAD_INTRIN.values()) + list(_STORE_INTRIN.values()):
            self.assertNotIn(intrin, code)
        self.assertNotIn('"wb"', code)

    def test_load_policies_render(self):
        for policy, intrin in _LOAD_INTRIN.items():
            code = _emit(_KERN_KW % (policy, "wb"))
            self.assertIn(f"{intrin}(x + (i))", code, f"policy {policy} missing its cache operator")
            # The policy string must not leak into the emitted code.
            self.assertNotIn('"%s"' % policy, code)

    def test_store_policies_render(self):
        for policy, intrin in _STORE_INTRIN.items():
            code = _emit(_KERN_KW % ("wb", policy))
            self.assertIn(f"{intrin}(y + (i), v);", code, f"policy {policy} missing its cache operator")
            self.assertNotIn('"%s"' % policy, code)

    def test_multiple_policies_coexist(self):
        """nc + lu loads and cs + wt stores in one kernel, mixed dtypes."""
        code = _emit(_KERN_MIX)
        self.assertIn("__ldg(x + (i))", code)  # int32 read-only path
        self.assertIn("__ldlu(p + (i))", code)  # int64 last-use path
        self.assertIn("__stcs(y + (i), v);", code)  # streaming store
        self.assertIn("__stwt(p + (i), q);", code)  # write-through store
        for token in ('"nc"', '"lu"', '"cs"', '"wt"'):
            self.assertNotIn(token, code)

    def test_positional_policy_string(self):
        code = _emit(_KERN_POS)
        self.assertIn("__ldcg(x + (i))", code)

    def test_illegal_load_policy_rejected(self):
        with self.assertRaises(ValueError):
            _emit(_KERN_KW % ("bogus", "wb"))

    def test_illegal_store_policy_rejected(self):
        # "nc" is load-only; it must not be accepted for stores.
        with self.assertRaises(ValueError):
            _emit(_KERN_KW % ("wb", "nc"))

    def test_store_rejects_load_only_policies(self):
        for load_only in ("ca", "lu", "nc", "cv"):
            with self.assertRaises(ValueError, msg=load_only):
                _emit(_KERN_KW % ("wb", load_only))

    def test_load_rejects_store_only_policy(self):
        # "wt" is store-only; it must not be accepted for loads.
        with self.assertRaises(ValueError):
            _emit(_KERN_KW % ("wt", "wb"))

    def test_non_tensor_mem_rejected(self):
        """A scalar first argument must fail at type inference, not emit."""
        bad = """
def kern(x: ll.Tensor[ll.int32]) -> ll.void:
    i = ll.threadIdx_x()
    v: ll.int32 = ll.gmem_ld_scalar(i, i)
"""
        with self.assertRaises(TypeInferenceError) as cm:
            _emit(bad)
        self.assertIn("expects Tensor as first argument", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
