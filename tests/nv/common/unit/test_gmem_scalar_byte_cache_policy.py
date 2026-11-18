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
"""Byte-element cache-policy legalization for gmem_ld_scalar/gmem_st_scalar.

The CUDA ``__ld*``/``__st*`` cache-operator family has no overload for 1-byte
integer payloads (``bool``, ``int8_t``, ``uint8_t``): ``__stcs(bool*, bool)``
is rejected by NVCC ("no instance of overloaded function __stcs matches").
When a cache policy is selected on a 1-byte integer element tensor, the
scalar load/store codegen must legalize the access through an
unsigned-char reinterpret so the cache-policy hint applies generically
while the moved byte stays bit-identical.  Wider dtypes must keep the
historical typed-pointer spelling, and the default (no policy) path must
stay a plain subscript for every dtype.
"""

import unittest
import ast

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(source: str) -> str:
    tree = ast.parse(source)
    return codegen_cuda(tree, ctx={"ll": ll}, emit_header=False)


_STORE_KERN = """
def kern(m: ll.Tensor[ll.%s], x: ll.Tensor[ll.int32]) -> ll.void:
    i = ll.threadIdx_x()
    v: ll.int32 = ll.gmem_ld_scalar(x, i)
    ll.gmem_st_scalar(m, i, v != 0, policy="%s")
"""

_LOAD_KERN = """
def kern(m: ll.Tensor[ll.%s], x: ll.Tensor[ll.int32]) -> ll.void:
    i = ll.threadIdx_x()
    v: ll.bool_ = ll.gmem_ld_scalar(m, i, policy="%s")
    ll.gmem_st_scalar(x, i, v)
"""

_BYTE_DTYPES = ("bool_", "int8", "uint8")

_STORE_INTRIN = {
    "cg": "__stcg",
    "cs": "__stcs",
    "wt": "__stwt",
}

_LOAD_INTRIN = {
    "ca": "__ldca",
    "cg": "__ldcg",
    "cs": "__ldcs",
    "lu": "__ldlu",
    "nc": "__ldg",
    "cv": "__ldcv",
}


class TestGmemScalarByteCachePolicy(unittest.TestCase):

    def test_store_byte_policies_legalized(self):
        """bool/int8/uint8 + policy -> unsigned-char store reinterpret."""
        for dtype in _BYTE_DTYPES:
            for policy, intrin in _STORE_INTRIN.items():
                # bool tensor stores take a bool value; use (v != 0) for all
                # so the expression type is bool regardless of dtype (int8 /
                # uint8 payloads accept the conversion).
                src = _STORE_KERN % (dtype, policy)
                code = _emit(src)
                self.assertIn(f"{intrin}(reinterpret_cast<unsigned char*>(m + (i)),", code,
                              f"{dtype} policy {policy} missing byte legalization")
                self.assertIn("(unsigned char)", code)
                # The illegal typed-pointer form must not be emitted.
                self.assertNotIn(f"{intrin}(m + (i)", code)

    def test_load_byte_policies_legalized(self):
        """bool/int8/uint8 + policy -> const unsigned-char load reinterpret."""
        for dtype in _BYTE_DTYPES:
            for policy, intrin in _LOAD_INTRIN.items():
                src = """
def kern(m: ll.Tensor[ll.%s], x: ll.Tensor[ll.int32]) -> ll.void:
    i = ll.threadIdx_x()
    v: ll.%s = ll.gmem_ld_scalar(m, i, policy="%s")
    ll.gmem_st_scalar(x, i, v)
""" % (dtype, "bool_" if dtype == "bool_" else "int32", policy)
                code = _emit(src)
                self.assertIn(f"{intrin}(reinterpret_cast<const unsigned char*>(m + (i)))", code,
                              f"{dtype} policy {policy} missing byte legalization")
                self.assertNotIn(f"{intrin}(m + (i))", code)

    def test_wide_dtypes_keep_typed_pointer(self):
        """int16/int32/int64/float32 stores keep __st*(typed_ptr) spelling."""
        for dtype in ("int16", "int32", "int64", "float32"):
            src = """
def kern(m: ll.Tensor[ll.%s]) -> ll.void:
    i = ll.threadIdx_x()
    v: ll.%s = ll.gmem_ld_scalar(m, i, policy="cs")
    ll.gmem_st_scalar(m, i, v, policy="cs")
""" % (dtype, dtype)
            code = _emit(src)
            self.assertIn("__ldcs(m + (i))", code, dtype)
            self.assertIn("__stcs(m + (i), v);", code, dtype)
            self.assertNotIn("unsigned char", code, dtype)

    def test_default_policy_stays_plain_subscript(self):
        """No policy on a bool tensor -> historical plain subscript, no cast."""
        src = """
def kern(m: ll.Tensor[ll.bool_]) -> ll.void:
    i = ll.threadIdx_x()
    v: ll.bool_ = ll.gmem_ld_scalar(m, i)
    ll.gmem_st_scalar(m, i, v)
"""
        code = _emit(src)
        self.assertIn("= m[i];", code)
        self.assertIn("m[i] = v;", code)
        for intrin in list(_LOAD_INTRIN.values()) + list(_STORE_INTRIN.values()):
            self.assertNotIn(intrin, code)
        self.assertNotIn("unsigned char", code)

    def test_bool_cs_mask_kernel(self):
        """The dependent-task reproducer: write-once bool mask, policy cs."""
        src = """
def kern(m: ll.Tensor[ll.bool_], n: ll.int32) -> ll.void:
    i = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        ll.gmem_st_scalar(m, i, True, policy="cs")
"""
        code = _emit(src)
        self.assertIn("__stcs(reinterpret_cast<unsigned char*>(m + (i)), (unsigned char)", code)
        self.assertNotIn("__stcs(m + (i)", code)


if __name__ == "__main__":
    unittest.main()
