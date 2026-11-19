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
"""Render-level contracts for the gmem_ld_v4 / gmem_st_v4 cache-policy arg.

The stock dtype-generic 16-byte vector intrinsics accept an optional
compile-time string literal ``policy`` selecting the CUDA cache operator
(loads: wb/ca/cg/cs/lu/nc/cv; stores: wb/cg/cs/wt).  Each used policy emits
one #ifndef-guarded per-policy helper; the default (no policy) path must
stay bit-identical to the historical helper so existing frontends are
unaffected.  Illegal policy names must fail at compile time, not emit
invalid C++.
"""

import unittest
import ast

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(source: str) -> str:
    tree = ast.parse(source)
    return codegen_cuda(tree, ctx={"ll": ll}, emit_header=False)


_KERN_DEFAULT = """
def kern(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.bfloat16]) -> ll.void:
    i = ll.threadIdx_x() * 8
    r0: ll.uint32
    r1: ll.uint32
    r2: ll.uint32
    r3: ll.uint32
    ll.gmem_ld_v4(x, i, r0, r1, r2, r3)
    ll.gmem_st_v4(y, i, r0, r1, r2, r3)
"""

_KERN_KW = """
def kern(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.bfloat16]) -> ll.void:
    i = ll.threadIdx_x() * 8
    r0: ll.uint32
    r1: ll.uint32
    r2: ll.uint32
    r3: ll.uint32
    ll.gmem_ld_v4(x, i, r0, r1, r2, r3, policy="%s")
    ll.gmem_st_v4(y, i, r0, r1, r2, r3, policy="%s")
"""

_KERN_DUAL = """
def kern(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.bfloat16],
         z: ll.Tensor[ll.bfloat16]) -> ll.void:
    i = ll.threadIdx_x() * 8
    r0: ll.uint32
    r1: ll.uint32
    r2: ll.uint32
    r3: ll.uint32
    ll.gmem_ld_v4(x, i, r0, r1, r2, r3, policy="nc")
    ll.gmem_ld_v4(y, i, r0, r1, r2, r3, policy="cs")
    ll.gmem_st_v4(z, i, r0, r1, r2, r3, policy="wt")
"""

_KERN_POS = """
def kern(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.bfloat16]) -> ll.void:
    i = ll.threadIdx_x() * 8
    r0: ll.uint32
    r1: ll.uint32
    r2: ll.uint32
    r3: ll.uint32
    ll.gmem_ld_v4(x, i, r0, r1, r2, r3, "lu")
"""

_LOAD_EXPR = {
    "wb": "*reinterpret_cast<const uint4*>(ptr)",
    "ca": "__ldca(reinterpret_cast<const uint4*>(ptr))",
    "cg": "__ldcg(reinterpret_cast<const uint4*>(ptr))",
    "cs": "__ldcs(reinterpret_cast<const uint4*>(ptr))",
    "lu": "__ldlu(reinterpret_cast<const uint4*>(ptr))",
    "nc": "__ldg(reinterpret_cast<const uint4*>(ptr))",
    "cv": "__ldcv(reinterpret_cast<const uint4*>(ptr))",
}

_STORE_STMT = {
    "wb": "*reinterpret_cast<uint4*>(ptr) = value;",
    "cg": "__stcg(reinterpret_cast<uint4*>(ptr), value);",
    "cs": "__stcs(reinterpret_cast<uint4*>(ptr), value);",
    "wt": "__stwt(reinterpret_cast<uint4*>(ptr), value);",
}


class TestGmemV4CachePolicy(unittest.TestCase):

    def test_default_is_plain_wb_helper(self):
        """No policy argument -> historical wb helpers, unsuffixed names."""
        code = _emit(_KERN_DEFAULT)
        self.assertIn("gmem_ld_v4_fn(", code)
        self.assertIn("gmem_st_v4_fn(", code)
        self.assertIn(_LOAD_EXPR["wb"], code)
        self.assertIn(_STORE_STMT["wb"], code)
        # No policy-specific helpers leaked into the default path.
        self.assertNotIn("gmem_ld_v4_", code.replace("gmem_ld_v4_fn", ""))
        self.assertNotIn("gmem_st_v4_", code.replace("gmem_st_v4_fn", ""))

    def test_load_policies_render(self):
        for policy in ("ca", "cg", "cs", "lu", "nc", "cv"):
            code = _emit(_KERN_KW % (policy, "wb"))
            self.assertIn(f"gmem_ld_v4_{policy}_fn(", code)
            self.assertIn(_LOAD_EXPR[policy], code, f"policy {policy} missing its cache operator")
            self.assertIn("#ifndef LK_GMEM_LD_V4_%s_FN_DEFINED" % policy.upper(), code)
            # The policy string must not leak into the emitted call line.
            self.assertNotIn('"%s"' % policy, code)

    def test_store_policies_render(self):
        for policy in ("cg", "cs", "wt"):
            code = _emit(_KERN_KW % ("wb", policy))
            self.assertIn(f"gmem_st_v4_{policy}_fn(", code)
            self.assertIn(_STORE_STMT[policy], code, f"policy {policy} missing its cache operator")
            self.assertIn("#ifndef LK_GMEM_ST_V4_%s_FN_DEFINED" % policy.upper(), code)
            self.assertNotIn('"%s"' % policy, code)

    def test_multiple_policies_coexist(self):
        """Two different load policies + one store policy in one kernel."""
        code = _emit(_KERN_DUAL)
        for helper in ("gmem_ld_v4_nc_fn(", "gmem_ld_v4_cs_fn(", "gmem_st_v4_wt_fn("):
            self.assertIn(helper, code)
        for stmt in (_LOAD_EXPR["nc"], _LOAD_EXPR["cs"], _STORE_STMT["wt"]):
            self.assertIn(stmt, code)
        # Each helper body is emitted exactly once (one guard, one define).
        self.assertEqual(code.count("gmem_ld_v4_nc_fn("), 2,  # 1 def + 1 call
                         "nc helper emitted more than once")
        self.assertEqual(code.count("gmem_ld_v4_cs_fn("), 2)
        self.assertEqual(code.count("gmem_st_v4_wt_fn("), 2)

    def test_positional_policy_string(self):
        code = _emit(_KERN_POS)
        self.assertIn("gmem_ld_v4_lu_fn(", code)
        self.assertIn(_LOAD_EXPR["lu"], code)

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


if __name__ == "__main__":
    unittest.main()
