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

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
"""8-byte (u32x2) read-only global load with the .nc cache policy
(load_64bit_nc), the non-coherent/read-only sibling of load_64bit /
load_64bit_cg / load_64bit_streaming. It must type as u32x2 (lanes
extracted via vec_u32x2_lane), emit __ldg on a reinterpret-cast uint2
pointer, keep the generic-pointer contract so a 1-byte element base
(uint8 tensor) is legal, and be importable from little_kernel.language.
"""

import ast
import unittest

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(source: str) -> str:
    tree = ast.parse(source)
    return codegen_cuda(tree, ctx={"ll": ll}, emit_header=False)


_KERN_NC_U8_BASE = """
def kern(x: ll.Tensor[ll.uint8], out: ll.Tensor[ll.uint32], n: ll.int32) -> ll.void:
    i = ll.threadIdx_x()
    if i < n:
        v = ll.load_64bit_nc(x + i * 8)
        out[i * 2] = ll.vec_u32x2_lane(v, 0)
        out[i * 2 + 1] = ll.vec_u32x2_lane(v, 1)
"""

_KERN_NC_U32_BASE = """
def kern(x: ll.Tensor[ll.uint32], out: ll.Tensor[ll.uint32]) -> ll.void:
    i = ll.threadIdx_x()
    v = ll.load_64bit_nc(x + i * 2)
    out[i * 2] = ll.vec_u32x2_lane(v, 0)
"""


class TestLoad64bitNc(unittest.TestCase):

    def test_nc_renders_ldg_on_uint2(self):
        code = _emit(_KERN_NC_U8_BASE)
        self.assertIn("__ldg(reinterpret_cast<const uint2*>", code)
        self.assertNotIn("__ldcs(", code)
        self.assertNotIn("__ldcg(", code)

    def test_nc_keeps_generic_pointer_contract_for_u8_base(self):
        # 1-byte element base: x + i*8 is a uint8*; must still compile and
        # yield uint2 lanes (no element-typed view forced by the intrinsic).
        code = _emit(_KERN_NC_U8_BASE)
        self.assertIn("load_64bit_nc_fn(", code)
        self.assertIn(".x", code)
        self.assertIn(".y", code)

    def test_nc_u32_base_lowers(self):
        code = _emit(_KERN_NC_U32_BASE)
        self.assertIn("__ldg(reinterpret_cast<const uint2*>", code)

    def test_intrinsic_exposed_on_language_module(self):
        self.assertTrue(hasattr(ll, "load_64bit_nc"))


if __name__ == "__main__":
    unittest.main()
