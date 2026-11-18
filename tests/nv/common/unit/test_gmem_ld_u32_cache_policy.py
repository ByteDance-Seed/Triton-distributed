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
"""Scalar 32-bit global loads with explicit cache-policy hints
(gmem_ld_cs_u32 / gmem_ld_cg_u32), the streaming (evict-first) and
L2-only siblings of gmem_ld_nc_u32. Each intrinsic must type as uint32 and
emit the matching __ldcs/__ldcg CUDA builtin on a reinterpret-cast byte
pointer; the intrinsic must be importable from little_kernel.language.
"""

import ast
import unittest

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(source: str) -> str:
    tree = ast.parse(source)
    return codegen_cuda(tree, ctx={"ll": ll}, emit_header=False)


_KERN_CS = """
def kern(x: ll.Tensor[ll.uint32], out: ll.Tensor[ll.uint32]) -> ll.void:
    i = ll.threadIdx_x()
    m: ll.uint32 = ll.gmem_ld_cs_u32(ll.ptr_byte_offset(x, i * 4))
    out[i] = m
"""

_KERN_CG = """
def kern(x: ll.Tensor[ll.uint32], out: ll.Tensor[ll.uint32]) -> ll.void:
    i = ll.threadIdx_x()
    m: ll.uint32 = ll.gmem_ld_cg_u32(ll.ptr_byte_offset(x, i * 4))
    out[i] = m
"""


class TestGmemLdU32CachePolicy(unittest.TestCase):

    def test_cs_u32_renders_ldcs(self):
        code = _emit(_KERN_CS)
        self.assertIn("__ldcs(reinterpret_cast<const uint32_t*>", code)
        self.assertNotIn("__ldg(", code)

    def test_cg_u32_renders_ldcg(self):
        code = _emit(_KERN_CG)
        self.assertIn("__ldcg(reinterpret_cast<const uint32_t*>", code)

    def test_intrinsics_exposed_on_language_module(self):
        self.assertTrue(hasattr(ll, "gmem_ld_cs_u32"))
        self.assertTrue(hasattr(ll, "gmem_ld_cg_u32"))


if __name__ == "__main__":
    unittest.main()
