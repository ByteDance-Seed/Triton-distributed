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

import unittest
import little_kernel.language as ll
import little_kernel.backends.nv.common.language.intrin.memory as memory
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, Builtin


class TestGmemStoreCgIntrinsics(unittest.TestCase):
    """Tests for the cache-global (.cg) 16-byte GMEM store family.

    The `_cg` stores are cache-hint twins of the plain typed vector stores:
    same lane signature, same value bits, lowered through `__stcg`
    (st.global.cg) instead of a plain vector store.
    """

    CG_NAMES = (
        "gmem_st_v4f32_cg",
        "gmem_st_v4s32_cg",
        "gmem_st_v8f16_cg",
        "gmem_st_v8bf16_cg",
    )

    def test_are_builtins(self):
        for name in self.CG_NAMES:
            with self.subTest(name=name):
                fn = getattr(memory, name)
                self.assertTrue(getattr(fn, BUILTIN_ATTR, False))

    def test_exported_from_language(self):
        for name in self.CG_NAMES:
            with self.subTest(name=name):
                self.assertTrue(hasattr(ll, name))

    def test_v4s32_codegen(self):
        codegen = getattr(memory.gmem_st_v4s32_cg, CODEGEN_FUNC_ATTR)
        result = codegen("out", "i", "a", "b", "c", "d")
        self.assertIsInstance(result, Builtin)
        self.assertIn("__stcg", result.body)
        self.assertIn("int4", result.body)
        self.assertIn("gmem_st_v4s32_cg_fn(out + i, a, b, c, d)", result.return_val)

    def test_v4f32_codegen(self):
        codegen = getattr(memory.gmem_st_v4f32_cg, CODEGEN_FUNC_ATTR)
        result = codegen("out", "i", "a", "b", "c", "d")
        self.assertIsInstance(result, Builtin)
        self.assertIn("__stcg", result.body)
        self.assertIn("float4", result.body)
        self.assertIn("gmem_st_v4f32_cg_fn(out + i, a, b, c, d)", result.return_val)

    def test_v8f16_codegen(self):
        codegen = getattr(memory.gmem_st_v8f16_cg, CODEGEN_FUNC_ATTR)
        result = codegen("out", "i", "a", "b", "c", "d", "e", "f", "g", "h")
        self.assertIsInstance(result, Builtin)
        self.assertIn("__stcg", result.body)
        self.assertIn("__floats2half2_rn", result.body)
        self.assertIn("<cuda_fp16.h>", result.includes)
        self.assertIn(
            "gmem_st_v8f16_cg_fn(out + i, a, b, c, d, e, f, g, h)",
            result.return_val,
        )

    def test_v8bf16_codegen(self):
        codegen = getattr(memory.gmem_st_v8bf16_cg, CODEGEN_FUNC_ATTR)
        result = codegen("out", "i", "a", "b", "c", "d", "e", "f", "g", "h")
        self.assertIsInstance(result, Builtin)
        self.assertIn("__stcg", result.body)
        self.assertIn("__floats2bfloat162_rn", result.body)
        self.assertIn("<cuda_bf16.h>", result.includes)

    def test_plain_stores_untouched(self):
        """The plain store codegens must not gain the cache hint."""
        codegen = getattr(memory.gmem_st_v4s32, CODEGEN_FUNC_ATTR)
        result = codegen("out", "i", "a", "b", "c", "d")
        self.assertNotIn("__stcg", result.body)


if __name__ == "__main__":
    unittest.main()
