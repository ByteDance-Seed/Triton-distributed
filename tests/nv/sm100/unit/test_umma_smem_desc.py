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
"""CPU regressions for the SM100 ``make_umma_smem_desc`` intrinsic.

The intrinsic lets a plain ``@lk.ll_kernel(is_entry=True)`` kernel build a
hardware UMMA SMEM descriptor anchored at a typed shared-memory pointer,
without going through the compiler-managed Flow/dataflow layout plan. These
tests never touch a GPU: they assert builtin wiring, the exact CUDA helper
shape (field encodings, masks, version bit) and backend.json registration.
"""

import ast as _ast
import json
import unittest
from pathlib import Path

import little_kernel.language as ll
import little_kernel.backends.nv.sm100.language.intrin.umma as umma
from little_kernel.backends.registry import resolve_language_symbol
from little_kernel.language.builtin_base import (
    BUILTIN_ATTR,
    CODEGEN_FUNC_ATTR,
    EVAL_RETURN_TYPE_ATTR,
    Builtin,
)


class TestMakeUmmaSmemDescBuiltin(unittest.TestCase):
    """Builtin wiring: attributes, return type, eager misuse."""

    def test_is_registered_builtin(self):
        self.assertTrue(getattr(umma.make_umma_smem_desc, BUILTIN_ATTR, False))
        self.assertTrue(callable(getattr(umma.make_umma_smem_desc, CODEGEN_FUNC_ATTR)))

    def test_eval_return_type_is_uint64(self):
        self.assertIs(getattr(umma.make_umma_smem_desc, EVAL_RETURN_TYPE_ATTR)(), ll.uint64)

    def test_direct_call_outside_compilation_raises(self):
        with self.assertRaises(RuntimeError):
            umma.make_umma_smem_desc("s_a", 0, 128, 256, 4)

    def test_distinct_from_encode_umma_smem_desc(self):
        # The pointer-anchored helper must not alias the addr16 encoder.
        self.assertIsNot(umma.make_umma_smem_desc, umma.encode_umma_smem_desc)


class TestMakeUmmaSmemDescCodegen(unittest.TestCase):
    """Shape of the emitted CUDA helper and its call expression."""

    def setUp(self):
        self.result = getattr(umma.make_umma_smem_desc, CODEGEN_FUNC_ATTR)("s_a", "0", "128", "256", "4")

    def test_returns_builtin_with_arguments_plumbed(self):
        self.assertIsInstance(self.result, Builtin)
        self.assertEqual(self.result.return_val, "make_umma_smem_desc_fn(s_a, 0, 128, 256, 4)")

    def test_anchors_pointer_through_cvta_shared(self):
        self.assertIn("__cvta_generic_to_shared(smem_ptr)", self.result.body)

    def test_converts_byte_address_to_16b_units_and_adds_masked_offset(self):
        self.assertIn("(uint32_t)((__cvta_generic_to_shared(smem_ptr) >> 4) & 0x3FFF)", self.result.body)
        self.assertIn("(offset_16b & 0x3FFF)", self.result.body)

    def test_encodes_descriptor_fields(self):
        body = self.result.body
        self.assertIn("(uint64_t)(start_addr_16b & 0x3FFF)", body)
        self.assertIn("(uint64_t)((leading_bytes >> 4) & 0x3FFF) << 16", body)
        self.assertIn("(uint64_t)((stride_bytes >> 4) & 0x3FFF) << 32", body)
        self.assertIn("(uint64_t)(layout_type & 0x7) << 61", body)

    def test_always_sets_sm100_version_one(self):
        self.assertIn("(uint64_t)1 << 46", self.result.body)

    def test_no_extra_includes(self):
        self.assertEqual(self.result.includes, [])


class TestMakeUmmaSmemDescKernelCodegen(unittest.TestCase):
    """End-to-end emission through the NV CUDA emitter, no GPU required."""

    def test_full_kernel_codegen(self):
        from little_kernel.backends.nv.common.codegen.emitter import (codegen_cuda)

        src = ("def kern(s_a: Tensor[bfloat16], off: uint32) -> uint64:\n"
               "    desc: uint64 = make_umma_smem_desc(s_a, off, 128, 256, 4)\n"
               "    return desc\n")
        namespace = dict(ll.__dict__)
        # ll resolves backend symbols lazily; bind the intrinsic explicitly
        # so this test never depends on import order.
        namespace["make_umma_smem_desc"] = umma.make_umma_smem_desc
        out = codegen_cuda(_ast.parse(src), namespace)
        self.assertIn("make_umma_smem_desc_fn(s_a, off, 128, 256, 4)", out)
        self.assertIn("make_umma_smem_desc_fn(", out)
        self.assertIn("__cvta_generic_to_shared", out)


class TestMakeUmmaSmemDescRegistration(unittest.TestCase):
    """The sm100 backend descriptor must export and resolve the symbol."""

    def test_backend_json_exports_symbol(self):
        backend_json = (Path(umma.__file__).resolve().parents[2] / "backend.json")
        data = json.loads(backend_json.read_text())
        exports = data.get("language_exports") or data.get("symbols") or {}
        self.assertEqual(exports.get("make_umma_smem_desc"), "little_kernel.backends.nv.sm100.language.intrin.umma:"
                         "make_umma_smem_desc")

    def test_registry_resolves_sm100_symbol(self):
        self.assertIs(resolve_language_symbol("make_umma_smem_desc", "nv/sm100"), umma.make_umma_smem_desc)


if __name__ == "__main__":
    unittest.main()
