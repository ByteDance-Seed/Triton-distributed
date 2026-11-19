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
"""CPU regression coverage for the tf32 dtype path of the mma_m16n8k8 IR atom.

Pins:
- legality: "tf32" (alias "tfloat32") is accepted by the m16n8k8 atom only;
  the 16-bit atoms and the f64 atom keep rejecting it, and unknown dtypes are
  rejected everywhere.
- codegen: the tf32 path emits exactly
  mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 with fp32 register
  operands (A: 4 regs/lane, B: 2 regs/lane, truncated to tf32 inside the
  tensor core) and 4 fp32 accumulators; the pre-existing bf16/f16 bodies are
  byte-for-byte unchanged.
- end-to-end: a kernel calling ll.mma_m16n8k8(..., dtype="tf32") lowers on
  CPU to a CUDA source containing the tf32 helper and instruction (both the
  nv/sm90 and nv/sm100 backends route through the shared atom).

No GPU is required: these tests exercise type checking and source emission
only. They use unittest.TestCase and are discoverable by both unittest and
pytest. Paired device-side validation is left to the queued GPU harness.
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends import get_backend
from little_kernel.backends.nv.common.language.intrin.memory_ops.matrix import (
    _mma_ptx_dtype,
    codegen_mma_m16n8k8,
    codegen_mma_m16n8k16,
    codegen_mma_m16n16k16,
    codegen_mma_m16n8k4_f64,
)
from little_kernel.core.passes import PASSES

# ---------------------------------------------------------------------------
# dtype legality
# ---------------------------------------------------------------------------


class TestTf32DtypeLegality(unittest.TestCase):
    """Whitelist behavior of _mma_ptx_dtype with and without the tf32 opt-in."""

    def test_tf32_accepted_with_opt_in(self):
        self.assertEqual(_mma_ptx_dtype("tf32", allow_tf32=True), "tf32")
        self.assertEqual(_mma_ptx_dtype("tfloat32", allow_tf32=True), "tf32")

    def test_tf32_quoted_and_uppercase_forms_accepted(self):
        # quoted tokens (frontend constant spelling) and case/b16 aliases behave
        self.assertEqual(_mma_ptx_dtype('"tf32"', allow_tf32=True), "tf32")
        self.assertEqual(_mma_ptx_dtype("TF32", allow_tf32=True), "tf32")

    def test_tf32_rejected_without_opt_in(self):
        with self.assertRaisesRegex(ValueError, "mma.sync operand dtype"):
            _mma_ptx_dtype("tf32")

    def test_existing_aliases_unchanged(self):
        self.assertEqual(_mma_ptx_dtype("bf16"), "bf16")
        self.assertEqual(_mma_ptx_dtype("bfloat16"), "bf16")
        self.assertEqual(_mma_ptx_dtype("f16"), "f16")
        self.assertEqual(_mma_ptx_dtype("half"), "f16")
        self.assertEqual(_mma_ptx_dtype("double"), "f64")

    def test_unknown_dtype_rejected_even_with_opt_in(self):
        with self.assertRaises(ValueError):
            _mma_ptx_dtype("int8", allow_tf32=True)

    def test_other_mma_atoms_reject_tf32(self):
        # PTX defines no tf32 opcode for m16n8k16/m16n16k16, and the f64 atom
        # is f64-only.
        for codegen in (codegen_mma_m16n8k16, codegen_mma_m16n16k16, codegen_mma_m16n8k4_f64):
            with self.subTest(codegen=codegen.__name__):
                with self.assertRaisesRegex(ValueError, "mma.sync operand dtype"):
                    codegen("c", "a", "b", 1, dtype="tf32")


# ---------------------------------------------------------------------------
# codegen body pins
# ---------------------------------------------------------------------------


class TestTf32CodegenBody(unittest.TestCase):
    """Pinned codegen body for the tf32 path and unchanged 16-bit bodies."""

    def test_tf32_body_pins_instruction_and_fragments(self):
        b = codegen_mma_m16n8k8("acc", "a", "b", 1, dtype="tf32")
        # exactly the PTX instruction from the capability spec
        self.assertEqual(
            b.body.count("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"),
            1,
        )
        # fp32 fragments: 4 accumulator ("+f"), 4 A + 2 B operand regs ("f")
        self.assertEqual(b.body.count('"+f"('), 4)
        self.assertEqual(b.body.count('"f"('), 6)
        # fp32 register-array signatures (truncation happens in-tensor-core only)
        self.assertIn("float (&c)[4]", b.body)
        self.assertIn("const float (&a)[4]", b.body)
        self.assertIn("const float (&b)[2]", b.body)
        # distinct helper name so bf16/f16/tf32 can coexist in one kernel
        self.assertEqual(b.return_val, "mma_m16n8k8_tf32_fn(acc, a, b, 1)")
        # accumulate flag preserved: zero-init prelude stays in the body
        self.assertIn("if (!accumulate)", b.body)

    def test_bf16_body_unchanged(self):
        b = codegen_mma_m16n8k8("acc", "a", "b", 1, dtype="bf16")
        self.assertIn("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32", b.body)
        self.assertEqual(b.return_val, "mma_m16n8k8_fn(acc, a, b, 1)")
        self.assertIn("const uint32_t (&a)[2], uint32_t b", b.body)

    def test_f16_body_unchanged(self):
        b = codegen_mma_m16n8k8("acc", "a", "b", 1, dtype="f16")
        self.assertIn("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32", b.body)
        self.assertEqual(b.return_val, "mma_m16n8k8_f16_fn(acc, a, b, 1)")


# ---------------------------------------------------------------------------
# end-to-end CPU lowering on both NV backends that route through the atom
# ---------------------------------------------------------------------------

_TF32_INSN = "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"


def _lower_tf32_probe(backend):
    codegen_cuda = get_backend(backend).codegen.generate

    @lk.ll_kernel(backend=backend, is_entry=True)
    def tf32_probe(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32]) -> ll.void:
        a0: ll.float32 = x[0]
        acc = ll.zeros([4], dtype=ll.float32, scope="local")
        ll.mma_m16n8k8(acc, (a0, a0, a0, a0), (a0, a0), 1, dtype="tf32")
        y[0] = acc[0]

    return tf32_probe.compile(PASSES[backend], codegen_cuda)


class TestTf32EndToEndLowering(unittest.TestCase):
    """CPU-only end-to-end lowering through both NV backends sharing the atom."""

    def test_end_to_end_tf32_lowering(self):
        for backend in ("nv/sm90", "nv/sm100"):
            with self.subTest(backend=backend):
                src = _lower_tf32_probe(backend)
                self.assertIn(_TF32_INSN, src)
                self.assertIn("mma_m16n8k8_tf32_fn", src)
                # 6 input operand registers + shared accumulators at the call
                # site body
                self.assertEqual(src.count('"+f"(c['), 4)

    def test_end_to_end_bad_dtype_still_raises(self):
        for backend in ("nv/sm90", "nv/sm100"):
            with self.subTest(backend=backend):
                codegen_cuda = get_backend(backend).codegen.generate

                @lk.ll_kernel(backend=backend, is_entry=True)
                def bad_probe(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32]) -> ll.void:
                    a0: ll.float32 = x[0]
                    acc = ll.zeros([4], dtype=ll.float32, scope="local")
                    ll.mma_m16n8k8(acc, (a0, a0, a0, a0), (a0, a0), 1, dtype="fp8")
                    y[0] = acc[0]

                with self.assertRaisesRegex(ValueError, "mma.sync operand dtype"):
                    bad_probe.compile(PASSES[backend], codegen_cuda)


if __name__ == "__main__":
    unittest.main()
