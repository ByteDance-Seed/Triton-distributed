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
"""CPU codegen regression for the prmt.b32 byte-permute intrinsic.

Covers the dtype-generic SIMT primitive ``ll.prmt`` from
``backends/nv/common/language/intrin/cuda_asm.py`` without a GPU:

* Export wiring: both nv backend descriptors (sm90, sm100) resolve ``prmt``
  through ``language_exports`` to the single canonical cuda_asm definition
  (no per-arch divergence, matching the registry discipline of the sibling
  bit ops ``popc_u32`` / ``bfind_u32`` / ``prefix_lsb_mask_i32``).
* Structural: the emitted helper body must issue ``prmt.b32`` with the
  output register followed by the a, b and selector register operands, all
  carried in 32-bit registers, and must type the helper as one
  uint32_t(a, b, sel) -> uint32_t function.
* End-to-end: a minimal entry kernel that loads a uint32 pack and permutes
  a pair of registers with a constant selector compiles to CUDA source on
  the ordinary scalar/SIMT pipeline, and the generated text contains the
  ``prmt.b32`` instruction -- this is the exact CPU lowering path the fp8
  quantize+transpose staging reproducer needs.

Semantics note: ``sel`` is a runtime uint32 operand; a call-site selector
literal folds to an immediate through the constant-propagation pass, so the
shared primitive stays dtype- and selector-generic rather than baking one
transpose pattern into the compiler.

No CUDA toolchain or GPU is needed; codegen output is consumed as source
text via the BuiltinOp ``codegen`` staticmethod and ``kernel.compile``.
"""

import unittest

import little_kernel.language as ll
from little_kernel.backends.nv.common.language.intrin import cuda_asm
from little_kernel.language import builtin_base as bb


def _codegen_body(*args):
    codegen = getattr(cuda_asm.prmt, bb.CODEGEN_FUNC_ATTR)
    return codegen(*args).body


class TestPrmtExportWiring(unittest.TestCase):
    """``ll.prmt`` must resolve on every nv backend descriptor."""

    def test_language_exports_resolve(self):
        from little_kernel.backends.registry import language_symbols, resolve_language_symbol
        for backend in ("nv/sm90", "nv/sm100"):
            self.assertIn("prmt", language_symbols(backend), backend)
            self.assertIs(resolve_language_symbol("prmt", backend), cuda_asm.prmt, backend)


class TestPrmtStructure(unittest.TestCase):
    """Structural checks on the emitted helper source."""

    def test_prmt_b32_instruction(self):
        body = _codegen_body("a_val", "b_val", "sel_val")
        self.assertIn('asm volatile("prmt.b32 %0, %1, %2, %3;"', body)

    def test_operand_constraints(self):
        body = _codegen_body("a_val", "b_val", "sel_val")
        # Single 32-bit output register; three 32-bit inputs (a, b, sel).
        self.assertIn(': "=r"(ret)', body)
        self.assertIn('"r"(a)', body)
        self.assertIn('"r"(b)', body)
        self.assertIn('"r"(sel)', body)

    def test_typed_uint32_signature(self):
        body = _codegen_body("a", "b", "sel")
        self.assertIn("uint32_t prmt(", body)
        self.assertIn("uint32_t a, uint32_t b, uint32_t sel", body)


class TestPrmtSemantics(unittest.TestCase):
    """Python model of prmt.b32 mode-0 byte gather vs the documented PTX
    contract on the 8-byte pair {b, a}."""

    @staticmethod
    def _model(a, b, sel):
        pool = [(a >> (8 * i)) & 0xFF for i in range(4)] + \
               [(b >> (8 * i)) & 0xFF for i in range(4)]
        out = 0
        for i in range(4):
            idx = (sel >> (4 * i)) & 0xF
            out |= pool[idx & 0x7] << (8 * i)
        return out

    def test_reproducer_transpose_selector(self):
        # The fp8 staging path's 4x4 byte-transpose selector 0x5140:
        # result byte 0 <- pool[0], byte 1 <- pool[4], byte 2 <- pool[1],
        # byte 3 <- pool[5].
        a = 0x33221100  # pool bytes 0..3 = 0x00, 0x11, 0x22, 0x33
        b = 0x77665544  # pool bytes 4..7 = 0x44, 0x55, 0x66, 0x77
        got = self._model(a, b, 0x5140)
        self.assertEqual(got, 0x55114400)

    def test_identity_and_swizzle_selectors(self):
        a = 0xDEADBEEF
        b = 0x01020304
        # Identity on a: bytes 0,1,2,3.
        self.assertEqual(self._model(a, b, 0x3210), a)
        # Identity on b: bytes 4,5,6,7.
        self.assertEqual(self._model(a, b, 0x7654), b)
        # Byte reverse of a.
        self.assertEqual(self._model(a, b, 0x0123), 0xEFBEADDE)


class TestPrmtEndToEnd(unittest.TestCase):
    """Entry-kernel CPU lowering through the ordinary cuda pipeline."""

    def test_entry_kernel_emits_prmt(self):
        from little_kernel.core.compile import ll_kernel
        from little_kernel.core.passes import PASSES
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

        @ll_kernel(backend="cuda", is_entry=True)
        def prmt_kernel(packs: ll.Tensor[ll.uint32]) -> ll.void:
            tid = ll.threadIdx_x()
            a = packs[tid]
            b = packs[tid + 32]
            t = ll.prmt(a, b, 0x5140)
            packs[tid] = t

        text = prmt_kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertIn("prmt.b32", text)
        self.assertIn("uint32_t prmt(", text)


if __name__ == "__main__":
    unittest.main()
