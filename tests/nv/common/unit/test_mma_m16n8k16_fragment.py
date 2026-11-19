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
# Fragment NAMES (acc, af, bf, ...) are compile-time bindings created by the
# ll.mma_acc_fragment_f32 / ll.mma_operand_fragment_u32 declaration builtins
# in each kernel; they intentionally have no Python-level definition (same
# mechanism as the typing-scope registration used by declaration
# intrinsics), so every bare use carries a per-line "noqa: F821"
# suppression (as in tests/nv/sm90/unit/test_wgmma_acc_fragment.py).
"""NV-common warp-level mma.sync register-fragment CPU regression tests.

Covers the capability ``ir.mma.m16n8k16.bf16.per_thread_fragment`` (the
m16n8k16 typed per-thread fragment model), registered on both the nv/sm90
and nv/sm100 backend lanes:

* ``mma_acc_fragment_f32(name, 4)`` / ``mma_operand_fragment_u32(name, N)``
  declare N INDEPENDENT per-thread register scalars (never an
  address-taken array);
* the fragment-aware ``mma_m16n8k16`` builtin binds each fragment scalar
  as a separate asm operand (``"+f"`` for the four accumulators, ``"r"``
  for the packed u32 A/B lanes) — the hand-CUDA operand shape — when
  acc/a/b ALL name declared fragments of the matching shapes;
* fragment elements are readable AND writable at compile-time-unrolled
  indices (``ll.unroll(range(N))`` counters, literals, provably-static int
  locals), lowering to the per-element scalar ``<name>_<i>`` so a
  multi-consumer elementwise epilogue keeps the fragment in registers;
* the accumulator / A / B coordinate helpers implement the documented PTX
  m16n8k16 fragment layout;
* legality: partial fragment operand sets (fragment acc + array b),
  size-mismatched fragments, undeclared fragment references and dynamic
  (non-constant) fragment indices are hard compile errors, never silent
  local-memory demotions; legacy array operands keep the array path.
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
import little_kernel.backends.nv.common.language.intrin.memory as memory_intrin
from little_kernel.language.builtin_base import (BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR)

# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_epilogue(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.float32]) -> ll.void:
    """Fragments flow through a MMA, then an elementwise read/scale/store."""
    ll.mma_acc_fragment_f32("acc", 4)
    ll.mma_operand_fragment_u32("af", 4)
    ll.mma_operand_fragment_u32("bf", 2)
    tid: ll.int32 = ll.threadIdx_x()
    lane: ll.int32 = tid % 32
    # Pack operands from x at unrolled compile-time indices.
    for i in ll.unroll(range(4)):
        af[i] = lane * 8 + i  # noqa: F821 - fragment binding
    for i in ll.unroll(range(2)):
        bf[i] = lane * 4 + i  # noqa: F821 - fragment binding
    ll.mma_m16n8k16(acc, af, bf, 0)  # noqa: F821 - fragment bindings
    # Same fragment read again for a second, accumulating MMA.
    ll.mma_m16n8k16(acc, af, bf, 1)  # noqa: F821 - fragment bindings
    # Elementwise epilogue on the per-thread accumulator fragment.
    m: ll.float32 = ll.neg_infinity()
    for i in ll.unroll(range(4)):
        m = ll.max_val(m, acc[i])  # noqa: F821 - fragment binding
    for i in ll.unroll(range(4)):
        acc[i] = acc[i] - m  # noqa: F821 - fragment binding
    for i in ll.unroll(range(4)):
        # inline the documented PTX accumulator layout (validated
        # host-side by TestAccumulatorCoordinateHelpers).
        row: ll.int32 = (lane // 4) + 8 * (i // 2)
        col: ll.int32 = 2 * (lane % 4) + (i % 2)
        y[row * 8 + col] = acc[i]  # noqa: F821 - fragment binding
    v: ll.bfloat16 = x[0]  # keep x live  # noqa: F841


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_partial_operands(y: ll.Tensor[ll.float32]) -> ll.void:
    """Illegal: fragment acc with a legacy-array A/B operand set."""
    ll.mma_acc_fragment_f32("acc", 4)
    a = ll.empty([4], dtype=ll.uint32, scope="local")
    b = ll.empty([2], dtype=ll.uint32, scope="local")
    ll.mma_m16n8k16(acc, a, b, 0)  # noqa: F821 - fragment binding
    tid: ll.int32 = ll.threadIdx_x()
    y[tid] = acc[0]  # noqa: F821 - fragment binding


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_size_mismatch(y: ll.Tensor[ll.float32]) -> ll.void:
    """Illegal: 2-elem u32 fragment fed to the 4-register A operand."""
    ll.mma_acc_fragment_f32("acc", 4)
    ll.mma_operand_fragment_u32("af", 2)
    ll.mma_operand_fragment_u32("bf", 2)
    ll.mma_m16n8k16(acc, af, bf, 0)  # noqa: F821 - fragment bindings
    tid: ll.int32 = ll.threadIdx_x()
    y[tid] = acc[0]  # noqa: F821 - fragment binding


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_dynamic_index(y: ll.Tensor[ll.float32]) -> ll.void:
    """Illegal: fragment element access at a runtime (non-constant) index."""
    ll.mma_acc_fragment_f32("acc", 4)
    tid: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = acc[tid]  # noqa: F821 - fragment binding
    y[tid] = v


@lk.ll_kernel(backend='cuda', is_entry=True)
def _legacy_array_mma(y: ll.Tensor[ll.float32]) -> ll.void:
    """Legacy float[4]/u32[4]/u32[2] arrays keep the array helper path."""
    acc = ll.empty([4], dtype=ll.float32, scope="local")
    a = ll.empty([4], dtype=ll.uint32, scope="local")
    b = ll.empty([2], dtype=ll.uint32, scope="local")
    ll.mma_m16n8k16(acc, a, b, 1)
    tid: ll.int32 = ll.threadIdx_x()
    y[tid] = acc[0]


def _compile(kernel):
    return kernel.compile(PASSES["cuda"], codegen_cuda)


# ---------------------------------------------------------------------------
# Declaration and lowering tests
# ---------------------------------------------------------------------------


class TestFragmentBuiltinsExported(unittest.TestCase):
    """The new symbols are real language builtins exported as ll.*."""

    def test_declaration_builtins_exposed(self):
        for fn in (ll.mma_acc_fragment_f32, ll.mma_operand_fragment_u32):
            self.assertTrue(getattr(fn, BUILTIN_ATTR, False))
            self.assertTrue(hasattr(fn, CODEGEN_FUNC_ATTR))
            self.assertTrue(hasattr(fn, EVAL_RETURN_TYPE_ATTR))

    def test_coordinate_helpers_importable(self):
        self.assertTrue(callable(ll.mma_m16n8k16_accumulator_coordinate))
        self.assertTrue(callable(ll.mma_m16n8k16_a_coordinate))
        self.assertTrue(callable(ll.mma_m16n8k16_b_coordinate))


class TestFragmentDeclaration(unittest.TestCase):
    """Declarations emit independent scalars, never arrays."""

    def test_declaration_emits_scalars_not_array(self):
        src = _compile(_frag_epilogue)
        for i in range(4):
            self.assertIn(f"float acc_{i} = 0.0f;", src)
            self.assertIn(f"uint32_t af_{i} = 0u;", src)
        for i in range(2):
            self.assertIn(f"uint32_t bf_{i} = 0u;", src)
        self.assertNotIn("float acc[", src)
        self.assertNotIn("uint32_t af[", src)
        self.assertNotIn("uint32_t bf[", src)


class TestFragmentMmaOperandBinding(unittest.TestCase):
    """Fragment-aware mma.sync binds per-scalar asm operands."""

    def test_mma_binds_per_scalar_operands(self):
        src = _compile(_frag_epilogue)
        self.assertIn("mma_m16n8k16_bf16_frag_fn", src)
        # per-scalar call site
        self.assertIn("mma_m16n8k16_bf16_frag_fn(acc_0, acc_1, acc_2, acc_3, "
                      "af_0, af_1, af_2, af_3, bf_0, bf_1, ", src)
        # one "+f" operand per accumulator scalar
        for i in range(4):
            self.assertIn(f'"+f"(c{i})', src)
            self.assertIn(f'"r"(a{i})', src)
        for i in range(2):
            self.assertIn(f'"r"(b{i})', src)
        self.assertIn("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", src)

    def test_element_subscripts_lower_to_scalars(self):
        src = _compile(_frag_epilogue)
        # epilogue rewrite: acc[i] lowers to acc_<i> on BOTH sides.
        self.assertIn("acc_3 = acc_3 - m", src)
        self.assertNotIn("acc[", src)
        self.assertNotIn("&acc", src)
        self.assertNotIn("af[", src)


class TestAccumulatorCoordinateHelpers(unittest.TestCase):
    """acc/A/B coordinate helpers implement the PTX m16n8k16 layout."""

    def test_accumulator_layout_and_coverage(self):
        seen = set()
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for reg in range(4):
                row, col = memory_intrin.mma_m16n8k16_accumulator_coordinate(lane, reg)
                self.assertEqual(row, g + 8 * (reg >> 1))
                self.assertEqual(col, 2 * t + (reg & 1))
                seen.add((row, col))
        # full 16x8 tile is covered exactly once per element
        self.assertEqual(len(seen), 16 * 8)

    def test_a_layout_and_coverage(self):
        seen = set()
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for reg in range(4):
                for e in range(2):
                    row, col = memory_intrin.mma_m16n8k16_a_coordinate(lane, reg, e)
                    self.assertEqual(row, g + 8 * (reg & 1))
                    self.assertEqual(col, 2 * t + e + 8 * (reg >> 1))
                    seen.add((row, col))
        self.assertEqual(len(seen), 16 * 16)

    def test_b_layout_and_coverage(self):
        seen = set()
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for reg in range(2):
                for e in range(2):
                    k, n = memory_intrin.mma_m16n8k16_b_coordinate(lane, reg, e)
                    self.assertEqual(k, 2 * t + e + 8 * reg)
                    self.assertEqual(n, g)
                    seen.add((k, n))
        self.assertEqual(len(seen), 16 * 8)

    def test_bounds_rejected(self):
        with self.assertRaises(ValueError):
            memory_intrin.mma_m16n8k16_accumulator_coordinate(32, 0)
        with self.assertRaises(ValueError):
            memory_intrin.mma_m16n8k16_accumulator_coordinate(0, 4)
        with self.assertRaises(ValueError):
            memory_intrin.mma_m16n8k16_accumulator_coordinate(0, True)
        with self.assertRaises(ValueError):
            memory_intrin.mma_m16n8k16_a_coordinate(0, 4, 0)
        with self.assertRaises(ValueError):
            memory_intrin.mma_m16n8k16_a_coordinate(0, 0, 2)
        with self.assertRaises(ValueError):
            memory_intrin.mma_m16n8k16_b_coordinate(0, 2, 0)
        with self.assertRaises(ValueError):
            memory_intrin.mma_m16n8k16_b_coordinate(-1, 0, 0)


# ---------------------------------------------------------------------------
# Legality tests (compile errors, never silent demotions)
# ---------------------------------------------------------------------------


class TestFragmentLegality(unittest.TestCase):
    """Illegal fragment uses are hard compile errors."""

    def test_partial_fragment_operands_rejected(self):
        with self.assertRaisesRegex(ValueError, "partial fragment operand set"):
            _compile(_frag_partial_operands)

    def test_size_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            _compile(_frag_size_mismatch)

    def test_dynamic_index_rejected(self):
        with self.assertRaisesRegex(IndexError, "compile-time-unrolled"):
            _compile(_frag_dynamic_index)

    def test_legacy_array_operand_still_array_path(self):
        src = _compile(_legacy_array_mma)
        self.assertIn("mma_m16n8k16_bf16_fn(", src)
        self.assertNotIn("_frag_fn", src)


if __name__ == "__main__":
    unittest.main()
