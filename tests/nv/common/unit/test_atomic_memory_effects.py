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
"""
Memory-effect coverage tests for the hardware-atomic intrinsics
(language/intrin/atomic.py).

Contract under test (mutation/aliasing-aware codegen audit):

- EVERY hardware-atomic builtin in the module declares a region effect via
  MEMORY_EFFECTS_ATTR, so effect-driven analyses (dataflow mem-event
  collection, channel/pipeline validation, race and write-only-init checks)
  never see an atomic read-modify-write or fire-and-forget reduction as
  side-effect free.  The decorator only sets MEMORY_EFFECTS_ATTR when a
  non-empty effects tuple is passed, so attribute presence is sufficient
  proof of declaration.
- Every declared effect uses the shared conservative kind
  RW_GMEM_ATOMIC_SYNC: RW access on GMEM with atomic=True (hardware
  serialization; concurrent atomics to the same region are not a data
  race).
- The effect operand and region binding match the builtin's actual
  parameter names: pointer-form ops bind operand "addr" (no operand-derived
  region), (tensor, index) element/scatter ops bind operand "tensor" with a
  single-element region at offset operand "index", and the red.global
  vector reductions bind operand "mem" with a 2- or 4-element region.

All assertions are on function attributes only (deterministic, no GPU).
"""

import inspect
import unittest

import little_kernel.backends.nv.common.language.intrin.atomic as atomic
from little_kernel.language.builtin_base import BUILTIN_ATTR, MEMORY_EFFECTS_ATTR, ORIGINAL_PY_FUNC_ATTR
from little_kernel.core.memory_effects import Access, MemorySpace, RW_GMEM_ATOMIC_SYNC

# Every public hardware-atomic builtin defined in language/intrin_atomic.
# Keeping the list explicit makes the coverage property testable: adding a
# new atomic intrinsic requires adding it HERE (with its declared effect
# binding), so a silently un-effected intrinsic fails this suite.
POINTER_FORM = (
    "atomic_add",
    "atomic_add_i64",
    "atomic_add_acq_rel_gpu",
    "atomic_cas_system",
    "atomic_cas",
    "atomic_cas_i64",
    "atomic_min",
    "atomic_max",
    "atomic_or",
    "atomic_and",
    "atomic_sub",
    "atomic_xor",
    "atomic_inc",
)

# (tensor, index, ...) element ops: one-element region at offset "index".
ELEMENT_FORM = (
    "atomic_add_global_i32",
    "atomic_add_global_u32",
    "atomic_add_global_u64",
    "atomic_add_global_f32",
    "atomic_cas_global_i32",
    "atomic_cas_global_u32",
    "atomic_exch_global_i32",
    "atomic_exch_global_u32",
    "atomic_sub_global_i32",
    "atomic_sub_global_u32",
    "atomic_xor_global_i32",
    "atomic_xor_global_u32",
)

# (mem, index, v0..vN) red.global.add vector reductions: N-element region.
VECTOR_FORM = (
    ("atomic_add_f32x2", 2),
    ("atomic_add_f32x4", 4),
)

ALL_ATOMICS = POINTER_FORM + ELEMENT_FORM + tuple(n for n, _ in VECTOR_FORM)


def _signature_param_names(builtin_fn):
    original = getattr(builtin_fn, ORIGINAL_PY_FUNC_ATTR, builtin_fn)
    return set(inspect.signature(original).parameters)


class TestAtomicMemoryEffectPresence(unittest.TestCase):
    """Every hardware-atomic builtin must declare at least one effect."""

    def test_all_atomics_declare_memory_effects(self):
        missing = []
        for name in ALL_ATOMICS:
            fn = getattr(atomic, name)
            self.assertTrue(getattr(fn, BUILTIN_ATTR, False), f"{name} is not a builtin")
            if not getattr(fn, MEMORY_EFFECTS_ATTR, ()):
                missing.append(name)
        self.assertEqual(missing, [], f"atomic intrinsics without MEMORY_EFFECTS_ATTR: "
                         f"{missing}")

    def test_no_unlisted_atomic_builtins(self):
        # Any new attribute named atomic_* must be enumerated above so its
        # effect binding is audited too (guards against future drift between
        # this list and the module).
        unlisted = [name for name in dir(atomic) if name.startswith("atomic_") and name not in ALL_ATOMICS]
        self.assertEqual(
            unlisted, [], f"unlisted atomic_* intrinsic(s) in atomic.py: "
            f"{unlisted}; add them to POINTER_FORM, "
            f"ELEMENT_FORM or VECTOR_FORM")


class TestAtomicMemoryEffectKind(unittest.TestCase):
    """All atomic effects share the conservative RW_GMEM_ATOMIC_SYNC kind."""

    def test_kind_is_gmem_atomic_rw(self):
        self.assertTrue(RW_GMEM_ATOMIC_SYNC.atomic)
        self.assertIs(RW_GMEM_ATOMIC_SYNC.access, Access.RW)
        self.assertIs(RW_GMEM_ATOMIC_SYNC.space, MemorySpace.GMEM)
        for name in ALL_ATOMICS:
            fn = getattr(atomic, name)
            for eff in getattr(fn, MEMORY_EFFECTS_ATTR):
                with self.subTest(intrinsic=name, operand=eff.operand):
                    self.assertIs(eff.kind, RW_GMEM_ATOMIC_SYNC)

    def test_atomic_kind_value_rendering(self):
        # The model carries no ordering vocabulary, so all scope/order
        # variants collapse into one kind; pin its rendering.
        self.assertEqual(RW_GMEM_ATOMIC_SYNC.value, "rw.gmem.generic.sync.atomic")


class TestAtomicEffectRegionBinding(unittest.TestCase):
    """Effect operands/regions bind to real parameters of each builtin."""

    def _check_operand_is_parameter(self, name):
        fn = getattr(atomic, name)
        params = _signature_param_names(fn)
        for eff in getattr(fn, MEMORY_EFFECTS_ATTR):
            self.assertIn(eff.operand, params, f"{name}: effect operand {eff.operand!r} is not a parameter")
            if eff.region is not None and eff.region.offset_operand is not None:
                self.assertIn(eff.region.offset_operand, params, f"{name}: region offset operand "
                              f"{eff.region.offset_operand!r} is not a parameter")

    def test_pointer_form_effects(self):
        for name in POINTER_FORM:
            with self.subTest(intrinsic=name):
                fn = getattr(atomic, name)
                effects = getattr(fn, MEMORY_EFFECTS_ATTR)
                self.assertEqual(len(effects), 1)
                eff = effects[0]
                self.assertEqual(eff.operand, "addr")
                # Pointer form touches exactly the element at *addr; no
                # operand-derived region.
                self.assertIsNone(eff.region)
                self._check_operand_is_parameter(name)

    def test_element_form_effects(self):
        for name in ELEMENT_FORM:
            with self.subTest(intrinsic=name):
                fn = getattr(atomic, name)
                effects = getattr(fn, MEMORY_EFFECTS_ATTR)
                self.assertEqual(len(effects), 1)
                eff = effects[0]
                self.assertEqual(eff.operand, "tensor")
                self.assertIsNotNone(eff.region)
                self.assertEqual(eff.region.offset_operand, "index")
                self.assertEqual(eff.region.extent, 1)
                self._check_operand_is_parameter(name)

    def test_vector_form_effects(self):
        for name, width in VECTOR_FORM:
            with self.subTest(intrinsic=name):
                fn = getattr(atomic, name)
                effects = getattr(fn, MEMORY_EFFECTS_ATTR)
                self.assertEqual(len(effects), 1)
                eff = effects[0]
                self.assertEqual(eff.operand, "mem")
                self.assertIsNotNone(eff.region)
                self.assertEqual(eff.region.offset_operand, "index")
                self.assertEqual(eff.region.extent, width)
                self._check_operand_is_parameter(name)


if __name__ == "__main__":
    unittest.main()
