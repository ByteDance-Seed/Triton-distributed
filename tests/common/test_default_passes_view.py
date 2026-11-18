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
"""Regression tests for the ``PASSES`` default-pipeline mapping view.

A backend alias (e.g. ``cuda`` for ``nv/sm90``) must never register a
second, alias-keyed stage list: every accessor resolves the requested name
to the canonical backend id first, so ``PASSES[alias]`` and
``PASSES[canonical]`` share one registry entry and one pipeline.
"""
import sys
import types
import unittest
from unittest import mock

from little_kernel.backends.interfaces import BackendUnavailable, BackendComponents
from little_kernel.backends.registry import backend_specs, resolve_backend_spec
from little_kernel.core.passes import PASSES
from little_kernel.core.passes.pass_registry import PassRegistry, get_pass_registry, get_passes

# Every (name, canonical id) pair the view must normalize: installed backend
# ids plus each declared alias, resolved the way the registry resolves them
# (a shared alias may point at a higher-priority backend than the descriptor
# that lists it).
PAIRS = tuple(
    dict.fromkeys([(spec.backend_id, spec.backend_id)
                   for spec in backend_specs()] + [(alias, resolve_backend_spec(alias).backend_id)
                                                   for spec in backend_specs()
                                                   for alias in spec.aliases]))


class _StubPassProvider:
    """Minimal PassProvider that registers one stage under the given key."""

    def __init__(self, registry):
        self._registry = registry

    def register(self, backend_key):
        self._registry.register_pass(backend_key, lambda tree, ctx: tree, pass_id="stub_pass")
        return self._registry.get_pipeline(backend_key)


def _stub_backend_modules(registry):
    """Build a stubbed ``little_kernel.backends`` package with one aliased spec."""
    from little_kernel.backends import interfaces

    spec = interfaces.BackendSpec(backend_id="nv/sm90", factory="stub:make", aliases=("cu", ))
    components = BackendComponents(
        spec=spec,
        codegen=mock.Mock(spec=interfaces.CodeGenerator),
        passes=_StubPassProvider(registry),
        toolchain=mock.Mock(spec=interfaces.Toolchain),
        builder=mock.Mock(spec=interfaces.KernelBuilder),
        runtime=mock.Mock(spec=interfaces.RuntimeProvider),
        target=mock.Mock(spec=interfaces.TargetPolicy),
    )

    def resolve_backend_spec_stub(backend, architecture=None):
        if backend in (spec.backend_id, *spec.aliases):
            return spec
        raise BackendUnavailable(f"Backend {backend!r} is not included")

    backends_pkg = types.ModuleType("little_kernel.backends")
    backends_pkg.resolve_backend_spec = resolve_backend_spec_stub
    backends_pkg.BackendUnavailable = BackendUnavailable
    backends_pkg.get_backend = lambda backend, architecture=None: components
    registry_mod = types.ModuleType("little_kernel.backends.registry")
    registry_mod.backend_specs = lambda: (spec, )
    registry_mod.resolve_backend_spec = resolve_backend_spec_stub
    return {"little_kernel.backends": backends_pkg, "little_kernel.backends.registry": registry_mod}


class TestDefaultPassesView(unittest.TestCase):

    def test_membership_matches_canonical_and_alias_keys(self):
        keys = PASSES.keys()
        for name, _canonical in PAIRS:
            with self.subTest(backend=name):
                self.assertIn(name, PASSES)
                self.assertIn(name, keys)

    def test_unknown_backend_is_not_contained(self):
        self.assertNotIn("missing_vendor/missing_hardware", PASSES)
        self.assertIs(PASSES.get("missing_vendor/missing_hardware"), None)
        self.assertEqual(PASSES.get("missing_vendor/missing_hardware", []), [])

    def test_alias_and_canonical_share_one_registry_entry(self):
        registry = get_pass_registry()
        for name, canonical in PAIRS:
            with self.subTest(backend=name):
                PASSES[name]
                if name != canonical:
                    self.assertEqual(get_passes(name), [], "an alias must never own an alias-keyed stage list")
                canonical_ids = registry.get_pass_ids(canonical)
                self.assertGreater(len(canonical_ids), 0)
                self.assertEqual(canonical_ids, tuple(dict.fromkeys(canonical_ids)),
                                 "duplicate pass ids registered for one backend")

    def test_alias_and_canonical_return_equal_fresh_lists(self):
        for name, canonical in PAIRS:
            if name == canonical:
                continue
            with self.subTest(backend=name):
                via_alias = PASSES[name]
                via_canonical = PASSES[canonical]
                self.assertEqual([getattr(p, "__name__", repr(p)) for p in via_alias],
                                 [getattr(p, "__name__", repr(p)) for p in via_canonical])
                again = PASSES[name]
                self.assertIsNot(via_alias, again)
                via_alias.clear()
                self.assertGreater(len(PASSES[name]), 0, "mutating one handed-out list must not affect later lookups")

    def test_pipeline_is_keyed_by_canonical_backend_id(self):
        for name, canonical in PAIRS:
            with self.subTest(backend=name):
                via_name = PASSES.pipeline(name)
                via_canonical = PASSES.pipeline(canonical)
                self.assertEqual(via_name, via_canonical)
                self.assertEqual(via_name.backend, canonical)
                self.assertEqual(via_name.pass_ids, via_canonical.pass_ids)

    def test_stubbed_aliased_backend_never_registers_an_alias_keyed_pipeline(self):
        """One spec {backend_id: 'nv/sm90', aliases: ['cu']} against a fresh registry."""
        registry = PassRegistry()
        stubs = _stub_backend_modules(registry)
        with mock.patch.dict(sys.modules, stubs), \
             mock.patch("little_kernel.core.passes.pass_registry._pass_registry", registry):
            self.assertEqual(PASSES.keys(), ("nv/sm90", "cu"))
            self.assertIn("cu", PASSES)
            via_alias = PASSES["cu"]
            via_canonical = PASSES["nv/sm90"]
            self.assertEqual(len(via_alias), 1)
            self.assertEqual(via_alias, via_canonical)
            self.assertIsNot(via_alias, via_canonical)
            via_alias.clear()
            self.assertEqual(len(PASSES["cu"]), 1, "mutating one handed-out list must not affect later lookups")
            self.assertEqual(PASSES.pipeline("cu"), PASSES.pipeline("nv/sm90"))
            self.assertEqual(PASSES.pipeline("cu").backend, "nv/sm90")
            self.assertEqual(sorted(registry._passes), ["nv/sm90"],
                             "the alias must never own an alias-keyed stage list")


if __name__ == "__main__":
    unittest.main()
