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
"""CPU regression for derived integer scalar references in shape expressions.

Covers the shared manifest layer ``export._spec`` + ``export._shapes``: an
output/workspace ``shape`` entry may name a previously declared non-runtime
derived integer scalar (``scalars`` entry with an ``expr``), not just tensor
dims, constructor ints and runtime ints. The generated binding defines each
derived scalar (``const int64_t name = expr;``) only *after* the output and
workspace ``at::empty`` sites, so the allocation cannot reference the
variable; the spec layer substitutes the already-validated computed C++
expression inline at the allocation site. Launch.grid and kernel-arg sites
run after the definition and keep the readable variable reference, so one
declared extent sizes both the grid and the workspace without duplicating
the inline expression at the author level.

A derived scalar joins the substitution map only when its own expression
resolves in the allocation-site environment (input tensor dims/strides,
constructor ints, runtime ints, host int-list helpers, and earlier
substitution-eligible derived scalars). One whose expression needs a later
extent — an output/workspace dim — stays out of the map: its definition
site still works, and a shape naming it keeps the historical rejection.

These tests run entirely on CPU: they validate the manifest-level
substitution and pin the emitted host binding ordering. Paired GPU
validation (the substituted allocation actually sizing a workspace used by
a launched kernel) is left to the queue.
"""

import copy
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.export import _spec
from little_kernel.backends.nv.common.export._render import render_bundle

_DRAFT = '''
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend='cuda', is_entry=True)
def pipe(sm: ll.Tensor[ll.bfloat16], out: ll.Tensor[ll.bfloat16]) -> ll.void:
    item: ll.int32 = ll.blockIdx_x()
    out[item] = sm[item]
'''


def _render(spec):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "draft.py"
        src.write_text(_DRAFT)
        return render_bundle(src, spec)


def _manifest(scalars, workspace_shape, output_shape=None):
    # _resolve mutates scalar items with derived render fields, so each call
    # fresh-copies the authored items.
    return {
        "kernel": "lk_kernel",
        "inputs": [{"name": "expert_idx", "dtype": "int32", "rank": 1}],
        "output": {"name": "out", "dtype": "int32", "shape": output_shape or ["expert_idx[0]"]},
        "scalars": copy.deepcopy(scalars),
        "workspaces": [{"name": "w", "dtype": "int32", "shape": workspace_shape}],
        "kernel_args": ["expert_idx", "out", "w"],
        "launch": {"grid": [1, 1, 1], "block": [128, 1, 1]},
    }


_N_PAIR_SCALARS = [
    {"name": "n_pair_blocks", "dtype": "int32", "expr": "min(max(device_sms() * 5, 1), max(expert_idx[0] // 1024, 1))"},
]


class TestWorkspaceShapeDerivedScalarRef(unittest.TestCase):
    """A workspace shape may name a previously declared derived int scalar."""

    def test_minimal_reproducer_resolves(self):
        spec = _spec._resolve(_manifest(_N_PAIR_SCALARS, ["n_pair_blocks * 64"]))
        self.assertEqual(
            spec["workspaces"][0]["_shape_cpp"],
            ["((lk_min(lk_max((lk_device_sms() * 5), 1), "
             "lk_max((expert_idx.size(0) / 1024), 1))) * 64)"])

    def test_output_shape_derived_scalar_ref(self):
        spec = _spec._resolve(_manifest(_N_PAIR_SCALARS, ["expert_idx[0]"], output_shape=["n_pair_blocks"]))
        self.assertEqual(spec["output"]["_shape_cpp"],
                         ["(lk_min(lk_max((lk_device_sms() * 5), 1), "
                          "lk_max((expert_idx.size(0) / 1024), 1)))"])

    def test_scalar_definition_keeps_unsubstituted_spelling(self):
        # The derived scalar's own definition site and launch.grid run after
        # the allocations; both keep the readable variable / plain validated
        # expression and are unaffected by the substitution map.
        spec = _spec._resolve(_manifest(_N_PAIR_SCALARS, ["n_pair_blocks * 64"]))
        self.assertEqual(spec["scalars"][0]["_expr_cpp"], "lk_min(lk_max((lk_device_sms() * 5), 1), "
                         "lk_max((expert_idx.size(0) / 1024), 1))")

    def test_nested_derived_refs_substitute_fully_expanded(self):
        # A later derived scalar whose expr names an earlier eligible one
        # records the *fully expanded* spelling, so a shape referencing it
        # needs no recursive lookup.
        scalars = _N_PAIR_SCALARS + [
            {"name": "flat_pairs", "dtype": "int32", "expr": "n_pair_blocks * 2"},
        ]
        spec = _spec._resolve(_manifest(scalars, ["flat_pairs + 1"]))
        self.assertEqual(
            spec["workspaces"][0]["_shape_cpp"],
            ["((((lk_min(lk_max((lk_device_sms() * 5), 1), "
             "lk_max((expert_idx.size(0) / 1024), 1))) * 2)) + 1)"])

    def test_derived_ref_over_runtime_scalar(self):
        # Runtime integer scalars stay visible inside derived expressions
        # substituted at the allocation site (they are binding parameters,
        # already in scope there).
        scalars = [
            {"name": "cap", "dtype": "int32", "runtime": True},
            {"name": "tiles", "dtype": "int32", "expr": "min(cap, expert_idx[0] // 16)"},
        ]
        spec = _spec._resolve(_manifest(scalars, ["tiles"]))
        self.assertEqual(spec["workspaces"][0]["_shape_cpp"], ["(lk_min(cap, (expert_idx.size(0) / 16)))"])

    def test_unknown_shape_name_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported shape expression"):
            _spec._resolve(_manifest([], ["n_pair_blocks * 64"]))

    def test_derived_depending_on_later_extent_stays_out_of_map(self):
        # A derived scalar whose expr names an output dim cannot substitute
        # at the allocation site (the output is not yet allocated); it stays
        # out of the map, so a workspace naming it keeps the historical
        # rejection. Its own definition site remains legal.
        scalars = [
            {"name": "echo", "dtype": "int32", "expr": "min(expert_idx[0], out[0])"},
        ]
        manifest = _manifest(scalars, ["echo"])
        with self.assertRaisesRegex(ValueError, "unsupported shape expression"):
            _spec._resolve(manifest)

    def test_ineligible_derived_name_stays_rejected_in_later_shapes(self):
        # A derived scalar whose expr references an output dim never enters
        # the substitution map (the output is not yet allocated at the
        # at::empty sites), so even a workspace declared after all scalars
        # keeps the historical rejection when naming it — while the scalar's
        # own definition site (rendered after the allocations) stays legal.
        scalars = _N_PAIR_SCALARS + [
            {"name": "echo", "dtype": "int32", "expr": "min(n_pair_blocks, out[0])"},
        ]
        with self.assertRaisesRegex(ValueError, "unsupported shape expression"):
            _spec._resolve(_manifest(scalars, ["echo * 2"]))


class TestBindingEmissionOrdering(unittest.TestCase):
    """End-to-end: the emitted host binding is coherent C++ ordering."""

    def _pair_spec(self):
        manifest = _manifest(_N_PAIR_SCALARS, ["n_pair_blocks * 64"])
        manifest["kernel"] = "pipe"
        manifest["function"] = "pipe"
        manifest["inputs"] = [{"name": "sm", "dtype": "bfloat16", "rank": 1}]
        manifest["scalars"] = [
            {
                "name": "n_pair_blocks", "dtype": "int32", "expr": "min(max(device_sms() * 5, 1), "
                "max(sm[0] // 1024, 1))"
            },
        ]
        manifest["outputs"] = [{"name": "out", "dtype": "bfloat16", "shape": ["sm[0]"]}]
        del manifest["output"]
        # The draft entry declares (sm, out) only; the workspace is a local
        # at::empty, not a kernel parameter.
        manifest["kernel_args"] = ["sm", "out"]
        manifest["launch"] = {"grid": ["n_pair_blocks", 1, 1], "block": [128, 1, 1]}
        return manifest

    def test_allocation_uses_substitution_before_scalar_definition(self):
        binding = _render(self._pair_spec()).binding
        empty_idx = binding.find("torch::Tensor w = at::empty(")
        defn_idx = binding.find("const int64_t n_pair_blocks =")
        self.assertGreater(empty_idx, 0, "workspace allocation not emitted")
        self.assertGreater(defn_idx, empty_idx, "derived scalar must be defined after the "
                           "allocation it sizes")
        # The allocation itself carries the substituted expression, and the
        # substitution must not reference the not-yet-declared variable.
        alloc_line = binding[empty_idx:binding.find("\n", empty_idx)]
        self.assertIn("lk_min(lk_max((lk_device_sms() * 5), 1)", alloc_line)
        self.assertNotIn("n_pair_blocks", alloc_line)
        # The definition site and the grid expression reference the scalar
        # by name.
        self.assertIn("lk_min(lk_max((lk_device_sms() * 5), 1)", binding[defn_idx:binding.find("\n", defn_idx)])
        self.assertIn("const int64_t lk_grid_x = n_pair_blocks;", binding)


if __name__ == "__main__":
    unittest.main()
