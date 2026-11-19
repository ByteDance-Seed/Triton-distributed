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
"""Bare-TmaDescriptor entry-parameter shorthand (semantic key
ir.tma_descriptor.entry_param_type).

CUDA requires a ``__global__`` CUtensorMap parameter to be declared
``const __grid_constant__``; writing the explicit
``ll.const[ll.grid_constant[ll.TmaDescriptor]]`` spelling at every entry
parameter is noise the compiler can resolve by itself. The bare
``ll.TmaDescriptor`` annotation is the documented shorthand for the
by-value grid-constant singleton transport:

  * ``normalize_entry_grid_constant_param`` (nv/common/core/types.py)
    resolves bare ``TmaDescriptor`` to
    ``const[grid_constant[TmaDescriptor]]`` and moves a ``const[...]``
    wrapper inside the grid-constant marker (never a second const);
    every other type -- including the already-explicit spelling and the
    device-array ``ptr[TmaDescriptor]`` bundle transport -- is returned
    unchanged;
  * the CUDA entry-signature emitter (nv/common/codegen/emitter.py)
    applies the normalization to entry parameters only, so all three
    spellings emit the CUDA-legal
    ``const __grid_constant__ TmaDescriptor dA`` while non-entry
    (device) functions keep the annotation exactly as written;
  * the export manifest validator
    (_frontend.py ``_descriptor_param_kind``) accepts the bare shorthand
    as the by-value ``grid_constant`` transport, so the emitter, the
    launcher and the host binding agree on the ONE transport spelling.

These CPU regressions pin the whole path without nvcc/GPU: type-level
normalization, export-side classification, byte-identical entry
signatures across the shorthand spellings, device-function annotation
preservation, and the by-value launcher/binding ABI.
"""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.backends.nv.common.core.types import (
    TmaDescriptor,
    TmaDescriptorType,
    normalize_entry_grid_constant_param,
)
from little_kernel.backends.nv.common.export._frontend import (
    _descriptor_param_kind, )
from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import (
    _load_kernel,
    _load_spec,
    _render_binding,
    _render_cuda,
)
from little_kernel.core.passes import PASSES
from little_kernel.core.type_system import Const, GridConstant, Pointer


def _load_fixtures():
    path = os.path.join(os.path.dirname(__file__), "tma_descriptor_entry_fixtures.py")
    spec = importlib.util.spec_from_file_location("tma_descriptor_entry_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__dict__


FIXTURES = _load_fixtures()


def _compile_fixture(fn_name):
    return FIXTURES[fn_name].compile(passes=PASSES["cuda"], codegen_func=codegen_cuda)


def _entry_signature(cuda_src, kernel_name):
    start = cuda_src.find("extern \"C\" __global__")
    assert start >= 0, "no entry kernel signature in generated source"
    end = cuda_src.find(") {", start)
    return cuda_src[start:end + 1].replace(kernel_name, "K")


DRAFT_BARE = """
import little_kernel as lk
import little_kernel.language as ll

BM = 8
BN = 64


@lk.ll_kernel(backend='cuda', is_entry=True)
def tma_bf16_tile_copy(dA: {annotation},
                       out: ll.Tensor[ll.bfloat16]) -> ll.void:
    tile = ll.alloc_shared_memory("s_tile", ll.bfloat16, BM * BN)
    bar = ll.alloc_shared_memory("s_bar", ll.int64, 1)
    ll.init_smem_barrier(bar, 1)
    ll.mbarrier_arrive_and_expect_tx(bar, BM * BN * 2)
    ll.tma_load_2d_1sm(dA, bar, tile, 0, 0)
    ll.mbarrier_wait(bar, 0)
    ll.__syncthreads()
"""


def _spec():
    return {
        "function":
        "lk_forward",
        "kernel":
        "tma_bf16_tile_copy",
        "inputs": [{"name": "A", "dtype": "bfloat16", "rank": 2}],
        "outputs": [{"name": "out", "dtype": "bfloat16", "shape": ["A[0]", "A[1]"]}],
        "scalars": [],
        "launch": {"grid": [1, 1, 1], "block": [128, 1, 1]},
        "tma_descriptors": [
            {
                "name": "dA", "tensor": "A", "rank": 2, "inner": "A[1]", "outer": "A[0]", "outer_stride": "A[1]", "box":
                [64, 8], "swizzle_bytes": 0, "oob_fill": True
            },
        ],
        "kernel_args": ["dA", "out"],
    }


def _lower(tmp_path: Path, annotation: str):
    (tmp_path / "draft.py").write_text(DRAFT_BARE.format(annotation=annotation))
    (tmp_path / "export.json").write_text(json.dumps(_spec()))
    spec = _load_spec(tmp_path / "export.json")
    kernel = _load_kernel(tmp_path / "draft.py", spec["_kernel"])
    return spec, _render_cuda(spec, kernel), _render_binding(spec)


class TestNormalizeEntryGridConstantParam(unittest.TestCase):
    """Type-level contract of the bare-shorthand normalization."""

    def test_bare_descriptor_resolves_to_const_grid_constant(self):
        resolved = normalize_entry_grid_constant_param(TmaDescriptor)
        # A plain const wrapper (not itself a grid-constant marker)...
        self.assertIsInstance(resolved, Const)
        self.assertNotIsInstance(resolved, GridConstant)
        # ...around the grid-constant marker...
        self.assertIsInstance(resolved.inner_type, GridConstant)
        # ...around the descriptor singleton.
        self.assertIsInstance(resolved.inner_type.inner_type, TmaDescriptorType)
        self.assertEqual(str(resolved), "const[grid_constant[TmaDescriptor]]")

    def test_const_wrapper_gains_grid_constant_inside(self):
        resolved = normalize_entry_grid_constant_param(Const(TmaDescriptor))
        self.assertIsInstance(resolved, Const)
        self.assertNotIsInstance(resolved, GridConstant)
        self.assertIsInstance(resolved.inner_type, GridConstant)
        # Never a doubly-nested const: the marker moves inside the const,
        # leaving exactly one plain Const wrapper around the marker.
        self.assertIsInstance(resolved.inner_type.inner_type, TmaDescriptorType)
        self.assertEqual(str(resolved), "const[grid_constant[TmaDescriptor]]")

    def test_explicit_spelling_is_unchanged(self):
        explicit = Const(GridConstant(TmaDescriptor))
        self.assertIs(normalize_entry_grid_constant_param(explicit), explicit)
        bare_grid_const = GridConstant(TmaDescriptor)
        self.assertIs(normalize_entry_grid_constant_param(bare_grid_const), bare_grid_const)

    def test_normalization_is_idempotent(self):
        once = normalize_entry_grid_constant_param(TmaDescriptor)
        self.assertIs(normalize_entry_grid_constant_param(once), once)

    def test_pointer_bundle_transport_unchanged(self):
        bundle = Pointer(TmaDescriptor)
        self.assertIs(normalize_entry_grid_constant_param(bundle), bundle)
        const_bundle = Const(Pointer(TmaDescriptor))
        self.assertIs(normalize_entry_grid_constant_param(const_bundle), const_bundle)

    def test_unrelated_types_unchanged(self):
        self.assertIs(normalize_entry_grid_constant_param(ll.int32), ll.int32)
        tensor = ll.Tensor[ll.bfloat16]
        self.assertIs(normalize_entry_grid_constant_param(tensor), tensor)


class TestDescriptorParamKind(unittest.TestCase):
    """Export validator classification of descriptor annotations."""

    def test_bare_shorthand_classifies_as_grid_constant(self):
        self.assertEqual(_descriptor_param_kind(TmaDescriptor), "grid_constant")

    def test_const_shorthand_classifies_as_grid_constant(self):
        self.assertEqual(_descriptor_param_kind(Const(TmaDescriptor)), "grid_constant")

    def test_explicit_spellings_classify_as_grid_constant(self):
        self.assertEqual(_descriptor_param_kind(GridConstant(TmaDescriptor)), "grid_constant")
        self.assertEqual(_descriptor_param_kind(Const(GridConstant(TmaDescriptor))), "grid_constant")

    def test_bundle_transport_classifies_as_pointer(self):
        self.assertEqual(_descriptor_param_kind(Pointer(TmaDescriptor)), "pointer")
        self.assertEqual(_descriptor_param_kind(Const(Pointer(TmaDescriptor))), "pointer")

    def test_illegal_annotations_rejected(self):
        self.assertIsNone(_descriptor_param_kind(None))
        self.assertIsNone(_descriptor_param_kind(ll.int32))
        self.assertIsNone(_descriptor_param_kind(ll.Tensor[ll.bfloat16]))


class TestEntrySignatureCodegen(unittest.TestCase):
    """Emitter-side resolution of the shorthand on entry parameters."""

    def test_bare_shorthand_emits_grid_constant_signature(self):
        src = _compile_fixture("bare_shorthand_entry")
        self.assertIn("using TmaDescriptor = CUtensorMap;", src)
        self.assertIn(
            "extern \"C\" __global__ void bare_shorthand_entry(\n"
            "    const __grid_constant__ TmaDescriptor dA,", src)
        # The by-value descriptor is forwarded to the TMA intrinsic by
        # address inside the body, never re-declared as a pointer param.
        self.assertNotIn("TmaDescriptor* dA", src)

    def test_const_shorthand_emits_grid_constant_signature(self):
        src = _compile_fixture("const_shorthand_entry")
        self.assertIn(
            "extern \"C\" __global__ void const_shorthand_entry(\n"
            "    const __grid_constant__ TmaDescriptor dA,", src)

    def test_all_spellings_emit_identical_signatures(self):
        bare = _entry_signature(_compile_fixture("bare_shorthand_entry"), "bare_shorthand_entry")
        const = _entry_signature(_compile_fixture("const_shorthand_entry"), "const_shorthand_entry")
        explicit = _entry_signature(_compile_fixture("explicit_grid_const_entry"), "explicit_grid_const_entry")
        self.assertEqual(bare, const)
        self.assertEqual(const, explicit)
        self.assertIn("const __grid_constant__ TmaDescriptor dA", bare)

    def test_non_entry_function_keeps_annotation_as_written(self):
        src = _compile_fixture("device_caller_entry")
        # The entry parameter resolves to the grid-constant transport...
        self.assertIn(
            "extern \"C\" __global__ void device_caller_entry(\n"
            "    const __grid_constant__ TmaDescriptor dA,", src)
        # ...while the device helper's bare annotation is NOT normalized:
        # it renders as a plain by-value descriptor parameter.
        self.assertIn("static __device__ __forceinline__ void tma_load_helper(\n"
                      "    TmaDescriptor d,", src)
        helper_start = src.find("void tma_load_helper(")
        self.assertNotIn("__grid_constant__", src[helper_start:src.find(") {", helper_start)])


class TestExportIntegration(unittest.TestCase):
    """Full export path: validator, emitter, launcher and binding agree."""

    def test_bare_shorthand_accepted_with_by_value_transport(self):
        with tempfile.TemporaryDirectory() as td:
            _, cuda, binding = _lower(Path(td), "ll.TmaDescriptor")
        self.assertIn(
            "extern \"C\" __global__ void agent_tma_bf16_tile_copy(\n"
            "    const __grid_constant__ TmaDescriptor dA,", cuda)
        # The launcher declares the descriptor by value and the binding
        # host-encodes and forwards it exactly as for the explicit spelling.
        self.assertIn("extern \"C\" cudaError_t agent_lk_launch_lk_forward(\n"
                      "    CUtensorMap dA,", cuda)
        self.assertNotIn("void* dA", cuda)
        self.assertNotIn(">(dA)", cuda)
        self.assertIn("agent_lk_launch_lk_forward(CUtensorMap dA,", binding)

    def test_const_shorthand_accepted_with_by_value_transport(self):
        with tempfile.TemporaryDirectory() as td:
            _, cuda, _ = _lower(Path(td), "ll.const[ll.TmaDescriptor]")
        self.assertIn(
            "extern \"C\" __global__ void agent_tma_bf16_tile_copy(\n"
            "    const __grid_constant__ TmaDescriptor dA,", cuda)

    def test_shorthand_and_explicit_spellings_render_identical_launcher(self):
        with tempfile.TemporaryDirectory() as td:
            _, cuda_bare, binding_bare = _lower(Path(td), "ll.TmaDescriptor")
        with tempfile.TemporaryDirectory() as td:
            _, cuda_explicit, binding_explicit = _lower(Path(td), "ll.const[ll.grid_constant[ll.TmaDescriptor]]")
        self.assertEqual(cuda_bare, cuda_explicit)
        self.assertEqual(binding_bare, binding_explicit)

    def test_singleton_still_rejects_pointer_transport(self):
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td)
            (tp / "draft.py").write_text(DRAFT_BARE.format(annotation="ll.const[ll.ptr[ll.TmaDescriptor]]"))
            (tp / "export.json").write_text(json.dumps(_spec()))
            spec = _load_spec(tp / "export.json")
            kernel = _load_kernel(tp / "draft.py", spec["_kernel"])
            with self.assertRaisesRegex(ValueError, "grid_constant\\[TmaDescriptor\\]"):
                _render_cuda(spec, kernel)

    def test_singleton_still_rejects_non_descriptor_annotation(self):
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td)
            (tp / "draft.py").write_text(DRAFT_BARE.format(annotation="ll.Tensor[ll.bfloat16]"))
            (tp / "export.json").write_text(json.dumps(_spec()))
            spec = _load_spec(tp / "export.json")
            kernel = _load_kernel(tp / "draft.py", spec["_kernel"])
            with self.assertRaisesRegex(ValueError, "TMA descriptor dA"):
                _render_cuda(spec, kernel)


if __name__ == "__main__":
    unittest.main()
