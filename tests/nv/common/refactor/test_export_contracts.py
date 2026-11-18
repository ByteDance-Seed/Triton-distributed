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
"""RF09 generic ``little_kernel.backends.nv.common.export`` contract tests (CPU only, no GPU).

These regressions pin the split export layer itself:

* the public typed raw/validated API (``RawSpec``/``ValidatedSpec`` and
  ``load_spec``/``validate``) is immutable and never mutates the authored dict;
* one schema owns every allowlist: unknown top-level and record keys fail
  instead of being silently ignored;
* shape expressions, launch/alias/constexpr/foreach failure contracts survive
  the move;
* a minimal authored DSL program is exported through the standalone
  ``little_kernel.backends.nv.common.export`` API without importing any evaluation adapter;
* the legacy private re-exports remain thin aliases of the canonical code;
* the ai_compiler_eval adapter depends on the public functions only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import little_kernel.backends.nv.common.export as export_api
from little_kernel.backends.nv.common.export import RawSpec, SchemaError, ValidatedSpec, check_identifier, load_spec, validate
from little_kernel.backends.nv.common.integrations import vibe_cuda_eval as legacy


def _base_spec(**updates) -> dict:
    spec = {
        "kernel": "vec_add",
        "function": "vec_add",
        "inputs": [
            {"name": "a", "dtype": "float32", "rank": 1},
            {"name": "b", "dtype": "float32", "rank": 1},
        ],
        "output": {"name": "out", "dtype": "float32", "shape": ["a[0]"]},
        "launch": {"grid": ["ceil_div(a[0], 256)", 1, 1], "block": [256, 1, 1]},
        "scalars": [],
    }
    spec.update(updates)
    return spec


def test_derived_constexpr_facts_are_deeply_immutable():
    import runpy
    fixture = runpy.run_path(str(Path(__file__).parents[1] / "unit/test_derived_constexpr_constructor.py"))
    resolved = validate(fixture["SPEC"])
    with pytest.raises(TypeError):
        resolved.facts.constexpr.derived[0]["name"] = "other"
    mapping = resolved.legacy_mapping()
    assert mapping["_constexpr_derived"][0]["name"] == "has_mom"


DRAFT_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def vec_add(a: ll.Tensor[ll.float32], b: ll.Tensor[ll.float32],
            out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < 1024:
        out[i] = a[i] + b[i]
'''

# ---------------------------------------------------------------------------
# Typed raw/validated/resolved specifications
# ---------------------------------------------------------------------------


def test_raw_spec_is_unvalidated_and_typed_specs_are_immutable():
    raw_obj = _base_spec()
    raw = RawSpec.from_object(raw_obj)
    assert raw.data is raw_obj
    resolved = validate(raw)
    # Authored and resolved layers are distinct immutable owners.
    assert isinstance(resolved, export_api.ResolvedSpec)
    assert isinstance(resolved.authored, ValidatedSpec)
    assert isinstance(resolved.authored.inputs[0], export_api.TensorInput)
    assert resolved.kernel == "vec_add"
    assert resolved.function == "vec_add"
    assert resolved.ranks == {"a": 1, "b": 1, "out": 1}
    assert resolved.tensor_dtypes["out"] == "float32"
    assert resolved.kernel_args == ("a", "b", "out")
    assert resolved.runtime_kernel_args == ("a", "b", "out")
    assert resolved.foreach_inputs == ()
    # The launch contract is a typed immutable owner, not a subscriptable dict.
    assert isinstance(resolved.launch, export_api.LaunchPlan)
    assert resolved.launch.grid == ("ceil_div(a[0], 256)", 1, 1)
    assert resolved.launch.block == (256, 1, 1)
    # Resolved accessors are read-only mappings.
    with pytest.raises(TypeError):
        resolved.ranks["a"] = 4
    with pytest.raises(TypeError):
        resolved.authored.inputs[0].item["name"] = "x"
    # The authored raw object is never mutated with derived fields.
    assert not any(key.startswith("_") for key in raw_obj)


def test_deep_immutability_nested_containers_and_caller_mutation():
    authored = _base_spec()
    resolved = validate(authored)
    # The caller can mutate its own dict afterwards; the validated model is
    # isolated (validation deep-copied the input).
    authored["launch"]["block"] = [1, 1, 1]
    authored["inputs"][0]["name"] = "hijacked"
    assert resolved.launch.block == (256, 1, 1)
    assert resolved.authored.input_names[0] == "a"
    # Nested containers cannot be mutated through the typed surface.
    with pytest.raises(TypeError):
        resolved.launch.grid[0] = 9
    with pytest.raises((TypeError, AttributeError)):
        resolved.launch.grid_cpp[0] = "x"
    with pytest.raises((TypeError, AttributeError)):
        resolved.authored.outputs[0].shape_cpp[0] = "x"


def test_validation_is_idempotent_on_resolved_spec():
    first = validate(_base_spec())
    second = load_spec(first)
    assert second is first  # already-validated models are reused, not re-parsed


def test_direct_validated_construction_requires_typed_records():
    # A caller cannot fabricate validated state from arbitrary fields.
    with pytest.raises(TypeError):
        ValidatedSpec(kernel="k", function="k", inputs=[{"name": "a"}], outputs=(), constructor=(), scalars=())


def test_validated_identifier_cannot_be_replaced_with_invalid_name():
    from dataclasses import replace

    resolved = validate(_base_spec())
    with pytest.raises(ValueError, match="identifier"):
        replace(resolved.authored, kernel="0invalid")
    with pytest.raises(TypeError, match="validate"):
        replace(resolved.authored, inputs=(export_api.TensorInput({"name": "a", "dtype": "invalid", "rank": True}), ))
    with pytest.raises(TypeError, match="validation"):
        export_api.ResolvedSpec(resolved.authored, resolved.facts, {"kernel": "unvalidated"})


def test_loaded_entry_render_options_do_not_leak_across_calls(tmp_path):
    from little_kernel.backends.nv.common.export._frontend import _load_kernel

    source = tmp_path / "program.py"
    source.write_text(DRAFT_SOURCE)
    kernel = _load_kernel(source, "vec_add")
    before = dict(kernel.ctx)
    occupied = _base_spec()
    occupied["launch"]["max_blocks_per_sm"] = 4
    export_api.render_bundle(kernel, occupied)
    assert kernel.ctx == before
    reused = export_api.render_bundle(kernel, _base_spec())
    fresh = export_api.render_bundle(source, _base_spec())
    assert reused.cuda == fresh.cuda
    with pytest.raises(ValueError, match="does not match spec"):
        export_api.render_bundle(kernel, _base_spec(kernel="other"))


def test_validate_from_mapping_and_path_equivalent(tmp_path):
    spec = _base_spec()
    path = tmp_path / "export.json"
    path.write_text(json.dumps(spec))
    assert load_spec(path).ranks == validate(spec).ranks


def test_legacy_mapping_view_is_isolated_per_call():
    validated = validate(_base_spec())
    view1 = validated.legacy_mapping()
    view1["_mutated_by_renderer"] = True
    view1["inputs"].append({"name": "injected"})
    view2 = validated.legacy_mapping()
    assert "_mutated_by_renderer" not in view2
    assert len(view2["inputs"]) == 2
    # Renderer-produced state never leaks into the typed model.
    assert "_mutated_by_renderer" not in validated.kernel_args


def test_non_object_spec_rejected():
    with pytest.raises(ValueError):
        validate([])
    with pytest.raises(ValueError):
        RawSpec.from_object([1, 2])


def test_check_identifier_rule_is_public():
    assert check_identifier("ok_name") == "ok_name"
    with pytest.raises(ValueError):
        check_identifier("1bad", "x")
    with pytest.raises(ValueError):
        check_identifier("for", "keyword")


# ---------------------------------------------------------------------------
# Single schema ownership: unknown keys are rejected, never ignored
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_rejected():
    spec = _base_spec(blockdims=[1, 1, 1])  # typo for launch.block
    with pytest.raises(SchemaError, match="unknown export spec fields"):
        validate(spec)
    assert issubclass(SchemaError, ValueError)


@pytest.mark.parametrize("record,key", [
    ({"inputs": [{"name": "a", "dtype": "float32", "rank": 1, "contigous": True}]}, "input"),
    ({"scalars": [{"name": "s", "dtype": "int32", "expr": "a[0]", "runtime": False, "extra": 1}]}, "scalar"),
    ({"launch": {"grid": [1, 1, 1], "block": [1, 1, 1], "blok_cap": [1, 1, 1]}}, "launch"),
    ({"foreach": {"inputs": ["a"], "mode": "loop", "maxtrix": 8}}, "foreach"),
])
def test_unknown_record_field_rejected(record, key):
    spec = _base_spec(**record)
    with pytest.raises(ValueError, match=f"unknown {key} fields"):
        validate(spec)


def test_known_optional_top_level_fields_accepted():
    # Regression guard: every documented top-level spelling must stay in the
    # single owned allowlist (no accidental narrowing).
    spec = _base_spec(
        inputs=[{"name": "a", "dtype": "float32", "rank": 2}, {"name": "b", "dtype": "float32", "rank": 2}],
        no_alias=False, wrap_single_as_tuple=False, equalities=[["a[0]", "b[0]"]])
    validated = validate(spec)
    assert validated.launch.grid_cpp[0].startswith("lk_ceil_div")


# ---------------------------------------------------------------------------
# Shape-expression and launch contracts
# ---------------------------------------------------------------------------


def test_shape_expression_language_unchanged():
    from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _shape_expr
    ranks = {"x": 2}
    assert _shape_expr("min(2112, ceil_div(x[0], 256))", ranks) == ("lk_min(2112, lk_ceil_div(x.size(0), 256))")
    assert _shape_expr(3, ranks) == "3"
    with pytest.raises(ValueError, match="nonnegative"):
        _shape_expr(-1, ranks)
    with pytest.raises(ValueError, match="dimension out of range"):
        _shape_expr("x[5]", ranks)
    with pytest.raises(ValueError, match="invalid tensor dimension"):
        _shape_expr("y[0]", ranks)
    with pytest.raises(ValueError, match="% requires integer operands"):
        _shape_expr("3 % 1.5", ranks, allow_float=True)


def test_dynamic_block_requires_valid_cap():
    spec = _base_spec(launch={"grid": [1, 1, 1], "block": ["a[0]", 1, 1], "block_cap": [2048, 1, 1]})
    with pytest.raises(ValueError, match="per-axis limit"):
        validate(spec)


def test_static_block_over_1024_rejected():
    spec = _base_spec(launch={"grid": [1, 1, 1], "block": [1025, 1, 1]})
    with pytest.raises(ValueError, match="1024"):
        validate(spec)


def test_min_blocks_per_sm_hardware_limit_rejected():
    spec = _base_spec(launch={"grid": [1, 1, 1], "block": [256, 1, 1], "min_blocks_per_sm": 32})
    with pytest.raises(ValueError, match="2048"):
        validate(spec)


# ---------------------------------------------------------------------------
# Alias / constexpr / foreach failure contracts preserved
# ---------------------------------------------------------------------------


def test_alias_of_unknown_input_rejected():
    spec = _base_spec(inputs=[{"name": "a", "dtype": "float32", "rank": 1}],
                      output={"name": "out", "dtype": "float32", "alias_of": "nope"})
    with pytest.raises(ValueError, match="must name a model input"):
        validate(spec)


def test_alias_of_dtype_mismatch_rejected():
    spec = _base_spec(inputs=[{"name": "a", "dtype": "float32", "rank": 1}],
                      output={"name": "out", "dtype": "bfloat16", "alias_of": "a"})
    with pytest.raises(ValueError, match="dtype must match"):
        validate(spec)


def test_foreach_cannot_combine_with_optional():
    # The trailing optional input b is legal at the input layer; foreach must
    # then reject the combination rather than emit a mis-bound group.
    spec = _base_spec(
        inputs=[{"name": "a", "dtype": "float32", "rank": 1},
                {"name": "b", "dtype": "float32", "rank": 1, "optional": True}],
        output={"name": "out", "dtype": "float32", "shape": ["a[0]"]}, foreach={"inputs": ["a", "b"]})
    with pytest.raises(ValueError, match="foreach cannot be combined"):
        validate(spec)


def test_derived_constexpr_needs_preceding_reference():
    spec = _base_spec(inputs=[{"name": "a", "dtype": "float32", "rank":
                               1}], output={"name": "out", "dtype": "float32", "shape": ["a[0]"]},
                      constructor=[{"name": "flag", "dtype": "bool", "constexpr": "momentum != 0"}])
    with pytest.raises(ValueError, match="preceding"):
        validate(spec)


# ---------------------------------------------------------------------------
# Standalone compiler API: no evaluator adapter involved
# ---------------------------------------------------------------------------


def _build_workspace(root: Path) -> tuple[Path, Path, Path]:
    workspace = root / "workspace"
    (workspace / "task").mkdir(parents=True)
    (workspace / "agent").mkdir()
    draft = workspace / "agent" / "draft.py"
    spec_path = workspace / "agent" / "export.json"
    draft.write_text(DRAFT_SOURCE)
    spec_path.write_text(json.dumps(_base_spec()))
    return workspace, draft, spec_path


def test_standalone_export_without_adapter(tmp_path, monkeypatch):
    # Block every evaluator-adapter import: the standalone compiler API must
    # not depend on an evaluation adapter.
    for forbidden in (
            "little_kernel.backends.nv.common.integrations.ai_compiler_eval",
            "little_kernel.backends.nv.common.integrations.ai_compiler_eval.adapter",
            "little_kernel.backends.nv.common.integrations.vibe_cuda_eval",
    ):
        monkeypatch.setitem(sys.modules, forbidden, None)
    workspace, draft, spec_path = _build_workspace(tmp_path)
    paths = export_api.export(draft, spec_path, workspace, force=True)
    cuda, binding, model = paths
    assert cuda.suffix == ".cu" and binding.suffix == ".cpp"
    cuda_text = cuda.read_text()
    binding_text = binding.read_text()
    model_text = model.read_text()
    # Generated entry/launcher and alias policy are present in the CUDA unit.
    assert "__global__" in cuda_text
    assert "agent_lk_launch_vec_add" in cuda_text
    assert "float* __restrict__ a" in cuda_text
    # Binding checks inputs and allocates the declared output shape.
    assert "a rank mismatch" in binding_text
    assert "at::empty({a.size(0)}" in binding_text
    # The model file is a self-contained drop-in.
    assert "class Model(torch.nn.Module)" in model_text
    assert "cuda_extension.agent_vec_add(" in model_text
    # check() confirms a byte-identical re-render.
    again = export_api.check(draft, spec_path, workspace)
    assert [str(p) for p in again] == [str(p) for p in paths]


def test_export_fails_honestly_for_missing_frontend(tmp_path):
    workspace, _draft, spec_path = _build_workspace(tmp_path)
    missing = workspace / "agent" / "nope.py"
    with pytest.raises(FileNotFoundError):
        export_api.export(missing, spec_path, workspace, force=True)


def test_generic_render_bundle_needs_no_eval_layout(tmp_path, monkeypatch):
    # The generic API renders from a plain directory: no task/ or agent/,
    # no compile.sh, and no evaluator adapter import.
    for forbidden in (
            "little_kernel.backends.nv.common.integrations.ai_compiler_eval",
            "little_kernel.backends.nv.common.integrations.ai_compiler_eval.adapter",
            "little_kernel.backends.nv.common.integrations.vibe_cuda_eval",
    ):
        monkeypatch.setitem(sys.modules, forbidden, None)
    src = tmp_path / "kernel.py"
    src.write_text(DRAFT_SOURCE)
    resolved = validate(_base_spec())
    # Render directly from the immutable typed model (idempotent reuse).
    bundle = export_api.render_bundle(src, resolved)
    assert isinstance(bundle, export_api.ExportBundle)
    assert bundle.spec is resolved
    assert "agent_lk_launch_vec_add" in bundle.cuda
    assert "a rank mismatch" in bundle.binding
    assert "cuda_extension.agent_vec_add(" in bundle.model
    assert bundle.filenames() == ("agent_lk_vec_add.cu", "agent_lk_vec_add_binding.cpp", "model_new.py")


def test_generic_write_bundle_into_plain_directory(tmp_path):
    src = tmp_path / "kernel.py"
    src.write_text(DRAFT_SOURCE)
    out = tmp_path / "generic_out"
    bundle = export_api.write_bundle(src, _base_spec(), out)
    cuda = out / "kernels" / "agent_lk_vec_add.cu"
    binding = out / "kernels" / "agent_lk_vec_add_binding.cpp"
    model = out / "model_new.py"
    assert cuda.read_text() == bundle.cuda
    assert binding.read_text() == bundle.binding
    assert model.read_text() == bundle.model
    # No evaluator workspace policy leaked into the generic directory.
    assert not (out / "task").exists()
    assert not (out / "agent").exists()
    # overwrite guard is honest.
    with pytest.raises(FileExistsError):
        export_api.write_bundle(src, _base_spec(), out)
    export_api.write_bundle(src, _base_spec(), out, overwrite=True)


def test_render_uses_fresh_legacy_view_and_is_repeatable(tmp_path):
    src = tmp_path / "kernel.py"
    src.write_text(DRAFT_SOURCE)
    resolved = validate(_base_spec())
    first = export_api.render_bundle(src, resolved)
    # A first render cannot pollute the second through shared mutable state.
    view = resolved.legacy_mapping()
    view["launch"]["_threads"] = 1
    second = export_api.render_bundle(src, resolved)
    assert first.cuda == second.cuda
    assert first.binding == second.binding


# ---------------------------------------------------------------------------
# Thin compatibility re-exports + public-adapter dependency
# ---------------------------------------------------------------------------


def test_legacy_private_imports_are_canonical_aliases():
    from little_kernel.backends.nv.common.export._spec import _load_spec as canonical_load
    from little_kernel.backends.nv.common.export._bindings import _render_binding
    from little_kernel.backends.nv.common.export._shapes import _shape_expr
    from little_kernel.backends.nv.common.export._schema import _DTYPES, _identifier
    # Thin facade: same callables, same single schema table (no divergent copy).
    assert legacy._load_spec is canonical_load
    assert legacy._render_binding is _render_binding
    assert legacy._shape_expr is _shape_expr
    assert legacy._identifier is _identifier
    assert legacy._DTYPES is _DTYPES
    assert legacy.export is export_api.export
    # Names the pinned corpus/tests import keep working.
    for name in (
            "_DTYPES",
            "_TMA_DTYPES",
            "_load_kernel",
            "_load_spec",
            "_launcher_arg_decl",
            "_cuda_kernel_arg",
            "_kernel_param_names",
            "_declared_tensor_dtypes",
            "_render_cuda",
            "_render_binding",
            "_render_model",
            "export",
            "check",
    ):
        assert hasattr(legacy, name)


def test_legacy_load_spec_matches_public_view(tmp_path):
    path = tmp_path / "export.json"
    path.write_text(json.dumps(_base_spec()))
    legacy_view = legacy._load_spec(path)
    public_view = load_spec(path).legacy_mapping()
    # Same derived contract, underscore fields included.
    assert legacy_view["_kernel_args"] == public_view["_kernel_args"]
    assert legacy_view["launch"]["_grid_cpp"] == public_view["launch"]["_grid_cpp"]
    assert legacy_view["_function"] == public_view["_function"]


def test_adapter_depends_only_on_public_export_api():
    import inspect
    import re
    from little_kernel.backends.nv.common.integrations.ai_compiler_eval import adapter
    source = inspect.getsource(adapter)
    assert "from little_kernel.backends.nv.common.export import" in source
    assert not re.search(r"little_kernel\.integrations\.vibe_cuda_eval", source)
    assert "load_spec" in source and "export" in source
    # No task dispatch / oracle fallback strings were introduced.
    assert "best_impl" not in source
