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
"""Spec-contract tests for the exporter ABI (no GPU required).

Covers the generic cycle-004 surface changes:
- uint8/int8 exporter dtypes (byte masks / flag arrays / i8 payloads),
- rank-0 device-scalar tensors (in-place counts, status words),
- the multi-output foreach grouped-binding contract (tuple of K-lists),
- the adapter honest missing-entry failure for a missing program case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from little_kernel.backends.nv.common.integrations.ai_compiler_eval.adapter import lower
from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_spec, _render_binding, _shape_expr


def _write_spec(tmp_path: Path, spec: dict) -> Path:
    path = tmp_path / "export.json"
    path.write_text(json.dumps(spec))
    return path


def _base_spec(**updates) -> dict:
    spec = {
        "kernel": "probe",
        "function": "probe",
        "inputs": [{"name": "src", "dtype": "float32", "rank": 1}],
        "output": {"name": "dst", "dtype": "float32", "shape": ["src[0]"]},
        "launch": {"grid": [1, 1, 1], "block": [64, 1, 1]},
        "scalars": [],
    }
    spec.update(updates)
    if "outputs" in updates:
        spec.pop("output", None)
    return spec


def test_uint8_and_int8_inputs_bind(tmp_path):
    spec = _base_spec(
        inputs=[{"name": "flags", "dtype": "uint8", "rank": 1}, {"name": "tags", "dtype": "int8", "rank": 1}],
        output={"name": "dst", "dtype": "uint8", "shape": ["flags[0]"]},
    )
    loaded = _load_spec(_write_spec(tmp_path, spec))
    binding = _render_binding(loaded)
    assert "at::kByte" in binding
    assert "at::kChar" in binding
    assert "at::empty({flags.size(0)}" in binding


def test_rank0_input_and_alias_output(tmp_path):
    spec = _base_spec(
        inputs=[{"name": "src", "dtype": "float32", "rank": 1}, {"name": "cnt", "dtype": "int64", "rank": 0}],
        outputs=[{"name": "dst", "dtype": "int64", "alias_of": "cnt"}],
    )
    loaded = _load_spec(_write_spec(tmp_path, spec))
    assert loaded["_outputs"][0]["shape"] == []
    assert loaded["_outputs"][0]["alias_of"] == "cnt"
    binding = _render_binding(loaded)
    assert "TORCH_CHECK(cnt.dim() == 0" in binding
    assert "at::empty" not in binding  # aliased outputs are never allocated


def test_rank0_fresh_output_allocates_scalar_tensor(tmp_path):
    spec = _base_spec(outputs=[{"name": "cnt_out", "dtype": "int64", "shape": []}], )
    loaded = _load_spec(_write_spec(tmp_path, spec))
    binding = _render_binding(loaded)
    assert "at::empty({}" in binding
    assert "cnt_out.numel() == 0" in binding  # rank-0 numel()==1: early return safe


def test_rank0_dimension_expression_rejected(tmp_path):
    spec = _base_spec(
        inputs=[{"name": "cnt", "dtype": "int64", "rank": 0}, {"name": "src", "dtype": "float32", "rank": 1}],
        output={"name": "dst", "dtype": "float32", "shape": ["cnt[0]"]},
    )
    with pytest.raises(ValueError, match="dimension out of range"):
        _load_spec(_write_spec(tmp_path, spec))


def test_negative_rank_rejected(tmp_path):
    spec = _base_spec(inputs=[{"name": "src", "dtype": "float32", "rank": -1}])
    with pytest.raises(ValueError, match="nonnegative integer"):
        _load_spec(_write_spec(tmp_path, spec))


def test_modulo_in_scalar_and_launch_grid_expressions(tmp_path):
    # ir.shape_expr.modulo: the Python % operator must lower to C++ % in
    # output shapes, declared scalars and launch grid expressions, with the
    # same integer-typing rule as // (int64 host arithmetic). This is the
    # wave-padding pattern: ceil rows up to a multiple of a wave size and
    # launch one extra block when rows is not divisible by it.
    spec = _base_spec(
        inputs=[{"name": "x", "dtype": "int32", "rank": 1}],
        output={"name": "y", "dtype": "int32", "shape": ["x[0] + (264 - x[0] % 264) % 264"]},
        launch={"grid": ["x[0] // 264 + select(x[0] % 264 == 0, 0, 1)", 1, 1], "block": [64, 1, 1]},
        scalars=[{"name": "rem", "expr": "x[0] % 264"}],
    )
    loaded = _load_spec(_write_spec(tmp_path, spec))
    binding = _render_binding(loaded)
    assert "(x.size(0) % 264)" in binding
    assert "const int64_t rem = (x.size(0) % 264);" in binding
    assert "% 264) == 0" in loaded["launch"]["_grid_cpp"][0]
    assert "lk_grid_x" in binding


def test_modulo_requires_integer_operands():
    # % shares the integer-typing rule of // but C % has no floating-point
    # form: any operand subtree that can evaluate to a float (a float literal
    # admitted under allow_float, or a / or sqrt subtree) would emit
    # uncompilable C++, so it must be rejected at spec load.
    ranks = {"x": 1}
    for bad in ("x[0] % 2.5", "2.5 % x[0]", "(x[0] / 2) % 3", "x[0] % sqrt(4.0)", "(x[0] + 1.0) % 2"):
        with pytest.raises(ValueError, match="% requires integer operands"):
            _shape_expr(bad, ranks, set(), allow_float=True)
    # Integer expressions still pass under allow_float (int literals, sizes).
    assert _shape_expr("(x[0] + 1) % 3", ranks, set(), allow_float=True) == \
        "((x.size(0) + 1) % 3)"
    # The % operand check fires before the allow_float constant check, so a
    # float literal is rejected with the same message without allow_float.
    with pytest.raises(ValueError, match="% requires integer operands"):
        _shape_expr("x[0] % 2.5", ranks, set())


def test_unary_minus_shape_expression():
    # ir.export.shape_expr_unary_minus: unary minus/plus lower through
    # ast.UnaryOp instead of dying with "unsupported shape expression". The
    # recursive emit keeps the same allow_float/scalar/dim resolution as
    # other expressions, so -1.5 needs allow_float just like 1.5 does.
    ranks = {"x": 1}
    assert _shape_expr("-1.5", {}, set(), allow_float=True) == "(-(1.5))"
    assert _shape_expr("+1.5", {}, set(), allow_float=True) == "(+(1.5))"
    assert _shape_expr("-3", ranks, set()) == "(-(3))"
    assert _shape_expr("-k", {}, {"k"}) == "(-(k))"
    assert _shape_expr("-(x[0] + 2)", ranks, set()) == "(-((x.size(0) + 2)))"
    with pytest.raises(ValueError, match="unsupported shape expression"):
        _shape_expr("-1.5", {}, set())
    # A signed float operand stays visible to the % integer-operand rule.
    with pytest.raises(ValueError, match="% requires integer operands"):
        _shape_expr("x[0] % -2.5", ranks, set(), allow_float=True)


def test_foreach_multi_output_alias_contract(tmp_path):
    inputs = [{"name": "xs", "dtype": "float32", "rank": 1}, {"name": "ys", "dtype": "float32", "rank": 1}]
    outputs = [{"name": "out_xs", "dtype": "float32", "alias_of": "xs"},
               {"name": "out_ys", "dtype": "float32", "alias_of": "ys"}]
    spec = _base_spec(inputs=inputs, outputs=outputs, foreach={"inputs": ["xs", "ys"]})
    loaded = _load_spec(_write_spec(tmp_path, spec))
    binding = _render_binding(loaded)
    assert "std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>" in binding
    assert "std::get<1>(lk_results).push_back" in binding
    with pytest.raises(ValueError, match="distinct foreach input"):
        bad = _base_spec(
            inputs=inputs, outputs=[{"name": "out_xs", "dtype": "float32", "alias_of": "xs"},
                                    {"name": "again", "dtype": "float32", "alias_of": "xs"}],
            foreach={"inputs": ["xs", "ys"]})
        _load_spec(_write_spec(tmp_path, bad))


def _tensor_list_spec(**updates) -> dict:
    spec = _base_spec(
        inputs=[{"name": "src", "dtype": "float16", "rank": 2}],
        output={"name": "dst", "dtype": "float16", "shape": ["src[0]", "src[1]"]},
        tensor_lists=[
            {
                "name": "logits", "dtype": "float16", "rank": 1, "max_len": 8, "kernel_params":
                ["lg0", "lg1", "lg2", "lg3", "lg4", "lg5", "lg6", "lg7"]
            },
            {"name": "ids", "dtype": "int32", "rank": 1, "max_len": 4, "kernel_params": ["id0", "id1", "id2", "id3"]},
        ],
        kernel_args=["src", "dst", "lg0", "lg1", "lg2", "lg3", "lg4", "lg5", "lg6", "lg7", "id0", "id1", "id2", "id3"],
    )
    spec.update(updates)
    return spec


def test_tensor_lists_binding_contract(tmp_path):
    loaded = _load_spec(_write_spec(tmp_path, _tensor_list_spec()))
    assert loaded["_list_param_types"] == {
        "lg0": "float16",
        "lg1": "float16",
        "lg2": "float16",
        "lg3": "float16",
        "lg4": "float16",
        "lg5": "float16",
        "lg6": "float16",
        "lg7": "float16",
        "id0": "int32",
        "id1": "int32",
        "id2": "int32",
        "id3": "int32",
    }
    binding = _render_binding(loaded)
    # pybind takes the two lists in declaration order after the plain inputs.
    assert "std::vector<torch::Tensor> logits" in binding
    assert ("static torch::Tensor agent_probe(torch::Tensor src, "
            "std::vector<torch::Tensor> logits, std::vector<torch::Tensor> ids)") in binding
    # Per-list length bounds and element contract checks.
    assert "logits.size() >= 1 && logits.size() <= 8" in binding
    assert "ids.size() >= 1 && ids.size() <= 4" in binding
    assert "lk_el_logits.scalar_type() == at::kHalf" in binding
    assert "lk_el_ids.dim() == 1" in binding
    # Flattened raw-pointer kernel params, nullptr-padded past the list length.
    assert "void* lg7 = (logits.size() > static_cast<size_t>(7)) ? logits[7].data_ptr() : nullptr;" in binding
    assert "void* id0 = (ids.size() > static_cast<size_t>(0)) ? ids[0].data_ptr() : nullptr;" in binding
    # The launcher is invoked with the flattened pointers, not .data_ptr() chains.
    launch_tail = binding.split("agent_lk_launch_probe(")[-1]
    assert "lg0," in launch_tail and "lg0.data_ptr()" not in launch_tail
    cuda = None  # launcher decl is rendered by _render_cuda; check arg types here
    from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _launcher_arg_decl, _cuda_kernel_arg
    assert _launcher_arg_decl(loaded, "lg0") == "void* lg0"
    assert _cuda_kernel_arg(loaded, "lg0") == "reinterpret_cast<__half*>(lg0)"
    assert _cuda_kernel_arg(loaded, "id1") == "reinterpret_cast<int32_t*>(id1)"
    del cuda


def test_tensor_lists_forward_args_and_model(tmp_path):
    from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _render_model
    spec = _tensor_list_spec(scalars=[{"name": "n", "expr": "src[0]"}], )
    spec["forward_args"] = ["logits", "src", "ids"]
    loaded = _load_spec(_write_spec(tmp_path, spec))
    model = _render_model(loaded)
    assert "def forward(self, logits, src, ids):" in model
    # Extension is still called in canonical (binding declaration) order.
    assert "cuda_extension.agent_probe(src, logits, ids)" in model


def test_tensor_lists_reject_illegal_specs(tmp_path):
    with pytest.raises(ValueError, match="distinct identifiers"):
        bad = _tensor_list_spec()
        bad["tensor_lists"][0]["kernel_params"] = bad["tensor_lists"][0]["kernel_params"][:-1]
        _load_spec(_write_spec(tmp_path, bad))
    with pytest.raises(ValueError, match="kernel param collides"):
        bad = _tensor_list_spec()
        bad["tensor_lists"][0]["kernel_params"][0] = "src"
        _load_spec(_write_spec(tmp_path, bad))
    with pytest.raises(ValueError, match="duplicate tensor_list name"):
        bad = _tensor_list_spec()
        bad["tensor_lists"][1] = dict(bad["tensor_lists"][0])
        _load_spec(_write_spec(tmp_path, bad))
    with pytest.raises(ValueError, match="foreach cannot be combined"):
        bad = _tensor_list_spec(foreach={"inputs": ["src"]})
        _load_spec(_write_spec(tmp_path, bad))
    with pytest.raises(ValueError, match="kernel_args"):
        bad = _tensor_list_spec()
        bad["kernel_args"] = ["logits"]  # the list object itself is not a kernel arg
        _load_spec(_write_spec(tmp_path, bad))
    with pytest.raises(ValueError, match="invalid tensor dimension"):
        bad = _tensor_list_spec()
        bad["scalars"] = [{"name": "v", "expr": "logits[0]"}]
        _load_spec(_write_spec(tmp_path, bad))


def test_adapter_missing_program_is_honest_unsupported(tmp_path):
    # Source policy: a case without an authored LittleKernel entry must fail
    # honestly (NotImplementedError). Reference-model delegation, eager-oracle
    # execution and no-op CUDA placeholders labelled as compiled are banned.
    task = tmp_path / "task"
    program = tmp_path / "program"
    output = tmp_path / "out"
    task.mkdir()
    program.mkdir()
    (task / "model.py").write_text("# task reference placeholder\n")
    (task / "test_cases.py").write_text("test_cases = []\n")
    with pytest.raises(NotImplementedError, match="no LittleKernel draft.py"):
        lower(task, program, output, 0)
    # A missing case must never delegate to the task reference model or
    # emit a placeholder candidate.
    assert not (output / "model_new.py").exists()


def test_adapter_requires_task_files(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "model.py").write_text("# placeholder\n")
    with pytest.raises(ValueError, match="model.py and test_cases.py"):
        lower(task, tmp_path / "program", tmp_path / "out", 0)


def test_int_literal_scalar_expr_accepted(tmp_path):
    # Merge repair (round-8, task 1151 gpu-06 case manifests): _shape_expr's
    # contract admits a bare nonnegative int as a constant expression, and
    # the present() detection walk must tolerate it — a literal cannot name
    # an optional input, so the ABI classification is unchanged.
    spec = _base_spec(
        inputs=[{"name": "src", "dtype": "float32", "rank": 1},
                {"name": "aux", "dtype": "float32", "rank": 1, "optional": True}],
        scalars=[{"name": "TILE", "expr": 8}, {"name": "HAS_AUX", "expr": "present(aux)"}],
        output={"name": "dst", "dtype": "float32", "shape": ["src[0]"]},
    )
    loaded = _load_spec(_write_spec(tmp_path, spec))
    cpp = {item["name"]: item["_expr_cpp"] for item in loaded["scalars"]}
    assert cpp["TILE"] == "8"
    assert loaded["_present_inputs"] == {"aux"}


def test_int_literal_scalar_expr_perturbed_and_negative_rejected(tmp_path):
    # A different literal value and no optional inputs: still a constant,
    # no present() classification. Fallback/legality: negative constants
    # remain rejected, matching _shape_expr's shape-constant rule.
    spec = _base_spec(scalars=[{"name": "K", "expr": 1}])
    loaded = _load_spec(_write_spec(tmp_path, spec))
    assert loaded["scalars"][0]["_expr_cpp"] == "1"
    assert loaded["_present_inputs"] == set()
    with pytest.raises(ValueError, match="nonnegative"):
        _load_spec(_write_spec(tmp_path, _base_spec(scalars=[{"name": "K", "expr": -2}])))


def test_negative_float_literal_scalar_expr_accepted(tmp_path):
    # ir.export.shape_expr_unary_minus end-to-end: a float32 constructor
    # scalar whose expr is a negative literal string ("-1.5") previously
    # failed spec load with "unsupported shape expression" because
    # _shape_expr.emit had no ast.UnaryOp branch.
    spec = _base_spec(scalars=[{"name": "k", "expr": "-1.5", "dtype": "float32"}])
    loaded = _load_spec(_write_spec(tmp_path, spec))
    assert loaded["scalars"][0]["_expr_cpp"] == "(-(1.5))"
    binding = _render_binding(loaded)
    assert "const float k = static_cast<float>((-(1.5)));" in binding
