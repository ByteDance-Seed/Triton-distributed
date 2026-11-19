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
"""CPU spec/render tests for the opt-in singleton-tuple return contract.

Some PyTorch reference models return a one-element tuple ``(y,)`` rather
than the bare tensor. Setting ``"wrap_single_as_tuple": true`` on a
single-output export makes ``_render_model`` emit ``return (call,)`` (or
``return (value,)`` on the aliased path) so the generated Model.forward
preserves that exact structure. Covers spec validation plus an end-to-end
run of the generated ``model_new.py`` source against a stub
``cuda_extension`` (no GPU, no kernel compile).
"""

import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from little_kernel.backends.nv.common.integrations import vibe_cuda_eval as vibe


def _base_spec(**updates):
    spec = {
        "kernel": "probe_kernel",
        "function": "probe",
        "inputs": [{"name": "x", "dtype": "float32", "rank": 2}],
        "output": {"name": "y", "dtype": "float32", "shape": ["x[0]", "x[1]"]},
        "scalars": [],
        "launch": {"grid": ["ceil_div(x[1], 256)", "x[0]", 1], "block": [256, 1, 1]},
    }
    spec.update(updates)
    if "outputs" in updates:
        spec.pop("output", None)
    return spec


def _load(spec_dict):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "export.json"
        path.write_text(json.dumps(spec_dict))
        return vibe._load_spec(path)


def _run_generated_model(model_src: str, x: torch.Tensor):
    """Execute the generated model_new.py against a stub cuda_extension."""
    sentinel = torch.empty_like(x)
    stub = types.ModuleType("cuda_extension")
    calls = []

    def agent_probe(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    stub.agent_probe = agent_probe
    sys.modules["cuda_extension"] = stub
    try:
        namespace = {"torch": torch}
        exec(compile(model_src, "model_new.py", "exec"), namespace)
        model = namespace["Model"]()
        result = model(x)
    finally:
        del sys.modules["cuda_extension"]
    return result, sentinel, calls


def test_wrap_single_as_tuple_renders_tuple_return():
    loaded = _load(_base_spec(wrap_single_as_tuple=True))
    model = vibe._render_model(loaded)
    assert "        return (cuda_extension.agent_probe(x),)\n" in model
    # Generated module must be valid Python and callable: a 1-tuple comes back.
    result, sentinel, calls = _run_generated_model(model, torch.zeros(2, 3))
    assert isinstance(result, tuple) and len(result) == 1
    assert result[0] is sentinel
    assert len(calls) == 1


def test_default_single_output_stays_bare_tensor():
    loaded = _load(_base_spec())
    assert loaded["_wrap_single_as_tuple"] is False
    model = vibe._render_model(loaded)
    assert "        return cuda_extension.agent_probe(x)\n" in model
    result, sentinel, _ = _run_generated_model(model, torch.zeros(2, 3))
    assert result is sentinel  # unchanged legacy contract


def test_wrap_single_as_tuple_aliased_output():
    spec = _base_spec(
        inputs=[{"name": "x", "dtype": "float32", "rank": 1}],
        outputs=[{"name": "y", "dtype": "float32", "alias_of": "x"}],
        launch={"grid": ["ceil_div(x[0], 256)", 1, 1], "block": [256, 1, 1]},
        wrap_single_as_tuple=True,
    )
    loaded = _load(spec)
    model = vibe._render_model(loaded)
    # Aliased output: the caller's buffer itself is returned, tuple-wrapped.
    assert "lk_outputs = cuda_extension.agent_probe(x)\n" in model
    assert "        return (x,)\n" in model


def test_wrap_single_as_tuple_rejects_illegal_specs():
    with pytest.raises(ValueError, match="must be a boolean"):
        _load(_base_spec(wrap_single_as_tuple=1))
    with pytest.raises(ValueError, match="exactly one output"):
        _load(
            _base_spec(
                outputs=[{"name": "a", "dtype": "float32", "shape": ["x[0]"]},
                         {"name": "b", "dtype": "float32", "shape": ["x[0]"]}],
                wrap_single_as_tuple=True,
            ))
    with pytest.raises(ValueError, match="cannot be combined with foreach"):
        _load(_base_spec(foreach={"inputs": ["x"]}, wrap_single_as_tuple=True))
