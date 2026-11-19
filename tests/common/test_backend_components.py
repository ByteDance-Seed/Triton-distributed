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
"""The same contracts run in a full checkout and every selective source export."""
import ast
from pathlib import Path
import sys

import pytest

from little_kernel.backends import BackendUnavailable, available_backends, get_backend
from little_kernel.backends.interfaces import BackendSpec, BuildRequest
from little_kernel.backends.registry import backend_specs
from little_kernel.core.artifact import CompilationArtifact
from little_kernel.core.passes import PASSES
from little_kernel.core.type_system import VectorType, float32


@pytest.mark.parametrize("backend_id", available_backends())
def test_component_factory_and_explicit_target(backend_id):
    backend = get_backend(backend_id)
    assert backend.backend_id == backend_id
    assert backend.target.resolve().backend_id == backend_id
    assert backend.toolchain.source_suffix.startswith(".")
    assert callable(backend.runtime.create)
    assert len(PASSES[backend_id]) > 0


@pytest.mark.parametrize("backend_id", available_backends())
def test_source_generation_uses_its_own_components(backend_id):
    backend = get_backend(backend_id)
    tree = ast.parse("def scalar_identity(x: ll.int32) -> ll.int32:\n    return x + 1\n")
    context = {"ll": backend.language}
    for p in PASSES[backend_id]:
        tree = p(tree, context)
    code = backend.codegen.generate(tree, context, emit_header=True)
    assert "scalar_identity" in code
    assert "x + 1" in code


@pytest.mark.parametrize("backend_id", available_backends())
def test_unknown_target_is_rejected(backend_id):
    with pytest.raises(ValueError, match="unsupported architecture"):
        get_backend(backend_id).target.resolve("nonexistent_architecture")


def test_missing_backend_is_not_a_fallback():
    with pytest.raises(BackendUnavailable, match="not included"):
        get_backend("missing_vendor/missing_hardware")


def test_core_type_and_artifact_invariants_are_shared():
    vector = VectorType(float32, 4)
    assert vector is VectorType(float32, 4)
    with pytest.raises(AttributeError):
        vector.lanes = 8
    artifact = CompilationArtifact("source", "kernel", metadata={"options": [1, 2]})
    assert artifact.metadata["options"] == (1, 2)
    with pytest.raises(TypeError):
        artifact.metadata["options"] = ()


def test_descriptors_only_name_physically_available_hardware():
    root = Path(__import__("little_kernel.backends", fromlist=["__file__"]).__file__).parent
    for spec in backend_specs():
        for module in spec.modules:
            path = root.parent.parent / Path(*module.split("."))
            assert path.is_dir(), (spec.backend_id, module)
    for name, module in sys.modules.items():
        if name.startswith("little_kernel") and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(root.parent)


@pytest.mark.parametrize("factory", ["module", "module:", ":factory", "module:invalid:name"])
def test_invalid_factory_is_rejected_when_descriptor_is_constructed(factory):
    with pytest.raises(ValueError, match="module:attribute"):
        BackendSpec("vendor/target", factory)


@pytest.mark.parametrize("backend_id", available_backends())
def test_canonical_build_selects_its_own_default_target_without_device_discovery(monkeypatch, backend_id):
    from dataclasses import replace
    import little_kernel as lk
    import little_kernel.backends as backends

    backend = get_backend(backend_id)

    class CaptureBuilder:

        def build(self, request):
            return request

    monkeypatch.setattr(backends, "get_backend", lambda *a, **kw: replace(backend, builder=CaptureBuilder()))

    @lk.ll_kernel(backend=backend_id, is_entry=True)
    def program(n: int):
        pass

    request = program.build(PASSES[backend_id], None, grid=(1, 1, 1), block=(1, 1, 1))
    assert isinstance(request, BuildRequest)
    assert request.architecture == backend.target.resolve().arch


@pytest.mark.parametrize("backend_id", available_backends())
def test_foreign_backend_build_options_cannot_be_silently_ignored(backend_id):
    from types import SimpleNamespace
    backend = get_backend(backend_id)
    for name in backend.builder.unsupported_options:
        request = BuildRequest(SimpleNamespace(backend=backend_id), (), None, (1, 1, 1), (1, 1, 1), options={name: 1})
        with pytest.raises(ValueError, match=f"does not support build option {name}"):
            backend.builder.build(request)
