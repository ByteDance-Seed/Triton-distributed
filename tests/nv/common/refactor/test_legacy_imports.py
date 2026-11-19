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
"""RF17 regressions: one inferencer/resolver owner with explicit legacy shims.

Covers:
* canonical, package-public and legacy import paths all succeed without an
  import-time GPU (torch/CUDA) dependency;
* the abandoned ``infer_type_new`` and flat ``method_resolver`` modules are thin
  re-export shims pointing at the single canonical implementation (identity,
  no divergent copy);
* the canonical ``MethodResolver`` strategy order and its preserved
  ``resolve_method_return_type(var_name, method_name, var_type=None)`` contract;
* the canonical inferencer still runs end to end.
"""

import ast
import importlib.util
import pathlib
import sys
import textwrap

import pytest

from little_kernel.core.type_system import TupleType, SpecialStructType, bool_, float32, int32

CANONICAL_RESOLVER = ("little_kernel.core.passes.utils.type_inference.method_resolver")
CANONICAL_INFER = "little_kernel.core.passes.utils.type_inference"
LEGACY_INFER = "little_kernel.core.passes.utils.infer_type_new"
LEGACY_RESOLVER = "little_kernel.core.passes.utils.method_resolver"


# --------------------------------------------------------------------------- #
# Import smoke: no broken modules, no import-time GPU dependency
# --------------------------------------------------------------------------- #
def test_canonical_and_public_imports_resolve():
    from little_kernel.core.passes.utils.type_inference import TypeInferencer as PackageInferencer, infer_type as package_infer_type
    from little_kernel.core.passes.utils.type_inference.infer_type import TypeInferencer, infer_type
    from little_kernel.core.passes.utils.type_inference.method_resolver import MethodResolver

    assert PackageInferencer is TypeInferencer
    assert package_infer_type is infer_type
    assert callable(MethodResolver)
    assert callable(infer_type)


def test_legacy_modules_import_where_they_previously_crashed():
    # infer_type_new raised ModuleNotFoundError before RF17 because it imported
    # sibling modules that only exist inside the type_inference package.
    import little_kernel.core.passes.utils.infer_type_new as legacy_infer
    import little_kernel.core.passes.utils.method_resolver as legacy_resolver

    assert hasattr(legacy_infer, "TypeInferencer")
    assert hasattr(legacy_infer, "infer_type")
    assert hasattr(legacy_resolver, "MethodResolver")


@pytest.mark.parametrize("module_name", [
    CANONICAL_INFER,
    CANONICAL_RESOLVER,
    LEGACY_INFER,
    LEGACY_RESOLVER,
])
def test_inference_imports_do_not_pull_torch_or_cuda(module_name):
    from tests.common._optional_imports import check_cpu_import
    check_cpu_import(module_name)


def test_legacy_shims_reexport_the_single_canonical_implementation():
    from little_kernel.core.passes.utils.type_inference import TypeInferencer as CanonicalInferencer, infer_type as canonical_infer_type
    canonical_resolver_mod = __import__(CANONICAL_RESOLVER, fromlist=["MethodResolver"])
    legacy_infer = __import__(LEGACY_INFER, fromlist=["TypeInferencer", "infer_type"])
    legacy_resolver = __import__(LEGACY_RESOLVER, fromlist=["MethodResolver"])

    # Identity, not just equality: legacy paths must share one implementation.
    assert legacy_infer.TypeInferencer is CanonicalInferencer
    assert legacy_infer.infer_type is canonical_infer_type
    assert legacy_resolver.MethodResolver is canonical_resolver_mod.MethodResolver


def test_legacy_shim_files_have_no_local_implementation():
    # A compatibility shim must not silently evolve a second algorithm variant.

    root = pathlib.Path(__file__).resolve().parents[4] / "python" / "little_kernel"
    shims = [
        root / "core/passes/utils/infer_type_new.py",
        root / "core/passes/utils/method_resolver.py",
    ]
    for path in shims:
        tree = ast.parse(path.read_text())
        local_defs = [node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
        assert local_defs == [], f"{path.name} defines divergent behavior: {local_defs}"


# --------------------------------------------------------------------------- #
# Canonical inferencer smoke
# --------------------------------------------------------------------------- #
def test_canonical_infer_type_runs_end_to_end():
    from little_kernel.core.passes.utils.type_inference import infer_type

    tree = ast.parse(textwrap.dedent("""
        x: int32 = 1
        y = 2
    """))
    scope = infer_type(tree, {"int32": int32})
    assert scope["x"] is int32
    assert scope["y"] is int32


# --------------------------------------------------------------------------- #
# MethodResolver behavior
# --------------------------------------------------------------------------- #
def _load_ctx_module(tmp_path):
    """An on-disk module so inspect.getsource can read its class definitions."""
    source = textwrap.dedent("""\
        from little_kernel.core.type_system import int32

        class Scheduler:
            def annotated_method(self) -> int32:
                return 0

            def next_block(self):
                return False, -1, -1
    """)
    module_path = tmp_path / "rf17_ctx_class.py"
    module_path.write_text(source)
    spec = importlib.util.spec_from_file_location("rf17_ctx_class", module_path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so inspect does not classify the class as built-in.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[spec.name]
        raise
    return module


def _annotation_resolver(node):
    mapping = {"int32": int32, "bool_": bool_}
    if isinstance(node, ast.Name):
        return mapping.get(node.id)
    return None


def test_resolver_ctx_annotation_and_tuple_inference(tmp_path):
    from little_kernel.core.passes.utils.type_inference.method_resolver import MethodResolver

    module = _load_ctx_module(tmp_path)
    resolver = MethodResolver({"Scheduler": module.Scheduler}, {}, _annotation_resolver)

    # Explicit return annotation wins.
    annotated = resolver.resolve_method_return_type("scheduler", "annotated_method")
    assert annotated is int32

    # Unannotated (bool, int32, int32) literal tuple is inferred.
    inferred = resolver.resolve_method_return_type("scheduler", "next_block")
    assert inferred == TupleType((bool_, int32, int32))


def test_resolver_unknown_method_returns_none_without_fallback(tmp_path):
    from little_kernel.core.passes.utils.type_inference.method_resolver import MethodResolver

    module = _load_ctx_module(tmp_path)
    resolver = MethodResolver({"Scheduler": module.Scheduler}, {}, _annotation_resolver)

    # An unresolvable method is legitimately unresolved (None lets the caller
    # continue its own strategies); it must not be guessed into a wrong type.
    assert resolver.resolve_method_return_type("scheduler", "no_such_method") is None


def test_resolver_special_struct_type_consults_struct_stub_registry(monkeypatch):
    from little_kernel.core.passes.utils.type_inference.method_resolver import MethodResolver
    from little_kernel.language.intrin import struct_stub as stub_registry

    stub_info = stub_registry.StructStubInfo(
        cpp_struct_name="RF17Stub",
        includes=['"rf17_stub.h"'],
        methods={
            "scalar":
            stub_registry.MethodStubInfo(python_method_name="scalar", return_type=float32),
            "packed":
            stub_registry.MethodStubInfo(
                python_method_name="packed",
                is_tuple_return=True,
                tuple_return_types=[bool_, int32],
            ),
        },
    )
    monkeypatch.setitem(stub_registry._struct_stub_registry, "RF17Stub", stub_info)

    resolver = MethodResolver({}, {}, _annotation_resolver)
    var_type = SpecialStructType("RF17Stub")

    assert resolver.resolve_method_return_type("obj", "scalar", var_type) is float32
    assert resolver.resolve_method_return_type("obj", "packed", var_type) == TupleType((bool_, int32))

    # Strategy 2: the struct name itself may be passed without a var_type.
    assert resolver.resolve_method_return_type("RF17Stub", "scalar") is float32


def test_legacy_resolver_uses_canonical_strategies(tmp_path):
    """The old flat import path must resolve exactly like the canonical one."""
    from little_kernel.core.passes.utils.method_resolver import MethodResolver

    module = _load_ctx_module(tmp_path)
    resolver = MethodResolver({"Scheduler": module.Scheduler}, {}, _annotation_resolver)

    # Preserved signature: var_type is the trailing optional keyword.
    assert resolver.resolve_method_return_type("scheduler", "annotated_method", var_type=None) is int32
    assert resolver.resolve_method_return_type("scheduler", "next_block") == TupleType((bool_, int32, int32))
