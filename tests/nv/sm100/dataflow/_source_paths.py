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
"""Canonical authored inputs for dataflow tests, with historical filename mapping."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

# Special role-demo names from the shared M13 naming convention. These do not
# follow the generic "strip v2_ / fuse -> fused" rule.
SPECIAL_EXAMPLE_NAMES = {
    "v2_fa4.py": "fa4_attention.py",
    "v2_gdn.py": "gated_delta_net.py",
    "v2_groupgemm.py": "grouped_gemm.py",
    "v2_empty_tx_sm100.py": "transaction_probe_sm100.py",
    "v2_empty_tma_store_sm100.py": "tma_store_probe_sm100.py",
    "v2_empty_umma_sm100.py": "umma_probe_sm100.py",
}


def canonical_example_name(basename: str) -> str:
    """Map a legacy ``v2_*`` role-demo basename to its semantic example name."""
    if basename in SPECIAL_EXAMPLE_NAMES:
        return SPECIAL_EXAMPLE_NAMES[basename]
    name = basename
    if name.startswith("v2_"):
        name = name[len("v2_"):]
    name = name.replace("_fuse_", "_fused_")
    if name.endswith("_fuse.py"):
        name = name[:-len("_fuse.py")] + "_fused.py"
    return name


def canonical_bench_name(basename: str) -> str:
    """Map a legacy ``bench_v2_*`` role-benchmark basename to ``bench_*``."""
    special = {
        "bench_v2_fa4_sm100.py": "bench_fa4_attention_sm100.py",
        "bench_v2_gdn.py": "bench_gated_delta_net.py",
        "bench_v2_groupgemm.py": "bench_grouped_gemm.py",
    }
    if basename in special:
        return special[basename]
    name = basename
    if name.startswith("bench_v2_"):
        name = "bench_" + name[len("bench_v2_"):]
    name = name.replace("_fuse_", "_fused_")
    if name.endswith("_fuse.py"):
        name = name[:-len("_fuse.py")] + "_fused.py"
    return name


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` until a directory containing pyproject.toml."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("could not locate repository root (pyproject.toml)")


REPO_ROOT = find_repo_root()
PYTHON_ROOT = REPO_ROOT / "python"
EXAMPLES_DATAFLOW = REPO_ROOT / "examples/nv/sm100"


def role_demo_path(basename: str) -> Path:
    path = EXAMPLES_DATAFLOW / canonical_example_name(basename)
    if not path.is_file():
        raise FileNotFoundError(f"Missing authored dataflow example: {path}")
    return path


def read_role_demo(basename: str) -> str:
    """Read the authored DSL source for a legacy ``v2_*`` basename."""
    return role_demo_path(basename).read_text(encoding="utf-8")


def torch_available() -> bool:
    """Whether an importable Torch is present, without importing it."""
    return importlib.util.find_spec("torch") is not None


def _module_imports_torch(path: Path) -> tuple[bool, list[str]]:
    """Return ``(imports_torch, local_role_leaves)`` for one demo source."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports_torch = False
    local_leaves: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            (alias.name == "torch" or alias.name.startswith("torch.")) for alias in node.names):
            imports_torch = True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "torch" or module.startswith("torch."):
                imports_torch = True
            elif node.level or module.startswith("examples.nv.sm100"):
                local_leaves.append(module.split(".")[-1])
    return imports_torch, local_leaves


def demo_requires_torch(basename: str) -> bool:
    """True if the demo or any in-package role sibling it imports needs Torch.

    This inspects authored source only (never importing Torch or the demo),
    following ``examples.nv.sm100`` sibling imports so a demo that pulls in a
    Torch-importing helper (e.g. ``v2_a2a_gemm_sm100`` -> ``v2_a2a_tma_push``)
    is classified honestly.
    """
    memo: dict[str, bool] = {}

    def needs(name: str, active: frozenset[str]) -> bool:
        if name in memo:
            return memo[name]
        path = role_demo_path(name)
        imports_torch, leaves = _module_imports_torch(path)
        result = imports_torch
        for leaf in leaves:
            if leaf in active:
                continue
            sibling = path.parent / f"{leaf}.py"
            if not sibling.is_file():
                continue
            if needs(f"{leaf}.py", active | {leaf}):
                result = True
                break
        memo[name] = result
        return result

    return needs(basename, frozenset({basename.removesuffix('.py')}))


def skip_if_demo_requires_torch(basename: str) -> None:
    """Skip the current test only when its demo truly needs absent Torch.

    A genuine compiler/assertion failure is never hidden: with Torch present no
    skip occurs, and demos that do not (transitively) import Torch always run.
    """
    if demo_requires_torch(basename) and not torch_available():
        pytest.skip(f"{canonical_example_name(basename)} (transitively) imports the "
                    "optional Torch dependency, which is not installed in this CPU stage")


BENCHMARKS_DATAFLOW = REPO_ROOT / "benchmarks" / "dataflow"


def role_bench_path(basename: str) -> Path:
    path = BENCHMARKS_DATAFLOW / canonical_bench_name(basename)
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataflow benchmark: {path}")
    return path
