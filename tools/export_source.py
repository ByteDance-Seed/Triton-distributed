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
"""Export shared infrastructure plus an explicit hardware dependency closure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from little_kernel.backends.registry import backend_specs, resolve_backend_spec


def _module_path(module):
    return Path(*module.split("."))


def _scope_selected(path: Path, selected: set[str], vendors: set[str]) -> bool:
    parts = path.parts
    if not parts:
        return False
    if parts[0] == "common":
        return True
    if parts[0] not in vendors:
        return False
    if len(parts) == 1 or parts[1] in {"__init__.py", "conftest.py", "common"}:
        return True
    return any(
        parts[0] == spec.backend_id.split("/")[0] and parts[1] == _module_path(spec.factory.split(":")[0]).parts[-2]
        for spec in backend_specs()
        if spec.backend_id in selected)


def export_source(destination: Path, backend_ids: list[str], *, write_lock=True):
    destination = destination.resolve()
    if destination == ROOT or ROOT.is_relative_to(destination):
        raise ValueError("Export destination must not contain the working checkout")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Export destination is not empty: {destination}")
    specs = [resolve_backend_spec(name) for name in backend_ids]
    selected = {spec.backend_id for spec in specs}
    if not selected:
        raise ValueError("Select at least one installed backend")
    vendors = {key.split("/")[0] for key in selected}
    modules = {_module_path(m) for spec in specs for m in spec.modules}
    destination.mkdir(parents=True, exist_ok=True)
    copied = []

    def copy(path):
        if path.is_symlink():
            raise ValueError(f"Source exports require regular files, not symlinks: {path}")
        rel = path.relative_to(ROOT)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(rel.as_posix())

    for path in (ROOT / "python/little_kernel").rglob("*"):
        if not path.is_file() or any(p in {"__pycache__", "build"} or p.endswith(".egg-info") for p in path.parts):
            continue
        rel = path.relative_to(ROOT / "python")
        if "backends" in rel.parts:
            component = Path(*rel.parts[:-1])
            shared_api = len(rel.parts) == 3  # backends/{interfaces,registry,shared,__init__}.py
            if not shared_api and not any(component == m or component.is_relative_to(m) or
                                          (path.name == "__init__.py" and m.is_relative_to(component))
                                          for m in modules):
                continue
        copy(path)
    for name in ("tests", "benchmarks", "examples", "assets", "tools"):
        base = ROOT / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(base)
            if rel.name == "__init__.py" and len(rel.parts) == 1:
                copy(path)
            elif _scope_selected(rel, selected, vendors):
                copy(path)
    # Release tooling is generic. Full-repository migration tests/tools are not exported.
    for name in ("export_source.py", "merge_rank_traces.py", "verify_backend.py", "verify.py", "__init__.py"):
        path = ROOT / "tools" / name
        if path.is_file():
            copy(path)
    testpaths = ["tests/common"]
    for path in sorted((destination / "tests").glob("*/*")):
        for name in ("unit", "refactor", "dataflow"):
            if (path / name).is_dir():
                testpaths.append((path / name).relative_to(destination).as_posix())
    pyproject = f'''[project]
name = "little-kernel"
version = "0.1.0"
description = "LittleKernel selected hardware source distribution"
requires-python = ">=3.11"
dependencies = ["numpy"]

[dependency-groups]
cpu-test = ["pytest", "numpy"]
build = ["setuptools>=69", "wheel"]

[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
include-package-data = false

[tool.setuptools.packages.find]
where = ["python"]

[tool.setuptools.package-data]
"little_kernel.backends" = ["**/*.json", "**/*.cuh", "**/*.h", "**/*.hpp"]

[tool.pytest.ini_options]
testpaths = {json.dumps(testpaths)}
markers = ["gpu: requires matching accelerator", "sm90: requires SM90", "sm100: requires SM100", "distributed: requires multiple accelerators", "torch: optional tensor integration", "benchmark: opt-in performance run"]
'''
    (destination / "pyproject.toml").write_text(pyproject)
    (destination /
     "README.md").write_text("# LittleKernel selected source distribution\n\n" + "Backends: " +
                             ", ".join(sorted(selected)) + "\n\n" +
                             "CPU validation: `uv sync --locked --group cpu-test --group build`, then "
                             "`uv run --locked pytest`. Device execution requires the matching vendor "
                             "toolchain/driver and tensor integration; CPU passes are not device acceptance.\n")
    if write_lock:
        subprocess.run(
            ["uv", "lock", "--offline", "--directory", str(destination)], check=True, capture_output=True, text=True)
    manifest = {
        "backends": sorted(selected), "modules": sorted(map(str, modules)), "files":
        {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
         for name in sorted(copied)}
    }
    (destination / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", required=True, help="comma-separated installed backend ids")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = export_source(args.output, [x.strip() for x in args.backends.split(",") if x.strip()])
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "backends": manifest["backends"], "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
