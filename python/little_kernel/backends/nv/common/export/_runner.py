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
"""Artifact orchestration: render, verify, write and compile the submission.

One exporter invocation owns one CUDA translation unit and binding; outputs are
written only after a successful full render and restored on compilation
failure. This module contains no evaluator-specific policy.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from little_kernel.backends.nv.common.export._render import render_bundle
from little_kernel.backends.nv.common.export._validation import validate_spec


def _render_submission(source: Path, spec_path: Path, workspace: Path):
    root = workspace.resolve(strict=True)
    agent = root / "agent"
    task = root / "task"
    if not agent.is_dir() or not task.is_dir():
        raise ValueError("workspace must contain task/ and agent/")
    source = source.resolve(strict=True)
    spec_path = spec_path.resolve(strict=True)
    if agent not in source.parents or agent not in spec_path.parents:
        raise ValueError("source and spec must be regular files under agent/")
    if not source.is_file() or not spec_path.is_file():
        raise ValueError("source and spec must be regular files")
    # Validation and rendering are evaluator-independent; the only eval policy
    # here is the required task/agent workspace layout and output locations.
    bundle = render_bundle(source, validate_spec(spec_path))
    function = bundle.function
    paths = [
        agent / "kernels" / f"agent_lk_{function}.cu", agent / "kernels" / f"agent_lk_{function}_binding.cpp",
        agent / "model_new.py"
    ]
    return paths, (bundle.cuda, bundle.binding, bundle.model)


def _compile_error_detail(agent: Path, output: str) -> str:
    log = agent / "compilation_full.log"
    errors = []
    if log.is_file():
        errors = [
            line.strip()[:500]
            for line in log.read_text(errors="replace").splitlines()
            if any(word in line.lower() for word in ("error:", "fatal", "undefined reference"))
        ]
    return "\n".join(errors[-8:]) or output[-3000:].strip()


def export(source: Path, spec_path: Path, workspace: Path, *, force: bool = False,
           compile_after: bool = False) -> list[Path]:
    paths, contents = _render_submission(source, spec_path, workspace)
    kernels = paths[0].parent
    # One exporter invocation owns one CUDA translation unit and binding.
    # A changed function name must not leave the previous generated pair in
    # the build, where it can cause duplicate registrations or fail --strict.
    managed = set(paths)
    managed.update(kernels.glob("agent_lk_*.cu"))
    managed.update(kernels.glob("agent_lk_*_binding.cpp"))
    if not force and any(path.exists() for path in managed):
        raise FileExistsError("generated output exists; pass --force to replace it")
    compile_script = workspace.resolve(strict=True) / "task" / "compile.sh"
    if compile_after and not compile_script.is_file():
        raise FileNotFoundError(f"compile script is missing: {compile_script}")
    previous = {path: path.read_bytes() if path.is_file() else None for path in managed}
    had_submission = (previous.get(paths[2]) is not None
                      and any(path.suffix == ".cu" and value is not None for path, value in previous.items())
                      and any(path.suffix == ".cpp" and value is not None for path, value in previous.items()))
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    try:
        for stale in managed.difference(paths):
            stale.unlink()
        for path, content in zip(paths, contents):
            path.write_text(content)
        if compile_after:
            result = subprocess.run(["bash", str(compile_script)], cwd=workspace, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, check=False)
            if result.returncode:
                detail = _compile_error_detail(workspace / "agent", result.stdout)
                raise RuntimeError(f"CUDA compilation failed (exit {result.returncode}):\n{detail}")
    except Exception as error:
        for path, contents_before in previous.items():
            if contents_before is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(contents_before)
        if compile_after and had_submission:
            # compile.sh may have removed the prior extension. Restore it too.
            recovered = subprocess.run(["bash", str(compile_script)], cwd=workspace, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, check=False)
            if recovered.returncode:
                raise RuntimeError(f"{error}\nPrevious sources were restored, but rebuilding their extension "
                                   f"also failed (exit {recovered.returncode}).") from error
        raise
    return paths


def check(source: Path, spec_path: Path, workspace: Path, *, strict: bool = False) -> list[Path]:
    """Confirm submitted CUDA, binding and model still match the frontend source."""
    paths, contents = _render_submission(source, spec_path, workspace)
    changed = [str(path) for path, content in zip(paths, contents) if not path.is_file() or path.read_text() != content]
    if strict:
        agent = workspace.resolve(strict=True) / "agent"
        expected = set(paths[:2])
        changed.extend(
            str(path)
            for directory in (agent, agent / "kernels")
            for pattern in ("*.cu", "*.cpp")
            for path in directory.glob(pattern)
            if path not in expected)
    if changed:
        raise ValueError("submission differs from frontend export: " + ", ".join(changed))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--compile", action="store_true",
                        help="compile exported CUDA and restore previous files if compilation fails")
    parser.add_argument("--strict", action="store_true", help="with --check, reject extra CUDA/C++ sources")
    args = parser.parse_args(argv)
    if args.strict and not args.check:
        parser.error("--strict requires --check")
    if args.check and args.compile:
        parser.error("--compile cannot be combined with --check")
    operation = check if args.check else export
    options = ({"strict": args.strict} if args.check else {"force": args.force, "compile_after": args.compile})
    for path in operation(args.source, args.spec, args.workspace, **options):
        print(path)
    return 0
