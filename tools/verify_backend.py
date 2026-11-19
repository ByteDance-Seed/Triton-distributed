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
"""Run a hardware-owned correctness gate with explicit verified/unavailable states."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from little_kernel.backends import available_backends, resolve_backend_spec, get_backend


def asset_roots(category, backend_id):
    spec = resolve_backend_spec(backend_id)
    vendor = spec.backend_id.split("/")[0]
    hardware = spec.factory.split(":")[0].split(".")[-2]
    paths = [ROOT / category / vendor / "common", ROOT / category / vendor / hardware]
    return [p for p in paths if p.is_dir()]


def suite_paths(backend_id, suite):
    if suite == "cpu":
        result = [ROOT / "tests/common"] if (ROOT / "tests/common").is_dir() else []
        for base in asset_roots("tests", backend_id):
            result.extend(base / name for name in ("unit", "refactor", "dataflow") if (base / name).is_dir())
        return result
    return [
        base / name
        for base in asset_roots("tests", backend_id)
        for name in ("integration", "distributed")
        if (base / name).is_dir()
    ]


def verify(backend_id, suite="cpu", report=None):
    backend_id = resolve_backend_spec(backend_id).backend_id
    paths = suite_paths(backend_id, suite)
    if suite == "gpu":
        try:
            get_backend(backend_id).target.detect()
        except (RuntimeError, ValueError, ImportError, OSError, subprocess.TimeoutExpired) as exc:
            result = {"backend": backend_id, "suite": suite, "state": "unavailable", "reason": str(exc), "passed": 0}
            if report is not None:
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(json.dumps(result, indent=2) + "\n")
            return result
    if not paths:
        result = {
            "backend": backend_id, "suite": suite, "state": "unavailable", "reason":
            "No gate is declared in this source distribution", "passed": 0
        }
        if report is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(result, indent=2) + "\n")
        return result
    with tempfile.TemporaryDirectory(prefix="little-kernel-verify-") as temp:
        junit = Path(temp) / "junit.xml"
        command = [
            "uv", "run", "--locked", "--no-sync", "pytest", "-q", "--tb=short",
            *[str(p.relative_to(ROOT)) for p in paths], "--junitxml",
            str(junit)
        ]
        env = os.environ.copy()
        if suite == "cpu":
            env.update(CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", ASCEND_RT_VISIBLE_DEVICES="")
        process = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        junit_error = None
        cases = []
        if junit.is_file():
            try:
                cases = list(ET.parse(junit).getroot().iter("testcase"))
            except (ET.ParseError, OSError) as exc:
                # An interrupted or killed runner can leave a truncated JUnit
                # report; unverifiable evidence must fail the gate, never crash
                # it or be read as a pass.
                junit_error = f"could not parse JUnit report: {exc}"
        skipped = sum(c.find("skipped") is not None for c in cases)
        failed = sum(c.find("failure") is not None or c.find("error") is not None for c in cases)
        passed = len(cases) - skipped - failed
        if junit_error is not None or failed or process.returncode not in {0, 5}:
            state = "failed"
        elif passed == 0:
            state = "unavailable"
        else:
            state = "passed"
        result = {
            "backend": backend_id, "suite": suite, "state": state, "passed": passed, "failed": failed, "skipped":
            skipped, "returncode": process.returncode, "command": command, "stdout": process.stdout, "stderr":
            process.stderr
        }
        if junit_error is not None:
            result["junit_error"] = junit_error
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=available_backends())
    parser.add_argument("--suite", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print(
            json.dumps(
                {
                    key: {
                        category: [str(p.relative_to(ROOT))
                                   for p in asset_roots(category, key)]
                        for category in ("tests", "benchmarks", "examples", "assets")
                    }
                    for key in available_backends()
                }, indent=2))
        return 0
    if args.backend is None:
        parser.error("Select an installed backend using --backend")
    result = verify(args.backend, args.suite, args.report)
    print(json.dumps({k: v for k, v in result.items() if k not in {"stdout", "stderr", "command"}}))
    if result["state"] != "passed":
        print(result.get("stdout", "")[-6000:])
        print(result.get("stderr", "")[-2000:])
    return {"passed": 0, "failed": 1, "unavailable": 3}[result["state"]]


if __name__ == "__main__":
    raise SystemExit(main())
