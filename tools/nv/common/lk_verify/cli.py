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

# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
"""Command-line interface for the authoritative verification suites.

Examples (every Python/test invocation uses locked uv)::

    uv run --locked python -m lk_verify --suites cpu
    uv run --locked python -m lk_verify --suites quick
    uv run --locked python -m lk_verify --suites cpu,cuda-compile --allow-missing
    uv run --locked python -m lk_verify --suites performance --include-performance
    uv run --locked python -m lk_verify --list
    uv run --locked python -m lk_verify --suites sm90 --only vector-memory,math-intrinsics

Exit codes: ``0`` admitted, ``1`` real failure/usage, ``2`` a required
capability (hardware/dependency) is unavailable. Required missing hardware
never exits 0 and is never counted as a pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from tools.nv.common.lk_verify.capabilities import detect_capabilities
from tools.nv.common.lk_verify.manifest import SUITES, expand_suite_names, validate_manifest_paths
from tools.nv.common.lk_verify.results import EXIT_FAILED, EXIT_OK, EXIT_USAGE, CaseResult, ResultState, admission_exit_code
from tools.nv.common.lk_verify.runner import REPO_ROOT, VerificationRunner


def _split(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lk_verify",
        description="Run LittleKernel's documented CPU/GPU/performance suites "
        "with typed results and nonzero failure exits.",
    )
    parser.add_argument(
        "--suites",
        default="quick",
        help="comma-separated suite names or aliases (default: quick; "
        "aliases: all, quick, v2)",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated case ids to run within the selected suites",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list suites/cases with their capabilities and exit",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="developer mode: do not fail admission when a *required* hardware/"
        "dependency capability is absent (a whole suite with zero real "
        "passes still fails). Missing capability is never counted as pass.",
    )
    parser.add_argument(
        "--include-performance",
        action="store_true",
        help="permit the expensive performance suite; it never runs otherwise",
    )
    parser.add_argument("--timeout", type=float, default=None, help="per-case subprocess timeout in seconds")
    parser.add_argument("--report", type=Path, default=None, help="write a structured JSON report to this path")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--detect-only", action="store_true", help="print detected host capabilities and exit")
    parser.add_argument(
        "--references", choices=("auto", "required", "none"), default="auto",
        help="external benchmark reference mode forwarded to "
        "benchmark gates (compat with verify_v2.sh)")
    return parser


def _render_list() -> str:
    lines = []
    for suite in SUITES.values():
        expensive = " [EXPENSIVE: explicit selection only]" if suite.expensive else ""
        req = ", ".join(sorted(suite.required)) or "-"
        lines.append(f"\n[{suite.name}]{expensive}")
        lines.append(f"  {suite.description}")
        lines.append(f"  suite-required: {req}")
        for case in suite.cases:
            reqc = ", ".join(sorted(case.required)) or "-"
            optc = ", ".join(sorted(case.optional)) or "-"
            paths = ", ".join(case.paths) or (case.kind and f"<{case.kind}>")
            lines.append(f"    - {case.case_id:<32} kind={case.kind:<8} "
                         f"required={reqc:<24} optional={optc:<12} {paths}")
    return "\n".join(lines)


def _result_to_dict(result: CaseResult) -> dict:
    return {
        "case": result.case_id,
        "state": result.state.value,
        "not_verified": result.not_verified,
        "detail": result.detail,
        "seconds": round(result.seconds, 3),
        **({"extra": result.extra} if result.extra else {}),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    caps = detect_capabilities()

    if args.detect_only:
        print(
            json.dumps(
                {
                    "nvcc": caps.has_nvcc,
                    "nvcc_path": caps.nvcc_path,
                    "nvcc_version": caps.nvcc_version,
                    "nvcc_supports_sm100": caps.satisfies("nvcc-sm100"),
                    "torch": caps.has_torch,
                    "gpu": caps.has_gpu,
                    "gpu_count": caps.gpu_count,
                    "sm_arch": caps.sm_arch,
                    "exact_sm90": caps.satisfies("sm90"),
                    "exact_sm100": caps.satisfies("sm100"),
                    "cuda_driver": caps.satisfies("cuda-driver"),
                    "selected_device_index": caps.selected_device_index,
                    "cuda_visible_devices": caps.visible_devices,
                    "gpu_name": caps.gpu_name,
                    "missing_modules": sorted(caps.missing_modules),
                }, indent=2))
        return EXIT_OK

    if args.list:
        print(_render_list())
        return EXIT_OK

    # Manifest integrity: a broken suite path must fail before any run.
    problems = validate_manifest_paths(REPO_ROOT)
    if problems:
        print("Manifest validation failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return EXIT_FAILED

    try:
        suite_names = expand_suite_names(_split(args.suites) or ["quick"])
    except KeyError as exc:
        parser.error(str(exc))
        return EXIT_USAGE

    if any(SUITES[name].expensive for name in suite_names) and not args.include_performance:
        print(
            "error: the performance suite is expensive and is never run by "
            "default; pass --include-performance to opt in deliberately.", file=sys.stderr)
        return EXIT_USAGE

    only = _split(args.only)
    if only:
        valid_ids = {case.case_id for name in suite_names for case in SUITES[name].cases}
        unknown = sorted(set(only) - valid_ids)
        if unknown:
            parser.error(f"unknown --only case id(s) for selected suites: {', '.join(unknown)}")
            return EXIT_USAGE

    runner = VerificationRunner(
        caps=caps,
        allow_missing=args.allow_missing,
        timeout=args.timeout,
        verbose=True,
        references_mode=args.references,
    )

    runs = []
    print(f"Host: nvcc={caps.nvcc_version or 'absent'} "
          f"nvcc-sm100={caps.satisfies('nvcc-sm100')} torch={caps.has_torch} "
          f"gpu={caps.has_gpu} sm={caps.sm_arch or '-'} "
          f"exact-sm90={caps.satisfies('sm90')} exact-sm100={caps.satisfies('sm100')}")
    print(f"      {caps.selected_device_label()}")
    for name in suite_names:
        print(f"\n=== suite: {name} ===")
        run = runner.run_suite(SUITES[name], only=only)
        runs.append(run)
        counts = run.counts()
        print(f"--- {name}: {counts['passed']} passed, {counts['failed']} failed, "
              f"{counts['skipped']} skipped, {counts['unavailable']} unavailable" +
              (f", {counts['not_verified']} not-verified" if counts['not_verified'] else "") + f" of {counts['total']}")

    # Final summary with distinct states (never collapse missing into pass).
    print("\n" + "=" * 72)
    print("SUMMARY")
    total = {"passed": 0, "failed": 0, "skipped": 0, "unavailable": 0, "not_verified": 0}
    problem_diag = []
    for run in runs:
        for result in run.results:
            total[result.state.value] += 1
            if result.not_verified:
                total["not_verified"] += 1
            if result.state in (ResultState.FAILED, ResultState.UNAVAILABLE):
                problem_diag.append(result.render_line())
    print(f"passed={total['passed']} failed={total['failed']} "
          f"skipped={total['skipped']} unavailable={total['unavailable']} "
          f"not-verified={total['not_verified']}")
    if problem_diag:
        print("\nFailed/unavailable cases (diagnostics retained above):")
        for line in problem_diag:
            print(f"  * {line}")
    print("=" * 72)

    if args.report:
        payload = {
            "host": {
                "nvcc": caps.nvcc_version,
                "nvcc_supports_sm100": caps.satisfies("nvcc-sm100"),
                "torch": caps.has_torch,
                "gpu": caps.has_gpu,
                "gpu_count": caps.gpu_count,
                "sm_arch": caps.sm_arch,
                "exact_sm100": caps.satisfies("sm100"),
                "selected_device": caps.selected_device_label(),
            },
            "suites": {
                run.suite_name: {
                    "counts": run.counts(),
                    "admitted": run.is_admitted(allow_missing=args.allow_missing),
                    "cases": [_result_to_dict(r)
                              for r in run.results],
                }
                for run in runs
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n")

    # Single source of truth for the exit code: a required gate that ran but
    # verified nothing fails even with --allow-missing.
    return admission_exit_code(runs, allow_missing=args.allow_missing)


if __name__ == "__main__":
    raise SystemExit(main())
