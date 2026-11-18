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
"""Suite execution engine (RF13).

The runner turns manifest cases into typed :class:`CaseResult` values. It never
launches GPU jobs itself: capability checks gate every case before execution,
and subprocesses are injectable so tests can exercise failure/timeout paths
deterministically.

Every Python subprocess is invoked through locked uv:

    uv run --locked [--group G ...] <argv>

JUnit XML (never exit code alone) is the source of truth for pytest cases, so a
suite path that collects zero tests -- or where every test skipped -- cannot be
reported as a passing gate.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from tools.nv.common.lk_verify.capabilities import HostCaps, detect_capabilities
from tools.nv.common.lk_verify.compile_probes import COMPILE_PROBES
from tools.nv.common.lk_verify.manifest import CaseSpec, SuiteSpec, expand_suite_names, SUITES
from tools.nv.common.lk_verify.results import CaseResult, ResultState, SuiteRun

REPO_ROOT = Path(__file__).resolve().parents[4]

# CompletedProcess-like signature: (cmd, cwd, env, timeout) -> object with
# returncode/stdout/stderr. Injectable for deterministic tests.
SubprocessRunner = Callable[..., "subprocess.CompletedProcess"]
CompileFn = Callable[[str, str, str], bytes]

_DIAG_LIMIT = 4000


def _tail(text: str, limit: int = _DIAG_LIMIT) -> str:
    text = text or ""
    if len(text) <= limit:
        return text.strip()
    return "...<truncated>...\n" + text[-limit:].strip()


def case_environment(case: CaseSpec) -> dict:
    """Keep CPU suites CPU-only even on a workstation with visible GPUs."""
    env = os.environ.copy()
    if "cpu" in case.required:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def default_subprocess_runner(cmd, *, cwd, env, timeout=None, capture=True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        timeout=timeout,
        capture_output=capture,
        text=True,
        check=False,
    )


def build_uv_command(case: CaseSpec, resolved_paths: Sequence[str] = (),
                     references_mode: Optional[str] = None) -> List[str]:
    """Build the locked-uv command for a pytest/command case.

    ``resolved_paths`` are the on-disk targets chosen by
    :meth:`CaseSpec.select_paths` (canonical when present, legacy otherwise).
    ``references_mode`` (``auto``/``required``/``none``) is appended to
    benchmark command cases that declare the ``references`` capability, so the
    old ``verify_v2.sh --references none`` contract is preserved explicitly.
    """
    cmd = ["uv", "run", "--locked"]
    for group in case.uv_groups:
        cmd += ["--group", group]
    if case.kind == "pytest":
        cmd += ["pytest", "-q", "-p", "no:cacheprovider"]
        cmd += list(resolved_paths)
        cmd += list(case.args)
    else:  # command: ``python <resolved-script> <args>``
        cmd += ["python"]
        cmd += list(resolved_paths)
        cmd += list(case.args)
        if references_mode and "references" in (case.required | case.optional):
            cmd += ["--references", references_mode]
    return cmd


def resolve_or_fail(case: CaseSpec) -> Optional[CaseResult]:
    """Resolve existing paths; return a FAILED result when neither set exists.

    A missing suite target is a broken manifest/migration problem, never a
    silent skip, and it is independent of whether Torch is installed.
    """
    origin, existing = case.select_paths(REPO_ROOT)
    if origin == "missing":
        return CaseResult(
            case.case_id, ResultState.FAILED, detail="manifest target missing on disk (neither canonical nor "
            f"legacy path exists): {list(case.paths) or list(case.legacy_paths)}",
            extra={"canonical": list(case.paths), "legacy": list(case.legacy_paths)})
    return None


def _missing_state(case: CaseSpec, caps: HostCaps, allow_missing: bool) -> Optional[ResultState]:
    """Return the gating state when a required/optional capability is absent."""
    missing_required = frozenset(k for k in case.required if not caps.satisfies(k))
    if missing_required:
        return ResultState.SKIPPED if allow_missing else ResultState.UNAVAILABLE
    missing_optional = frozenset(k for k in case.optional if not caps.satisfies(k))
    if missing_optional:
        return ResultState.SKIPPED
    return None


def _gate_detail(case: CaseSpec, caps: HostCaps) -> str:
    missing_req = sorted(k for k in case.required if not caps.satisfies(k))
    missing_opt = sorted(k for k in case.optional if not caps.satisfies(k))
    parts = []
    if missing_req:
        parts.append("required capability absent: " + ", ".join(missing_req))
    if missing_opt:
        parts.append("optional capability absent: " + ", ".join(missing_opt))
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# pytest result parsing
# --------------------------------------------------------------------------- #


@dataclass
class PytestOutcome:
    tests: int
    failures: int
    errors: int
    skipped: int
    passed: int
    failure_diag: str


def parse_junit_xml(xml_text: str) -> PytestOutcome:
    """Parse pytest JUnit XML into typed counts and a retained diagnostic."""
    root = ET.fromstring(xml_text)
    suites = root.findall("testsuite") or ([root] if root.tag == "testsuite" else [])
    tests = failures = errors = skipped = passed = 0
    diag_chunks: List[str] = []
    for suite in suites:
        try:
            tests += int(suite.get("tests", 0))
            failures += int(suite.get("failures", 0))
            errors += int(suite.get("errors", 0))
            skipped += int(suite.get("skipped", 0))
        except ValueError:
            pass
        for tc in suite.findall("testcase"):
            for bad in tc.findall("failure") + tc.findall("error"):
                message = (bad.get("message") or "").strip()
                body = (bad.text or "").strip()
                diag_chunks.append(f"{tc.get('classname', '')}::{tc.get('name', '')}: "
                                   f"{message or body.splitlines()[-1] if body else message}")
    passed = max(tests - failures - errors - skipped, 0)
    return PytestOutcome(
        tests=tests,
        failures=failures,
        errors=errors,
        skipped=skipped,
        passed=passed,
        failure_diag=_tail("\n".join(diag_chunks)),
    )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class VerificationRunner:
    """Execute manifest cases with injected subprocess/compile collaborators."""

    def __init__(
        self,
        *,
        caps: Optional[HostCaps] = None,
        subprocess_runner: Optional[SubprocessRunner] = None,
        compile_fn: Optional[CompileFn] = None,
        allow_missing: bool = False,
        timeout: Optional[float] = None,
        verbose: bool = False,
        references_mode: Optional[str] = None,
    ) -> None:
        self.caps = caps if caps is not None else detect_capabilities()
        self._run_subprocess = subprocess_runner or default_subprocess_runner
        self._compile_fn = compile_fn
        self.allow_missing = allow_missing
        self.timeout = timeout
        self.verbose = verbose
        self.references_mode = references_mode

    # -- capability gating ------------------------------------------------- #
    def run_case(self, case: CaseSpec) -> CaseResult:
        state = _missing_state(case, self.caps, self.allow_missing)
        if state is not None:
            detail = _gate_detail(case, self.caps)
            if state is ResultState.SKIPPED:
                detail += " (not verified; not counted as pass)"
            # Capability-carried skips are an explicit developer/optional
            # decision; they do NOT enforce the hard required-not-verified
            # gate (that is reserved for cases that *executed* but verified
            # nothing despite their declared capabilities being present).
            return CaseResult(case.case_id, state, detail=detail, not_verified=state is ResultState.SKIPPED,
                              required_case=False)
        # Path integrity is checked regardless of Torch availability.
        # Compile cases synthesize their own kernels and have no on-disk
        # manifest targets.
        if case.kind != "compile":
            missing = resolve_or_fail(case)
            if missing is not None:
                return missing
        if case.kind == "pytest":
            return self._run_pytest_case(case)
        if case.kind == "command":
            return self._run_command_case(case)
        if case.kind == "compile":
            return self._run_compile_case(case)
        return CaseResult(case.case_id, ResultState.FAILED, detail=f"unsupported kind {case.kind!r}")

    @staticmethod
    def _resolved(case: CaseSpec) -> Tuple[str, ...]:
        _, paths = case.select_paths(REPO_ROOT)
        return paths

    # -- pytest ------------------------------------------------------------ #
    def _run_pytest_case(self, case: CaseSpec) -> CaseResult:
        with tempfile.TemporaryDirectory(prefix="lk-verify-") as tmp:
            junit = Path(tmp) / "result.xml"
            cmd = build_uv_command(case, self._resolved(case), self.references_mode) + ["--junitxml", str(junit)]
            started = time.monotonic()
            try:
                proc = self._run_subprocess(cmd, cwd=REPO_ROOT, env=case_environment(case), timeout=self.timeout)
            except subprocess.TimeoutExpired:
                return CaseResult(case.case_id, ResultState.FAILED,
                                  detail=f"timed out after {self.timeout}s: {' '.join(cmd)}",
                                  seconds=time.monotonic() - started, extra={"cmd": cmd})
            seconds = time.monotonic() - started
            stdout = getattr(proc, "stdout", "") or ""
            stderr = getattr(proc, "stderr", "") or ""
            returncode = int(getattr(proc, "returncode", 1))

            if not junit.exists():
                return CaseResult(
                    case.case_id, ResultState.FAILED, detail=f"pytest produced no JUnit report (exit {returncode}); "
                    f"{_tail(stderr or stdout)}", seconds=seconds, extra={"cmd": cmd})
            try:
                outcome = parse_junit_xml(junit.read_text())
            except ET.ParseError as exc:
                return CaseResult(case.case_id, ResultState.FAILED,
                                  detail=f"could not parse JUnit report: {exc}; {_tail(stderr)}", seconds=seconds)

        if outcome.failures or outcome.errors or returncode != 0:
            detail = (f"{outcome.failures} failure(s), {outcome.errors} error(s) "
                      f"of {outcome.tests} collected (exit {returncode})")
            if outcome.failure_diag:
                detail += f"\n{outcome.failure_diag}"
            elif returncode != 0:
                detail += f"\n{_tail(stderr or stdout)}"
            return CaseResult(case.case_id, ResultState.FAILED, detail=detail, seconds=seconds,
                              extra={"tests": outcome.tests, "passed": outcome.passed, "skipped": outcome.skipped})

        # Exit 0 is not acceptance by itself.
        if outcome.tests == 0:
            return CaseResult(
                case.case_id, ResultState.FAILED, detail="pytest collected 0 tests; suite path/selection is broken "
                "(exit 0 is not acceptance)", seconds=seconds, required_case=True, extra={"cmd": cmd})
        if outcome.passed == 0:
            # Required capabilities were present (otherwise the case gated
            # before launch), yet the gate verified nothing: every collected
            # test self-skipped. This is a REQUIRED executed gate with zero
            # real passes: it is UNAVAILABLE/not-verified and a hard admission
            # failure that can neither hide behind a sibling pass nor be
            # carried by --allow-missing (unlike an optional capability skip).
            return CaseResult(
                case.case_id, ResultState.UNAVAILABLE, detail=f"REQUIRED gate verified nothing: all "
                f"{outcome.skipped} collected test(s) self-skipped "
                f"(not an optional capability skip, not a pass)", seconds=seconds, not_verified=True,
                required_case=True, extra={"tests": outcome.tests, "passed": 0, "skipped": outcome.skipped})
        detail = f"{outcome.passed} passed"
        if outcome.skipped:
            detail += f", {outcome.skipped} skipped (not counted as pass)"
        return CaseResult(case.case_id, ResultState.PASSED, detail=detail, seconds=seconds,
                          extra={"tests": outcome.tests, "passed": outcome.passed, "skipped": outcome.skipped})

    # -- command ----------------------------------------------------------- #
    def _run_command_case(self, case: CaseSpec) -> CaseResult:
        cmd = build_uv_command(case, self._resolved(case), self.references_mode)
        started = time.monotonic()
        try:
            proc = self._run_subprocess(cmd, cwd=REPO_ROOT, env=case_environment(case), timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return CaseResult(case.case_id, ResultState.FAILED,
                              detail=f"timed out after {self.timeout}s: {' '.join(cmd)}",
                              seconds=time.monotonic() - started, extra={"cmd": cmd})
        seconds = time.monotonic() - started
        returncode = int(getattr(proc, "returncode", 1))
        combined = (getattr(proc, "stdout", "") or "") + "\n" + (getattr(proc, "stderr", "") or "")
        if returncode != 0:
            return CaseResult(case.case_id, ResultState.FAILED, detail=f"command exited {returncode}: {' '.join(cmd)}\n"
                              f"{_tail(combined)}", seconds=seconds, extra={"cmd": cmd, "exit": returncode})
        return CaseResult(case.case_id, ResultState.PASSED, detail="exit 0", seconds=seconds, extra={"cmd": cmd})

    # -- compile ----------------------------------------------------------- #
    def _run_compile_case(self, case: CaseSpec) -> CaseResult:
        if case.case_id not in COMPILE_PROBES:
            return CaseResult(case.case_id, ResultState.FAILED, detail=f"no compile probe named {case.case_id!r}")
        kernel, threads, markers, arch = COMPILE_PROBES[case.case_id]
        started = time.monotonic()
        try:
            from little_kernel.core.passes import PASSES
            from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

            source = kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=threads)
            missing_markers = [m for m in markers if m not in source]
            if missing_markers:
                return CaseResult(case.case_id, ResultState.FAILED,
                                  detail=f"generated CUDA is missing markers {missing_markers}",
                                  seconds=time.monotonic() - started, extra={"arch": arch})
            if self._compile_fn is not None:
                binary = self._compile_fn(source, case.case_id, arch)
            else:
                from little_kernel.backends.nv.common.runtime.compiler import compile_cuda
                binary = compile_cuda(source, case.case_id, arch=arch, target_format="ptx")
            if not binary:
                return CaseResult(case.case_id, ResultState.FAILED, detail="NVCC produced empty PTX",
                                  seconds=time.monotonic() - started)
        except Exception as exc:  # lower or NVCC failure is a real failure
            return CaseResult(case.case_id, ResultState.FAILED, detail=f"{type(exc).__name__}: {_tail(str(exc), 2000)}",
                              seconds=time.monotonic() - started, extra={"arch": arch})
        return CaseResult(
            case.case_id, ResultState.PASSED, detail=f"lowered and NVCC cross-compiled to PTX for {arch} "
            f"(toolchain result; not a GPU numerical/performance pass)", seconds=time.monotonic() - started,
            extra={"arch": arch, "ptx_bytes": len(binary)})

    # -- suites ------------------------------------------------------------ #
    def run_suite(self, suite: SuiteSpec, only: Optional[Sequence[str]] = None) -> SuiteRun:
        wanted = set(only) if only else None
        run = SuiteRun(suite.name, [])
        for case in suite.cases:
            if wanted is not None and case.case_id not in wanted:
                continue
            result = self.run_case(case)
            run.results.append(result)
            if self.verbose:
                print(result.render_line(), flush=True)
        return run


def select_cases(suite_names: Sequence[str], only: Optional[Sequence[str]] = None) -> Dict[str, SuiteSpec]:
    expanded = expand_suite_names(suite_names)
    return {name: SUITES[name] for name in expanded}
