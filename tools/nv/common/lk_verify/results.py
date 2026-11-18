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
"""Typed verification result states, admission rules and exit codes (RF13).

The four states keep fundamentally different outcomes apart:

* PASSED       -- the case actually ran and verified something;
* FAILED       -- the case ran and a real assertion / compile / error /
                  collection failure occurred (diagnostics retained);
* SKIPPED      -- an **optional** capability was absent, or an explicitly
                  optional (``--allow-missing``) required capability was
                  absent. Never counted as a pass; marked *not verified*;
* UNAVAILABLE  -- a **required** capability was absent and the developer did
                  not opt into ``--allow-missing``. A required gate cannot
                  admit this way.

Additionally, a **required** case that *executed* (pytest returned exit 0)
but produced **zero real passes** -- all tests skipped / zero collected -- is
``NOT VERIFIED`` and becomes UNAVAILABLE: it cannot be admitted, even if an
unrelated case passes and even under ``--allow-missing``. An **optional**
test-file that self-skips because its optional capability (e.g. Torch) is
absent stays SKIPPED and a sibling real pass can still admit the suite.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3  # distinct so CI/root GPU acceptance can branch on cause


class ResultState(str, enum.Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"


_STATE_GLYPH = {
    ResultState.PASSED: "PASS",
    ResultState.FAILED: "FAIL",
    ResultState.SKIPPED: "SKIP",
    ResultState.UNAVAILABLE: "UNAV",
}


@dataclass
class CaseResult:
    case_id: str
    state: ResultState
    detail: str = ""
    seconds: float = 0.0
    extra: Dict[str, object] = field(default_factory=dict)
    # True when the case ran/skipped but verified nothing. A required
    # not-verified case cannot be admitted, even with --allow-missing.
    not_verified: bool = False
    # True when this is a *required* gate that executed/should have executed.
    # A required not-verified case rejects admission even with --allow-missing.
    required_case: bool = True

    @property
    def ok(self) -> bool:
        return self.state is ResultState.PASSED

    def render_line(self) -> str:
        glyph = _STATE_GLYPH[self.state]
        suffix = f" [{self.seconds:.1f}s]" if self.state is ResultState.PASSED else ""
        not_verified = " (NOT VERIFIED)" if self.not_verified else ""
        detail = f" -- {self.detail}" if self.detail else ""
        return f"[{glyph}] {self.case_id}{suffix}{not_verified}{detail}"


@dataclass
class SuiteRun:
    suite_name: str
    results: List[CaseResult]

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {
            "passed": 0, "failed": 0, "skipped": 0, "unavailable": 0, "not_verified": 0, "total": len(self.results)
        }
        for result in self.results:
            counts[result.state.value] += 1
            if result.not_verified:
                counts["not_verified"] += 1
        return counts

    def is_admitted(self, *, allow_missing: bool = False) -> bool:
        """Whether this suite passes its admission gate.

        Rules, in order:
        * any FAILED case (incl. an executed-but-zero-real-pass case) rejects;
        * any case that verified nothing required rejects -- even if another
          case passed and even under --allow-missing;
        * any UNAVAILABLE case rejects unless the developer asked to carry it
          as explicit not-verified SKIP;
        * a suite with zero real passes cannot admit.
        """
        counts = self.counts()
        if counts["failed"]:
            return False
        for result in self.results:
            if result.not_verified and result.required_case:
                return False
        if counts["unavailable"] and not allow_missing:
            return False
        if counts["passed"] == 0:
            return False
        return True


def admission_exit_code(runs: Sequence[SuiteRun], *, allow_missing: bool = False) -> int:
    """Aggregate suite admission into the documented process exit code."""
    if not runs:
        # Nothing was executed at all: that is never a successful admission.
        return EXIT_FAILED
    any_failure = any(r.counts()["failed"] for r in runs)
    if any_failure:
        return EXIT_FAILED
    if not all(r.is_admitted(allow_missing=allow_missing) for r in runs):
        # If the only obstruction is a developer-carried missing capability
        # vs. required-not-verified, exit FAILED: nothing "unavailable" is
        # being hidden. UNAVAILABLE without --allow-missing is distinct.
        any_unavailable = any(r.counts()["unavailable"] for r in runs)
        if any_unavailable and not allow_missing:
            return EXIT_UNAVAILABLE
        return EXIT_FAILED
    return EXIT_OK
