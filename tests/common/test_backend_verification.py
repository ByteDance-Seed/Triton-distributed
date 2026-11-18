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
"""Required accelerator gates must report unverified work without admitting it."""
import json
from types import SimpleNamespace

import pytest

from little_kernel.backends import available_backends
from tools import verify_backend as runner


def test_missing_declared_gate_writes_an_unavailable_report(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "suite_paths", lambda *a: [])
    report = tmp_path / "report.json"
    result = runner.verify(available_backends()[0], report=report)
    assert result["state"] == "unavailable" and result["passed"] == 0
    assert json.loads(report.read_text()) == result


@pytest.mark.parametrize("skipped,expected", [(True, "unavailable"), (False, "passed")])
def test_junit_requires_a_real_pass(monkeypatch, tmp_path, skipped, expected):
    monkeypatch.setattr(runner, "suite_paths", lambda *a: [runner.ROOT / "tests/common"])

    def run(command, **kwargs):
        junit = command[command.index("--junitxml") + 1]
        from pathlib import Path
        child = "<skipped/>" if skipped else ""
        Path(junit).write_text(f"<testsuites><testsuite><testcase>{child}</testcase></testsuite></testsuites>")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", run)
    result = runner.verify(available_backends()[0])
    assert result["state"] == expected
    assert result["passed"] == (0 if skipped else 1)
