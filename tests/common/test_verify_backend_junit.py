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
"""Regression: an unparseable JUnit report must fail the gate, not crash it.

tools/verify_backend.py used to call ET.parse on the pytest report without
guarding. A runner that is interrupted or killed (returncode outside {0, 5},
or a report truncated mid-write) left a malformed junit.xml whose ParseError
propagated out of verify(): the gate crashed with a traceback, wrote no JSON
report, and never reached its documented 0/1/3 exit-code contract. Unverifiable
evidence must report state 'failed' instead.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
# Same portable discovery as tools/verify_backend.py: make the src layout and
# the tools package importable without requiring an installed wheel.
for _path in (str(REPO_ROOT), str(REPO_ROOT / "python")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from little_kernel.backends import available_backends
from tools import verify_backend as runner

TRUNCATED_JUNIT = "<?xml version='1.0'?><testsuites><testsuite><testcase"
VALID_ONE_PASS = "<testsuites><testsuite><testcase/></testsuite></testsuites>"
VALID_ONE_FAIL = ("<testsuites><testsuite><testcase>"
                  "<failure message='mismatch'>trace</failure>"
                  "</testcase></testsuite></testsuites>")


class _VerifyGateTestBase(unittest.TestCase):
    """Drive verify() with an injected pytest subprocess; never runs pytest."""

    def run_verify(self, returncode, junit_content, report=None):

        def fake_run(command, **kwargs):
            path = Path(command[command.index("--junitxml") + 1])
            if junit_content is not None:
                path.write_text(junit_content)
            return SimpleNamespace(returncode=returncode, stdout="", stderr="")

        with mock.patch.object(runner, "suite_paths",
                               lambda *args: [runner.ROOT / "tests/common"]), \
             mock.patch.object(runner.subprocess, "run", fake_run):
            return runner.verify(available_backends()[0], report=report)


class CorruptJunitAlwaysFails(_VerifyGateTestBase):

    def test_interrupted_runner_with_truncated_junit_reports_failed(self):
        result = self.run_verify(2, TRUNCATED_JUNIT)
        self.assertEqual(result["state"], "failed")
        self.assertIn("junit_error", result)
        self.assertEqual(result["returncode"], 2)
        self.assertEqual(result["passed"], 0)

    def test_killed_runner_with_truncated_junit_reports_failed(self):
        # Killed by SIGKILL: returncode is negative, junit may be partial.
        result = self.run_verify(-9, TRUNCATED_JUNIT)
        self.assertEqual(result["state"], "failed")
        self.assertIn("junit_error", result)

    def test_zero_exit_with_truncated_junit_fails_closed(self):
        # rc=0 claims success but the evidence is unverifiable: never pass,
        # and never claim 'unavailable' for a runner that says it ran.
        result = self.run_verify(0, TRUNCATED_JUNIT)
        self.assertEqual(result["state"], "failed")
        self.assertIn("junit_error", result)

    def test_no_tests_collected_with_truncated_junit_fails_closed(self):
        # rc=5 with a healthy run writes a valid empty report; a corrupt one
        # is anomalous and cannot be admitted as either passed or unavailable.
        result = self.run_verify(5, TRUNCATED_JUNIT)
        self.assertEqual(result["state"], "failed")
        self.assertIn("junit_error", result)

    def test_corrupt_junit_still_writes_json_report(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "report.json"
            result = self.run_verify(2, TRUNCATED_JUNIT, report=report)
            self.assertTrue(report.is_file())
            on_disk = json.loads(report.read_text())
            self.assertEqual(on_disk, result)
            self.assertEqual(on_disk["state"], "failed")


class WellFormedJunitBehaviourUnchanged(_VerifyGateTestBase):

    def test_valid_junit_with_nonzero_returncode_is_failed(self):
        result = self.run_verify(2, VALID_ONE_PASS)
        self.assertEqual(result["state"], "failed")
        self.assertNotIn("junit_error", result)

    def test_valid_junit_with_failure_is_failed(self):
        result = self.run_verify(1, VALID_ONE_FAIL)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["failed"], 1)
        self.assertNotIn("junit_error", result)

    def test_valid_junit_with_real_pass_stays_passed(self):
        result = self.run_verify(0, VALID_ONE_PASS)
        self.assertEqual(result["state"], "passed")
        self.assertEqual(result["passed"], 1)
        self.assertNotIn("junit_error", result)

    def test_missing_junit_with_zero_exit_stays_unavailable(self):
        result = self.run_verify(0, None)
        self.assertEqual(result["state"], "unavailable")
        self.assertEqual(result["passed"], 0)
        self.assertNotIn("junit_error", result)


if __name__ == "__main__":
    unittest.main()
