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
"""HIP runtime library discovery: configured SDK roots (lib and lib64 layouts)."""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from little_kernel.backends.amd.cdna4.runtime import hip_runtime
from little_kernel.backends.amd.cdna4.runtime.hip_runtime import HIPRuntime

_ENV_VARS = ("LK_LIBAMDHIP64", "ROCM_PATH", "HIP_PATH")


class LoadLibraryCandidates(unittest.TestCase):
    """_load_library follows caller configuration instead of a host installation."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        for var in _ENV_VARS:
            os.environ.pop(var, None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.library = SimpleNamespace()
        self.seen = []

    def _sdk(self, name, libdirs):
        root = os.path.join(self.tmp.name, name)
        for libdir in libdirs:
            os.makedirs(os.path.join(root, libdir))
            open(os.path.join(root, libdir, "libamdhip64.so"), "w").close()
        return root

    def _load_library(self, fail_paths=()):

        def load(path, **kwargs):
            self.seen.append(path)
            if path in fail_paths:
                raise OSError("unavailable")
            return self.library

        with mock.patch.object(hip_runtime.ctypes, "CDLL", load):
            return HIPRuntime._load_library()

    def test_rocm_path_lib64_only_tree(self):
        # Regression: a ROCm tree whose runtime lives only under lib64 used to be invisible.
        root = self._sdk("rocm", ["lib64"])
        os.environ["ROCM_PATH"] = root
        fail = {os.path.join(root, "lib", "libamdhip64.so")}
        self.assertIs(self._load_library(fail_paths=fail), self.library)
        self.assertEqual(self.seen,
                         [os.path.join(root, "lib", "libamdhip64.so"),
                          os.path.join(root, "lib64", "libamdhip64.so")])

    def test_hip_path_lib64_only_tree(self):
        root = self._sdk("hip", ["lib64"])
        os.environ["HIP_PATH"] = root
        fail = {os.path.join(root, "lib", "libamdhip64.so")}
        self.assertIs(self._load_library(fail_paths=fail), self.library)
        self.assertEqual(self.seen,
                         [os.path.join(root, "lib", "libamdhip64.so"),
                          os.path.join(root, "lib64", "libamdhip64.so")])

    def test_lib_candidate_precedes_lib64(self):
        root = self._sdk("rocm", ["lib", "lib64"])
        os.environ["ROCM_PATH"] = root
        self.assertIs(self._load_library(), self.library)
        self.assertEqual(self.seen, [os.path.join(root, "lib", "libamdhip64.so")])

    def test_rocm_path_precedes_hip_path(self):
        rocm = self._sdk("rocm", ["lib"])
        hip = self._sdk("hip", ["lib64"])
        os.environ["ROCM_PATH"] = rocm
        os.environ["HIP_PATH"] = hip
        self.assertIs(self._load_library(), self.library)
        self.assertEqual(self.seen, [os.path.join(rocm, "lib", "libamdhip64.so")])

    def test_loader_bare_name_is_last_resort(self):
        root = self._sdk("rocm", ["lib64"])
        os.environ["ROCM_PATH"] = root
        explicit = {os.path.join(root, libdir, "libamdhip64.so") for libdir in ("lib", "lib64")}
        self.assertIs(self._load_library(fail_paths=explicit), self.library)
        self.assertEqual(self.seen, [
            os.path.join(root, "lib", "libamdhip64.so"),
            os.path.join(root, "lib64", "libamdhip64.so"), "libamdhip64.so"
        ])

    def test_explicit_override_wins(self):
        override = os.path.join(self.tmp.name, "custom.so")
        os.environ["LK_LIBAMDHIP64"] = override
        os.environ["ROCM_PATH"] = self._sdk("rocm", ["lib64"])
        self.assertIs(self._load_library(), self.library)
        self.assertEqual(self.seen, [override])

    def test_unset_roots_try_loader_bare_name(self):
        self.assertIs(self._load_library(), self.library)
        self.assertEqual(self.seen, ["libamdhip64.so"])

    def test_all_candidates_fail_reports_every_path(self):
        root = self._sdk("rocm", ["lib64"])
        os.environ["ROCM_PATH"] = root
        with mock.patch.object(hip_runtime.ctypes, "CDLL", side_effect=OSError("unavailable")):
            with self.assertRaises(RuntimeError) as ctx:
                HIPRuntime._load_library()
        message = str(ctx.exception)
        self.assertIn("Cannot load libamdhip64.so", message)
        self.assertIn(os.path.join(root, "lib", "libamdhip64.so"), message)
        self.assertIn(os.path.join(root, "lib64", "libamdhip64.so"), message)


if __name__ == "__main__":
    unittest.main()
