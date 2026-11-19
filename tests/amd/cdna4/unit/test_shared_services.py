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
"""Typed ABI/cache/lifetime behavior with injected HIP resources, no GPU claims."""
import ctypes
import struct

import pytest

from little_kernel.backends import get_backend
from little_kernel.backends.interfaces import ExecutableKernel
from little_kernel.backends.amd.cdna4.runtime.hip_kernel import CompiledHipKernel
from little_kernel.backends.amd.cdna4.runtime.hip_runtime import HIPRuntime
from little_kernel.runtime.abi import KernelParameter, KernelSignature
from little_kernel.core.type_system import int64, float64


class RecordingRuntime:

    def __init__(self):
        self.calls = []
        self.fail_unload = False

    def load_module(self, code):
        self.calls.append(("load", code))
        return 1

    def get_function(self, module, name):
        return 2

    def launch(self, function, grid, block, arguments, shared, stream):
        self.calls.append(("launch", arguments))

    def synchronize(self, stream=None):
        self.calls.append(("sync", stream))

    def unload_module(self, module):
        self.calls.append(("unload", module))
        if self.fail_unload:
            raise RuntimeError("unload failed")


def signature():
    return KernelSignature([KernelParameter.scalar("n", int64), KernelParameter.scalar("scale", float64)])


def kernel(runtime, **kwargs):
    return CompiledHipKernel("source", "typed", (1, 1, 1), (64, 1, 1), arch="gfx950", signature=signature(),
                             runtime=runtime, compiler=lambda *a, **kw: b"code-object", **kwargs)


def test_typed_launch_preserves_int64_and_exact_float64():
    runtime = RecordingRuntime()
    executable = kernel(runtime)
    assert isinstance(executable, ExecutableKernel)
    executable(1, 1.0000000001)
    assert runtime.calls[-1] == ("launch", struct.pack("<qd", 1, 1.0000000001))
    executable.close()
    assert runtime.calls[-1] == ("unload", 1)
    with pytest.raises(RuntimeError, match="closed"):
        executable(1, 0.0)


def test_bad_typed_argument_fails_before_loading_or_launching():
    runtime = RecordingRuntime()
    with pytest.raises(OverflowError):
        kernel(runtime)(1 << 64, 0.0)
    assert runtime.calls == []


def test_failed_close_can_retry_without_losing_module():
    runtime = RecordingRuntime()
    executable = kernel(runtime)
    executable(1, 0.0)
    runtime.fail_unload = True
    with pytest.raises(RuntimeError, match="unload failed"):
        executable.close()
    runtime.fail_unload = False
    executable.close()
    executable.close()
    assert [c for c in runtime.calls if c[0] == "unload"] == [("unload", 1), ("unload", 1)]


def test_shared_cache_invalidates_when_user_header_changes(tmp_path):
    header = tmp_path / "value.h"
    header.write_text("#define VALUE 1\n")
    calls = []

    def compile_source(*args, **kwargs):
        calls.append(1)
        return b"object-" + header.read_bytes()

    def make():
        return CompiledHipKernel('#include "value.h"\nsource', "typed", (1, 1, 1), (64, 1, 1), arch="gfx950",
                                 cache_dir=tmp_path / "cache", include_paths=[tmp_path], signature=signature(),
                                 compiler=compile_source, toolchain_id="test-hipcc")

    first = make().code_object
    assert make().code_object == first and len(calls) == 1
    header.write_text("#define VALUE 2\n")
    assert make().code_object != first and len(calls) == 2


def test_runtime_component_honors_selected_device(monkeypatch):

    class Function:

        def __init__(self):
            self.values = []

        def __call__(self, device):
            self.values.append(device)
            return 0

    class Library:
        hipSetDevice = Function()

    library = Library()
    monkeypatch.setattr(HIPRuntime, "_load_library", staticmethod(lambda: library))
    monkeypatch.setattr(HIPRuntime, "_setup", lambda self: None)
    get_backend("amd/cdna4").runtime.create(device=3)
    assert library.hipSetDevice.values == [3]
