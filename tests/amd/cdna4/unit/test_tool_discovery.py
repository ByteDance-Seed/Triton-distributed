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
"""Tool discovery follows caller configuration instead of a host installation."""
from types import SimpleNamespace

import pytest

from little_kernel.backends.amd.cdna4.runtime import compiler
from little_kernel.backends.amd.cdna4.runtime.hip_runtime import HIPRuntime


def test_hipcc_explicit_override_and_configured_sdk(monkeypatch, tmp_path):
    monkeypatch.setenv('HIPCC', 'caller-selected-compiler')
    assert compiler.get_hipcc_path() == 'caller-selected-compiler'
    monkeypatch.delenv('HIPCC')
    monkeypatch.delenv('ROCM_PATH', raising=False)
    monkeypatch.setattr(compiler.shutil, 'which', lambda _: None)
    root = tmp_path / 'sdk'
    (root / 'bin').mkdir(parents=True)
    tool = root / 'bin/hipcc'
    tool.write_text('tool placeholder')
    monkeypatch.setenv('HIP_PATH', str(root))
    assert compiler.get_hipcc_path() == str(tool)
    monkeypatch.delenv('HIP_PATH')
    with pytest.raises(RuntimeError, match='HIPCC or ROCM_PATH'):
        compiler.get_hipcc_path()


def test_hip_library_uses_caller_root_then_loader(monkeypatch, tmp_path):
    from little_kernel.backends.amd.cdna4.runtime import hip_runtime
    monkeypatch.delenv('LK_LIBAMDHIP64', raising=False)
    monkeypatch.delenv('ROCM_PATH', raising=False)
    monkeypatch.setenv('HIP_PATH', str(tmp_path / 'sdk'))
    seen = []
    library = SimpleNamespace()

    def load(path, **kwargs):
        seen.append(path)
        if path.endswith('libamdhip64.so'): return library
        raise OSError('unavailable')

    monkeypatch.setattr(hip_runtime.ctypes, 'CDLL', load)
    assert HIPRuntime._load_library() is library
    assert seen == [str(tmp_path / 'sdk/lib/libamdhip64.so')]
