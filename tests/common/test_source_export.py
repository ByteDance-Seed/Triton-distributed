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
"""Selective publication must remove implementations, not merely disable imports."""
from pathlib import Path

import pytest

from little_kernel.backends import available_backends
from little_kernel.backends.registry import resolve_backend_spec
from tools.export_source import export_source


@pytest.mark.parametrize("backend_id", available_backends())
def test_selected_source_contains_only_the_requested_hardware(tmp_path, backend_id):
    output = tmp_path / "release"
    manifest = export_source(output, [backend_id], write_lock=False)
    assert manifest["backends"] == [backend_id]
    selected = resolve_backend_spec(backend_id)  # noqa: F841
    for other_id in available_backends():
        other = resolve_backend_spec(other_id)
        module_path = Path(*other.factory.split(":")[0].split(".")[:-1])
        exists = (output / "python" / module_path).is_dir()
        assert exists == (other_id == backend_id), (backend_id, other_id)
    assert (output / "python/little_kernel/core/type_system.py").is_file()
    assert (output / "python/little_kernel/backends/interfaces.py").is_file()
    for root in ("tests", "benchmarks", "examples", "assets", "tools"):
        for path in (output / root).glob("*"):
            if path.is_dir() and path.name not in {"common", "__pycache__"}:
                assert path.name == backend_id.split("/")[0]


def test_export_never_overwrites_existing_data(tmp_path):
    marker = tmp_path / "important.txt"
    marker.write_text("preserved")
    with pytest.raises(FileExistsError):
        export_source(tmp_path, [available_backends()[0]], write_lock=False)
    assert marker.read_text() == "preserved"
