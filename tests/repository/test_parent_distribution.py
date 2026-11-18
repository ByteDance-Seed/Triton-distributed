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
"""Validate the parent packaging entry point without building device extensions."""
from pathlib import Path
import runpy

import pytest
import setuptools

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def setup_metadata(monkeypatch):

    def capture(enabled):
        monkeypatch.chdir(ROOT / "python")
        monkeypatch.setenv("TRITON_BUILD_LITTLE_KERNEL", "ON" if enabled else "OFF")
        result = {}
        monkeypatch.setattr(setuptools, "setup", lambda **kwargs: result.update(kwargs))
        runpy.run_path(str(ROOT / "python/setup.py"), run_name="parent_packaging_check")
        return result

    return capture


def test_parent_discovers_every_bundled_backend_and_shared_component(setup_metadata):
    metadata = setup_metadata(True)
    packages = {name.replace("/", ".") for name in metadata["packages"]}
    assert {
        "little_kernel",
        "little_kernel.core",
        "little_kernel.language.intrin",
        "little_kernel.backends.common.gpu.passes",
        "little_kernel.backends.nv.common.runtime",
        "little_kernel.backends.nv.sm90",
        "little_kernel.backends.nv.sm100.passes.dataflow",
        "little_kernel.backends.amd.cdna4.runtime",
    } <= packages
    assert not any("__pycache__" in name for name in packages)
    assert "numpy" in metadata["install_requires"]


def test_parent_resource_patterns_include_descriptors_and_pipeline_header(setup_metadata):
    metadata = setup_metadata(True)
    root = ROOT / "python/little_kernel/backends"
    matched = {
        str(path.relative_to(root))
        for pattern in metadata["package_data"]["little_kernel.backends"]
        for path in root.glob(pattern)
        if path.is_file()
    }
    assert {
        "nv/sm90/backend.json",
        "nv/sm100/backend.json",
        "amd/cdna4/backend.json",
        "nv/sm100/runtime/include/role_pipeline.cuh",
    } <= matched


def test_disabling_bundled_compiler_preserves_parent_package_switch(setup_metadata):
    metadata = setup_metadata(False)
    assert not any(name.startswith("little_kernel") for name in metadata["packages"])
    assert "numpy" not in metadata["install_requires"]
