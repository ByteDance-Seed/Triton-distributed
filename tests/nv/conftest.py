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
"""Shared pytest fixtures and hardware-capability marks.

This module is discovered by pytest from the canonical test roots
(``tests/unit``, ``tests/integration``, ``tests/nv/sm100/dataflow``,
``tests/nv/sm90/distributed``). Test modules use the marks/fixtures defined here rather
than importing a helper from inside the production package.

Capability policy (RF11/RF12):

* CPU collection never imports Torch at module import time. Torch is probed
  lazily through :func:`torch_available`, and only selected hardware tests
  import/use it behind the marks below.
* A missing optional dependency or absent device is a *skip* with a stated
  capability; a genuine compiler/import failure is never converted to a skip.
"""

from __future__ import annotations

import importlib.util

import pytest


def torch_available() -> bool:
    """Return whether an importable Torch is present.

    Uses :func:`importlib.util.find_spec` so this check itself never imports
    Torch into the CPU-only test process.
    """
    return importlib.util.find_spec("torch") is not None


def cuda_available() -> bool:
    """Return whether Torch reports a usable CUDA device. Never hides errors."""
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _is_rocm():
    if not torch_available():
        return False
    import torch
    return torch.version.hip is not None


def _get_cuda_capability():
    """Return ``(major, minor)`` CUDA compute capability, or ``None``."""
    if not cuda_available() or _is_rocm():
        return None
    import torch

    return torch.cuda.get_device_capability(0)


def _get_sm_arch():
    """Return the SM architecture integer (e.g. 90, 100), or ``None``."""
    cap = _get_cuda_capability()
    if cap is None:
        return None
    return cap[0] * 10 + cap[1]


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

requires_torch = pytest.mark.skipif(
    not torch_available(),
    reason="Requires the optional Torch dependency",
)

requires_cuda = pytest.mark.skipif(
    _get_cuda_capability() is None,
    reason="No CUDA GPU available",
)

requires_sm90 = pytest.mark.skipif(
    _get_sm_arch() is None or _get_sm_arch() < 90,
    reason="Requires Hopper (SM90+) GPU",
)

requires_sm100 = pytest.mark.skipif(
    _get_sm_arch() is None or _get_sm_arch() < 100,
    reason="Requires Blackwell (SM100+) GPU",
)

# FlashComm multi-GPU / IPC design programs need Torch and a CUDA device.
requires_multi_gpu = pytest.mark.skipif(
    _get_cuda_capability() is None,
    reason="Requires a CUDA GPU (and Torch) for the FlashComm distributed program",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cuda_device():
    """Provide a CUDA device, skip if unavailable."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("Torch is not installed")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    return torch.device("cuda:0")


@pytest.fixture
def sm_arch():
    """Return the SM architecture of the first CUDA device."""
    arch = _get_sm_arch()
    if arch is None:
        pytest.skip("No CUDA GPU available")
    return arch


@pytest.fixture
def gpu_props():
    """Return torch.cuda device properties for device 0."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("Torch is not installed")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    return torch.cuda.get_device_properties(0)
