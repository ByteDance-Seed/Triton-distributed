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
"""Shared pytest fixtures for LittleKernel tests."""

import pytest
import torch


def _get_cuda_capability():
    """Return (major, minor) CUDA compute capability, or None if no CUDA GPU."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(0)


def _get_sm_arch():
    """Return SM architecture integer (e.g. 90, 100), or None."""
    cap = _get_cuda_capability()
    if cap is None:
        return None
    return cap[0] * 10 + cap[1]


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cuda_device():
    """Provide a CUDA device, skip if unavailable."""
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
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    return torch.cuda.get_device_properties(0)
