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
"""Host capability detection for suite admission (RF13).

Detection is read-only and never launches a GPU kernel, benchmark or job: it
only asks the CUDA Driver/Toolkit about what is present. All probes are
best-effort and never raise; missing capabilities are reported, not guessed.

Architecture gates are *exact family* gates, deliberately:

* ``sm90``  -- a Hopper-family device, packed arch in ``[90, 100)``. The SM90
  gate exercises the Hopper wgmma/TMA pipeline.
* ``sm100`` -- a Blackwell-family device, packed arch in ``[100, 110)``, **and**
  an NVCC toolchain new enough for the SM100 ``tcgen05`` pipeline (CUDA 12.8+).
  This is intentionally NOT ``arch >= 100``: an SM120 device is newer but does
  not implement the SM100 ``tcgen05`` pipeline, so ``sm120`` must NOT satisfy
  the ``sm100`` gate.
* ``cuda-driver`` -- distinct from the kernel-compatibility gates above: the
  real-Driver TMA/runtime-lifecycle checks only need a Driver exposing an
  SM90-or-newer device (``arch >= 90``); they need neither Torch nor a kernel
  launch.

Capability keys:

================  ============================================================
key               meaning
================  ============================================================
``cpu``           always available; pure CPU compiler/runtime checks
``nvcc``          an ``nvcc`` executable is on PATH (cross-compile host)
``nvcc-sm100``    NVCC toolchain new enough to target the SM100 tcgen05 ISA
                  (CUDA 12.8+ / major >= 13)
``torch``         PyTorch is importable (numerical/reference integration)
``gpu``           at least one CUDA GPU visible to the Driver
``cuda-driver``   libcuda initializes and exposes an SM90-or-newer device
                  (real Driver test; no Torch and no kernel launch)
``sm90``          visible Hopper-family device (exact family)
``sm100``         visible Blackwell-family device AND sufficient NVCC
``multigpu2``     at least two visible CUDA GPUs
``multigpu4``     at least four visible CUDA GPUs
``bench-deps``    benchmark-only dependencies importable (triton/tabulate)
``references``    external benchmark references (optional; never auto-required)
================  ============================================================
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import FrozenSet, Optional, Tuple

# Exact architecture families. A "newer" device is not automatically an
# "equivalent" device: SM120 is deliberately outside the SM100 tcgen05 family.
SM90_FAMILY = range(90, 100)
SM100_FAMILY = range(100, 110)

# The SM100 tcgen05/UMMA ISA is first supported by CUDA 12.8.
SM100_MIN_NVCC = (12, 8)


def _nvcc_supports_sm100(major: Optional[int], minor: Optional[int]) -> bool:
    if major is None or minor is None:
        return False
    if major > SM100_MIN_NVCC[0]:
        return True
    return (major, minor) >= SM100_MIN_NVCC


@dataclass(frozen=True)
class HostCaps:
    """Immutable snapshot of relevant host capabilities."""

    has_nvcc: bool = False
    nvcc_path: Optional[str] = None
    nvcc_version: Optional[str] = None
    nvcc_major: Optional[int] = None
    nvcc_minor: Optional[int] = None
    has_torch: bool = False
    has_gpu: bool = False
    gpu_count: int = 0
    sm_arch: Optional[int] = None  # packed integer of the *selected* device: 90,100,120
    gpu_name: Optional[str] = None
    driver_ok: bool = False
    selected_device_index: Optional[int] = None
    visible_devices: Optional[str] = None
    missing_modules: FrozenSet[str] = field(default_factory=frozenset)

    # -- requirement evaluation ------------------------------------------- #
    def satisfies(self, key: str) -> bool:
        if key == "cpu":
            return True
        if key == "nvcc":
            return self.has_nvcc
        if key == "nvcc-sm100":
            return self.has_nvcc and _nvcc_supports_sm100(self.nvcc_major, self.nvcc_minor)
        if key == "torch":
            return self.has_torch
        if key == "gpu":
            return self.has_gpu
        if key == "cuda-driver":
            # Real-Driver/TMA and runtime-lifecycle gate: SM90-or-newer Driver.
            # Deliberately a lower-bound, distinct from exact kernel families.
            return self.driver_ok and self.sm_arch is not None and self.sm_arch >= 90
        if key == "sm90":
            return self.has_gpu and self.sm_arch in SM90_FAMILY
        if key == "sm100":
            return (self.has_gpu and self.sm_arch in SM100_FAMILY
                    and _nvcc_supports_sm100(self.nvcc_major, self.nvcc_minor))
        if key == "multigpu2":
            return self.gpu_count >= 2
        if key == "multigpu4":
            return self.gpu_count >= 4
        if key == "bench-deps":
            return self.has_torch and "triton" not in self.missing_modules
        if key == "references":
            # External references are optional by policy; concrete path
            # detection happens inside the bench runner. Never auto-satisfied.
            return False
        raise KeyError(f"unknown capability key: {key!r}")

    def missing_of(self, keys: FrozenSet[str]) -> FrozenSet[str]:
        return frozenset(k for k in keys if not self.satisfies(k))

    def selected_device_label(self) -> str:
        """Human-readable selected-device metadata (reporting only)."""
        if not self.has_gpu:
            visible = self.visible_devices
            return f"no visible GPU (CUDA_VISIBLE_DEVICES={visible!r})"
        label = f"selected device cuda:{self.selected_device_index or 0}"
        if self.gpu_name:
            label += f" ({self.gpu_name})"
        label += f" sm_{self.sm_arch}" if self.sm_arch is not None else ""
        if self.gpu_count:
            label += f", {self.gpu_count} visible"
        if self.visible_devices:
            label += f", CUDA_VISIBLE_DEVICES={self.visible_devices!r}"
        return label

    @classmethod
    def for_testing(cls, **overrides) -> "HostCaps":
        """Construct an explicit snapshot (tests / injected runners)."""
        base = dict(
            has_nvcc=False,
            nvcc_path=None,
            nvcc_version=None,
            nvcc_major=None,
            nvcc_minor=None,
            has_torch=False,
            has_gpu=False,
            gpu_count=0,
            sm_arch=None,
            gpu_name=None,
            driver_ok=False,
            selected_device_index=None,
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            missing_modules=frozenset(),
        )
        base.update(overrides)
        return cls(**base)


def _module_importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, OSError):
        return False


def _detect_nvcc() -> Tuple[bool, Optional[str], Optional[str], Optional[int], Optional[int]]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return False, None, None, None, None
    version = major = minor = None
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=15, check=False)
        match = re.search(r"release (\d+)\.(\d+)", out.stdout + out.stderr)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            version = f"{major}.{minor}"
    except (OSError, subprocess.SubprocessError):
        pass
    return True, nvcc, version, major, minor


def _detect_torch(visible: Optional[str]):
    """Return torch capability snapshot, respecting visible/current device."""
    if not _module_importable("torch"):
        return False, False, 0, None, None, None
    try:
        import torch
    except Exception:
        return False, False, 0, None, None, None
    try:
        if not torch.cuda.is_available():
            return True, False, 0, None, None, None
        count = int(torch.cuda.device_count())
        # The *current/selected* device, not blindly device 0.
        idx = int(torch.cuda.current_device())
        major, minor = (int(x) for x in torch.cuda.get_device_capability(idx))
        name = str(torch.cuda.get_device_name(idx))
        return True, True, count, major * 10 + minor, name, idx
    except Exception:
        return True, False, 0, None, None, None


def _first_visible_ordinal(visible: Optional[str]) -> int:
    """Physical ordinal selected by CUDA_VISIBLE_DEVICES (0 when unset/empty).

    CUDA renumbers hidden devices, so when the variable is set the first listed
    ordinal is the physical device that becomes logical cuda:0. Invalid values
    fall back to 0 and let the Driver reject at launch time.
    """
    if not visible:
        return 0
    first = visible.split(",", 1)[0].strip()
    try:
        ordinal = int(first)
    except ValueError:
        return 0
    return max(ordinal, 0)


def _detect_driver(selected_ordinal: int = 0):
    """Probe libcuda directly (matches the real-Driver tests' own gate).

    Honors CUDA_VISIBLE_DEVICES by probing the first visible physical ordinal
    rather than blindly device 0; physical visibility translation is otherwise
    left to the Driver at launch time. No kernel/job is launched.
    """
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False, 0, None
    try:
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        if driver.cuInit(0) != 0:
            return False, 0, None
        count = ctypes.c_int()
        driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDeviceGetCount.restype = ctypes.c_int
        if driver.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value <= 0:
            return True, max(count.value, 0), None
        if selected_ordinal >= count.value:
            # Asked for a hidden/absent ordinal: effectively no usable GPU.
            return True, count.value, None
        major = ctypes.c_int()
        minor = ctypes.c_int()
        driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        driver.cuDeviceGetAttribute.restype = ctypes.c_int
        ok_major = driver.cuDeviceGetAttribute(ctypes.byref(major), 75, selected_ordinal) == 0
        ok_minor = driver.cuDeviceGetAttribute(ctypes.byref(minor), 76, selected_ordinal) == 0
        sm = major.value * 10 + minor.value if ok_major and ok_minor else None
        return True, count.value, sm
    except Exception:
        return True, 0, None


def detect_capabilities(*, use_driver_probe: bool = True) -> HostCaps:
    """Detect current host capabilities once, without side effects."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    (has_nvcc, nvcc_path, nvcc_version, nvcc_major, nvcc_minor) = _detect_nvcc()
    (has_torch, has_gpu, gpu_count, sm_arch, gpu_name, selected_idx) = _detect_torch(visible)

    driver_ok = False
    selected_ordinal = _first_visible_ordinal(visible)
    if use_driver_probe:
        driver_ok, driver_count, driver_sm = _detect_driver(selected_ordinal)
        # The raw Driver is authoritative for "a GPU exists" even without Torch.
        if driver_count > 0:
            has_gpu = True
            gpu_count = gpu_count or driver_count
            if sm_arch is None:
                sm_arch = driver_sm
            if selected_idx is None:
                # Logical cuda:0 corresponds to the first visible ordinal.
                selected_idx = 0
    else:
        driver_ok = has_gpu

    missing = frozenset(name for name in ("triton", "tabulate") if not _module_importable(name))
    return HostCaps(
        has_nvcc=has_nvcc,
        nvcc_path=nvcc_path,
        nvcc_version=nvcc_version,
        nvcc_major=nvcc_major,
        nvcc_minor=nvcc_minor,
        has_torch=has_torch,
        has_gpu=has_gpu,
        gpu_count=gpu_count,
        sm_arch=sm_arch,
        gpu_name=gpu_name,
        driver_ok=driver_ok,
        selected_device_index=selected_idx,
        visible_devices=visible,
        missing_modules=missing,
    )


@lru_cache(maxsize=1)
def cached_capabilities() -> HostCaps:
    """Process-wide cached snapshot (detection runs once per invocation)."""
    return detect_capabilities()
