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
"""CUDA target / device-capability resolution service (RF03).

This module is the single owner of the question "which compute architecture are
we compiling for?". Automatic discovery always inspects exactly one *selected*
device (the current Torch device by default, consistently with the Driver
runtime), and major/minor compute capabilities are parsed as integers, so
``8.6`` -> :class:`CudaTarget` ``sm_86`` rather than ``sm_80``.

Design rules enforced here:

* An explicit ``arch=`` is a cross-compilation request: it is parsed and
  canonicalized but it never probes Torch, ``nvidia-smi`` or a Driver.
* Failed discovery raises :class:`TargetResolutionError` with actionable
  context; it never silently guesses an architecture (in particular it never
  falls back to ``sm_90a``).
* Feature suffixes are deliberate. Explicit spellings are honored verbatim
  (portable ``sm_90`` stays ``sm_90``; nvcc family variants such as ``sm_100f``
  are preserved). Automatic discovery instead selects the device's native
  architecture-specific policy (Hopper and later resolve to ``sm_90a``/
  ``sm_100a`` so native instructions such as ``wgmma.fence`` are available),
  but discovery failure never guesses a target.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Optional, Tuple, Union

# CUDA arch feature suffixes nvcc defines for sm_/compute_ spellings:
#   "a" - architecture-specific features (sm_90a); "f" - the Blackwell
#   family variant (sm_100f, supported by NVCC 13). They are deliberately
#   explicit: automatic native discovery selects the native architecture-
#   specific policy, but an explicit spelling is honored verbatim.
_KNOWN_FEATURES: FrozenSet[str] = frozenset({"a", "f"})

_PACKED_ARCH_RE = re.compile(r"^(?:sm|compute)_?(\d+)([a-z]*)$")
_DOTTED_CAP_RE = re.compile(r"^(\d+)\.(\d+)$")

# Discovery backends, recorded on the resolved target for diagnostics/cache keys.
SOURCE_EXPLICIT = "explicit"
SOURCE_TORCH = "torch"
SOURCE_NVIDIA_SMI = "nvidia-smi"


class TargetResolutionError(RuntimeError):
    """Raised when an automatic target cannot be established.

    The message always explains what was attempted and how to fix the situation
    (typically: pass ``arch=`` explicitly for cross compilation).
    """


def _coerce_capability_component(value: Any, what: str) -> int:
    """Coerce one compute-capability component without silent truncation.

    Integers and parsed digit strings (as produced by the arch regexes) are
    accepted. Booleans, fractional floats and any other non-integral value are
    rejected rather than rounded/truncated (``8.9`` must not become ``8``).
    """
    if isinstance(value, bool):
        raise ValueError(f"compute capability {what} must be an integer, got bool {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        return int(value.strip())
    raise ValueError(f"compute capability {what} must be an integer (or digit string), got {value!r}")


def _validate_capability(major: Any, minor: Any) -> Tuple[int, int]:
    major_i = _coerce_capability_component(major, "major")
    minor_i = _coerce_capability_component(minor, "minor")
    if major_i < 1:
        raise ValueError(f"invalid compute capability major version {major_i}; expected an integer >= 1")
    # Real CUDA compute capabilities use a single-digit minor (8.6, 9.0, 10.0, 12.0).
    if not 0 <= minor_i <= 9:
        raise ValueError(f"invalid compute capability minor version {minor_i}; expected a single digit in [0, 9] "
                         f"(got capability {major_i}.{minor_i})")
    return major_i, minor_i


def _native_features(major: int) -> FrozenSet[str]:
    """Native architecture-specific feature policy for a *discovered* device.

    Preserves the project's valid native default: Hopper (major 9) and later
    architectures are natively compiled with the architecture-specific "a"
    feature set (WGMMA instructions such as wgmma.fence require sm_90a and do
    not exist in portable sm_90). Older architectures use the portable spelling.
    This applies only to automatic discovery; an explicitly requested
    ``sm_90`` stays portable.
    """
    return frozenset({"a"}) if major >= 9 else frozenset()


def _canonical_suffix(features: FrozenSet[str]) -> str:
    return "".join(sorted(features))


@dataclass(frozen=True, slots=True)
class CudaTarget:
    """An immutable CUDA compilation target.

    Attributes
    ----------
    major, minor:
        Integer compute-capability components (e.g. 8, 6).
    features:
        Frozen set of architecture feature suffixes (nvcc ``"a"`` and the
        NVCC-13 Blackwell family ``"f"``). An explicit spelling is honored
        verbatim; a discovered target carries the device's native
        architecture-specific policy (``"a"`` on major >= 9).
    device:
        Logical index of the selected device this target was resolved from, or
        ``None`` for an explicit cross-compilation target.
    source:
        How the target was established (``"explicit"``, ``"torch"`` or
        ``"nvidia-smi"``).
    """

    major: int
    minor: int
    features: FrozenSet[str] = field(default_factory=frozenset)
    device: Optional[int] = None
    source: str = SOURCE_EXPLICIT

    def __post_init__(self) -> None:
        major, minor = _validate_capability(self.major, self.minor)
        features = frozenset(self.features)
        if len(features) > 1:
            raise ValueError(f"CUDA target feature suffixes {sorted(features)} are mutually exclusive; "
                             f"choose one of {sorted(_KNOWN_FEATURES)}")
        unknown = features - _KNOWN_FEATURES
        if unknown:
            raise ValueError(f"unknown CUDA arch feature suffix(es) {sorted(unknown)}; "
                             f"known features are {sorted(_KNOWN_FEATURES)}")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("target source must be a non-empty string")
        # frozen dataclass: bypass __setattr__ to store normalized values.
        object.__setattr__(self, "major", major)
        object.__setattr__(self, "minor", minor)
        object.__setattr__(self, "features", features)

    @property
    def capability(self) -> Tuple[int, int]:
        """Compute capability as an integer ``(major, minor)`` tuple."""
        return (self.major, self.minor)

    @property
    def arch(self) -> str:
        """Canonical nvcc architecture spelling, e.g. ``sm_86`` / ``sm_90a``."""
        return f"sm_{self.major}{self.minor}{_canonical_suffix(self.features)}"

    @property
    def virtual_arch(self) -> str:
        """Canonical nvcc virtual architecture, e.g. ``compute_90a``."""
        return f"compute_{self.major}{self.minor}{_canonical_suffix(self.features)}"

    @property
    def explicit(self) -> bool:
        """True when the caller named the architecture instead of discovery."""
        return self.source == SOURCE_EXPLICIT

    def nvcc_arch_args(self) -> list:
        """nvcc command-line arguments selecting this architecture.

        Feature-suffixed targets need an explicit ``-gencode`` pairing of the
        matching virtual and real architectures; plain targets use ``-arch``.
        """
        if self.features:
            return ["-gencode", f"arch={self.virtual_arch},code={self.arch}"]
        return [f"-arch={self.arch}"]

    def __str__(self) -> str:
        return self.arch

    @classmethod
    def from_arch(
        cls,
        arch: Union[str, "CudaTarget", Tuple[int, int]],
        *,
        device: Optional[int] = None,
    ) -> "CudaTarget":
        """Parse an explicit architecture specification.

        Accepts packed nvcc spellings (``sm_90a``, ``SM100``, ``compute_120``),
        dotted compute capabilities (``"9.0"``) and ``(major, minor)`` tuples.
        No device/Driver is probed.
        """
        if isinstance(arch, CudaTarget):
            return arch
        if isinstance(arch, tuple):
            if len(arch) != 2:
                raise ValueError(f"explicit CUDA arch tuple must be (major, minor), got {arch!r}")
            major, minor = _validate_capability(arch[0], arch[1])
            return cls(major, minor, device=device, source=SOURCE_EXPLICIT)
        if not isinstance(arch, str):
            raise ValueError(f"explicit CUDA arch must be a string like 'sm_90a' or '9.0', got {arch!r}")
        text = arch.strip().lower()
        if not text:
            raise ValueError("explicit CUDA arch string is empty")

        dotted = _DOTTED_CAP_RE.match(text)
        if dotted is not None:
            major, minor = _validate_capability(dotted.group(1), dotted.group(2))
            return cls(major, minor, device=device, source=SOURCE_EXPLICIT)

        packed = _PACKED_ARCH_RE.match(text)
        if packed is None:
            raise ValueError(f"unrecognized CUDA arch {arch!r}; expected a form like 'sm_90a', 'sm_100', "
                             f"'compute_120' or dotted capability '9.0'")
        digits, suffix_letters = packed.group(1), packed.group(2)
        if len(digits) < 2:
            raise ValueError(f"CUDA arch {arch!r} is missing its minor digit; packed spellings concatenate "
                             f"major and single-digit minor, e.g. '86' for capability 8.6 and '100' for 10.0")
        features = frozenset(suffix_letters)
        if len(features) > 1 or len(suffix_letters) > 1:
            raise ValueError(f"CUDA arch {arch!r} combines incompatible feature suffixes {suffix_letters!r}; "
                             f"choose one of {sorted(_KNOWN_FEATURES)}")
        unknown = features - _KNOWN_FEATURES
        if unknown:
            raise ValueError(f"CUDA arch {arch!r} uses unknown feature suffix(es) {sorted(unknown)}; "
                             f"known features are {sorted(_KNOWN_FEATURES)}")
        major, minor = _validate_capability(digits[:-1], digits[-1])
        return cls(major, minor, features, device=device, source=SOURCE_EXPLICIT)


def _coerce_device_index(device: Any) -> Optional[int]:
    """Normalize a device selector to a logical non-negative integer index.

    Accepts ``None`` (meaning "current device"), an int, ``"cuda:N"`` /
    ``"gpu:N"`` strings and Torch-device-like objects.
    """
    if device is None:
        return None
    if isinstance(device, bool):
        raise ValueError(f"device index must be an int or 'cuda:N', got {device!r}")
    if isinstance(device, int):
        idx = device
    elif isinstance(device, str):
        text = device.strip().lower()
        if text in ("cuda", "gpu"):
            return None
        match = re.match(r"^(?:cuda|gpu):(\d+)$", text)
        if match is None:
            if text.isdigit():
                idx = int(text)
            else:
                raise ValueError(f"device selector must be an int or 'cuda:N', got {device!r}")
        else:
            idx = int(match.group(1))
    else:
        dev_type = getattr(device, "type", None)
        dev_index = getattr(device, "index", None)
        if dev_type not in (None, "cuda"):
            raise ValueError(f"device selector must refer to a CUDA device, got {device!r}")
        if dev_index is None:
            return None
        idx = dev_index
    if idx < 0:
        raise ValueError(f"device index must be non-negative, got {idx}")
    return idx


def _discover_torch(device: Optional[int], reasons: list) -> Optional[CudaTarget]:
    """Resolve from an installed, initialized Torch without touching device 0."""
    try:
        import torch
    except ImportError:
        reasons.append("PyTorch is not installed")
        return None
    try:
        if not torch.cuda.is_available():
            reasons.append("PyTorch reports torch.cuda.is_available() == False")
            return None
        count = int(torch.cuda.device_count())
        if device is None:
            idx = int(torch.cuda.current_device())
        else:
            idx = device
            if count and idx >= count:
                raise TargetResolutionError(f"requested CUDA device {idx} but only {count} visible device(s) are "
                                            f"available to PyTorch; check CUDA_VISIBLE_DEVICES or pass arch= for "
                                            f"cross compilation")
        props = torch.cuda.get_device_properties(idx)
        major, minor = _validate_capability(props.major, props.minor)
        return CudaTarget(major, minor, _native_features(major), device=idx, source=SOURCE_TORCH)
    except TargetResolutionError:
        raise
    except Exception as exc:  # driver/runtime init failures: try the next source
        reasons.append(f"PyTorch discovery failed: {type(exc).__name__}: {exc}")
        return None


def _visible_entries() -> Optional[list]:
    """Parse CUDA_VISIBLE_DEVICES into ordered selector entries.

    Returns ``None`` when the variable is unset (all physical devices visible),
    and raises when it explicitly hides every device.
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return None
    raw = os.environ["CUDA_VISIBLE_DEVICES"]
    if raw is None or raw.strip() == "":
        raise TargetResolutionError("CUDA_VISIBLE_DEVICES is set to an empty value, so no CUDA device is visible; "
                                    "unset it, name a device, or pass arch= explicitly for cross compilation")
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not entries:
        raise TargetResolutionError("CUDA_VISIBLE_DEVICES names no usable devices; check its value or pass arch= "
                                    "explicitly for cross compilation")
    return entries


def _parse_smi_row(line: str) -> Tuple[str, str, Tuple[int, int]]:
    """Parse one ``index,gpu_uuid,compute_cap`` row from nvidia-smi."""
    fields = [part.strip() for part in line.split(",")]
    if len(fields) != 3:
        raise TargetResolutionError(f"unexpected nvidia-smi output line {line!r}; expected "
                                    f"'<index>, <gpu_uuid>, <major.minor>'")
    phys_index, gpu_uuid, cap_text = fields
    cap_match = _DOTTED_CAP_RE.match(cap_text)
    if cap_match is None:
        raise TargetResolutionError(f"nvidia-smi reported unparseable compute capability {cap_text!r} for device "
                                    f"{phys_index}; pass arch= explicitly (e.g. arch='sm_90a') for cross compilation")
    major, minor = _validate_capability(cap_match.group(1), cap_match.group(2))
    return phys_index, gpu_uuid, (major, minor)


def _discover_nvidia_smi(device: Optional[int], reasons: list) -> Optional[CudaTarget]:
    """Resolve from nvidia-smi, honoring multi-GPU output and visible devices."""
    # An explicitly empty CUDA_VISIBLE_DEVICES hides every device: fail fast
    # without probing the tool.
    entries = _visible_entries()
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,gpu_uuid,compute_cap",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        reasons.append("nvidia-smi was not found on PATH")
        return None
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        reasons.append(f"nvidia-smi exited with status {exc.returncode}: {stderr or '<no stderr>'}")
        return None
    except OSError as exc:  # pragma: no cover - defensive
        reasons.append(f"nvidia-smi could not be executed: {exc}")
        return None

    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            rows.append(_parse_smi_row(line))
    if not rows:
        reasons.append("nvidia-smi returned no GPU rows")
        return None

    selected_logical = 0 if device is None else device
    if entries is None:
        if selected_logical >= len(rows):
            raise TargetResolutionError(f"requested CUDA device {selected_logical} but nvidia-smi lists only "
                                        f"{len(rows)} device(s); check CUDA_VISIBLE_DEVICES or pass arch= for "
                                        f"cross compilation")
        _, _, (major, minor) = rows[selected_logical]
    else:
        if selected_logical >= len(entries):
            raise TargetResolutionError(f"requested logical CUDA device {selected_logical} but CUDA_VISIBLE_DEVICES="
                                        f"{os.environ['CUDA_VISIBLE_DEVICES']!r} exposes only {len(entries)} device(s)")
        wanted = entries[selected_logical]
        match = None
        if wanted.isdigit():
            match = next((row for row in rows if row[0] == wanted), None)
        else:
            wanted_uuid = wanted.upper()
            match = next((row for row in rows if row[1].upper() == wanted_uuid), None)
        if match is None:
            raise TargetResolutionError(f"CUDA_VISIBLE_DEVICES entry {wanted!r} does not match any GPU listed by "
                                        f"nvidia-smi ({len(rows)} GPU(s) listed); pass arch= explicitly for cross "
                                        f"compilation")
        _, _, (major, minor) = match

    return CudaTarget(
        major,
        minor,
        _native_features(major),
        device=selected_logical,
        source=SOURCE_NVIDIA_SMI,
    )


def resolve_target(
    arch: Union[str, CudaTarget, Tuple[int, int], None] = None,
    device: Any = None,
) -> CudaTarget:
    """Resolve exactly one CUDA compilation target.

    Parameters
    ----------
    arch:
        Explicit target (``"sm_90a"``, ``"9.0"``, ``(8, 6)`` ...). When given,
        this is cross compilation: the value is canonicalized and **no** Driver,
        Torch or nvidia-smi probe is performed.
    device:
        Selected device for automatic discovery. ``None`` means the current
        Torch device; an int / ``"cuda:N"`` / ``torch.device`` selects a logical
        visible device consistently with the runtime.

    Returns
    -------
    CudaTarget
        The resolved target with a canonical :attr:`CudaTarget.arch` spelling.

    Raises
    ------
    ValueError
        For malformed explicit architecture/device arguments.
    TargetResolutionError
        When discovery is required but no capability can be established; the
        message lists what was attempted and how to proceed.
    """
    selected = _coerce_device_index(device)

    if arch is not None:
        # Explicit cross compilation: never probe a Driver for capability.
        return CudaTarget.from_arch(arch, device=selected)

    reasons: list = []
    target = _discover_torch(selected, reasons)
    if target is None:
        target = _discover_nvidia_smi(selected, reasons)
    if target is not None:
        return target

    detail = "; ".join(reasons) if reasons else "no discovery backend was available"
    raise TargetResolutionError("Unable to determine the CUDA target automatically ("
                                f"{detail}). Pass an explicit arch= (for example arch='sm_80' or arch='sm_100', "
                                "matching your hardware/toolchain) for cross compilation, or fix the selected "
                                "device/CUDA driver installation.")
