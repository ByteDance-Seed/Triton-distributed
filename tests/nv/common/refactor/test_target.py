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
"""RF03 regressions for the CUDA target/device-capability service.

All tests are CPU-only: Torch and nvidia-smi discovery are exercised through
explicit fakes injected into ``sys.modules`` / ``target.subprocess``.
"""

import dataclasses
import subprocess
import sys
import types

import pytest

import little_kernel.backends.nv.common.runtime.compiler as compiler_mod
from little_kernel.backends.nv.common.runtime.compiler import compile_cuda, get_cuda_arch
from little_kernel.backends.nv.common.runtime.target import SOURCE_EXPLICIT, SOURCE_NVIDIA_SMI, SOURCE_TORCH, CudaTarget, TargetResolutionError, resolve_target


@dataclasses.dataclass
class _FakeProps:
    major: int
    minor: int
    name: str = "FakeGPU"


class _FakeCuda:
    """Minimal torch.cuda stand-in with a configurable current device."""

    def __init__(self, devices, current=0, available=True, fail=False):
        self._devices = dict(devices)
        self._current = current
        self._available = available
        self._fail = fail
        self.get_properties_calls = []

    def is_available(self):
        if self._fail:
            raise RuntimeError("simulated CUDA initialization failure")
        return self._available

    def device_count(self):
        return len(self._devices)

    def current_device(self):
        return self._current

    def get_device_properties(self, idx):
        self.get_properties_calls.append(idx)
        return _FakeProps(*self._devices[idx])


def _install_fake_torch(monkeypatch, fake_cuda):
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = fake_cuda

    class _Device:

        def __init__(self, spec):
            if isinstance(spec, str) and spec.startswith("cuda"):
                self.type = "cuda"
                self.index = int(spec.split(":")[1]) if ":" in spec else None
            else:
                self.type = "cpu"
                self.index = None

    torch_mod.device = _Device
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    return torch_mod


def _no_torch(monkeypatch):
    # Force ImportError for the lazy `import torch` regardless of host install,
    # while leaving every other import untouched for pytest internals.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("simulated missing torch")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    yield


# ---------------------------------------------------------------------------
# Canonical integer major/minor parsing (RF03 core defect)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "capability,expected",
    [((8, 6), "sm_86"), ((9, 0), "sm_90"), ((10, 0), "sm_100"), ((12, 0), "sm_120")],
)
def test_canonical_capability_integer_parse(capability, expected):
    target = CudaTarget(*capability)
    assert target.arch == expected
    assert target.capability == capability
    assert str(target) == expected


@pytest.mark.parametrize(
    "text,expected,major,minor",
    [
        ("8.6", "sm_86", 8, 6),
        ("9.0", "sm_90", 9, 0),
        ("10.0", "sm_100", 10, 0),
        ("12.0", "sm_120", 12, 0),
        ("sm_86", "sm_86", 8, 6),
        ("SM_100", "sm_100", 10, 0),
        ("sm120", "sm_120", 12, 0),
        ("compute_90", "sm_90", 9, 0),
    ],
)
def test_from_arch_text_forms(text, expected, major, minor):
    target = CudaTarget.from_arch(text)
    assert target.arch == expected
    assert target.capability == (major, minor)
    assert target.explicit
    assert target.source == SOURCE_EXPLICIT


def test_feature_suffix_is_deliberate():
    # Discovery-style dotted capability never gains an automatic "a".
    plain = CudaTarget.from_arch("9.0")
    assert plain.features == frozenset()
    assert plain.arch == "sm_90"
    assert plain.nvcc_arch_args() == ["-arch=sm_90"]

    suffixed = CudaTarget.from_arch("sm_90a")
    assert suffixed.features == frozenset({"a"})
    assert suffixed.arch == "sm_90a"
    assert suffixed.nvcc_arch_args() == ["-gencode", "arch=compute_90a,code=sm_90a"]

    assert CudaTarget.from_arch("SM_100A").arch == "sm_100a"


def test_f_family_target_preserved():
    """NVCC 13 Blackwell family suffix f is a legitimate explicit target."""
    for text in ("sm_100f", "SM100F", "compute_100f"):
        target = CudaTarget.from_arch(text)
        assert target.arch == "sm_100f"
        assert target.virtual_arch == "compute_100f"
        assert target.features == frozenset({"f"})
        assert target.capability == (10, 0)
        assert target.nvcc_arch_args() == ["-gencode", "arch=compute_100f,code=sm_100f"]


def test_a_and_f_are_distinct_explicit_families():
    assert CudaTarget.from_arch("sm_100a").arch == "sm_100a"
    assert CudaTarget.from_arch("sm_100f").arch == "sm_100f"
    # Unknown suffixes remain rejected.
    for bad in ("sm_100x", "sm_90af", "sm_90fa"):
        with pytest.raises(ValueError, match="feature suffix|CUDA arch"):
            CudaTarget.from_arch(bad)


@pytest.mark.parametrize("bad", ["sm_90b", "sm_9", "sm_", "hopper", "8.6.1", "9x", ""])
def test_invalid_explicit_arch_is_actionable(bad):
    with pytest.raises(ValueError, match="CUDA arch|architecture"):
        CudaTarget.from_arch(bad)


def test_target_is_immutable_and_value_based():
    a = CudaTarget(9, 0, frozenset({"a"}))
    b = CudaTarget.from_arch("sm_90a")
    assert a == b
    assert hash(a) == hash(b)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.major = 10  # type: ignore[misc]


def test_unknown_feature_suffix_rejected():
    with pytest.raises(ValueError, match="feature suffix"):
        CudaTarget(9, 0, features=frozenset({"b"}))


@pytest.mark.parametrize("major,minor", [(8.9, 0), (8.6, 6), (9.0, 0), (0, 0), (-1, 0)])
def test_nonintegral_or_invalid_capability_rejected(major, minor):
    with pytest.raises(ValueError, match="integer|invalid compute"):
        CudaTarget(major, minor)


@pytest.mark.parametrize("major,minor", [(True, 0), (9, False), (9.9, 0), ("nine", 0)])
def test_bool_and_nonintegral_capability_rejected(major, minor):
    with pytest.raises(ValueError, match="integer"):
        CudaTarget(major, minor)


@pytest.mark.parametrize("major,minor", [(8, 6), (9, 0), (10, 0), (12, 0), ("8", "6")])
def test_integer_and_digit_string_capability_supported(major, minor):
    target = CudaTarget(major, minor)
    assert isinstance(target.major, int) and isinstance(target.minor, int)
    assert target.capability == (int(major), int(minor))


def test_tuple_with_fractional_component_rejected():
    with pytest.raises(ValueError, match="integer"):
        CudaTarget.from_arch((8.9, 0))


# ---------------------------------------------------------------------------
# Current/selected device consistency
# ---------------------------------------------------------------------------


def test_torch_uses_current_device_not_zero(monkeypatch):
    fake = _FakeCuda({0: (8, 6), 1: (8, 9), 2: (10, 0)}, current=2)
    _install_fake_torch(monkeypatch, fake)

    target = resolve_target()
    # Automatic discovery applies the native architecture-specific policy.
    assert target.arch == "sm_100a"
    assert target.device == 2
    assert target.source == SOURCE_TORCH
    assert target.features == frozenset({"a"})
    assert fake.get_properties_calls == [2]


def test_explicit_device_selection_forms(monkeypatch):
    fake = _FakeCuda({0: (8, 6), 1: (9, 0), 2: (10, 0)}, current=2)
    torch_mod = _install_fake_torch(monkeypatch, fake)

    assert resolve_target(device=1).device == 1
    assert resolve_target(device="cuda:0").arch == "sm_86"
    assert resolve_target(device=torch_mod.device("cuda:2")).arch == "sm_100a"
    assert fake.get_properties_calls == [1, 0, 2]


def test_selected_device_out_of_range_is_actionable(monkeypatch):
    fake = _FakeCuda({0: (8, 6)}, current=0)
    _install_fake_torch(monkeypatch, fake)
    with pytest.raises(TargetResolutionError, match="device 3"):
        resolve_target(device=3)


def test_negative_device_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        resolve_target(device=-1)


# ---------------------------------------------------------------------------
# Explicit cross compilation never probes a Driver
# ---------------------------------------------------------------------------


def test_explicit_arch_does_not_probe(monkeypatch):
    # Torch import forced to fail and any subprocess call explodes; explicit
    # arch must still resolve without touching either discovery backend.
    def _boom(*args, **kwargs):
        raise AssertionError("nvidia-smi must not be invoked for explicit arch")

    monkeypatch.setattr("little_kernel.backends.nv.common.runtime.target.subprocess.run", _boom)
    _no_torch(monkeypatch)

    for text in ("sm_86", "9.0", "sm_100a"):
        target = resolve_target(arch=text)
        assert target.explicit
        assert target.source == SOURCE_EXPLICIT


def test_explicit_arch_preferred_over_bad_device(monkeypatch):
    # Even an unusable auto-discovery environment must not matter when the
    # caller names the target.
    _no_torch(monkeypatch)
    monkeypatch.setattr(
        "little_kernel.backends.nv.common.runtime.target.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no nvidia-smi")),
    )
    target = resolve_target(arch="sm_120")
    assert target.arch == "sm_120"


# ---------------------------------------------------------------------------
# nvidia-smi fallback: multi-line data and visible devices
# ---------------------------------------------------------------------------

SMI_TWO_GPUS = ("0, GPU-aaaa-aaaa, 8.6\n"
                "1, GPU-bbbb-bbbb, 9.0\n")


def _fake_completed(stdout):
    return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout=stdout, stderr="")


def test_nvidia_smi_multiline_selects_device(monkeypatch):
    _no_torch(monkeypatch)
    monkeypatch.setattr(
        "little_kernel.backends.nv.common.runtime.target.subprocess.run",
        lambda *a, **k: _fake_completed(SMI_TWO_GPUS),
    )
    first = resolve_target()
    second = resolve_target(device=1)
    assert first.arch == "sm_86"
    assert first.device == 0
    assert first.source == SOURCE_NVIDIA_SMI
    assert first.features == frozenset()
    # Native Hopper policy for automatic discovery.
    assert second.arch == "sm_90a"
    assert second.device == 1


def test_nvidia_smi_integer_minor_regression(monkeypatch):
    # The old parser produced sm_80/sm_10/sm_12 from this exact data.
    _no_torch(monkeypatch)
    out = "0, GPU-x, 8.6\n1, GPU-y, 10.0\n2, GPU-z, 12.0\n"
    monkeypatch.setattr(
        "little_kernel.backends.nv.common.runtime.target.subprocess.run",
        lambda *a, **k: _fake_completed(out),
    )
    assert resolve_target().arch == "sm_86"
    assert resolve_target(device=1).arch == "sm_100a"
    assert resolve_target(device=2).arch == "sm_120a"


def test_automatic_native_policy_versus_explicit_portable(monkeypatch):
    """RF03 review: native Hopper default is sm_90a; explicit sm_90 stays portable."""
    fake = _FakeCuda({0: (9, 0)}, current=0)
    _install_fake_torch(monkeypatch, fake)

    discovered = resolve_target()
    assert discovered.arch == "sm_90a"
    assert discovered.features == frozenset({"a"})
    assert discovered.nvcc_arch_args() == ["-gencode", "arch=compute_90a,code=sm_90a"]

    # Explicit portable spelling is honored verbatim and never probes discovery.
    monkeypatch.setattr(
        fake,
        "get_device_properties",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    explicit = resolve_target(arch="sm_90")
    assert explicit.arch == "sm_90"
    assert explicit.features == frozenset()
    assert explicit.nvcc_arch_args() == ["-arch=sm_90"]

    explicit_dotted = resolve_target(arch="9.0")
    assert explicit_dotted.arch == "sm_90"

    # Pre-major-9 devices are portable even when discovered.
    fake2 = _FakeCuda({0: (8, 6)}, current=0)
    _install_fake_torch(monkeypatch, fake2)
    assert resolve_target().arch == "sm_86"


def test_visible_devices_uuid_mapping(monkeypatch):
    _no_torch(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb-bbbb")
    monkeypatch.setattr(
        "little_kernel.backends.nv.common.runtime.target.subprocess.run",
        lambda *a, **k: _fake_completed(SMI_TWO_GPUS),
    )
    # Only the second physical GPU is visible: logical 0 resolves to native sm_90a.
    target = resolve_target()
    assert target.arch == "sm_90a"
    assert target.device == 0


def test_empty_visible_devices_is_actionable(monkeypatch):
    _no_torch(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        "little_kernel.backends.nv.common.runtime.target.subprocess.run",
        lambda *a, **k: _fake_completed(SMI_TWO_GPUS),
    )
    with pytest.raises(TargetResolutionError, match="CUDA_VISIBLE_DEVICES"):
        resolve_target()


def test_malformed_smi_output_is_actionable(monkeypatch):
    _no_torch(monkeypatch)
    monkeypatch.setattr(
        "little_kernel.backends.nv.common.runtime.target.subprocess.run",
        lambda *a, **k: _fake_completed("0, GPU-x, not-a-cap\n"),
    )
    with pytest.raises(TargetResolutionError, match="compute capability"):
        resolve_target()


def test_failed_discovery_raises_instead_of_sm90a(monkeypatch):
    _no_torch(monkeypatch)

    def _missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr("little_kernel.backends.nv.common.runtime.target.subprocess.run", _missing)
    with pytest.raises(TargetResolutionError, match="arch=") as exc:
        resolve_target()
    assert "sm_90a" not in str(exc.value)


def test_torch_failure_falls_through_to_smi(monkeypatch):
    fake = _FakeCuda({}, available=False)
    _install_fake_torch(monkeypatch, fake)
    monkeypatch.setattr(
        "little_kernel.backends.nv.common.runtime.target.subprocess.run",
        lambda *a, **k: _fake_completed("0, GPU-x, 12.0\n"),
    )
    target = resolve_target()
    assert target.arch == "sm_120a"
    assert target.source == SOURCE_NVIDIA_SMI


# ---------------------------------------------------------------------------
# compiler.py compatibility wrapper and compile_cuda integration
# ---------------------------------------------------------------------------


def test_get_cuda_arch_wrapper(monkeypatch):
    fake = _FakeCuda({0: (8, 6), 1: (9, 0)}, current=1)
    _install_fake_torch(monkeypatch, fake)
    # Native Hopper default preserved (WGMMA requires sm_90a).
    assert get_cuda_arch() == "sm_90a"
    assert get_cuda_arch(device=0) == "sm_86"


def test_get_cuda_arch_no_silent_default(monkeypatch):
    _no_torch(monkeypatch)

    def _missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr("little_kernel.backends.nv.common.runtime.target.subprocess.run", _missing)
    with pytest.raises(TargetResolutionError):
        get_cuda_arch()


def _make_fake_nvcc(monkeypatch, captured):
    monkeypatch.setattr(compiler_mod, "get_nvcc_path", lambda: "nvcc")

    def _fake_run(cmd, **kwargs):
        captured.append(cmd)
        # Emulate nvcc producing its -o output file.
        out_idx = cmd.index("-o") + 1
        with open(cmd[out_idx], "wb") as f:
            f.write(b"CUBIN-BYTES")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(compiler_mod.subprocess, "run", _fake_run)


def test_compile_cuda_uses_resolved_target_args(monkeypatch, tmp_path):
    captured = []
    _make_fake_nvcc(monkeypatch, captured)

    binary = compile_cuda(
        "__global__ void k() {}",
        "k",
        arch="sm_90a",
        cache_dir=str(tmp_path),
    )
    assert binary == b"CUBIN-BYTES"
    cmd = captured[0]
    assert "-gencode" in cmd
    assert "arch=compute_90a,code=sm_90a" in cmd


def test_compile_cuda_explicit_arch_no_probe(monkeypatch, tmp_path):
    captured = []
    _make_fake_nvcc(monkeypatch, captured)
    # No torch and a real subprocess.run would fail; explicit arch must win.
    _no_torch(monkeypatch)
    compile_cuda("__global__ void k() {}", "k", arch="sm_100", cache_dir=str(tmp_path))
    cmd = captured[0]
    assert "-arch=sm_100" in cmd


def test_compile_cuda_f_family_gencode(monkeypatch, tmp_path):
    """NVCC 13 explicit family target sm_100f must be preserved verbatim."""
    captured = []
    _make_fake_nvcc(monkeypatch, captured)
    compile_cuda("__global__ void k() {}", "k", arch="sm_100f", cache_dir=str(tmp_path))
    cmd = captured[0]
    assert "-gencode" in cmd
    assert "arch=compute_100f,code=sm_100f" in cmd
    assert not any(arg.startswith("-arch=") for arg in cmd)


def test_compile_cuda_bad_format_rejected_before_probe(monkeypatch):
    with pytest.raises(ValueError, match="target_format"):
        compile_cuda("code", "k", arch="sm_90", target_format="elf")  # type: ignore[arg-type]


def test_compile_cuda_selected_device_discovery(monkeypatch, tmp_path):
    fake = _FakeCuda({0: (8, 6), 1: (10, 0)}, current=0)
    _install_fake_torch(monkeypatch, fake)
    captured = []
    _make_fake_nvcc(monkeypatch, captured)
    compile_cuda("__global__ void k() {}", "k", device=1, cache_dir=str(tmp_path))
    cmd = captured[0]
    # Automatic Blackwell discovery selects native architecture-specific codegen.
    assert "arch=compute_100a,code=sm_100a" in cmd
