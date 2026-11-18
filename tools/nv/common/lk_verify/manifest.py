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
"""Authoritative suite manifest (RF13).

Every maintained integration module belongs to exactly one documented suite.
The old ``tools/nv/sm100/verify_dataflow.sh`` only ran the SM100 GEMM path and silently omitted
vector-memory, math-intrinsic, TMA-Driver and SM90 coverage; this manifest makes
kinds, required capabilities and paths explicit.

Canonical path convention:

* ``tests/unit/``                                         -- CPU unit suite
* ``tests/integration/<old-basename>``                    -- driver/TMA, vector,
  math, runtime lifecycle, SM90/SM100 GEMM integration
* ``tests/nv/sm100/dataflow/<old-role-test-basename>``             -- role/dataflow
  validation and the generated-kernel gate
* ``tests/nv/sm90/distributed/<old-design-test-basename>``        -- multi-GPU
* ``examples/nv/sm100/{gemm_sm100,fa4_attention,...}``    -- role kernels
* ``benchmarks/dataflow/``, ``benchmarks/gemm/``                                -- benchmarks.nv.common

``paths`` are the **canonical** targets. ``legacy_paths`` retain historical
locations for tooling that reads older checkouts;
:meth:`CaseSpec.select_paths` chooses the set that exists on disk and the
runner fails (never silently skips) if neither set is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, List, Sequence, Tuple

VALID_KINDS = ("pytest", "command", "compile")


@dataclass(frozen=True)
class CaseSpec:
    """One runnable verification case.

    Attributes
    ----------
    case_id:
        Stable identifier used in reports and ``--only`` selection.
    kind:
        ``pytest`` / ``command`` / ``compile`` (see docs/verification.md).
    paths:
        Canonical repository-relative targets after the layout migration.
    legacy_paths:
        Historical paths in older checkouts. Never preferred over canonical paths.
    required / optional:
        Capability keys. A missing **required** key yields UNAVAILABLE
        (admission failure); a missing **optional** key yields SKIPPED.
    args / uv_groups / expensive / note:
        Extra argv, locked-uv dependency groups, opt-in performance flag.
    """

    case_id: str
    kind: str
    paths: Tuple[str, ...] = ()
    legacy_paths: Tuple[str, ...] = ()
    required: FrozenSet[str] = field(default_factory=frozenset)
    optional: FrozenSet[str] = field(default_factory=frozenset)
    args: Tuple[str, ...] = ()
    uv_groups: Tuple[str, ...] = ()
    expensive: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"{self.case_id}: unknown kind {self.kind!r}")
        if not self.case_id:
            raise ValueError("case_id must be non-empty")

    def select_paths(self, repo_root: Path) -> Tuple[str, Tuple[str, ...]]:
        """Return (origin, existing-paths) choosing canonical over legacy.

        ``origin`` is ``"canonical"`` or ``"legacy"``. If neither set exists,
        returns ``("missing", canonical paths)`` so the caller fails loudly
        instead of silently collecting zero tests.
        """
        canonical = tuple(self.paths)
        if canonical and all((repo_root / p).exists() for p in canonical):
            return "canonical", canonical
        legacy = tuple(self.legacy_paths)
        if legacy and all((repo_root / p).exists() for p in legacy):
            return "legacy", legacy
        return "missing", canonical or legacy


@dataclass(frozen=True)
class SuiteSpec:
    """A named acceptance suite with explicit capabilities."""

    name: str
    description: str
    cases: Tuple[CaseSpec, ...]
    required: FrozenSet[str] = field(default_factory=frozenset)
    expensive: bool = False

    def case(self, case_id: str) -> CaseSpec:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(f"suite {self.name!r} has no case {case_id!r}")


def _p(case_id: str, *canonical: str, legacy: Sequence[str] = (), **kw) -> CaseSpec:
    required = kw.get("required", ())
    group = ("gpu-sm90" if "sm90" in required else "gpu-test") if "torch" in required else "cpu-test"
    return CaseSpec(case_id, "pytest", paths=tuple(canonical), uv_groups=(group, ), legacy_paths=tuple(legacy), **kw)


def _cmd(case_id: str, *canonical_argv: str, script: str = "", legacy_script: str = "", **kw) -> CaseSpec:
    # canonical_argv holds the fixed options AFTER the script token. The script
    # itself comes from the resolved canonical/legacy path.
    return CaseSpec(case_id, "command", paths=(script, ) if script else (),
                    legacy_paths=(legacy_script, ) if legacy_script else (), args=tuple(canonical_argv), **kw)


CPU_SUITE = SuiteSpec(
    "cpu",
    "Pure-CPU compiler/runtime unit, refactor and role-frontend validation; "
    "no GPU. CPU numerical-ABI mocks run without Torch and are never skipped as "
    "a whole file merely because Torch is absent.",
    (
        _p(
            "cpu-unit",
            "tests/nv/common/unit",
            "tests/nv/sm90/unit",
            "tests/nv/sm100/unit",
            legacy=["python/little_kernel/tests/unit"],
            required=frozenset({"cpu"}),
            note="Canonical CPU unit suite (types, passes, ABI, codegen).",
        ),
        _p(
            "cpu-refactor",
            "tests/nv/common/refactor",
            required=frozenset({"cpu"}),
            note="Refactor regression suite incl. CPU numerical-ABI mocks and "
            "verification-runner tests; Torch-free.",
        ),
        _p(
            "cpu-role-channels",
            "tests/nv/sm100/dataflow/test_channel_validation.py",
            required=frozenset({"cpu"}),
            note="CPU-only role/dataflow analysis and two-CTA code-generation checks.",
        ),
        _p(
            "cpu-reference-contracts",
            "tests/nv/sm100/dataflow/test_benchmark_references.py",
            required=frozenset({"cpu"}),
            note="Benchmark reference and performance-gate parser contracts; "
            "its own GPU case self-skips at runtime.",
        ),
    ),
    required=frozenset({"cpu"}),
)

CUDA_COMPILE_SUITE = SuiteSpec(
    "cuda-compile",
    "Lower representative scalar/vector/reduction/MMA kernels and cross-compile "
    "them with the available NVCC. Toolchain/codegen acceptance only -- never a "
    "GPU numerical or performance result.",
    (
        CaseSpec("compile-scalar", "compile", required=frozenset({"cpu", "nvcc"}),
                 note="Scalar elementwise kernel lowers and NVCC-compiles."),
        CaseSpec("compile-vector", "compile", required=frozenset({"cpu", "nvcc"}),
                 note="V4 global vector load/store lowers and NVCC-compiles."),
        CaseSpec("compile-reduction", "compile", required=frozenset({"cpu", "nvcc"}),
                 note="Warp-shuffle reduction lowers and NVCC-compiles."),
        CaseSpec("compile-mma-atom", "compile", required=frozenset({"cpu", "nvcc"}),
                 note="FP64 MMA atom lowers to its documented PTX and compiles."),
    ),
    required=frozenset({"nvcc"}),
)

SM90_SUITE = SuiteSpec(
    "sm90",
    "Hopper-family (exact sm90 arch) numerical integration: TMA Driver, memory "
    "vector access, math intrinsics, SM90 GEMM and causal varlen attention.",
    (
        _p(
            "driver-tma-descriptor",
            "tests/nv/common/integration/test_tma_runtime_driver.py",
            legacy=["python/little_kernel/tests/nv/common/integration/test_tma_runtime_driver.py"],
            required=frozenset({"cuda-driver"}),
            note="Real Driver descriptor encode/rebind and runtime lifecycle. "
            "Needs an SM90-or-newer Driver but NOT Torch and launches no "
            "kernel.",
        ),
        _p(
            "vector-memory",
            "tests/nv/common/integration/test_global_vector_ldst.py",
            legacy=["python/little_kernel/tests/nv/common/integration/test_global_vector_ldst.py"],
            required=frozenset({"sm90", "torch"}),
            note="Vector load/store policies, atomics, bf16/fp8 conversions: "
            "numerical mismatches assert, never print success.",
        ),
        _p(
            "math-intrinsics",
            "tests/nv/common/integration/test_math_intrin_exports.py",
            legacy=["python/little_kernel/tests/nv/common/integration/test_math_intrin_exports.py"],
            required=frozenset({"sm90", "torch"}),
            note="Scalar math intrinsics vs Torch references with assertions.",
        ),
        _p(
            "sm90-gemm",
            "tests/nv/sm90/integration/test_sm90_gemm.py",
            legacy=["python/little_kernel/tests/nv/sm90/integration/test_sm90_gemm.py"],
            required=frozenset({"sm90", "torch"}),
            note="All maintained SM90 GEMM variants vs reference.",
        ),
        _p(
            "sm90-flashattention",
            "tests/nv/sm90/integration/test_flashattention.py",
            required=frozenset({"sm90", "torch"}),
            note="Causal BF16 head-dimension-128 varlen attention vs Torch math.",
        ),
    ),
    required=frozenset({"gpu"}),
)

SM100_SUITE = SuiteSpec(
    "sm100",
    "Blackwell-family numerical integration. SM100 is an EXACT architecture "
    "gate (sm_10x plus a CUDA-12.8+ toolchain for tcgen05): sm_120 does not "
    "satisfy it.",
    (
        _p(
            "sm100-gemm",
            "tests/nv/sm100/integration/test_sm100_gemm.py",
            legacy=["python/little_kernel/tests/nv/sm100/integration/test_sm100_gemm.py"],
            required=frozenset({"sm100", "torch"}),
            note="SM100 GEMM levels 1-9 vs reference.",
        ),
        _p(
            "sm100-clustered-topology",
            "tests/nv/sm100/dataflow/test_clustered_gemm_topology_sm100.py",
            required=frozenset({"sm100", "torch"}),
            note="SM100 clustered-GEMM topology numerical coverage.",
        ),
        _cmd(
            "sm100-generated-kernels",
            "--require-all",
            "--timeout",
            "300",
            script="tests/nv/sm100/dataflow/test_generated_kernels_sm100.py",
            required=frozenset({"sm100", "nvcc-sm100", "torch", "bench-deps"}),
            optional=frozenset({"references"}),
            uv_groups=("bench", ),
            note="Every role SM100 kernel correctness gate; references "
            "auto/required/none forwarded explicitly.",
        ),
    ),
    required=frozenset({"gpu"}),
)

MULTIGPU_SUITE = SuiteSpec(
    "multigpu",
    "Distributed / multi-GPU correctness. Requires peer-accessible GPUs; run by "
    "root on real hardware, never launched from this development worktree.",
    (
        _p(
            "multigpu-flashcomm",
            "tests/nv/sm90/distributed/test_flashcomm_multi_gpu.py",
            legacy=["python/little_kernel/design/test_flashcomm_multi_gpu.py"],
            required=frozenset({"multigpu2", "torch"}),
            note="FlashComm dispatch/combine across 2/4/8 GPUs.",
        ),
        _cmd(
            "multigpu-role-distributed",
            "--only",
            "ag_gemm,gemm_rs,gemm_a2a,a2a_gemm,dispatch_swiglu",
            "--require-all",
            "--timeout",
            "600",
            script="tests/nv/sm100/dataflow/test_generated_kernels_sm100.py",
            required=frozenset({"multigpu2", "sm100", "nvcc-sm100", "torch", "bench-deps"}),
            optional=frozenset({"references"}),
            uv_groups=("bench", ),
            note="Role communication-GEMM cases over torch.distributed.run.",
        ),
    ),
    required=frozenset({"multigpu2"}),
)

PERFORMANCE_SUITE = SuiteSpec(
    "performance",
    "Expensive performance gates with fixed baselines. Never run automatically "
    "on every change; select deliberately. Numbers are valid only on matching "
    "hardware/toolchain.",
    (
        _cmd(
            "perf-role-sm100-baselines",
            "--check-perf",
            "--timeout",
            "1800",
            script="tests/nv/sm100/dataflow/test_generated_kernels_sm100.py",
            required=frozenset({"sm100", "nvcc-sm100", "torch", "bench-deps"}),
            optional=frozenset({"references"}),
            uv_groups=("bench", ),
            expensive=True,
            note="Recorded GB200 latency table; measured on GB200 x4 only and "
            "never replaced by a reduced smoke run.",
        ),
        _cmd(
            "perf-sm100-gemm-levels",
            script="benchmarks/nv/sm100/gemm/run_sm100_levels.py",
            legacy_script="python/little_kernel/benchmark/gemm_sm100/test_all_levels.py",
            required=frozenset({"sm100", "nvcc-sm100", "torch", "bench-deps"}),
            uv_groups=("bench", ),
            expensive=True,
            note="SM100 GEMM correctness and timing catalog; no threshold claim.",
        ),
        _cmd(
            "perf-sm90-gemm",
            "--category",
            "sm90",
            script="benchmarks/nv/common/run_all.py",
            legacy_script="python/little_kernel/benchmark/run_all.py",
            required=frozenset({"sm90", "torch", "bench-deps"}),
            uv_groups=("bench", ),
            expensive=True,
            note="Hopper SM90 GEMM correctness and timing catalog; no threshold claim.",
        ),
        _cmd(
            "perf-hopper-microbench",
            script="benchmarks/nv/common/run_all.py",
            legacy_script="python/little_kernel/benchmark/run_all.py",
            required=frozenset({"sm90", "torch"}),
            uv_groups=("bench", ),
            expensive=True,
            note="Full Hopper microbenchmark suite; scheduled, not per-change.",
        ),
    ),
    required=frozenset({"gpu"}),
    expensive=True,
)

SUITES = {
    suite.name: suite
    for suite in (
        CPU_SUITE,
        CUDA_COMPILE_SUITE,
        SM90_SUITE,
        SM100_SUITE,
        MULTIGPU_SUITE,
        PERFORMANCE_SUITE,
    )
}

SUITE_ALIASES = {
    "all": ("cpu", "cuda-compile", "sm90", "sm100", "multigpu"),
    "quick": ("cpu", "cuda-compile"),
    "v2": ("cpu", "sm100"),
}


def expand_suite_names(names: Sequence[str]) -> List[str]:
    out: List[str] = []
    for name in names:
        if name in SUITE_ALIASES:
            targets = SUITE_ALIASES[name]
        elif name in SUITES:
            targets = (name, )
        else:
            raise KeyError(f"unknown suite {name!r}; known suites: "
                           f"{', '.join(sorted(SUITES))}; aliases: "
                           f"{', '.join(sorted(SUITE_ALIASES))}")
        for target in targets:
            if target not in out:
                out.append(target)
    return out


def get_suite(name: str) -> SuiteSpec:
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}")
    return SUITES[name]


def validate_manifest_paths(repo_root: Path, *, strict_canonical: bool = False) -> List[str]:
    """Return manifest consistency problems (empty when consistent).

    * unknown capability keys are errors;
    * a case whose canonical AND legacy paths are both missing is an error
      (independent of Torch -- path validation never silently skips);
    * with ``strict_canonical`` (used by root after the migration), every
      canonical path must exist and legacy fallbacks must be gone.
    """
    from tools.nv.common.lk_verify.capabilities import HostCaps

    problems: List[str] = []
    known_caps = {
        "cpu",
        "nvcc",
        "nvcc-sm100",
        "torch",
        "gpu",
        "cuda-driver",
        "sm90",
        "sm100",
        "multigpu2",
        "multigpu4",
        "bench-deps",
        "references",
    }
    probe = HostCaps.for_testing(has_nvcc=True, nvcc_major=99, nvcc_minor=0, has_torch=True, has_gpu=True, gpu_count=8,
                                 sm_arch=100, driver_ok=True)
    for suite in SUITES.values():
        ids = set()
        unknown = set(suite.required) - known_caps
        if unknown:
            problems.append(f"suite {suite.name}: unknown required caps {sorted(unknown)}")
        for case in suite.cases:
            if case.case_id in ids:
                problems.append(f"suite {suite.name}: duplicate case id {case.case_id!r}")
            ids.add(case.case_id)
            for cap in case.required | case.optional:
                if cap not in known_caps:
                    problems.append(f"suite {suite.name} case {case.case_id}: unknown cap {cap!r}")
                try:
                    probe.satisfies(cap)
                except KeyError:
                    problems.append(f"suite {suite.name} case {case.case_id}: cap {cap!r} "
                                    f"not evaluable by HostCaps")
            if case.kind == "compile":
                continue
            origin, existing = case.select_paths(repo_root)
            if origin == "missing":
                problems.append(f"suite {suite.name} case {case.case_id}: neither canonical "
                                f"nor legacy paths exist: {list(case.paths) or list(case.legacy_paths)}")
                continue
            if strict_canonical:
                if origin != "canonical":
                    problems.append(f"suite {suite.name} case {case.case_id}: canonical "
                                    f"paths missing while legacy used: {list(case.paths)}")
    return problems
