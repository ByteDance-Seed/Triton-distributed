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
"""Correctness + performance gate for every role-proposal SM100 kernel.

Every run must inspect both dimensions: all bench correctness checks must pass,
and the tracked LittleKernel latency/ratio must be compared with the fixed table
below. After a large compiler or pipeline change, update the affected rows with
the new default-argument median and an Asia/Shanghai timestamp; do not silently
accept a performance regression because correctness passed.

PERF_BASELINE_TABLE (the single fixed update location; bench default arguments):

| case | tracked output | LK median ms | reference median ms | ratio | validation | device | measured at |
|---|---|---:|---:|---:|---|---|---|
| gemm | role_v2, 8192^3 | 0.726 | 0.705 torch.matmul | 0.97x | PASS | GB200 | 2026-08-20 23:47 +08:00 |
| mxfp4 | 2CTA, 8192^3 | 0.213 | 0.204 F.scaled_mm | 0.96x | PASS | GB200 | 2026-08-24 20:31 +08:00 |
| mxfp8 | 2CTA, 8192^3 | 0.359 | 0.337 F.scaled_mm | 0.94x | PASS | GB200 | 2026-08-24 20:31 +08:00 |
| nvfp4 | 2CTA, 8192^3 | 0.189 | 0.179 F.scaled_mm | 0.94x | PASS | GB200 | 2026-08-24 20:31 +08:00 |
| w4a8_groupgemm | LK worklist/online E40/M29440/N5120/K8192 | 0.628 / 0.628 | 0.611 DeepGEMM | 0.97x / 0.97x | PASS | GB200 | 2026-08-25 00:44 +08:00 |
| fa4 | role_v2, B=4 T=16384 | 25.993 | 24.422 fa4_ref | 0.940x | PASS | GB200 | 2026-08-20 23:47 +08:00 |
| gdn | LittleKernel, large_5x8k | 0.684 | 4.302 FLA | 6.29x | PASS | GB200 | 2026-09-02 21:58 +08:00 |
| groupgemm | LittleKernel 2CTA | 1.642 | 1.540 torch.grouped_mm | 0.94x | PASS | GB200 | 2026-08-20 23:47 +08:00 |
| ag_gemm | fuse_overlap | 0.950 | 1.102 torch_sequential | 1.16x | PASS | GB200 x4 | 2026-08-25 10:31 +08:00 |
| gemm_rs | fuse_overlap | 1.009 | 1.447 torch_sequential | 1.43x | PASS | GB200 x4 | 2026-08-20 23:47 +08:00 |
| gemm_a2a | role_fused | 2.478 | 3.296 torch_sequential | 1.33x | PASS | GB200 x4 | 2026-08-20 23:47 +08:00 |
| a2a_gemm | role_fused | 2.378 | 2.825 torch_sequential | 1.19x | PASS | GB200 x4 | 2026-08-20 23:47 +08:00 |
| dispatch_swiglu | LK fused | 2.119 | 2.077 CuTeDSL | 0.98x | PASS | GB200 x4 | 2026-08-20 23:47 +08:00 |

The ratio is reference_ms / LK_ms, so larger is better. A partial ``--only``
run updates only its rows. Use default shapes/warmup/iters for recorded numbers;
reduced runs are smoke tests and must not replace the baseline. The distributed
cases disable optional external comparators not tracked in this table.

    python test_generated_kernels_sm100.py
    python test_generated_kernels_sm100.py --only gemm,fa4
    python test_generated_kernels_sm100.py --skip ag_gemm --warmup 5 --iters 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tests.nv.sm100.dataflow._source_paths import REPO_ROOT, role_bench_path

HERE = Path(__file__).resolve().parent
PYTHON_ROOT = REPO_ROOT / "python"


@dataclass(frozen=True)
class Case:
    name: str
    script: str
    # Single-process benches stay on one GPU. Distributed benches use every
    # visible GPU via torch.distributed.run (same as a manual torchrun).
    distributed: bool = False
    bench_args: tuple[str, ...] = ()


# One row per bench_v2_*.py. Kernel sources (v2_*.py) are compiled by the bench.
CASES: tuple[Case, ...] = (
    Case("gemm", "bench_v2_gemm_sm100.py"),
    Case("mxfp4", "bench_v2_mxfp4_gemm.py"),
    Case("mxfp8", "bench_v2_mxfp8_gemm.py"),
    Case("nvfp4", "bench_v2_nvfp4_gemm.py"),
    Case("w4a8_groupgemm", "bench_v2_mxfp_w4a8_groupgemm.py"),
    Case("fa4", "bench_v2_fa4_sm100.py"),
    Case("gdn", "bench_v2_gdn.py"),
    Case("groupgemm", "bench_v2_groupgemm.py"),
    Case("ag_gemm", "bench_v2_ag_gemm_sm100.py", True, ("--no-triton-dist", "--no-thunderkittens")),
    Case("gemm_rs", "bench_v2_gemm_rs_sm100.py", True,
         ("--no-triton-dist", "--no-triton-dist-reduce-st", "--no-thunderkittens")),
    Case("gemm_a2a", "bench_v2_gemm_a2a_sm100.py", True, ("--no-triton-dist", )),
    Case("a2a_gemm", "bench_v2_a2a_gemm_sm100.py", True, ("--no-triton-dist", )),
    Case("dispatch_swiglu", "bench_v2_dispatch_group_gemm_swiglu_sm100.py", distributed=True),
)

# Extract only the tracked default-shape row, not small correctness shapes.
PERF_PATTERNS = {
    "gemm": r"role_v2:\s+([\d.]+) ms",
    "mxfp4": r"role_v2:\s+([\d.]+) ms",
    "mxfp8": r"role_v2:\s+([\d.]+) ms",
    "nvfp4": r"role_v2:\s+([\d.]+) ms",
    "w4a8_groupgemm": r"LK (?:worklist|online):\s+([\d.]+) ms",
    "fa4": r"=== FA4 B=4 T=16384 ===\s+role_v2:\s+([\d.]+) ms",
    "gdn": r"=== GDN large_5x8k:[\s\S]*?LittleKernel\s+([\d.]+)",
    "groupgemm": r"LittleKernel 2CTA:\s+([\d.]+) ms",
    "ag_gemm": r"fuse_overlap:\s+([\d.]+) ms",
    "gemm_rs": r"fuse_overlap:\s+([\d.]+) ms",
    "gemm_a2a": r"role_fused:\s+([\d.]+) ms",
    "a2a_gemm": r"role_fused:\s+([\d.]+) ms",
    "dispatch_swiglu": r"LittleKernel role fused:\s+([\d.]+) ms",
}
REFERENCE_CASES = {"w4a8_groupgemm", "fa4", "gdn", "dispatch_swiglu"}


def check_performance(name, output, tolerance):
    values = [float(value) for value in re.findall(PERF_PATTERNS[name], output)]
    row = next(line for line in __doc__.splitlines() if line.startswith(f"| {name} |"))
    baseline = [float(value.strip()) for value in row.split("|")[3].split("/")]
    if name != "w4a8_groupgemm":
        values = values[-1:]
    if len(values) != len(baseline):
        raise ValueError(f"{name}: missing tracked performance output")
    passed = all(value <= limit * (1 + tolerance) for value, limit in zip(values, baseline))
    return {"measured_ms": values, "baseline_ms": baseline, "passed": passed}


def _gpu_count() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


def _pythonpath(distributed: bool) -> str:
    paths = [str(PYTHON_ROOT)]
    triton_dist = REPO_ROOT / "Triton-distributed" / "python"
    if distributed and triton_dist.is_dir():
        paths.append(str(triton_dist))
    existing = os.environ.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    return os.pathsep.join(paths)


def _env(case: Case) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath(case.distributed)
    # Distributed benches init NCCL. Default driver logging is INFO and drowns
    # correctness/perf output; keep WARN unless the caller asked for more.
    if env.get("NCCL_DEBUG", "INFO").upper() in {"", "VERSION", "INFO"}:
        env["NCCL_DEBUG"] = "WARN"
    return env


def _command(case: Case, forwarded: list[str], nproc: int) -> list[str]:
    # Locate the real bench script through the canonical benchmark mapping,
    # never a compatibility facade __file__.
    script = str(role_bench_path(case.script))
    if not case.distributed:
        return [sys.executable, script, *case.bench_args, *forwarded]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--standalone",
        script,
        *case.bench_args,
        *forwarded,
    ]


def _select(only: str | None, skip: str | None) -> list[Case]:
    names = {case.name: case for case in CASES}
    chosen = list(CASES)
    if only:
        wanted = [name.strip() for name in only.split(",") if name.strip()]
        unknown = [name for name in wanted if name not in names]
        if unknown:
            raise SystemExit(f"unknown --only names: {', '.join(unknown)}")
        chosen = [names[name] for name in wanted]
    if skip:
        dropped = {name.strip() for name in skip.split(",") if name.strip()}
        unknown = sorted(dropped - set(names))
        if unknown:
            raise SystemExit(f"unknown --skip names: {', '.join(unknown)}")
        chosen = [case for case in chosen if case.name not in dropped]
    return chosen


def _run(case: Case, forwarded: list[str], nproc: int, timeout: float | None) -> tuple[str, float, str]:
    cmd = _command(case, forwarded, nproc)
    print("\n" + "=" * 72)
    print(f"CASE {case.name}")
    print(" ".join(cmd))
    print("=" * 72, flush=True)
    started = time.monotonic()
    with subprocess.Popen(cmd, env=_env(case), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          start_new_session=True) as process:
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            print(output, flush=True)
            return "TIMEOUT", time.monotonic() - started, output
    print(output, flush=True)
    status = "PASS" if process.returncode == 0 else f"FAIL({process.returncode})"
    if status == "PASS" and "REFERENCE UNAVAILABLE:" in output:
        status = "PASS_NO_REFERENCE"
    return status, time.monotonic() - started, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated case names to run")
    parser.add_argument("--skip", help="comma-separated case names to skip")
    parser.add_argument("--list", action="store_true", help="print the case table and exit")
    parser.add_argument("--warmup", type=int, help="forwarded to each bench")
    parser.add_argument("--iters", type=int, help="forwarded to each bench")
    parser.add_argument("--timeout", type=float, default=None, help="seconds per case; default is no limit")
    parser.add_argument("--references", choices=("auto", "required", "none"), default="auto")
    parser.add_argument("--require-all", action="store_true", help="fail on hardware skips")
    parser.add_argument("--check-perf", action="store_true",
                        help="check LK default-shape latency against the table (GB200)")
    parser.add_argument("--perf-tolerance", type=float, default=0.10, help="allowed latency regression, default 10%%")
    parser.add_argument("--report", type=Path, help="write JSON results including measured/baseline latency")
    args = parser.parse_args()
    if args.perf_tolerance < 0:
        parser.error("--perf-tolerance must be non-negative")
    if args.check_perf and (args.warmup is not None or args.iters is not None):
        parser.error("performance acceptance requires default warmup/iters")

    if args.list:
        for case in CASES:
            kind = "torchrun" if case.distributed else "python"
            print(f"{case.name:20} {kind:10} {case.script}")
        return

    gpus = _gpu_count()
    if gpus == 0:
        raise SystemExit("no CUDA GPU visible; SM100 benches require a Blackwell GPU")

    selected = _select(args.only, args.skip)
    if not selected:
        parser.error("no cases selected")
    if args.check_perf:
        import torch
        if "GB200" not in torch.cuda.get_device_name():
            parser.error("the recorded performance baseline requires GB200")
        if any(case.distributed for case in selected) and gpus != 4:
            parser.error("distributed performance acceptance requires exactly four visible GB200 GPUs")
    forwarded: list[str] = []
    if args.warmup is not None:
        forwarded += ["--warmup", str(args.warmup)]
    if args.iters is not None:
        forwarded += ["--iters", str(args.iters)]
    rows: list[tuple[str, str, float]] = []
    report = {}
    for case in selected:
        if case.distributed and gpus < 2:
            print(f"SKIP {case.name}: needs >=2 GPUs, have {gpus}")
            rows.append((case.name, "SKIP", 0.0))
            report[case.name] = {"status": "SKIP"}
            continue
        case_args = forwarded + (["--references", args.references] if case.name in REFERENCE_CASES else [])
        status, elapsed, output = _run(case, case_args, gpus, args.timeout)
        result = {"status": status, "seconds": elapsed}
        if status.startswith("PASS") and args.check_perf:
            try:
                result.update(check_performance(case.name, output, args.perf_tolerance))
                if not result["passed"]:
                    status = "PERF_REGRESSION"
            except ValueError as exc:
                result["error"] = str(exc)
                status = "PERF_MISSING"
        result["status"] = status
        if args.check_perf:
            print(f"PERF {case.name}: {result}", flush=True)
        report[case.name] = result
        rows.append((case.name, status, elapsed))

    print("\n" + "=" * 72)
    print(f"{'case':20} {'status':12} {'sec':>8}")
    print("-" * 72)
    for name, status, elapsed in rows:
        print(f"{name:20} {status:12} {elapsed:8.1f}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    failed = [
        name for name, status, _ in rows
        if not status.startswith("PASS") and (status != "SKIP" or args.require_all or args.check_perf)
    ]
    if failed:
        raise SystemExit(f"{len(failed)} case(s) failed: {', '.join(failed)}")
    print("selected cases completed; see per-case reference availability and skip status above")


if __name__ == "__main__":
    main()
