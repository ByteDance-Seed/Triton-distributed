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

import os
import random
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

# torchrun-populated env vars; available after ``init_dist``.
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
    "float32": torch.float32,
}


def init_seed(seed: int = 0) -> None:
    """Deterministic-ish seeding for Python/NumPy/Torch."""
    import numpy as np

    os.environ.setdefault("NCCL_DEBUG", "ERROR")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_printoptions(precision=4)
    torch.manual_seed(3 + seed)
    torch.cuda.manual_seed_all(3 + seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + seed)
    random.seed(3 + seed)


def init_dist(timeout_seconds: int = 1800) -> torch.distributed.ProcessGroup:
    """Initialise the default + EP NCCL group and return the EP group."""
    import datetime

    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )
    assert torch.distributed.is_initialized()
    init_seed(RANK)
    ep_group = torch.distributed.new_group(
        ranks=list(range(WORLD_SIZE)),
        backend="nccl",
    )
    torch.distributed.barrier(group=ep_group)
    torch.cuda.synchronize()
    return ep_group


# ---------------------------------------------------------------------------
# Comparison primitives.
# ---------------------------------------------------------------------------


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Strict bit-for-bit equality (shape, dtype, device, AND bytes)."""
    assert a.is_contiguous() and b.is_contiguous()
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        return False
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def assert_close_strict(a: torch.Tensor, b: torch.Tensor, *, label: str = "tensor") -> None:
    """``torch.testing.assert_close(rtol=0, atol=0)`` plus byte-equal check."""
    torch.testing.assert_close(a, b, rtol=0, atol=0, msg=lambda m: f"[{label}] {m}")
    assert bitwise_equal(a, b), f"[{label}] bitwise mismatch"


def assert_close_bf16_ulp(
    got: torch.Tensor,
    ref: torch.Tensor,
    *,
    max_ulps: int = 3,
    label: str = "tensor",
) -> Dict[str, float]:
    """Strict bf16 ULP-distance check.

    Allows up to ``max_ulps`` BF16 ULPs per element (budget
    ``max_ulps * (|ref| * 2^-7 + 2^-7)`` so subnormal/zero refs are
    covered). Returns per-tensor stats (``byte_equal_pct``, ``max_abs``,
    ``max_rel``, ``max_ulp_drift``, ``num_violations``) for triage on
    pass; a violation aborts the test.
    """
    if got.shape != ref.shape:
        raise AssertionError(f"[{label}] shape mismatch: {tuple(got.shape)} vs {tuple(ref.shape)}")
    if got.dtype != torch.bfloat16 or ref.dtype != torch.bfloat16:
        raise AssertionError(f"[{label}] both tensors must be bf16; got {got.dtype} / {ref.dtype}")
    got_f = got.to(torch.float32)
    ref_f = ref.to(torch.float32)
    diff = (got_f - ref_f).abs()
    bf16_ulp_unit = 2.0**-7
    tol = max_ulps * (ref_f.abs() * bf16_ulp_unit + bf16_ulp_unit)
    bad_mask = diff > tol
    n_bad = int(bad_mask.sum().item())
    n = int(got.numel())
    n_byte_eq = int((diff == 0).sum().item())
    max_abs = float(diff.max().item()) if n > 0 else 0.0
    safe_ref = ref_f.abs().clamp(min=1e-9)
    max_rel = float((diff / safe_ref).max().item()) if n > 0 else 0.0
    # diff / (|ref| * 2^-7), floored at 1 ULP so subnormals don't blow up.
    approx_ulp = diff / (safe_ref * bf16_ulp_unit + bf16_ulp_unit)
    max_ulp = float(approx_ulp.max().item()) if n > 0 else 0.0
    stats = {
        "n_total": n,
        "byte_equal_pct": 100.0 * n_byte_eq / max(1, n),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "max_ulp_drift": max_ulp,
        "num_violations": n_bad,
        "max_ulps_allowed": max_ulps,
    }
    if n_bad:
        raise AssertionError(f"[{label}] BF16 ULP check FAILED: "
                             f"{n_bad}/{n} elements ({100.0 * n_bad / n:.4f}%) exceed "
                             f"{max_ulps} BF16 ULPs.  "
                             f"max_abs={max_abs:.6f}  max_rel={max_rel:.6e}  "
                             f"max_ulp_drift={max_ulp:.2f}  "
                             f"byte_equal_pct={stats['byte_equal_pct']:.4f}%  "
                             f"(see stats: {stats})")
    return stats


@contextmanager
def nvtx_range(label: Optional[str]):
    """Push/pop an NVTX range AND a torch profiler record."""
    if label is None:
        yield
        return
    torch.cuda.nvtx.range_push(label)
    try:
        with torch.profiler.record_function(label):
            yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextmanager
def get_torch_prof_ctx(enabled: bool):
    """Return a profiler context manager (or a no-op context)."""
    if not enabled:
        yield None
        return
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
    ) as prof:
        yield prof


def record_function_range(label: Optional[str], enabled: bool = True):
    """Return a torch profiler range only for explicit test-side tracing."""
    if not enabled or label is None:
        return nullcontext()
    return torch.profiler.record_function(label)


def export_trace(prof, *, label: str, rank: int, round_id: int, profile_dir: Optional[str] = None) -> str:
    """Dump a chrome trace for the given label / rank / round."""
    if profile_dir is None:
        run_id = os.environ.get("TORCHELASTIC_RUN_ID", "default")
        profile_dir = f"prof/{run_id}"
    os.makedirs(profile_dir, exist_ok=True)
    trace_path = os.path.join(
        profile_dir,
        f"{label}_rank{rank}_round{round_id}.json.gz",
    )
    prof.export_chrome_trace(trace_path)
    return trace_path


def perf_event_ms(
    fn,
    iters: int,
    warmup_iters: int,
    label: Optional[str] = None,
    *,
    device_sleep_cycles: int = 0,
    ep_group: Optional["torch.distributed.ProcessGroup"] = None,
    inter_iter_barrier: bool = False,
    return_stats: bool = False,
) -> Tuple[Any, "float | Dict[str, float]"]:
    """CUDA-event timer. Default returns ``(output, mean_ms)``.

    With ``return_stats=True`` returns ``(output, {mean,min,p10,median,max})``
    so callers can surface the per-iter distribution. The historical
    perf table in the team doc was sampled from Nsight kernel traces
    (i.e. the per-call GPU duration); that lines up with this timer's
    ``min``/``median`` rather than ``mean`` because (a) the inter-rank
    ``_sync_ranks`` barrier needed for combine/dispatch peer-buffer
    safety serialises ranks on the slowest one, and (b) NCCL kernels
    occasionally steal SMs from the timed GEMM. Both pad the right
    tail of the per-iter distribution without affecting the kernel's
    own GPU duration.

    ``inter_iter_barrier=False`` (defaults True) drops the per-iter
    NCCL barrier from the timing loop -- only safe for baselines that
    do NOT push into peer symmetric buffers (e.g. standalone
    ``group_gemm`` / topk reduce). Combine, dispatch and the fused
    group_gemm+combine paths MUST keep the barrier; without it
    rank-N+1's push can race rank-M's still-in-flight reduce.
    """
    iters = int(iters)
    warmup_iters = int(warmup_iters)
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stop_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    def _sync_ranks() -> None:
        if ep_group is not None:
            torch.distributed.barrier(group=ep_group)

    _sync_ranks()
    with nvtx_range(None if label is None else f"{label}.warmup"):
        for _ in range(warmup_iters):
            _ = fn()
    torch.cuda.synchronize()

    output = None
    with nvtx_range(None if label is None else f"{label}.measured"):
        for i in range(iters):
            if inter_iter_barrier:
                _sync_ranks()
            start_events[i].record()
            output = fn()
            stop_events[i].record()
            if device_sleep_cycles > 0:
                torch.cuda._sleep(int(device_sleep_cycles))
    torch.cuda.synchronize()

    deltas = [s.elapsed_time(e) for s, e in zip(start_events, stop_events)]
    deltas_sorted = sorted(deltas)
    n = len(deltas_sorted)
    mean_ms = sum(deltas) / n
    if not return_stats:
        return output, mean_ms
    stats = {
        "mean": mean_ms,
        "min": deltas_sorted[0],
        "p10": deltas_sorted[max(0, (n - 1) * 10 // 100)],
        "median": deltas_sorted[n // 2] if n % 2 else (deltas_sorted[n // 2 - 1] + deltas_sorted[n // 2]) / 2,
        "max": deltas_sorted[-1],
    }
    return output, stats


def print_perf_table(rank: int, round_id: int, rows: List[Tuple[str, float]],
                     extras: Optional[Dict[str, str]] = None) -> None:
    """One-line-per-baseline table with optional extra columns."""
    parts = [f"[RANK {rank}] round {round_id}:"]
    for name, ms in rows:
        parts.append(f"{name}={ms:.4f}ms")
    if extras:
        for k, v in extras.items():
            parts.append(f"{k}={v}")
    print("  ".join(parts))


def generate_random_exp_indices(
    token_num: int,
    total_num_experts: int,
    topk: int,
    drop_ratio: float = 0.0,
) -> torch.Tensor:
    """Uniform-random topk routing with optional per-slot drop."""
    exp_indices = []
    exp_list = list(range(total_num_experts))
    for _ in range(token_num):
        top_selected = random.sample(exp_list, topk)
        for i, _ in enumerate(top_selected):
            if random.uniform(0, 1) < drop_ratio:
                top_selected[i] = total_num_experts
        exp_indices.append(top_selected)
    return torch.tensor(exp_indices, dtype=torch.int32)


def generate_stress_exp_indices(token_num: int, total_num_experts: int, topk: int, drop_ratio: float, case_id: int,
                                rank: int, world_size: int) -> torch.Tensor:
    """Varied routing patterns for stress testing.

    ``case_id`` modulo 7 selects:
      0: uniform random,
      1: all-experts-on-own-rank (self-mostly),
      2: all-experts-on-one-peer (rotating across cases),
      3: round-robin across ranks,
      4: linear sequence,
      5: edges + random,
      6: heavy drop.
    """
    experts_per_rank = total_num_experts // world_size
    exp_list = list(range(total_num_experts))
    case = case_id % 7

    def maybe_drop(vals, local_drop_ratio):
        out = list(vals)
        for i in range(len(out)):
            if random.uniform(0, 1) < local_drop_ratio:
                out[i] = total_num_experts
        return out

    def sample_pool(pool):
        if len(pool) >= topk:
            return random.sample(pool, topk)
        return random.sample(exp_list, topk)

    rows = []
    for t in range(token_num):
        if case == 0:
            selected = sample_pool(exp_list)
            local_drop = drop_ratio
        elif case == 1:
            pool = list(range(rank * experts_per_rank, (rank + 1) * experts_per_rank))
            selected = sample_pool(pool)
            local_drop = drop_ratio
        elif case == 2:
            base_rank = (rank + 1 + (t % max(1, world_size - 1))) % world_size
            pool = list(range(base_rank * experts_per_rank, (base_rank + 1) * experts_per_rank))
            selected = sample_pool(pool)
            local_drop = drop_ratio
        elif case == 3:
            base_rank = t % world_size
            pool = list(range(base_rank * experts_per_rank, (base_rank + 1) * experts_per_rank))
            selected = sample_pool(pool)
            local_drop = drop_ratio
        elif case == 4:
            selected = [(t + k) % total_num_experts for k in range(topk)]
            local_drop = drop_ratio
        elif case == 5:
            edge = [0, total_num_experts - 1]
            rest = sample_pool(exp_list)[:max(0, topk - len(edge))]
            selected = (edge + rest)[:topk]
            local_drop = drop_ratio
        else:
            selected = sample_pool(exp_list)
            local_drop = max(drop_ratio, 0.5)
        rows.append(maybe_drop(selected, local_drop))
    return torch.tensor(rows, dtype=torch.int32)


def densify_aligned_output(aligned_out: torch.Tensor, expert_counts: torch.Tensor,
                           expert_alignment: int) -> torch.Tensor:
    """Strip per-expert tile padding from an expert-aligned output buffer."""
    parts = []
    offset = 0
    for le in range(len(expert_counts)):
        n = expert_counts[le].item()
        parts.append(aligned_out[offset:offset + n])
        aligned_n = ((n + expert_alignment - 1) // expert_alignment) * expert_alignment
        offset += aligned_n
    return torch.cat(parts, dim=0)


def straggler(rank: int) -> None:
    """Rank-skewed device-side delay to stress the barrier discipline."""
    clock_rate = torch.cuda.clock_rate() * 1e6
    cycles = random.randint(0, int(clock_rate * 0.0001)) * (rank + 1)
    torch.cuda._sleep(cycles)


def cool_down(seconds: float) -> None:
    """Brief SM cooldown between perf baselines (frequency-throttling mitigation)."""
    if seconds > 0:
        torch.cuda.synchronize()
        time.sleep(seconds)


def summarize_tensor(t: torch.Tensor) -> str:
    """One-line summary for log lines (shape, dtype, sum)."""
    return (f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"sum={t.to(torch.float32).sum().item():.4f}")


def iter_with_label(items: Iterable, *, label: str):
    """Like ``enumerate`` but yields a per-step nvtx label."""
    for idx, item in enumerate(items):
        yield idx, item, f"{label}.case{idx}"
