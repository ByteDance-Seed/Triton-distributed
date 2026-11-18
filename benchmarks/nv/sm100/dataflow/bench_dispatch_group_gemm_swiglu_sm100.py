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
"""Validate and benchmark role-fused MoE dispatch + grouped GEMM + SwiGLU.

FlashComm/CuTeDSL is an optional baseline; PyTorch supplies standalone routing and checks.
The LittleKernel implementation uses one fused role kernel; it does not
provide a separate stream-overlap variant.
"""

from __future__ import annotations
from benchmarks.nv.sm100.dataflow._paths import PYTHON_INCLUDE_ROOT, REPO_ROOT

import argparse
import gc
import os
import random
import statistics
import sys
import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from types import SimpleNamespace
import torch.nn.functional as F

from benchmarks.nv.sm100.dataflow.bench_report import ReportRow, SpeedupRow, print_benchmark_report, print_speedups
from benchmarks.nv.sm100.dataflow.bench_optional import add_reference_argument, optional_import
from tools.merge_rank_traces import merge_rank_traces
from examples.nv.sm100.dispatch_group_gemm_swiglu_fused_sm100 import DispatchGroupGemmSwigluFuse

REFRESH_SECONDS = 0.4

CUDAEPKernels = EPCommLayoutDesc = EPOverlapKernels = None


def _load_flashcomm(mode):
    global CUDAEPKernels, EPCommLayoutDesc, EPOverlapKernels
    cutlass = optional_import("cutlass", "CUTLASS_DSL_PYTHON_PATH", [
        Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" /
        "site-packages/nvidia_cutlass_dsl/python_packages", REPO_ROOT.parent / "cutlass/python/CuTeDSL"
    ], mode=mode)
    if cutlass is None:
        return
    root = REPO_ROOT.parent / "Triton-distributed/FlashComm/python"
    overlap = optional_import("flash_comm.ep_overlap", "FLASHCOMM_PYTHON_PATH", [root], mode=mode)
    if overlap is not None:
        from flash_comm.ep import EPKernels
        from flash_comm.ep.ep_kernels import EPCommLayoutDesc as Layout
        CUDAEPKernels, EPCommLayoutDesc, EPOverlapKernels = EPKernels, Layout, overlap.EPOverlapKernels


CHECK_RTOL = 8e-2
CHECK_ATOL = 8e-2
CUTEDSL_EXPERT_ALIGNMENT = 256
_CUDA_EP_SUPPORTED_HIDDEN = (1024, 1536, 2048, 2304, 3328, 3584, 4096, 5120, 6144, 7168, 8192)
_CUDA_EP_SUPPORTED_TOPK = (6, 8)
_CUTEDSL_POINTER_DEPRECATION = ("Use explicit `struct.scalar.ptr` for pointer instead.")


@contextmanager
def _suppress_cutedsl_pointer_deprecation():
    """Drop CuTeDSL's forced pointer warning while preserving all others."""
    with warnings.catch_warnings(record=True) as captured:
        yield
    for warning in captured:
        if (warning.category is DeprecationWarning and str(warning.message) == _CUTEDSL_POINTER_DEPRECATION):
            continue
        warnings.warn_explicit(
            warning.message,
            warning.category,
            warning.filename,
            warning.lineno,
            source=warning.source,
        )


def _padded_offsets(expert_counts: torch.Tensor, alignment: int) -> tuple[int, ...]:
    offsets = [0]
    for count in expert_counts.cpu().tolist():
        count = int(count)
        offsets.append(offsets[-1] + (count + alignment - 1) // alignment * alignment)
    return tuple(offsets)


def _pack_valid_rows(tensor: torch.Tensor, counts: tuple[int, ...], offsets: tuple[int, ...]) -> torch.Tensor:
    parts = [tensor[offsets[expert]:offsets[expert] + count] for expert, count in enumerate(counts) if count]
    if not parts:
        return torch.empty((0, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
    return torch.cat(parts, dim=0)


def _generate_topk(
    *,
    num_tokens: int,
    num_experts: int,
    topk: int,
    drop_ratio: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if not 0 <= drop_ratio <= 1:
        raise ValueError("drop_ratio must be in [0, 1]")
    if not 1 <= topk <= num_experts:
        raise ValueError("topk must be in [1, num_experts]")
    rng = random.Random(seed)
    rows = []
    experts = range(num_experts)
    for _ in range(num_tokens):
        selected = rng.sample(experts, topk)
        rows.append([num_experts if rng.random() < drop_ratio else selected[topk_idx] for topk_idx in range(topk)])
    return torch.tensor(rows, dtype=torch.int32, device=device)


def _make_weights(
    *,
    experts_per_rank: int,
    intermediate_size: int,
    hidden_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    standard = torch.randn(
        (experts_per_rank, 2 * intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.1)
    tile = DispatchGroupGemmSwigluFuse.SWIGLU_EPI_TILE_N
    interleaved = standard.reshape(experts_per_rank, 2, intermediate_size // tile, tile,
                                   hidden_size).transpose(1, 2).reshape_as(standard).contiguous()
    # CuTeDSL requires the K-major (N, K, E) view, not a contiguous repack.
    cutedsl_view = interleaved.permute(1, 2, 0)
    return standard, interleaved, cutedsl_view


@dataclass
class DispatchGroupGemmSwiglu:
    flashcomm: EPOverlapKernels
    cuda_ep: CUDAEPKernels | None
    role_fused: DispatchGroupGemmSwigluFuse
    inputs: torch.Tensor
    topk_indices: torch.Tensor
    weights_standard: torch.Tensor
    weights_cutedsl: torch.Tensor
    layout: EPCommLayoutDesc
    recv_count: torch.Tensor
    cutedsl_a: torch.Tensor
    cutedsl_out: torch.Tensor
    expert_counts: torch.Tensor
    counts_cpu: tuple[int, ...]
    padded_offsets: tuple[int, ...]
    role_padded_offsets: tuple[int, ...]
    rank: int
    world_size: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk: int
    dispatch_num_stages: int
    gemm_num_sm: int
    input_buffer: torch.Tensor | None = None
    input_handle: object = None

    @property
    def device(self) -> torch.device:
        return self.inputs.device

    @property
    def valid_m(self) -> int:
        return sum(self.counts_cpu)

    @property
    def padded_m(self) -> int:
        return self.padded_offsets[-1]

    @property
    def role_padded_m(self) -> int:
        return self.role_padded_offsets[-1]

    def stage_inputs(self) -> None:
        if self.flashcomm is None:
            self.input_buffer.copy_(self.inputs)
            dist.barrier(device_ids=[self.device.index])
            return
        overlap = self.flashcomm.overlap_context
        overlap.dispatch_input_buf[:self.inputs.shape[0]].copy_(self.inputs)
        self.flashcomm.ep_group_barrier()

    def launch_cutedsl(self, stream: torch.cuda.Stream | None = None) -> None:
        """Invoke cutedsl_dispatch_group_gemm_swiglu.py as the baseline."""
        current = torch.cuda.current_stream() if stream is None else stream
        with torch.cuda.stream(current):
            self.flashcomm.overlap_context.reset_expert_signals()
            with _suppress_cutedsl_pointer_deprecation():
                self.flashcomm._dispatch_group_gemm_swiglu_op.run(
                    A_padded=self.cutedsl_a,
                    B_swiglu=self.weights_cutedsl,
                    token_src_rank_topk_and_indices=self.layout.token_src_rank_topk_and_indices,
                    recv_count=self.recv_count,
                    recv_expert_counts=self.expert_counts,
                    output=self.cutedsl_out,
                    dispatch_num_stages=self.dispatch_num_stages,
                    num_sm=self.gemm_num_sm,
                )

    def launch_role_fused(self, stream: torch.cuda.Stream | None = None) -> None:
        self.role_fused.launch(stream=torch.cuda.current_stream() if stream is None else stream)


def _run_cuda_dispatch_reference(ctx: DispatchGroupGemmSwiglu, ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if ctx.cuda_ep is None:
        raise RuntimeError("CUDA EP reference was not created")
    layout = EPCommLayoutDesc()
    dispatched, weights, layout = ctx.cuda_ep.dispatch(ctx.inputs, ctx.topk_indices, None, layout)
    a_ref, _, layout = ctx.cuda_ep.dispatch_postprocess(dispatched.clone(), weights, layout)
    counts = layout.recv_expert_counts.clone()
    offsets = _padded_offsets(counts, CUTEDSL_EXPERT_ALIGNMENT)
    return a_ref.clone(), counts, offsets


def _torch_dispatch_reference(ctx):
    inputs = [torch.empty_like(ctx.inputs) for _ in range(ctx.world_size)]
    routes = [torch.empty_like(ctx.topk_indices) for _ in range(ctx.world_size)]
    dist.all_gather(inputs, ctx.inputs)
    dist.all_gather(routes, ctx.topk_indices)
    rows = []
    for expert, count in enumerate(ctx.counts_cpu):
        parts = []
        for source in range(ctx.world_size):
            token, _ = torch.where(routes[source] == ctx.rank * len(ctx.counts_cpu) + expert)
            parts.append(inputs[source][token])
        row = torch.cat(parts)
        assert row.shape[0] == count
        rows.append(F.pad(row, (0, 0, 0, ctx.padded_offsets[expert + 1] - ctx.padded_offsets[expert] - count)))
    return torch.cat(rows), ctx.expert_counts, ctx.padded_offsets


def _torch_reference(
    ctx: DispatchGroupGemmSwiglu,
    a_ref: torch.Tensor,
    counts: tuple[int, ...],
    offsets: tuple[int, ...],
) -> torch.Tensor:
    out = torch.zeros((offsets[-1], ctx.intermediate_size), dtype=torch.bfloat16, device=ctx.device)
    for expert, count in enumerate(counts):
        if not count:
            continue
        start = offsets[expert]
        accum_bf16 = torch.matmul(a_ref[start:start + count], ctx.weights_standard[expert].t())
        accum = accum_bf16.float()
        gate = accum[:, :ctx.intermediate_size]
        up = accum[:, ctx.intermediate_size:]
        out[start:start + count] = (F.silu(gate) * up).to(torch.bfloat16)
    return out


def _assert_distributed_bool(ok: bool, *, device: torch.device, label: str) -> None:
    failed = torch.tensor(0 if ok else 1, dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise AssertionError(f"{label} failed on at least one rank")


def _assert_distributed_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    rtol: float,
    atol: float,
) -> float:
    same_shape = actual.shape == expected.shape
    finite = bool(torch.isfinite(actual).all()) if same_shape else False
    close = (bool(torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol))
             if same_shape and finite else False)
    local_max = ((actual.float() - expected.float()).abs().max() if same_shape and actual.numel() else torch.tensor(
        float("inf"), device=actual.device))
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
    _assert_distributed_bool(close, device=actual.device, label=f"{label} (max_abs={local_max.item():.6g})")
    return float(local_max.item())


def _assert_distributed_bf16_ulp(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_ulps: int,
    label: str,
) -> dict[str, float]:
    if actual.shape != expected.shape:
        _assert_distributed_bool(False, device=actual.device, label=f"{label} shape")
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    unit = 2.0**-7
    tolerance = max_ulps * (expected_f.abs() * unit + unit)
    violations = (diff > tolerance).sum().to(torch.int64)
    total = torch.tensor(actual.numel(), dtype=torch.int64, device=actual.device)
    byte_equal = (diff == 0).sum().to(torch.int64)
    max_abs = (diff.max() if diff.numel() else torch.tensor(0.0, device=actual.device))
    approx_ulp = diff / (expected_f.abs() * unit + unit)
    max_ulp = (approx_ulp.max() if approx_ulp.numel() else torch.tensor(0.0, device=actual.device))
    dist.all_reduce(violations)
    dist.all_reduce(total)
    dist.all_reduce(byte_equal)
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    dist.all_reduce(max_ulp, op=dist.ReduceOp.MAX)
    if violations.item():
        raise AssertionError(f"{label}: {violations.item()}/{total.item()} elements exceed "
                             f"{max_ulps} approximate BF16 ULPs; "
                             f"max_abs={max_abs.item():.6g}, max_ulp={max_ulp.item():.3f}")
    return {
        "byte_equal_pct": (100.0 * byte_equal.item() / max(1, total.item())),
        "max_abs": float(max_abs.item()),
        "max_ulp": float(max_ulp.item()),
    }


def check_correctness(
    ctx: DispatchGroupGemmSwiglu,
    *,
    iterations: int,
    max_bf16_ulps: int,
    rtol: float,
    atol: float,
) -> None:
    for iteration in range(iterations):
        if iteration:
            ctx.inputs.add_(0.01)

        dist.barrier(device_ids=[ctx.device.index])
        a_ref, cuda_counts, cuda_offsets = (_run_cuda_dispatch_reference(ctx)
                                            if ctx.flashcomm is not None else _torch_dispatch_reference(ctx))
        torch.cuda.synchronize(ctx.device)
        dist.barrier(device_ids=[ctx.device.index])

        cuda_counts_cpu = tuple(int(x) for x in cuda_counts.cpu().tolist())
        _assert_distributed_bool(
            cuda_counts_cpu == ctx.counts_cpu,
            device=ctx.device,
            label="CUDA/CuTeDSL expert-count layout",
        )
        _assert_distributed_bool(
            cuda_offsets == ctx.padded_offsets,
            device=ctx.device,
            label="CUDA/CuTeDSL padded offsets",
        )

        ctx.stage_inputs()
        if ctx.flashcomm is not None:
            ctx.cutedsl_a.fill_(float("nan"))
            ctx.cutedsl_out.fill_(float("nan"))
            ctx.launch_cutedsl()
            torch.cuda.synchronize(ctx.device)

        ctx.role_fused.workspace.fill_(float("nan"))
        ctx.role_fused.out.fill_(float("nan"))
        ctx.launch_role_fused()
        torch.cuda.synchronize(ctx.device)

        ref_a_valid = _pack_valid_rows(a_ref, cuda_counts_cpu, cuda_offsets)
        role_a_valid = _pack_valid_rows(
            ctx.role_fused.workspace,
            ctx.counts_cpu,
            ctx.role_padded_offsets,
        )
        if ctx.flashcomm is not None:
            cutedsl_a_valid = _pack_valid_rows(ctx.cutedsl_a, ctx.counts_cpu, ctx.padded_offsets)
            _assert_distributed_bool(torch.equal(cutedsl_a_valid, ref_a_valid), device=ctx.device,
                                     label="CuTeDSL dispatch bytes")
        _assert_distributed_bool(
            torch.equal(role_a_valid, ref_a_valid),
            device=ctx.device,
            label="LittleKernel dispatch bytes",
        )

        role_out_valid = _pack_valid_rows(
            ctx.role_fused.out,
            ctx.counts_cpu,
            ctx.role_padded_offsets,
        )
        torch_ref = _torch_reference(ctx, a_ref, cuda_counts_cpu, cuda_offsets)
        torch_ref_valid = _pack_valid_rows(torch_ref, cuda_counts_cpu, cuda_offsets)
        if ctx.flashcomm is not None:
            cutedsl_out_valid = _pack_valid_rows(ctx.cutedsl_out, ctx.counts_cpu, ctx.padded_offsets)
            cross_stats = _assert_distributed_bf16_ulp(role_out_valid, cutedsl_out_valid, max_ulps=max_bf16_ulps,
                                                       label="LittleKernel vs CuTeDSL fused output")
            _assert_distributed_close(cutedsl_out_valid, torch_ref_valid, label="CuTeDSL vs PyTorch", rtol=rtol,
                                      atol=atol)
            if ctx.rank == 0:
                print(f"LittleKernel vs CuTeDSL: {cross_stats}")
        role_torch_error = _assert_distributed_close(
            role_out_valid,
            torch_ref_valid,
            label="LittleKernel vs PyTorch",
            rtol=rtol,
            atol=atol,
        )
        if ctx.rank == 0:
            print(f"correctness iteration {iteration}: pass; "
                  f"valid_M={ctx.valid_m}, "
                  f"CuTe_padded_M={ctx.padded_m}, "
                  f"LK_padded_M={ctx.role_padded_m}, "
                  f"torch_max_abs={role_torch_error:.6g}")


def _time_launch_once(launch: Callable[..., None]) -> float:
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    launch(stream=stream)
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end)


def _bench(ctx: DispatchGroupGemmSwiglu, once: Callable[[], float], *, warmup: int, iters: int) -> float:
    device_id = ctx.device.index
    torch.cuda.synchronize(device_id)
    dist.barrier(device_ids=[device_id])
    time.sleep(REFRESH_SECONDS)
    dist.barrier(device_ids=[device_id])
    once()
    for _ in range(warmup):
        dist.barrier(device_ids=[device_id])
        once()
    samples = []
    for _ in range(iters):
        dist.barrier(device_ids=[ctx.device.index])
        samples.append(once())
    rank_max = torch.tensor(samples, dtype=torch.float64, device=ctx.device)
    dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
    return statistics.median(rank_max.tolist())


def bench_compare(ctx: DispatchGroupGemmSwiglu, *, warmup: int, iters: int) -> None:
    ctx.stage_inputs()
    funcs: dict[str, Callable[[], float]] = {
        "role_fused": partial(_time_launch_once, ctx.launch_role_fused),
    }
    if ctx.flashcomm is not None:
        funcs = {"cutedsl_fused": partial(_time_launch_once, ctx.launch_cutedsl), **funcs}
    times = {name: _bench(ctx, fn, warmup=warmup, iters=iters) for name, fn in funcs.items()}
    if ctx.rank != 0:
        return

    n_full = 2 * ctx.intermediate_size
    flops = 2.0 * ctx.valid_m * n_full * ctx.hidden_size
    # Count only rows sourced from remote ranks in the balanced expectation.
    remote_bytes = (ctx.valid_m * ctx.hidden_size * ctx.inputs.element_size() * (ctx.world_size - 1) /
                    max(1, ctx.world_size))
    print_benchmark_report(
        title=("Dispatch-GroupGEMM-SwiGLU "
               f"tokens/rank={ctx.inputs.shape[0]} valid_M={ctx.valid_m} "
               f"CuTe_padded_M={ctx.padded_m} "
               f"LK_padded_M={ctx.role_padded_m} "
               f"E/rank={len(ctx.counts_cpu)} "
               f"H={ctx.hidden_size} I={ctx.intermediate_size} "
               f"topk={ctx.topk} ranks={ctx.world_size}"),
        times=times,
        rows=[
            ReportRow("CuTeDSL fused:", "cutedsl_fused", "tflops", "max", "(optional baseline)"),
            ReportRow("LittleKernel role fused:", "role_fused", "tflops", "max", "(2x2 cluster, 2-CTA UMMA)"),
        ],
        flops=flops,
        comm_bytes=remote_bytes,
        device=ctx.device,
        comm_label="estimated remote payload",
    )
    print_speedups(
        times,
        [SpeedupRow("role_speedup:", "cutedsl_fused", "role_fused", "(CuTeDSL fused / LittleKernel role fused)")],
    )


def export_trace(
    ctx: DispatchGroupGemmSwiglu,
    trace_dir: str | Path,
    *,
    merge: bool,
    trace_iters: int,
    with_stack: bool,
) -> None:
    from torch.profiler import ProfilerActivity, profile, record_function

    ctx.stage_inputs()
    funcs: dict[str, Callable[[], None]] = {
        "littlekernel_role_fused": lambda: _time_launch_once(ctx.launch_role_fused),
    }
    if ctx.flashcomm is not None:
        funcs["cutedsl_dispatch_group_gemm_swiglu"] = lambda: _time_launch_once(ctx.launch_cutedsl)
    for fn in funcs.values():
        dist.barrier(device_ids=[ctx.device.index])
        fn()

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=with_stack,
    )
    dist.barrier(device_ids=[ctx.device.index])
    prof.start()
    for label, fn in funcs.items():
        for _ in range(trace_iters):
            dist.barrier(device_ids=[ctx.device.index])
            with record_function(label):
                fn()
    prof.stop()

    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"dispatch_group_gemm_swiglu_t{ctx.inputs.shape[0]}"
            f"_h{ctx.hidden_size}_i{ctx.intermediate_size}"
            f"_e{ctx.num_experts}_k{ctx.topk}_r{ctx.world_size}")
    rank_trace = trace_dir / f"{stem}_rank{ctx.rank}.json.gz"
    prof.export_chrome_trace(str(rank_trace))
    dist.barrier(device_ids=[ctx.device.index])
    if merge and ctx.rank == 0:
        rank_traces = [trace_dir / f"{stem}_rank{rank}.json.gz" for rank in range(ctx.world_size)]
        merged = merge_rank_traces(rank_traces, trace_dir / f"{stem}_merged.json.gz")
        print(f"merged trace: {merged}")
    elif ctx.rank == 0:
        print(f"rank traces: {trace_dir / f'{stem}_rank*.json.gz'}")
    dist.barrier(device_ids=[ctx.device.index])


def prepare(args, compiled_kernel, ep_group, *, create_cuda_reference: bool) -> DispatchGroupGemmSwiglu:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank)
    if args.num_experts % world_size:
        raise ValueError("num_experts must be divisible by world_size")
    create_cuda_reference = create_cuda_reference and EPOverlapKernels is not None
    if (create_cuda_reference and args.hidden_size not in _CUDA_EP_SUPPORTED_HIDDEN):
        raise ValueError("the CUDA dispatch correctness reference supports hidden_size "
                         f"in {_CUDA_EP_SUPPORTED_HIDDEN}; pass --no-check to benchmark "
                         "another LittleKernel/CuTeDSL shape")
    if create_cuda_reference and args.topk not in _CUDA_EP_SUPPORTED_TOPK:
        raise ValueError("the CUDA dispatch correctness reference supports topk in "
                         f"{_CUDA_EP_SUPPORTED_TOPK}; pass --no-check to benchmark "
                         "another LittleKernel/CuTeDSL shape")
    experts_per_rank = args.num_experts // world_size
    if args.intermediate_size % DispatchGroupGemmSwigluFuse.OUTPUT_TILE_N:
        raise ValueError(f"intermediate_size must be divisible by "
                         f"{DispatchGroupGemmSwigluFuse.OUTPUT_TILE_N}")
    if args.hidden_size % DispatchGroupGemmSwigluFuse.K_TILE:
        raise ValueError(f"hidden_size must be divisible by "
                         f"{DispatchGroupGemmSwigluFuse.K_TILE}")

    flashcomm = EPOverlapKernels(
        max_m=args.num_tokens,
        hidden=args.hidden_size,
        topk=args.topk,
        num_experts=args.num_experts,
        local_world_size=world_size,
        ep_group=ep_group,
        num_sm=args.num_sm,
        num_worst_tokens=args.num_worst_tokens,
        expert_alignment=CUTEDSL_EXPERT_ALIGNMENT,
    ) if EPOverlapKernels is not None else None
    cuda_ep = (CUDAEPKernels(
        max_m=args.num_tokens,
        hidden=args.hidden_size,
        topk=args.topk,
        num_experts=args.num_experts,
        local_world_size=world_size,
        ep_group=ep_group,
        num_sm=args.num_sm,
        num_worst_tokens=args.num_worst_tokens,
        expert_alignment=CUTEDSL_EXPERT_ALIGNMENT,
    ) if create_cuda_reference else None)

    topk_indices = _generate_topk(
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        topk=args.topk,
        drop_ratio=args.drop_ratio,
        seed=42 + rank,
        device=device,
    )
    input_generator = torch.Generator(device=device)
    input_generator.manual_seed(2026 + rank)
    inputs = torch.randn(
        (args.num_tokens, args.hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=input_generator,
    ).mul_(0.1)
    weights_standard, weights_interleaved, weights_cutedsl = (_make_weights(
        experts_per_rank=experts_per_rank,
        intermediate_size=args.intermediate_size,
        hidden_size=args.hidden_size,
        device=device,
        seed=1729 + rank,
    ))

    input_buffer = input_handle = None
    if flashcomm is not None:
        layout = EPCommLayoutDesc()
        _, recv_count, layout = flashcomm._prepare_dispatch_buffers(inputs, topk_indices, layout)
        expert_counts = layout.recv_expert_counts
    else:
        # Routing/setup is outside the timed region, just as for FlashComm.
        routes = [torch.empty_like(topk_indices) for _ in range(world_size)]
        dist.all_gather(routes, topk_indices)
        routes = torch.stack(routes)
        counts, encoded_rows = [], []
        for expert in range(experts_per_rank):
            source, token, topk_slot = torch.where(routes == rank * experts_per_rank + expert)
            counts.append(token.numel())
            encoded = (source << 48) | (topk_slot << 32) | token
            padding = (-token.numel()) % CUTEDSL_EXPERT_ALIGNMENT
            encoded_rows.append(F.pad(encoded, (0, padding)))
        expert_counts = torch.tensor(counts, dtype=torch.int32, device=device)
        metadata = torch.cat(encoded_rows)
        layout = SimpleNamespace(token_src_rank_topk_and_indices=metadata)
        recv_count = torch.tensor(metadata.numel(), dtype=torch.int32, device=device)
        input_buffer = symm_mem.empty(inputs.shape, dtype=inputs.dtype, device=device)
        input_handle = symm_mem.rendezvous(input_buffer, group=ep_group)
        input_buffer.copy_(inputs)
        dist.barrier(device_ids=[local_rank])
    counts_cpu = tuple(int(x) for x in expert_counts.cpu().tolist())
    offsets = _padded_offsets(expert_counts, CUTEDSL_EXPERT_ALIGNMENT)
    padded_m = offsets[-1]
    if padded_m == 0:
        raise ValueError("routing produced no received tokens; lower drop_ratio")
    if padded_m % CUTEDSL_EXPERT_ALIGNMENT:
        raise AssertionError("CuTeDSL padded M is not tile aligned")

    metadata = layout.token_src_rank_topk_and_indices
    if metadata.numel() < padded_m:
        raise RuntimeError(f"metadata has {metadata.numel()} rows but padded M is {padded_m}")
    role_offsets = _padded_offsets(expert_counts, DispatchGroupGemmSwigluFuse.EXPERT_ALIGNMENT)
    role_padded_m = role_offsets[-1]
    role_metadata = torch.zeros((role_padded_m, ), dtype=metadata.dtype, device=device)
    for expert, count in enumerate(counts_cpu):
        if count:
            role_metadata[role_offsets[expert]:role_offsets[expert] + count].copy_(
                metadata[offsets[expert]:offsets[expert] + count])
    workspace = torch.empty(
        (role_padded_m, args.hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    input_peers = [
        flashcomm.overlap_context.dispatch_input_symm_tensor.get_peer_tensor(peer)
        if flashcomm is not None else input_handle.get_buffer(peer, inputs.shape, inputs.dtype)
        for peer in range(world_size)
    ]
    role_fused = DispatchGroupGemmSwigluFuse.create(
        compiled_kernel,
        input_peers=input_peers,
        workspace=workspace,
        weights_interleaved=weights_interleaved,
        token_src_rank_topk_and_indices=role_metadata,
        expert_counts=expert_counts,
        num_ctas=args.num_ctas,
    )
    cutedsl_a = torch.empty((padded_m, args.hidden_size), dtype=torch.bfloat16, device=device)
    cutedsl_out = torch.empty((padded_m, args.intermediate_size), dtype=torch.bfloat16, device=device)
    gemm_num_sm = flashcomm._resolve_gemm_num_sm(args.gemm_num_sm) if flashcomm is not None else 0
    return DispatchGroupGemmSwiglu(
        input_buffer=input_buffer,
        input_handle=input_handle,
        flashcomm=flashcomm,
        cuda_ep=cuda_ep,
        role_fused=role_fused,
        inputs=inputs,
        topk_indices=topk_indices,
        weights_standard=weights_standard,
        weights_cutedsl=weights_cutedsl,
        layout=layout,
        recv_count=recv_count,
        cutedsl_a=cutedsl_a,
        cutedsl_out=cutedsl_out,
        expert_counts=expert_counts,
        counts_cpu=counts_cpu,
        padded_offsets=offsets,
        role_padded_offsets=role_offsets,
        rank=rank,
        world_size=world_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        topk=args.topk,
        dispatch_num_stages=args.dispatch_num_stages,
        gemm_num_sm=gemm_num_sm,
    )


def _parse_args():
    p = argparse.ArgumentParser(description=("Benchmark LittleKernel role-fused pull dispatch + grouped "
                                             "FC1 GEMM + SwiGLU against the FlashComm CuTeDSL kernel"))
    p.add_argument("--num-tokens", type=int, default=4096)
    p.add_argument("--hidden-size", type=int, default=8192)
    p.add_argument("--intermediate-size", type=int, default=2560)
    p.add_argument("--num-experts", type=int, default=160)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--drop-ratio", type=float, default=0.1)
    p.add_argument("--num-sm", type=int, default=32)
    p.add_argument("--gemm-num-sm", type=int, default=None)
    p.add_argument(
        "--num-ctas",
        type=int,
        default=None,
        help="LittleKernel persistent CTA budget; default=min(tiles, SMs)",
    )
    p.add_argument("--num-worst-tokens", type=int, default=-1)
    p.add_argument("--dispatch-num-stages", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--check-iters", type=int, default=2)
    p.add_argument("--max-bf16-ulps", type=int, default=3)
    p.add_argument("--check-rtol", type=float, default=CHECK_RTOL)
    p.add_argument("--check-atol", type=float, default=CHECK_ATOL)
    p.add_argument("--no-check", action="store_true")
    p.add_argument(
        "--cache-dir",
        default=(f"{os.path.dirname(__file__)}/"
                 "lk_cache_role_v2_dispatch_group_gemm_swiglu"),
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument(
        "--trace-dir",
        default=(f"{os.path.dirname(__file__)}/"
                 "lk_trace_role_v2_dispatch_group_gemm_swiglu"),
    )
    p.add_argument("--trace-iters", type=int, default=5)
    p.add_argument("--trace-with-stack", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--merge-trace", action=argparse.BooleanOptionalAction, default=True)
    add_reference_argument(p)
    args = p.parse_args()
    if args.num_tokens <= 0:
        p.error("--num-tokens must be positive")
    if args.hidden_size <= 0 or args.intermediate_size <= 0:
        p.error("--hidden-size and --intermediate-size must be positive")
    if args.num_experts <= 0 or args.topk <= 0:
        p.error("--num-experts and --topk must be positive")
    if args.warmup < 0 or args.iters <= 0:
        p.error("--warmup must be non-negative and --iters must be positive")
    if args.check_iters <= 0 or args.trace_iters <= 0:
        p.error("--check-iters and --trace-iters must be positive")
    if args.max_bf16_ulps < 0:
        p.error("--max-bf16-ulps must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    _load_flashcomm(args.references)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if local_world_size != world_size:
        raise ValueError("this benchmark currently supports one node")
    ep_group = dist.group.WORLD

    compiled_kernel = DispatchGroupGemmSwigluFuse.build(cache_dir=args.cache_dir, verbose=args.verbose)

    # Compile once on rank 0 so shared-cache users do not race on the cubin.
    if rank == 0:
        _ = compiled_kernel.launcher
    dist.barrier(device_ids=[local_rank])
    if rank != 0:
        _ = compiled_kernel.launcher
    dist.barrier(device_ids=[local_rank])

    ctx = prepare(args, compiled_kernel, ep_group, create_cuda_reference=not args.no_check)
    torch.cuda.synchronize(ctx.device)
    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        print(f"args: {args}")
        print(f"cache_dir: {args.cache_dir}; "
              f"role_ctas={ctx.role_fused.num_ctas}; "
              f"cutedsl_gemm_sms={ctx.gemm_num_sm}; "
              f"valid_M={ctx.valid_m}, "
              f"CuTe_padded_M={ctx.padded_m}, "
              f"LK_padded_M={ctx.role_padded_m}")

    if not args.no_check:
        check_correctness(
            ctx,
            iterations=args.check_iters,
            max_bf16_ulps=args.max_bf16_ulps,
            rtol=args.check_rtol,
            atol=args.check_atol,
        )
    bench_compare(ctx, warmup=args.warmup, iters=args.iters)
    if args.trace:
        export_trace(
            ctx,
            args.trace_dir,
            merge=args.merge_trace,
            trace_iters=args.trace_iters,
            with_stack=args.trace_with_stack,
        )

    torch.cuda.synchronize(ctx.device)
    dist.barrier(device_ids=[local_rank])
    ctx.role_fused.release()
    ctx.input_handle = None
    ctx.input_buffer = None
    del ctx
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
