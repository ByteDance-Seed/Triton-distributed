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
"""Build, check, and benchmark the fused varlen SM100 GDN v2 kernel."""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import triton.testing

from benchmarks.nv.sm100.dataflow.bench_optional import add_reference_argument, optional_import
from examples.nv.sm100.gated_delta_net import BT, D, H, GDNVarlenV2

REFRESH_SECONDS = 0.4
CHECK_TOKENS = 4096
_FLA = None
_FLASHINFER = None


@dataclass(frozen=True)
class Shape:
    name: str
    lengths: tuple[int, ...]


SHAPES = (
    Shape("tiny_packed", (17, 31)),
    Shape("small_edges", (63, 64, 65)),
    Shape("medium_ragged", (511, 777, 1024, 1537)),
    Shape("large_5x8k", (7937, 8063, 8192, 8257, 8319)),
    Shape("throughput_128k", (8192, ) * 16),
    Shape("throughput_512k", (8192, ) * 64),
)


def _fla_reference(q, k, v, g, beta, h0, cu):
    return _FLA.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0,
        output_final_state=True,
        state_v_first=True,
        cu_seqlens=cu,
    )


def _flashinfer_target(q, k, v, g, beta, h0, cu):
    q3, k3, v3, beta2 = (tensor[0] for tensor in (q, k, v, beta))
    g2 = g[0].exp()
    out3 = torch.empty_like(q3)
    out = out3.unsqueeze(0)
    ht = torch.empty_like(h0)
    cu32 = cu.to(torch.int32)

    def launch():
        _FLASHINFER.chunk_gated_delta_rule_sm100(q3, k3, v3, g2, beta2, out3, cu32, h0, ht, 1.0 / math.sqrt(D))
        return out, ht

    return launch, out, ht


def _inputs(shape: Shape):
    total_t = sum(shape.lengths)
    torch.manual_seed(2026 + total_t + len(shape.lengths))
    q = F.normalize(torch.randn((1, total_t, H, D), device="cuda", dtype=torch.float32), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn((1, total_t, H, D), device="cuda", dtype=torch.float32), dim=-1).to(torch.bfloat16)
    v = torch.randn((1, total_t, H, D), device="cuda", dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn((1, total_t, H), device="cuda", dtype=torch.float32))
    beta = torch.sigmoid(torch.randn((1, total_t, H), device="cuda", dtype=torch.float32))
    h0 = torch.randn((len(shape.lengths), H, D, D), device="cuda", dtype=torch.float32) * 0.1
    cu = torch.tensor((0, *torch.tensor(shape.lengths).cumsum(0).tolist()), device="cuda", dtype=torch.int64)
    return q, k, v, g, beta, h0, cu


def torch_reference(q, k, v, g, beta, h0, cu):
    """FP32 chunk recurrence; no external kernels or BF16 state rounding.

    Solve (I + tril(beta * K K^T * decay, -1)) delta =
    beta * (V - exp(cumsum(g)) * K state), then update the recurrent state.
    Batch sequences and heads so long/varlen correctness remains practical.
    """
    out = torch.empty_like(v)
    state = h0.transpose(-1, -2).contiguous().clone()
    starts, ends = cu[:-1], cu[1:]
    max_length = int((ends - starts).max())
    offsets = torch.arange(BT, device=q.device)
    eye = torch.eye(BT, device=q.device)
    for start in range(0, max_length, BT):
        indices = starts[:, None] + start + offsets
        valid = indices < ends[:, None]
        indices = indices.clamp(max=q.shape[1] - 1)
        mask = valid[:, None, :, None]
        qc, kc, vc = (x[0, indices].permute(0, 2, 1, 3).float() * mask for x in (q, k, v))
        gc = (g[0, indices].permute(0, 2, 1) * valid[:, None]).cumsum(-1)
        bc = beta[0, indices].permute(0, 2, 1) * valid[:, None]
        # Mask before exp: upper-triangle differences may otherwise overflow.
        decay = (gc[..., :, None] - gc[..., None, :]).tril().exp().tril()
        system = eye + ((kc @ kc.transpose(-1, -2)) * decay * bc[..., None]).tril(-1)
        rhs = bc[..., None] * (vc - gc.exp()[..., None] * (kc @ state))
        delta = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
        qc = qc / math.sqrt(q.shape[-1])
        values = gc.exp()[..., None] * (qc @ state) + ((qc @ kc.transpose(-1, -2)) * decay) @ delta
        out[0, indices[valid]] = values.permute(0, 2, 1, 3)[valid].to(out.dtype)
        tail_decay = (gc[..., -1, None] - gc).exp()
        state = gc[..., -1, None, None].exp() * state + kc.transpose(-1, -2) @ (tail_decay[..., None] * delta)
    return out, state.transpose(-1, -2).contiguous()


def _refresh(run) -> None:
    torch.cuda.synchronize()
    time.sleep(REFRESH_SECONDS)
    run()
    torch.cuda.synchronize()


def _modeled_flops(shape: Shape) -> float:
    chunks = sum((length + BT - 1) // BT for length in shape.lengths)
    inverse_flops = 4 * (4 * 8**3 + 2 * 16**3 + 32**3)
    return float(chunks * H * (8 * BT * BT * D + 6 * BT * D * D + inverse_flops))


def _effective_bytes(shape: Shape) -> float:
    total_t = sum(shape.lengths)
    nseq = len(shape.lengths)
    return float(8 * total_t * H * D + 8 * total_t * H + 8 * nseq * H * D * D + 8 * (nseq + 1))


def _print_table(rows) -> None:
    headers = ("implementation", "median ms", "modeled TFLOP/s", "effective TB/s", "speedup")
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def _assert_output_close(actual, expected, *, rtol: float, atol: float) -> None:
    for start in range(0, actual.shape[1], CHECK_TOKENS):
        end = min(start + CHECK_TOKENS, actual.shape[1])
        try:
            torch.testing.assert_close(actual[:, start:end].float(), expected[:, start:end].float(), rtol=rtol,
                                       atol=atol)
        except AssertionError as error:
            raise AssertionError(f"output mismatch in token range [{start}, {end})") from error


def check_correctness(compiled_kernel, shape: Shape, inputs, flashinfer_out, flashinfer_ht) -> None:
    q, k, v, g, beta, h0, cu = inputs
    target = GDNVarlenV2.create(
        compiled_kernel,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        state_v_first=True,
        cu_seqlens=cu,
    )
    out, ht = target.launch()
    torch.cuda.synchronize()
    ref_o, ref_ht = (_fla_reference if _FLA is not None else torch_reference)(q, k, v, g, beta, h0, cu)
    torch.cuda.synchronize()
    _assert_output_close(out, ref_o, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(ht.float(), ref_ht.float(), rtol=2e-2, atol=3e-2)
    if flashinfer_out is not None:
        _assert_output_close(flashinfer_out, ref_o, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(flashinfer_ht.float(), ref_ht.float(), rtol=2e-2, atol=2e-2)
    reference_name = "FLA" if _FLA is not None else "PyTorch FP32"
    print(f"correctness: {shape.name} LittleKernel pass vs {reference_name}")


def bench_compare(compiled_kernel, shape: Shape, inputs, flashinfer, *, warmup: int, iters: int) -> None:
    q, k, v, g, beta, h0, cu = inputs
    target = GDNVarlenV2.create(
        compiled_kernel,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        state_v_first=True,
        cu_seqlens=cu,
    )
    implementations = []
    if _FLA is not None:
        implementations.append(("FLA", lambda: _fla_reference(q, k, v, g, beta, h0, cu)))
    if flashinfer is not None:
        implementations.append(("FlashInfer", flashinfer))
    implementations.append(("LittleKernel", target.launch))
    times = {}
    for name, run in implementations:
        _refresh(run)
        times[name] = triton.testing.do_bench(run, warmup=warmup, rep=iters, return_mode="median")

    flops = _modeled_flops(shape)
    effective_bytes = _effective_bytes(shape)
    rows = []
    for name, _ in implementations:
        ms = times[name]
        rows.append((name, f"{ms:.4f}", f"{flops / ms / 1e9:.2f}", f"{effective_bytes / ms / 1e9:.3f}",
                     f"{times[implementations[0][0]] / ms:.3f}x"))
    chunks = sum((length + BT - 1) // BT for length in shape.lengths)
    print(f"\n=== GDN {shape.name}: N={len(shape.lengths)}, total_T={sum(shape.lengths)}, "
          f"min_len={min(shape.lengths)}, max_len={max(shape.lengths)}, chunks={chunks}, H={H}, D={D} ===")
    _print_table(rows)


def export_trace(
    compiled_kernel,
    cases,
    trace_dir: str | Path,
    *,
    trace_iters: int,
    with_stack: bool,
) -> Path:
    from torch.profiler import ProfilerActivity, profile, record_function

    shape_runners = []
    for shape, inputs, flashinfer, _, _ in cases:
        q, k, v, g, beta, h0, cu = inputs
        target = GDNVarlenV2.create(
            compiled_kernel,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            state_v_first=True,
            cu_seqlens=cu,
        )
        runners = {"littlekernel_gdn": target.launch}
        if _FLA is not None:
            runners["fla_reference"] = lambda q=q, k=k, v=v, g=g, beta=beta, h0=h0, cu=cu: _fla_reference(
                q, k, v, g, beta, h0, cu)
        if flashinfer is not None:
            runners["flashinfer_gdn"] = flashinfer
        shape_runners.append((shape.name, runners))
        for run in runners.values():
            run()
    torch.cuda.synchronize()

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=with_stack,
    )
    prof.start()
    for shape_name, runners in shape_runners:
        for label, run in runners.items():
            for _ in range(trace_iters):
                with record_function(f"{shape_name}/{label}"):
                    run()
    prof.stop()

    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_name = cases[0][0].name if len(cases) == 1 else "all"
    trace = trace_dir / f"gdn_{trace_name}.json.gz"
    prof.export_chrome_trace(str(trace))
    print(f"trace: {trace}")
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fused varlen role-proposal GDN v2")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--cache-dir", default=f"{os.path.dirname(__file__)}/lk_cache_role_v2_gdn")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--shape", choices=("all", *(shape.name for shape in SHAPES)), default="all")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--trace", action="store_true", help="Export a compressed torch.profiler trace")
    parser.add_argument("--trace-dir", default=f"{os.path.dirname(__file__)}/lk_trace_role_v2_gdn")
    parser.add_argument("--trace-iters", type=int, default=5)
    parser.add_argument("--trace-with-stack", action=argparse.BooleanOptionalAction, default=False)
    add_reference_argument(parser)
    args = parser.parse_args()
    if args.trace_iters <= 0:
        parser.error("--trace-iters must be positive")

    global _FLA, _FLASHINFER
    os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
    _FLA = optional_import("fla.ops.gated_delta_rule", "FLA_PYTHON_PATH",
                           [os.environ["FLA_ROOT"]] if os.environ.get("FLA_ROOT") else (), mode=args.references)
    _FLASHINFER = optional_import("flashinfer.gdn_kernels.blackwell.gdn_prefill", "FLASHINFER_PYTHON_PATH",
                                  [os.environ["FLASHINFER_ROOT"]] if os.environ.get("FLASHINFER_ROOT") else (),
                                  mode=args.references)
    compiled_kernel = GDNVarlenV2.build(cache_dir=args.cache_dir, verbose=args.verbose)
    selected = SHAPES if args.shape == "all" else tuple(shape for shape in SHAPES if shape.name == args.shape)
    print(f"cache_dir: {args.cache_dir}")
    cases = []
    for shape in selected:
        inputs = _inputs(shape)
        flashinfer = flashinfer_out = flashinfer_ht = None
        if _FLASHINFER is not None:
            flashinfer, flashinfer_out, flashinfer_ht = _flashinfer_target(*inputs)
            flashinfer()
            torch.cuda.synchronize()
        if not args.no_check:
            check_correctness(compiled_kernel, shape, inputs, flashinfer_out, flashinfer_ht)
        bench_compare(compiled_kernel, shape, inputs, flashinfer, warmup=args.warmup, iters=args.iters)
        if args.trace:
            cases.append((shape, inputs, flashinfer, flashinfer_out, flashinfer_ht))
    if args.trace:
        export_trace(
            compiled_kernel,
            cases,
            args.trace_dir,
            trace_iters=args.trace_iters,
            with_stack=args.trace_with_stack,
        )


if __name__ == "__main__":
    main()
