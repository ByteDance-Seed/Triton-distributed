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

import argparse
import random
from typing import Dict

import torch

from flash_comm.ep import EPKernels as _CudaEPKernels
from flash_comm.ep.ep_kernels import EPCommLayoutDesc
from flash_comm.ep_overlap import EPOverlapKernels
from flash_comm.utils.test_utils import (
    LOCAL_WORLD_SIZE,
    RANK,
    WORLD_SIZE,
    bitwise_equal,
    cool_down,
    export_trace,
    generate_random_exp_indices,
    get_torch_prof_ctx,
    init_dist,
    perf_event_ms,
    record_function_range,
)

from _torch_refs import grouped_matmul_bf16


def counts_to_padded_offsets(expert_counts: torch.Tensor, pad_to: int) -> torch.Tensor:
    counts_i64 = expert_counts.to(torch.int64)
    padded = ((counts_i64 + pad_to - 1) // pad_to) * pad_to
    out = torch.empty(expert_counts.shape[0] + 1, dtype=torch.int64, device=expert_counts.device)
    out[0].zero_()
    torch.cumsum(padded, dim=0, out=out[1:])
    return out


# Untimed FC1 prep so the timed window only covers the FC2 fused kernel.


def make_topk_weights(args, token_num: int) -> torch.Tensor:
    """Per-rank FP32 ``(token_num, topk)`` weight matrix.

    Used by ``--has_weight`` to exercise the optional combine weight
    push side channel. The values are bit-aliased through Int32 in
    flight (no arithmetic), so a bitwise round-trip check is valid.
    """
    return torch.randn(
        token_num,
        args.topk,
        dtype=torch.float32,
        device="cuda",
    )


def prepare_dispatch(args, ep_kernels: EPOverlapKernels, B_fc1, input_data, exp_indices, topk_weights=None) -> Dict:
    """Untimed dispatch + FC1 setup for the combine perf/correctness loop.

    When ``topk_weights`` is supplied we route via the fused
    ``dispatch_group_gemm`` kernel so the test can capture the 1D
    ``dispatch_weights`` vector (one FP32 scalar per dispatched row)
    in addition to the FC1 output. The fused path is bit-equivalent to
    ``dispatch_cutedsl`` + ``group_gemm`` (covered by the dispatch
    test suite), so swapping it in here does not perturb downstream
    numerics.  The weightless path keeps the legacy two-step flow so
    perf-only smoke tests stay unaffected.
    """
    layout_desc = EPCommLayoutDesc()
    if topk_weights is not None:
        A_fc1, dispatch_weights, layout_desc = ep_kernels.dispatch_group_gemm(
            input_data,
            exp_indices,
            B_fc1,
            layout_desc,
            gemm_num_sm=args.gemm_num_sm,
            comm_num_sm=args.comm_num_sm,
            topk_weights=topk_weights,
        )
    else:
        A_dispatch, layout_desc = ep_kernels.dispatch_cutedsl(
            input_data,
            exp_indices,
            layout_desc,
            comm_num_sm=args.comm_num_sm,
        )
        A_fc1 = ep_kernels.group_gemm(
            A_dispatch,
            B_fc1,
            layout_desc.recv_expert_counts,
            gemm_num_sm=args.gemm_num_sm,
        )
        dispatch_weights = None

    # Clone layout tensors so successive cases do not alias each other's
    # symmetric metadata slices.
    for attr in ("token_src_rank_topk_and_indices", "recv_token_count", "recv_aligned_token_count",
                 "recv_expert_counts", "topk_indices"):
        v = getattr(layout_desc, attr, None)
        if isinstance(v, torch.Tensor):
            setattr(layout_desc, attr, v.clone())
    expert_counts = layout_desc.recv_expert_counts.clone()
    padded_offsets = counts_to_padded_offsets(
        expert_counts,
        ep_kernels.cluster_tile_m,
    )
    gemm_actual_padded = int(padded_offsets[-1].item())
    recv_unpadded = int(layout_desc.recv_token_count[RANK].item())

    # SwiGLU-stand-in: elementwise gate*up suffices for byte-exact tests.
    gate, up = A_fc1.split(args.intermediate_size, dim=-1)
    A_fc2_in = (gate * up).contiguous()
    # Clone dispatch_weights so reusing the symmetric staging buffer in
    # later cases cannot corrupt the captured 1D vector we hand to
    # ``group_gemm_combine``.
    return {
        "A": A_fc2_in,
        "A_fc1": A_fc1.clone(),
        "layout": layout_desc,
        "expert_counts": expert_counts,
        "padded_offsets": padded_offsets,
        "total_padded": A_fc1.shape[0],
        "gemm_actual_padded": gemm_actual_padded,
        "recv_unpadded": recv_unpadded,
        "dispatch_weights": (dispatch_weights.clone() if dispatch_weights is not None else None),
        "topk_weights": topk_weights,
    }


def make_baseline_runners(args, ep_kernels: EPOverlapKernels, prep: Dict, B_fc2, scratch_C: torch.Tensor,
                          prebuilt_C: torch.Tensor, *, record_fused_call: bool = False) -> Dict[str, callable]:
    layout_desc = prep["layout"]

    def gemm_only():
        return ep_kernels.group_gemm(
            prep["A"],
            B_fc2,
            prep["expert_counts"],
            output=scratch_C,
            gemm_num_sm=args.gemm_num_sm,
        )

    def combine_only():
        # Full combine on a pre-baked C tensor (no FC2 GEMM cost).
        return ep_kernels.combine_cutedsl(
            prebuilt_C,
            layout_desc=layout_desc,
            push_mode="tile",
            tile_m=args.combine_tile_m,
            tile_n=args.combine_tile_n,
            comm_num_sm=args.comm_num_sm,
        )

    def reduce_only():
        return ep_kernels.topk_reduce_only(
            topk_indices=layout_desc.topk_indices,
            use_ggc_staging=True,
        )

    def sequential():
        # gemm -> combine (apples-to-apples with ``fused_e2e``).
        C = ep_kernels.group_gemm(
            prep["A"],
            B_fc2,
            prep["expert_counts"],
            output=scratch_C,
            gemm_num_sm=args.gemm_num_sm,
        )
        return ep_kernels.combine_cutedsl(
            C,
            layout_desc=layout_desc,
            push_mode="tile",
            tile_m=args.combine_tile_m,
            tile_n=args.combine_tile_n,
            comm_num_sm=args.comm_num_sm,
        )

    # Threading optional dispatched_weights through the two fused
    # runners so timing/correctness paths both exercise the
    # ``has_weight`` side channel when --has_weight is set. We always
    # pull the captured 1D weight vector from ``prep`` (None when
    # disabled), so the runner closures stay stable across perf rounds.
    dispatched_weights = prep.get("dispatch_weights")

    def fused_kernel():
        # Fused kernel only -- skip the post-push NVL barrier + reduce.
        # The returned weight view is unsafe to read here (no barrier);
        # we drop it so timing stays apples-to-apples.
        with record_function_range("group_gemm_combine.run", record_fused_call):
            _none, _w = ep_kernels.group_gemm_combine(
                A_padded=prep["A"],
                B=B_fc2,
                layout_desc=layout_desc,
                run_reduce=False,
                gemm_num_sm=args.gemm_num_sm,
                dispatched_weights=dispatched_weights,
            )
        return None

    def fused_e2e():
        with record_function_range("group_gemm_combine.run", record_fused_call):
            out, w = ep_kernels.group_gemm_combine(
                A_padded=prep["A"],
                B=B_fc2,
                layout_desc=layout_desc,
                gemm_num_sm=args.gemm_num_sm,
                dispatched_weights=dispatched_weights,
            )
        return (out, w) if w is not None else out

    return {
        "group_gemm_only": gemm_only,
        "combine_only": combine_only,
        "reduce_only": reduce_only,
        "sequential": sequential,
        "fused_kernel": fused_kernel,
        "fused_e2e": fused_e2e,
    }


# Independent reference: CUDA dispatch + torch.matmul (FC1, FC2) +
# CUDA combine. Per-component byte equality is established elsewhere
# (cutedsl tests); this is the integration witness for the full pipeline.


def run_torch_cuda_e2e_reference(args, cuda_kernels: _CudaEPKernels, ep_kernels: EPOverlapKernels,
                                 B_fc1_raw: torch.Tensor, B_fc2_raw: torch.Tensor, input_data, exp_indices):
    """Full e2e via CUDA dispatch + torch.matmul (FC1, FC2) + CUDA combine.

    ``B_fc1_raw`` is ``(E, 2*I, H)``; ``B_fc2_raw`` is ``(E, H, I)``.
    Returns the final dense output tensor. The weight side channel is
    not exercised here (combine in this path collapses the topk
    dimension via an unweighted sum); :func:`_verify_combine_weights`
    independently round-trip-checks the fused kernel's W output
    against the original ``topk_weights`` input.
    """
    layout_desc = EPCommLayoutDesc()
    disp_out, weights, layout_desc = cuda_kernels.dispatch(
        input_data,
        exp_indices,
        None,
        layout_desc,
    )
    A_padded, _, layout_desc = cuda_kernels.dispatch_postprocess(
        disp_out.clone(),
        weights,
        layout_desc,
    )
    expert_counts = layout_desc.recv_expert_counts.clone()
    padded_offsets = counts_to_padded_offsets(
        expert_counts,
        ep_kernels.cluster_tile_m,
    )

    # FC1 group-GEMM via torch.matmul (bf16 in / bf16 out).
    A_fc1 = grouped_matmul_bf16(
        A_padded,
        B_fc1_raw,
        expert_counts,
        padded_offsets,
    )
    gate, up = A_fc1.split(args.intermediate_size, dim=-1)
    A_fc2_in = (gate * up).contiguous()
    C = grouped_matmul_bf16(
        A_fc2_in,
        B_fc2_raw,
        expert_counts,
        padded_offsets,
    )

    # CUDA combine: stage into the symmetric combine buffer, fill the
    # send mask, and invoke production combine.
    buf = cuda_kernels.get_combine_buffer(C.shape[0], dtype=C.dtype)
    buf.copy_(C)
    layout_desc.token_topk_send_mask.fill_(1)
    out, _ = cuda_kernels.combine(buf, layout_desc)
    return out


def run_gemm_num_sm_sweep(args, ep_kernels, B_fc1, B_fc2) -> None:
    """Bitwise check across ``gemm_num_sm`` budgets on one fixed case.

    Both FC1 (in ``prepare_dispatch``) and FC2 (inside
    ``group_gemm_combine``) are M-contig cluster-persistent GEMMs with
    no split-K, so ``gemm_num_sm`` controls only how many clusters are
    resident -- the per-tile accumulation order is invariant. The
    final combine output is dense ``(num_tokens, n_out)`` (no
    cluster-tile tail padding), so a direct bitwise compare is well
    defined for every byte.

    Inputs are generated *once* outside the per-value sample fn so that
    each ``gemm_num_sm`` value is exercised on identical data; reusing
    the same buffers also ensures peer ranks pull-read the same staged
    input at every iteration.
    """
    sweep_values = list({args.gemm_num_sm, 64, 136, args.gemm_num_sm_sweep_n})
    sweep_values = [v for v in sweep_values if v != args.gemm_num_sm]
    if not sweep_values:
        return
    if RANK == 0:
        print(f"\n--- gemm_num_sm sweep "
              f"(reference={args.gemm_num_sm}, "
              f"others={sweep_values}) ---")

    torch.manual_seed(2026 + RANK)
    exp_indices = generate_random_exp_indices(
        args.num_tokens,
        args.num_experts,
        args.topk,
        args.drop_ratio,
    ).cuda()
    input_data = torch.randn(
        args.num_tokens,
        args.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )

    def _run(value):
        prep = prepare_dispatch(args, ep_kernels, B_fc1, input_data, exp_indices)
        out, _w = ep_kernels.group_gemm_combine(
            A_padded=prep["A"],
            B=B_fc2,
            layout_desc=prep["layout"],
            gemm_num_sm=value,
        )
        return out.clone()

    ref = _run(args.gemm_num_sm)
    for v in sweep_values:
        out = _run(v)
        if not bitwise_equal(ref, out):
            diff = (ref.float() - out.float()).abs()
            raise AssertionError(f"gemm_num_sm sweep: gemm_num_sm={v} not byte-equal to "
                                 f"reference (gemm_num_sm={args.gemm_num_sm}); "
                                 f"max_abs={diff.max().item():.6e}")
        if RANK == 0:
            print(f"  gemm_num_sm={v}: bitwise=PASS")
    if RANK == 0:
        print("gemm_num_sm sweep PASSED.\n")


def _verify_combine_weights(case_label: str, expected: torch.Tensor, got: torch.Tensor, exp_indices: torch.Tensor,
                            drop_sentinel: int) -> None:
    """Assert ``got[i, k]`` matches the FlashComm
    ``kernel_combine_intranode_v2`` semantics: valid slots
    (``exp_indices[i, k] != drop_sentinel``) carry the byte-equal
    transported weight; dropped slots carry exactly ``0``.

    The transport is an ``Int{W}`` alias of the source dtype with no
    arithmetic, so we expect byte-exact equality on every valid slot
    regardless of drop ratio or receive-side scheduling order. The
    final mask-and-copy kernel (see
    :func:`masked_weight_copy`) materialises the ``0``-on-drop
    contract and is exercised by this check end-to-end.
    """
    mask = (exp_indices != drop_sentinel)
    valid_count = int(mask.sum().item())
    # Valid-slot byte-equality.
    if valid_count > 0:
        exp_valid = expected[mask]
        got_valid = got[mask]
        if not bitwise_equal(exp_valid, got_valid):
            diff = (exp_valid - got_valid).abs()
            raise AssertionError(f"{case_label}: combine_weights byte mismatch on valid "
                                 f"slots; max_abs={diff.max().item():.3e} "
                                 f"({valid_count} valid entries)")
    # Drop-slot zeroing (FlashComm parity).
    drop_mask = ~mask
    drop_count = int(drop_mask.sum().item())
    if drop_count > 0:
        got_drop = got[drop_mask]
        zero_ref = torch.zeros_like(got_drop)
        if not bitwise_equal(got_drop, zero_ref):
            nz = (got_drop != 0).nonzero(as_tuple=False)
            raise AssertionError(f"{case_label}: combine_weights drop slots must be 0 "
                                 f"(FlashComm parity); {nz.shape[0]} non-zero entries "
                                 f"out of {drop_count} drop slots")


def run_check(args, ep_kernels, cuda_kernels, B_fc1, B_fc2, B_fc1_raw, B_fc2_raw):
    if RANK == 0:
        weight_tag = "  has_weight=1" if args.has_weight else ""
        print(f"\n=== correctness phase: rounds={args.rounds} "
              f"verify_iters={args.verify_iters}{weight_tag} ===")
    cases_total = 0
    for round_idx in range(args.rounds):
        # Pre-generate deterministic per-case input data so every phase
        # below sees identical inputs case-by-case.
        case_inputs = []
        for case_idx in range(args.verify_iters):
            token_num = (args.num_tokens if round_idx == 0 and case_idx == 0 else random.randint(
                max(1, args.num_tokens // 2), args.num_tokens))
            exp_indices = generate_random_exp_indices(
                token_num,
                args.num_experts,
                args.topk,
                args.drop_ratio,
            ).cuda()
            input_data = torch.randn(
                token_num,
                args.hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            )
            topk_weights = (make_topk_weights(args, token_num) if args.has_weight else None)
            case_inputs.append((case_idx, token_num, input_data, exp_indices, topk_weights))

        def _run_baseline(name):
            outs, weight_outs, padded = [], [], []
            for (case_idx, token_num, input_data, exp_indices, topk_weights) in case_inputs:
                torch.distributed.barrier()
                prep = prepare_dispatch(
                    args,
                    ep_kernels,
                    B_fc1,
                    input_data,
                    exp_indices,
                    topk_weights=topk_weights,
                )
                scratch_C = torch.empty(
                    prep["total_padded"],
                    args.hidden_size,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                runners = make_baseline_runners(
                    args,
                    ep_kernels,
                    prep,
                    B_fc2,
                    scratch_C=scratch_C,
                    prebuilt_C=scratch_C,
                )
                result = runners[name]()
                # ``fused_e2e`` with weights returns
                # ``(output, combine_weights)``; other paths return a
                # bare output tensor.
                if isinstance(result, tuple):
                    out, weights = result
                    weight_outs.append(weights.clone())
                else:
                    weight_outs.append(None)
                    out = result
                outs.append(out.clone())
                padded.append(prep["total_padded"])
            torch.cuda.synchronize()
            torch.distributed.barrier()
            return outs, weight_outs, padded

        # Phase A: ``sequential`` -- the byte-exact target for fused_e2e
        # and torch_cuda_ref.
        seq_outs, _, padded_M = _run_baseline("sequential")
        # Phase B: ``fused_e2e`` (apples-to-apples with sequential).
        fused_outs, fused_weights, _ = _run_baseline("fused_e2e")
        # Phase C: independent torch+CUDA reference.
        torch_cuda_outs = []
        for case_idx, _, input_data, exp_indices, _ in case_inputs:
            torch.distributed.barrier()
            torch_cuda_outs.append(
                run_torch_cuda_e2e_reference(
                    args,
                    cuda_kernels,
                    ep_kernels,
                    B_fc1_raw,
                    B_fc2_raw,
                    input_data,
                    exp_indices,
                ).clone())
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # Phase D: unified bit-exact comparison.
        for ((case_idx, token_num, _, exp_indices, topk_weights), seq_out, fused_out, torch_out, fused_w,
             pm) in zip(case_inputs, seq_outs, fused_outs, torch_cuda_outs, fused_weights, padded_M):
            if not bitwise_equal(seq_out, fused_out):
                diff = (seq_out.float() - fused_out.float()).abs()
                rel = (diff / (seq_out.float().abs() + 1e-9)).max().item()
                raise AssertionError(f"FAIL fused_e2e rank={RANK} round={round_idx} "
                                     f"case={case_idx} tokens={token_num} "
                                     f"max_abs={diff.max().item():.6e} max_rel={rel:.6e}")
            if not bitwise_equal(seq_out, torch_out):
                diff = (seq_out.float() - torch_out.float()).abs()
                rel = (diff / (seq_out.float().abs() + 1e-9)).max().item()
                raise AssertionError(f"FAIL torch_cuda_ref rank={RANK} round={round_idx} "
                                     f"case={case_idx} tokens={token_num} "
                                     f"max_abs={diff.max().item():.6e} max_rel={rel:.6e}")
            weight_tag = ""
            if args.has_weight and fused_w is not None:
                _verify_combine_weights(
                    f"rank={RANK} round={round_idx} "
                    f"case={case_idx} tokens={token_num}",
                    expected=topk_weights,
                    got=fused_w,
                    exp_indices=exp_indices,
                    drop_sentinel=args.num_experts,
                )
                weight_tag = "  combine_weights=bitwise=PASS"
            if RANK == 0:
                print(f"round {round_idx} case {case_idx}: "
                      f"tokens={token_num}  total_padded={pm}  "
                      f"fused_e2e=bitwise=PASS  "
                      f"torch_cuda_ref=bitwise=PASS"
                      f"{weight_tag}")
            cases_total += 1

    if RANK == 0:
        print(f"=== correctness PASSED ({cases_total} cases) ===\n")


_GEMM_BASELINES = {"group_gemm_only", "sequential", "fused_kernel", "fused_e2e"}
_COMBINE_BASELINES = {"combine_only", "sequential", "fused_kernel", "fused_e2e"}
_REDUCE_BASELINES = {"reduce_only"}


def compute_metrics(args, prep, *, num_local_tokens, dtype_bytes=2):
    """FLOPs / byte counters for the perf headline (TFLOPs use valid M)."""
    valid_M = int(prep["expert_counts"].sum().item())
    padded_M = prep["gemm_actual_padded"]
    recv_unpadded = prep["recv_unpadded"]
    return {
        "num_local_tokens": num_local_tokens,
        "valid_M": valid_M,
        "padded_M": padded_M,
        "total_padded": prep["total_padded"],
        "recv_unpadded": recv_unpadded,
        "fc2_N": args.hidden_size,
        "fc2_K": args.intermediate_size,
        # FC2 GEMM FLOPs: 2 * M_valid * N * K (FMA = 2 ops).
        "gemm_flops": 2 * valid_M * args.hidden_size * args.intermediate_size,
        # Per-rank combine bytes = received rows we push back * hidden * 2B.
        "combine_bytes": recv_unpadded * args.hidden_size * dtype_bytes,
        # Reduce reads ``num_local * topk`` rows and writes ``num_local`` rows.
        "reduce_bytes": num_local_tokens * (args.topk + 1) * args.hidden_size * dtype_bytes,
    }


def print_workload_header(round_id, m, *, has_weight=False):
    weight_tag = "  has_weight=1" if has_weight else ""
    print(f"[RANK {RANK}] workload round {round_id}: "
          f"group_gemm valid_M={m['valid_M']} (padded_M={m['padded_M']}) "
          f"N={m['fc2_N']} K={m['fc2_K']}  "
          f"gemm_flops={m['gemm_flops'] / 1e12:.3f}TF  "
          f"combine recv_rows={m['recv_unpadded']}  "
          f"combine_bytes={m['combine_bytes'] / 1e9:.3f}GB  "
          f"local_tokens={m['num_local_tokens']}  "
          f"reduce_bytes={m['reduce_bytes'] / 1e9:.3f}GB"
          f"{weight_tag}")


def annotate_baseline(name: str, ms: float, m: Dict) -> str:
    parts = [f"{name}={ms:.4f}ms"]
    if name in _GEMM_BASELINES:
        tflops = m["gemm_flops"] / (ms * 1e-3) / 1e12 if ms > 0 else 0.0
        parts.append(f"gemm_tflops={tflops:.2f}")
    if name in _COMBINE_BASELINES:
        bw = m["combine_bytes"] / ms / 1e6 if ms > 0 else 0.0
        parts.append(f"combine_bw={bw:.2f}GB/s")
    if name in _REDUCE_BASELINES:
        bw = m["reduce_bytes"] / ms / 1e6 if ms > 0 else 0.0
        parts.append(f"reduce_bw={bw:.2f}GB/s")
    return " ".join(parts)


def print_overlap_analysis(round_id, metrics, timings):
    """Headline overlap savings + phase decomposition."""
    need = {"group_gemm_only", "combine_only", "sequential", "fused_e2e"}
    if not need.issubset(timings):
        return
    gemm_ms = timings["group_gemm_only"]
    comb_ms = timings["combine_only"]
    seq_ms = timings["sequential"]
    e2e_ms = timings["fused_e2e"]
    reduce_ms = timings.get("reduce_only")
    kernel_ms = timings.get("fused_kernel")

    overlap_savings = seq_ms - e2e_ms
    ideal_lb = max(gemm_ms, comb_ms)
    print(f"[RANK {RANK}] overlap analysis round {round_id}: "
          f"seq={seq_ms:.4f}  fused_e2e={e2e_ms:.4f}  "
          f"savings={overlap_savings:.4f}ms "
          f"({overlap_savings / seq_ms * 100:.1f}%)  "
          f"perfect_overlap_lb={ideal_lb:.4f}ms")
    # Subtract reduce from combine_only and fused_e2e to surface the
    # push-only times; if fused_kernel is timed, also report the
    # e2e barrier+reduce tail.
    if reduce_ms is not None:
        comb_push_ms = max(comb_ms - reduce_ms, 1e-9)
        e2e_push_ms = max(e2e_ms - reduce_ms, 1e-9)
        theoretical_ms = max(gemm_ms, comb_push_ms) + reduce_ms
        bottleneck = "gemm" if gemm_ms >= comb_push_ms else "combine_push"
        extra = ""
        if kernel_ms is not None:
            tail_ms = max(e2e_ms - kernel_ms, 0.0)
            extra = (f"  fused_kernel={kernel_ms:.4f}ms  "
                     f"e2e_tail(barrier+reduce)={tail_ms:.4f}ms")
        print(f"[RANK {RANK}] phase decomp round {round_id}: "
              f"reduce={reduce_ms:.4f}ms  "
              f"combine_push_only={comb_push_ms:.4f}ms  "
              f"fused_e2e_push_only={e2e_push_ms:.4f}ms  "
              f"theoretical_e2e={theoretical_ms:.4f}ms "
              f"(bottleneck={bottleneck})  "
              f"e2e/theory={theoretical_ms / e2e_ms * 100:.1f}%"
              f"{extra}")


def run_perf_round(args, ep_kernels, B_fc1, B_fc2, round_id: int, *, profile: bool, baselines):
    ep_group = ep_kernels.ep_group
    token_num = (args.num_tokens if round_id == 0 else random.randint(max(1, args.num_tokens // 2), args.num_tokens))
    if RANK == 0:
        weight_tag = "  has_weight=1" if args.has_weight else ""
        print(f"\n--- perf round {round_id}: tokens={token_num}"
              f"{weight_tag} ---")
    exp_indices = generate_random_exp_indices(
        token_num,
        args.num_experts,
        args.topk,
        args.drop_ratio,
    ).cuda()
    input_data = torch.randn(
        token_num,
        args.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_weights = (make_topk_weights(args, token_num) if args.has_weight else None)
    nvtx_prefix = f"rank{RANK}.round{round_id}" if profile else None

    torch.distributed.barrier()
    prep = prepare_dispatch(args, ep_kernels, B_fc1, input_data, exp_indices, topk_weights=topk_weights)
    scratch_C = torch.empty(
        prep["total_padded"],
        args.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # ``combine_only`` runs combine on a fixed C; pre-bake outside timing.
    prebuilt_C = ep_kernels.group_gemm(
        prep["A"],
        B_fc2,
        prep["expert_counts"],
        output=torch.empty_like(scratch_C),
        gemm_num_sm=args.gemm_num_sm,
    )
    # Warm the ggc staging buffer for ``reduce_only``. Pass through
    # ``dispatched_weights`` so the warm-up exercises the same kernel
    # path that the timed iterations will use. Keep this outside the
    # profiler so compile/warmup does not create a separate trace or skew
    # the reported timing pass.
    _ = ep_kernels.group_gemm_combine(
        A_padded=prep["A"],
        B=B_fc2,
        layout_desc=prep["layout"],
        gemm_num_sm=args.gemm_num_sm,
        dispatched_weights=prep.get("dispatch_weights"),
    )
    torch.cuda.synchronize()
    torch.distributed.barrier()

    runners = make_baseline_runners(
        args,
        ep_kernels,
        prep,
        B_fc2,
        scratch_C=scratch_C,
        prebuilt_C=prebuilt_C,
    )
    runners = {k: v for k, v in runners.items() if k in baselines}

    timings = {}
    for name, fn in runners.items():
        _, stats = perf_event_ms(
            fn,
            iters=args.iters,
            warmup_iters=args.warmup_iters,
            label=None,
            device_sleep_cycles=args.device_sleep_cycles,
            ep_group=ep_group,
            inter_iter_barrier=False,
            return_stats=True,
        )
        timings[name] = stats["mean"]
        # Track best/median per baseline. The min/median line up with
        # Nsight kernel-trace numbers (= the historical perf table); mean
        # is what the loop average sees.
        timings.setdefault("_stats", {})[name] = stats
        # Per-baseline cooldown to avoid clock-throttling bleed.
        if args.cooldown_s > 0:
            cool_down(args.cooldown_s)

    torch.cuda.synchronize()
    torch.distributed.barrier()

    if profile:
        profile_runners = make_baseline_runners(
            args,
            ep_kernels,
            prep,
            B_fc2,
            scratch_C=scratch_C,
            prebuilt_C=prebuilt_C,
            record_fused_call=True,
        )
        profile_runners = {k: v for k, v in profile_runners.items() if k in baselines}
        with get_torch_prof_ctx(True) as prof:
            for name, fn in profile_runners.items():
                label = f"{nvtx_prefix}.profile.{name}" if nvtx_prefix is not None else None
                perf_event_ms(
                    fn,
                    iters=args.iters,
                    warmup_iters=args.warmup_iters,
                    label=label,
                    device_sleep_cycles=args.device_sleep_cycles,
                    ep_group=ep_group,
                    inter_iter_barrier=False,
                    return_stats=True,
                )
                if args.cooldown_s > 0:
                    cool_down(args.cooldown_s)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        trace = export_trace(
            prof,
            label="ep_overlap_group_gemm_combine",
            rank=RANK,
            round_id=round_id,
            profile_dir=args.profile_dir or None,
        )
        if RANK == 0:
            print(f"  trace written to {trace}")

    if RANK == 0:
        metrics = compute_metrics(args, prep, num_local_tokens=token_num)
        print_workload_header(round_id, metrics, has_weight=args.has_weight)
        stats_table = timings.pop("_stats", {})
        for name, ms in timings.items():
            print(f"[RANK {RANK}] round {round_id}: " + annotate_baseline(name, ms, metrics))
            st = stats_table.get(name)
            if st is not None:
                # Min/median align with Nsight kernel-trace samples; the
                # mean reported above is what the event-loop sees once
                # per-iter NCCL serialisation noise is folded in.
                print(f"[RANK {RANK}] round {round_id}:   "
                      f"{name} stats min={st['min']:.4f} "
                      f"p10={st['p10']:.4f} median={st['median']:.4f} "
                      f"max={st['max']:.4f} (ms)")
        print_overlap_analysis(round_id, metrics, timings)


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------

# Default order, left-to-right: standalone phases -> the non-overlap
# sequential reference -> production overlap path at both kernel-only
# and end-to-end granularity.
_DEFAULT_BASELINES = ("group_gemm_only,combine_only,reduce_only,"
                      "sequential,fused_kernel,fused_e2e")


def parse_args():
    parser = argparse.ArgumentParser()
    # MoE geometry (unified across tests/ep_overlap).
    parser.add_argument("--num_tokens", type=int, default=4096, help="Per-rank input token count (was -M).")
    parser.add_argument("--hidden_size", type=int, default=8192, help="MoE hidden dim H.")
    parser.add_argument(
        "--intermediate_size", type=int, default=2560, help="MoE intermediate dim I. FC1 weight is "
        "(experts_per_rank, 2*I, H); FC2 weight is "
        "(experts_per_rank, H, I).")
    parser.add_argument("--num_experts", type=int, default=160)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--drop_ratio", type=float, default=0.1)
    # Kernel / runtime knobs.
    parser.add_argument(
        "--gemm_num_sm", type=int, default=None, help="Per-call GEMM scheduler SM budget. None (default)"
        " = full device SM count (queried from torch).")
    parser.add_argument("--comm_num_sm", type=int, default=None,
                        help="Per-call dispatch/combine SM budget. None = full device.")
    parser.add_argument("--combine_tile_m", type=int, default=128)
    parser.add_argument("--combine_tile_n", type=int, default=256)
    parser.add_argument("--num_worst_tokens", type=int, default=131072)
    # Test loop.
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=50, help="Untimed warmup iters per baseline before "
                        "perf measurement (avoids cold-start bias).")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--verify_iters", type=int, default=4)
    parser.add_argument("--cooldown_s", type=float, default=0.001)
    parser.add_argument(
        "--device_sleep_cycles", type=int, default=2_000_000, help="torch.cuda._sleep cycles inserted between "
        "successive perf iters to prevent SM "
        "frequency throttling on sustained heavy "
        "workloads (default 2e6 ~= 1.4 ms at 1.5 GHz; "
        "set 0 to disable).")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_dir", type=str, default="")
    parser.add_argument(
        "--baselines", type=str, default=_DEFAULT_BASELINES, help="Comma-separated subset of the available "
        "baselines.  Categories: "
        "{group_gemm_only,combine_only,reduce_only} "
        "standalone-phase isolators; "
        "{sequential} non-overlap reference "
        "(group_gemm + combine_cutedsl); "
        "{fused_kernel,fused_e2e} production overlap: "
        "fused_kernel = kernel-only (barrier + reduce "
        "excluded), fused_e2e = full pipeline "
        "(kernel + barrier + topk_reduce).")
    parser.add_argument(
        "--gemm_num_sm_sweep", action="store_true", help="Append a bitwise sweep over multiple GEMM SM "
        "budgets to the check phase (each distinct value "
        "triggers a fresh CuTeDSL compile, so this is "
        "off-by-default).")
    parser.add_argument("--gemm_num_sm_sweep_n", type=int, default=96,
                        help="Extra SM budget probed by the sweep (in addition "
                        "to 64 and 136).")
    parser.add_argument("--check", action="store_true", help="Run correctness comparison vs sequential.")
    parser.add_argument(
        "--has_weight", action="store_true", help="Exercise the optional combine weight side "
        "channel: per-rank (num_tokens, topk) FP32 "
        "weights are routed through the fused "
        "dispatch_group_gemm to produce a 1D "
        "dispatched_weights vector, then "
        "group_gemm_combine push-writes one 4 B "
        "FP32 scalar per dispatched row back to "
        "the source rank's symmetric "
        "(num_tokens, topk) buffer. The transport "
        "is byte-exact (Int32 alias load + store, "
        "no arithmetic) so correctness is verified "
        "as a bitwise round-trip vs the original "
        "topk_weights at valid slots.")
    parser.add_argument("--skip_perf_after_check", action="store_true")
    args = parser.parse_args()
    args.baselines = [s.strip() for s in args.baselines.split(",") if s.strip()]
    valid = {"group_gemm_only", "combine_only", "reduce_only", "sequential", "fused_kernel", "fused_e2e"}
    for b in args.baselines:
        if b not in valid:
            raise ValueError(f"unknown baseline: {b}")
    return args


def main():
    args = parse_args()
    ep_group = init_dist()

    assert args.num_experts % WORLD_SIZE == 0
    experts_per_rank = args.num_experts // WORLD_SIZE
    fc1_n_out = 2 * args.intermediate_size
    expert_alignment = EPOverlapKernels.cluster_tile_m

    if RANK == 0:
        print(f"args = {args}")
        print(f"experts_per_rank={experts_per_rank}  fc1_n_out=2*I={fc1_n_out}  "
              f"expert_alignment={expert_alignment}  "
              f"gemm_num_sm={args.gemm_num_sm}")

    ep_kernels = EPOverlapKernels(
        max_m=args.num_tokens,
        hidden=args.hidden_size,
        topk=args.topk,
        num_experts=args.num_experts,
        local_world_size=LOCAL_WORLD_SIZE or WORLD_SIZE,
        ep_group=ep_group,
        num_worst_tokens=args.num_worst_tokens,
        expert_alignment=expert_alignment,
    )
    cuda_kernels = _CudaEPKernels(
        max_m=args.num_tokens,
        hidden=args.hidden_size,
        topk=args.topk,
        num_experts=args.num_experts,
        local_world_size=LOCAL_WORLD_SIZE or WORLD_SIZE,
        ep_group=ep_group,
        num_sm=args.comm_num_sm if args.comm_num_sm is not None else 16,
        num_worst_tokens=args.num_worst_tokens,
        expert_alignment=expert_alignment,
    )

    torch.manual_seed(42 + RANK)
    # FC2 weight per MoE convention: (E, H, I) -> permute to (N, K, L) = (H, I, E).
    # The permuted view is K-major (strides (K, 1, N*K)); the kernel
    # explicitly rejects ``.contiguous()`` repacks because that flips
    # the leading dim to E and silently breaks the GEMM (see
    # ``_validate_gemm_B`` and ``_b_layout_probe.py``).  We keep both
    # the K-major view (consumed by the kernel) AND the human-readable
    # ``(E, N, K)`` source (consumed by ``torch.matmul`` per expert in
    # the independent torch + CUDA reference path).
    B_fc2_raw = torch.randn(
        experts_per_rank,
        args.hidden_size,
        args.intermediate_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    B_fc2 = B_fc2_raw.permute(1, 2, 0)
    B_fc1_raw = torch.randn(
        experts_per_rank,
        fc1_n_out,
        args.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    B_fc1 = B_fc1_raw.permute(1, 2, 0)

    if args.check:
        run_check(args, ep_kernels, cuda_kernels, B_fc1, B_fc2, B_fc1_raw, B_fc2_raw)
        if args.gemm_num_sm_sweep:
            run_gemm_num_sm_sweep(args, ep_kernels, B_fc1, B_fc2)
        if args.skip_perf_after_check:
            torch.distributed.destroy_process_group(ep_group)
            return

    for round_id in range(args.rounds):
        if round_id > 0:
            cool_down(args.cooldown_s * 4)
        run_perf_round(args, ep_kernels, B_fc1, B_fc2, round_id, profile=args.profile, baselines=args.baselines)

    torch.cuda.synchronize()
    torch.distributed.barrier()
    if RANK == 0:
        print("\nFused GroupGEMM+Combine test completed.")
    torch.distributed.destroy_process_group(ep_group)


if __name__ == "__main__":
    main()
