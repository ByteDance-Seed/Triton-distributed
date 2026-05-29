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
import os
import random
from typing import Dict, List, Optional, Tuple

import torch

from flash_comm.ep import EPKernels as _CudaEPKernels
from flash_comm.ep.ep_kernels import EPCommLayoutDesc
from flash_comm.ep_overlap import EPOverlapKernels
from flash_comm.utils.test_utils import (
    LOCAL_WORLD_SIZE,
    RANK,
    WORLD_SIZE,
    assert_close_strict,
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
    """Per-expert padded prefix sum, returned as ``(num_experts+1,)`` int64."""
    counts_i64 = expert_counts.to(torch.int64)
    padded = ((counts_i64 + pad_to - 1) // pad_to) * pad_to
    out = torch.empty(expert_counts.shape[0] + 1, dtype=torch.int64, device=expert_counts.device)
    out[0].zero_()
    torch.cumsum(padded, dim=0, out=out[1:])
    return out


def pack_valid_rows(tensor: torch.Tensor, expert_counts, padded_offsets):
    """Strip per-expert tile padding from a (padded_rows, *) tensor."""
    parts = []
    for e, count in enumerate(expert_counts):
        count = int(count)
        if count <= 0:
            continue
        start = int(padded_offsets[e])
        parts.append(tensor[start:start + count])
    if not parts:
        return torch.empty((0, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
    return torch.cat(parts, dim=0)


def run_cuda_reference(cuda_kernels: _CudaEPKernels, cutedsl_kernels: EPOverlapKernels, input_data, exp_indices, B, *,
                       gemm_num_sm=None, topk_weights=None) -> Dict:
    """CUDA push dispatch + CuTeDSL group-GEMM (shared GEMM with cutedsl_ref).

    When ``topk_weights`` is provided the CUDA dispatch path is asked to
    gather the weights too; the densified result is returned alongside
    ``A`` / ``C`` so a downstream check can compare it against the fused
    kernel's side-loaded output bitwise.
    """
    layout_desc = EPCommLayoutDesc()
    disp_out, weights, layout_desc = cuda_kernels.dispatch(
        input_data,
        exp_indices,
        topk_weights,
        layout_desc,
    )
    A_padded, dispatch_weights, layout_desc = cuda_kernels.dispatch_postprocess(
        disp_out.clone(),
        weights.clone() if weights is not None else None,
        layout_desc,
    )
    expert_counts = layout_desc.recv_expert_counts.clone()
    C = cutedsl_kernels.group_gemm(A_padded, B, expert_counts, gemm_num_sm=gemm_num_sm)
    out = {
        "A": A_padded.clone(),
        "C": C.clone(),
        "expert_counts": expert_counts,
        "padded_offsets": counts_to_padded_offsets(expert_counts, cutedsl_kernels.cluster_tile_m).clone(),
    }
    if dispatch_weights is not None:
        out["W"] = dispatch_weights.clone()
    return out


def run_torch_reference(cuda_kernels: _CudaEPKernels, cutedsl_kernels: EPOverlapKernels, input_data, exp_indices,
                        B_raw: torch.Tensor, *, topk_weights: Optional[torch.Tensor] = None) -> Dict:
    """CUDA dispatch + per-expert bf16 ``torch.matmul`` GEMM (no split-K).

    When ``topk_weights`` is provided the gathered (dense) weights are
    threaded through ``cuda_kernels.dispatch`` and surfaced via ``W`` so
    the downstream :func:`compare_against_cuda` also exercises the weight
    side channel against the fused capture.
    """
    layout_desc = EPCommLayoutDesc()
    disp_out, weights, layout_desc = cuda_kernels.dispatch(
        input_data,
        exp_indices,
        topk_weights,
        layout_desc,
    )
    A_padded, dispatch_weights, layout_desc = cuda_kernels.dispatch_postprocess(
        disp_out.clone(),
        weights.clone() if weights is not None else None,
        layout_desc,
    )
    expert_counts = layout_desc.recv_expert_counts.clone()
    padded_offsets = counts_to_padded_offsets(
        expert_counts,
        cutedsl_kernels.cluster_tile_m,
    )
    C = grouped_matmul_bf16(A_padded, B_raw, expert_counts, padded_offsets)
    out = {
        "A": A_padded.clone(),
        "C": C.clone(),
        "expert_counts": expert_counts,
        "padded_offsets": padded_offsets.clone(),
    }
    if dispatch_weights is not None:
        out["W"] = dispatch_weights.clone()
    return out


def run_cutedsl_ref(cutedsl_kernels: EPOverlapKernels, input_data, exp_indices, B, *, gemm_num_sm=None,
                    comm_num_sm=None) -> Dict:
    """CuTeDSL dispatch + standalone CuTeDSL group-GEMM (no overlap)."""
    layout_desc = EPCommLayoutDesc()
    A_padded, layout_desc = cutedsl_kernels.dispatch_cutedsl(
        input_data,
        exp_indices,
        layout_desc,
        comm_num_sm=comm_num_sm,
    )
    expert_counts = layout_desc.recv_expert_counts.clone()
    C = cutedsl_kernels.group_gemm(A_padded, B, expert_counts, gemm_num_sm=gemm_num_sm)
    return {
        "A": A_padded.clone(),
        "C": C.clone(),
        "layout": layout_desc,
        "expert_counts": expert_counts,
        "padded_offsets": counts_to_padded_offsets(expert_counts, cutedsl_kernels.cluster_tile_m).clone(),
    }


def run_fused(cutedsl_kernels: EPOverlapKernels, *, input_data, exp_indices, B, dispatch_num_stages: int,
              gemm_num_sm=None, comm_num_sm=None, topk_weights=None) -> Dict:
    """Drive the production fused dispatch + group-GEMM kernel.

    Passing ``topk_weights`` enables the side-loaded weight path; the
    returned dict gains a ``W`` field with the gathered per-row weights
    (1D float32, sized to ``A_padded.shape[0]``).
    """
    layout_desc = EPCommLayoutDesc()
    C, dispatch_weights, layout_desc = cutedsl_kernels.dispatch_group_gemm(
        input_data,
        exp_indices,
        B,
        layout_desc,
        dispatch_num_stages=dispatch_num_stages,
        gemm_num_sm=gemm_num_sm,
        comm_num_sm=comm_num_sm,
        topk_weights=topk_weights,
    )
    expert_counts = layout_desc.recv_expert_counts.clone()
    out = {
        "C": C.clone(),
        "layout": layout_desc,
        "expert_counts": expert_counts,
        "padded_offsets": counts_to_padded_offsets(expert_counts, cutedsl_kernels.cluster_tile_m).clone(),
    }
    if dispatch_weights is not None:
        out["W"] = dispatch_weights.clone()
    return out


def _pack_valid_weights(tensor: torch.Tensor, expert_counts, padded_offsets) -> torch.Tensor:
    """Densify a (padded_rows,) 1D weight tensor by stripping per-expert pad."""
    parts = []
    for e, count in enumerate(expert_counts):
        count = int(count)
        if count <= 0:
            continue
        start = int(padded_offsets[e])
        parts.append(tensor[start:start + count])
    if not parts:
        return torch.empty((0, ), dtype=tensor.dtype, device=tensor.device)
    return torch.cat(parts, dim=0)


def compare_against_cuda(label: str, capture: Dict, cuda: Dict) -> None:
    """Assert bit-exact equality for ``C`` (and ``A`` / ``W`` when present)
    after densifying.
    """
    ec = cuda["expert_counts"].cpu().tolist()
    po = cuda["padded_offsets"].cpu().tolist()
    ref_C = pack_valid_rows(cuda["C"], ec, po)

    cap_ec = capture["expert_counts"].cpu().tolist()
    cap_po = capture["padded_offsets"].cpu().tolist()
    assert cap_ec == ec, f"{label}: expert_counts mismatch"
    assert cap_po == po, f"{label}: padded_offsets mismatch"

    if "A" in capture and capture["A"] is not None:
        ref_A = pack_valid_rows(cuda["A"], ec, po)
        cap_A = pack_valid_rows(capture["A"], cap_ec, cap_po)
        assert bitwise_equal(ref_A, cap_A), f"{label}: A bytes differ"
    cap_C = pack_valid_rows(capture["C"], cap_ec, cap_po)
    assert_close_strict(ref_C, cap_C, label=f"{label} C")

    # Side-loaded weights: dense 1D float32, byte-equal because the
    # kernel does an aliased Int32 ld + st with no arithmetic.
    if "W" in capture and "W" in cuda:
        ref_W = _pack_valid_weights(cuda["W"], ec, po)
        cap_W = _pack_valid_weights(capture["W"], cap_ec, cap_po)
        assert bitwise_equal(ref_W, cap_W), (f"{label}: dispatch weights differ "
                                             f"(max_abs={(ref_W - cap_W).abs().max().item():.3e})")


def make_case(args, token_num: int):
    """Return ``(input, exp_indices, topk_weights_or_None)``.

    The optional FP32 ``topk_weights`` is only allocated when
    ``args.has_weight`` is True so the default (weightless) workload
    keeps an identical memory footprint.
    """
    exp_indices = generate_random_exp_indices(
        token_num,
        args.num_experts,
        args.topk,
        args.drop_ratio,
    ).cuda()
    input_data = torch.randn(token_num, args.hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_weights = None
    if args.has_weight:
        topk_weights = torch.randn(
            token_num,
            args.topk,
            dtype=torch.float32,
            device="cuda",
        )
    return input_data, exp_indices, topk_weights


def case_sizes(args, num_cases: int):
    """Reproducible mix: at least one full-num_tokens case + random fillers."""
    sizes = [args.num_tokens]
    for _ in range(num_cases - 1):
        sizes.append(random.randint(max(1, args.num_tokens // 2), args.num_tokens))
    return sizes


def run_gemm_num_sm_sweep(args, cutedsl_kernels, B) -> None:
    """Bitwise check of valid GEMM output across ``gemm_num_sm`` budgets.

    Different ``gemm_num_sm`` produce different ``max_active_clusters``,
    which is a purely scheduling-side constexpr in the M-contig group
    GEMM (cluster-persistent tile distribution). No split-K is used, so
    every valid output tile's accumulator order is identical regardless
    of which cluster computes it -- the *valid* portion of C must be
    byte-equal across SM budgets.

    The kernel writes only the per-expert cluster-tile-padded prefix
    of ``output`` (rows ``[0, sum(padded_expert_counts))``); the tail
    of the ``torch.empty`` allocation up to ``num_worst_tokens`` is
    untouched and its uninitialised bytes depend on the caching
    allocator state, which in turn depends on the kernel's RMEM/SMEM
    bookkeeping allocations (i.e. on ``gemm_num_sm``). We therefore
    densify with :func:`pack_valid_rows` and compare only the valid
    rows -- exactly what :func:`compare_against_cuda` does for the
    main check.
    """
    sweep_values = list({args.gemm_num_sm, 64, 136, args.gemm_num_sm_sweep_n})
    sweep_values = [v for v in sweep_values if v != args.gemm_num_sm]
    if not sweep_values:
        return
    if RANK == 0:
        print(f"\n--- gemm_num_sm sweep "
              f"(reference={args.gemm_num_sm}, "
              f"others={sweep_values}) ---")

    input_data, exp_indices, topk_weights = make_case(args, args.num_tokens)

    def _sample(v):
        out, _w, layout_desc = cutedsl_kernels.dispatch_group_gemm(
            input_data,
            exp_indices,
            B,
            None,
            dispatch_num_stages=args.dispatch_num_stages,
            gemm_num_sm=v,
            comm_num_sm=args.comm_num_sm,
            topk_weights=topk_weights,
        )
        ec = layout_desc.recv_expert_counts.cpu().tolist()
        po = counts_to_padded_offsets(
            layout_desc.recv_expert_counts,
            cutedsl_kernels.cluster_tile_m,
        ).cpu().tolist()
        return pack_valid_rows(out, ec, po)

    ref = _sample(args.gemm_num_sm)
    for v in sweep_values:
        out = _sample(v)
        if not bitwise_equal(ref, out):
            diff = (ref.float() - out.float()).abs()
            raise AssertionError(f"gemm_num_sm sweep: gemm_num_sm={v} not byte-equal to "
                                 f"reference (gemm_num_sm={args.gemm_num_sm}) on "
                                 f"valid rows; max_abs={diff.max().item():.6e}")
        if RANK == 0:
            print(f"  gemm_num_sm={v}: bitwise=PASS")
    if RANK == 0:
        print("gemm_num_sm sweep PASSED.\n")


def run_check(args, cutedsl_kernels, cuda_kernels, B, B_raw, baselines):

    total = 0

    for round_idx in range(args.rounds):
        if RANK == 0:
            print(f"\n--- check round {round_idx} ---")
        sizes = case_sizes(args, args.verify_iters)
        case_data = [(idx, n, *make_case(args, n)) for idx, n in enumerate(sizes)]

        refs = []
        torch_refs = []
        for case_idx, token_num, input_data, exp_indices, topk_weights in case_data:
            torch.distributed.barrier()
            refs.append(
                run_cuda_reference(
                    cuda_kernels,
                    cutedsl_kernels,
                    input_data,
                    exp_indices,
                    B,
                    gemm_num_sm=args.gemm_num_sm,
                    topk_weights=topk_weights,
                ))
            torch_refs.append(
                run_torch_reference(
                    cuda_kernels,
                    cutedsl_kernels,
                    input_data,
                    exp_indices,
                    B_raw,
                    topk_weights=topk_weights,
                ))
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # Independent witness: kernel group_gemm vs torch.matmul on the
        # same CUDA-dispatched A (both bf16, no split-K) -> byte-equal.
        # When --has_weight is on the torch reference also gathers the
        # side-loaded weights through ``cuda_kernels.dispatch``; assert
        # they round-trip byte-equal to the kernel-side reference too.
        for case_idx, _, _, _, _ in case_data:
            ec = refs[case_idx]["expert_counts"].cpu().tolist()
            po = refs[case_idx]["padded_offsets"].cpu().tolist()
            cuda_C = pack_valid_rows(refs[case_idx]["C"], ec, po)
            torch_C = pack_valid_rows(torch_refs[case_idx]["C"], ec, po)
            assert bitwise_equal(cuda_C, torch_C), (f"round{round_idx} case{case_idx}: kernel group_gemm "
                                                    f"vs torch.matmul bf16 byte mismatch")
            if "W" in refs[case_idx] and "W" in torch_refs[case_idx]:
                cuda_W = _pack_valid_weights(refs[case_idx]["W"], ec, po)
                torch_W = _pack_valid_weights(torch_refs[case_idx]["W"], ec, po)
                assert bitwise_equal(cuda_W, torch_W), (f"round{round_idx} case{case_idx}: torch_ref W "
                                                        f"byte mismatch vs cuda_ref")

        # ``cuda`` is the reference (``refs``); other baselines are compared
        # against it.
        capture_targets = [b for b in baselines if b != "cuda"]
        captures = {b: [] for b in capture_targets}
        if "cutedsl_ref" in captures:
            for _, _, input_data, exp_indices, _ in case_data:
                # cutedsl standalone dispatch has no weight side channel;
                # we just don't compare W for this baseline.
                captures["cutedsl_ref"].append(
                    run_cutedsl_ref(
                        cutedsl_kernels,
                        input_data,
                        exp_indices,
                        B,
                        gemm_num_sm=args.gemm_num_sm,
                        comm_num_sm=args.comm_num_sm,
                    ))
            torch.cuda.synchronize()
            torch.distributed.barrier()
        if "fused_kernel" in captures:
            for _, _, input_data, exp_indices, topk_weights in case_data:
                captures["fused_kernel"].append(
                    run_fused(
                        cutedsl_kernels,
                        input_data=input_data,
                        exp_indices=exp_indices,
                        B=B,
                        dispatch_num_stages=args.dispatch_num_stages,
                        gemm_num_sm=args.gemm_num_sm,
                        comm_num_sm=args.comm_num_sm,
                        topk_weights=topk_weights,
                    ))
            torch.cuda.synchronize()
            torch.distributed.barrier()
        if "fused_e2e" in captures:
            for _, _, input_data, exp_indices, topk_weights in case_data:
                captures["fused_e2e"].append(
                    run_fused(
                        cutedsl_kernels,
                        input_data=input_data,
                        exp_indices=exp_indices,
                        B=B,
                        dispatch_num_stages=args.dispatch_num_stages,
                        gemm_num_sm=args.gemm_num_sm,
                        comm_num_sm=args.comm_num_sm,
                        topk_weights=topk_weights,
                    ))
            torch.cuda.synchronize()
            torch.distributed.barrier()

        for case_idx, token_num, _, _, _ in case_data:
            ref = refs[case_idx]
            statuses = [
                f"tokens={token_num}", f"A_shape={tuple(ref['A'].shape)}", f"C_shape={tuple(ref['C'].shape)}",
                "torch_ref=bitwise=PASS"
            ]
            for name, caps in captures.items():
                compare_against_cuda(
                    f"round{round_idx} case{case_idx} {name}",
                    caps[case_idx],
                    ref,
                )
                statuses.append(f"{name}=bitwise=PASS")
            if RANK == 0:
                print(f"round {round_idx} case {case_idx}: " + "  ".join(statuses))
            total += 1

    if RANK == 0:
        print(f"\nCorrectness check passed: {total} cases.")


def build_perf_runners(args, cutedsl_kernels: EPOverlapKernels, cuda_kernels: _CudaEPKernels, B: torch.Tensor,
                       input_data, exp_indices, topk_weights=None, *,
                       record_fused_call: bool = False) -> Tuple[Dict[str, callable], Dict]:
    """Return ``(runners, workload)`` for the perf loop.

    ``workload`` carries FC1 GEMM M/N/K and dispatch volume so the
    headline printer can derive TFLOPs / BW. When ``topk_weights`` is
    supplied the runners exercise the ``has_weight`` dispatch path so
    we can spot any regression vs the weightless build.
    """
    fc1_n_out = 2 * args.intermediate_size
    cutedsl_state = run_cutedsl_ref(
        cutedsl_kernels,
        input_data,
        exp_indices,
        B,
        gemm_num_sm=args.gemm_num_sm,
        comm_num_sm=args.comm_num_sm,
    )
    ctx = cutedsl_kernels.overlap_context
    layout_desc = cutedsl_state["layout"]
    expert_counts = cutedsl_state["expert_counts"]
    A_padded_buf = cutedsl_state["A"].clone()
    recv_count = (layout_desc.recv_aligned_token_count
                  if layout_desc.expert_alignment > 1 else layout_desc.recv_token_count)

    has_weight = topk_weights is not None

    # Pre-stage input (and weight, if applicable) so fused pull dispatch
    # reads stable data per iter. ``ensure_dispatch_input_weight`` is
    # cheap (single lazy SymmetricTensor alloc) and only ever runs once.
    ctx.ensure_dispatch_input()
    ctx.dispatch_input_buf[:input_data.shape[0]].copy_(input_data)
    if has_weight:
        ctx.ensure_dispatch_input_weight()
        ctx.dispatch_input_weight_buf[:topk_weights.shape[0]].copy_(topk_weights)
    cutedsl_kernels.ep_group_barrier()
    torch.cuda.synchronize()

    C_scratch = torch.empty(
        (A_padded_buf.shape[0], fc1_n_out),
        dtype=torch.bfloat16,
        device="cuda",
    )
    # Pre-allocate the weight output scratch so the kernel-only runner
    # stays allocation-free (matches the no-weight C_scratch pattern).
    W_scratch = (torch.empty(
        (A_padded_buf.shape[0], ),
        dtype=torch.float32,
        device="cuda",
    ) if has_weight else None)
    input_weight_ptrs = (ctx.dispatch_input_weight_ptrs if has_weight else None)

    valid_M = int(expert_counts.sum().item())
    padded_M = int(A_padded_buf.shape[0])
    recv_unpadded = int(layout_desc.recv_token_count[RANK].item())
    workload = {
        "valid_M": valid_M,
        "padded_M": padded_M,
        "fc1_N": fc1_n_out,
        "fc1_K": args.hidden_size,
        # FLOPs on valid (un-padded) M only.
        "gemm_flops": 2 * valid_M * fc1_n_out * args.hidden_size,
        "dispatch_bytes": recv_unpadded * args.hidden_size * 2,
        "recv_unpadded": recv_unpadded,
        "has_weight": has_weight,
    }

    runners: Dict[str, callable] = {}

    # ``gemm_num_sm=None`` -> full device; resolve once here so the
    # kernel-only runner (which bypasses the public wrapper) sees the
    # same value the public API would use.
    gemm_num_sm = cutedsl_kernels._resolve_gemm_num_sm(args.gemm_num_sm)

    def cuda_runner():
        return run_cuda_reference(
            cuda_kernels,
            cutedsl_kernels,
            input_data,
            exp_indices,
            B,
            gemm_num_sm=args.gemm_num_sm,
            topk_weights=topk_weights,
        )["C"]

    runners["cuda"] = cuda_runner

    def cutedsl_ref_runner():
        return cutedsl_kernels.group_gemm(
            cutedsl_state["A"],
            B,
            expert_counts,
            output=C_scratch,
            gemm_num_sm=args.gemm_num_sm,
        )

    runners["cutedsl_ref"] = cutedsl_ref_runner

    topk_const = int(exp_indices.shape[1])
    if "fused_kernel" in args.baselines:
        # Kernel-only: bypass the public wrapper so layout compute /
        # input copy / barrier are not in the timed window.
        def fused_kernel_runner():
            ctx.reset_expert_signals()
            with record_function_range("dispatch_group_gemm.run", record_fused_call):
                cutedsl_kernels._dispatch_group_gemm_op.run(
                    A_padded=A_padded_buf,
                    B=B,
                    token_src_rank_topk_and_indices=layout_desc.token_src_rank_topk_and_indices,
                    recv_count=recv_count,
                    recv_expert_counts=expert_counts,
                    output=C_scratch,
                    dispatch_input_ptrs=ctx.dispatch_input_ptrs,
                    expert_signals=ctx.expert_signals,
                    expert_signal_counters=ctx.expert_signal_counters,
                    dispatch_num_stages=args.dispatch_num_stages,
                    num_sm=gemm_num_sm,
                    input_weight_ptrs=input_weight_ptrs,
                    output_weight=W_scratch,
                    topk=topk_const,
                    weight_dtype=torch.float32,
                )
            return C_scratch

        runners["fused_kernel"] = fused_kernel_runner

    if "fused_e2e" in args.baselines:
        # Public API: includes layout / input copy / barrier overhead.
        def fused_e2e_runner():
            with record_function_range("dispatch_group_gemm.run", record_fused_call):
                out, _w, _ld = cutedsl_kernels.dispatch_group_gemm(
                    input_data,
                    exp_indices,
                    B,
                    layout_desc=None,
                    dispatch_num_stages=args.dispatch_num_stages,
                    gemm_num_sm=args.gemm_num_sm,
                    comm_num_sm=args.comm_num_sm,
                    topk_weights=topk_weights,
                )
            return out

        runners["fused_e2e"] = fused_e2e_runner

    return runners, workload


_GEMM_BASELINES = {"cuda", "cutedsl_ref", "fused_kernel", "fused_e2e"}
_DISPATCH_BASELINES = {"cuda", "fused_kernel", "fused_e2e"}


def print_workload_header(rank: int, round_id: int, workload: Dict) -> None:
    """One-shot summary of the FC1 GEMM workload + dispatch volume."""
    gemm_flops = workload["gemm_flops"]
    dispatch_bytes = workload["dispatch_bytes"]
    weight_flag = "  has_weight=1" if workload.get("has_weight") else ""
    print(f"[RANK {rank}] workload round {round_id}: "
          f"group_gemm valid_M={workload['valid_M']} (padded_M={workload['padded_M']}) "
          f"N={workload['fc1_N']} K={workload['fc1_K']}  "
          f"gemm_flops={gemm_flops / 1e12:.3f}TF  "
          f"dispatch recv_rows={workload['recv_unpadded']}  "
          f"dispatch_bytes={dispatch_bytes / 1e9:.3f}GB"
          f"{weight_flag}")


def annotate_baseline(name: str, ms: float, workload: Dict) -> str:
    """Append TFLOPs / BW columns to a baseline's headline ms."""
    parts = [f"{name}={ms:.4f}ms"]
    if name in _GEMM_BASELINES:
        tflops = workload["gemm_flops"] / (ms * 1e-3) / 1e12 if ms > 0 else 0.0
        parts.append(f"gemm_tflops={tflops:.2f}")
    if name in _DISPATCH_BASELINES:
        bw_gbps = workload["dispatch_bytes"] / ms / 1e6 if ms > 0 else 0.0
        parts.append(f"dispatch_bw={bw_gbps:.2f}GB/s")
    return " ".join(parts)


def run_perf_round(
    args,
    runners: Dict[str, callable],
    workload: Dict,
    round_id: int,
    *,
    profile: bool,
    profile_runners: Optional[Dict[str, callable]] = None,
    ep_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> None:
    timings = {}
    nvtx_prefix = (f"rank{RANK}.round{round_id}" if profile else None)

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
        timings.setdefault("_stats", {})[name] = stats
        # Cooldown between baselines to dodge clock throttling.
        if args.cooldown_s > 0:
            cool_down(args.cooldown_s)

    torch.cuda.synchronize()
    torch.distributed.barrier()

    if profile:
        if profile_runners is None:
            profile_runners = runners
        profile_runners = {k: v for k, v in profile_runners.items() if k in runners}
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
            label="ep_overlap_dispatch_group_gemm",
            rank=RANK,
            round_id=round_id,
            profile_dir=args.profile_dir or None,
        )
        if RANK == 0:
            print(f"  trace written to {trace}")

    # Gather per-rank timings + workload so we can spot cross-rank skew
    # (a slow rank in the fused pull dispatch bottlenecks every peer).
    payload = {"rank": RANK, "timings": timings, "workload": workload}
    gathered: List[Dict] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, payload)
    if RANK == 0:
        for entry in sorted(gathered, key=lambda e: e["rank"]):
            print_workload_header(entry["rank"], round_id, entry["workload"])
            stats_table = entry["timings"].pop("_stats", {})
            for name, ms in entry["timings"].items():
                print(f"[RANK {entry['rank']}] round {round_id}: " + annotate_baseline(name, ms, entry["workload"]))
                st = stats_table.get(name)
                if st is not None:
                    # Min/median align with Nsight kernel-trace samples.
                    print(f"[RANK {entry['rank']}] round {round_id}:   "
                          f"{name} stats min={st['min']:.4f} "
                          f"p10={st['p10']:.4f} median={st['median']:.4f} "
                          f"max={st['max']:.4f} (ms)")
    cool_down(args.cooldown_s)


_DEFAULT_BASELINES = "cuda,cutedsl_ref,fused_kernel,fused_e2e"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, default=4096)
    parser.add_argument("--hidden_size", type=int, default=8192)
    parser.add_argument("--intermediate_size", type=int, default=2560, help="MoE intermediate dim I; FC1 N = 2*I.")
    parser.add_argument("--num_experts", type=int, default=160)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--drop_ratio", type=float, default=0.1)
    parser.add_argument(
        "--gemm_num_sm", type=int, default=None, help="Per-call GEMM scheduler SM budget. None (default)"
        " = full device SM count (queried from torch).")
    parser.add_argument("--comm_num_sm", type=int, default=None,
                        help="Per-call dispatch/layout SM budget. None = full device.")
    parser.add_argument("--num_worst_tokens", type=int, default=131072)
    parser.add_argument("--dispatch_num_stages", type=int, default=1)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup_iters", type=int, default=50, help="Untimed warmup iters per baseline before "
                        "perf measurement (avoids cold-start bias).")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--verify_iters", type=int, default=4)
    parser.add_argument("--cooldown_s", type=float, default=0.01)
    parser.add_argument(
        "--device_sleep_cycles", type=int, default=2_000_000, help="torch.cuda._sleep cycles between perf iters "
        "to prevent SM throttling on sustained heavy "
        "workloads (default 2e6 ~= 1.4 ms at 1.5 GHz; "
        "set 0 to disable).")
    parser.add_argument(
        "--baselines", type=str, default=_DEFAULT_BASELINES, help="Comma-separated subset: cuda, cutedsl_ref, "
        "fused_kernel (kernel-only), fused_e2e (full API).")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_dir", type=str, default="")
    parser.add_argument("--check", action="store_true", help="Run multi-round correctness comparison.")
    parser.add_argument(
        "--gemm_num_sm_sweep", action="store_true", help="Append a bitwise sweep over multiple GEMM SM "
        "budgets to the check phase (each distinct value "
        "triggers a fresh CuTeDSL compile, so this is "
        "off-by-default).")
    parser.add_argument("--gemm_num_sm_sweep_n", type=int, default=96,
                        help="Extra SM budget probed by the sweep (in addition "
                        "to 64 and 136).")
    parser.add_argument("--skip_perf_after_check", action="store_true")
    parser.add_argument(
        "--has_weight",
        action="store_true",
        help="Exercise the side-loaded topk_weights dispatch path. "
        "Check phase asserts the gathered weights are bitwise-equal "
        "against the CUDA reference; perf phase confirms parity vs "
        "the weightless build.",
    )
    args = parser.parse_args()
    args.baselines = [s.strip() for s in args.baselines.split(",") if s.strip()]
    valid = {"cuda", "cutedsl_ref", "fused_kernel", "fused_e2e"}
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

    cutedsl_kernels = EPOverlapKernels(
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

    seed_off = int(os.environ.get("PROBE_SEED_OFFSET", "42"))
    torch.manual_seed(seed_off + RANK)
    # FC1 weight (E, 2*I, H); kernel consumes K-major view via permute.
    B_ref = torch.randn(experts_per_rank, fc1_n_out, args.hidden_size, dtype=torch.bfloat16, device="cuda")
    B = B_ref.permute(1, 2, 0)

    if args.check:
        run_check(args, cutedsl_kernels, cuda_kernels, B, B_ref, args.baselines)
        if args.gemm_num_sm_sweep:
            run_gemm_num_sm_sweep(args, cutedsl_kernels, B)
        if args.skip_perf_after_check:
            torch.distributed.destroy_process_group(ep_group)
            return

    for round_idx in range(args.rounds):
        token_num = (args.num_tokens if round_idx == 0 else random.randint(max(1, args.num_tokens //
                                                                               2), args.num_tokens))
        if RANK == 0:
            print(f"\n--- perf round {round_idx}: tokens={token_num} ---")
        input_data, exp_indices, topk_weights = make_case(args, token_num)
        runners, workload = build_perf_runners(
            args,
            cutedsl_kernels,
            cuda_kernels,
            B,
            input_data,
            exp_indices,
            topk_weights=topk_weights,
        )
        profile_runners = None
        if args.profile:
            profile_runners, _ = build_perf_runners(
                args,
                cutedsl_kernels,
                cuda_kernels,
                B,
                input_data,
                exp_indices,
                topk_weights=topk_weights,
                record_fused_call=True,
            )
        run_perf_round(
            args,
            runners,
            workload,
            round_idx,
            profile=args.profile,
            profile_runners=profile_runners,
            ep_group=ep_group,
        )

    torch.cuda.synchronize()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group(ep_group)


if __name__ == "__main__":
    main()
