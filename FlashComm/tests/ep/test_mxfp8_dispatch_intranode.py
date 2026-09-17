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
import datetime
import gzip
import json
import os
from pathlib import Path

import torch

import flash_comm._C.ep_intranode as _ep
import flash_comm._C.quantization as _quantization
import test_ep_intranode as ep_ref
from flash_comm.ep import EPKernels, EPCommLayoutDesc

PROFILE_KERNELS = (
    ("dispatch_index", "kernel_compute_stable_local_token_within_expert_offset"),
    ("dispatch_layout", "kernel_compute_dispatch_layout"),
    ("quant", "kernel_mxfp8_quantize"),
    ("fused_dispatch", "kernel_dispatch_mxfp8_quant_fused_intranode"),
    ("prequantized_dispatch", "kernel_dispatch_mxfp8_prequantized_intranode"),
    ("postprocess_unpack", "kernel_dispatch_mxfp8_postprocess_unpack"),
    ("postprocess_metadata", "kernel_dispatch_mxfp8_postprocess_metadata"),
)

MXFP8_BLOCK_SIZE = 32


def mxfp8_quantize_reference(input: torch.Tensor):
    rows, hidden = input.shape
    blocks = input.float().reshape(rows, hidden // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1)
    scale_value = amax * (1.0 / 448.0)
    bits = scale_value.contiguous().view(torch.int32)
    exponent = torch.bitwise_right_shift(bits, 23)
    mantissa = torch.bitwise_and(bits, 0x7FFFFF)
    round_up = ((mantissa > 0) & (exponent != 0xFE) & ~((exponent == 0) & (mantissa <= 0x400000)))
    exponent = exponent + round_up.to(exponent.dtype)
    exponent = torch.where(scale_value == 0, torch.zeros_like(exponent), exponent)
    exponent = torch.where(torch.isinf(scale_value), torch.full_like(exponent, 0xFE), exponent)
    scales = torch.where(torch.isnan(scale_value), torch.full_like(exponent, 0xFF), exponent).to(torch.uint8)

    scale_i32 = scales.to(torch.int32)
    reciprocal_bits = torch.bitwise_left_shift(254 - scale_i32, 23)
    reciprocal_bits = torch.where(scale_i32 == 254, torch.full_like(reciprocal_bits, 0x00400000), reciprocal_bits)
    reciprocal_bits = torch.where(scale_i32 == 255, torch.full_like(reciprocal_bits, 0x7FFFFFFF), reciprocal_bits)
    reciprocal = reciprocal_bits.contiguous().view(torch.float32)
    data = (blocks * reciprocal.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(rows, hidden)
    return data, scales


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=7168)
    parser.add_argument("-G", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iters", type=int, default=1, help="outer stress rounds used by --check")
    parser.add_argument("--verify-iters", type=int, default=30,
                        help="inputs generated up front and dispatched consecutively in each outer stress round")
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--num-sm", type=int, default=8)
    parser.add_argument("--quant-num-sm", type=int, default=None,
                        help="standalone quant SM count; defaults to --num-sm, 0 uses all SMs")
    parser.add_argument("--drop-ratio", type=float, default=0.0)
    parser.add_argument("--without-weights", action="store_true")
    parser.add_argument("--expert-alignment", type=int, default=1)
    parser.add_argument("--check", action="store_true", help="run batched bitwise state stress")
    parser.add_argument("--profile", action="store_true", help="profile each kernel in both dispatch baselines")
    parser.add_argument("--profile-iters", type=int, default=5)
    parser.add_argument("--prof-dir", default="prof")
    parser.add_argument("--hbm-bandwidth-gbps", type=float, default=3350.0,
                        help="device peak HBM bandwidth used for postprocess utilization reporting")
    parser.add_argument("--min-postprocess-hbm-utilization", type=float, default=0.8,
                        help="minimum postprocess HBM utilization required by --profile")
    parser.add_argument("--pg-timeout", type=int, default=60,
                        help="process-group timeout in seconds; a deadlock must fail fast")
    return parser.parse_args()


def timed_ms(op, iters: int, warmup_iters: int, barrier) -> float:
    for _ in range(warmup_iters):
        op()
    barrier()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        op()
    end.record()
    torch.cuda.synchronize()
    barrier()
    return start.elapsed_time(end) / iters


def aggregate(group, world_size: int, latency_ms: float, num_bytes: int):
    metrics = [None] * world_size
    torch.distributed.all_gather_object(metrics, (latency_ms, num_bytes), group=group)
    slowest_ms = max(item[0] for item in metrics)
    mean_bytes = sum(item[1] for item in metrics) / world_size
    return slowest_ms, mean_bytes, mean_bytes / slowest_ms / 1e6


def bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return (actual.shape == expected.shape and actual.dtype == expected.dtype
            and torch.equal(actual.contiguous().view(torch.uint8),
                            expected.contiguous().view(torch.uint8)))


def mismatch_message(label: str, actual: torch.Tensor, expected: torch.Tensor) -> str:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return (f"{label}: shape/dtype mismatch: actual={tuple(actual.shape)}/{actual.dtype}, "
                f"expected={tuple(expected.shape)}/{expected.dtype}")
    actual_bytes = actual.contiguous().view(torch.uint8)
    expected_bytes = expected.contiguous().view(torch.uint8)
    mismatch = actual_bytes != expected_bytes
    first = mismatch.nonzero()[0].tolist()
    return (f"{label}: bytes_differing={int(mismatch.sum())}/{mismatch.numel()}, first={first}, "
            f"actual={int(actual_bytes[tuple(first)])}, expected={int(expected_bytes[tuple(first)])}")


def check_aligned(actual: torch.Tensor, expected: torch.Tensor, expert_counts: torch.Tensor, expert_alignment: int,
                  label: str):
    if expert_alignment == 1:
        return [] if bitwise_equal(actual, expected) else [mismatch_message(label, actual, expected)]
    if actual.dtype != expected.dtype or actual.shape[1:] != expected.shape[1:]:
        return [
            f"{label}: shape/dtype mismatch: actual={tuple(actual.shape)}/{actual.dtype}, "
            f"expected={tuple(expected.shape)}/{expected.dtype}"
        ]

    errors = []
    aligned_offset = 0
    dense_offset = 0
    for expert, count in enumerate(expert_counts.tolist()):
        actual_slice = actual[aligned_offset:aligned_offset + count]
        expected_slice = expected[dense_offset:dense_offset + count]
        if not bitwise_equal(actual_slice, expected_slice):
            errors.append(mismatch_message(f"{label}/expert{expert}", actual_slice, expected_slice))
        dense_offset += count
        aligned_offset += (count + expert_alignment - 1) // expert_alignment * expert_alignment
    if dense_offset != expected.shape[0] or aligned_offset != actual.shape[0]:
        errors.append(f"{label}: aligned size mismatch: actual={actual.shape[0]}/{aligned_offset}, "
                      f"expected={expected.shape[0]}/{dense_offset}")
    return errors


def load_profile_kernel_averages(trace_path: Path):
    totals = {name: 0.0 for name, _ in PROFILE_KERNELS}
    counts = {name: 0 for name, _ in PROFILE_KERNELS}
    with gzip.open(trace_path, "rt") as file:
        events = json.load(file).get("traceEvents", [])
    for event in events:
        kernel_name = event.get("name", "")
        duration_us = event.get("dur")
        if duration_us is None:
            continue
        for name, needle in PROFILE_KERNELS:
            if needle in kernel_name:
                totals[name] += duration_us / 1000.0
                counts[name] += 1
                break
    averages = {name: totals[name] / counts[name] if counts[name] else 0.0 for name in totals}
    averages["postprocess_total"] = averages["postprocess_unpack"] + averages["postprocess_metadata"]
    return averages


def profile_baseline(op, baseline: str, round_index: int, args, rank: int, world_size: int, group, barrier,
                     kernel_bytes):
    for _ in range(2):
        op()
    barrier()
    torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
    ) as profiler:
        for _ in range(args.profile_iters):
            op()
            profiler.step()
    torch.cuda.synchronize()
    barrier()

    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "none")
    trace_dir = Path(args.prof_dir) / run_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_name = f"mxfp8_{baseline}_M{args.M}_N{args.N}_round{round_index}_rank{rank}.json.gz"
    trace_path = trace_dir / trace_name
    profiler.export_chrome_trace(str(trace_path))
    averages = load_profile_kernel_averages(trace_path)
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, (averages, kernel_bytes), group=group)

    slowest_post_ms = max(item[0]["postprocess_total"] for item in gathered)
    mean_post_bytes = sum(item[1]["postprocess_total"] for item in gathered) / world_size
    postprocess_bandwidth = mean_post_bytes / slowest_post_ms / 1e6
    postprocess_utilization = postprocess_bandwidth / args.hbm_bandwidth_gbps

    if rank == 0:
        print(f"\nprofile baseline={baseline}, "
              f"traces={trace_dir}/mxfp8_{baseline}_M{args.M}_N{args.N}_round{round_index}_rank*.json.gz")
        print(f"{'kernel':28s} {'ms':>10s} {'MB/rank':>10s} {'GB/s':>10s}")
        for name, _ in PROFILE_KERNELS:
            slowest_ms = max(item[0][name] for item in gathered)
            if slowest_ms == 0:
                continue
            mean_bytes = sum(item[1].get(name, 0) for item in gathered) / world_size
            bandwidth = mean_bytes / slowest_ms / 1e6 if mean_bytes else 0.0
            bandwidth_text = f"{bandwidth:10.2f}" if mean_bytes else f"{'-':>10s}"
            print(f"{name:28s} {slowest_ms:10.4f} {mean_bytes / 1e6:10.2f} {bandwidth_text}")
        print(f"{'postprocess_total':28s} {slowest_post_ms:10.4f} {mean_post_bytes / 1e6:10.2f} "
              f"{postprocess_bandwidth:10.2f}")
        print(
            f"postprocess HBM utilization: {postprocess_utilization * 100:.2f}% of "
            f"{args.hbm_bandwidth_gbps:.0f} GB/s (required >= "
            f"{args.min_postprocess_hbm_utilization * 100:.1f}%)", flush=True)

    if postprocess_utilization < args.min_postprocess_hbm_utilization:
        raise AssertionError(f"{baseline} postprocess HBM utilization {postprocess_utilization * 100:.2f}% is below "
                             f"{args.min_postprocess_hbm_utilization * 100:.1f}%")


def main():
    args = parse_args()
    if args.profile and args.check:
        raise ValueError("--profile is a performance option and cannot be combined with --check")
    if min(args.iters, args.verify_iters, args.bench_iters, args.warmup_iters, args.rounds, args.profile_iters) < 1:
        raise ValueError("iteration counts must be positive")
    if args.hbm_bandwidth_gbps <= 0:
        raise ValueError("--hbm-bandwidth-gbps must be positive")
    if not 0 < args.min_postprocess_hbm_utilization <= 1:
        raise ValueError("--min-postprocess-hbm-utilization must be in (0, 1]")
    if args.quant_num_sm is None:
        args.quant_num_sm = args.num_sm

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=args.pg_timeout),
    )
    group = torch.distributed.new_group(ranks=list(range(world_size)), backend="nccl")
    ep_ref.EP_GROUP = group
    ep_ref.init_seed(rank)

    if args.G % world_size:
        raise ValueError("num experts must be divisible by world size")
    if args.N % 32:
        raise ValueError("hidden size must be divisible by 32")

    kernels = EPKernels(
        max_m=args.M,
        hidden=args.N,
        topk=args.topk,
        num_experts=args.G,
        local_world_size=world_size,
        ep_group=group,
        num_sm=args.num_sm,
        expert_alignment=args.expert_alignment,
    )

    def make_data():
        exp_indices = ep_ref.generate_random_exp_indices(args.M, args.G, args.topk, args.drop_ratio).cuda()
        input_data = torch.randn((args.M, args.N), dtype=torch.bfloat16, device="cuda")
        topk_weights = None if args.without_weights else torch.softmax(torch.randn(
            (args.M, args.topk), device="cuda"), dim=-1)
        return input_data, topk_weights, exp_indices

    def run_dispatch(input_data, topk_weights, exp_indices, baseline: str):
        layout = EPCommLayoutDesc()
        if baseline == "quant_fused":
            packed, recv_weights, layout = kernels.dispatch_mxfp8(input_data, exp_indices, topk_weights, layout)
        elif baseline == "prequantized":
            packed_row_bytes = int(_quantization.mxfp8_packed_row_bytes(args.N))
            packed_input = torch.empty((input_data.shape[0], packed_row_bytes // 2), dtype=torch.bfloat16,
                                       device="cuda")
            _quantization.mxfp8_quantize(input_data, packed_input, args.quant_num_sm)
            packed, recv_weights, layout = kernels.dispatch_mxfp8_prequantized(packed_input, exp_indices, topk_weights,
                                                                               layout)
        else:
            raise ValueError(f"unknown baseline {baseline!r}")
        data, scales, weights, layout = kernels.dispatch_mxfp8_postprocess_unpack(packed, recv_weights, layout)
        if layout.recv_expert_counts is not None:
            layout.recv_expert_counts = layout.recv_expert_counts.clone()
        return data, scales, weights, layout

    def check_result(result, reference, label: str):
        data, scales, weights, layout = result
        ref_data, ref_scales, ref_weights, expert_counts = reference
        errors = []
        errors.extend(check_aligned(data, ref_data, expert_counts, args.expert_alignment, f"{label}/data"))
        errors.extend(check_aligned(scales, ref_scales, expert_counts, args.expert_alignment, f"{label}/scales"))
        if (weights is None) != (ref_weights is None):
            errors.append(f"{label}/weights: presence mismatch")
        elif ref_weights is not None:
            errors.extend(check_aligned(weights, ref_weights, expert_counts, args.expert_alignment, f"{label}/weights"))
        if layout.recv_expert_counts is None or not torch.equal(layout.recv_expert_counts.cpu(),
                                                                expert_counts.to(torch.int32)):
            errors.append(f"{label}/expert_counts: mismatch")
        return errors

    def reference(input_data, topk_weights, exp_indices):
        ref_bf16, ref_weights, expert_counts = ep_ref.torch_forward_single(input_data, topk_weights, exp_indices,
                                                                           args.G)
        rows = ref_bf16.shape[0]
        ref_data = torch.empty_like(ref_bf16, dtype=torch.float8_e4m3fn)
        ref_scales = torch.empty((rows, args.N // 32), dtype=torch.uint8, device="cuda")
        for start in range(0, rows, 4096):
            data, scales = mxfp8_quantize_reference(ref_bf16[start:start + 4096].contiguous())
            ref_data[start:start + data.shape[0]].copy_(data)
            ref_scales[start:start + scales.shape[0]].copy_(scales)
        return ref_data, ref_scales, ref_weights, expert_counts

    def barrier():
        kernels.ep_group_barrier()

    try:
        if rank == 0:
            print(f"args = {args}", flush=True)

        if args.check:
            for stress_iter in range(args.iters):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                inputs = [make_data() for _ in range(args.verify_iters)]
                errors = []

                # Match test_ep.py: generate the whole input batch first, then
                # run one implementation continuously before any oracle work.
                for baseline in ("quant_fused", "prequantized"):
                    outputs = []
                    for input_data, topk_weights, exp_indices in inputs:
                        ep_ref.straggler(rank)
                        torch.distributed.barrier(group=group)
                        outputs.append(run_dispatch(input_data, topk_weights, exp_indices, baseline))
                    torch.cuda.synchronize()

                    for input_index, (input_data, topk_weights, exp_indices) in enumerate(inputs):
                        ref = reference(input_data, topk_weights, exp_indices)
                        errors.extend(
                            check_result(outputs[input_index], ref, f"iter{stress_iter}/input{input_index}/{baseline}"))
                        outputs[input_index] = None
                    del outputs, ref
                    torch.cuda.empty_cache()

                passed = torch.tensor([not errors], dtype=torch.int32, device="cuda")
                torch.distributed.all_reduce(passed, op=torch.distributed.ReduceOp.MIN, group=group)
                for error in errors[:20]:
                    print(f"RANK[{rank}] {error}", flush=True)
                if not int(passed.item()):
                    raise AssertionError("MXFP8 dispatch bitwise stress failed")
                if rank == 0:
                    print(f"check iter {stress_iter}: {args.verify_iters} inputs x 2 baselines bitwise OK", flush=True)
                del inputs, input_data, topk_weights, exp_indices

            if rank == 0:
                print("MXFP8 dispatch state stress passed.", flush=True)
            return

        for round_index in range(args.rounds):
            input_data, topk_weights, exp_indices = make_data()
            ref = reference(input_data, topk_weights, exp_indices)

            fused_e2e_ms = timed_ms(lambda: run_dispatch(input_data, topk_weights, exp_indices, "quant_fused"),
                                    args.bench_iters, args.warmup_iters, barrier)
            prequantized_e2e_ms = timed_ms(lambda: run_dispatch(input_data, topk_weights, exp_indices, "prequantized"),
                                           args.bench_iters, args.warmup_iters, barrier)

            errors = check_result(run_dispatch(input_data, topk_weights, exp_indices, "quant_fused"), ref,
                                  f"round{round_index}/quant_fused")
            errors.extend(
                check_result(run_dispatch(input_data, topk_weights, exp_indices, "prequantized"), ref,
                             f"round{round_index}/prequantized"))
            passed = torch.tensor([not errors], dtype=torch.int32, device="cuda")
            torch.distributed.all_reduce(passed, op=torch.distributed.ReduceOp.MIN, group=group)
            for error in errors[:20]:
                print(f"RANK[{rank}] {error}", flush=True)
            if not int(passed.item()):
                raise AssertionError("MXFP8 dispatch performance case failed bitwise verification")

            # Isolate the kernels on the same input/layout for communication and
            # quantization bandwidth attribution.
            layout = EPCommLayoutDesc()
            _, _, layout = kernels.dispatch_mxfp8(input_data, exp_indices, topk_weights, layout)
            packed_row_bytes = int(_quantization.mxfp8_packed_row_bytes(args.N))
            packed_input = torch.empty((args.M, packed_row_bytes // 2), dtype=torch.bfloat16, device="cuda")
            experts_per_rank = args.G // world_size
            context = kernels.ep_context

            def fused_dispatch():
                _ep.dispatch_mxfp8_quant_fused_intranode(input_data, layout.token_topk_send_mask, topk_weights,
                                                         exp_indices, layout.token_dst_scatter_indices,
                                                         context.dispatch_output_buf_ptrs,
                                                         context.dispatch_topk_weights_buf_ptrs,
                                                         context.dispatch_topk_scatter_indices_buf_ptrs, rank,
                                                         world_size, experts_per_rank, args.num_sm)

            def quant_only():
                _quantization.mxfp8_quantize(input_data, packed_input, args.quant_num_sm)

            def prequantized_dispatch():
                _ep.dispatch_mxfp8_prequantized_intranode(packed_input, layout.token_topk_send_mask, topk_weights,
                                                          exp_indices, layout.token_dst_scatter_indices,
                                                          context.dispatch_output_buf_ptrs,
                                                          context.dispatch_topk_weights_buf_ptrs,
                                                          context.dispatch_topk_scatter_indices_buf_ptrs, args.N, rank,
                                                          world_size, experts_per_rank, args.num_sm)

            def prequantized_pipeline():
                quant_only()
                prequantized_dispatch()

            quant_only()
            fused_ms = timed_ms(fused_dispatch, args.bench_iters, args.warmup_iters, barrier)
            pipeline_ms = timed_ms(prequantized_pipeline, args.bench_iters, args.warmup_iters, barrier)
            packed_ms = timed_ms(prequantized_dispatch, args.bench_iters, args.warmup_iters, barrier)
            quant_ms = timed_ms(quant_only, args.bench_iters, args.warmup_iters, barrier)

            valid = exp_indices < args.G
            remote = valid & (exp_indices // experts_per_rank != rank)
            remote_rows = int((layout.token_topk_send_mask.bool() & remote).sum().item())
            remote_bytes = remote_rows * packed_row_bytes
            quant_bytes = args.M * (args.N * input_data.element_size() + packed_row_bytes)
            e2e_bytes = remote_bytes

            rows = []
            for label, latency, num_bytes in (
                ("fused end-to-end", fused_e2e_ms, e2e_bytes),
                ("prequantized end-to-end", prequantized_e2e_ms, e2e_bytes),
                ("fused dispatch kernel", fused_ms, remote_bytes),
                (f"quant({args.quant_num_sm} SM)+prequantized dispatch", pipeline_ms, remote_bytes),
                ("prequantized dispatch kernel", packed_ms, remote_bytes),
                (f"standalone quant ({args.quant_num_sm} SM, HBM)", quant_ms, quant_bytes),
            ):
                rows.append((label, *aggregate(group, world_size, latency, num_bytes)))

            if rank == 0:
                print(
                    f"\nround {round_index}: M={args.M} N={args.N} G={args.G} topk={args.topk} "
                    f"num_sm={args.num_sm} packed_row_bytes={packed_row_bytes} bitwise OK", flush=True)
                print("bandwidth numerator: deduplicated remote wire payload for dispatch; BF16 read + packed write "
                      "for quant")
                print(f"{'stage':45s} {'ms':>9s} {'MB/rank':>10s} {'GB/s':>9s}")
                for label, latency, mean_bytes, bandwidth in rows:
                    print(f"{label:45s} {latency:9.4f} {mean_bytes / 1e6:10.2f} {bandwidth:9.2f}")
                comm_roof_ms = rows[4][1]
                print(f"fused/comm-only roof: {comm_roof_ms / rows[2][1] * 100:.1f}%, "
                      f"exposed quant={max(0.0, rows[2][1] - comm_roof_ms) * 1000:.1f} us")
                print(f"fused speedup over separate quant+dispatch: {rows[3][1] / rows[2][1]:.2f}x", flush=True)

            if args.profile:
                fused_dispatch()
                barrier()
                post_count = (layout.recv_aligned_token_count if args.expert_alignment > 1 else layout.recv_token_count)
                processed_rows = int(post_count[rank].item())
                scatter = context.dispatch_topk_scatter_indices_buf[:processed_rows]
                physical_rows = int(scatter.ne(-1).any(dim=1).sum().item())
                valid_entries = int(scatter.ne(-1).sum().item())
                metadata_entries = processed_rows * args.topk
                unpack_bytes = (physical_rows * packed_row_bytes + valid_entries * (args.N + args.N // 32) +
                                metadata_entries * 2 * 4)
                metadata_bytes = metadata_entries * 3 * 4
                if topk_weights is not None:
                    metadata_bytes += valid_entries * 2 * topk_weights.element_size()
                profile_bytes = {
                    "quant": quant_bytes,
                    "fused_dispatch": remote_bytes,
                    "prequantized_dispatch": remote_bytes,
                    "postprocess_unpack": unpack_bytes,
                    "postprocess_metadata": metadata_bytes,
                    "postprocess_total": unpack_bytes + metadata_bytes,
                }
                profile_baseline(lambda: run_dispatch(input_data, topk_weights, exp_indices, "quant_fused"),
                                 "quant_fused", round_index, args, rank, world_size, group, barrier, profile_bytes)
                profile_baseline(lambda: run_dispatch(input_data, topk_weights, exp_indices, "prequantized"),
                                 "prequantized", round_index, args, rank, world_size, group, barrier, profile_bytes)
    finally:
        kernels.finalize()
        torch.distributed.barrier(group=group)
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
