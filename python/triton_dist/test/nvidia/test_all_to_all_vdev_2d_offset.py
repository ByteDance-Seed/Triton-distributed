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
import torch
import os
import torch.distributed as dist
from triton_dist.kernels.nvidia.all_to_all_vdev_2d_offset import all_to_all_v_offset_op, all_to_all_v_offset_op_v2, create_context
from triton_dist.profiler_utils import perf_func, group_profile
from triton_dist.utils import initialize_distributed, dist_print, sleep_async
import random
from functools import partial


def straggler(rank):
    clock_rate = torch.cuda.clock_rate() * 1e6
    r = max(int(clock_rate * 0.00001), 1000)
    cycles = random.randint(0, r) * (rank + 1)
    torch.cuda._sleep(cycles)


def get_golden_results(inp_splits: torch.Tensor, inp: torch.Tensor, world_size: int, ne: int, align: int,
                       rank_in_row: bool, padded_offset: torch.Tensor = None, group: dist.ProcessGroup = None):
    device = inp_splits.device
    dtype = inp.dtype
    dist.barrier()

    if inp.dim() > 1:
        token_dim = inp.shape[1]
    else:
        token_dim = 1

    if padded_offset is None:
        inp_splits_flat = inp_splits.view(-1)
        inp_offsets_inclusive = torch.cumsum(inp_splits_flat, dim=0)
        inp_offsets = torch.cat((torch.tensor([0], device=device,
                                              dtype=inp_splits.dtype), inp_offsets_inclusive[:-1])).to(torch.int32)
    else:
        assert not rank_in_row, "padded_offset is not supported in rank-major mode"
        inp_offsets = padded_offset.view(-1).to(torch.int32)

    # Step 1: exchange splits and offset
    if rank_in_row:
        assert inp_splits.shape[0] == world_size and inp_splits.shape[1] == ne
        inp_offsets = inp_offsets.reshape(world_size, ne)
        recv_splits_rm = torch.empty_like(inp_splits)
        recv_offsets_rm = torch.empty_like(inp_splits)
        dist.all_to_all_single(recv_splits_rm, inp_splits, group=group)
        dist.all_to_all_single(recv_offsets_rm, inp_offsets, group=group)
        expected_out_splits = recv_splits_rm.t().contiguous()
        expected_source_offsets = recv_offsets_rm.t().contiguous()

        raw_recv_splits = recv_splits_rm
    else:
        assert inp_splits.shape[0] == ne and inp_splits.shape[1] == world_size
        inp_offsets = inp_offsets.reshape(ne, world_size)
        split_trans = inp_splits.t().contiguous()
        offset_trans = inp_offsets.t().contiguous()
        expected_out_splits = torch.empty_like(split_trans)
        expected_source_offsets = torch.empty_like(offset_trans)
        dist.all_to_all_single(expected_out_splits, split_trans, group=group)
        dist.all_to_all_single(expected_source_offsets, offset_trans, group=group)

        raw_recv_splits = expected_out_splits

    # Step 2: align recv out splits, if needed
    if align > 1:
        out_split_list = expected_out_splits.tolist()
        split_sum_each_row = expected_out_splits.sum(1, keepdim=False).tolist()
        for row in range(len(out_split_list)):
            expert_sum_aligned = (split_sum_each_row[row] + align - 1) // align * align
            expert_sum_aligned = max(expert_sum_aligned, align)
            out_split_list[row][-1] += expert_sum_aligned - split_sum_each_row[row]
        out_splits_aligned_flat = torch.tensor(out_split_list, device=device).view(-1)
    else:
        out_splits_aligned_flat = expected_out_splits.view(-1)

    local_offsets_inclusive = torch.cumsum(out_splits_aligned_flat, dim=0)
    expected_local_offsets = torch.cat((torch.tensor([0], device=device,
                                                     dtype=torch.int64), local_offsets_inclusive[:-1]))

    inp_offsets_flat = inp_offsets.view(-1)
    inp_splits_flat = inp_splits.view(-1)
    valid_chunks = []

    if inp.dim() == 1:
        inp_reshaped = inp.view(-1, 1)
    else:
        inp_reshaped = inp

    for i in range(inp_splits_flat.numel()):
        start = inp_offsets_flat[i].item()
        length = inp_splits_flat[i].item()
        valid_chunks.append(inp_reshaped[start:start + length])

    if rank_in_row:
        compact_inp = torch.cat(valid_chunks).view(-1)
    else:
        # Transpose chunks: (NE, World) -> (World, NE)
        chunks_grid = [valid_chunks[i:i + world_size] for i in range(0, len(valid_chunks), world_size)]
        chunks_transposed = list(map(list, zip(*chunks_grid)))
        ordered_chunks = [c for row in chunks_transposed for c in row]
        compact_inp = torch.cat(ordered_chunks).view(-1)

    out_numel_tokens = expected_out_splits.sum().item()
    out_numel_elements = out_numel_tokens * token_dim
    expected_buffer_flat = torch.empty(out_numel_elements, dtype=dtype, device=device)

    if rank_in_row:
        inp_splits_rank_tokens = inp_splits.reshape(world_size, ne).sum(1)
    else:
        inp_splits_rank_tokens = inp_splits.t().contiguous().reshape(world_size, ne).sum(1)

    out_splits_rank_tokens = raw_recv_splits.reshape(world_size, ne).sum(1)

    inp_splits_rank_elements = [s * token_dim for s in inp_splits_rank_tokens.tolist()]
    out_splits_rank_elements = [s * token_dim for s in out_splits_rank_tokens.tolist()]

    dist.all_to_all_single(expected_buffer_flat, compact_inp, out_splits_rank_elements, inp_splits_rank_elements,
                           group=group)

    expected_chunks = []

    if rank_in_row:  # raw_recv_splits is always rank-major
        recv_splits_flat = raw_recv_splits.view(-1)
        recv_splits_elements = recv_splits_flat * token_dim

        buffer_offsets_elements = torch.cat((torch.tensor([0], device=device), torch.cumsum(recv_splits_elements,
                                                                                            0)[:-1]))

        for j in range(ne):
            for i in range(world_size):
                flat_idx_rm = i * ne + j
                offset_elem = buffer_offsets_elements[flat_idx_rm].item()
                length_elem = recv_splits_elements[flat_idx_rm].item()
                chunk_flat = expected_buffer_flat[offset_elem:offset_elem + length_elem]
                chunk_reshaped = chunk_flat.view(-1, token_dim)
                expected_chunks.append(chunk_reshaped)
    else:
        # Linear cut
        recv_splits_flat = raw_recv_splits.view(-1)
        start_elem = 0
        for i in range(recv_splits_flat.numel()):
            length_elem = recv_splits_flat[i].item() * token_dim
            chunk_flat = expected_buffer_flat[start_elem:start_elem + length_elem]
            chunk_reshaped = chunk_flat.view(-1, token_dim)
            expected_chunks.append(chunk_reshaped)
            start_elem += length_elem

    return (expected_out_splits.to(torch.int32).view(-1), expected_source_offsets.to(torch.int32).view(-1),
            expected_local_offsets.to(torch.int32).view(-1), expected_chunks)


def check_with_golden(golden_results: tuple, received_out_data: torch.Tensor, received_out_splits: torch.Tensor = None,
                      received_source_offsets: torch.Tensor = None, received_local_offsets: torch.Tensor = None,
                      verbose: bool = True, allowed_ranks: list = None) -> None:
    rank = dist.get_rank()
    if allowed_ranks is None:
        allowed_ranks = list(range(dist.get_world_size()))
    expected_out_splits, expected_source_offset, expected_local_offset, expected_chunks = golden_results

    if received_out_splits is not None:
        received_out_splits = received_out_splits.to(torch.int32)
        try:
            torch.testing.assert_close(received_out_splits, expected_out_splits, msg="Output splits mismatch")
        except AssertionError:
            if verbose and rank in allowed_ranks:
                print("expected_out_splits", expected_out_splits)
                print("received_out_splits", received_out_splits)
            return -1

    if received_source_offsets is not None:
        received_source_offsets = received_source_offsets.to(torch.int32)
        try:
            torch.testing.assert_close(received_source_offsets, expected_source_offset, msg="Source offsets mismatch")
        except AssertionError:
            if verbose and rank in allowed_ranks:
                print("expected_source_offset", expected_source_offset)
                print("received_source_offsets", received_source_offsets)
            return -1

    if received_local_offsets is not None:
        received_local_offsets = received_local_offsets.to(torch.int32)
        try:
            torch.testing.assert_close(received_local_offsets, expected_local_offset, msg="Local offsets mismatch")
        except AssertionError:
            if verbose and rank in allowed_ranks:
                print("expected_local_offset", expected_local_offset)
                print("received_local_offsets", received_local_offsets)
            return -1

    if received_out_data.dim() == 1 and len(expected_chunks) > 0 and expected_chunks[0].dim() > 1:
        token_dim = expected_chunks[0].shape[1]
    elif received_out_data.dim() > 1:
        token_dim = received_out_data.shape[1]
    else:
        token_dim = 1

    for i, expected_chunk in enumerate(expected_chunks):
        start_token_idx = expected_local_offset[i].item()
        length_token_cnt = expected_out_splits[i].item()
        if received_out_data.dim() > 1:
            received_chunk = received_out_data[start_token_idx:start_token_idx + length_token_cnt]
        else:
            start_elem = start_token_idx * token_dim
            end_elem = (start_token_idx + length_token_cnt) * token_dim
            received_chunk = received_out_data[start_elem:end_elem].view(-1, token_dim)

        try:
            torch.testing.assert_close(received_chunk, expected_chunk, msg=f"Data mismatch idx {i}")
        except AssertionError:
            if verbose and rank in allowed_ranks:
                print(f"expected_chunk {expected_chunk}")
                print(f"received_chunk {received_chunk}")
            return -1
    return 0


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ne", type=int, default=8, help="number of experts per rank")
    parser.add_argument("--token_len", type=int, default=16, help="length of each token, unit: elements")
    parser.add_argument("-K", type=int, default=10, help="max val per splits")
    parser.add_argument("--grid_size", default=1, type=int, help="grid size")
    parser.add_argument("--world_size", default=8, type=int, help="world size when test kernels")
    parser.add_argument("--rank_is_row_in", action="store_true", help="if rank is row in", default=False)
    parser.add_argument("--rounds", default=1, type=int, help="number of rounds to run")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--warmup_iters", default=10, type=int, help="warmup iterations")
    parser.add_argument("--align", default=1, type=int, help="align size", choices=[1, 8, 16, 32])
    parser.add_argument("--dtype", default="float32", help="data type for token data", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--check", action="store_true", help="check with golden results", default=False)
    parser.add_argument("--info", default="", type=str, help="info string")
    parser.add_argument("--verbose", action="store_true", help="print verbose info", default=False)
    parser.add_argument("--test_with_offset", action="store_true", help="test with random gaps/offsets", default=False)
    parser.add_argument("--test_v2", action="store_true", help="test v2 kernel", default=False)
    return parser.parse_args()


def test_all_to_all_v_offset(args):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    ne, k, token_len, token_dtype = args.ne, args.K, args.token_len, DTYPE_MAP[args.dtype]
    test_with_offset, test_v2 = args.test_with_offset, args.test_v2
    ctx = create_context(rank, world_size, ne, k, token_len, token_dtype, use_v2_kernel=test_v2)
    output = torch.empty((ctx.max_num_token, token_len), dtype=token_dtype, device="cuda")

    deterministic = False  # set to True for debugging
    print_kwargs = {"allowed_ranks": list(range(world_size)), "need_sync": True}
    out_splits_offsets = ctx.combine_splits_offsets if args.rank_is_row_in else ctx.dispatch_splits_offsets

    def create_data(rank_in_row):
        if rank_in_row:
            M, N = world_size, ne
        else:
            M, N = ne, world_size

        in_offsets = None
        if test_with_offset:
            in_splits = torch.randint(1, k, (M, N), dtype=torch.int32, device="cuda")
            # Generate random padding (gaps)
            gaps = torch.randint(0, 5, (M, N), dtype=torch.int32, device="cuda")
            flat_splits = in_splits.view(-1)
            flat_gaps = gaps.view(-1)
            offsets_inclusive = torch.cumsum(flat_splits + flat_gaps, dim=0)
            in_offsets_flat = torch.cat((torch.tensor([0], device="cuda", dtype=torch.int32), offsets_inclusive[:-1]))
            in_offsets = in_offsets_flat.view(M, N).to(torch.int32)
            total_tokens = offsets_inclusive[-1].item()
        else:
            if deterministic:
                in_splits = torch.full((M, N), rank + 1, dtype=torch.int32, device="cuda")
                total_tokens = in_splits.sum().item()
            else:
                in_splits = torch.randint(1, k, (M, N), dtype=torch.int32, device="cuda")
                total_tokens = in_splits.sum().item()

        if deterministic:
            tokens = torch.full((total_tokens, token_len), rank + 1, dtype=token_dtype, device="cuda")
        else:
            tokens = torch.randn((total_tokens, token_len), dtype=token_dtype, device="cuda")

        return in_splits, in_offsets, tokens, total_tokens

    # 1. Correctness Check
    if args.check:
        has_fault = False
        for rid in range(args.rounds):
            if rank == 0:
                print(f"round {rid}")
            in_splits, in_offsets, tokens, num_in_tokens = create_data(args.rank_is_row_in)
            golden = get_golden_results(in_splits, tokens, world_size, ne, args.align, padded_offset=in_offsets,
                                        rank_in_row=args.rank_is_row_in)
            straggler(rank)
            if test_v2:
                _, send_token_num, recv_token_num = all_to_all_v_offset_op_v2(
                    ctx,
                    rank_in_row=args.rank_is_row_in,
                    input=tokens,
                    output=output,
                    in_splits=in_splits,
                    out_splits=golden[0],
                    copy_to_symm_buffer=True,
                    return_stat=True,
                    return_tensor_sliced=False,
                    grid_size=args.grid_size,
                )
                ret_code = check_with_golden(
                    received_out_data=output,
                    golden_results=golden,
                )
                if num_in_tokens != send_token_num or recv_token_num != golden[0].sum().item():
                    ret_code = -1
            else:
                ret, _ = all_to_all_v_offset_op(
                    ctx,
                    input=tokens,
                    output=output,
                    in_splits=in_splits,
                    in_offset=in_offsets,
                    copy_to_symm_buffer=True,
                    grid_size=args.grid_size,
                    major_align=args.align,
                    rank_in_row=args.rank_is_row_in,
                    has_input_offset=test_with_offset,
                )
                ret_code = check_with_golden(
                    received_out_splits=out_splits_offsets[:ctx.nsplits],
                    received_local_offsets=out_splits_offsets[ctx.nsplits:],
                    received_out_data=ret,
                    golden_results=golden,
                )
            if ret_code != 0:
                has_fault = True
        torch.cuda.current_stream().synchronize()
        msg_suffix = "(Explicit Offset)" if test_with_offset else "(Dense)"
        if not has_fault:
            dist_print(f"✅ rank {rank} correctness check passed {msg_suffix}", **print_kwargs)
        else:
            dist_print(f"❌ rank {rank} correctness check failed {msg_suffix}", **print_kwargs)

    # 2. Performance Profiling
    in_splits, in_offsets, tokens, num_in_tokens = create_data(args.rank_is_row_in)
    golden = get_golden_results(in_splits, tokens, world_size, ne, args.align, padded_offset=in_offsets,
                                rank_in_row=args.rank_is_row_in)

    def op():
        if test_v2:
            all_to_all_v_offset_op_v2(
                ctx,
                rank_in_row=args.rank_is_row_in,
                input=tokens,
                output=output,
                in_splits=in_splits,
                out_splits=golden[0],
                copy_to_symm_buffer=True,
                return_stat=False,
                return_tensor_sliced=False,
                grid_size=args.grid_size,
                profiling=args.profile,
            )
        else:
            all_to_all_v_offset_op(ctx, input=tokens, output=output, in_splits=in_splits, in_offset=in_offsets,
                                   copy_to_symm_buffer=False,  # already copied
                                   grid_size=args.grid_size, major_align=args.align, rank_in_row=args.rank_is_row_in,
                                   return_tensor=False, has_input_offset=test_with_offset, profiling=args.profile)

    # with group_profile(f"all2all_v_offset_{args.info}", do_prof=args.profile,
    #                     compress=True, group=dist.group.WORLD):
    #     dist.barrier()
    #     sleep_async(20)
    #     for _ in range(1 + args.warmup_iters + args.iters):
    #         op()

    sleep_async(20)  # incase of CPU bound
    _, perf_ms = perf_func(op, iters=args.iters, warmup_iters=args.warmup_iters)
    if args.profile:
        ctx.dump_profiler_trace(info=args.info)

    total_valid_tokens = out_splits_offsets[:ctx.nsplits].sum().item() if not test_v2 else golden[0][:ctx.nsplits].sum(
    ).item()
    total_bytes = total_valid_tokens * tokens.element_size() * token_len
    gbps = (total_bytes / 1e9) / (perf_ms / 1000)
    total_copied_bytes = in_splits.sum().item() * token_len * tokens.element_size()  # copied to comm buffer
    dist_print(
        f"rank {rank} | Total copy bytes {total_copied_bytes} | Latency: {perf_ms * 1000:.2f} us | Throughput: {gbps:.2f} GB/s",
        **print_kwargs)
    torch.cuda.current_stream().synchronize()
    ctx.finalize()


def profile_real_aether_testcases(args):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    token_dtype = DTYPE_MAP[args.dtype]
    ne, seq_len, topk, token_len = 48, 8192, 8, 2048
    grid_size = 48
    print_kwargs = {"allowed_ranks": list(range(world_size)), "need_sync": True}

    base_path = "dev_output/checked_splits_2"
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Folder {base_path} does not exist.")

    def load_splits_to_gpu(idx) -> tuple[torch.Tensor, torch.Tensor]:
        in_path = os.path.join(base_path, f"rank_{rank}_insplits_step_{idx}_layer_0")
        out_path = os.path.join(base_path, f"rank_{rank}_outsplits_step_{idx}_layer_0")
        device = torch.device(f"cuda:{rank}")
        in_tensor = torch.load(in_path, map_location=device, weights_only=False)
        out_tensor = torch.load(out_path, map_location=device, weights_only=False)
        assert in_tensor.shape == (world_size, ne) and out_tensor.shape == (ne, world_size)
        return in_tensor, out_tensor

    def analyze_testcases():
        import re
        global_max = torch.iinfo(torch.int32).min
        min_idx, max_idx = float('inf'), -1
        idx_pattern = re.compile(r"step_(\d+)")
        for filename in os.listdir(base_path):
            if "insplits" in filename or "outsplits" in filename:
                idx_match = idx_pattern.search(filename)
                if idx_match:
                    current_idx = int(idx_match.group(1))
                    max_idx = max(max_idx, current_idx)
                    min_idx = min(min_idx, current_idx)
                file_path = os.path.join(base_path, filename)
                tensor = torch.load(file_path, map_location="cpu").to(torch.int32)
                current_max = tensor.max().item()
                if current_max > global_max:
                    global_max = current_max
        return global_max, int(min_idx), int(max_idx)

    k, min_idx, max_idx = analyze_testcases()
    ctx = create_context(rank, world_size, ne, k, token_len, token_dtype, use_v2_kernel=True,
                         max_token_num_per_rank=seq_len * topk, overflow_factor=world_size)
    input = torch.randn((seq_len * topk * world_size, token_len), dtype=token_dtype,
                        device="cuda")  # same shape as ctx's max symm buffer
    triton_out = torch.empty_like(input)
    torch_out = torch.empty_like(input)

    for idx in range(min_idx, max_idx + 1):
        dist.barrier()
        if rank == 0:
            print("=" * 84 + f" step {idx} " + "=" * 84)
        dispatch_splits, combine_splits = load_splits_to_gpu(idx)  # (ws, ne) and (ne, ws)
        combine_splits_t = combine_splits.t().contiguous()
        dispatch_splits_per_rank = dispatch_splits.sum(dim=1)
        combine_splits_per_rank = combine_splits_t.sum(dim=1)
        num_dispatch_tokens = dispatch_splits.sum().item()
        num_combine_tokens = combine_splits.sum().item()

        def triton_op(rank_is_row_in):
            all_to_all_v_offset_op_v2(
                ctx,
                rank_in_row=rank_is_row_in,
                input=input,
                output=triton_out,
                in_splits=dispatch_splits if rank_is_row_in else combine_splits,
                out_splits=combine_splits if rank_is_row_in else dispatch_splits,
                copy_to_symm_buffer=True,
                return_stat=False,
                return_tensor_sliced=False,
                grid_size=grid_size,
                profiling=args.profile,
            )
            return triton_out

        def torch_op(rank_is_row_in):
            split_sizes = tuple(combine_splits_t.flatten().tolist())
            if rank_is_row_in:
                inp = input[:num_dispatch_tokens]
                out = torch_out[:num_combine_tokens]
                dist.all_to_all_single(
                    out,
                    inp,
                    output_split_sizes=combine_splits_per_rank.tolist(),
                    input_split_sizes=dispatch_splits_per_rank.tolist(),
                )
                data_splits = out.split(split_sizes, dim=0)
                ret = torch.cat(
                    [data_splits[i % world_size * ne + i // world_size] for i in range(world_size * ne)],
                    dim=0,
                ).contiguous()
                return ret
            else:
                inp = input[:num_combine_tokens]
                out = torch_out[:num_dispatch_tokens]
                data_splits = inp.split(
                    [split_sizes[i % world_size * ne + i // world_size] for i in range(world_size * ne)],
                    dim=0,
                )
                inp_unpermute = torch.cat(
                    [data_splits[i % ne * world_size + i // ne] for i in range(world_size * ne)],
                    dim=0,
                )
                dist.all_to_all_single(
                    out,
                    inp_unpermute,
                    output_split_sizes=dispatch_splits_per_rank.tolist(),
                    input_split_sizes=combine_splits_per_rank.tolist(),
                )
                return out

        with group_profile(f"{args.info}/triton_dispatch_idx_{idx}_merged", do_prof=args.profile, compress=True,
                           group=dist.group.WORLD):
            sleep_async(20)
            _, perf_triton_dispatch = perf_func(partial(triton_op, True), iters=args.iters,
                                                warmup_iters=args.warmup_iters)

        with group_profile(f"{args.info}/torch_dispatch_idx_{idx}_merged", do_prof=args.profile, compress=True,
                           group=dist.group.WORLD):
            sleep_async(20)
            torch_res, perf_torch_dispatch = perf_func(partial(torch_op, True), iters=args.iters,
                                                       warmup_iters=args.warmup_iters)

        torch.cuda.current_stream().synchronize()
        torch.testing.assert_close(torch_res, triton_out[:num_combine_tokens], msg="dispatch not equal")
        # if rank == 0:
        #     print("combine_splits:\n", combine_splits)
        #     print("workspace:\n", ctx.local_splits_workspace)
        # assert_allclose(torch_res, triton_out[:num_combine_tokens], atol=1e-3, rtol=1e-3)

        # with group_profile(f"dev_output/real_testcases_triton_combine_idx_{idx}",
        #                    do_prof=args.profile, compress=True, group=dist.group.WORLD):
        sleep_async(20)
        _, perf_triton_combine = perf_func(partial(triton_op, False), iters=args.iters, warmup_iters=args.warmup_iters)
        sleep_async(20)
        torch_res, perf_torch_combine = perf_func(partial(torch_op, False), iters=args.iters,
                                                  warmup_iters=args.warmup_iters)
        torch.cuda.current_stream().synchronize()
        torch.testing.assert_close(torch_res, triton_out[:num_dispatch_tokens], msg="combine not equal")

        total_dispatch_bytes = num_dispatch_tokens * input.element_size() * token_len
        triton_dispatch_gbps = (total_dispatch_bytes / 1e9) / (perf_triton_dispatch / 1000)
        torch_dispatch_gbps = (total_dispatch_bytes / 1e9) / (perf_torch_dispatch / 1000)
        total_combine_bytes = num_combine_tokens * input.element_size() * token_len
        triton_combine_gbps = (total_combine_bytes / 1e9) / (perf_triton_combine / 1000)
        torch_combine_gbps = (total_combine_bytes / 1e9) / (perf_torch_combine / 1000)
        dispatch_speedup = perf_torch_dispatch / perf_triton_dispatch if perf_triton_dispatch != 0 else -1
        combine_speedup = perf_torch_combine / perf_triton_combine if perf_triton_combine != 0 else -1
        dist_print(
            f"rank {rank:2d} | Dispatch | Triton Latency: {perf_triton_dispatch * 1000:8.2f} us | Torch Latency: {perf_torch_dispatch * 1000:8.2f} us | Triton Throughput: {triton_dispatch_gbps:8.2f} GB/s | Torch Throughput: {torch_dispatch_gbps:8.2f} GB/s | Throughput Speedup: {dispatch_speedup:5.2f}x\n"
            f"rank {rank:2d} | Combine  | Triton Latency: {perf_triton_combine * 1000:8.2f} us | Torch Latency: {perf_torch_combine * 1000:8.2f} us | Triton Throughput: {triton_combine_gbps:8.2f} GB/s | Torch Throughput: {torch_combine_gbps:8.2f} GB/s | Throughput Speedup: {combine_speedup:5.2f}x",
            **print_kwargs)
        torch.cuda.current_stream().synchronize()
        if args.profile:
            ctx.dump_profiler_trace(info=args.info)

    torch.cuda.current_stream().synchronize()
    ctx.finalize()


if __name__ == "__main__":
    args = parse_args()
    WORLD = initialize_distributed()
    test_all_to_all_v_offset(args)
    #profile_real_aether_testcases(args)
    # to run:
    # bash +x scripts/launch.sh python/triton_dist/test/nvidia/test_all_to_all_vdev_2d_offset.py --ne 16 --grid_size 16 --check --rounds 10 --rank_is_row_in --test_v2
    # bash +x scripts/launch.sh python/triton_dist/test/nvidia/test_all_to_all_vdev_2d_offset.py --ne 48 --grid_size 48 --check --rounds 10 --token_len 8192  --test_v2
    # bash +x scripts/launch.sh python/triton_dist/test/nvidia/test_all_to_all_vdev_2d_offset.py --ne 16 --grid_size 16 --check --rounds 10 --rank_is_row_in
    # bash +x scripts/launch.sh python/triton_dist/test/nvidia/test_all_to_all_vdev_2d_offset.py --ne 16 --grid_size 16 --check --rounds 10 --rank_is_row_in --align 32
    # bash +x scripts/launch.sh python/triton_dist/test/nvidia/test_all_to_all_vdev_2d_offset.py --ne 16 --grid_size 16 --check --rounds 10 --test_with_offset
    # bash +x scripts/launch.sh python/triton_dist/test/nvidia/test_all_to_all_vdev_2d_offset.py --ne 16 --grid_size 16 --check --rounds 10 --test_with_offset --token_len 4096
    # bash +x scripts/launch.sh python/triton_dist/test/nvidia/test_all_to_all_vdev_2d_offset.py --warmup_iters 10 --iters 10 --rank_is_row_in --check --align 32
