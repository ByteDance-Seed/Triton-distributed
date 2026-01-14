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
import torch.distributed as dist
from triton_dist.kernels.nvidia.all_to_all_vdev_2d_offset_inter_node import create_context, all_to_all_v_offset_op
from triton_dist.profiler_utils import perf_func, group_profile
from triton_dist.utils import initialize_distributed, dist_print, sleep_async, nvshmem_barrier_all_on_stream
import random


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


def check_with_golden(golden_results: tuple, received_out_data: torch.Tensor, received_out_splits: torch.Tensor,
                      received_source_offsets: torch.Tensor = None, received_local_offsets: torch.Tensor = None,
                      verbose: bool = True, allowed_ranks: list = None) -> None:
    rank = dist.get_rank()
    if allowed_ranks is None:
        allowed_ranks = list(range(dist.get_world_size()))
    received_out_splits = received_out_splits.to(torch.int32)
    expected_out_splits, expected_source_offset, expected_local_offset, expected_chunks = golden_results

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
                print(f"expected_chunk idx {i}", expected_chunk)
                print(f"received_chunk idx {i}'s sum", received_chunk.sum().item())
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
    parser.add_argument("--grid_size", default=0, type=int, help="grid size")
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
    return parser.parse_args()


def test_all_to_all_v_offset(args):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    ne, k, token_len, token_dtype = args.ne, args.K, args.token_len, DTYPE_MAP[args.dtype]
    test_with_offset = args.test_with_offset
    deterministic = False  # set to True for debugging
    if deterministic:
        k = world_size
        torch.set_printoptions(
            linewidth=100000,
            threshold=torch.inf,
        )
    ctx = create_context(rank, world_size, ne, k, token_len, token_dtype)
    output = torch.empty((ctx.max_token_elem, ), dtype=token_dtype, device="cuda")
    print_kwargs = {"allowed_ranks": list(range(world_size)), "need_sync": True}

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

        return in_splits, in_offsets, tokens

    def pre_copy(in_splits, in_offsets, tokens):
        ctx.in_splits_offsets[:ctx.nsplits].copy_(in_splits.view(-1))
        if in_offsets is not None:
            ctx.in_splits_offsets[ctx.nsplits:].copy_(in_offsets.view(-1))
        input_flat = tokens.contiguous().view(-1)
        ctx.send_data[:input_flat.numel()].copy_(input_flat)

    # 1. Correctness Check
    if args.check:
        has_fault = False
        for _ in range(args.rounds):
            in_splits, in_offsets, tokens = create_data(args.rank_is_row_in)
            straggler(rank)
            ret, _ = all_to_all_v_offset_op(
                ctx,
                input=tokens,
                output=output,
                in_splits=in_splits,
                in_offset=in_offsets,
                copy_to_symm_buffer=True,
                grid_size=args.grid_size if args.grid_size > 0 else None,
                major_align=args.align,
                rank_in_row=args.rank_is_row_in,
                has_input_offset=test_with_offset,
            )
            golden = get_golden_results(in_splits, tokens, world_size, ne, args.align, padded_offset=in_offsets,
                                        rank_in_row=args.rank_is_row_in)
            ret_code = check_with_golden(
                received_out_splits=ctx.out_splits_offsets[:ctx.nsplits],
                received_local_offsets=ctx.out_splits_offsets[ctx.nsplits:],
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
    in_splits, in_offsets, tokens = create_data(args.rank_is_row_in)
    pre_copy(in_splits, in_offsets, tokens)

    def op():
        all_to_all_v_offset_op(ctx, input=tokens, output=None, in_splits=in_splits, in_offset=in_offsets,
                               copy_to_symm_buffer=False,  # already copied
                               grid_size=args.grid_size if args.grid_size > 0 else None, major_align=args.align,
                               rank_in_row=args.rank_is_row_in, return_tensor=False, has_input_offset=test_with_offset,
                               profiling=args.profile)

    sleep_async(20)  # incase of CPU bound
    _, perf_ms = perf_func(op, iters=args.iters, warmup_iters=args.warmup_iters)

    if args.profile:
        with group_profile(f"all2all_v_offset_{args.info}", do_prof=True, compress=True, group=dist.group.WORLD):
            sleep_async(20)
            for _ in range(args.warmup_iters + args.iters):
                op()
        ctx.dump_profiler_trace(info=args.info)

    total_valid_tokens = ctx.out_splits_offsets[:ctx.nsplits].sum().item()
    total_bytes = total_valid_tokens * tokens.element_size() * token_len
    gbps = (total_bytes / 1e9) / (perf_ms / 1000)
    dist_print(f"rank {rank} | Latency: {perf_ms * 1000:.2f} us | Throughput: {gbps:.2f} GB/s", **print_kwargs)
    torch.cuda.current_stream().synchronize()
    ctx.finalize()


def profile_triton_all_testcases(args):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    k, token_dtype = args.K, DTYPE_MAP[args.dtype]
    test_with_offset = args.test_with_offset

    nes = [2, 4, 6, 8, 10, 12, 14, 16]
    token_lens = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    if rank == 0:
        print(f"== Start profiling with nes={nes} and token_lens={token_lens} ==")

    max_ne, max_token_len = nes[-1], token_lens[-1]
    ctx = create_context(rank, world_size, max_ne, k, max_token_len, token_dtype)
    print_kwargs = {
        "allowed_ranks": [0, 8],
    }

    max_send_len = ctx.send_data.numel()
    ctx.send_data.copy_(torch.randn(max_send_len, dtype=token_dtype, device="cuda"))

    for ne in nes:
        for token_len in token_lens:
            ctx.ne = ne
            ctx.token_len_elem = token_len
            ctx.__post_init__()
            dist.barrier()
            num_sms = min(ne, ctx.nsplits // 32)  # max num_warps=32 per block

            def op():
                all_to_all_v_offset_op(ctx, copy_to_symm_buffer=False,  # already copied
                                       grid_size=num_sms, major_align=args.align, rank_in_row=args.rank_is_row_in,
                                       return_tensor=False, has_input_offset=test_with_offset, profiling=False)

            avg_latency_ms = 0.0
            avg_throughput_gbps = 0.0
            for _ in range(args.rounds):
                ctx.in_splits_offsets[:ne * world_size].copy_(
                    torch.randint(1, k, (ne * world_size, ), dtype=torch.int32, device="cuda"))
                nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
                sleep_async(20)  # incase of CPU bound
                _, perf_ms = perf_func(op, iters=args.iters, warmup_iters=args.warmup_iters)

                total_valid_tokens = ctx.out_splits_offsets[:ctx.nsplits].sum().item()
                total_bytes = total_valid_tokens * token_dtype.itemsize * token_len
                gbps = (total_bytes / 1e9) / (perf_ms / 1000)
                avg_latency_ms += perf_ms
                avg_throughput_gbps += gbps
            dist_print(
                f"rank {rank} ne {ne} token_len {token_len} | Latency: {avg_latency_ms * 1000 / args.rounds:.2f} us | Throughput: {avg_throughput_gbps / args.rounds:.2f} GB/s",
                **print_kwargs)
    torch.cuda.current_stream().synchronize()
    ctx.finalize()


if __name__ == "__main__":
    args = parse_args()
    WORLD = initialize_distributed()
    test_all_to_all_v_offset(args)
    #profile_triton_all_testcases(args)
