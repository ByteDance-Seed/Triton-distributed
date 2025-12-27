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
import torch
import triton
import argparse
import random
import os
from functools import partial
from typing import Optional

from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import (dist_print, initialize_distributed, nvshmem_barrier_all_on_stream, finalize_distributed,
                               sleep_async, rand_tensor)
from triton_dist.layers.nvidia import GemmARLayer


def _make_data(M):
    global scale
    scale = RANK + 1
    if input_dtype == torch.int8:
        scale_factor = 1
    else:
        scale_factor = 0.01 * scale
    device = torch.cuda.current_device()
    A = rand_tensor((M, local_K), dtype=input_dtype, device=device) * scale_factor
    weight = rand_tensor((args.N, local_K), dtype=input_dtype, device=device)
    bias = rand_tensor((M, args.N), dtype=input_dtype, device=device) if args.has_bias else None
    return A, weight, bias


def torch_gemm_ar(
    A: torch.Tensor,  # [M, local_k]
    weight: torch.Tensor,  # [N, local_K]
    bias: Optional[torch.Tensor],
    tp_group,
):
    output = torch.matmul(A, weight.T)
    if bias:
        output = output + bias
    torch.distributed.all_reduce(output, group=tp_group)
    return output


# per-channel quantization for int8_w8a8
def torch_scaled_gemm_ar(A, weight, scale_a, scale_b, out_dtype, tp_group):
    output = torch.matmul(A.to(torch.float32), weight.T.to(torch.float32))
    if bias is not None:
        output = output.to(torch.float32) * scale_a.view(-1, 1) * scale_b.view(1, -1) + bias
    else:
        output = output.to(torch.float32) * scale_a.view(-1, 1) * scale_b.view(1, -1)
    output = output.to(out_dtype)
    torch.distributed.all_reduce(output, group=tp_group)
    return output


def cutlass_scaled_gemm_ar(A, weight, scale_a, scale_b, out_dtype, tp_group):
    output = int8_scaled_mm(A, weight.T, scale_a, scale_b, out_dtype)
    torch.distributed.all_reduce(output, group=tp_group)
    return output


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

THRESHOLD_MAP = {
    torch.float16: 1e-2, torch.bfloat16: 6e-2, torch.float8_e4m3fn: 1e-2, torch.float8_e5m2: 1e-2, torch.int8: 0.1
}


def straggler(rank):
    clock_rate = torch.cuda.clock_rate() * 1e6
    cycles = random.randint(0, int(clock_rate * 0.01)) * (rank + 1)
    torch.cuda._sleep(cycles)


def print_tile_info(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE, NUM_GEMM_SM, NUM_COMM_SM, world_size):

    def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        return pid_m, pid_n

    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_m_per_rank = triton.cdiv(num_pid_m, world_size)
    num_tiles_per_rank = triton.cdiv(num_tiles, world_size)
    for rank in range(world_size):
        print(
            f"rank={rank}, M={M}, N={N}, BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, num_pid_m={num_pid_m}, num_pid_n={num_pid_n}, num_tiles={num_tiles}, world_size={world_size}, num_tiles_per_rank={num_tiles_per_rank}, num_sms={NUM_GEMM_SM}"
        )
        for sm_id in range(NUM_GEMM_SM):
            for tile_id in range(sm_id, num_tiles, NUM_GEMM_SM):
                pid_m, pid_n = _compute_pid(tile_id, GROUP_SIZE * num_pid_n, num_pid_m, GROUP_SIZE, NUM_GEMM_SM)
                gemm_barrier_idx = pid_m * num_pid_n + pid_n
                row_gemm_barrier_idx = pid_m

                alpha = 0
                beta = 0
                pid_m = (pid_m + ((((rank ^ alpha) + beta) % world_size) * num_pid_m_per_rank)) % num_pid_m
                tile_rank_id = (pid_m * num_pid_n + pid_n) // num_tiles_per_rank

                print(
                    f"sm_id=>{sm_id}, tile_id=>{tile_id}, pid_m=>{pid_m}, pid_n=>{pid_n}, offset=>{tile_id * BLOCK_SIZE_N * BLOCK_SIZE_M}"
                )
                print(
                    f"tile-wise barrier=>{gemm_barrier_idx}, row-wise barrier=>{row_gemm_barrier_idx}, rank-wise barrier=>{tile_rank_id}"
                )

    numel = M * N
    elem_per_pid = BLOCK_SIZE_M * N
    BLOCK_SIZE_COMM = 8192
    num_tiles = triton.cdiv(numel, BLOCK_SIZE_COMM)

    for rank in range(world_size):
        print(f"rank={rank}, NUM_COMM_SM={NUM_COMM_SM}, elem={numel}, n_blocks={num_tiles}")
        for sm_id in range(NUM_COMM_SM):
            for tile_id in range(sm_id, num_tiles, NUM_COMM_SM):
                pid_m = tile_id * BLOCK_SIZE_COMM // elem_per_pid
                pid_n = tile_id * BLOCK_SIZE_COMM % elem_per_pid // BLOCK_SIZE_COMM
                print(
                    f"sm_id {sm_id}, tile_id {tile_id}, barrier id {tile_id * BLOCK_SIZE_COMM // elem_per_pid}, pid_m {pid_m}, pid_n {pid_n}"
                )


def print_ring_info(world_size):
    for rank in range(world_size):
        segment = rank
        for i in range(0, world_size):
            to_rank = (rank + world_size - 1 - i) % world_size
            print(f"rank {rank}, stage {i}, to_rank {to_rank}, segment {segment}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", help="data type")

    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true", help="correctness check")
    parser.add_argument("--verify-iters", default=10, type=int)
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction,
                        default=torch.cuda.get_device_capability() >= (9, 0))

    parser.add_argument("--low-latency", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--copy-to-local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_comm_sms", default=16, type=int, help="num sm for allreduce")
    parser.add_argument("--row-wise", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument("--quant", default=False, action="store_true")
    parser.add_argument("--quant_out_dtype", default="bfloat16", help="output data type with scale")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print", default=False, action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    # init
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    torch.cuda.set_device(LOCAL_RANK)

    args = parse_args()
    dist_print(args)
    tp_group = initialize_distributed(args.seed)
    if torch.cuda.get_device_capability()[0] < 9:
        assert not args.persistent, "persistent is not supported on cuda < 9.0"

    input_dtype = DTYPE_MAP[args.dtype]
    if args.quant:
        assert input_dtype == torch.int8 or input_dtype == torch.float8_e4m3fn or input_dtype == torch.float8_e5m2, "only support fp8/int8 for quant"
        output_dtype = DTYPE_MAP[args.quant_out_dtype]
        from sgl_kernel import int8_scaled_mm
    else:
        output_dtype = input_dtype
    atol = THRESHOLD_MAP[output_dtype]
    rtol = THRESHOLD_MAP[output_dtype]

    assert args.K % WORLD_SIZE == 0
    local_K = args.K // WORLD_SIZE

    if args.quant:
        scale_a = torch.randn((args.M, ), device="cuda", dtype=torch.float32)
        scale_b = torch.randn((args.N, ), device="cuda", dtype=torch.float32)
    else:
        scale_a = None
        scale_b = None

    scale = 0

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    NUM_GEMM_SMS = NUM_SMS - args.num_comm_sms
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 1
    gemm_config = triton.Config(
        {
            'BLOCK_SIZE_M': BLOCK_SIZE_M, 'BLOCK_SIZE_N': BLOCK_SIZE_N, "BLOCK_SIZE_K": BLOCK_SIZE_K, "GROUP_SIZE_M":
            GROUP_SIZE_M, "NUM_GEMM_SMS": NUM_GEMM_SMS
        }, num_stages=3, num_warps=8)
    TILE_MAP_LEVEL = (args.row_wise == 1)

    gemm_ar_op = GemmARLayer(tp_group, args.M, args.N, args.K, input_dtype, output_dtype, LOCAL_WORLD_SIZE,
                             persistent=args.persistent, use_ll_kernel=args.low_latency,
                             copy_to_local=args.copy_to_local, NUM_COMM_SMS=args.num_comm_sms,
                             TILE_MAP_LEVEL=TILE_MAP_LEVEL, user_gemm_config=gemm_config)

    if args.print and RANK == 0:
        print_tile_info(args.M, args.N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, GROUP_SIZE=GROUP_SIZE_M,
                        NUM_GEMM_SM=NUM_GEMM_SMS, NUM_COMM_SM=args.num_comm_sms, world_size=WORLD_SIZE)

        print_ring_info(WORLD_SIZE)

    if args.check:
        assert args.copy_to_local
        for n in range(args.iters):
            torch.cuda.empty_cache()
            input_list = [_make_data(args.M) for _ in range(args.verify_iters)]
            dist_out_list, torch_out_list, cutlass_out_list = [], [], []
            # torch impl
            for A, weight, bias in input_list:
                if args.quant:
                    torch_out = torch_scaled_gemm_ar(A, weight, scale_a, scale_b, output_dtype, tp_group)
                    cutlass_out = cutlass_scaled_gemm_ar(A, weight, scale_a, scale_b, output_dtype, tp_group)
                    cutlass_out_list.append(cutlass_out)
                else:
                    torch_out = torch_gemm_ar(A, weight, bias, tp_group)
                torch_out_list.append(torch_out)

            # dist triton impl
            for A, weight, bias in input_list:
                straggler(RANK)
                dist_out = gemm_ar_op.forward(A, weight, bias, scale_a, scale_b)
                dist_out_list.append(dist_out)

            # verify
            for idx, (torch_out, dist_out) in enumerate(zip(torch_out_list, dist_out_list)):
                assert_allclose(torch_out, dist_out, rtol=rtol, atol=atol, verbose=False)

            if args.quant:
                for idx, (cutlass_out, dist_out) in enumerate(zip(cutlass_out_list, dist_out_list)):
                    assert_allclose(cutlass_out, dist_out, rtol=rtol, atol=atol, verbose=False)

        print(f"RANK[{RANK}]: pass.")
        gemm_ar_op.finalize()
        finalize_distributed()
        exit(0)

    # warm up
    A, weight, bias = _make_data(args.M)
    if args.quant:
        ar_input = torch.matmul(A.to(torch.float32), weight.T.to(torch.float32)).to(torch.bfloat16)
    else:
        ar_input = torch.matmul(A, weight.T)

    x = gemm_ar_op.forward(A, weight, bias, scale_a, scale_b)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    with group_profile(f"gemm_ar_{args.M}x{args.N}x{args.K}_{os.environ['TORCHELASTIC_RUN_ID']}", args.profile,
                       group=tp_group):
        dist_triton_output, dist_triton_perf = perf_func(partial(gemm_ar_op.forward, A, weight, bias, scale_a, scale_b),
                                                         iters=args.iters, warmup_iters=args.warmup)

        if args.quant:
            torch_output, torch_perf = perf_func(
                partial(cutlass_scaled_gemm_ar, A, weight, scale_a, scale_b, output_dtype, tp_group), iters=args.iters,
                warmup_iters=args.warmup)
        else:
            torch_output, torch_perf = perf_func(partial(torch_gemm_ar, A, weight, bias, tp_group), iters=args.iters,
                                                 warmup_iters=args.warmup)

        if args.quant:
            _, torch_gemm_perf = perf_func(partial(int8_scaled_mm, A, weight.T, scale_a, scale_b, output_dtype),
                                           iters=args.iters, warmup_iters=args.warmup)
        else:
            _, torch_gemm_perf = perf_func(partial(torch.matmul, A, weight.T), iters=args.iters,
                                           warmup_iters=args.warmup)

        torch.cuda.synchronize()
        sleep_async(100)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        _, dist_triton_gemm_perf = perf_func(partial(gemm_ar_op.forward_gemm, A, weight, bias, scale_a, scale_b),
                                             iters=args.iters, warmup_iters=args.warmup)

        torch.cuda.synchronize()
        sleep_async(100)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        _, dist_triton_ar_perf = perf_func(partial(gemm_ar_op.forward_ar, ar_input), iters=args.iters,
                                           warmup_iters=args.warmup)

        torch.cuda.synchronize()
        sleep_async(100)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        _, torch_ar_perf = perf_func(partial(torch.distributed.all_reduce, ar_input, group=tp_group), iters=args.iters,
                                     warmup_iters=args.warmup)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    assert_allclose(torch_output, dist_triton_output, rtol=rtol, atol=atol)

    dist_print(f"dist-triton #{RANK}, gemm={dist_triton_gemm_perf:0.4f}", need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(
        f"dist-triton #{RANK}, ar={dist_triton_ar_perf:0.4f}, band={ar_input.element_size()*args.M*args.N/1024/1024/1024/dist_triton_ar_perf*1000:0.4f}GBps",
        need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"dist-triton #{RANK}, total={dist_triton_perf:0.4f}", need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))

    dist_print(f"torch #{RANK}, gemm={torch_gemm_perf:0.4f}", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(
        f"torch #{RANK}, ar={torch_ar_perf:0.4f}, band={ar_input.element_size()*args.M*args.N/1024/1024/1024/torch_ar_perf*1000:0.4f}GBps",
        need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK}, total={torch_perf:0.4f}", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    gemm_ar_op.finalize()
    finalize_distributed()
