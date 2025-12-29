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
import csv
from contextlib import nullcontext
from functools import partial
from typing import List

import random
import torch
import torch.distributed

from triton_dist.test.utils import assert_allclose, bitwise_equal
from triton_dist.utils import initialize_distributed, finalize_distributed, is_fp8_dtype
from triton_dist.profiler_utils import perf_func
from triton_dist.test.nvidia.ep_a2a_utils import quant_bf16_fp8

from triton_dist.kernels.nvidia.ulysses_sp_infer_gemm_a2a import (UlyssesSpInferPreAttnContext,
                                                                  ulysses_sp_infer_gemm_a2a_op)

print = partial(print, flush=True)


def _verify_and_check_bitwise(torch_outs: List[torch.Tensor], triton_dist_outs: List[torch.Tensor], atol, rtol):
    if isinstance(torch_outs, (torch.Tensor, )):
        torch_outs = [torch_outs]
        triton_dist_outs = [triton_dist_outs]
    is_bitwise = True
    for ref_out, triton_dist_out in zip(torch_outs, triton_dist_outs):
        if ref_out is None and triton_dist_out is None:
            continue
        if ref_out.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            ref_out = ref_out.to(torch.bfloat16)
        if triton_dist_out.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            triton_dist_out = triton_dist_out.to(torch.bfloat16)
        triton_dist_out = triton_dist_out.reshape(ref_out.shape)
        assert_allclose(ref_out, triton_dist_out, atol=atol, rtol=rtol)
        if not bitwise_equal(ref_out, triton_dist_out):
            is_bitwise = False
    return is_bitwise


def matmul_int8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    INT8 matrix multiplication using torch._int_mm
    torch._int_mm requires A.size(0) needs to be greater than 16
    b is expected to be (N, K) and will be transposed to (K, N)
    """
    M, _ = a.shape
    b_t = b.t()  # Transpose from (N, K) to (K, N)
    if M <= 16:
        return torch._int_mm(torch.nn.functional.pad(a, (0, 0, 0, 32 - M)), b_t)[:M, :]
    return torch._int_mm(a, b_t)


@torch.no_grad()
def torch_gemm_a2a(
    sp_group: torch.distributed.ProcessGroup,
    full_input: torch.Tensor,  # [seq_len, hidden]
    weight: torch.
    Tensor,  # [(local_q_nheads * k_head_dim + local_kv_nheads * k_head_dim + local_kv_nheads * v_head_dim) * sp_size, hidden]
    full_input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_nheads: int,
    kv_nheads: int,
    k_head_dim: int,
    v_head_dim: int,
    quant_out: bool,
):
    gemm_n = weight.shape[0]
    seq_len, hidden_dim = full_input.shape
    sp_size = sp_group.size()
    rank = sp_group.rank()

    is_s8 = full_input.dtype == torch.int8
    is_fp8 = is_fp8_dtype(full_input.dtype)
    assert seq_len % sp_size == 0
    assert q_nheads % sp_size == 0 and kv_nheads % sp_size == 0

    local_q_nheads = q_nheads // sp_size
    local_kv_nheads = kv_nheads // sp_size
    local_out_feat = local_q_nheads * k_head_dim + local_kv_nheads * k_head_dim + local_kv_nheads * v_head_dim
    assert local_out_feat * sp_size == gemm_n

    weight_a2a = weight[rank * local_out_feat:(rank + 1) * local_out_feat, :].contiguous()
    weight_scale_a2a = weight_scale[:, rank * local_out_feat:(rank + 1) * local_out_feat]
    torch.distributed.barrier()

    if is_s8:
        accum = matmul_int8(full_input, weight_a2a).to(torch.float32)
        output = full_input_scale * weight_scale_a2a * accum
    elif is_fp8:
        dequant_scale = full_input_scale * weight_scale_a2a
        full_input_bf16 = full_input.to(torch.bfloat16)
        weight_a2a_bf16 = weight_a2a.to(torch.bfloat16)
        output = dequant_scale * torch.matmul(full_input_bf16, weight_a2a_bf16.t())
    else:
        output = torch.matmul(full_input, weight_a2a.t())

    if (is_s8 or is_fp8) and output.dtype != torch.bfloat16:
        output = output.to(torch.bfloat16)
    if quant_out:
        output, output_scale = quant_bf16_fp8(output, gsize=k_head_dim)
    else:
        output_scale = None

    return output, output_scale


@torch.no_grad()
def dist_triton_gemm_a2a(
    sp_group: torch.distributed.ProcessGroup,
    sp_op_ctx: UlyssesSpInferPreAttnContext,
    input: torch.Tensor,  # [local_seq_len, hidden]
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_nheads: int,
    kv_nheads: int,
    k_head_dim: int,
    v_head_dim: int,
    quant_out: bool,
):
    output, output_scale = ulysses_sp_infer_gemm_a2a_op(sp_op_ctx, input, weight, input_scale, weight_scale,
                                                        quant_out=quant_out)
    return output, output_scale


def make_data(local_seq_len, hidden_dim, out_features, input_dtype):
    is_fp8 = is_fp8_dtype(input_dtype)
    is_int8 = input_dtype == torch.int8
    assert is_fp8 or is_int8
    full_seq_len = local_seq_len * SP_GROUP.size()
    full_input_shape = [full_seq_len, hidden_dim]
    weight_shape = [out_features, hidden_dim]
    full_input_scale_shape = [full_seq_len, 1]
    weight_scale_shape = [1, out_features]

    if is_int8:
        full_input = torch.randint(-127, 127, full_input_shape, dtype=dtype, device="cuda")
        weight = torch.randint(-127, 127, weight_shape, dtype=dtype, device="cuda")
        full_input_scale = torch.rand(full_input_scale_shape, dtype=torch.float32, device="cuda") * 2 - 1
        weight_scale = torch.rand(weight_scale_shape, dtype=torch.float32, device="cuda") * 2 - 1
    elif is_fp8:
        # Generate FP8 tensors with controlled range
        full_input_f16 = (torch.rand(full_input_shape, dtype=torch.float16, device="cuda") * 2 - 1) * 0.01
        weight_f16 = (torch.rand(weight_shape, dtype=torch.float16, device="cuda") * 2 - 1) * 0.01
        full_input = full_input_f16.to(input_dtype)
        weight = weight_f16.to(input_dtype)
        full_input_scale = torch.rand(full_input_scale_shape, dtype=torch.float32, device="cuda") * 2 - 1
        weight_scale = torch.rand(weight_scale_shape, dtype=torch.float32, device="cuda") * 2 - 1

    input = full_input[RANK * local_seq_len:(RANK + 1) * local_seq_len].contiguous()
    input_scale = full_input_scale[RANK * local_seq_len:(RANK + 1) * local_seq_len].contiguous()
    return full_input, input, weight, full_input_scale, input_scale, weight_scale


def benchmark(SP_GROUP, args):
    max_seq_len = args.seq_len
    out_features = args.q_nheads * args.k_head_dim + args.kv_nheads * (args.k_head_dim + args.v_head_dim)
    step = args.step
    assert step % SP_GROUP.size() == 0
    input_dtype = DTYPE_MAP[args.dtype]
    op_ctx_overlap = UlyssesSpInferPreAttnContext.create(max_seq=args.seq_len, q_nheads=args.q_nheads,
                                                         kv_nheads=args.kv_nheads, k_head_dim=args.k_head_dim,
                                                         v_head_dim=args.v_head_dim, rank=RANK, world_size=WORLD_SIZE,
                                                         local_world_size=LOCAL_WORLD_SIZE, data_dtype=torch.bfloat16,
                                                         a2a_stream=a2a_stream)
    op_ctx_non_overlap = UlyssesSpInferPreAttnContext.create(max_seq=args.seq_len, q_nheads=args.q_nheads,
                                                             kv_nheads=args.kv_nheads, k_head_dim=args.k_head_dim,
                                                             v_head_dim=args.v_head_dim, rank=RANK,
                                                             world_size=WORLD_SIZE, local_world_size=LOCAL_WORLD_SIZE,
                                                             data_dtype=torch.bfloat16,
                                                             a2a_stream=torch.cuda.current_stream())

    perf_table = {}
    for cur_seq_len in range(step, max_seq_len + 1, step):
        cur_local_seq_len = cur_seq_len // 8
        full_input, input, weight, full_input_scale, input_scale, weight_scale = make_data(
            cur_local_seq_len, args.hidden_dim, out_features, input_dtype)
        torch_out, torch_perf = perf_func(
            partial(torch_gemm_a2a, SP_GROUP, full_input, weight, full_input_scale, weight_scale, args.q_nheads,
                    args.kv_nheads, args.k_head_dim, args.v_head_dim, args.quant_out), iters=50, warmup_iters=10)

        dist_triton_out, dist_triton_perf = perf_func(
            partial(dist_triton_gemm_a2a, SP_GROUP, op_ctx_overlap, input, weight, input_scale, weight_scale,
                    args.q_nheads, args.kv_nheads, args.k_head_dim, args.v_head_dim, args.quant_out), iters=50,
            warmup_iters=10)

        _, dist_triton_non_overlap_perf = perf_func(
            partial(dist_triton_gemm_a2a, SP_GROUP, op_ctx_non_overlap, input, weight, input_scale, weight_scale,
                    args.q_nheads, args.kv_nheads, args.k_head_dim, args.v_head_dim, args.quant_out), iters=50,
            warmup_iters=10)

        _verify_and_check_bitwise(torch_out, dist_triton_out, atol=0, rtol=0)
        if cur_seq_len not in perf_table.keys():
            perf_table[cur_seq_len] = {}
        perf_table[cur_seq_len]["overlap"] = dist_triton_perf
        perf_table[cur_seq_len]["non_overlap"] = dist_triton_non_overlap_perf

    if SP_GROUP.rank() == 0:
        config_info = f"hidden{args.hidden_dim}_qnheads{args.q_nheads}_kvnheads{args.kv_nheads}_khead{args.k_head_dim}_vhead{args.v_head_dim}"
        filename = f"perf_{config_info}.csv"

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['seq_len', 'overlap', 'non_overlap'])

            for k, v in perf_table.items():
                overlap_perf = v["overlap"]
                non_overlap_perf = v["non_overlap"]
                writer.writerow([k, overlap_perf, non_overlap_perf])
                print(
                    f"seq_len = {k}, hidden = {args.hidden_dim}, out_feat = {out_features}, overlap time = {overlap_perf}ms, non_overlap_perf={non_overlap_perf}ms"
                )

        print(f"save perf table to {os.path.abspath(filename)}")

    op_ctx_overlap.finalize()
    op_ctx_non_overlap.finalize()
    finalize_distributed()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_len", type=int)
    parser.add_argument("hidden_dim", type=int)
    parser.add_argument("q_nheads", type=int)
    parser.add_argument("k_head_dim", type=int)
    parser.add_argument("kv_nheads", type=int)
    parser.add_argument("v_head_dim", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--verify_iters", default=20, type=int, help="verify iterations")
    parser.add_argument("--dtype", default="s8", type=str, help="data type")
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true", help="stress test")
    parser.add_argument("--quant_out", default=False, action="store_true", help="apply quant for gemm out")
    parser.add_argument("--benchmark", default=False, action="store_true", help="stress test")
    parser.add_argument("--step", type=int, default=1024)

    return parser.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "fp8e4m3": torch.float8_e4m3fn,
    "fp8e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
    torch.int8: 0,
}

if __name__ == "__main__":
    args = parse_args()

    SP_GROUP = initialize_distributed(seed=0)
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    dtype = DTYPE_MAP[args.dtype]

    out_features = args.q_nheads * args.k_head_dim + args.kv_nheads * (args.k_head_dim + args.v_head_dim)
    a2a_stream = torch.cuda.Stream()

    assert args.seq_len % SP_GROUP.size() == 0
    max_local_seq_len = args.seq_len // SP_GROUP.size()

    if args.benchmark:
        benchmark(SP_GROUP, args)
        exit(0)

    ctx = (torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) if args.profile else nullcontext())

    sp_op_ctx = UlyssesSpInferPreAttnContext.create(max_seq=args.seq_len, q_nheads=args.q_nheads,
                                                    kv_nheads=args.kv_nheads, k_head_dim=args.k_head_dim,
                                                    v_head_dim=args.v_head_dim, rank=RANK, world_size=WORLD_SIZE,
                                                    local_world_size=LOCAL_WORLD_SIZE, data_dtype=torch.bfloat16,
                                                    a2a_stream=a2a_stream)

    if args.check:
        for n in range(args.iters):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            input_list = [
                make_data(random.randint(1, max_local_seq_len), args.hidden_dim, out_features, dtype)
                for _ in range(args.verify_iters)
            ]
            torch_outs, dist_triton_outs = [], []
            for full_input, input, weight, full_input_scale, input_scale, weight_scale in input_list:
                torch_out = torch_gemm_a2a(SP_GROUP, full_input, weight, full_input_scale, weight_scale, args.q_nheads,
                                           args.kv_nheads, args.k_head_dim, args.v_head_dim, args.quant_out)
                torch_outs.append(torch_out)

            for full_input, input, weight, full_input_scale, input_scale, weight_scale in input_list:
                dist_triton_out = dist_triton_gemm_a2a(SP_GROUP, sp_op_ctx, input, weight, input_scale, weight_scale,
                                                       args.q_nheads, args.kv_nheads, args.k_head_dim, args.v_head_dim,
                                                       args.quant_out)
                dist_triton_outs.append(dist_triton_out)
            for idx, ((torch_out, triton_out_scale),
                      (dist_triton_out, dist_triton_out_scale)) in enumerate(zip(torch_outs, dist_triton_outs)):
                if RANK == 0:
                    print(
                        f"output shape = {dist_triton_out.shape}, sum_torch_out = {torch_out.to(torch.float32).sum()}, sum_dist_triton_out = {dist_triton_out.to(torch.float32).sum()}"
                    )
                try:
                    torch.testing.assert_close(torch_out, dist_triton_out, atol=0, rtol=0)
                except Exception as e:
                    raise e

        print(f"RANK[{RANK}]: pass.")
        sp_op_ctx.finalize()
        finalize_distributed()
        exit(0)

    full_input, input, weight, full_input_scale, input_scale, weight_scale = make_data(
        max_local_seq_len, args.hidden_dim, out_features, dtype)

    with ctx:
        torch_out, _ = perf_func(
            partial(torch_gemm_a2a, SP_GROUP, full_input, weight, full_input_scale, weight_scale, args.q_nheads,
                    args.kv_nheads, args.k_head_dim, args.v_head_dim, args.quant_out), iters=50, warmup_iters=10)

        dist_triton_out, dist_triton_perf = perf_func(
            partial(dist_triton_gemm_a2a, SP_GROUP, sp_op_ctx, input, weight, input_scale, weight_scale, args.q_nheads,
                    args.kv_nheads, args.k_head_dim, args.v_head_dim, args.quant_out), iters=50, warmup_iters=10)

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{SP_GROUP.rank()}.json.gz")

    for i in range(LOCAL_WORLD_SIZE):
        if i == LOCAL_RANK:
            print(f"RANK[{RANK}]: dist_triton_perf = {dist_triton_perf} ms")
        torch.distributed.barrier()

    torch.cuda.synchronize()
    torch.distributed.barrier()

    atol = THRESHOLD_MAP[dtype]
    rtol = THRESHOLD_MAP[dtype]

    is_bitwise = _verify_and_check_bitwise(torch_out, dist_triton_out, atol=atol, rtol=rtol)
    if is_bitwise:
        print("✅  torch vs triton_dist bitwise match")
    else:
        print("❌  torch vs triton_dist not bitwise match")
    SP_GROUP.barrier()
    torch.cuda.synchronize()
    sp_op_ctx.finalize()
    finalize_distributed()
