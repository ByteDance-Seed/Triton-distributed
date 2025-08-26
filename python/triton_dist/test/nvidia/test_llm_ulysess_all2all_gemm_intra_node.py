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
import time
from contextlib import nullcontext
from functools import partial
from typing import Optional
import random

import numpy as np
import torch
import torch.distributed

from triton_dist.utils import assert_allclose, bitwise_equal, initialize_distributed, nvshmem_barrier_all_on_stream, finalize_distributed

from triton_dist.kernels.nvidia.sp_ulysess_o_all2all_gemm import SpUlysessOAll2AllGemmKernel

gemm_a2a_op = None


def triton_dist_init(world_group: torch.distributed.ProcessGroup, nnodes: int, sp_size: int, max_batch_size: int,
                     num_head: int, max_seq_len: int, head_dim: int, input_dtype=torch.bfloat16,
                     output_dtype=torch.bfloat16, max_num_comm_buf: int = 1, fuse_sync: bool = True):
    global gemm_a2a_op
    if gemm_a2a_op is None:
        gemm_a2a_op = SpUlysessOAll2AllGemmKernel(
            world_group,
            nnodes,
            sp_size,
            max_batch_size,
            num_head,
            max_seq_len,
            head_dim,
            max_num_comm_buf,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            a2a_only=True,
            fuse_sync=fuse_sync,
        )
    nvshmem_barrier_all_on_stream()


def triton_dist_sp_barrier_all():
    global gemm_a2a_op
    gemm_a2a_op.sp_group_barrier_all_intra_node()


def triton_dist_post_attn_a2a_gemm(
    attention_outputs,
    weight,
    seq_lens_cpu=None,
    bias=None,
    outputs=None,
    a2a_outputs=None,
    num_comm_sms=16,
    sm_margin=0,
):
    global gemm_a2a_op
    outputs = gemm_a2a_op.forward(
        attention_outputs,
        weight,
        seq_lens_cpu=seq_lens_cpu,
        bias=bias,
        output=outputs,
        a2a_output=a2a_outputs,
        transpose_weight=False,
        num_comm_sms=num_comm_sms,
        sm_margin=sm_margin,
    )

    return outputs


print = partial(print, flush=True)


class PerfResult:

    def __init__(
        self,
        name: str,
        output: torch.Tensor,
        a2a_output: torch.Tensor,
        total_ms: float,
        time1: str,
        gemm_time_ms: float,
        time2: str,
        comm_time_ms: float,
        time3: str = "gemm_only",
        gemm_only_time_ms: float = 0,
    ) -> None:
        self.name = name
        self.output = output
        self.a2a_output = a2a_output
        self.total_ms = total_ms
        self.time1 = time1
        self.time2 = time2
        self.gemm_time_ms = gemm_time_ms
        self.comm_time_ms = comm_time_ms
        self.time3 = time3
        self.gemm_only_time_ms = gemm_only_time_ms

    def __repr__(self) -> str:
        if self.gemm_only_time_ms == 0.0:
            return (f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms"
                    f", {self.time2} {self.comm_time_ms:.3f} ms")
        else:
            return (f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms"
                    f", {self.time2} {self.comm_time_ms:.3f} ms, {self.time3} {self.gemm_only_time_ms:.3f} ms")


def torch_post_attn_all_to_all_transpose(sp_group, input, a2a_only, is_dp, seq_lens_cpu=None):
    if not a2a_only:
        bs, local_nh, seq_len, hd = input.shape
    else:
        bs, seq_len, local_nh, hd = input.shape
    local_seq_len = seq_len // sp_group.size()
    hidden_dim = local_nh * hd * sp_group.size()

    if is_dp:
        local_seq_len = seq_lens_cpu[sp_group.rank()].item()

    # All to all input tensors from all gpus
    input_after_a2a = torch.zeros(
        (local_seq_len * sp_group.size(), bs, local_nh, hd),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    if is_dp:
        input_before_a2a = input.permute(1, 0, 2, 3).contiguous()  # [seq_len, bs, local_nh, hd]
        output_splits = [local_seq_len for i in range(sp_group.size())]
        input_splits = seq_lens_cpu.tolist()
        torch.distributed.all_to_all_single(input_after_a2a, input_before_a2a, output_splits, input_splits,
                                            group=sp_group)
        gemm_input = (input_after_a2a.reshape(sp_group.size(), local_seq_len, bs, local_nh,
                                              hd).permute(2, 1, 0, 3, 4).reshape(bs, local_seq_len, hidden_dim))
    else:
        if not a2a_only:
            input_before_a2a = input.permute(2, 0, 1, 3).contiguous()
            torch.distributed.all_to_all_single(input_after_a2a, input_before_a2a, group=sp_group)
            gemm_input = (input_after_a2a.reshape(sp_group.size(), local_seq_len, bs, local_nh,
                                                  hd).permute(2, 1, 0, 3, 4).reshape(bs, local_seq_len, hidden_dim))
        else:
            input_before_a2a = input.permute(1, 0, 2, 3).contiguous()  # [seq_len, bs, local_nh, hd]
            torch.distributed.all_to_all_single(input_after_a2a, input_before_a2a, group=sp_group)
            gemm_input = (input_after_a2a.reshape(sp_group.size(), local_seq_len, bs, local_nh,
                                                  hd).permute(2, 1, 0, 3, 4).reshape(bs, local_seq_len, hidden_dim))
    return gemm_input


def check_correctness(sp_group, args):
    random.seed(42 + RANK // sp_group.size())

    num_iteration = args.iters

    max_local_seq_len = args.seq_len // sp_group.size()
    hidden_dim = args.nh * args.hd
    out_features = args.out_features
    dtype = DTYPE_MAP[args.dtype]

    def _gen_inputs(max_local_seq_len):
        if not args.dp:
            seq_lens_cpu = None
            total_seq_len = max_local_seq_len * sp_group.size()
        else:
            seq_lens_list = list(
                np.random.randint(max(1, max_local_seq_len - 32), max_local_seq_len, size=(sp_group.size(), )))
            seq_lens_gpu = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")
            torch.distributed.broadcast(seq_lens_gpu, src=0, group=sp_group)
            seq_lens_cpu = seq_lens_gpu.cpu()
            seq_lens_list = seq_lens_cpu.tolist()
            total_seq_len = sum(seq_lens_list)
            local_seq_len = seq_lens_list[sp_group.rank()]
            if sp_group.rank() == 0:
                print(f"sp_group id = {RANK // sp_group.size()}, seq_lens_list = {seq_lens_list}")
        input_shape = [args.bs, total_seq_len, local_nh, args.hd]

        weight_shape = [out_features, hidden_dim]

        input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) * (sp_group.rank() + 1)
        weight = (-2 * torch.rand(weight_shape, dtype=dtype).cuda() + 1) * (sp_group.rank() + 1)

        bias = None
        local_seq_len = seq_lens_list[sp_group.rank()] if args.dp else max_local_seq_len
        gemm_m = args.bs * local_seq_len
        bias_shape = [gemm_m, args.out_features]
        if args.has_bias:
            bias = torch.rand(bias_shape, dtype=dtype).cuda() / 10 * (sp_group.rank() + 1)
        return (input, weight, bias, seq_lens_cpu)

    def _torch_impl(input, weight, bias, seq_lens_cpu):
        gemm_input = torch_post_attn_all_to_all_transpose(sp_group, input, True, args.dp, seq_lens_cpu=seq_lens_cpu)
        output = torch.matmul(gemm_input, weight.t())

        if bias is not None:
            bias = bias.reshape(output.shape)
            output += bias
        if args.copy_a2a_output:
            return output, gemm_input
        else:
            return output

    def _triton_dist_impl(input, weight, bias, seq_lens_cpu):
        bs, seq_len, local_nh, hd = input.shape
        local_seq_len = seq_len // sp_group.size()
        hidden_dim = local_nh * hd * sp_group.size()

        if args.dp:
            local_seq_len = seq_lens_cpu[sp_group.rank()].item()

        a2a_transpose_output = (torch.empty(
            (bs, local_seq_len, hidden_dim), dtype=input.dtype, device=input.device) if args.copy_a2a_output else None)
        output = triton_dist_post_attn_a2a_gemm(
            input,
            weight,
            seq_lens_cpu,
            bias=bias,
            outputs=None,
            a2a_outputs=a2a_transpose_output,
            num_comm_sms=args.num_comm_sm,
            sm_margin=args.sm_margin,
        )
        if args.copy_a2a_output:
            return output, a2a_transpose_output
        else:
            return output

    all_inputs = [_gen_inputs(random.randint(1, max_local_seq_len)) for _ in range(num_iteration)]
    torch_outputs = [_torch_impl(*inputs) for inputs in all_inputs]

    torch.distributed.barrier()
    torch.cuda.synchronize()

    triton_dist_outputs = [_triton_dist_impl(*inputs) for inputs in all_inputs]

    torch.distributed.barrier()

    torch.cuda.synchronize()

    for i in range(WORLD_SIZE):
        if i == RANK:
            for triton_dist_output, torch_output in zip(triton_dist_outputs, torch_outputs):
                if not isinstance(triton_dist_output, (list, tuple)):
                    triton_dist_output = [triton_dist_output]
                if not isinstance(torch_output, (list, tuple)):
                    torch_output = [torch_output]
                for triton_dist_tensor, torch_tensor in zip(triton_dist_output, torch_output):
                    triton_dist_tensor = triton_dist_tensor.reshape(torch_tensor.shape)
                    if not bitwise_equal(triton_dist_tensor, torch_tensor):
                        print("Warning: torch vs triton_dist not bitwise match")

                    atol = THRESHOLD_MAP[dtype]
                    rtol = THRESHOLD_MAP[dtype]
                    assert_allclose(triton_dist_tensor, torch_tensor, atol=atol, rtol=rtol)

            print("✅ triton_dist check passed")
        torch.distributed.barrier()

    TP_GROUP.barrier()
    torch.cuda.synchronize()


@torch.no_grad()
def perf_torch(
    sp_group: torch.distributed.ProcessGroup,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    warmup: int,
    iters: int,
    is_dp: bool = False,
    seq_lens_cpu: Optional[torch.Tensor] = None,
):
    torch.distributed.barrier()

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    all2all_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        gemm_input = torch_post_attn_all_to_all_transpose(sp_group, input, True, is_dp, seq_lens_cpu=seq_lens_cpu)
        all2all_end_events[i].record()

        output = torch.matmul(gemm_input, weight.t())

        if bias is not None:
            bias = bias.reshape(output.shape)
            output += bias
        end_events[i].record()

    comm_times = []  # all to all
    gemm_times = []  # gemm
    for i in range(total_iters):
        all2all_end_events[i].synchronize()
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(all2all_end_events[i]) / 1000)
            gemm_times.append(all2all_end_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times) / iters * 1000
    gemm_time = sum(gemm_times) / iters * 1000

    return PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        output=output,
        a2a_output=gemm_input,
        total_ms=gemm_time + comm_time,
        time1="gemm",
        gemm_time_ms=gemm_time,
        time2="comm",
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_triton_dist(
    sp_group: torch.distributed.ProcessGroup,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    transpose_weight: bool = True,
    save_gemm_input: bool = False,
    warmup: int = 5,
    iters: int = 10,
    num_comm_sm: int = -1,
    sm_margin: int = 0,
    fuse_sync: bool = False,
    fast_acc: bool = False,
    is_dp: bool = False,
    seq_lens_cpu: Optional[torch.Tensor] = None,
    copy_a2a_output: bool = False,
):
    if not is_dp:
        assert seq_lens_cpu is None

    bs, seq_len, local_nh, hd = input.shape

    local_seq_len = seq_len // sp_group.size()
    hidden_dim = local_nh * hd * sp_group.size()
    if is_dp:
        local_seq_len = seq_lens_cpu[sp_group.rank()].item()

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    gemm_input = torch_post_attn_all_to_all_transpose(sp_group, input, True, is_dp, seq_lens_cpu=seq_lens_cpu)

    gemm_output = torch.matmul(gemm_input, weight.T)

    time.sleep(1)
    torch.distributed.barrier()

    for i in range(total_iters):
        start_events[i].record()
        a2a_transpose_output = (torch.empty(
            (bs, local_seq_len, hidden_dim), dtype=input.dtype, device=input.device) if copy_a2a_output else None)
        a2a_gemm_output = triton_dist_post_attn_a2a_gemm(
            input,
            weight,
            seq_lens_cpu,
            bias=bias,
            outputs=None,
            a2a_outputs=a2a_transpose_output,
            num_comm_sms=args.num_comm_sm,
            sm_margin=args.sm_margin,
        )
        end_events[i].record()

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    a2a_gemm_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            a2a_gemm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    a2a_gemm_time = sum(a2a_gemm_times)

    a2a_gemm_time_ms = a2a_gemm_time / iters * 1000
    gemm_time_ms = a2a_gemm_time / iters * 1000
    comm_time_ms = 0

    is_bitwise_match = bitwise_equal(gemm_output, a2a_gemm_output[0])
    if sp_group.rank() == 0:
        print("is bitwise match: ", is_bitwise_match)

    return PerfResult(
        name=f"triton_dist  #{TP_GROUP.rank()}",
        output=a2a_gemm_output[0],
        a2a_output=a2a_transpose_output,
        total_ms=a2a_gemm_time_ms,
        time1="gemm",
        gemm_time_ms=gemm_time_ms,
        time2="comm",
        comm_time_ms=comm_time_ms,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bs", type=int)
    parser.add_argument("nh", type=int)
    parser.add_argument("seq_len", type=int)
    parser.add_argument("hd", type=int)
    parser.add_argument("out_features", type=int)
    parser.add_argument("--num_comm_sm", type=int, required=True, help="num sm for a2a")
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--sm_margin", default=0, type=int, help="sm margin")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument("--copy_a2a_output", default=False, action="store_true", help="whether to copy a2a output")
    parser.add_argument(
        "--fastacc",
        default=False,
        action="store_true",
        help="whether to use fast accumulation (FP8 Gemm only)",
    )
    parser.add_argument(
        "--transpose_weight",
        dest="transpose_weight",
        action=argparse.BooleanOptionalAction,
        help="transpose weight",
        default=False,
    )
    parser.add_argument("--fuse_sync", default=False, action="store_true", help="fuse sync into all2all kernel")
    parser.add_argument(
        "--verify",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="run once to verify correctness",
    )
    parser.add_argument(
        "--save_gemm_input",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="save gemm input",
    )
    parser.add_argument("--dp", default=False, action="store_true", help="dp per rank")
    parser.add_argument("--sp_size", default=0, type=int, help="sp size")
    parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
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
}

if __name__ == "__main__":
    args = parse_args()

    TP_GROUP = initialize_distributed()
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    dtype = DTYPE_MAP[args.dtype]

    triton_dist_init(
        world_group=TP_GROUP,
        nnodes=WORLD_SIZE // LOCAL_WORLD_SIZE,
        sp_size=args.sp_size,
        max_batch_size=args.bs,
        num_head=args.nh,
        max_seq_len=args.seq_len,
        head_dim=args.hd,
        input_dtype=dtype,
        output_dtype=dtype,
        max_num_comm_buf=3,
        fuse_sync=args.fuse_sync,
    )

    if dtype not in [torch.bfloat16]:
        raise NotImplementedError("A2A Gemm only support BF16.")

    if args.transpose_weight:
        raise NotImplementedError("A2A Gemm only support RCR layout.")

    # init sp process group
    assert args.sp_size > 0 and LOCAL_WORLD_SIZE % args.sp_size == 0
    num_sp_group = WORLD_SIZE // args.sp_size
    all_sp_subgroups = []
    sp_group = None
    for i in range(num_sp_group):
        cur_group_ranks = [i * args.sp_size + j for j in range(args.sp_size)]
        all_sp_subgroups.append(torch.distributed.new_group(cur_group_ranks))
        if i == RANK // args.sp_size:
            sp_group = all_sp_subgroups[-1]
    assert sp_group is not None

    assert args.nh % sp_group.size() == 0
    assert args.seq_len % sp_group.size() == 0

    local_nh = args.nh // sp_group.size()
    local_seq_len = args.seq_len // sp_group.size()
    hidden_dim = args.nh * args.hd

    if not args.dp:
        seq_lens_cpu = None
        total_seq_len = args.seq_len
    else:
        seq_lens_list = list(np.random.randint(max(1, local_seq_len - 32), local_seq_len, size=(sp_group.size(), )))
        seq_lens_gpu = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")
        torch.distributed.broadcast(seq_lens_gpu, src=0, group=sp_group)
        seq_lens_cpu = seq_lens_gpu.cpu()
        seq_lens_list = seq_lens_cpu.tolist()
        total_seq_len = sum(seq_lens_list)
        local_seq_len = seq_lens_list[sp_group.rank()]
        if sp_group.rank() == 0:
            print(f"sp_group id = {RANK // sp_group.size()}, seq_lens_list = {seq_lens_list}")
    input_shape = [args.bs, total_seq_len, local_nh, args.hd]

    # weight: [out_features, hidden_dim]
    weight_shape = [args.out_features, hidden_dim]

    input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) * (sp_group.rank() + 1)
    weight = (-2 * torch.rand(weight_shape, dtype=dtype).cuda() + 1) * (sp_group.rank() + 1)

    input_scale = None
    weight_scale = None

    bias = None
    gemm_m = args.bs * local_seq_len
    bias_shape = [gemm_m, args.out_features]
    if args.has_bias:
        bias = torch.rand(bias_shape, dtype=dtype).cuda() / 10 * (sp_group.rank() + 1)

    if args.debug:
        input.zero_()
        input[:, 0].fill_(sp_group.rank() + 1)
        weight.fill_(1)
        if input_scale is not None:
            input_scale.fill_(1)
            weight_scale.fill_(1)
        if bias is not None:
            bias.zero_()
    torch.distributed.barrier()

    ctx = (torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) if args.profile else nullcontext())

    if args.verify:
        check_correctness(sp_group, args)
        gemm_a2a_op.finalize()
        del gemm_a2a_op
        finalize_distributed()
        exit(0)

    with ctx:
        perf_res_torch = perf_torch(
            sp_group,
            input,
            weight,
            bias,
            input_scale,
            weight_scale,
            args.warmup,
            args.iters,
            args.dp,
            seq_lens_cpu=seq_lens_cpu,
        )
        perf_res_triton_dist = perf_triton_dist(
            sp_group,
            input,
            weight,
            bias,
            input_scale,
            weight_scale,
            args.transpose_weight,
            args.save_gemm_input,
            args.warmup,
            args.iters,
            args.num_comm_sm,
            args.sm_margin,
            args.fuse_sync,
            args.fastacc,
            args.dp,
            seq_lens_cpu=seq_lens_cpu,
            copy_a2a_output=args.copy_a2a_output,
        )

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            print(perf_res_torch)
            print(perf_res_triton_dist)
        torch.distributed.barrier()

    torch_output = perf_res_torch.output
    triton_dist_output = perf_res_triton_dist.output.reshape(torch_output.shape)
    torch_gathered_data = perf_res_torch.a2a_output
    triton_dist_gathered_data = perf_res_triton_dist.a2a_output
    torch.distributed.barrier()
    if bitwise_equal(torch_output, triton_dist_output):
        print("✅  torch vs triton_dist bitwise match")
    else:
        print("❌  torch vs triton_dist not bitwise match")

    if args.save_gemm_input and TP_GROUP.rank() == 0:
        try:
            assert_allclose(triton_dist_gathered_data, torch_gathered_data, atol=1e-9, rtol=1e-9)
        except Exception as e:
            torch.save(triton_dist_gathered_data, f"triton_dist_gathered_data_{TP_GROUP.rank()}.pt")
            torch.save(torch_gathered_data, f"torch_gathered_data_{TP_GROUP.rank()}.pt")
            print("❌ triton_dist gathered data check failed")
            raise e
        else:
            print("✅ triton_dist gathered data check passed")

    atol = THRESHOLD_MAP[dtype]
    rtol = THRESHOLD_MAP[dtype]
    try:
        assert_allclose(triton_dist_output, torch_output, atol=atol, rtol=rtol)
    except Exception as e:
        torch.save(triton_dist_output, f"triton_dist_{TP_GROUP.rank()}.pt")
        torch.save(torch_output, f"torch_{TP_GROUP.rank()}.pt")
        print("❌ triton_dist check failed")
        raise e
    else:
        print("✅ triton_dist check passed")

    TP_GROUP.barrier()
    torch.cuda.synchronize()

    gemm_a2a_op.finalize()
    del gemm_a2a_op
    finalize_distributed()
