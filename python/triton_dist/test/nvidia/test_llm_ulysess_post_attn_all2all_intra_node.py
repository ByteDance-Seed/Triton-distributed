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


def triton_dist_post_attn_a2a(
    input,
    seq_lens_cpu=None,
    num_comm_sm=-1,
):
    global gemm_a2a_op
    return gemm_a2a_op.post_attn_a2a(
        input,
        seq_lens_cpu=seq_lens_cpu,
        num_comm_sms=num_comm_sm,
    )


def triton_dist_post_attn_a2a_no_cpy(
    input,
    seq_lens_cpu=None,
    num_comm_sm=-1,
    comm_buf_idx=0,
):
    global gemm_a2a_op
    return gemm_a2a_op.post_attn_a2a_no_cpy(
        input,
        seq_lens_cpu=seq_lens_cpu,
        num_comm_sms=num_comm_sm,
        comm_buf_idx=comm_buf_idx,
    )


print = partial(print, flush=True)


class PerfResult:

    def __init__(
        self,
        name: str,
        a2a_output: torch.Tensor,
        total_ms: float,
    ) -> None:
        self.name = name
        self.a2a_output = a2a_output
        self.total_ms = total_ms

    def __repr__(self) -> str:
        return f"{self.name}: total {self.total_ms:.3f} ms"


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
            if sp_group.rank() == 0:
                print(f"sp_group id = {RANK // sp_group.size()}, seq_lens_list = {seq_lens_list}")
        if not args.a2a_only:
            input_shape = [args.bs, local_nh, total_seq_len, args.hd]
        else:
            input_shape = [args.bs, total_seq_len, local_nh, args.hd]

        input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) * (sp_group.rank() + 1)

        return (input, seq_lens_cpu)

    def _torch_impl(input, seq_lens_cpu):
        a2a_output = torch_post_attn_all_to_all_transpose(sp_group, input, args.a2a_only, args.dp,
                                                          seq_lens_cpu=seq_lens_cpu)
        return a2a_output

    def _triton_dist_impl(input, seq_lens_cpu):
        if not args.local_copy:
            output = triton_dist_post_attn_a2a(
                input,
                seq_lens_cpu=seq_lens_cpu,
                num_comm_sm=args.num_comm_sm,
            )
        else:
            output = triton_dist_post_attn_a2a_no_cpy(
                input,
                seq_lens_cpu=seq_lens_cpu,
                num_comm_sm=args.num_comm_sm,
                comm_buf_idx=0,
            )
            triton_dist_sp_barrier_all()
            new_output = torch.empty(output.shape, dtype=output.dtype, device=output.device)
            new_output.copy_(output)
            return new_output
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

            print(f"rank {RANK} ✅ triton_dist check passed")
        torch.distributed.barrier()

    TP_GROUP.barrier()
    torch.cuda.synchronize()


@torch.no_grad()
def perf_torch(
    sp_group: torch.distributed.ProcessGroup,
    input: torch.Tensor,
    warmup: int,
    iters: int,
    a2a_only: bool = False,
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
        a2a_output = torch_post_attn_all_to_all_transpose(sp_group, input, a2a_only, is_dp, seq_lens_cpu=seq_lens_cpu)
        all2all_end_events[i].record()
        end_events[i].record()

    comm_times = []
    for i in range(total_iters):
        all2all_end_events[i].synchronize()
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times) / iters * 1000

    return PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        a2a_output=a2a_output,
        total_ms=comm_time,
    )


@torch.no_grad()
def perf_triton_dist(
    sp_group: torch.distributed.ProcessGroup,
    input: torch.Tensor,
    warmup: int = 5,
    iters: int = 10,
    num_comm_sm: int = -1,
    fuse_sync: bool = False,
    a2a_only: bool = False,
    is_dp: bool = False,
    local_copy: bool = False,
    seq_lens_cpu: Optional[torch.Tensor] = None,
):
    if not is_dp:
        assert seq_lens_cpu is None

    assert a2a_only

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()

    for i in range(total_iters):
        start_events[i].record()
        if not local_copy:
            a2a_output = triton_dist_post_attn_a2a(
                input,
                seq_lens_cpu=seq_lens_cpu,
                num_comm_sm=num_comm_sm,
            )
        else:
            comm_buf = triton_dist_post_attn_a2a_no_cpy(
                input,
                seq_lens_cpu=seq_lens_cpu,
                num_comm_sm=num_comm_sm,
                comm_buf_idx=0,
            )
            triton_dist_sp_barrier_all()
            a2a_output = torch.empty(comm_buf.shape, dtype=comm_buf.dtype, device=comm_buf.device)
            a2a_output.copy_(comm_buf)
        end_events[i].record()

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    comm_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times)

    comm_time_ms = comm_time / iters * 1000

    return PerfResult(
        name=f"triton_dist  #{TP_GROUP.rank()}",
        a2a_output=a2a_output,
        total_ms=comm_time_ms,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bs", type=int)
    parser.add_argument("seq_len", type=int)
    parser.add_argument("nh", type=int)
    parser.add_argument("hd", type=int)
    parser.add_argument("--num_comm_sm", type=int, required=True, help="num sm for a2a")
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--fuse_sync", default=False, action="store_true", help="fuse sync into all2all kernel")
    parser.add_argument(
        "--verify",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="run once to verify correctness",
    )
    parser.add_argument("--a2a_only", default=False, action="store_true", help="whether have transpose")
    parser.add_argument("--dp", default=False, action="store_true", help="dp per rank")
    parser.add_argument("--sp_size", default=0, type=int, help="sp size")
    parser.add_argument(
        "--local-copy",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If local-copy is true, the user needs to copy output from comm buffer to user buffer",
    )
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
    # input: [bs, local_nh, seq_len, hd] for a2a_transpose, [bs, seq_len, local_nh, hd] for a2a_only
    if args.dp and not args.a2a_only:
        raise NotImplementedError("dp mode only support for a2a only")

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
    if not args.a2a_only:
        input_shape = [args.bs, local_nh, total_seq_len, args.hd]
    else:
        input_shape = [args.bs, total_seq_len, local_nh, args.hd]

    input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) * (sp_group.rank() + 1)

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
            args.warmup,
            args.iters,
            args.a2a_only,
            args.dp,
            seq_lens_cpu=seq_lens_cpu,
        )
        perf_res_triton_dist = perf_triton_dist(
            sp_group,
            input,
            args.warmup,
            args.iters,
            args.num_comm_sm,
            args.fuse_sync,
            args.a2a_only,
            args.dp,
            args.local_copy,
            seq_lens_cpu=seq_lens_cpu,
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

    torch_output = perf_res_torch.a2a_output
    triton_dist_output = perf_res_triton_dist.a2a_output.reshape(torch_output.shape)
    torch.distributed.barrier()
    if bitwise_equal(torch_output, triton_dist_output):
        print("✅  torch vs triton_dist bitwise match")
    else:
        print("❌  torch vs triton_dist not bitwise match")

    atol, rtol = 0.0, 0.0
    try:
        torch.allclose(triton_dist_output, torch_output, atol=atol, rtol=rtol)
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
