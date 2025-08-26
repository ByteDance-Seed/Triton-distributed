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
from typing import List, Optional

import random
import numpy as np
import torch
import torch.distributed

from triton_dist.utils import assert_allclose, bitwise_equal, initialize_distributed, nvshmem_barrier_all_on_stream, finalize_distributed

from triton_dist.kernels.nvidia.sp_ulysess_qkv_gemm_all2all import SpUlysessQKVGemmAll2AllKernel

print = partial(print, flush=True)

gemm_a2a_op = None


def triton_dist_init(world_group: torch.distributed.ProcessGroup, nnodes: int, sp_size: int, max_batch_size: int,
                     max_seq_len: int, hidden_size: int, head_dim: int,
                     qkv_out_features: int,  # qkv_out_features can be different from 3 * hidden_size
                     input_dtype=torch.bfloat16, output_dtype=torch.bfloat16, gqa: int = 1, max_num_comm_buf: int = 1):
    global gemm_a2a_op
    if gemm_a2a_op is None:
        gemm_a2a_op = SpUlysessQKVGemmAll2AllKernel(
            world_group,
            nnodes,
            sp_size,
            max_batch_size,
            max_seq_len,
            hidden_size,
            head_dim,
            qkv_out_features,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            gqa=gqa,
            max_num_comm_buf=max_num_comm_buf,
        )
    nvshmem_barrier_all_on_stream()


def triton_dist_sp_barrier_all():
    global gemm_a2a_op
    gemm_a2a_op.sp_group_barrier_all_intra_node()


def triton_dist_pre_attn_a2a(
    inputs,
    seq_lens_cpu=None,
    num_comm_sm=-1,
):
    return gemm_a2a_op.pre_attn_a2a(
        inputs,
        seq_lens_cpu=seq_lens_cpu,
        num_comm_sms=num_comm_sm,
        is_input_in_comm_buf=False,
        comm_buf_idx=0,
    )


def triton_dist_pre_attn_a2a_no_cpy(
    inputs,
    seq_lens_cpu=None,
    num_comm_sm=-1,
    comm_buf_idx=0,
):
    global gemm_a2a_op
    return gemm_a2a_op.pre_attn_a2a_no_cpy(
        inputs,
        seq_lens_cpu=seq_lens_cpu,
        num_comm_sms=num_comm_sm,
        comm_buf_idx=comm_buf_idx,
    )


def triton_dist_pre_attn_qkv_pack_a2a(
    inputs,
    seq_lens_cpu=None,
    num_comm_sm=-1,
    comm_buf_idx=0,
):
    global gemm_a2a_op
    return gemm_a2a_op.qkv_pack_a2a(
        inputs,
        seq_lens_cpu=seq_lens_cpu,
        num_comm_sms=num_comm_sm,
        is_input_in_comm_buf=False,
        comm_buf_idx=comm_buf_idx,
    )


def triton_dist_pre_attn_qkv_pack_a2a_no_cpy(
    inputs,
    seq_lens_cpu=None,
    num_comm_sm=-1,
    comm_buf_idx=0,
):
    global gemm_a2a_op
    return gemm_a2a_op.qkv_pack_a2a_no_cpy(
        inputs,
        seq_lens_cpu=seq_lens_cpu,
        num_comm_sms=num_comm_sm,
        comm_buf_idx=comm_buf_idx,
    )


def triton_dist_get_input_comm_buf(
    tensor,
    comm_buf_idx=0,
):
    global gemm_a2a_op
    return gemm_a2a_op.get_input_comm_buf(tensor, comm_buf_idx)


def triton_dist_sp_group_barrier_all(stream=None):
    global gemm_a2a_op
    gemm_a2a_op.sp_group_barrier_all_intra_node(stream)


class PerfResult:

    def __init__(
        self,
        name: str,
        outputs: List[torch.Tensor],
        total_ms: float,
    ) -> None:
        self.name = name
        self.outputs = outputs
        self.total_ms = total_ms

    def __repr__(self) -> str:
        return f"{self.name}: total {self.total_ms:.3f} ms"


def _verify_and_check_bitwise(torch_outs: List[torch.Tensor], triton_dist_outs: List[torch.Tensor], atol, rtol):
    is_bitwise = True
    for ref_out, triton_dist_out in zip(torch_outs, triton_dist_outs):
        triton_dist_out = triton_dist_out.reshape(ref_out.shape)
        assert_allclose(ref_out, triton_dist_out, atol=atol, rtol=rtol)
        if not bitwise_equal(ref_out, triton_dist_out):
            is_bitwise = False
    return is_bitwise


def torch_pre_attn_qkv_pack_a2a(sp_group, input, bs, seq_len, nh, head_dim, gqa, seq_lens_cpu=None):
    world_size = sp_group.size()
    rank = sp_group.rank()
    local_seq_len = (seq_len // sp_group.size() if seq_lens_cpu is None else seq_lens_cpu[rank].item())
    local_nh = nh // sp_group.size()
    input = input.reshape(bs, local_seq_len, nh, head_dim)
    local_q_nh = local_nh // (gqa + 2) * gqa
    local_k_nh = local_nh // (gqa + 2)

    q_input = input[:, :, :local_q_nh * world_size, :].contiguous()
    k_input = input[:, :, local_q_nh * world_size:(local_q_nh + local_k_nh) * world_size, :].contiguous()
    v_input = input[:, :, (local_q_nh + local_k_nh) * world_size:, :].contiguous()

    def _a2a(a2a_tensor):
        if seq_lens_cpu is None:
            a2a_input = a2a_tensor.permute(2, 1, 0, 3).contiguous()  # [nh, local_seq_len, bs, hd]
            a2a_nh, a2a_local_seq_len, a2a_bs, a2a_hd = a2a_input.shape
            a2a_buffer = torch.empty(
                (world_size, a2a_nh // world_size, a2a_local_seq_len, a2a_bs, a2a_hd),
                dtype=a2a_input.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            torch.distributed.all_to_all_single(a2a_buffer, a2a_input, group=sp_group)
            a2a_buffer = (a2a_buffer.permute(3, 0, 2, 1, 4).reshape(a2a_bs, a2a_local_seq_len * world_size,
                                                                    a2a_nh // world_size, a2a_hd).contiguous())
            return a2a_buffer
        else:
            a2a_nh = a2a_tensor.shape[2]
            a2a_local_nh = a2a_nh // world_size
            a2a_input = (a2a_tensor.permute(2, 1, 0, 3).reshape(-1, bs,
                                                                head_dim).contiguous())  # [nh * local_seq_len , bs, hd]
            _, a2a_bs, a2a_hd = a2a_input.shape
            sum_seq_len = seq_lens_cpu.sum().item()
            a2a_buffer = torch.empty(
                (a2a_local_nh * sum_seq_len, a2a_bs, a2a_hd),
                dtype=a2a_input.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            output_splits = [val * a2a_local_nh for val in seq_lens_cpu.tolist()]
            input_splits = [output_splits[rank] for i in range(world_size)]
            torch.distributed.all_to_all_single(a2a_buffer, a2a_input, output_splits, input_splits, group=sp_group)
            tensor_list = []
            start = 0
            for i in range(world_size):
                cur_slice = a2a_buffer[start:start + output_splits[i], :, :].reshape(a2a_local_nh, -1, a2a_bs, a2a_hd)
                start += output_splits[i]
                tensor_list.append(cur_slice)
            a2a_buffer = torch.cat(tensor_list, dim=1)  # [a2a_local_nh, sum_seq_len, a2a_bs, a2a_hd]
            a2a_buffer = a2a_buffer.permute(2, 1, 0, 3).contiguous()
            return a2a_buffer

    q = _a2a(q_input)
    k = _a2a(k_input)
    v = _a2a(v_input)
    return [q, k, v]


def check_correctness(sp_group, args):
    random.seed(42 + RANK // sp_group.size())

    num_iteration = args.iters
    bs, max_seq_len, nh, head_dim = args.bs, args.seq_len, args.nheads, args.head_dim
    max_local_seq_len = max_seq_len // sp_group.size()

    dtype = DTYPE_MAP[args.dtype]

    gqa = args.gqa

    def _gen_inputs(max_local_seq_len, iter=0, is_debug=False):
        if not args.dp:
            seq_lens_cpu = None
            local_seq_len = max_local_seq_len
        else:
            seq_lens_list = list(
                np.random.randint(max(1, max_local_seq_len - 32), max_local_seq_len, size=(sp_group.size(), )))
            local_seq_len = seq_lens_list[sp_group.rank()]
            seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int32, device="cpu")
            if sp_group.rank() == 0:
                print(f"max_local_seq_len = {max_local_seq_len}, seq_lens_cpu = {seq_lens_cpu}")

        input_shape = [bs, local_seq_len, args.nheads, args.head_dim]
        input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) / 10 * (sp_group.rank() + 1)
        return (input, seq_lens_cpu)

    def _torch_impl(input, seq_lens_cpu):
        seq_len = input.size(1) * sp_group.size() if seq_lens_cpu is None else seq_lens_cpu.sum()
        outputs = torch_pre_attn_qkv_pack_a2a(sp_group, input, bs, seq_len, nh, head_dim, gqa, seq_lens_cpu)
        return outputs

    def _triton_dist_impl(input, seq_lens_cpu):
        if not args.local_copy:
            if args.apply_pack:
                outputs = triton_dist_pre_attn_qkv_pack_a2a(
                    input,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                )
            else:
                q_nh = nh // (gqa + 2) * gqa
                k_nh = nh // (gqa + 2)
                q = input[:, :, :q_nh].contiguous()
                k = input[:, :, q_nh:q_nh + k_nh].contiguous()
                v = input[:, :, q_nh + k_nh:].contiguous()
                out_q = triton_dist_pre_attn_a2a(
                    q,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                )
                out_k = triton_dist_pre_attn_a2a(
                    k,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                )
                out_v = triton_dist_pre_attn_a2a(
                    v,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                )
                outputs = [out_q, out_k, out_v]
        else:
            if args.apply_pack:
                comm_buf = triton_dist_get_input_comm_buf(input, 0)
                comm_buf.copy_(input)
                triton_dist_sp_group_barrier_all()
                outputs = triton_dist_pre_attn_qkv_pack_a2a_no_cpy(
                    input,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                    comm_buf_idx=0,
                )
                triton_dist_sp_group_barrier_all()
            else:
                q_nh = nh // (gqa + 2) * gqa
                k_nh = nh // (gqa + 2)
                q = input[:, :, :q_nh].contiguous()
                k = input[:, :, q_nh:q_nh + k_nh].contiguous()
                v = input[:, :, q_nh + k_nh:].contiguous()
                q_comm_buf = triton_dist_get_input_comm_buf(q, 0)
                k_comm_buf = triton_dist_get_input_comm_buf(k, 1)
                v_comm_buf = triton_dist_get_input_comm_buf(v, 2)
                q_comm_buf.copy_(q)
                k_comm_buf.copy_(k)
                v_comm_buf.copy_(v)
                triton_dist_sp_group_barrier_all()
                out_q = triton_dist_pre_attn_a2a_no_cpy(
                    q,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                    comm_buf_idx=0,
                )
                out_k = triton_dist_pre_attn_a2a_no_cpy(
                    k,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                    comm_buf_idx=1,
                )
                out_v = triton_dist_pre_attn_a2a_no_cpy(
                    v,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                    comm_buf_idx=2,
                )
                triton_dist_sp_group_barrier_all()
                outputs = [out_q, out_k, out_v]
        return outputs

    all_inputs = [_gen_inputs(random.randint(1, max_local_seq_len), idx) for idx in range(num_iteration)]
    torch.cuda.synchronize()
    torch.distributed.barrier()

    all_torch_outputs = [_torch_impl(*inputs) for inputs in all_inputs]

    torch.cuda.synchronize()
    torch.distributed.barrier()

    all_triton_dist_outputs = [_triton_dist_impl(*inputs) for inputs in all_inputs]

    torch.cuda.synchronize()
    torch.distributed.barrier()

    is_bitwise = True
    for i in range(WORLD_SIZE):
        if i == RANK:
            for idx, (triton_dist_outs, torch_outs) in enumerate(zip(all_torch_outputs, all_triton_dist_outputs)):

                atol, rtol = 0, 0
                if not _verify_and_check_bitwise(torch_outs, triton_dist_outs, atol, rtol):
                    is_bitwise = False
        torch.distributed.barrier()

    if is_bitwise:
        print(f"rank[{TP_GROUP.rank()}]: ✅  torch vs triton_dist bitwise match")
    else:
        print(f"rank[{TP_GROUP.rank()}]: ❌  torch vs triton_dist not bitwise match")
    TP_GROUP.barrier()
    torch.cuda.synchronize()


@torch.no_grad()
def perf_torch(
    sp_group: torch.distributed.ProcessGroup,
    input: torch.Tensor,
    seq_lens_cpu: Optional[torch.Tensor],
    warmup: int,
    iters: int,
    gqa: int = 0,
):
    bs, local_seq_len, nh, head_dim = input.shape
    seq_len = (local_seq_len * sp_group.size() if seq_lens_cpu is None else sum(seq_lens_cpu.tolist()))
    # All to all input tensors from all gpus

    torch.distributed.barrier()

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()

        outputs = torch_pre_attn_qkv_pack_a2a(sp_group, input, bs, seq_len, nh, head_dim, gqa, seq_lens_cpu)
        end_events[i].record()

    comm_times = []  # all to all
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times) / iters * 1000

    return PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        outputs=outputs,
        total_ms=comm_time,
    )


@torch.no_grad()
def perf_triton_dist(
    sp_group: torch.distributed.ProcessGroup,
    input: torch.Tensor,
    seq_lens_cpu: Optional[torch.Tensor],
    warmup: int = 5,
    iters: int = 10,
    num_comm_sm: int = -1,
    gqa: int = 0,
    apply_pack: bool = True,
    local_copy: bool = False,
):
    bs, local_seq_len, nh, head_dim = input.shape
    q_nh = nh // (gqa + 2) * gqa
    k_nh = nh // (gqa + 2)
    q = input[:, :, :q_nh].contiguous()
    k = input[:, :, q_nh:q_nh + k_nh].contiguous()
    v = input[:, :, q_nh + k_nh:].contiguous()

    torch.distributed.barrier()

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    time.sleep(1)

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        if not local_copy:
            if apply_pack:
                outs = triton_dist_pre_attn_qkv_pack_a2a(
                    input,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                )
                print(f"outs = {outs}")
            else:
                out_q = triton_dist_pre_attn_a2a(
                    q,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                )
                out_k = triton_dist_pre_attn_a2a(
                    k,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                )
                out_v = triton_dist_pre_attn_a2a(
                    v,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                )
                outs = [out_q, out_k, out_v]
        else:
            if apply_pack:
                comm_buf = triton_dist_get_input_comm_buf(input, 0)
                comm_buf.copy_(input)
                triton_dist_sp_group_barrier_all()
                outs = triton_dist_pre_attn_qkv_pack_a2a_no_cpy(input, seq_lens_cpu=seq_lens_cpu,
                                                                num_comm_sm=num_comm_sm, comm_buf_idx=0)
                triton_dist_sp_group_barrier_all()
            else:
                q_comm_buf = triton_dist_get_input_comm_buf(q, 0)
                k_comm_buf = triton_dist_get_input_comm_buf(k, 1)
                v_comm_buf = triton_dist_get_input_comm_buf(v, 2)
                q_comm_buf.copy_(q)
                k_comm_buf.copy_(k)
                v_comm_buf.copy_(v)
                triton_dist_sp_group_barrier_all()
                out_q = triton_dist_pre_attn_a2a_no_cpy(
                    q,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                    comm_buf_idx=0,
                )
                out_k = triton_dist_pre_attn_a2a_no_cpy(
                    k,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                    comm_buf_idx=1,
                )
                out_v = triton_dist_pre_attn_a2a_no_cpy(
                    v,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                    comm_buf_idx=2,
                )
                triton_dist_sp_group_barrier_all()
                outs = [out_q, out_k, out_v]
        end_events[i].record()

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    a2a_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            a2a_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    a2a_time = sum(a2a_times)

    a2a_time_ms = a2a_time / iters * 1000

    return PerfResult(name=f"triton_dist  #{TP_GROUP.rank()}", outputs=outs, total_ms=a2a_time_ms)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bs", type=int)
    parser.add_argument("seq_len", type=int)
    parser.add_argument("nheads", type=int)
    parser.add_argument("head_dim", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--num_comm_sm", type=int, required=True, help="num sm for a2a")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument(
        "--verify",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="verify correctness",
    )
    parser.add_argument("--dp", default=False, action="store_true", help="dp per rank")
    parser.add_argument(
        "--comm_op",
        default="QKVPackA2A",
        choices=["QKVPackA2A"],
        help="pre attn all to all communication operation",
    )
    parser.add_argument(
        "--local-copy",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If local-copy is true, the user needs to ensure that the input is copied to comm buffer",
    )
    parser.add_argument("--gqa", default=0, type=int, help="group size of group query attn")
    parser.add_argument("--sp_size", default=0, required=True, type=int, help="sp size")
    parser.add_argument(
        "--apply_pack",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="whether to pack q/k/v",
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
        max_seq_len=args.seq_len,
        hidden_size=args.nheads * args.head_dim,
        head_dim=args.head_dim,
        qkv_out_features=args.nheads * args.head_dim,
        input_dtype=dtype,
        output_dtype=dtype,
        gqa=args.gqa,
        max_num_comm_buf=1 if args.apply_pack else 3,
    )

    if args.comm_op != "QKVPackA2A":
        raise NotImplementedError("only QKVPackA2A supported")

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
    assert args.seq_len % sp_group.size() == 0
    np.random.seed(3 + RANK // args.sp_size)

    max_local_seq_len = args.seq_len // sp_group.size()
    if args.dp is not None:
        seq_lens_cpu = None
        local_seq_len = max_local_seq_len
    else:
        seq_lens_list = list(
            np.random.randint(
                max(1, max_local_seq_len - 32),
                max_local_seq_len,
                size=(sp_group.size(), ),
            ))
        local_seq_len = seq_lens_list[sp_group.rank()]
        seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int32, device="cpu")
        if sp_group.rank() == 0:
            print(f"sp_group_id = {TP_GROUP.rank() // args.sp_size}, seq_lens_list = {seq_lens_list}")
    input_shape = [args.bs, local_seq_len, args.nheads, args.head_dim]
    input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) / 10 * (sp_group.rank() + 1)

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
            seq_lens_cpu,
            args.warmup,
            args.iters,
            args.gqa,
        )
        perf_res_triton_dist = perf_triton_dist(
            sp_group,
            input,
            seq_lens_cpu,
            args.warmup,
            args.iters,
            args.num_comm_sm,
            args.gqa,
            args.apply_pack,
            args.local_copy,
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

    torch_outputs = perf_res_torch.outputs
    triton_dist_outputs = perf_res_triton_dist.outputs

    torch.cuda.synchronize()
    torch.distributed.barrier()

    atol, rtol = 0, 0
    is_bitwise = _verify_and_check_bitwise(torch_outputs, triton_dist_outputs, atol=atol, rtol=rtol)
    if is_bitwise:
        print("✅  torch vs triton_dist bitwise match")
    else:
        print("❌  torch vs triton_dist not bitwise match")
    TP_GROUP.barrier()
    torch.cuda.synchronize()

    gemm_a2a_op.finalize()
    del gemm_a2a_op
    finalize_distributed()
