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
from typing import List, Optional

import random
import numpy as np
import torch
import torch.distributed
from triton_dist.utils import initialize_distributed, finalize_distributed
from triton_dist.profiler_utils import perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.layers.nvidia import UlyssesSPAllToAllLayer

print = partial(print, flush=True)


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


def torch_pre_attn_qkv_a2a(sp_group, q_input, k_input, v_input, bs, seq_len, q_nh, head_dim, gqa, skip_q_a2a,
                           seq_lens_cpu=None):
    world_size = sp_group.size()

    def _a2a(a2a_tensor):
        if seq_lens_cpu is None:
            a2a_input = a2a_tensor.permute(2, 1, 0, 3).contiguous()  # [nh, local_seq_len, bs, hd]
            a2a_nh, a2a_local_seq_len, a2a_bs, a2a_hd = a2a_input.shape
            if a2a_nh < world_size:
                assert world_size % a2a_nh == 0
                repeats = world_size // a2a_nh
                a2a_input = torch.repeat_interleave(a2a_input, repeats=repeats, dim=0).contiguous()
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
            raise NotImplementedError("not supported")

    if not skip_q_a2a:
        q = _a2a(q_input)
    else:
        q = None
    k = _a2a(k_input)
    v = _a2a(v_input)
    return [q, k, v]


@torch.no_grad()
def perf_torch(sp_group: torch.distributed.ProcessGroup, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               seq_lens_cpu: Optional[torch.Tensor], warmup: int, iters: int, gqa: int = 0, skip_q_a2a: bool = False):
    local_seq_len = q.shape[1]
    seq_len = (local_seq_len * sp_group.size() if seq_lens_cpu is None else sum(seq_lens_cpu.tolist()))
    # All to all input tensors from all gpus

    torch.distributed.barrier()

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    bs, _, q_nh, head_dim = q.shape
    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()

        outputs = torch_pre_attn_qkv_a2a(sp_group, q, k, v, bs, seq_len, q_nh, head_dim, gqa, skip_q_a2a, seq_lens_cpu)
        end_events[i].record()

    comm_times = []  # all to all
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times) / iters * 1000

    return PerfResult(
        name=f"torch #{SP_GROUP.rank()}",
        outputs=outputs,
        total_ms=comm_time,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bs", type=int)
    parser.add_argument("seq_len", type=int)
    parser.add_argument("nheads", type=int)
    parser.add_argument("head_dim", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--num_comm_sm", default=8, type=int, help="num sm for a2a")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--gqa", default=1, type=int, help="group size of group query attn")
    parser.add_argument("--sp_size", default=0, type=int, help="sp size")
    parser.add_argument("--verify-iters", default=30, type=int)
    parser.add_argument("--check", default=False, action="store_true", help="stress test")
    parser.add_argument("--return_comm_buf", default=False, action="store_true", help="output in comm buf")
    parser.add_argument("--skip_q_a2a", default=False, action="store_true", help="skip a2a of q")

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

    SP_GROUP = initialize_distributed(0)
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    if LOCAL_RANK == 0:
        print(f"args = {args}")
    dtype = DTYPE_MAP[args.dtype]
    if args.sp_size == 0:
        args.sp_size = WORLD_SIZE

    # init sp process group
    assert args.sp_size > 0
    num_sp_group = WORLD_SIZE // args.sp_size
    all_sp_subgroups = []

    assert SP_GROUP is not None
    assert args.seq_len % SP_GROUP.size() == 0
    np.random.seed(3 + RANK // args.sp_size)

    max_local_seq_len = args.seq_len // SP_GROUP.size()

    seq_lens_list = [max_local_seq_len for i in range(SP_GROUP.size())]
    local_seq_len = seq_lens_list[SP_GROUP.rank()]
    if SP_GROUP.rank() == 0:
        print(f"sp_group_id = {SP_GROUP.rank() // args.sp_size}, seq_lens_list = {seq_lens_list}")

    def _make_data(cur_local_seq_len):
        q_shape = [args.bs, cur_local_seq_len, args.nheads, args.head_dim]
        kv_shape = [args.bs, cur_local_seq_len, args.nheads // args.gqa, args.head_dim]
        q = (-2 * torch.rand(q_shape, dtype=dtype).cuda() + 1) / 10 * (SP_GROUP.rank() + 1)
        k = (-2 * torch.rand(kv_shape, dtype=dtype).cuda() + 1) / 10 * (SP_GROUP.rank() + 1)
        v = (-2 * torch.rand(kv_shape, dtype=dtype).cuda() + 1) / 10 * (SP_GROUP.rank() + 1)
        return q, k, v

    ulysses_sp_layer = UlyssesSPAllToAllLayer(SP_GROUP, args.bs, max_seq=args.seq_len, q_nheads=args.nheads,
                                              kv_nheads=args.nheads // args.gqa, k_head_dim=args.head_dim,
                                              v_head_dim=args.head_dim, group_size=args.gqa, dtype=dtype,
                                              local_world_size=LOCAL_WORLD_SIZE)
    torch.distributed.barrier()

    if args.check:
        for n in range(args.iters):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            input_list = [_make_data(random.randint(1, local_seq_len)) for _ in range(args.verify_iters)]
            torch_outs, triton_outs = [], []

            for cur_q, cur_k, cur_v in input_list:
                perf_res_torch = perf_torch(
                    SP_GROUP,
                    cur_q,
                    cur_k,
                    cur_v,
                    seq_lens_cpu=None,
                    warmup=0,
                    iters=1,
                    gqa=args.gqa,
                    skip_q_a2a=args.skip_q_a2a,
                )
                torch_q_ref, torch_k_ref, torch_v_ref = perf_res_torch.outputs
                torch_outs.append((torch_q_ref, torch_k_ref, torch_v_ref))

            for cur_q, cur_k, cur_v in input_list:
                triton_out_q, triton_out_k, triton_out_v = ulysses_sp_layer.pre_attn_qkv_pack_a2a(
                    cur_q, cur_k, cur_v, skip_q_a2a=args.skip_q_a2a)
                triton_outs.append((triton_out_q, triton_out_k, triton_out_v))

            for idx, (torch_out_qkv, triton_out_qkv) in enumerate(zip(torch_outs, triton_outs)):
                torch_q_ref, torch_k_ref, torch_v_ref = torch_out_qkv
                triton_q, triton_k, triton_v = triton_out_qkv

                if RANK == 0:
                    print(f"shape = {triton_q.shape if not args.skip_q_a2a else triton_k.shape}")
                try:
                    if not args.skip_q_a2a:
                        torch.testing.assert_close(torch_q_ref, triton_q, atol=0, rtol=0)
                    torch.testing.assert_close(torch_k_ref, triton_k, atol=0, rtol=0)
                    torch.testing.assert_close(torch_v_ref, triton_v, atol=0, rtol=0)

                except Exception as e:
                    raise e
        ulysses_sp_layer.finalize()
        finalize_distributed()
        exit(0)

    assert args.gqa > 0 and args.nheads % args.gqa == 0
    q_shape = [args.bs, local_seq_len, args.nheads, args.head_dim]
    kv_shape = [args.bs, local_seq_len, args.nheads // args.gqa, args.head_dim]
    q, k, v = _make_data(local_seq_len)
    # q.fill_(RANK)
    ctx = (torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) if args.profile else nullcontext())

    with ctx:
        perf_res_torch = perf_torch(
            SP_GROUP,
            q,
            k,
            v,
            None,
            args.warmup,
            args.iters,
            args.gqa,
            args.skip_q_a2a,
        )

        (triton_out_q, triton_out_k, triton_out_v), triton_perf = perf_func(
            partial(ulysses_sp_layer.pre_attn_qkv_pack_a2a, q, k, v, args.skip_q_a2a, args.return_comm_buf), iters=100,
            warmup_iters=10)

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{SP_GROUP.rank()}.json.gz")

    for i in range(SP_GROUP.size()):
        if i == SP_GROUP.rank():
            print(perf_res_torch)
            print(f"triton #{RANK}: total {triton_perf:.3f} ms")
        torch.distributed.barrier()
    inter_node_nbytes_wo_opt, intra_node_nbytes_wo_opt, inter_node_nbytes_with_opt, intra_node_nbytes_with_opt = ulysses_sp_layer.get_comm_nbytes(
        q, k, v, args.skip_q_a2a)
    if LOCAL_RANK == 0:
        print(
            f"inter_node_nbytes_wo_opt = {inter_node_nbytes_wo_opt / 1e6} MB, intra_node_nbytes_wo_opt = {intra_node_nbytes_wo_opt / 1e6} MB, inter_node_nbytes_with_opt = {inter_node_nbytes_with_opt / 1e6} MB, intra_node_nbytes_with_opt = {intra_node_nbytes_with_opt / 1e6} MB"
        )

    torch_outputs = perf_res_torch.outputs
    torch_q_ref, torch_k_ref, torch_v_ref = torch_outputs
    if not args.skip_q_a2a:
        assert_allclose(torch_q_ref, triton_out_q, atol=0, rtol=0)
        torch.testing.assert_close(torch_q_ref, triton_out_q, atol=0, rtol=0)
    torch.testing.assert_close(torch_k_ref, triton_out_k, atol=0, rtol=0)
    torch.testing.assert_close(torch_v_ref, triton_out_v, atol=0, rtol=0)

    ulysses_sp_layer.finalize()
    finalize_distributed()
