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

import nvshmem.core
import torch
import torch.distributed

from triton_dist.kernels.nvidia.reduce_scatter import (create_reduce_scater_2d_ctx, reduce_scatter_2d_op,
                                                       reduce_scatter_ring_push_1d_intra_node_ce,
                                                       reduce_scatter_ring_push_1d_intra_node_sm,
                                                       reduce_scatter_ring_push_1d_intra_node_sm_rma)
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import initialize_distributed, nvshmem_barrier_all_on_stream, nvshmem_create_tensors, nvshmem_create_tensor, sleep_async


def fill_random(tensor: torch.Tensor):
    if tensor.dtype == torch.int32:
        tensor.random_(0, 12345)
    elif tensor.dtype.is_floating_point:
        tensor.random_()
    else:
        raise NotImplementedError


def test_reduce_scatter_ring_push_1d_intra_node(M_per_rank, N, dtype: torch.dtype, method, warmup_iters=30, iters=500,
                                                debug: bool = False, profile: bool = False):

    M = M_per_rank * WORLD_SIZE

    input_tensor = nvshmem_create_tensor((M, N), dtype)
    fill_random(input_tensor)
    if debug:
        input_tensor.fill_(RANK + 1)

    symm_reduce_buffers = nvshmem_create_tensors((M, N), dtype, RANK, LOCAL_WORLD_SIZE)
    symm_reduce_buffer = symm_reduce_buffers[LOCAL_RANK]

    input_flag = torch.ones((WORLD_SIZE, ), device="cuda", dtype=torch.int32)

    symm_reduce_flags = nvshmem_create_tensors((WORLD_SIZE, ), torch.int32, RANK, LOCAL_WORLD_SIZE)
    symm_reduce_flag = symm_reduce_flags[LOCAL_RANK]
    symm_reduce_flag.zero_()

    if method != "ce":
        grid_barrier = torch.zeros((1, ), device="cuda", dtype=torch.int32)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    ref_output = torch.empty((M_per_rank, N), dtype=dtype).cuda()
    torch.distributed.reduce_scatter_tensor(ref_output, input_tensor, group=TP_GROUP)

    def _reduce_scatter_fn():
        if method == "ce":
            output = reduce_scatter_ring_push_1d_intra_node_ce(
                RANK,
                WORLD_SIZE,
                input_tensor,
                input_flag,
                symm_reduce_buffers,
                symm_reduce_flags,
            )
        elif method == "sm":
            output = reduce_scatter_ring_push_1d_intra_node_sm(
                RANK,
                WORLD_SIZE,
                input_tensor,
                input_flag,
                symm_reduce_buffer,
                symm_reduce_flag,
                grid_barrier,
                num_sms=8,
            )
        elif method == "sm_rma":
            output = reduce_scatter_ring_push_1d_intra_node_sm_rma(
                RANK,
                WORLD_SIZE,
                input_tensor,
                input_flag,
                symm_reduce_buffer,
                symm_reduce_flag,
                grid_barrier,
                num_sms=8,
            )
        else:
            raise Exception(f"Unsupported method {method}")

        symm_reduce_flag.zero_()
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        return output

    output = _reduce_scatter_fn()
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    try:
        assert_allclose(output, ref_output, atol=0, rtol=0)
    except Exception as e:
        print(f"❌ RANK[{RANK}] check failed")
        torch.save(input_tensor, f"input_tensor_{LOCAL_RANK}.pt")
        torch.save(output, f"output_{LOCAL_RANK}.pt")
        torch.save(ref_output, f"ref_output_{LOCAL_RANK}.pt")
        torch.save(symm_reduce_buffers[LOCAL_RANK], f"symm_reduce_buffers_{LOCAL_RANK}.pt")
        raise e
    else:
        print(f"✅ RANK[{RANK}] check passed")

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    sleep_async(1000)
    _run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    with group_profile(f"reduce_scatter_1d_{method}_{M}x{N}_{_run_id}", group=TP_GROUP, do_prof=profile):
        _, duration_ms = perf_func(_reduce_scatter_fn, iters, warmup_iters)

    gbps = (lambda ms: input_tensor.nbytes * 1e-9 / (ms * 1e-3) * (WORLD_SIZE - 1) / WORLD_SIZE)
    print(f"[{method}] RANK = {RANK}, Bandwith = {gbps(duration_ms):0.2f} GB/S")


def test_reduce_scatter_2d_op(M_per_rank, N, dtype, profile, warmup_iters=30, iters=500, debug=False):
    M = M_per_rank * WORLD_SIZE
    input_tensor = torch.empty((M, N), dtype=dtype, device="cuda")
    fill_random(input_tensor)
    if debug:
        input_tensor.fill_(RANK + 1)
    ref_output = torch.empty((M_per_rank, N), dtype=dtype, device="cuda")
    torch.distributed.reduce_scatter_tensor(ref_output, input_tensor, group=TP_GROUP)
    ctx = create_reduce_scater_2d_ctx(M, N, RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, dtype, overlap_with_gemm=True)
    ctx.scatter_signal_buf.fill_(1)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    def _reduce_scatter_fn():
        output = reduce_scatter_2d_op(input_tensor, ctx)
        ctx.scatter_signal_buf.fill_(1)
        return output

    output = _reduce_scatter_fn()
    try:
        assert_allclose(output, ref_output, atol=0, rtol=0)
    except Exception as e:
        print(f"❌ RANK[{RANK}] check failed")
        torch.save(output, f"output_{LOCAL_RANK}.pt")
        torch.save(ref_output, f"ref_output_{LOCAL_RANK}.pt")
        torch.save(ctx.scatter_bufs, f"scatter_bufs_{LOCAL_RANK}.pt")
        torch.save(ctx.p2p_buf, f"p2p_buf_{LOCAL_RANK}.pt")
        raise e
    else:
        print(f"✅ RANK[{RANK}] check passed")

    _run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    with group_profile(f"reduce_scatter_2d_{M}x{N}_{_run_id}", group=TP_GROUP, do_prof=profile):
        sleep_async(1000)  # in case CPU bound
        _, duration_ms = perf_func(_reduce_scatter_fn, iters, warmup_iters)

    gbps = (lambda ms: input_tensor.nbytes * 1e-9 / (ms * 1e-3) * (WORLD_SIZE - 1) / WORLD_SIZE)
    print(f"RANK = {RANK}, Bandwith = {gbps(duration_ms):0.2f} GB/S")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="float16", type=str, help="data type")
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check-only", default=False, action="store_true", help="correctness check")
    parser.add_argument("--debug", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed()
    args = parse_args()
    assert args.M % WORLD_SIZE == 0
    M_per_rank = args.M // WORLD_SIZE
    N = args.N
    warmup_iters = args.warmup
    iters = args.iters

    if LOCAL_WORLD_SIZE == WORLD_SIZE:
        for method in ["ce", "sm", "sm_rma"]:
            test_reduce_scatter_ring_push_1d_intra_node(M_per_rank, N, torch.int32, method=method,
                                                        warmup_iters=warmup_iters, iters=iters, debug=args.debug,
                                                        profile=args.profile)
        torch.cuda.synchronize()

    test_reduce_scatter_2d_op(M_per_rank, N, torch.int32, profile=args.profile, warmup_iters=warmup_iters, iters=iters,
                              debug=args.debug)
    torch.cuda.synchronize()

    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
