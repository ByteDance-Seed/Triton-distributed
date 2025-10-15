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
import random

import nvshmem.core
import torch
import torch.distributed

from triton_dist.kernels.nvidia import ag_gemm, create_ag_gemm_context
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import initialize_distributed

TP_GROUP = initialize_distributed()


def torch_ag_gemm(
    pg: torch.distributed.ProcessGroup,
    local_input: torch.Tensor,
    local_weight: torch.Tensor,
    ag_out: torch.Tensor,
):
    torch.distributed.all_gather_into_tensor(ag_out, local_input, pg)
    ag_gemm_output = torch.matmul(ag_out, local_weight)
    return ag_gemm_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=11008)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--verify_shapes", type=int, default=5)
    parser.add_argument("--verify_hang", type=int, default=40)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--simulate_straggler", default=False, action="store_true")
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    dtype = torch.float16
    random.seed(args.seed)  # set all ranks to the same seed

    def _make_data(M):
        if TP_GROUP.rank() == 0:
            print(f"test M {M}")

        M_per_rank = M // TP_GROUP.size()
        N_per_rank = args.N // TP_GROUP.size()

        A = torch.randn([M_per_rank, args.K], dtype=dtype, device="cuda")
        if args.trans_b:
            B = torch.randn([N_per_rank, args.K], dtype=dtype, device="cuda").T
        else:
            B = torch.randn([args.K, N_per_rank], dtype=dtype, device="cuda")

        torch_ag_out = torch.empty([M, args.K], dtype=dtype, device="cuda")
        ctx = create_ag_gemm_context(A, B, TP_GROUP.rank(), TP_GROUP.size(), max_M=M)

        return A, B, ctx, torch_ag_out

    def _run_dist_triton(A: torch.Tensor, B: torch.Tensor, ctx, straggler_option=None):
        return ag_gemm(A, B, ctx, straggler_option=straggler_option)

    for n in range(args.iters):
        # generate data for verify
        tensor_inputs = []
        for _ in range(args.verify_shapes):
            test_m = random.randint(1, args.max_M // TP_GROUP.size()) * TP_GROUP.size()
            tensor_inputs.append(_make_data(test_m))

        triton_out_list, torch_out_list = [], []

        for input, weight, ctx, _ in tensor_inputs:
            res = _run_dist_triton(input, weight, ctx)
            triton_out_list.append(res)

        for input, weight, _, torch_ag_out in tensor_inputs:
            ag_gemm_res = torch_ag_gemm(TP_GROUP, input, weight, torch_ag_out)
            torch_out_list.append(ag_gemm_res)

        # verify
        for triton_res, torch_res in zip(triton_out_list, torch_out_list):
            check_failed = False
            for i in range(TP_GROUP.size()):
                torch.distributed.barrier(TP_GROUP)
                if TP_GROUP.rank() == i:
                    try:
                        assert_allclose(triton_res, torch_res, atol=1e-3, rtol=1e-3)
                    except Exception:
                        check_failed = True
            if check_failed:
                exit(1)

        # just runs, check if hangs
        straggler_option = None if not args.simulate_straggler else (random.randint(
            0,
            TP_GROUP.size() - 1), random.randint(1e9, 1e9 + 1e8))  # straggler id, straggler_latency (ns)
        if straggler_option:
            print(f"straggler id {straggler_option[0]}, latency {straggler_option[1] / 1000 / 1000 / 1000} s")

        for j in range(args.verify_hang):
            _run_dist_triton(input, weight, ctx, straggler_option)

        if (n + 1) % 10 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(f"runs {n + 1} iterations done")

    if TP_GROUP.rank() == 0:
        print("Pass the stree test!")

    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
