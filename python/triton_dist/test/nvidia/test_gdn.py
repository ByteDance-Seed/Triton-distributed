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
import os
import argparse
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule
from triton_dist.utils import perf_func
from triton_dist.kernels.nvidia import chunk_gated_delta_rule_fwd

RANK = int(os.getenv("RANK", "0"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

# you can enable FLA baseline if installed
HAS_FLA = False

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_device("cuda")
    dtype = torch.bfloat16
    CHUNK_SIZE = 64

    parser = argparse.ArgumentParser()
    parser.add_argument("--prof", action="store_true", help="whether to run the profiler")
    parser.add_argument("--num_heads", type=int, default=16, choices=[12, 16, 18, 20], help="number of heads")
    args = parser.parse_args()

    def get_profiler(name: str):
        handler = torch.profiler.tensorboard_trace_handler(name, use_gzip=True)
        return profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            # profile_memory=True,
            # record_shapes=True,
            with_stack=True,
            # with_flops=True,
            schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=handler,
        )

    def fused_triton():
        return chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state=None, output_final_state=False,
                                          cu_seqlens=cu_seqlens)

    def unfused_fla():
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as gdn_fwd_fla
        return gdn_fwd_fla(q, k, v, g, beta, scale, initial_state=None, output_final_state=False, cu_seqlens=cu_seqlens)

    func_table = {"gdn_dist_triton": fused_triton}

    if HAS_FLA:
        func_table["gdn_fla"] = unfused_fla

    warmup_iters = 20
    bench_iters = 200

    for T in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        B, H, K, V = 1, args.num_heads, 128, 128
        scale = 1.0 / (K**0.5)
        q = torch.randn(1, T, H, K, dtype=dtype)
        k = F.normalize(torch.randn(1, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
        v = torch.randn([B, T, H, V], dtype=dtype)
        beta = torch.randn([B, T, H], dtype=dtype).sigmoid()
        g = F.logsigmoid(torch.rand(1, T, H, dtype=torch.float32))
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32)

        outputs = {}
        for func_name, func in func_table.items():
            x, t = perf_func(func, warmup_iters=warmup_iters, iters=bench_iters)
            print(f"{func_name:<20} {T:<8}: {t:.3f} ms")
            # compare `O`
            outputs[func_name] = x[1]

        # if T in [1024, 8192, 32768]:
        #     profiler = get_profiler(f"prof/gdn_fused_triton_{T}")
        #     with profiler:
        #         for func in func_table.values():
        #             for _ in range(5):
        #                 func()

        try:
            ref = outputs["gdn_fla"] if HAS_FLA else outputs["gdn_dist_triton"]
            outputs.pop("gdn_fla" if HAS_FLA else "gdn_dist_triton")

            for func_name, res in outputs.items():
                torch.testing.assert_close(res, ref, rtol=1e-4, atol=1e-4)
                print(f"✅ Correctness check for `{func_name}`passed.")
        except Exception as e:
            print("❌ Correctness check failed.")
            raise e
