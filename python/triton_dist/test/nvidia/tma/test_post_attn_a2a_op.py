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
"""Test PostAttnA2AOp (triton_dist TMA push).

Aligned with Flux test_post_attn_a2a_op.py -- verify correctness against
torch.distributed.all_to_all_single and measure bandwidth.

Usage:
    torchrun --nproc_per_node=8 python/triton_dist/test/nvidia/tma/test_post_attn_a2a_op.py \\
        --verify --q_nheads 64 --hd 128 --max_seq 16384 --iters 20
"""
import argparse
import os
import random
from dataclasses import dataclass
from functools import partial
import torch
import torch.distributed
from triton_dist.utils import initialize_distributed, finalize_distributed
from triton_dist.profiler_utils import get_torch_prof_ctx

print = partial(print, flush=True)
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


def torch_post_attn_a2a_reference(sp_group, input_3d):
    """Reference: (seq_len, local_nh, hd) -> (local_seq, nh, hd)."""
    seq_len, local_nh, hd = input_3d.shape
    ws = sp_group.size()
    local_seq = seq_len // ws
    input_chunked = input_3d.view(ws, local_seq, local_nh, hd)
    output_chunked = torch.empty_like(input_chunked)
    torch.distributed.all_to_all_single(output_chunked.view(-1), input_chunked.contiguous().view(-1), group=sp_group)
    return output_chunked.permute(1, 0, 2, 3).reshape(local_seq, local_nh * ws, hd).contiguous()


@dataclass
class TestConfig:
    max_seq_len: int
    nh: int
    hd: int
    sp_size: int
    dtype: torch.dtype = torch.bfloat16

    @property
    def local_nh(self) -> int:
        return self.nh // self.sp_size

    @property
    def max_local_seq(self) -> int:
        return self.max_seq_len // self.sp_size

    @property
    def total_bytes(self) -> int:
        bytes_per_elem = torch.tensor([], dtype=self.dtype).element_size()
        return self.max_seq_len * self.local_nh * self.hd * bytes_per_elem


@dataclass
class PerfResult:
    name: str
    time_ms: float
    total_bytes: int

    @property
    def bandwidth_gbps(self) -> float:
        return self.total_bytes / (self.time_ms / 1000) / 1e9

    def __str__(self) -> str:
        return f"{self.name}: {self.time_ms:.3f} ms ({self.bandwidth_gbps:.2f} GB/s)"


def create_input_tensor(config: TestConfig, seq_len: int, sp_rank: int) -> torch.Tensor:
    return (torch.rand(
        (seq_len, config.local_nh, config.hd), device="cuda", dtype=config.dtype) * 2.0 - 1.0) * (sp_rank + 1)


def benchmark(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(int(1e8))
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_verify(sp_group, config: TestConfig, iters: int, num_sm: int):
    from triton_dist.layers.nvidia.post_attn_a2a_layer import PostAttnA2AOp

    random.seed(42)
    op = PostAttnA2AOp(sp_group=sp_group, max_seq_len=config.max_seq_len, nheads=config.nh, head_dim=config.hd,
                       dtype=config.dtype)

    max_local_seq = config.max_local_seq
    random_seq_lens = [random.randint(1, max_local_seq) * config.sp_size for _ in range(iters)]

    all_inputs = [create_input_tensor(config, s, sp_group.rank()) for s in random_seq_lens]

    torch.distributed.barrier()
    torch.cuda.synchronize()

    torch_outputs = [torch_post_attn_a2a_reference(sp_group, inp) for inp in all_inputs]

    torch.distributed.barrier()
    torch.cuda.synchronize()

    triton_outputs = [op.forward(inp, num_sm=num_sm) for inp in all_inputs]

    torch.distributed.barrier()
    torch.cuda.synchronize()

    for idx, (tri_out, torch_out) in enumerate(zip(triton_outputs, torch_outputs)):
        try:
            torch.testing.assert_close(tri_out, torch_out, atol=0, rtol=0)
        except Exception as e:
            print(f"[Rank {RANK}] FAIL at iter {idx}, shape = {all_inputs[idx].shape}")
            raise e

    print(f"[Rank {RANK}] nh={config.nh} max_seq={config.max_seq_len}: "
          f"all {iters} iterations passed (bitwise)")

    op.finalize()
    torch.distributed.barrier()
    torch.cuda.synchronize()


def run_perf(sp_group, config: TestConfig, warmup: int, iters: int, num_sm: int):
    from triton_dist.layers.nvidia.post_attn_a2a_layer import PostAttnA2AOp

    input_3d = create_input_tensor(config, config.max_seq_len, sp_group.rank())
    op = PostAttnA2AOp(sp_group=sp_group, max_seq_len=config.max_seq_len, nheads=config.nh, head_dim=config.hd,
                       dtype=config.dtype)

    ref = torch_post_attn_a2a_reference(sp_group, input_3d)
    out = op.forward(input_3d, num_sm=num_sm)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    if RANK == 0:
        print(f"nh={config.nh} max_seq={config.max_seq_len}: correctness pass")

    torch.distributed.barrier(sp_group)
    torch_ms = benchmark(lambda: torch_post_attn_a2a_reference(sp_group, input_3d), warmup=warmup, iters=iters)
    triton_ms = benchmark(lambda: op.forward(input_3d, return_comm_buf=False, num_sm=num_sm), warmup=warmup,
                          iters=iters)
    torch.distributed.barrier(sp_group)
    torch.cuda.synchronize()

    tb = config.total_bytes
    torch_res = PerfResult("torch", torch_ms, tb)
    triton_res = PerfResult("triton_dist", triton_ms, tb)

    for i in range(WORLD_SIZE):
        if i == RANK:
            print(f"rank={RANK} nh={config.nh} seq={config.max_seq_len} | {torch_res} | {triton_res}")
        torch.distributed.barrier()

    op.finalize()
    torch.distributed.barrier()
    torch.cuda.synchronize()


def main():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    sp_group = initialize_distributed()

    p = argparse.ArgumentParser()
    p.add_argument("--max_seq", type=int, default=16384)
    p.add_argument("--q_nheads", type=int, default=64)
    p.add_argument("--hd", type=int, default=128)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--num_sm", type=int, default=16)
    p.add_argument("--verify", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--profile", default=False, action="store_true")
    args = p.parse_args()

    ws = sp_group.size()
    assert args.q_nheads % ws == 0, f"q_nheads={args.q_nheads} must be divisible by sp_size={ws}"
    assert args.max_seq % ws == 0, f"max_seq={args.max_seq} must be divisible by sp_size={ws}"

    config = TestConfig(max_seq_len=args.max_seq, nh=args.q_nheads, hd=args.hd, sp_size=ws)

    torch.distributed.barrier()

    if args.verify:
        if RANK == 0:
            print(f"=== Verify: nh={args.q_nheads}, max_seq={args.max_seq}, hd={args.hd} ===")
        run_verify(sp_group, config, args.iters, num_sm=args.num_sm)
    else:
        ctx = get_torch_prof_ctx(args.profile)
        if RANK == 0:
            print(f"=== Perf: nh={args.q_nheads}, max_seq={args.max_seq}, hd={args.hd} ===")

        for i in range(2):
            with ctx:
                run_perf(sp_group, config, args.warmup, args.iters, args.num_sm)
            if args.profile:
                prof_dir = "prof/post_attn_a2a"
                os.makedirs(prof_dir, exist_ok=True)
                ctx.export_chrome_trace(f"{prof_dir}/trace_rank{sp_group.rank()}.json.gz")

    if RANK == 0:
        print("Done!")
    finalize_distributed()


if __name__ == "__main__":
    main()
