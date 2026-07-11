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
"""Test PreAttnA2AOp / PreAttnQKVPackA2AOp (triton_dist TMA pull).

Aligned with Flux test_pre_attn_a2a_op.py.

Usage (single tensor):
    torchrun --nproc_per_node=8 python/triton_dist/test/nvidia/tma/test_pre_attn_a2a_op.py \
        --verify --q_nheads 64 --hd 128 --max_seq 16384 --iters 20

Usage (QKV pack):
    torchrun --nproc_per_node=8 python/triton_dist/test/nvidia/tma/test_pre_attn_a2a_op.py \
        --verify --qkv_pack --q_nheads 64 --k_nheads 8 --v_nheads 8 --max_seq 16384
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


def torch_pre_attn_qkv_ref(sp_group, inp, q_nh, k_nh, v_nh):
    local_seq, total_nh, hd = inp.shape
    ws = sp_group.size()
    seq = local_seq * ws
    lq = q_nh // ws
    lk = k_nh // ws if k_nh > 0 else 0
    lv = v_nh // ws if v_nh > 0 else 0

    def do(t, nh, lnh):
        if t is None or nh == 0:
            return torch.empty((seq, 0, hd), dtype=inp.dtype, device=inp.device)
        a = t.permute(1, 0, 2).contiguous()
        b = torch.empty((ws, lnh, local_seq, hd), dtype=t.dtype, device=t.device)
        torch.distributed.all_to_all_single(b.view(-1), a.view(-1), group=sp_group)
        return b.permute(0, 2, 1, 3).reshape(seq, lnh, hd).contiguous()

    q_in = inp[:, :q_nh, :]
    k_in = inp[:, q_nh:q_nh + k_nh, :] if k_nh > 0 else None
    v_in = inp[:, q_nh + k_nh:, :] if v_nh > 0 else None
    return do(q_in, q_nh, lq), do(k_in, k_nh, lk), do(v_in, v_nh, lv)


def torch_pre_attn_ref(sp_group, inp):
    q, _, _ = torch_pre_attn_qkv_ref(sp_group, inp, inp.size(1), 0, 0)
    return q


@dataclass
class TestConfig:
    max_seq_len: int
    q_nh: int
    k_nh: int
    v_nh: int
    hd: int
    sp_size: int
    dtype: torch.dtype = torch.bfloat16

    @property
    def total_nh(self) -> int:
        return self.q_nh + self.k_nh + self.v_nh

    @property
    def max_local_seq(self) -> int:
        return self.max_seq_len // self.sp_size

    def total_bytes(self, local_seq: int) -> int:
        bytes_per_elem = torch.tensor([], dtype=self.dtype).element_size()
        return local_seq * self.total_nh * self.hd * bytes_per_elem

    @property
    def is_qkv_pack(self) -> bool:
        return self.k_nh > 0 or self.v_nh > 0


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


def create_input_tensor(local_seq: int, nh: int, hd: int, sp_rank: int, dtype: torch.dtype) -> torch.Tensor:
    return (torch.rand((local_seq, nh, hd), device="cuda", dtype=dtype) * 2.0 - 1.0) * (sp_rank + 1)


def benchmark(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(int(1e8))
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def run_verify_single(sp_group, config: TestConfig, iters: int, num_sm: int):
    from triton_dist.layers.nvidia.pre_attn_a2a_layer import PreAttnA2AOp

    random.seed(42)
    op = PreAttnA2AOp(
        sp_group=sp_group,
        max_seq_len=config.max_seq_len,
        nheads=config.q_nh,
        head_dim=config.hd,
    )

    max_local_seq = config.max_local_seq
    local_seq_lens = [random.randint(1, max_local_seq) for _ in range(iters)]

    all_inputs = [
        create_input_tensor(ls, config.q_nh, config.hd, sp_group.rank(), config.dtype) for ls in local_seq_lens
    ]

    torch.distributed.barrier()
    torch.cuda.synchronize()

    torch_outputs = [torch_pre_attn_ref(sp_group, inp) for inp in all_inputs]

    torch.distributed.barrier()
    torch.cuda.synchronize()

    triton_outputs = [op.forward(inp, num_sm=num_sm) for inp in all_inputs]

    torch.distributed.barrier()
    torch.cuda.synchronize()

    for idx, (tri_out, torch_out) in enumerate(zip(triton_outputs, torch_outputs)):
        try:
            torch.testing.assert_close(tri_out, torch_out, atol=0, rtol=0)
        except Exception as e:
            local_seq = all_inputs[idx].shape[0]
            print(f"[Rank {RANK}] FAIL at iter {idx}, local_seq={local_seq}")
            raise e

    print(f"[Rank {RANK}] single q_nh={config.q_nh} "
          f"max_seq={config.max_seq_len}: "
          f"all {len(local_seq_lens)} iterations passed (bitwise)")

    op.finalize()
    torch.distributed.barrier()
    torch.cuda.synchronize()


def run_verify_qkv(sp_group, config: TestConfig, iters: int, num_sm: int):
    from triton_dist.layers.nvidia.pre_attn_a2a_layer import (
        PreAttnQKVPackA2AOp, )

    random.seed(42)
    op = PreAttnQKVPackA2AOp(
        sp_group=sp_group,
        max_seq_len=config.max_seq_len,
        q_nheads=config.q_nh,
        k_nheads=config.k_nh,
        v_nheads=config.v_nh,
        head_dim=config.hd,
    )

    max_local_seq = config.max_local_seq
    local_seq_lens = [random.randint(1, max_local_seq) for _ in range(iters)]

    total_nh = config.total_nh
    all_inputs = [create_input_tensor(ls, total_nh, config.hd, sp_group.rank(), config.dtype) for ls in local_seq_lens]

    torch.distributed.barrier()
    torch.cuda.synchronize()

    torch_outputs = [torch_pre_attn_qkv_ref(sp_group, inp, config.q_nh, config.k_nh, config.v_nh) for inp in all_inputs]

    torch.distributed.barrier()
    torch.cuda.synchronize()

    triton_outputs = [op.forward(inp, num_sm=num_sm) for inp in all_inputs]

    torch.distributed.barrier()
    torch.cuda.synchronize()

    for idx, ((fq, fk, fv), (tq, tk, tv)) in enumerate(zip(triton_outputs, torch_outputs)):
        try:
            torch.testing.assert_close(fq, tq, atol=0, rtol=0)
            if config.k_nh > 0:
                torch.testing.assert_close(fk, tk, atol=0, rtol=0)
            if config.v_nh > 0:
                torch.testing.assert_close(fv, tv, atol=0, rtol=0)
        except Exception as e:
            local_seq = all_inputs[idx].shape[0]
            print(f"[Rank {RANK}] FAIL at iter {idx}, local_seq={local_seq}")
            raise e

    print(f"[Rank {RANK}] QKV q={config.q_nh} k={config.k_nh} "
          f"v={config.v_nh} max_seq={config.max_seq_len}: "
          f"all {len(local_seq_lens)} iterations passed (bitwise)")

    op.finalize()
    torch.distributed.barrier()
    torch.cuda.synchronize()


def run_perf(
    sp_group,
    config: TestConfig,
    warmup: int,
    iters: int,
    num_sm: int,
):
    is_qkv = config.is_qkv_pack
    local_seq = config.max_local_seq

    if is_qkv:
        from triton_dist.layers.nvidia.pre_attn_a2a_layer import (
            PreAttnQKVPackA2AOp, )
        input_3d = create_input_tensor(
            local_seq,
            config.total_nh,
            config.hd,
            sp_group.rank(),
            config.dtype,
        )
        op = PreAttnQKVPackA2AOp(
            sp_group=sp_group,
            max_seq_len=config.max_seq_len,
            q_nheads=config.q_nh,
            k_nheads=config.k_nh,
            v_nheads=config.v_nh,
            head_dim=config.hd,
        )
    else:
        from triton_dist.layers.nvidia.pre_attn_a2a_layer import PreAttnA2AOp
        input_3d = create_input_tensor(
            local_seq,
            config.q_nh,
            config.hd,
            sp_group.rank(),
            config.dtype,
        )
        op = PreAttnA2AOp(
            sp_group=sp_group,
            max_seq_len=config.max_seq_len,
            nheads=config.q_nh,
            head_dim=config.hd,
        )

    if is_qkv:
        ref = torch_pre_attn_qkv_ref(
            sp_group,
            input_3d,
            config.q_nh,
            config.k_nh,
            config.v_nh,
        )
        out = op.forward(input_3d, num_sm=num_sm)
        torch.testing.assert_close(out[0], ref[0], atol=0, rtol=0)
    else:
        ref = torch_pre_attn_ref(sp_group, input_3d)
        out = op.forward(input_3d, num_sm=num_sm)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)
    if RANK == 0:
        print("correctness: pass")

    torch.distributed.barrier(sp_group)
    if is_qkv:
        torch_ms = benchmark(
            lambda: torch_pre_attn_qkv_ref(
                sp_group,
                input_3d,
                config.q_nh,
                config.k_nh,
                config.v_nh,
            ),
            warmup,
            iters,
        )
        triton_ms = benchmark(
            lambda: op.forward(input_3d, num_sm=num_sm),
            warmup,
            iters,
        )
    else:
        torch_ms = benchmark(
            lambda: torch_pre_attn_ref(sp_group, input_3d),
            warmup,
            iters,
        )
        triton_ms = benchmark(
            lambda: op.forward(input_3d, num_sm=num_sm),
            warmup,
            iters,
        )
    torch.distributed.barrier(sp_group)
    torch.cuda.synchronize()

    tb = config.total_bytes(local_seq)
    mode_str = (f"QKV({config.q_nh},{config.k_nh},{config.v_nh})" if is_qkv else f"single({config.q_nh})")
    torch_res = PerfResult("torch", torch_ms, tb)
    triton_res = PerfResult("triton_dist", triton_ms, tb)

    for i in range(WORLD_SIZE):
        if i == RANK:
            print(f"rank={RANK} {mode_str} seq={config.max_seq_len} "
                  f"| {torch_res} | {triton_res}")
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
    p.add_argument("--k_nheads", type=int, default=8)
    p.add_argument("--v_nheads", type=int, default=8)
    p.add_argument("--hd", type=int, default=128)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--num_sm", type=int, default=16)
    p.add_argument(
        "--verify",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    p.add_argument("--profile", default=False, action="store_true")
    p.add_argument("--qkv_pack", default=False, action="store_true")
    args = p.parse_args()

    ws = sp_group.size()
    assert args.q_nheads % ws == 0, (f"q_nheads={args.q_nheads} must be divisible by sp_size={ws}")
    assert args.max_seq % ws == 0, (f"max_seq={args.max_seq} must be divisible by sp_size={ws}")

    k = args.k_nheads if args.qkv_pack else 0
    v = args.v_nheads if args.qkv_pack else 0
    if k > 0:
        assert k % ws == 0, (f"k_nheads={k} must be divisible by sp_size={ws}")
    if v > 0:
        assert v % ws == 0, (f"v_nheads={v} must be divisible by sp_size={ws}")

    config = TestConfig(
        max_seq_len=args.max_seq,
        q_nh=args.q_nheads,
        k_nh=k,
        v_nh=v,
        hd=args.hd,
        sp_size=ws,
    )

    torch.distributed.barrier()

    if args.verify:
        if args.qkv_pack:
            if RANK == 0:
                print(f"=== Verify QKV: q={args.q_nheads} k={k} "
                      f"v={v} seq={args.max_seq} ===")
            run_verify_qkv(sp_group, config, args.iters, args.num_sm)
        else:
            if RANK == 0:
                print(f"=== Verify single: q={args.q_nheads} "
                      f"seq={args.max_seq} ===")
            run_verify_single(
                sp_group,
                config,
                args.iters,
                args.num_sm,
            )
    else:
        ctx = get_torch_prof_ctx(args.profile)
        m = (f"QKV({args.q_nheads},{k},{v})" if args.qkv_pack else f"single({args.q_nheads})")
        if RANK == 0:
            print(f"=== Perf {m} seq={args.max_seq} ===")
        for i in range(2):
            with ctx:
                run_perf(
                    sp_group,
                    config,
                    args.warmup,
                    args.iters,
                    args.num_sm,
                )
            if args.profile:
                prof_dir = "prof/pre_attn_a2a"
                os.makedirs(prof_dir, exist_ok=True)
                ctx.export_chrome_trace(f"{prof_dir}/trace_rank{sp_group.rank()}.json.gz")

    if RANK == 0:
        print("Done!")
    finalize_distributed()


if __name__ == "__main__":
    main()
