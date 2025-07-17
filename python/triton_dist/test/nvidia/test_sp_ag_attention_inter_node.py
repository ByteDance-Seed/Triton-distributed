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
import datetime
import os
import random
from functools import partial
from itertools import accumulate

import numpy as np
import torch
import nvshmem.core
from triton_dist.kernels.nvidia import (create_sp_ag_attention_context_inter_node, fused_sp_ag_attn_inter_node)
from triton_dist.utils import (dist_print, get_torch_prof_ctx, init_nvshmem_by_torch_process_group, perf_func,
                               nvshmem_barrier_all_on_stream)

##################################################


class FusedSequenceParallelAttn(torch.nn.Module):

    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        local_world_size: int,
        nnodes: int,
        batch_size: int,
        q_head: int,
        kv_head: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        head_dim: int,
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        device="cuda",
        is_causal=True,
        enable_zig_zag=True,
    ):
        super(FusedSequenceParallelAttn, self).__init__()
        self.pg = pg
        self.rank = pg.rank()
        self.world_size = pg.size()
        self.local_world_size = local_world_size
        self.nnodes = nnodes
        self.local_rank = self.rank % self.local_world_size

        self.batch_size = batch_size
        self.q_head = q_head
        self.kv_head = kv_head
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.head_dim = head_dim

        assert (max_seqlen_q % self.world_size == 0 and max_seqlen_q % self.world_size
                == 0), f"sequence length should be multiple of world_size({self.world_size})"
        self.max_q_shard_len = self.max_seqlen_q // self.world_size

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        assert self.input_dtype == self.output_dtype
        self.device = device
        self.is_causal = is_causal
        self.enable_zig_zag = enable_zig_zag

        self.ctx = create_sp_ag_attention_context_inter_node(
            self.batch_size,
            self.q_head,
            self.kv_head,
            self.max_seqlen_k,
            self.max_q_shard_len,
            self.head_dim,
            self.input_dtype,
            self.output_dtype,
            self.rank,
            self.local_world_size,
            self.world_size,
            self.device,
        )

    def forward(self, q_shard, k_shard, v_shard, cu_seqlens_q, cu_seqlens_k):
        total_q_shard = cu_seqlens_q[-1]
        output_buffer = self.ctx.attn_output_buffer[:total_q_shard]

        fused_sp_ag_attn_inter_node(
            self.ctx,
            q_shard,
            k_shard,
            v_shard,
            output_buffer,
            cu_seqlens_q,
            cu_seqlens_k,
            self.max_q_shard_len,
            self.max_seqlen_k,
            self.rank,
            self.world_size,
            self.nnodes,
            self.local_rank,
            self.is_causal,
            self.enable_zig_zag,
        )

        return output_buffer


class TorchSequenceParallelAttn(torch.nn.Module):

    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        batch_size: int,
        q_head: int,
        kv_head: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        head_dim: int,
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        device="cuda",
        is_causal=True,
        enable_zig_zag=True,
    ):
        super(TorchSequenceParallelAttn, self).__init__()
        self.pg = pg
        self.rank = pg.rank()
        self.world_size = pg.size()

        self.batch_size = batch_size
        self.q_head = q_head
        self.kv_head = kv_head
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.head_dim = head_dim

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.device = device
        self.is_causal = is_causal
        self.enable_zig_zag = enable_zig_zag
        assert self.input_dtype == self.output_dtype

        self.max_q_shard_len = max_seqlen_q // self.world_size
        self.max_kv_shard_ken = max_seqlen_q // self.world_size
        assert (max_seqlen_q % self.world_size == 0 and max_seqlen_q % self.world_size
                == 0), f"sequence length should be multiple of world_size({self.world_size})"

        self.ag_k_buffer: torch.Tensor = torch.empty(
            self.batch_size * self.max_seqlen_k,
            self.kv_head,
            self.head_dim,
            dtype=self.input_dtype,
            device=self.device,
        )
        self.ag_v_buffer: torch.Tensor = torch.empty(
            self.batch_size * self.max_seqlen_k,
            self.kv_head,
            self.head_dim,
            dtype=self.input_dtype,
            device=self.device,
        )

    def forward(self, q_shard, k_shard, v_shard, cu_seqlens_q, cu_seqlens_k):
        # construct casual mask with offset
        def _gen_mask(offset, q_shard_len, kv_len):
            if self.is_causal:
                mask = torch.zeros((q_shard_len, kv_len), dtype=torch.bool, device=self.device)
                mask[:, :offset + q_shard_len] = True
                if offset < kv_len:
                    mask[:, offset:offset + q_shard_len].tril_()
                return mask
            return None

        batch_size = cu_seqlens_q.shape[0] - 1
        total_q_shard, q_head, head_dim = q_shard.shape
        total_kv_shard, kv_head, head_dim = k_shard.shape
        out_list = []
        for i in range(batch_size):
            cu_seqlens_q_start = cu_seqlens_q[i].item()
            cu_seqlens_q_end = cu_seqlens_q[i + 1].item()
            q_shard_len = cu_seqlens_q_end - cu_seqlens_q_start
            q_len = q_shard_len * self.world_size

            cu_seqlens_k_start = cu_seqlens_k[i].item() // self.world_size
            cu_seqlens_k_end = cu_seqlens_k[i + 1].item() // self.world_size
            kv_shard_len = cu_seqlens_k_end - cu_seqlens_k_start
            kv_len = kv_shard_len * self.world_size

            if self.enable_zig_zag:
                half_q_shard_len = q_shard_len // 2
                half_kv_shard_len = kv_shard_len // 2

                q0_shard = q_shard[cu_seqlens_q_start:cu_seqlens_q_start + half_q_shard_len, :, :].clone()
                q1_shard = q_shard[cu_seqlens_q_start + half_q_shard_len:cu_seqlens_q_end, :, :].clone()

                q0_shard_permute = torch.permute(q0_shard.reshape(1, half_q_shard_len, q_head, head_dim),
                                                 (0, 2, 1, 3)).contiguous()
                q1_shard_permute = torch.permute(q1_shard.reshape(1, half_q_shard_len, q_head, head_dim),
                                                 (0, 2, 1, 3)).contiguous()

                k0_shard = k_shard[cu_seqlens_k_start:cu_seqlens_k_start + half_kv_shard_len, :, :].clone()
                k1_shard = k_shard[cu_seqlens_k_start + half_kv_shard_len:cu_seqlens_k_end, :, :].clone()
                v0_shard = v_shard[cu_seqlens_k_start:cu_seqlens_k_start + half_kv_shard_len, :, :].clone()
                v1_shard = v_shard[cu_seqlens_k_start + half_kv_shard_len:cu_seqlens_k_end, :, :].clone()

                buffer_size = (half_kv_shard_len * kv_head * head_dim * self.world_size)

                ag_k0 = self.ag_k_buffer.reshape(-1)[:buffer_size].reshape(half_kv_shard_len * self.world_size, kv_head,
                                                                           head_dim)
                ag_k1 = self.ag_k_buffer.reshape(-1)[buffer_size:2 * buffer_size].reshape(
                    half_kv_shard_len * self.world_size, kv_head, head_dim)
                ag_v0 = self.ag_v_buffer.reshape(-1)[:buffer_size].reshape(half_kv_shard_len * self.world_size, kv_head,
                                                                           head_dim)
                ag_v1 = self.ag_v_buffer.reshape(-1)[buffer_size:2 * buffer_size].reshape(
                    half_kv_shard_len * self.world_size, kv_head, head_dim)
                torch.distributed.all_gather_into_tensor(
                    ag_k0,
                    k0_shard,
                    group=self.pg,
                )
                torch.distributed.all_gather_into_tensor(
                    ag_k1,
                    k1_shard,
                    group=self.pg,
                )
                torch.distributed.all_gather_into_tensor(
                    ag_v0,
                    v0_shard,
                    group=self.pg,
                )
                torch.distributed.all_gather_into_tensor(
                    ag_v1,
                    v1_shard,
                    group=self.pg,
                )
                ag_k1 = ag_k1.reshape(self.world_size, half_kv_shard_len, kv_head, head_dim)
                ag_k1 = torch.flip(ag_k1, [0]).reshape(self.world_size * half_kv_shard_len, kv_head, head_dim)
                ag_k = torch.cat((ag_k0, ag_k1), dim=0)
                ag_k = torch.permute(ag_k.reshape(1, kv_len, kv_head, head_dim), (0, 2, 1, 3)).contiguous()
                ag_k = ag_k.repeat_interleave(q_head // kv_head, -3)

                ag_v1 = ag_v1.reshape(self.world_size, half_kv_shard_len, kv_head, head_dim)
                ag_v1 = torch.flip(ag_v1, [0]).reshape(self.world_size * half_kv_shard_len, kv_head, head_dim)
                ag_v = torch.cat((ag_v0, ag_v1), dim=0)
                ag_v = torch.permute(ag_v.reshape(1, kv_len, kv_head, head_dim), (0, 2, 1, 3)).contiguous()
                ag_v = ag_v.repeat_interleave(q_head // kv_head, -3)

                offset_q0 = half_q_shard_len * self.rank
                offset_q1 = q_len - half_q_shard_len * (self.rank + 1)
                prefix = kv_len - q_len
                mask0 = _gen_mask(prefix + offset_q0, half_q_shard_len, kv_len)
                mask1 = _gen_mask(prefix + offset_q1, half_q_shard_len, kv_len)
                out0 = torch.nn.functional.scaled_dot_product_attention(q0_shard_permute, ag_k, ag_v, attn_mask=mask0)
                out1 = torch.nn.functional.scaled_dot_product_attention(q1_shard_permute, ag_k, ag_v, attn_mask=mask1)
                out = torch.cat((out0, out1), dim=2)  # [1, q_head, q_shard_len, head_dim]
            else:
                cu_q_shard = q_shard[cu_seqlens_q_start:cu_seqlens_q_end, :, :].clone()
                cu_q_shard_permute = torch.permute(cu_q_shard.reshape(1, q_shard_len, q_head, head_dim),
                                                   (0, 2, 1, 3)).contiguous()

                total_size = kv_len * kv_head * head_dim
                ag_k = self.ag_k_buffer.reshape(-1)[:total_size].reshape(kv_len, kv_head, head_dim)
                cu_k_shard = k_shard[cu_seqlens_k_start:cu_seqlens_k_end, :, :].clone()
                torch.distributed.all_gather_into_tensor(
                    ag_k,
                    cu_k_shard,
                    group=self.pg,
                )
                ag_v = self.ag_v_buffer.reshape(-1)[:total_size].reshape(kv_len, kv_head, head_dim)
                cu_v_shard = v_shard[cu_seqlens_k_start:cu_seqlens_k_end, :, :].clone()
                torch.distributed.all_gather_into_tensor(
                    ag_v,
                    cu_v_shard,
                    group=self.pg,
                )
                ag_k = torch.permute(ag_k.reshape(1, kv_len, kv_head, head_dim), (0, 2, 1, 3)).contiguous()
                ag_k = ag_k.repeat_interleave(q_head // kv_head, -3)
                ag_v = torch.permute(ag_v.reshape(1, kv_len, kv_head, head_dim), (0, 2, 1, 3)).contiguous()
                ag_v = ag_v.repeat_interleave(q_head // kv_head, -3)

                offset = self.rank * q_shard_len
                prefix = kv_len - q_len
                mask = _gen_mask(prefix + offset, q_shard_len, kv_len)
                out = torch.nn.functional.scaled_dot_product_attention(
                    cu_q_shard_permute, ag_k, ag_v, attn_mask=mask)  # [1, q_head, q_shard_len, head_dim]

            out = torch.permute(out.reshape(q_head, q_shard_len, head_dim), (1, 0, 2)).contiguous()
            out_list.append(out)

        output = torch.cat(out_list)

        return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--q_head", type=int, default=32, help="num q heads")
    parser.add_argument("--kv_head", type=int, default=8, help="num kv heads")
    parser.add_argument("--max_seqlen_q", type=int, default=8192, help="max sequence length of q")
    parser.add_argument("--max_seqlen_k", type=int, default=12288, help="max sequence length of k/v")
    parser.add_argument("--head_dim", type=int, default=128, help="head dim")
    parser.add_argument("--seqlens_q", type=int, nargs='+', default=[4096, 8192], help="sequence lengths of q")
    parser.add_argument("--seqlens_k", type=int, nargs='+', default=[6144, 12288], help="sequence lengths of k/v")
    parser.add_argument(
        "--causal",
        "--no-causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="causal attn",
    )
    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="dump torch.profiler.profile",
    )
    parser.add_argument(
        "--zig_zag",
        "--no-zig-zag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable zig zag opt",
    )
    parser.add_argument("--warmup", default=10, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--check", default=False, action="store_true", help="correctness check")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(TP_GROUP)

    os.environ["NCCL_DEBUG"] = "ERROR"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)  # True or False
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + RANK)
    torch.cuda.manual_seed_all(3 + RANK)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + RANK)
    random.seed(42)

    current_stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    init_nvshmem_by_torch_process_group(TP_GROUP)

    rank = TP_GROUP.rank()
    world_size = TP_GROUP.size()

    batch_size = args.batch_size
    q_head = args.q_head
    kv_head = args.kv_head
    max_seqlen_q = args.max_seqlen_q
    max_seqlen_k = args.max_seqlen_k
    head_dim = args.head_dim
    input_dtype = torch.float16
    output_dtype = torch.float16
    device = "cuda"
    is_causal = args.causal
    enable_zig_zag = args.zig_zag

    iters = args.iters
    warmup_iters = args.warmup
    check = args.check

    seqlens_q = args.seqlens_q
    cu_seqlens_q_list = [0] + list(accumulate(seqlens_q))
    seqlens_k = args.seqlens_k
    cu_seqlens_k_list = [0] + list(accumulate(seqlens_k))

    # assertions about sequence length
    assert len(seqlens_q) == len(seqlens_k) and len(seqlens_q) == batch_size
    for i in range(len(seqlens_q)):
        assert seqlens_q[i] <= max_seqlen_q
        assert seqlens_k[i] <= max_seqlen_k
        assert seqlens_q[i] <= seqlens_k[i]
        assert seqlens_q[i] % world_size == 0 and seqlens_k[i] % world_size == 0

    with torch.no_grad():
        module = FusedSequenceParallelAttn(
            TP_GROUP,
            LOCAL_WORLD_SIZE,
            NNODES,
            batch_size,
            q_head,
            kv_head,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            input_dtype,
            output_dtype,
            device,
            is_causal,
            enable_zig_zag,
        )

        torch_module = TorchSequenceParallelAttn(
            TP_GROUP,
            batch_size,
            q_head,
            kv_head,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            input_dtype,
            output_dtype,
            device,
            is_causal,
            enable_zig_zag,
        )

        cu_seqlens_q = torch.tensor(cu_seqlens_q_list, dtype=torch.int32, device=device)
        cu_seqlens_q = cu_seqlens_q // world_size
        cu_seqlens_k = torch.tensor(cu_seqlens_k_list, dtype=torch.int32, device=device)

        q_shard = torch.empty(
            (cu_seqlens_q[-1], q_head, head_dim),
            dtype=input_dtype,
            device=device,
        ).normal_(mean=0.0, std=0.5)
        k_shard = torch.empty(
            (cu_seqlens_k[-1] // world_size, kv_head, head_dim),
            dtype=input_dtype,
            device=device,
        ).normal_(mean=0.0, std=0.5)
        v_shard = torch.empty(
            (cu_seqlens_k[-1] // world_size, kv_head, head_dim),
            dtype=input_dtype,
            device=device,
        ).normal_(mean=0.0, std=0.5)

        prof_ctx = get_torch_prof_ctx(args.profile)
        with prof_ctx:
            torch_output, torch_perf = perf_func(
                partial(torch_module.forward, q_shard, k_shard, v_shard, cu_seqlens_q, cu_seqlens_k), iters=iters,
                warmup_iters=warmup_iters)

            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            torch.cuda.synchronize()

            output, perf = perf_func(partial(module.forward, q_shard, k_shard, v_shard, cu_seqlens_q, cu_seqlens_k),
                                     iters=iters, warmup_iters=warmup_iters)

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        prof_ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    if check:
        atol = 1e-2
        rtol = 1e-2
        torch.testing.assert_close(output, torch_output, atol=atol, rtol=rtol)
        torch.cuda.synchronize()

    dist_print(f"dist-triton #{RANK}", perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
