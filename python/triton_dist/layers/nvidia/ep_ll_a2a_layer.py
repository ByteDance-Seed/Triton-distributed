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
import torch
import triton
from typing import Optional
from dataclasses import dataclass
import os

from triton_dist.kernels.nvidia.low_latency_all_to_all_v2 import (
    dispatch_kernel_v2,
    create_ep_ll_a2a_ctx,
    combine_kernel_v2,
)

from triton_dist.tools.profiler import (alloc_profiler_buffer, reset_profiler_buffer, export_to_perfetto_trace)


@dataclass
class DispatchMetaInfo:
    recv_token_source_indices: torch.Tensor  # [num_experts_per_rank, world_size * max_m] int32
    recv_token_source_count_and_start: torch.Tensor  # [num_experts_per_rank, world_size] int64(count, start)


class EPLowLatencyAllToAllLayer:

    def __init__(
        self,
        max_m: int,
        hidden: int,
        topk: int,
        online_quant_fp8: bool,
        rank: int,
        num_experts: int,
        local_world_size: int,
        world_size: int,
        fp8_gsize: int = 128,
        dtype=torch.bfloat16,
        enable_profiling=False,
    ):
        """
            - max_m: max number of tokens per rank. e.g. 128 or 256
        """
        assert online_quant_fp8 is True
        self.scale_dtype = torch.float32
        self.max_m = max_m
        self.hidden = hidden
        self.topk = topk
        self.online_quant_fp8 = online_quant_fp8
        self.fp8_gsize = fp8_gsize
        self.num_groups = hidden // fp8_gsize

        self.num_experts = num_experts
        self.num_experts_per_rank = num_experts // world_size

        self.world_size = world_size
        self.local_world_size = local_world_size
        self.rank = rank
        self.enable_profiling = enable_profiling
        self.dispatch_ctx, self.combine_ctx = create_ep_ll_a2a_ctx(max_m, hidden, topk, num_experts, online_quant_fp8,
                                                                   fp8_gsize, dtype, world_size, rank)
        self.dispatch_profile_buf = alloc_profiler_buffer(max_num_profile_slots=1000000)
        self.combine_profile_buf = alloc_profiler_buffer(max_num_profile_slots=1000000)

        reset_profiler_buffer(self.dispatch_profile_buf)
        reset_profiler_buffer(self.combine_profile_buf)

    def finalize(self):
        self.dispatch_ctx.finalize()
        self.combine_ctx.finalize()

    def dispatch(
        self,
        send_tokens: torch.Tensor,
        send_scales: Optional[torch.Tensor],
        topk_indices: torch.Tensor,
    ):
        if not self.online_quant_fp8:
            assert send_tokens.dtype == self.dispatch_ctx.dtype
        else:
            assert send_tokens.dtype == torch.bfloat16

        assert send_scales is None, "currently only support online quant"

        if send_scales is not None:
            assert send_scales.dtype == self.scale_dtype
        num_tokens = send_tokens.shape[0]

        # [num_local_experts, world_size * max_m]
        recv_token_source_count_and_start = torch.empty([self.num_experts_per_rank, self.world_size], dtype=torch.int64,
                                                        device=torch.cuda.current_device())
        recv_token_source_indices = torch.empty([self.num_experts_per_rank, self.world_size * self.max_m],
                                                dtype=torch.int32, device=torch.cuda.current_device())
        recv_scale = torch.empty([self.num_experts_per_rank, self.world_size *
                                  self.max_m, self.num_groups], dtype=torch.float32,
                                 device=torch.cuda.current_device()) if self.online_quant_fp8 else None
        recv_token = torch.empty([self.num_experts_per_rank, self.world_size * self.max_m, self.hidden],
                                 dtype=self.dispatch_ctx.dtype, device=torch.cuda.current_device())

        # zero init in dispatch_kernel_v2
        expert_recv_count = torch.empty([
            self.num_experts_per_rank,
        ], dtype=torch.int32, device=torch.cuda.current_device())

        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = (min(num_sms, num_tokens), )

        dispatch_kernel_v2[grid](
            self.dispatch_profile_buf,
            send_tokens,
            send_scales,
            topk_indices,
            num_tokens,
            self.dispatch_ctx.send_token_buffer,
            self.dispatch_ctx.recv_token_buffer,
            self.dispatch_ctx.send_count_buffer,
            self.dispatch_ctx.recv_count_buffer,
            self.dispatch_ctx.recv_slot_counter,
            self.dispatch_ctx.signal_buffer,
            recv_token_source_indices,
            recv_scale,
            recv_token,
            expert_recv_count,
            recv_token_source_count_and_start,
            grid_sync_counter=self.dispatch_ctx.grid_sync_counter,
            signal_val=self.dispatch_ctx.signal_val,
            TOPK=self.dispatch_ctx.topk,
            ONLINE_QUANT_FP8=self.dispatch_ctx.online_quant_fp8,
            FP8_GSIZE=self.dispatch_ctx.fp8_gsize,
            WORLD_SIZE=self.dispatch_ctx.world_size,
            HIDDEN=self.dispatch_ctx.hidden,
            MAX_M=self.dispatch_ctx.max_m,
            NUM_EXPERTS=self.dispatch_ctx.num_experts,
            BN=triton.next_power_of_2(self.dispatch_ctx.hidden),
            BLOCK_SCALE=triton.next_power_of_2(self.num_groups),
            BLOCK_EXPERT_PER_RANK=triton.next_power_of_2(self.num_experts_per_rank),
            META_BYTES=self.dispatch_ctx.meta_bytes,
            MSG_SIZE=self.dispatch_ctx.msg_size,
            ENABLE_PROFILING=self.enable_profiling,
            num_warps=32,
        )
        self.dispatch_ctx.update_phase()
        dispatch_meta = DispatchMetaInfo(recv_token_source_indices=recv_token_source_indices,
                                         recv_token_source_count_and_start=recv_token_source_count_and_start)
        return recv_token, recv_scale, expert_recv_count, dispatch_meta

    def dump_dispatch_trace(self):
        if self.enable_profiling:
            profiler_dir = "./prof"
            os.makedirs(profiler_dir, exist_ok=True)
            trace_file = os.path.join(profiler_dir, f"ll_dispatch_RANK_{self.rank}")
            export_to_perfetto_trace(
                profiler_buffer=self.dispatch_profile_buf,
                task_names=["quant_and_put", "count_put", "wait", "postprocess"],
                file_name=trace_file,
            )

    def dump_combine_trace(self):
        if self.enable_profiling:
            profiler_dir = "./prof"
            os.makedirs(profiler_dir, exist_ok=True)
            trace_file = os.path.join(profiler_dir, f"ll_combine_RANK_{self.rank}")
            export_to_perfetto_trace(
                profiler_buffer=self.combine_profile_buf,
                task_names=["copy_and_put", "recv_wait", "topk_reduce"],
                file_name=trace_file,
            )

    def combine(
        self,
        send_tokens: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        dispatch_meta: DispatchMetaInfo,
        zero_copy=False,
    ):
        """
        param:
            - send_tokens: [num_experts_per_rank, world_size * max_m, hidden].
            - topk_indices: [num_combined_tokens, topk]
            - topk_weights: [num_combined_tokens, topk]
            - dispatch_meta: DispatchMetaInfo, meta info returned in dispatch
            - zero_copy: whether the `send_tokens` is already copied into the communication buffer
        return:
            - combined_tokens: [num_combined_tokens, hidden]
        """
        assert send_tokens.dtype == self.combine_ctx.dtype
        num_combined_tokens = topk_indices.shape[0]
        hidden = send_tokens.shape[-1]
        assert hidden == self.combine_ctx.hidden
        combined_tokens = torch.empty([num_combined_tokens, self.combine_ctx.hidden], dtype=self.combine_ctx.dtype,
                                      device=torch.cuda.current_device())
        dispatch_recv_token_source_count_and_start = dispatch_meta.recv_token_source_count_and_start
        dispatch_recv_token_source_indices = dispatch_meta.recv_token_source_indices
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

        grid = (min(num_sms, self.num_experts), )
        BM = 16

        combine_kernel_v2[grid](
            self.combine_profile_buf,
            send_tokens=send_tokens,  # [num_experts_per_rank, world_size * max_m, hidden]
            send_tokens_comm_buf=self.combine_ctx.
            send_tokens_comm_buf,  # [num_experts_per_rank, world_size * max_m, hidden]
            topk_indices=topk_indices,  # [num_combined_tokens, topk]
            topk_weights=topk_weights,  # [num_combined_tokens, topk]
            combined_out=combined_tokens,  # [num_combined_tokens, hidden]
            recv_token_buffer=self.combine_ctx.recv_token_buffer,  # [num_experts, max_m, hidden], comm buf
            signal_buf=self.combine_ctx.signal_buffer,  # [num_expert], comm buf
            dispatch_recv_token_source_indices=
            dispatch_recv_token_source_indices,  # [num_experts_per_rank, world_size * max_m] int32
            dispatch_recv_token_source_count_and_start=
            dispatch_recv_token_source_count_and_start,  # [num_experts_per_rank, world_size] int64(count, start)
            grid_sync_counter=self.combine_ctx.grid_sync_counter,  # [1, ] torch.int32 zero init
            num_combined_tokens=num_combined_tokens,
            signal_val=self.combine_ctx.signal_val,
            BM=BM,  # token dim
            TOPK=self.combine_ctx.topk,
            HIDDEN=self.combine_ctx.hidden,
            MAX_M=self.combine_ctx.max_m,
            NUM_EXPERTS=self.combine_ctx.num_experts,
            LOCAL_WORLD_SIZE=self.local_world_size,
            ZERO_COPY=zero_copy,
            ENABLE_PROFILING=self.enable_profiling,
            num_warps=32,
        )

        self.combine_ctx.update_phase()

        return combined_tokens
