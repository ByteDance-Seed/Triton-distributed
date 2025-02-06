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
import os
from triton.distributed.kernels.nvidia import (create_fast_allgather_context, get_triton_combine_kv_algo_info,
                                               gqa_fwd_batch_decode_intra_rank_aot, gqa_fwd_batch_decode_intra_rank,
                                               kernel_inter_rank_gqa_fwd_batch_decode_combine_kv)
from .low_latency_allgather_layer import AllGatherLayer
import pynvshmem
if "USE_TRITON_DISTRIBUTED_AOT" in os.environ and os.environ["USE_TRITON_DISTRIBUTED_AOT"] in [
        "1", "true", "on", "ON", "On", True
]:
    use_aot = True
else:
    use_aot = False

if use_aot:
    from triton._C.libtriton_distributed import distributed


class SpGQAFlashDecodeAttention(torch.nn.Module):

    def __init__(self, rank, node, num_ranks, num_nodes, num_q_heads, num_kv_heads, q_head_dim, v_head_dim, page_size=1,
                 scale=1, soft_cap=0, max_allowed_batch=1, thrink_buffer_threshold=500, stages=20):
        super().__init__()
        self.rank = rank
        self.num_ranks = num_ranks
        self.node = node
        self.num_nodes = num_nodes

        self.workspace = None  # gqa_fwd doesn't need workspace
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.q_head_dim = q_head_dim
        self.v_head_dim = v_head_dim
        self.page_size = page_size
        self.soft_cap = soft_cap
        self.scale = scale
        self.kv_split = 32
        self.max_allowed_batch = max_allowed_batch
        self.stages = stages

        # allgather
        self.max_allgather_buffer_size = self.num_ranks * self.num_q_heads * self.v_head_dim * 8  # bytes
        self.ag_layer = AllGatherLayer(self.num_nodes, self.num_ranks, self.rank,
                                       max_buffer_size=self.max_allgather_buffer_size, stages=self.stages)
        self.ag_buffer = pynvshmem.nvshmem_create_tensor((
            self.stages,
            self.max_allgather_buffer_size,
        ), torch.int8)

        # track buffer size
        self.count_less_than_half = 0
        self.shrink_buffer_threshold = thrink_buffer_threshold

    def forward(self, q, k_cache, v_cache, global_kv_lens, block_table):
        """
        q: each rank has the same q
        k_cache: each rank's shard of k_cache
        v_cache: each rank's shard of v_cache
        global_kv_lens: all the rank's kv shard's length
        block_table: each rank's kv shard's kv_table
        """
        batch = q.shape[0]
        assert global_kv_lens.shape[0] == self.num_ranks
        assert global_kv_lens.shape[1] == batch
        assert batch <= self.max_allowed_batch, f"Only support {self.max_allowed_batch} queries decode now"
        output_split = torch.empty([batch, self.num_q_heads, self.kv_split, self.v_head_dim + 1], dtype=q.dtype,
                                   device=q.device)
        output_combine = torch.empty([batch, self.num_q_heads, self.v_head_dim + 1], dtype=q.dtype, device=q.device)
        final_output = torch.empty([batch, self.num_q_heads, self.v_head_dim], dtype=q.dtype, device=q.device)

        current_stream = torch.cuda.current_stream()
        if use_aot:
            gqa_fwd_batch_decode_intra_rank_aot(current_stream, q, k_cache, v_cache, self.workspace, [1] * q.shape[0],
                                                global_kv_lens[self.rank], block_table, self.scale,
                                                soft_cap=self.soft_cap, output_split=output_split,
                                                output_combine=output_combine, kv_split=self.kv_split)
        else:
            gqa_fwd_batch_decode_intra_rank(q, k_cache, v_cache, self.workspace, [1] * q.shape[0],
                                            global_kv_lens[self.rank], block_table, self.scale, soft_cap=self.soft_cap,
                                            output_split=output_split, output_combine=output_combine,
                                            kv_split=self.kv_split)
        ################
        # allgather part
        nbytes_per_rank = output_combine.nbytes
        nbytes = nbytes_per_rank * self.num_ranks

        if nbytes >= self.max_allgather_buffer_size:
            self.max_allgather_buffer_size *= 2
            self.allgather_ctx = create_fast_allgather_context(self.rank, self.node, self.num_ranks, self.num_nodes,
                                                               max_buffer_size=self.max_allgather_buffer_size)
            self.ag_buffer = pynvshmem.nvshmem_create_tensor((
                self.stages,
                self.max_allgather_buffer_size,
            ), torch.int8)
        if nbytes < self.max_allgather_buffer_size // 2:
            self.count_less_than_half += 1
        if self.count_less_than_half >= self.shrink_buffer_threshold:
            self.max_allgather_buffer_size = self.max_allgather_buffer_size // 2
            del self.allgather_ctx
            del self.ag_buffer
            self.allgather_ctx = create_fast_allgather_context(self.rank, self.node, self.num_ranks, self.num_nodes,
                                                               max_buffer_size=self.max_allgather_buffer_size)
            self.ag_buffer = pynvshmem.nvshmem_create_tensor((
                self.stages,
                self.max_allgather_buffer_size,
            ), torch.int8)
            # reset counter
            self.count_less_than_half = 0

        # local copy
        index_start, index_end = nbytes_per_rank * self.rank, nbytes_per_rank * (self.rank + 1)
        self.ag_buffer[self.ag_layer.signal_target % self.stages][index_start:index_end].copy_(
            output_combine.view(-1).view(torch.int8))
        ag_buffer = self.ag_layer.forward_push_2d_ll(self.ag_buffer[self.ag_layer.signal_target % self.stages][:nbytes])

        ################
        # final combine
        all_ranks_output_combine = ag_buffer.view(output_combine.dtype)
        all_ranks_output_combine = all_ranks_output_combine.view(
            [self.num_ranks, batch, self.num_q_heads, self.v_head_dim + 1])

        if use_aot:
            if output_split.dtype == torch.float32:
                kernel_combine = distributed.inter_rank_gqa_fwd_batch_decode_combine_kv_fp32_fp16
                combine_algo_info = distributed.inter_rank_gqa_fwd_batch_decode_combine_kv_fp32_fp16__triton_algo_info_t(
                )
            elif output_split.dtype == torch.float16:
                kernel_combine = distributed.inter_rank_gqa_fwd_batch_decode_combine_kv_fp16_fp16
                combine_algo_info = distributed.inter_rank_gqa_fwd_batch_decode_combine_kv_fp16_fp16__triton_algo_info_t(
                )
            else:
                raise RuntimeError("Unsupported data type of intermediate output:", output_split.dtype)
            py_combine_algo_info = get_triton_combine_kv_algo_info(split_kv=self.num_ranks, v_head_dim=self.v_head_dim,
                                                                   block_dv=1024)

            for k, v in py_combine_algo_info.items():
                setattr(combine_algo_info, k, v)

            kernel_combine(current_stream.cuda_stream, all_ranks_output_combine.data_ptr(), final_output.data_ptr(),
                           global_kv_lens.data_ptr(), batch, self.num_q_heads,
                           all_ranks_output_combine.stride(1),  # batch
                           all_ranks_output_combine.stride(2),  # head
                           all_ranks_output_combine.stride(0),  # num_ranks
                           final_output.stride(0),  # batch
                           final_output.stride(1),  # head
                           combine_algo_info)
        else:
            kernel_inter_rank_gqa_fwd_batch_decode_combine_kv[(batch, self.num_q_heads, 1)](
                all_ranks_output_combine, final_output, global_kv_lens, batch, self.num_q_heads,
                all_ranks_output_combine.stride(1),  # batch
                all_ranks_output_combine.stride(2),  # head
                all_ranks_output_combine.stride(0),  # num_ranks
                final_output.stride(0),  # batch
                final_output.stride(1),  # head
                self.num_ranks,  # split_kv
                512,  # BLOCK_DV
                self.v_head_dim,  # Lv
            )

        return final_output
