################################################################################
#
# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
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

from triton_dist.kernels.nvidia import (mla_fwd_batch_decode_inter_rank_combine, mla_fwd_batch_decode_intra_rank,
                                        sp_decode_allgather_push_intra_node)
from triton_dist.utils import (NVSHMEM_SIGNAL_DTYPE, nvshmem_barrier_all_on_stream, nvshmem_create_tensor,
                               nvshmem_free_tensor_sync)


class SpMLAFlashDecodeAttention(torch.nn.Module):
    """Sequence-parallel MLA decode on a single node.

    Each rank holds the full absorbed query and a shard of the compressed KV
    cache (FlashMLA layout, ``[num_pages, page_size, 1, kv_lora_rank + qk_rope_head_dim]``).
    ``forward`` computes the local partial, all-gathers the partials over NVSHMEM
    on the current stream and combines them, so the output is the attention over
    the concatenation of all shards.

    All communication buffers are allocated here, and ``forward`` does not read
    anything back to the host, so it can be captured into a CUDA graph. Every
    rank must call ``forward`` the same number of times with the same batch size.
    """

    def __init__(self, rank, num_ranks, num_heads, kv_lora_rank=512, qk_rope_head_dim=64, page_size=1, scale=1.0,
                 soft_cap=0.0, max_allowed_batch=1, kv_split=32, block_n=32, num_stages=2):
        super().__init__()
        assert num_ranks & (num_ranks - 1) == 0, f"num_ranks ({num_ranks}) must be a power of two"
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.page_size = page_size
        self.scale = scale
        self.soft_cap = soft_cap
        self.max_allowed_batch = max_allowed_batch
        self.kv_split = kv_split
        self.block_n = block_n
        self.num_stages = num_stages

        # Partials stay in fp32 end to end, so the exchange does not lose precision.
        partial_shape = (max_allowed_batch, num_heads, kv_lora_rank + 1)
        self.output_split = torch.empty((max_allowed_batch, num_heads, kv_split, kv_lora_rank + 1), dtype=torch.float32,
                                        device="cuda")
        self.output_combine = torch.empty(partial_shape, dtype=torch.float32, device="cuda")

        partial_numel = max_allowed_batch * num_heads * (kv_lora_rank + 1)
        self.symm_buffer = nvshmem_create_tensor((num_ranks * partial_numel, ), torch.float32)
        self.symm_signal = nvshmem_create_tensor((num_ranks, ), NVSHMEM_SIGNAL_DTYPE)
        self.symm_signal.fill_(0)
        self.grid_barrier = torch.zeros((1, ), dtype=torch.int32, device="cuda")
        nvshmem_barrier_all_on_stream()
        torch.cuda.synchronize()

    def finalize(self):
        nvshmem_free_tensor_sync(self.symm_buffer)
        nvshmem_free_tensor_sync(self.symm_signal)

    def forward(self, q, kv_cache, global_kv_lens, block_table):
        """
        q: [batch, num_heads, kv_lora_rank + qk_rope_head_dim], fp16 or bf16.
        kv_cache: [num_pages, page_size, 1, kv_lora_rank + qk_rope_head_dim], this rank's shard.
        global_kv_lens: [num_ranks, batch] int32, contiguous. Row ``r`` is the KV length held by rank ``r``.
        block_table: [batch, max_pages_per_seq] int32 into this rank's ``kv_cache``.

        Returns [batch, num_heads, kv_lora_rank] in ``q.dtype``.
        """
        batch, num_heads, head_dim = q.shape
        assert batch <= self.max_allowed_batch, f"batch {batch} > max_allowed_batch {self.max_allowed_batch}"
        assert num_heads == self.num_heads and head_dim == self.kv_lora_rank + self.qk_rope_head_dim
        assert kv_cache.shape[1] == self.page_size
        assert global_kv_lens.shape == (self.num_ranks, batch) and global_kv_lens.dtype == torch.int32

        partial = mla_fwd_batch_decode_intra_rank(q, kv_cache, global_kv_lens[self.rank], block_table, self.scale,
                                                  self.kv_lora_rank, soft_cap=self.soft_cap,
                                                  output_split=self.output_split[:batch],
                                                  output_combine=self.output_combine[:batch], kv_split=self.kv_split,
                                                  block_n=self.block_n, num_stages=self.num_stages)
        all_partials = sp_decode_allgather_push_intra_node(partial, self.symm_buffer, self.symm_signal,
                                                           self.grid_barrier, self.rank, self.num_ranks)
        output = torch.empty((batch, num_heads, self.kv_lora_rank), dtype=q.dtype, device=q.device)
        return mla_fwd_batch_decode_inter_rank_combine(all_partials, global_kv_lens, output)
