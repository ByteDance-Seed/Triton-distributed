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
"""Kernels for sequence-parallel MLA decode.

Every rank holds the same absorbed query and a shard of the compressed KV cache.
A rank computes a partial ``(o, lse)`` over its own shard, the partials are
all-gathered, and a final combine reduces them with the log-sum-exp rule.

The local partial and the final combine reuse the split-KV kernels in
``flash_decode.py``: MLA is the ``kv_heads == 1`` case of that kernel, with
``K_DIM = kv_lora_rank + qk_rope_head_dim`` and the value read from the first
``kv_lora_rank`` channels of the same cache entry.

The all-gather resets its signals inside the kernel and synchronizes the ranks
with a device-side barrier, so a call carries no per-call host state. That keeps
it valid when it is captured into a CUDA graph and replayed.
"""

import torch
import triton
import triton.language as tl

import triton_dist
from triton_dist.kernels.nvidia.common_ops import barrier_on_this_grid
from triton_dist.kernels.nvidia.flash_decode import (gqa_fwd_batch_decode_intra_rank,
                                                     kernel_inter_rank_gqa_fwd_batch_decode_combine_kv)
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.cuda.language_extra import __syncthreads, tid
from triton_dist.utils import launch_cooperative_grid_options


@triton_dist.jit(do_not_specialize=["rank"])
def kernel_sp_decode_allgather_push_intra_node(
    input_ptr,
    symm_buffer_ptr,
    symm_signal_ptr,
    grid_barrier_ptr,
    rank,
    world_size: tl.constexpr,
    n_elements,
    use_cooperative: tl.constexpr,
):
    """Push this rank's partial into slot ``rank`` of every peer's ``symm_buffer``.

    The signals are cleared and all ranks pass a barrier before anyone pushes,
    so a peer cannot write into a buffer that is still being combined from the
    previous call, and no signal value has to be carried over between calls.
    """
    thread_idx = tid(0)
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)
    elem_size = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8
    symm_buffer_ptr = tl.cast(symm_buffer_ptr, input_ptr.dtype)

    if pid == 0:
        offs = tl.arange(0, world_size)
        tl.store(symm_signal_ptr + offs, 0)
        libshmem_device.barrier_all_block()

    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    for peer in range(pid, world_size, num_pid):
        libshmem_device.putmem_signal_block(
            symm_buffer_ptr + n_elements * rank,
            input_ptr,
            n_elements * elem_size,
            symm_signal_ptr + rank,
            1,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )

    if thread_idx < world_size:
        libshmem_device.signal_wait_until(symm_signal_ptr + thread_idx, libshmem_device.NVSHMEM_CMP_EQ, 1)
    __syncthreads()


def sp_decode_allgather_push_intra_node(partial: torch.Tensor, symm_buffer: torch.Tensor, symm_signal: torch.Tensor,
                                        grid_barrier: torch.Tensor, rank: int, num_ranks: int) -> torch.Tensor:
    """All-gather ``partial`` from every rank on the current stream.

    partial: contiguous local tensor, same shape on every rank.
    symm_buffer: 1-D symmetric tensor of ``partial.dtype`` with at least ``num_ranks * partial.numel()`` elements.
    symm_signal: symmetric ``NVSHMEM_SIGNAL_DTYPE`` tensor with ``num_ranks`` elements.
    grid_barrier: int32 device tensor with one element, zero-initialized.

    Returns a ``[num_ranks, *partial.shape]`` view of ``symm_buffer``. It stays valid
    until the next call on the same buffers.
    """
    assert partial.is_cuda and partial.is_contiguous()
    assert symm_buffer.dim() == 1 and symm_buffer.dtype == partial.dtype
    assert num_ranks & (num_ranks - 1) == 0, f"num_ranks ({num_ranks}) must be a power of two"
    n_elements = partial.numel()
    assert symm_buffer.numel() >= n_elements * num_ranks
    assert symm_signal.numel() >= num_ranks

    cooperative_options = launch_cooperative_grid_options()
    kernel_sp_decode_allgather_push_intra_node[(num_ranks, )](
        partial,
        symm_buffer,
        symm_signal,
        grid_barrier,
        rank,
        num_ranks,
        n_elements,
        use_cooperative=len(cooperative_options) > 0,
        num_warps=32,
        **cooperative_options,
    )
    return symm_buffer[:n_elements * num_ranks].view(num_ranks, *partial.shape)


def mla_fwd_batch_decode_intra_rank(q: torch.Tensor, kv_cache: torch.Tensor, kv_lens: torch.Tensor,
                                    block_table: torch.Tensor, scale: float, kv_lora_rank: int, soft_cap: float = 0.0,
                                    output_split: torch.Tensor = None, output_combine: torch.Tensor = None,
                                    kv_split: int = -1, block_n: int = 32, num_stages: int = 2) -> torch.Tensor:
    """Local MLA decode partial over this rank's KV shard.

    q: [batch, num_heads, kv_lora_rank + qk_rope_head_dim], the absorbed query.
    kv_cache: [num_pages, page_size, 1, kv_lora_rank + qk_rope_head_dim]. The first
        ``kv_lora_rank`` channels are the compressed KV and double as the value.
    kv_lens: [batch] int32, this rank's KV length per sequence.
    block_table: [batch, max_pages_per_seq] int32 into this rank's ``kv_cache``.

    Returns ``output_combine`` of shape [batch, num_heads, kv_lora_rank + 1]: the
    normalized partial output followed by its log-sum-exp. Rows with
    ``kv_lens == 0`` are undefined and must be skipped by the caller.

    ``block_n`` (tokens per tile) and ``num_stages`` are exposed because the GQA
    defaults were picked for 128-wide heads; with 576-wide keys a smaller tile
    is the safer starting point. Tune them per GPU.
    """
    assert kv_cache.dim() == 4 and kv_cache.shape[2] == 1, "MLA uses a single latent KV head"
    assert q.shape[-1] == kv_cache.shape[-1]
    assert 0 < kv_lora_rank < q.shape[-1]
    v_cache = kv_cache[..., :kv_lora_rank]
    return gqa_fwd_batch_decode_intra_rank(q, kv_cache, v_cache, None, [1] * q.shape[0], kv_lens, block_table, scale,
                                           soft_cap=soft_cap, output_split=output_split, output_combine=output_combine,
                                           kv_split=kv_split, block_n=block_n, num_stages=num_stages)


def mla_fwd_batch_decode_inter_rank_combine(all_partials: torch.Tensor, global_kv_lens: torch.Tensor,
                                            output: torch.Tensor) -> torch.Tensor:
    """Combine the gathered per-rank partials into the final output.

    all_partials: [num_ranks, batch, num_heads, kv_lora_rank + 1].
    global_kv_lens: [num_ranks, batch] int32, contiguous. Ranks with zero length for a
        sequence are skipped. At least one rank must hold tokens for every sequence.
    output: [batch, num_heads, kv_lora_rank].
    """
    num_ranks, batch, num_heads, v_dim_plus_one = all_partials.shape
    v_dim = v_dim_plus_one - 1
    assert global_kv_lens.shape == (num_ranks, batch) and global_kv_lens.is_contiguous()
    assert output.shape == (batch, num_heads, v_dim)
    kernel_inter_rank_gqa_fwd_batch_decode_combine_kv[(batch, num_heads, 1)](
        all_partials, output, global_kv_lens, batch, num_heads, all_partials.stride(1),  # batch
        all_partials.stride(2),  # head
        all_partials.stride(0),  # num_ranks
        output.stride(0),  # batch
        output.stride(1),  # head
        num_ranks,  # NUM_KV_SPLITS
        triton.next_power_of_2(v_dim),  # BLOCK_DV
        v_dim,  # V_DIM
    )
    return output
