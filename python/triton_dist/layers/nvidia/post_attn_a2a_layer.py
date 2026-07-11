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
"""Post-attention AllToAll operator using warp-specialized TMA pipeline.

Data layout
-----------
  Input:  (seq_len, local_nh, hd)  -- full sequence, partial heads per rank
  Output: (local_seq, nh, hd)      -- partial sequence, full heads per rank

Uses a 2-warp kernel (producer TMA G2S + consumer TMA S2G) with multi-stage
pipeline.  Aligned with Flux PostAttnA2AOp ``use_tma_store`` path.
"""
import torch
import torch.distributed
import triton

from triton_dist.utils import (
    nvshmem_create_tensor,
    nvshmem_free_tensor_sync,
    nvshmem_barrier_all_on_stream,
)
from triton_dist.tma_utils import create_tmap_scratch
from triton_dist.kernels.nvidia.post_attn_a2a import kernel_post_attn_a2a

_SMEM_CAPACITY = 228 * 1024


def _compute_stages(bm, bn, elem_bytes=2):
    """Max pipeline stages that fit in shared memory (minus barrier overhead)."""
    mbar_overhead = 1024
    bytes_per_stage = bm * bn * elem_bytes
    return (_SMEM_CAPACITY - mbar_overhead) // bytes_per_stage


def _select_tile_bn(target_n, max_bn=128, min_bn=64):
    """Pick the largest BN <= max_bn that divides target_n."""
    bn = min(max_bn, target_n)
    while target_n % bn != 0 and bn > min_bn:
        bn //= 2
    assert target_n % bn == 0, (f"Cannot find BN in [{min_bn}, {max_bn}] that divides {target_n}")
    return bn


class PostAttnA2AOp:
    """Post-attention AllToAll via warp-specialized TMA pipeline."""

    def __init__(self, sp_group, max_seq_len, nheads, head_dim, dtype=torch.bfloat16):
        self.sp_group = sp_group
        self.max_seq_len = max_seq_len
        self.nheads = nheads
        self.head_dim = head_dim
        self.dtype = dtype
        self.world_size = sp_group.size()
        self.rank = sp_group.rank()

        assert nheads % self.world_size == 0, (f"nheads ({nheads}) must be divisible by world_size ({self.world_size})")
        assert max_seq_len % self.world_size == 0, (
            f"max_seq_len ({max_seq_len}) must be divisible by world_size ({self.world_size})")

        self.local_nh = nheads // self.world_size
        self.max_local_seq = max_seq_len // self.world_size

        self._comm_output_buf = nvshmem_create_tensor(
            (self.max_local_seq, self.nheads, self.head_dim),
            dtype=self.dtype,
        )

        torch.distributed.barrier(group=self.sp_group)
        torch.cuda.synchronize()

    def _canonicalize(self, inp):
        if inp.ndim == 4:
            assert inp.size(0) == 1, "batch must be 1 for 4D"
            inp = inp.squeeze(0)
        assert inp.ndim == 3
        assert inp.dtype == self.dtype and inp.is_cuda
        assert inp.size(1) == self.local_nh and inp.size(2) == self.head_dim
        seq = inp.size(0)
        assert seq <= self.max_seq_len and seq % self.world_size == 0
        return inp.contiguous()

    def finalize(self):
        nvshmem_free_tensor_sync(self._comm_output_buf)

    def forward(self, input_tensor, *, output=None, return_comm_buf=False, num_sm=16):
        inp3d = self._canonicalize(input_tensor)
        seq = inp3d.size(0)
        local_seq = seq // self.world_size

        nvshmem_barrier_all_on_stream()

        BM = 128
        SRC_N = self.local_nh * self.head_dim
        BN = _select_tile_bn(SRC_N)
        NUM_STAGES = _compute_stages(BM, BN)
        assert NUM_STAGES >= 2, (f"BM={BM} BN={BN} yields only {NUM_STAGES} stages; need >=2 for pipeline")
        assert num_sm >= self.world_size, (f"num_sm ({num_sm}) must be >= world_size ({self.world_size})")
        PIPE_CNT = min(2, NUM_STAGES - 1)

        tmap_scratch = create_tmap_scratch(num_sm, num_descs_per_cta=2)

        def alloc_fn(size, alignment, stream):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        kernel_post_attn_a2a[(num_sm, )](
            inp3d,
            self._comm_output_buf,
            tmap_scratch,
            local_seq,
            self.local_nh,
            self.head_dim,
            self.world_size,
            self.rank,
            self.rank,
            BM,
            BN,
            NUM_STAGES,
            num_sm,
            PIPE_CNT,
            num_warps=2,
        )

        nvshmem_barrier_all_on_stream()
        view = self._comm_output_buf.narrow(0, 0, local_seq)
        if return_comm_buf:
            return view
        out = output if output is not None else torch.empty_like(view)
        out.copy_(view)
        return out
