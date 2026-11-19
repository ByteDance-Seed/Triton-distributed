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
import sys
import math
import ctypes
import torch

# Define NEG_INF as a constant for use in DSL
NEG_INF = -math.inf


# FastDivmod helper class for efficient integer division
# Uses multiply-high + shift instead of expensive integer division
class FastDivmod:
    """Precompute multiplier and shift for fast division."""

    def __init__(self, divisor: int):
        self.divisor = divisor
        if divisor == 1:
            self.multiplier = 0
            self.shift_right = 0
        else:
            # Find ceil(log2(d))
            log2_ceil = 0
            d_unsigned = divisor
            while (1 << log2_ceil) < d_unsigned:
                log2_ceil += 1

            # p = 31 + ceil(log2(d))
            p = 31 + log2_ceil

            # Compute multiplier = ceil(2^p / d)
            self.multiplier = ((1 << p) + d_unsigned - 1) // d_unsigned
            self.shift_right = p - 32


# Add little_kernel to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'python'))

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_3d_descriptor
from little_kernel.language.builtin_base import builtin, Builtin, BuiltinOp


# ============================================================================
# Custom integer builtins for Flash Attention.
# ============================================================================
@builtin
class umulhi(BuiltinOp):
    """Unsigned multiply high - returns high 32 bits of a*b.

    Used for FastDivmod optimization to avoid expensive integer division.

    Usage in kernel:
        result = umulhi(a, b)  # where a, b are uint32
    """

    @staticmethod
    def return_type(a, b):
        """Return type is always uint32."""
        return ll.uint32

    @staticmethod
    def codegen(a_cpp, b_cpp):
        """Generate __umulhi(a, b) intrinsic call."""
        return Builtin(body="", includes=[], return_val=f"__umulhi({a_cpp}, {b_cpp})")


@builtin
class fast_div(BuiltinOp):
    """Fast integer division using precomputed multiplier and shift.

    Implements: (umulhi(dividend, multiplier) >> shift_right)

    This is used with FastDivmod class which precomputes multiplier and shift
    on the host side for efficient division on the device.

    Usage in kernel:
        # Host side: fd = FastDivmod(divisor)
        # Device side: quotient = fast_div(dividend, fd.multiplier, fd.shift_right)
    """

    @staticmethod
    def return_type(dividend, multiplier, shift_right):
        """Return type is always int32."""
        return ll.int32

    @staticmethod
    def codegen(dividend_cpp, multiplier_cpp, shift_right_cpp):
        """Generate fast division expression using __umulhi."""
        return Builtin(
            body="", includes=[],
            # FastDivmod encodes division by one with the multiplier-zero
            # sentinel. In MHA this must retain the Q head index.
            return_val=f"(({multiplier_cpp}) == 0u ? (unsigned int)({dividend_cpp}) : "
            f"(__umulhi((unsigned int)({dividend_cpp}), {multiplier_cpp}) >> {shift_right_cpp}))")


# ============================================================================
# Constants (matching flash_attn_varlen.cu exactly)
# ============================================================================
kStages = 2
kBlockM = 128
kBlockN = 128
kHeadDim = 128

# Thread configuration
kWarpSize = 32
kWarpsPerWarpGroup = 4
kThreadsPerWarpGroup = kWarpSize * kWarpsPerWarpGroup  # 128
kNumMmaWarpGroups = 2
kNumMmaThreads = kNumMmaWarpGroups * kThreadsPerWarpGroup  # 256
kNumThreads = kNumMmaThreads + kThreadsPerWarpGroup  # 384

# Shared memory sizes
kSmemQSize = kBlockM * kHeadDim  # 128 * 128 = 16K elements
kSmemKSize = kBlockN * kHeadDim * kStages  # 128 * 128 * 2 = 32K elements
kSmemVSize = kHeadDim * kBlockN * kStages  # 128 * 128 * 2 = 32K elements
kSmemOSize = kBlockM * kHeadDim  # 128 * 128 = 16K elements

# TMA transaction bytes
kTmaTileCols = 64  # Max for 128B swizzle with bf16
kTmaTransactionBytesQ = kBlockM * kTmaTileCols * 2  # 16KB per call
kTmaTransactionBytesK = kBlockN * kTmaTileCols * 2  # 16KB per call
kTmaTransactionBytesV = kBlockN * kTmaTileCols * 2  # 16KB per call

# Named barrier IDs
kBarrierConsumerDone = 8
kBarrierProducerDone = 9
kBarrierEpilogue = 10
kBarrierWarpSchedulerWG1 = 11
kBarrierWarpSchedulerWG2 = 12

# Number of scheduler threads
kNumSchedulerThreads = kWarpSize + kNumMmaThreads  # 32 + 256 = 288

# Epilogue constants
kNumEpilogueThreads = 256

# TMA cache hint
kCacheHintEvictLast = 0x14F0000000000000

# WGMMA descriptor constants
WGMMA_K = 16
DESC_Q_HI = (64 << 0) | (1 << 30)  # stride=64, swizzle=128B
DESC_K_HI = (64 << 0) | (1 << 30)  # stride=64, swizzle=128B

# Offset constants (in units of 16 bytes)
K_STRIDE = (16 * 2) >> 4  # 16 elements = 32 bytes = 2 units
STAGE_STRIDE = (kBlockN * kHeadDim * 2) >> 4  # Stage offset
Q_SECOND_HALF = (kBlockM * 64 * 2) >> 4  # Second half of Q
K_SECOND_HALF = (kBlockM * 64 * 2) >> 4  # Second half of K

# Precomputed col_offset table for mask application
COL_OFFSETS = [
    0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 64, 65, 72, 73, 80, 81, 88, 89, 96, 97, 104, 105, 112,
    113, 120, 121
]

# Shared memory total size
SMEM_TOTAL = (kSmemQSize + kSmemKSize + kSmemVSize + kSmemOSize) * 2 + 1024

passes = PASSES["cuda"]

# ============================================================================
# Helper functions using existing ll intrinsics
# ============================================================================


# pack_acc: Pack two float32 values into uint32 as bf16x2
# Uses ll.pack_bf16 which takes float bits as uint32
def pack_acc_helper(acc1, acc2):
    """Helper to pack two float32 values into bf16x2 uint32."""
    return ll.pack_bf16(ll.float_as_uint(acc1), ll.float_as_uint(acc2))


# ============================================================================
# Kernel definition
# ============================================================================


@lk.ll_kernel(backend="cuda", is_entry=True, launch_bounds=kNumThreads)
def flash_fwd_varlen_kernel(
    # TMA descriptors
    tma_Q: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    tma_K: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    tma_V: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    # Tensor pointers
    ptr_O: ll.Tensor[ll.bfloat16],
    # Sequence info
    cu_seqlens_q: ll.Tensor[ll.int32],
    cu_seqlens_k: ll.Tensor[ll.int32],
    # Dimensions
    total_q: ll.int32,
    total_k: ll.int32,
    max_seqlen_q: ll.int32,
    max_seqlen_k: ll.int32,
    batch: ll.int32,
    nheads: ll.int32,
    nkv_heads: ll.int32,
    headdim: ll.int32,
    # Causal mask
    is_causal: ll.int32,
    # Softmax
    softmax_scale_log2: ll.float32,
    # Strides
    stride_O_row: ll.int64,
    # Scheduler
    tile_count_semaphore: ll.Tensor[ll.int32],
    # FastDivmod for GQA: kv_head_idx = bidh / (nheads / nkv_heads)
    qhead_per_khead_multiplier: ll.uint32,
    qhead_per_khead_shift: ll.int32,
) -> ll.void:
    """Flash Attention 3 forward kernel - direct translation from CUDA."""

    # ========== Shared memory allocation ==========
    # Use single large arrays instead of pointer arrays to avoid stack usage
    ll.align_memory(128, scope="dynamic_shared")
    smem_q = ll.empty([kBlockM, kHeadDim], dtype=ll.bfloat16, scope="dynamic_shared")
    # Single array for K with kStages stages - access via smem_k_base + stage * kBlockN * kHeadDim
    smem_k_base = ll.empty([kBlockN * kHeadDim * kStages], dtype=ll.bfloat16, scope="dynamic_shared")
    # Single array for V with kStages stages
    smem_v_base = ll.empty([kBlockN * kHeadDim * kStages], dtype=ll.bfloat16, scope="dynamic_shared")
    smem_o = ll.empty([kBlockM, kHeadDim], dtype=ll.bfloat16, scope="dynamic_shared")

    # Barriers - use single arrays with 2 elements each
    full_barrier_K_base = ll.empty([kStages], dtype=ll.uint64, scope="dynamic_shared")
    empty_barrier_K_base = ll.empty([kStages], dtype=ll.uint64, scope="dynamic_shared")
    full_barrier_V_base = ll.empty([kStages], dtype=ll.uint64, scope="dynamic_shared")
    empty_barrier_V_base = ll.empty([kStages], dtype=ll.uint64, scope="dynamic_shared")
    full_barrier_Q = ll.empty([1], dtype=ll.uint64, scope="dynamic_shared")
    empty_barrier_Q = ll.empty([1], dtype=ll.uint64, scope="dynamic_shared")

    # Work info (int4 packed)
    work_info = ll.empty([4], dtype=ll.int32, scope="dynamic_shared")

    # Note: 16384 = kBlockN * kHeadDim = 16384 elements per stage
    # We use the literal value 16384 directly in the code to help compiler optimization

    # ========== Thread indexing ==========
    thread_idx: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = thread_idx >> 5
    lane_idx: ll.int32 = thread_idx & 31
    warp_group_idx: ll.int32 = thread_idx >> 7
    warp_idx_in_warpgroup: ll.int32 = warp_idx & 3

    is_warp0_lane0: ll.int32 = 1 if ((warp_idx == 0) & (lane_idx == 0)) else 0

    # ========== Barrier initialization ==========
    if is_warp0_lane0 == 1:
        ll.init_smem_barrier(full_barrier_Q, 1)
        ll.init_smem_barrier(empty_barrier_Q, kNumMmaWarpGroups)
        for s in ll.unroll(range(kStages)):
            ll.init_smem_barrier(full_barrier_K_base + s, 1)
            ll.init_smem_barrier(full_barrier_V_base + s, 1)
            ll.init_smem_barrier(empty_barrier_K_base + s, kNumMmaWarpGroups)
            ll.init_smem_barrier(empty_barrier_V_base + s, kNumMmaWarpGroups)
        ll.fence_smem_barrier_init()

    # Prefetch TMA descriptors
    if is_warp0_lane0 == 1:
        ll.prefetch_tma_descriptor(tma_Q)
        ll.prefetch_tma_descriptor(tma_K)
        ll.prefetch_tma_descriptor(tma_V)

    ll.__syncthreads()

    # ========== Producer path (warp_group_idx == 0) ==========
    if warp_group_idx == 0:
        ll.warpgroup_reg_dealloc(24)

        # Only warp 0 in producer warpgroup participates
        if warp_idx_in_warpgroup == 0:
            is_leader: ll.int32 = 1 if (thread_idx == 0) else 0

            # Get initial tile
            tile_idx: ll.int32 = ll.val_cast(ll.blockIdx_x(), ll.int32)

            # Pipeline state
            pipe_k_index: ll.int32 = 0
            pipe_k_phase: ll.int32 = 1
            pipe_v_index: ll.int32 = 0
            pipe_v_phase: ll.int32 = 1

            # Decode initial tile using warp-level parallelism (FA3-style)
            bidb: ll.int32 = 0
            bidh: ll.int32 = 0
            m_block: ll.int32 = 0

            # Fast tile decode using warp prefix sum
            # Each warp processes 31 batches at a time (lanes 0-30, lane 31 unused)
            group_start_tile: ll.int32 = 0

            # Each lane reads num_m_blocks for batch (bidb + lane)
            batch_idx_lane: ll.int32 = lane_idx + bidb
            num_m_blocks_lane: ll.int32 = 0
            if (batch_idx_lane < batch) & (lane_idx < 31):
                seqlen_q_lane: ll.int32 = cu_seqlens_q[batch_idx_lane + 1] - cu_seqlens_q[batch_idx_lane]
                num_m_blocks_lane = (seqlen_q_lane + kBlockM - 1) >> 7

            # Warp prefix sum using shfl_up (unrolled manually for powers of 2)
            num_m_blocks_cumulative: ll.int32 = num_m_blocks_lane
            partial_sum_1: ll.int32 = ll.shfl_up_sync(0xffffffff, num_m_blocks_cumulative, 1)
            if lane_idx >= 1:
                num_m_blocks_cumulative = num_m_blocks_cumulative + partial_sum_1
            partial_sum_2: ll.int32 = ll.shfl_up_sync(0xffffffff, num_m_blocks_cumulative, 2)
            if lane_idx >= 2:
                num_m_blocks_cumulative = num_m_blocks_cumulative + partial_sum_2
            partial_sum_4: ll.int32 = ll.shfl_up_sync(0xffffffff, num_m_blocks_cumulative, 4)
            if lane_idx >= 4:
                num_m_blocks_cumulative = num_m_blocks_cumulative + partial_sum_4
            partial_sum_8: ll.int32 = ll.shfl_up_sync(0xffffffff, num_m_blocks_cumulative, 8)
            if lane_idx >= 8:
                num_m_blocks_cumulative = num_m_blocks_cumulative + partial_sum_8
            partial_sum_16: ll.int32 = ll.shfl_up_sync(0xffffffff, num_m_blocks_cumulative, 16)
            if lane_idx >= 16:
                num_m_blocks_cumulative = num_m_blocks_cumulative + partial_sum_16

            m_blocks_in_group: ll.int32 = ll.__shfl_sync(0xffffffff, num_m_blocks_cumulative, 31)
            group_end_tile: ll.int32 = group_start_tile + m_blocks_in_group * nheads

            # Skip groups of 31 batches until we find the one containing tile_idx
            while group_end_tile <= tile_idx:
                bidb = bidb + 31
                if bidb >= batch:
                    bidb = batch
                    break
                group_start_tile = group_end_tile

                # Reload for new group
                batch_idx_lane = lane_idx + bidb
                num_m_blocks_lane = 0
                if (batch_idx_lane < batch) & (lane_idx < 31):
                    seqlen_q_lane = cu_seqlens_q[batch_idx_lane + 1] - cu_seqlens_q[batch_idx_lane]
                    num_m_blocks_lane = (seqlen_q_lane + kBlockM - 1) >> 7

                # Warp prefix sum (unrolled manually)
                num_m_blocks_cumulative = num_m_blocks_lane
                partial_sum_1 = ll.shfl_up_sync(0xffffffff, num_m_blocks_cumulative, 1)
                if lane_idx >= 1:
                    num_m_blocks_cumulative = num_m_blocks_cumulative + partial_sum_1
                partial_sum_2 = ll.shfl_up_sync(0xffffffff, num_m_blocks_cumulative, 2)
                if lane_idx >= 2:
                    num_m_blocks_cumulative = num_m_blocks_cumulative + partial_sum_2
                partial_sum_4 = ll.shfl_up_sync(0xffffffff, num_m_blocks_cumulative, 4)
                if lane_idx >= 4:
                    num_m_blocks_cumulative = num_m_blocks_cumulative + partial_sum_4
                partial_sum_8 = ll.shfl_up_sync(0xffffffff, num_m_blocks_cumulative, 8)
                if lane_idx >= 8:
                    num_m_blocks_cumulative = num_m_blocks_cumulative + partial_sum_8
                partial_sum_16 = ll.shfl_up_sync(0xffffffff, num_m_blocks_cumulative, 16)
                if lane_idx >= 16:
                    num_m_blocks_cumulative = num_m_blocks_cumulative + partial_sum_16

                m_blocks_in_group = ll.__shfl_sync(0xffffffff, num_m_blocks_cumulative, 31)
                group_end_tile = group_start_tile + m_blocks_in_group * nheads

            # Now tile_idx is within [group_start_tile, group_end_tile)
            # Use ballot to find which batch within the group
            ballot_pred: ll.int32 = 1 if (group_start_tile + num_m_blocks_cumulative * nheads <= tile_idx) else 0
            batch_idx_in_group: ll.int32 = ll.popc(ll.ballot_sync(0xffffffff, ballot_pred))

            bidb = bidb + batch_idx_in_group
            num_m_blocks: ll.int32 = ll.__shfl_sync(0xffffffff, num_m_blocks_lane, batch_idx_in_group)

            # Compute tile offset within this batch
            batch_start_tile: ll.int32 = group_start_tile
            if batch_idx_in_group > 0:
                prev_cumsum: ll.int32 = ll.__shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1)
                batch_start_tile = batch_start_tile + prev_cumsum * nheads

            tile_in_batch: ll.int32 = tile_idx - batch_start_tile
            bidh = tile_in_batch // num_m_blocks
            m_block = tile_in_batch - bidh * num_m_blocks

            # Write initial work info
            if is_leader == 1:
                work_info[0] = tile_idx
                work_info[1] = m_block
                work_info[2] = bidh
                work_info[3] = bidb

            # Signal consumers
            ll.named_barrier_arrive(kBarrierProducerDone, kNumSchedulerThreads)

            # Main producer loop
            q_phase_producer: ll.int32 = 0
            prev_tile_loaded_q: ll.int32 = 0

            while True:
                # Read current work info (set by previous iteration or initial setup)
                cur_m_block: ll.int32 = work_info[1]
                cur_bidh: ll.int32 = work_info[2]
                cur_bidb: ll.int32 = work_info[3]

                if cur_bidb >= batch:
                    break

                # Get sequence lengths
                offset_q: ll.int32 = cu_seqlens_q[cur_bidb]
                offset_k: ll.int32 = cu_seqlens_k[cur_bidb]
                seqlen_q: ll.int32 = cu_seqlens_q[cur_bidb + 1] - offset_q
                seqlen_k: ll.int32 = cu_seqlens_k[cur_bidb + 1] - offset_k

                # Compute n_block range
                n_block_max: ll.int32 = (seqlen_k + kBlockN - 1) >> 7
                if is_causal == 1:
                    m_idx_max: ll.int32 = (cur_m_block + 1) << 7
                    n_idx_max: ll.int32 = m_idx_max + seqlen_k - seqlen_q
                    n_block_max_causal: ll.int32 = (n_idx_max + kBlockN - 1) >> 7
                    n_block_max = ll.min_val(n_block_max, n_block_max_causal)

                n_blocks: ll.int32 = n_block_max

                # Prefetch next tile
                if is_leader == 1:
                    tile_idx = ll.atomic_add(tile_count_semaphore, 1) + ll.val_cast(ll.gridDim_x(), ll.int32)
                tile_idx = ll.__shfl_sync(0xffffffff, tile_idx, 0)

                # Compute KV head index for GQA using FastDivmod (matches CUDA reference exactly)
                # kv_head_idx = cur_bidh / (nheads / nkv_heads)
                # Use precomputed multiplier and shift for efficient division
                kv_head_idx: ll.int32 = fast_div(cur_bidh, qhead_per_khead_multiplier, qhead_per_khead_shift)

                if n_blocks > 0:
                    # Wait for Q buffer to be free
                    if prev_tile_loaded_q == 1:
                        ll.mbarrier_wait(empty_barrier_Q, q_phase_producer)
                        q_phase_producer = q_phase_producer ^ 1

                    # Load Q using TMA
                    if is_leader == 1:
                        ll.mbarrier_arrive_and_expect_tx(full_barrier_Q, kTmaTransactionBytesQ * 2)
                        ll.tma_load_3d_1sm(tma_Q, full_barrier_Q, smem_q, 0, offset_q + cur_m_block * kBlockM, cur_bidh)
                        ll.tma_load_3d_1sm(tma_Q, full_barrier_Q, smem_q + kBlockM * kTmaTileCols, kTmaTileCols,
                                           offset_q + cur_m_block * kBlockM, cur_bidh)

                    # Load K/V with pipeline
                    n_block_prev: ll.int32 = n_block_max - 1

                    # First K
                    if n_blocks > 0:
                        n_block: ll.int32 = n_block_max - 1
                        k_stage: ll.int32 = pipe_k_index

                        if is_leader == 1:
                            ll.mbarrier_arrive_and_expect_tx(full_barrier_K_base + k_stage, kTmaTransactionBytesK * 2)
                            smem_k_stage = smem_k_base + k_stage * 16384
                            ll.tma_load_3d_1sm(tma_K, full_barrier_K_base + k_stage, smem_k_stage, 0,
                                               offset_k + n_block * kBlockN, kv_head_idx, kCacheHintEvictLast)
                            ll.tma_load_3d_1sm(tma_K, full_barrier_K_base + k_stage,
                                               smem_k_stage + kBlockN * kTmaTileCols, kTmaTileCols,
                                               offset_k + n_block * kBlockN, kv_head_idx, kCacheHintEvictLast)

                        n_block_prev = n_block
                        pipe_k_index = pipe_k_index + 1
                        if pipe_k_index == kStages:
                            pipe_k_index = 0
                            pipe_k_phase = pipe_k_phase ^ 1

                    # Middle iterations: K[i] and V[i-1]
                    i: ll.int32 = 1
                    while i < n_blocks:
                        n_block = n_block_max - 1 - i
                        k_stage = pipe_k_index
                        v_stage: ll.int32 = pipe_v_index

                        if i >= kStages:
                            ll.mbarrier_wait(empty_barrier_K_base + k_stage, pipe_k_phase)
                        if i - 1 >= kStages:
                            ll.mbarrier_wait(empty_barrier_V_base + v_stage, pipe_v_phase)

                        if is_leader == 1:
                            # Load K[i]
                            ll.mbarrier_arrive_and_expect_tx(full_barrier_K_base + k_stage, kTmaTransactionBytesK * 2)
                            smem_k_stage = smem_k_base + k_stage * 16384
                            ll.tma_load_3d_1sm(tma_K, full_barrier_K_base + k_stage, smem_k_stage, 0,
                                               offset_k + n_block * kBlockN, kv_head_idx, kCacheHintEvictLast)
                            ll.tma_load_3d_1sm(tma_K, full_barrier_K_base + k_stage,
                                               smem_k_stage + kBlockN * kTmaTileCols, kTmaTileCols,
                                               offset_k + n_block * kBlockN, kv_head_idx, kCacheHintEvictLast)

                            # Load V[i-1]
                            ll.mbarrier_arrive_and_expect_tx(full_barrier_V_base + v_stage, kTmaTransactionBytesV * 2)
                            smem_v_stage = smem_v_base + v_stage * 16384
                            ll.tma_load_3d_1sm(tma_V, full_barrier_V_base + v_stage, smem_v_stage, 0,
                                               offset_k + n_block_prev * kBlockN, kv_head_idx, kCacheHintEvictLast)
                            ll.tma_load_3d_1sm(tma_V, full_barrier_V_base + v_stage,
                                               smem_v_stage + kBlockN * kTmaTileCols, kTmaTileCols,
                                               offset_k + n_block_prev * kBlockN, kv_head_idx, kCacheHintEvictLast)

                        n_block_prev = n_block
                        pipe_k_index = pipe_k_index + 1
                        if pipe_k_index == kStages:
                            pipe_k_index = 0
                            pipe_k_phase = pipe_k_phase ^ 1
                        pipe_v_index = pipe_v_index + 1
                        if pipe_v_index == kStages:
                            pipe_v_index = 0
                            pipe_v_phase = pipe_v_phase ^ 1
                        i = i + 1

                    # Final V
                    if n_blocks > 0:
                        v_stage = pipe_v_index
                        if n_blocks - 1 >= kStages:
                            ll.mbarrier_wait(empty_barrier_V_base + v_stage, pipe_v_phase)

                        if is_leader == 1:
                            ll.mbarrier_arrive_and_expect_tx(full_barrier_V_base + v_stage, kTmaTransactionBytesV * 2)
                            smem_v_stage = smem_v_base + v_stage * 16384
                            ll.tma_load_3d_1sm(tma_V, full_barrier_V_base + v_stage, smem_v_stage, 0,
                                               offset_k + n_block_prev * kBlockN, kv_head_idx, kCacheHintEvictLast)
                            ll.tma_load_3d_1sm(tma_V, full_barrier_V_base + v_stage,
                                               smem_v_stage + kBlockN * kTmaTileCols, kTmaTileCols,
                                               offset_k + n_block_prev * kBlockN, kv_head_idx, kCacheHintEvictLast)

                        pipe_v_index = pipe_v_index + 1
                        if pipe_v_index == kStages:
                            pipe_v_index = 0
                            pipe_v_phase = pipe_v_phase ^ 1

                    prev_tile_loaded_q = 1
                else:
                    prev_tile_loaded_q = 0

                # Wait for consumers
                ll.named_barrier_sync(kBarrierConsumerDone, kNumSchedulerThreads)

                # Decode next tile using warp prefix sum (same as initial decode)
                next_bidb: ll.int32 = 0
                next_bidh: ll.int32 = 0
                next_m_block: ll.int32 = 0
                next_group_start_tile: ll.int32 = 0

                # Each lane reads num_m_blocks for batch (next_bidb + lane)
                next_batch_idx_lane: ll.int32 = lane_idx + next_bidb
                next_num_m_blocks_lane: ll.int32 = 0
                if (next_batch_idx_lane < batch) & (lane_idx < 31):
                    next_seqlen_q_lane: ll.int32 = cu_seqlens_q[next_batch_idx_lane +
                                                                1] - cu_seqlens_q[next_batch_idx_lane]
                    next_num_m_blocks_lane = (next_seqlen_q_lane + kBlockM - 1) >> 7

                # Warp prefix sum
                next_num_m_blocks_cumulative: ll.int32 = next_num_m_blocks_lane
                next_partial_1: ll.int32 = ll.shfl_up_sync(0xffffffff, next_num_m_blocks_cumulative, 1)
                if lane_idx >= 1:
                    next_num_m_blocks_cumulative = next_num_m_blocks_cumulative + next_partial_1
                next_partial_2: ll.int32 = ll.shfl_up_sync(0xffffffff, next_num_m_blocks_cumulative, 2)
                if lane_idx >= 2:
                    next_num_m_blocks_cumulative = next_num_m_blocks_cumulative + next_partial_2
                next_partial_4: ll.int32 = ll.shfl_up_sync(0xffffffff, next_num_m_blocks_cumulative, 4)
                if lane_idx >= 4:
                    next_num_m_blocks_cumulative = next_num_m_blocks_cumulative + next_partial_4
                next_partial_8: ll.int32 = ll.shfl_up_sync(0xffffffff, next_num_m_blocks_cumulative, 8)
                if lane_idx >= 8:
                    next_num_m_blocks_cumulative = next_num_m_blocks_cumulative + next_partial_8
                next_partial_16: ll.int32 = ll.shfl_up_sync(0xffffffff, next_num_m_blocks_cumulative, 16)
                if lane_idx >= 16:
                    next_num_m_blocks_cumulative = next_num_m_blocks_cumulative + next_partial_16

                next_m_blocks_in_group: ll.int32 = ll.__shfl_sync(0xffffffff, next_num_m_blocks_cumulative, 31)
                next_group_end_tile: ll.int32 = next_group_start_tile + next_m_blocks_in_group * nheads

                # Skip groups until we find the one containing tile_idx
                while next_group_end_tile <= tile_idx:
                    next_bidb = next_bidb + 31
                    if next_bidb >= batch:
                        next_bidb = batch
                        break
                    next_group_start_tile = next_group_end_tile

                    next_batch_idx_lane = lane_idx + next_bidb
                    next_num_m_blocks_lane = 0
                    if (next_batch_idx_lane < batch) & (lane_idx < 31):
                        next_seqlen_q_lane = cu_seqlens_q[next_batch_idx_lane + 1] - cu_seqlens_q[next_batch_idx_lane]
                        next_num_m_blocks_lane = (next_seqlen_q_lane + kBlockM - 1) >> 7

                    next_num_m_blocks_cumulative = next_num_m_blocks_lane
                    next_partial_1 = ll.shfl_up_sync(0xffffffff, next_num_m_blocks_cumulative, 1)
                    if lane_idx >= 1:
                        next_num_m_blocks_cumulative = next_num_m_blocks_cumulative + next_partial_1
                    next_partial_2 = ll.shfl_up_sync(0xffffffff, next_num_m_blocks_cumulative, 2)
                    if lane_idx >= 2:
                        next_num_m_blocks_cumulative = next_num_m_blocks_cumulative + next_partial_2
                    next_partial_4 = ll.shfl_up_sync(0xffffffff, next_num_m_blocks_cumulative, 4)
                    if lane_idx >= 4:
                        next_num_m_blocks_cumulative = next_num_m_blocks_cumulative + next_partial_4
                    next_partial_8 = ll.shfl_up_sync(0xffffffff, next_num_m_blocks_cumulative, 8)
                    if lane_idx >= 8:
                        next_num_m_blocks_cumulative = next_num_m_blocks_cumulative + next_partial_8
                    next_partial_16 = ll.shfl_up_sync(0xffffffff, next_num_m_blocks_cumulative, 16)
                    if lane_idx >= 16:
                        next_num_m_blocks_cumulative = next_num_m_blocks_cumulative + next_partial_16

                    next_m_blocks_in_group = ll.__shfl_sync(0xffffffff, next_num_m_blocks_cumulative, 31)
                    next_group_end_tile = next_group_start_tile + next_m_blocks_in_group * nheads

                # Use ballot to find batch within group
                if next_bidb < batch:
                    next_ballot_pred: ll.int32 = 1 if (next_group_start_tile +
                                                       next_num_m_blocks_cumulative * nheads <= tile_idx) else 0
                    next_batch_idx_in_group: ll.int32 = ll.popc(ll.ballot_sync(0xffffffff, next_ballot_pred))

                    next_bidb = next_bidb + next_batch_idx_in_group
                    next_num_m_blocks: ll.int32 = ll.__shfl_sync(0xffffffff, next_num_m_blocks_lane,
                                                                 next_batch_idx_in_group)

                    next_batch_start_tile: ll.int32 = next_group_start_tile
                    if next_batch_idx_in_group > 0:
                        next_prev_cumsum: ll.int32 = ll.__shfl_sync(0xffffffff, next_num_m_blocks_cumulative,
                                                                    next_batch_idx_in_group - 1)
                        next_batch_start_tile = next_batch_start_tile + next_prev_cumsum * nheads

                    next_tile_in_batch: ll.int32 = tile_idx - next_batch_start_tile
                    next_bidh = next_tile_in_batch // next_num_m_blocks
                    next_m_block = next_tile_in_batch - next_bidh * next_num_m_blocks

                if is_leader == 1:
                    if next_bidb < batch:
                        work_info[0] = tile_idx
                        work_info[1] = next_m_block
                        work_info[2] = next_bidh
                        work_info[3] = next_bidb
                    else:
                        work_info[0] = 0
                        work_info[1] = 0
                        work_info[2] = 0
                        work_info[3] = batch

                # Signal consumers
                ll.named_barrier_arrive(kBarrierProducerDone, kNumSchedulerThreads)

    else:
        # ========== Consumer path (warp_group_idx >= 1) ==========
        ll.warpgroup_reg_alloc(240)

        is_warpgroup_leader: ll.int32 = 1 if ((thread_idx & 127) == 0) else 0

        # Pipeline state (separate for K and V)
        k_stage: ll.int32 = 0
        k_phase: ll.int32 = 0
        v_stage: ll.int32 = 0
        v_phase: ll.int32 = 0
        q_phase: ll.int32 = 0

        # Scheduler barrier init
        if warp_group_idx == 1:
            ll.named_barrier_arrive(kBarrierWarpSchedulerWG1, 2 * kThreadsPerWarpGroup)

        # Main consumer loop
        while True:
            # Wait for producer
            ll.named_barrier_sync(kBarrierProducerDone, kNumSchedulerThreads)

            # Read work info
            cur_bidb: ll.int32 = work_info[3]
            if cur_bidb >= batch:
                break

            cur_m_block: ll.int32 = work_info[1]
            cur_bidh: ll.int32 = work_info[2]

            # Get sequence lengths
            offset_q: ll.int32 = cu_seqlens_q[cur_bidb]
            offset_k: ll.int32 = cu_seqlens_k[cur_bidb]
            seqlen_q: ll.int32 = cu_seqlens_q[cur_bidb + 1] - offset_q
            seqlen_k: ll.int32 = cu_seqlens_k[cur_bidb + 1] - offset_k

            # Compute n_block range
            n_block_max: ll.int32 = (seqlen_k + kBlockN - 1) >> 7
            if is_causal == 1:
                m_idx_max: ll.int32 = (cur_m_block + 1) << 7
                n_idx_max: ll.int32 = m_idx_max + seqlen_k - seqlen_q
                n_block_max_causal: ll.int32 = (n_idx_max + kBlockN - 1) >> 7
                n_block_max = ll.min_val(n_block_max, n_block_max_causal)

            n_blocks: ll.int32 = n_block_max

            # Signal work info read
            ll.named_barrier_arrive(kBarrierConsumerDone, kNumSchedulerThreads)

            if n_blocks > 0:
                # Wait for Q (FA3-style two-step pattern)
                q_ready: ll.bool_ = ll.mbarrier_try_wait_parity(full_barrier_Q, q_phase)
                ll.mbarrier_wait_if_needed(full_barrier_Q, q_phase, q_ready)
                q_phase = q_phase ^ 1

                # Initialize accumulators
                # Use empty + manual init to match CUDA reference exactly
                # The compiler should be able to optimize this away since acc_o is first used
                # in PV WGMMA with scale_d=1 (accumulate mode)
                acc_o = ll.empty([64], dtype=ll.float32, scope="local")
                for _init_i in ll.unroll(range(64)):
                    acc_o[_init_i] = 0.0
                row_max = ll.full([2], dtype=ll.float32, scope="local", fill_value=NEG_INF)
                row_sum = ll.zeros([2], dtype=ll.float32, scope="local")

                # Precompute barrier IDs (FA3 optimization: use ternary to help compiler)
                # my_barrier: barrier this warpgroup syncs on
                # other_barrier: barrier this warpgroup arrives on (to signal other warpgroup)
                my_barrier: ll.int32 = kBarrierWarpSchedulerWG1 if warp_group_idx == 1 else kBarrierWarpSchedulerWG2
                other_barrier: ll.int32 = kBarrierWarpSchedulerWG2 if warp_group_idx == 1 else kBarrierWarpSchedulerWG1

                # Scheduler barrier sync
                ll.named_barrier_sync(my_barrier, 2 * kThreadsPerWarpGroup)

                # Thread layout constants
                thread_row_base: ll.int32 = ((thread_idx >> 2) & 7) | ((thread_idx >> 1) & 48)
                thread_col_offset: ll.int32 = (thread_idx & 3) << 1
                m_offset: ll.int32 = (warp_group_idx - 1) << 6

                # Descriptor bases
                base_desc_q_lo: ll.uint32 = ll.cvta_generic_to_shared(smem_q + m_offset * 64) >> 4
                base_desc_k_lo: ll.uint32 = (ll.cvta_generic_to_shared(smem_k_base) >> 4) | (1 << 16)

                # P matrix storage (use empty to avoid unnecessary initialization - matches CUDA)
                # acc_p_bf16 is initialized to 0 before first use, acc_s is set to 0 in each iteration
                acc_p_bf16 = ll.zeros([32], dtype=ll.uint32, scope="local")  # Need init for first iteration
                acc_s = ll.empty([64], dtype=ll.float32, scope="local")  # No init needed - set to 0 in loop
                prev_scores_scale = ll.full([2], dtype=ll.float32, scope="local", fill_value=1.0)

                # Causal mask boundary
                # Use arithmetic right shift instead of division for efficiency
                # Since max(0, ...) is applied, the result is the same:
                # - If n_idx_right >= 0: n_idx_right >> 7 == n_idx_right / 128
                # - If n_idx_right < 0: n_idx_right >> 7 < 0, max(0, ...) = 0
                n_idx_right: ll.int32 = cur_m_block * kBlockM + seqlen_k - seqlen_q
                n_block_min_causal_mask: ll.int32 = ll.max_val(0, n_idx_right >> 7)

                # ===== First iteration: QK + softmax only =====
                n_block: ll.int32 = n_block_max - 1
                first_k_stage: ll.int32 = k_stage
                first_k_phase: ll.int32 = k_phase

                # FA3-style two-step wait for K
                k_ready: ll.bool_ = ll.mbarrier_try_wait_parity(full_barrier_K_base + first_k_stage, first_k_phase)
                ll.mbarrier_wait_if_needed(full_barrier_K_base + first_k_stage, first_k_phase, k_ready)

                # Zero acc_s
                for j in ll.unroll(range(64)):
                    acc_s[j] = 0.0

                # Fence operands before WGMMA (FA3 optimization)
                ll.wgmma_fence_operand_array(acc_s, 64)

                # QK WGMMA
                ll.wgmma_fence()
                for ki in ll.unroll(range(4)):
                    k_off: ll.uint32 = ll.val_cast(ki, ll.uint32) * K_STRIDE
                    q_lo: ll.uint32 = base_desc_q_lo + k_off
                    k_lo: ll.uint32 = base_desc_k_lo + ll.val_cast(first_k_stage, ll.uint32) * STAGE_STRIDE + k_off
                    desc_q: ll.uint64 = ll.make_desc(q_lo, DESC_Q_HI)
                    desc_k: ll.uint64 = ll.make_desc(k_lo, DESC_K_HI)
                    scale_d: ll.int32 = 1 if (ki > 0) else 0
                    ll.wgmma_m64n128k16_ss(acc_s, desc_q, desc_k, scale_d)
                for ki in ll.unroll(range(4)):
                    k_off = ll.val_cast(ki, ll.uint32) * K_STRIDE
                    q_lo = base_desc_q_lo + Q_SECOND_HALF + k_off
                    k_lo = base_desc_k_lo + ll.val_cast(first_k_stage, ll.uint32) * STAGE_STRIDE + K_SECOND_HALF + k_off
                    desc_q = ll.make_desc(q_lo, DESC_Q_HI)
                    desc_k = ll.make_desc(k_lo, DESC_K_HI)
                    ll.wgmma_m64n128k16_ss(acc_s, desc_q, desc_k, 1)
                ll.wgmma_commit()

                # Fence operands after WGMMA commit (FA3 optimization)
                ll.wgmma_fence_operand_array(acc_s, 64)

                # Scheduler barrier arrive (signal other warpgroup)
                ll.named_barrier_arrive(other_barrier, 2 * kThreadsPerWarpGroup)

                ll.wgmma_wait()

                # Release K
                if is_warpgroup_leader == 1:
                    ll.mbarrier_arrive(empty_barrier_K_base + first_k_stage)
                # Update K pipeline state
                k_stage = k_stage + 1
                if k_stage == kStages:
                    k_stage = 0
                    k_phase = k_phase ^ 1

                # Apply mask (first iteration always needs mask for causal)
                # NOTE: This is the FIRST QK result, always apply mask (matches CUDA FWD_STEP with apply_mask_flag=true)
                seqlenk_col_limit: ll.int32 = seqlen_k - n_block * kBlockN - thread_col_offset
                causal_row_offset: ll.int32 = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset
                for mi in ll.unroll(range(2)):
                    row: ll.int32 = thread_row_base + mi * 8
                    row_idx: ll.int32 = cur_m_block * kBlockM + m_offset + row
                    # Causal col limit: min(row_idx + causal_row_offset, seqlenk_col_limit)
                    # Use __viaddmin_s32 for efficient single-instruction computation
                    # NOTE: Always use viaddmin_s32 here - this is the first iteration which always needs mask
                    col_limit_right: ll.int32 = ll.viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit)
                    row_oob: ll.bool_ = row_idx >= seqlen_q
                    for ni in ll.unroll(range(32)):
                        reg: ll.int32 = mi * 2 + (ni & 1) + ((ni >> 1) << 2)
                        # col_offset = (ni >> 1) * 8 + (ni & 1)
                        col_offset: ll.int32 = ((ni >> 1) << 3) + (ni & 1)
                        # Use || (logical or) to combine conditions - matches CUDA's || behavior
                        if row_oob | (col_offset >= col_limit_right):
                            acc_s[reg] = ll.val_cast(NEG_INF, ll.float32)

                # Softmax (first iteration)
                local_max = ll.full([2], dtype=ll.float32, scope="local", fill_value=NEG_INF)
                for j in ll.unroll(range(64)):
                    mi: ll.int32 = (j >> 1) & 1
                    local_max[mi] = ll.max_val(local_max[mi], acc_s[j])

                for mi in ll.unroll(range(2)):
                    local_max[mi] = ll.max_val(local_max[mi], ll.shfl_xor_sync(0xffffffff, local_max[mi], 2))
                    local_max[mi] = ll.max_val(local_max[mi], ll.shfl_xor_sync(0xffffffff, local_max[mi], 1))
                    row_max[mi] = local_max[mi]

                # FA3-style: precompute max_scaled to avoid redundant computation in loop
                max_scaled_0: ll.float32 = row_max[0] * softmax_scale_log2
                max_scaled_1: ll.float32 = row_max[1] * softmax_scale_log2

                # Exp and sum (FA3-style: ni outer, mi inner for better register reuse)
                for ni in ll.unroll(range(32)):
                    reg0: ll.int32 = 0 * 2 + (ni & 1) + ((ni >> 1) << 2)
                    reg1: ll.int32 = 1 * 2 + (ni & 1) + ((ni >> 1) << 2)
                    val0: ll.float32 = ll.exp2f(acc_s[reg0] * softmax_scale_log2 - max_scaled_0)
                    val1: ll.float32 = ll.exp2f(acc_s[reg1] * softmax_scale_log2 - max_scaled_1)
                    acc_s[reg0] = val0
                    acc_s[reg1] = val1
                    row_sum[0] = row_sum[0] + val0
                    row_sum[1] = row_sum[1] + val1

                # Store P[0] as bf16 (use pack_bf16_float for efficiency)
                for j in ll.unroll(range(16)):
                    j4: ll.int32 = j * 4
                    acc_p_bf16[j * 2] = ll.pack_bf16_float(acc_s[j4 + 0], acc_s[j4 + 1])
                    acc_p_bf16[j * 2 + 1] = ll.pack_bf16_float(acc_s[j4 + 2], acc_s[j4 + 3])

                # ===== Middle iterations: QK[i] + PV[i-1] =====
                # FA3-style: split into two loops - one with mask, one without
                # This allows compiler to generate more efficient code for no-mask case

                # Loop 1: iterations that need causal mask (n_block >= n_block_min_causal_mask)
                for iter_n_block in range(n_block_max - 2, n_block_min_causal_mask - 1, -1):
                    # FA3-style two-step wait for K (non-blocking check first)
                    # Use k_stage/v_stage directly (no copy) - compiler will optimize
                    k_ready: ll.bool_ = ll.mbarrier_try_wait_parity(full_barrier_K_base + k_stage, k_phase)
                    # FA3-style two-step wait for V (non-blocking check first)
                    v_ready: ll.bool_ = ll.mbarrier_try_wait_parity(full_barrier_V_base + v_stage, v_phase)

                    # Blocking wait for K if not ready
                    ll.mbarrier_wait_if_needed(full_barrier_K_base + k_stage, k_phase, k_ready)

                    # Scheduler barrier sync (use precomputed barrier ID)
                    ll.named_barrier_sync(my_barrier, 2 * kThreadsPerWarpGroup)

                    # Zero acc_s for QK
                    for j in ll.unroll(range(64)):
                        acc_s[j] = 0.0

                    # Fence operands before WGMMA (FA3 optimization)
                    ll.wgmma_fence_operand_array(acc_s, 64)

                    # QK WGMMA
                    ll.wgmma_fence()
                    for ki in ll.unroll(range(4)):
                        k_off = ll.val_cast(ki, ll.uint32) * K_STRIDE
                        q_lo = base_desc_q_lo + k_off
                        k_lo = base_desc_k_lo + ll.val_cast(k_stage, ll.uint32) * STAGE_STRIDE + k_off
                        desc_q = ll.make_desc(q_lo, DESC_Q_HI)
                        desc_k = ll.make_desc(k_lo, DESC_K_HI)
                        scale_d = 1 if (ki > 0) else 0
                        ll.wgmma_m64n128k16_ss(acc_s, desc_q, desc_k, scale_d)
                    for ki in ll.unroll(range(4)):
                        k_off = ll.val_cast(ki, ll.uint32) * K_STRIDE
                        q_lo = base_desc_q_lo + Q_SECOND_HALF + k_off
                        k_lo = base_desc_k_lo + ll.val_cast(k_stage, ll.uint32) * STAGE_STRIDE + K_SECOND_HALF + k_off
                        desc_q = ll.make_desc(q_lo, DESC_Q_HI)
                        desc_k = ll.make_desc(k_lo, DESC_K_HI)
                        ll.wgmma_m64n128k16_ss(acc_s, desc_q, desc_k, 1)
                    ll.wgmma_commit()

                    # Blocking wait for V if not ready
                    ll.mbarrier_wait_if_needed(full_barrier_V_base + v_stage, v_phase, v_ready)

                    # Scale acc_o by prev_scores_scale before PV
                    for j in ll.unroll(range(64)):
                        mi = (j >> 1) & 1
                        acc_o[j] = acc_o[j] * prev_scores_scale[mi]

                    # Fence operands before PV WGMMA (FA3 optimization)
                    ll.wgmma_fence_operand_array(acc_o, 64)
                    ll.wgmma_fence_u32_array(acc_p_bf16, 32)

                    # PV WGMMA (using previous P stored in acc_p_bf16)
                    v_addr: ll.uint32 = ll.cvta_generic_to_shared(smem_v_base + v_stage * 16384)
                    desc_v_base: ll.uint64 = (ll.uint64(v_addr >> 4)) | (1024 << 16) | (64 << 32) | (1 << 62)

                    ll.wgmma_fence()
                    for ki in ll.unroll(range(8)):
                        desc_v: ll.uint64 = desc_v_base + ((ki * 2048) >> 4)
                        a0 = acc_p_bf16[ki * 4 + 0]
                        a1 = acc_p_bf16[ki * 4 + 1]
                        a2 = acc_p_bf16[ki * 4 + 2]
                        a3 = acc_p_bf16[ki * 4 + 3]
                        ll.wgmma_m64n128k16_rs(acc_o, a0, a1, a2, a3, desc_v, 1)
                    ll.wgmma_commit()

                    # Fence operands after WGMMA commit (FA3 optimization)
                    ll.wgmma_fence_operand_array(acc_o, 64)

                    # Scheduler barrier arrive (signal other warpgroup)
                    ll.named_barrier_arrive(other_barrier, 2 * kThreadsPerWarpGroup)

                    # Wait for QK WGMMA only (allow PV to continue in background)
                    ll.wgmma_wait_group(1)

                    # Release K (use k_stage before update)
                    if is_warpgroup_leader == 1:
                        ll.mbarrier_arrive(empty_barrier_K_base + k_stage)

                    # Update K pipeline state
                    k_stage = k_stage + 1
                    if k_stage == kStages:
                        k_stage = 0
                        k_phase = k_phase ^ 1

                    # Apply mask (always for this loop - Loop 1 with apply_mask_flag=true)
                    # NOTE: No is_causal check needed - this loop only runs when mask is needed
                    iter_seqlenk_col_limit: ll.int32 = seqlen_k - iter_n_block * kBlockN - thread_col_offset
                    iter_causal_row_offset: ll.int32 = 1 + seqlen_k - iter_n_block * kBlockN - seqlen_q - thread_col_offset
                    for mi in ll.unroll(range(2)):
                        iter_row: ll.int32 = thread_row_base + mi * 8
                        iter_row_idx: ll.int32 = cur_m_block * kBlockM + m_offset + iter_row
                        # Use __viaddmin_s32 for efficient single-instruction computation
                        # NOTE: Always use viaddmin_s32 - this is Loop 1 which always needs causal mask
                        iter_col_limit_right: ll.int32 = ll.viaddmin_s32(iter_row_idx, iter_causal_row_offset,
                                                                         iter_seqlenk_col_limit)
                        iter_row_oob: ll.bool_ = iter_row_idx >= seqlen_q
                        for ni in ll.unroll(range(32)):
                            iter_reg: ll.int32 = mi * 2 + (ni & 1) + ((ni >> 1) << 2)
                            iter_col_offset: ll.int32 = ((ni >> 1) << 3) + (ni & 1)
                            # Use || (logical or) to combine conditions - matches CUDA's || behavior
                            if iter_row_oob | (iter_col_offset >= iter_col_limit_right):
                                acc_s[iter_reg] = ll.val_cast(NEG_INF, ll.float32)

                    # Softmax for this iteration
                    local_max[0] = ll.val_cast(NEG_INF, ll.float32)
                    local_max[1] = ll.val_cast(NEG_INF, ll.float32)
                    for j in ll.unroll(range(64)):
                        mi = (j >> 1) & 1
                        local_max[mi] = ll.max_val(local_max[mi], acc_s[j])

                    for mi in ll.unroll(range(2)):
                        local_max[mi] = ll.max_val(local_max[mi], ll.shfl_xor_sync(0xffffffff, local_max[mi], 2))
                        local_max[mi] = ll.max_val(local_max[mi], ll.shfl_xor_sync(0xffffffff, local_max[mi], 1))

                    # Update running max and compute scale
                    for mi in ll.unroll(range(2)):
                        old_max: ll.float32 = row_max[mi]
                        new_max: ll.float32 = local_max[mi]
                        row_max[mi] = ll.max_val(old_max, new_max)
                        prev_scores_scale[mi] = ll.exp2f((old_max - row_max[mi]) * softmax_scale_log2)
                        row_sum[mi] = row_sum[mi] * prev_scores_scale[mi]

                    # FA3-style: precompute max_scaled to avoid redundant computation in loop
                    max_scaled_0: ll.float32 = row_max[0] * softmax_scale_log2
                    max_scaled_1: ll.float32 = row_max[1] * softmax_scale_log2

                    # Wait for PV WGMMA to complete before exp (FA3 optimization)
                    ll.wgmma_wait()

                    # Exp and sum (FA3-style: ni outer, explicit reg0/reg1 for better register reuse)
                    for ni in ll.unroll(range(32)):
                        reg0: ll.int32 = 0 * 2 + (ni & 1) + ((ni >> 1) << 2)
                        reg1: ll.int32 = 1 * 2 + (ni & 1) + ((ni >> 1) << 2)
                        val0: ll.float32 = ll.exp2f(acc_s[reg0] * softmax_scale_log2 - max_scaled_0)
                        val1: ll.float32 = ll.exp2f(acc_s[reg1] * softmax_scale_log2 - max_scaled_1)
                        acc_s[reg0] = val0
                        acc_s[reg1] = val1
                        row_sum[0] = row_sum[0] + val0
                        row_sum[1] = row_sum[1] + val1

                    # Release V (Loop 1) - use v_stage before update
                    if is_warpgroup_leader == 1:
                        ll.mbarrier_arrive(empty_barrier_V_base + v_stage)

                    # Update V pipeline state
                    v_stage = v_stage + 1
                    if v_stage == kStages:
                        v_stage = 0
                        v_phase = v_phase ^ 1

                    # Store P for next iteration (use pack_bf16_float for efficiency)
                    for j in ll.unroll(range(16)):
                        j4 = j * 4
                        acc_p_bf16[j * 2] = ll.pack_bf16_float(acc_s[j4 + 0], acc_s[j4 + 1])
                        acc_p_bf16[j * 2 + 1] = ll.pack_bf16_float(acc_s[j4 + 2], acc_s[j4 + 3])

                # Loop 2: iterations that don't need mask (n_block < n_block_min_causal_mask)
                for iter_n_block in range(n_block_min_causal_mask - 1, -1, -1):
                    # FA3-style two-step wait for K (non-blocking check first)
                    # Use k_stage/v_stage directly (no copy) - compiler will optimize
                    k_ready = ll.mbarrier_try_wait_parity(full_barrier_K_base + k_stage, k_phase)
                    # FA3-style two-step wait for V (non-blocking check first)
                    v_ready = ll.mbarrier_try_wait_parity(full_barrier_V_base + v_stage, v_phase)

                    # Blocking wait for K if not ready
                    ll.mbarrier_wait_if_needed(full_barrier_K_base + k_stage, k_phase, k_ready)

                    # Scheduler barrier sync (use precomputed barrier ID)
                    ll.named_barrier_sync(my_barrier, 2 * kThreadsPerWarpGroup)

                    # Zero acc_s for QK
                    for j in ll.unroll(range(64)):
                        acc_s[j] = 0.0

                    # Fence operands before WGMMA (FA3 optimization)
                    ll.wgmma_fence_operand_array(acc_s, 64)

                    # QK WGMMA
                    ll.wgmma_fence()
                    for ki in ll.unroll(range(4)):
                        k_off = ll.val_cast(ki, ll.uint32) * K_STRIDE
                        q_lo = base_desc_q_lo + k_off
                        k_lo = base_desc_k_lo + ll.val_cast(k_stage, ll.uint32) * STAGE_STRIDE + k_off
                        desc_q = ll.make_desc(q_lo, DESC_Q_HI)
                        desc_k = ll.make_desc(k_lo, DESC_K_HI)
                        scale_d = 1 if (ki > 0) else 0
                        ll.wgmma_m64n128k16_ss(acc_s, desc_q, desc_k, scale_d)
                    for ki in ll.unroll(range(4)):
                        k_off = ll.val_cast(ki, ll.uint32) * K_STRIDE
                        q_lo = base_desc_q_lo + Q_SECOND_HALF + k_off
                        k_lo = base_desc_k_lo + ll.val_cast(k_stage, ll.uint32) * STAGE_STRIDE + K_SECOND_HALF + k_off
                        desc_q = ll.make_desc(q_lo, DESC_Q_HI)
                        desc_k = ll.make_desc(k_lo, DESC_K_HI)
                        ll.wgmma_m64n128k16_ss(acc_s, desc_q, desc_k, 1)
                    ll.wgmma_commit()

                    # Blocking wait for V if not ready
                    ll.mbarrier_wait_if_needed(full_barrier_V_base + v_stage, v_phase, v_ready)

                    # Scale acc_o by prev_scores_scale before PV
                    for j in ll.unroll(range(64)):
                        mi = (j >> 1) & 1
                        acc_o[j] = acc_o[j] * prev_scores_scale[mi]

                    # Fence operands before PV WGMMA (FA3 optimization)
                    ll.wgmma_fence_operand_array(acc_o, 64)
                    ll.wgmma_fence_u32_array(acc_p_bf16, 32)

                    # PV WGMMA (using previous P stored in acc_p_bf16)
                    v_addr = ll.cvta_generic_to_shared(smem_v_base + v_stage * 16384)
                    desc_v_base = (ll.uint64(v_addr >> 4)) | (1024 << 16) | (64 << 32) | (1 << 62)

                    ll.wgmma_fence()
                    for ki in ll.unroll(range(8)):
                        desc_v = desc_v_base + ((ki * 2048) >> 4)
                        a0 = acc_p_bf16[ki * 4 + 0]
                        a1 = acc_p_bf16[ki * 4 + 1]
                        a2 = acc_p_bf16[ki * 4 + 2]
                        a3 = acc_p_bf16[ki * 4 + 3]
                        ll.wgmma_m64n128k16_rs(acc_o, a0, a1, a2, a3, desc_v, 1)
                    ll.wgmma_commit()

                    # Fence operands after WGMMA commit (FA3 optimization)
                    ll.wgmma_fence_operand_array(acc_o, 64)

                    # Scheduler barrier arrive (signal other warpgroup)
                    ll.named_barrier_arrive(other_barrier, 2 * kThreadsPerWarpGroup)

                    # Wait for QK WGMMA only (allow PV to continue in background)
                    ll.wgmma_wait_group(1)

                    # Release K (use k_stage before update)
                    if is_warpgroup_leader == 1:
                        ll.mbarrier_arrive(empty_barrier_K_base + k_stage)

                    # Update K pipeline state
                    k_stage = k_stage + 1
                    if k_stage == kStages:
                        k_stage = 0
                        k_phase = k_phase ^ 1

                    # No mask needed for this loop

                    # Softmax for this iteration
                    local_max[0] = ll.val_cast(NEG_INF, ll.float32)
                    local_max[1] = ll.val_cast(NEG_INF, ll.float32)
                    for j in ll.unroll(range(64)):
                        mi = (j >> 1) & 1
                        local_max[mi] = ll.max_val(local_max[mi], acc_s[j])

                    for mi in ll.unroll(range(2)):
                        local_max[mi] = ll.max_val(local_max[mi], ll.shfl_xor_sync(0xffffffff, local_max[mi], 2))
                        local_max[mi] = ll.max_val(local_max[mi], ll.shfl_xor_sync(0xffffffff, local_max[mi], 1))

                    # Update running max and compute scale
                    for mi in ll.unroll(range(2)):
                        old_max: ll.float32 = row_max[mi]
                        new_max: ll.float32 = local_max[mi]
                        row_max[mi] = ll.max_val(old_max, new_max)
                        prev_scores_scale[mi] = ll.exp2f((old_max - row_max[mi]) * softmax_scale_log2)
                        row_sum[mi] = row_sum[mi] * prev_scores_scale[mi]

                    # FA3-style: precompute max_scaled to avoid redundant computation in loop
                    max_scaled_0 = row_max[0] * softmax_scale_log2
                    max_scaled_1 = row_max[1] * softmax_scale_log2

                    # Wait for PV WGMMA to complete before exp (FA3 optimization)
                    ll.wgmma_wait()

                    # Exp and sum (FA3-style: ni outer, explicit reg0/reg1 for better register reuse)
                    for ni in ll.unroll(range(32)):
                        reg0: ll.int32 = 0 * 2 + (ni & 1) + ((ni >> 1) << 2)
                        reg1: ll.int32 = 1 * 2 + (ni & 1) + ((ni >> 1) << 2)
                        val0: ll.float32 = ll.exp2f(acc_s[reg0] * softmax_scale_log2 - max_scaled_0)
                        val1: ll.float32 = ll.exp2f(acc_s[reg1] * softmax_scale_log2 - max_scaled_1)
                        acc_s[reg0] = val0
                        acc_s[reg1] = val1
                        row_sum[0] = row_sum[0] + val0
                        row_sum[1] = row_sum[1] + val1

                    # Release V (use v_stage before update)
                    if is_warpgroup_leader == 1:
                        ll.mbarrier_arrive(empty_barrier_V_base + v_stage)

                    # Update V pipeline state
                    v_stage = v_stage + 1
                    if v_stage == kStages:
                        v_stage = 0
                        v_phase = v_phase ^ 1

                    # Store P for next iteration (use pack_bf16_float for efficiency)
                    for j in ll.unroll(range(16)):
                        j4 = j * 4
                        acc_p_bf16[j * 2] = ll.pack_bf16_float(acc_s[j4 + 0], acc_s[j4 + 1])
                        acc_p_bf16[j * 2 + 1] = ll.pack_bf16_float(acc_s[j4 + 2], acc_s[j4 + 3])

                # ===== Final PV =====
                final_v_stage: ll.int32 = v_stage
                final_v_phase: ll.int32 = v_phase

                # FA3-style two-step wait for final V
                final_v_ready: ll.bool_ = ll.mbarrier_try_wait_parity(full_barrier_V_base + final_v_stage,
                                                                      final_v_phase)
                ll.mbarrier_wait_if_needed(full_barrier_V_base + final_v_stage, final_v_phase, final_v_ready)

                # Scheduler barrier sync (use precomputed barrier ID)
                ll.named_barrier_sync(my_barrier, 2 * kThreadsPerWarpGroup)

                # Apply final scale
                for j in ll.unroll(range(64)):
                    mi = (j >> 1) & 1
                    acc_o[j] = acc_o[j] * prev_scores_scale[mi]

                # PV WGMMA
                final_v_addr: ll.uint32 = ll.cvta_generic_to_shared(smem_v_base + final_v_stage * 16384)
                desc_v_base: ll.uint64 = (ll.uint64(final_v_addr >> 4)) | (1024 << 16) | (64 << 32) | (1 << 62)

                ll.wgmma_fence()
                for ki in ll.unroll(range(8)):
                    desc_v: ll.uint64 = desc_v_base + ((ki * 2048) >> 4)
                    # Extract 4 uint32 values for this k iteration
                    a0: ll.uint32 = acc_p_bf16[ki * 4 + 0]
                    a1: ll.uint32 = acc_p_bf16[ki * 4 + 1]
                    a2: ll.uint32 = acc_p_bf16[ki * 4 + 2]
                    a3: ll.uint32 = acc_p_bf16[ki * 4 + 3]
                    ll.wgmma_m64n128k16_rs(acc_o, a0, a1, a2, a3, desc_v, 1)
                ll.wgmma_commit()
                ll.wgmma_wait()

                if is_warpgroup_leader == 1:
                    ll.mbarrier_arrive(empty_barrier_V_base + final_v_stage)

                # Update V pipeline state after final PV
                v_stage = v_stage + 1
                if v_stage == kStages:
                    v_stage = 0
                    v_phase = v_phase ^ 1

                # Scheduler barrier arrive (signal other warpgroup)
                ll.named_barrier_arrive(other_barrier, 2 * kThreadsPerWarpGroup)

                # Signal Q consumed
                if is_warpgroup_leader == 1:
                    ll.mbarrier_arrive(empty_barrier_Q)

                # Final softmax reduction
                for mi in ll.unroll(range(2)):
                    row_sum[mi] = row_sum[mi] + ll.shfl_xor_sync(0xffffffff, row_sum[mi], 2)
                    row_sum[mi] = row_sum[mi] + ll.shfl_xor_sync(0xffffffff, row_sum[mi], 1)

                # Normalize and store output
                # ... epilogue code ...

                # R2S: stmatrix
                warp_in_wg: ll.int32 = (thread_idx & 127) >> 5
                lane_id: ll.int32 = thread_idx & 31
                matrix_id: ll.int32 = lane_id >> 3
                lane_in_matrix: ll.int32 = lane_id & 7
                mat_row: ll.int32 = matrix_id >> 1
                mat_col: ll.int32 = matrix_id & 1
                row_in_tile: ll.int32 = (mat_row << 3) + lane_in_matrix
                warp_base_row: ll.int32 = warp_in_wg << 4
                warpgroup_base_row: ll.int32 = (warp_group_idx - 1) << 6
                row: ll.int32 = warpgroup_base_row + warp_base_row + row_in_tile
                col_offset_matrix: ll.int32 = mat_col << 3
                swizzle_xor: ll.int32 = (row << 3) & 0x70

                # Normalize acc_o
                # Use ll.val_cast to ensure single precision division (avoid DFMA instructions)
                inv_sum_0: ll.float32 = ll.val_cast(0.0, ll.float32) if (row_sum[0] == ll.val_cast(
                    0.0, ll.float32)) else (ll.val_cast(1.0, ll.float32) / row_sum[0])
                inv_sum_1: ll.float32 = ll.val_cast(0.0, ll.float32) if (row_sum[1] == ll.val_cast(
                    0.0, ll.float32)) else (ll.val_cast(1.0, ll.float32) / row_sum[1])
                for i in ll.unroll(range(64)):
                    mi = ((i >> 1) & 1)
                    inv_sum: ll.float32 = inv_sum_0 if (mi == 0) else inv_sum_1
                    acc_o[i] = acc_o[i] * inv_sum

                # stmatrix.x4 loop
                for col_group in ll.unroll(range(8)):
                    acc_base: ll.int32 = col_group * 8
                    r0: ll.uint32 = ll.pack_bf16_float(acc_o[acc_base + 0], acc_o[acc_base + 1])
                    r1: ll.uint32 = ll.pack_bf16_float(acc_o[acc_base + 4], acc_o[acc_base + 5])
                    r2: ll.uint32 = ll.pack_bf16_float(acc_o[acc_base + 2], acc_o[acc_base + 3])
                    r3: ll.uint32 = ll.pack_bf16_float(acc_o[acc_base + 6], acc_o[acc_base + 7])

                    col: ll.int32 = (col_group << 4) + col_offset_matrix
                    col_lo: ll.int32 = col & 63
                    col_hi: ll.int32 = (col >> 6) << 13
                    base: ll.int32 = col_lo + (row << 6) + col_hi
                    smem_idx: ll.int32 = base ^ swizzle_xor

                    ll.stmatrix_x4(smem_o + smem_idx, r0, r1, r2, r3)

                # Sync after R2S
                ll.named_barrier_sync(kBarrierEpilogue, kNumEpilogueThreads)

                # S2G: vectorized store
                mma_thread_idx: ll.int32 = thread_idx - 128
                base_col: ll.int32 = (mma_thread_idx & 15) << 3
                row_offset: ll.int32 = mma_thread_idx >> 4

                if base_col < headdim:
                    col_contrib: ll.int32 = base_col if (base_col < 64) else (base_col - 64 + 8192)
                    s2g_swizzle_xor: ll.int32 = (row_offset << 3) & 0x70
                    base_row_offset: ll.int32 = col_contrib + (row_offset << 6)

                    rows_remaining: ll.int32 = seqlen_q - cur_m_block * kBlockM
                    gmem_o_base: ll.int64 = ll.int64(offset_q + cur_m_block * kBlockM) * stride_O_row + ll.int64(
                        cur_bidh * headdim)

                    for m in ll.unroll(range(8)):
                        s2g_row: ll.int32 = m * 16 + row_offset
                        if s2g_row < rows_remaining:
                            s2g_base: ll.int32 = base_row_offset + (m << 10)
                            s2g_smem_idx: ll.int32 = s2g_base ^ s2g_swizzle_xor

                            # 128-bit vectorized load/store
                            ll.store_128bit(ptr_O + gmem_o_base + ll.int64(s2g_row) * stride_O_row + base_col,
                                            ll.load_128bit(smem_o + s2g_smem_idx))


def build_kernel(num_sm=132, disable_var_hoisting=True, cache_dir=None):
    """Build the flash attention kernel.

    Args:
        num_sm: Number of SMs to use (default 132 for H100)
        disable_var_hoisting: If True, disable variable hoisting for better register allocation.
                              Default True for performance-critical kernels.
        cache_dir: Explicit cache directory, or a platform temporary directory.
    """
    if cache_dir is None:
        import tempfile
        cache_dir = os.path.join(tempfile.gettempdir(), 'lk_cache')
    return flash_fwd_varlen_kernel.build(
        passes,
        codegen_cuda,
        grid=(num_sm, 1, 1),
        block=(kNumThreads, 1, 1),
        shared_mem_bytes=SMEM_TOTAL,
        arch="sm_90a",
        disable_var_hoisting=disable_var_hoisting,
        cache_dir=cache_dir,
        extra_nvcc_options=["-DNDEBUG", "--use_fast_math"],
    )


# ============================================================================
# Test code
# ============================================================================


def test_cuda_correctness():
    """Test the pre-compiled CUDA kernel."""
    print("Testing pre-compiled CUDA kernel...")

    # Load CUDA kernel
    cuda_lib_path = os.path.join(PROJECT_ROOT, 'workspace', 'flash_attn_varlen.so')
    if not os.path.exists(cuda_lib_path):
        print(f"CUDA library not found at {cuda_lib_path}")
        return False

    lib = ctypes.CDLL(cuda_lib_path)

    # Test parameters
    batch = 4
    seqlen = 512
    nheads = 32
    headdim = 128

    # Create test data
    Q = torch.randn(batch * seqlen, nheads, headdim, dtype=torch.bfloat16, device='cuda')
    K = torch.randn(batch * seqlen, nheads, headdim, dtype=torch.bfloat16, device='cuda')
    V = torch.randn(batch * seqlen, nheads, headdim, dtype=torch.bfloat16, device='cuda')
    O = torch.zeros_like(Q)

    cu_seqlens_q = torch.arange(0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device='cuda')
    cu_seqlens_k = cu_seqlens_q.clone()

    # Workspace
    workspace_lse = torch.zeros(nheads * batch * seqlen, dtype=torch.float32, device='cuda')
    workspace_semaphore = torch.zeros(1, dtype=torch.int32, device='cuda')

    # Call kernel
    lib.flash_attn_fwd_varlen(ctypes.c_void_p(Q.data_ptr()), ctypes.c_void_p(K.data_ptr()),
                              ctypes.c_void_p(V.data_ptr()), ctypes.c_void_p(O.data_ptr()),
                              ctypes.c_void_p(cu_seqlens_q.data_ptr()), ctypes.c_void_p(cu_seqlens_k.data_ptr()),
                              ctypes.c_int(batch * seqlen), ctypes.c_int(batch * seqlen), ctypes.c_int(seqlen),
                              ctypes.c_int(seqlen), ctypes.c_int(batch), ctypes.c_int(nheads), ctypes.c_int(nheads),
                              ctypes.c_int(headdim), ctypes.c_float(1.0 / math.sqrt(headdim)),
                              ctypes.c_int(1),  # is_causal
                              ctypes.c_void_p(0),  # stream
                              ctypes.c_void_p(workspace_lse.data_ptr()),
                              ctypes.c_void_p(workspace_semaphore.data_ptr()), ctypes.c_void_p(0),  # seqused_q
                              ctypes.c_void_p(0),  # seqused_k
                              ctypes.c_void_p(0),  # leftpad_k
                              )
    torch.cuda.synchronize()

    # Reference
    Q_ref = Q.view(batch, seqlen, nheads, headdim).transpose(1, 2).float()
    K_ref = K.view(batch, seqlen, nheads, headdim).transpose(1, 2).float()
    V_ref = V.view(batch, seqlen, nheads, headdim).transpose(1, 2).float()

    scale = 1.0 / math.sqrt(headdim)
    attn = torch.matmul(Q_ref, K_ref.transpose(-2, -1)) * scale

    # Causal mask
    mask = torch.triu(torch.ones(seqlen, seqlen, device='cuda'), diagonal=1).bool()
    attn = attn.masked_fill(mask, float('-inf'))

    attn = torch.softmax(attn, dim=-1)
    O_ref = torch.matmul(attn, V_ref)
    O_ref = O_ref.transpose(1, 2).reshape(batch * seqlen, nheads, headdim).bfloat16()

    # Compare
    cos_sim = torch.nn.functional.cosine_similarity(O.float().flatten().unsqueeze(0),
                                                    O_ref.float().flatten().unsqueeze(0)).item()

    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max diff: {(O - O_ref).abs().max().item():.6f}")

    return cos_sim > 0.99


# ============================================================================
# Optimized kernel launcher with precompiled arguments
# ============================================================================


class OptimizedDSLLauncher:
    """Optimized launcher that precompiles arguments to minimize Python overhead."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.runtime = kernel.runtime
        self.func = kernel.launcher.func
        self.grid = kernel.grid
        self.block = kernel.block
        self.shared_mem_bytes = kernel.shared_mem_bytes
        self._cached_args = {}

    def _precompile_args(self, args):
        """Precompile arguments into a reusable format."""
        arg_ptrs = []
        arg_data = []  # Keep references to prevent GC

        for arg in args:
            if isinstance(arg, ctypes.Structure):
                arg_data.append(arg)
                struct_ptr = ctypes.cast(ctypes.byref(arg), ctypes.c_void_p)
                arg_ptrs.append(struct_ptr)
            elif isinstance(arg, int):
                val = int(arg)
                if -2**31 <= val < 2**31:
                    c_val = ctypes.c_int32(val)
                else:
                    c_val = ctypes.c_int64(val)
                arg_data.append(c_val)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(c_val), ctypes.c_void_p))
            elif hasattr(arg, "data_ptr"):
                ptr_val = ctypes.c_void_p(arg.data_ptr())
                arg_data.append(ptr_val)
                ptr_to_ptr = ctypes.cast(ctypes.pointer(ptr_val), ctypes.c_void_p)
                arg_ptrs.append(ptr_to_ptr)
            elif isinstance(arg, float):
                c_val = ctypes.c_float(float(arg))
                arg_data.append(c_val)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(c_val), ctypes.c_void_p))

        kernel_params_array = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)
        arg_data.append(kernel_params_array)
        kernel_params = ctypes.cast(kernel_params_array, ctypes.POINTER(ctypes.c_void_p))

        return kernel_params, arg_data

    def launch(self, *args):
        """Launch kernel with precompiled arguments."""
        kernel_params, arg_data = self._precompile_args(args)

        # Direct cuLaunchKernel call
        self.runtime._libcuda.cuLaunchKernel(
            self.func,
            self.grid[0],
            self.grid[1],
            self.grid[2],
            self.block[0],
            self.block[1],
            self.block[2],
            self.shared_mem_bytes,
            ctypes.c_void_p(0),  # stream
            kernel_params,
            None,
        )


def run_dsl_kernel(kernel, q, k, v, cu_q, cu_k, batch, max_q, max_k, launcher=None):
    """Run DSL kernel.

    Args:
        kernel: Compiled DSL kernel
        q, k, v: Input tensors
        cu_q, cu_k: Cumulative sequence lengths
        batch, max_q, max_k: Batch size and max sequence lengths
        launcher: Optional OptimizedDSLLauncher for reduced overhead
    """
    total_q = q.size(0)
    total_k = k.size(0)
    nheads, nkv_heads, headdim = q.size(1), k.size(1), q.size(2)
    out = torch.zeros_like(q)
    sem = torch.zeros(1, dtype=torch.int32, device=q.device)
    scale = 1.0 / math.sqrt(headdim)
    softmax_scale_log2 = scale * math.log2(math.e)

    # Compute FastDivmod for GQA: kv_head_idx = bidh / (nheads / nkv_heads)
    qhead_per_khead = nheads // nkv_heads
    fd = FastDivmod(qhead_per_khead)

    # Strides
    q_row_stride = nheads * headdim
    kv_row_stride = nkv_heads * headdim
    head_stride = headdim

    # Create TMA descriptors
    # Q: 3D tensor [headdim, total_q, nheads] in TMA's column-major view
    tma_Q = create_tma_3d_descriptor(q, dim0=headdim,  # innermost
                                     dim1=total_q,  # middle
                                     dim2=nheads,  # outermost
                                     stride1_bytes=q_row_stride * 2,  # bf16 = 2 bytes
                                     stride2_bytes=head_stride * 2, tile_dim0=kTmaTileCols,  # 64
                                     tile_dim1=kBlockM,  # 128
                                     )

    # K: 3D tensor [headdim, total_k, nkv_heads]
    tma_K = create_tma_3d_descriptor(
        k,
        dim0=headdim,
        dim1=total_k,
        dim2=nkv_heads,
        stride1_bytes=kv_row_stride * 2,
        stride2_bytes=head_stride * 2,
        tile_dim0=kTmaTileCols,
        tile_dim1=kBlockN,
    )

    # V: Same as K (transposed view for WGMMA)
    tma_V = create_tma_3d_descriptor(
        v,
        dim0=headdim,
        dim1=total_k,
        dim2=nkv_heads,
        stride1_bytes=kv_row_stride * 2,
        stride2_bytes=head_stride * 2,
        tile_dim0=kTmaTileCols,
        tile_dim1=kBlockN,
    )

    args = (
        tma_Q, tma_K, tma_V, out, cu_q, cu_k, total_q, total_k, max_q, max_k, batch, nheads, nkv_heads, headdim,
        1,  # is_causal
        softmax_scale_log2, nheads * headdim,  # stride_O_row
        sem, fd.multiplier,  # qhead_per_khead_multiplier
        fd.shift_right,  # qhead_per_khead_shift
    )

    # Use optimized launcher if provided
    if launcher is not None:
        launcher.launch(*args)
    else:
        kernel(*args)

    torch.cuda.synchronize()
    return out


def run_cuda_kernel(lib, q, k, v, cu_q, cu_k, batch, max_q, max_k):
    """Run CUDA kernel."""
    total_q = q.size(0)
    nheads, nkv_heads, headdim = q.size(1), k.size(1), q.size(2)
    out = torch.zeros_like(q)
    lse = torch.empty(nheads * total_q, dtype=torch.float32, device=q.device)
    sem = torch.zeros(1, dtype=torch.int32, device=q.device)
    scale = 1.0 / math.sqrt(headdim)
    stream = torch.cuda.current_stream().cuda_stream
    lib.flash_attn_fwd_varlen(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        out.data_ptr(),
        cu_q.data_ptr(),
        cu_k.data_ptr(),
        total_q,
        total_q,
        max_q,
        max_k,
        batch,
        nheads,
        nkv_heads,
        headdim,
        scale,
        True,
        stream,
        lse.data_ptr(),
        sem.data_ptr(),
        None,
        None,
        None,
    )
    torch.cuda.synchronize()
    return out


def load_cuda_lib():
    """Load CUDA library."""
    cuda_lib_path = os.path.join(PROJECT_ROOT, 'workspace', 'flash_attn_varlen.so')
    if not os.path.exists(cuda_lib_path):
        raise RuntimeError(f"CUDA library not found at {cuda_lib_path}")
    ctypes.CDLL('libcuda.so.1', mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(cuda_lib_path)
    lib.flash_attn_fwd_varlen.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.flash_attn_fwd_varlen.restype = None
    return lib


# Test shapes: (batch, seqlen, nheads, nkv_heads, headdim)
TEST_SHAPES = [
    # Uniform length (all batches same seqlen)
    {"batch": 1, "seqlen": 256, "nheads": 4, "nkv_heads": 1, "headdim": 128},
    {"batch": 2, "seqlen": 512, "nheads": 8, "nkv_heads": 2, "headdim": 128},
    {"batch": 4, "seqlen": 1024, "nheads": 16, "nkv_heads": 4, "headdim": 128},
    {"batch": 8, "seqlen": 2048, "nheads": 32, "nkv_heads": 8, "headdim": 128},
    {"batch": 8, "seqlen": 4096, "nheads": 32, "nkv_heads": 8, "headdim": 128},
    {"batch": 8, "seqlen": 4096, "nheads": 64, "nkv_heads": 8, "headdim": 128},
    # Large shapes for peak performance
    {"batch": 4, "seqlen": 8192, "nheads": 32, "nkv_heads": 8, "headdim": 128},
    {"batch": 8, "seqlen": 8192, "nheads": 32, "nkv_heads": 8, "headdim": 128},
]

import random


def create_test_data(shape, device):
    """Create Q,K,V,cu_seqlens from shape spec."""
    dtype = torch.bfloat16
    nheads, nkv_heads, headdim = shape["nheads"], shape["nkv_heads"], shape["headdim"]

    if "seqlens" in shape:
        seq_lens = shape["seqlens"]
        batch = len(seq_lens)
    elif "min_len" in shape:
        random.seed(42)
        seq_lens = [random.randint(shape["min_len"], shape["max_len"]) for _ in range(shape["batch"])]
        batch = shape["batch"]
    else:
        seq_lens = [shape["seqlen"]] * shape["batch"]
        batch = shape["batch"]

    total = sum(seq_lens)
    max_len = max(seq_lens)
    cu = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seq_lens, dtype=torch.int32, device=device).cumsum(0)

    q = torch.randn(total, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(total, nkv_heads, headdim, dtype=dtype, device=device)
    v = torch.randn(total, nkv_heads, headdim, dtype=dtype, device=device)

    return q, k, v, cu, cu, batch, max_len, max_len, seq_lens


def compute_flops(seq_lens, nheads, headdim, causal=True):
    """Compute FLOPs for attention."""
    total = 0
    for s in seq_lens:
        tokens = s * (s + 1) // 2 if causal else s * s
        total += 4 * tokens * headdim * nheads
    return total


def shape_str(shape):
    """Format shape for display."""
    if "seqlens" in shape:
        return "seqlens=%s h=%d/%d" % (shape["seqlens"], shape["nheads"], shape["nkv_heads"])
    elif "min_len" in shape:
        return "b=%d len=%d-%d h=%d/%d" % (shape["batch"], shape["min_len"], shape["max_len"], shape["nheads"],
                                           shape["nkv_heads"])
    else:
        return "b=%d s=%d h=%d/%d" % (shape["batch"], shape["seqlen"], shape["nheads"], shape["nkv_heads"])


def test_dsl_vs_cuda():
    """Test DSL kernel correctness against CUDA reference."""
    print("=" * 70)
    print("DSL vs CUDA Correctness Test")
    print("=" * 70)

    lib = load_cuda_lib()
    kernel = build_kernel()
    device = torch.device("cuda")
    all_pass = True

    for shape in TEST_SHAPES:
        torch.manual_seed(42)
        q, k, v, cu_q, cu_k, batch, max_q, max_k, seq_lens = create_test_data(shape, device)

        # DSL output
        out_dsl = run_dsl_kernel(kernel, q, k, v, cu_q, cu_k, batch, max_q, max_k)

        # CUDA output
        out_cuda = run_cuda_kernel(lib, q, k, v, cu_q, cu_k, batch, max_q, max_k)

        max_diff = (out_dsl.float() - out_cuda.float()).abs().max().item()
        ok = max_diff < 0.005  # Match CUDA reference tolerance
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print("  %-45s diff=%.6f %s" % (shape_str(shape), max_diff, status))

    print()
    return all_pass


def benchmark_dsl_vs_cuda(warmup=100, iters=100, rounds=5):
    """Benchmark DSL vs CUDA kernel using interleaved measurement.

    This benchmark measures pure kernel execution time by:
    1. Pre-creating all tensors and TMA descriptors
    2. Pre-compiling kernel arguments
    3. Using direct cuLaunchKernel calls
    4. Interleaving CUDA and DSL measurements to eliminate GPU frequency drift
    """
    print("=" * 70)
    print("DSL vs CUDA Performance Benchmark (warmup=%d, iters=%d, rounds=%d)" % (warmup, iters, rounds))
    print("=" * 70)
    print("  %-45s %10s %10s %8s" % ("Shape", "CUDA", "DSL", "Ratio"))
    print("  " + "-" * 75)

    lib = load_cuda_lib()
    kernel = build_kernel()
    device = torch.device("cuda")

    # Get kernel launch info
    runtime = kernel.runtime
    func = kernel.launcher.func
    grid = kernel.grid
    block = kernel.block
    shared_mem_bytes = kernel.shared_mem_bytes

    # Load cudart once
    import ctypes as ct
    cudart = ct.CDLL('libcudart.so')
    cudart.cudaMemsetAsync.argtypes = [ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_void_p]
    cudart.cudaMemsetAsync.restype = ct.c_int

    def precompile_dsl_args(args):
        """Precompile arguments for direct cuLaunchKernel."""
        arg_ptrs = []
        arg_data = []

        for arg in args:
            if isinstance(arg, ctypes.Structure):
                arg_data.append(arg)
                struct_ptr = ctypes.cast(ctypes.byref(arg), ctypes.c_void_p)
                arg_ptrs.append(struct_ptr)
            elif isinstance(arg, int):
                val = int(arg)
                if -2**31 <= val < 2**31:
                    c_val = ctypes.c_int32(val)
                else:
                    c_val = ctypes.c_int64(val)
                arg_data.append(c_val)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(c_val), ctypes.c_void_p))
            elif hasattr(arg, "data_ptr"):
                ptr_val = ctypes.c_void_p(arg.data_ptr())
                arg_data.append(ptr_val)
                ptr_to_ptr = ctypes.cast(ctypes.pointer(ptr_val), ctypes.c_void_p)
                arg_ptrs.append(ptr_to_ptr)
            elif isinstance(arg, float):
                c_val = ctypes.c_float(float(arg))
                arg_data.append(c_val)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(c_val), ctypes.c_void_p))

        kernel_params_array = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)
        arg_data.append(kernel_params_array)
        kernel_params = ctypes.cast(kernel_params_array, ctypes.POINTER(ctypes.c_void_p))

        return kernel_params, arg_data

    for shape in TEST_SHAPES:
        torch.manual_seed(42)
        q, k, v, cu_q, cu_k, batch, max_q, max_k, seq_lens = create_test_data(shape, device)
        nheads, nkv_heads, headdim = shape["nheads"], shape["nkv_heads"], shape["headdim"]
        flops = compute_flops(seq_lens, nheads, headdim, causal=True)
        total_q = q.size(0)
        total_k = k.size(0)

        # Pre-create all tensors for CUDA
        out_cuda = torch.zeros_like(q)
        lse = torch.empty(nheads * total_q, dtype=torch.float32, device=device)
        sem_cuda = torch.zeros(1, dtype=torch.int32, device=device)
        scale = 1.0 / math.sqrt(headdim)
        stream = torch.cuda.current_stream().cuda_stream

        # Pre-create all tensors for DSL
        out_dsl = torch.zeros_like(q)
        sem_dsl = torch.zeros(1, dtype=torch.int32, device=device)
        softmax_scale_log2 = scale * math.log2(math.e)

        # Pre-create TMA descriptors
        q_row_stride = nheads * headdim
        kv_row_stride = nkv_heads * headdim
        head_stride = headdim

        tma_Q = create_tma_3d_descriptor(
            q,
            dim0=headdim,
            dim1=total_q,
            dim2=nheads,
            stride1_bytes=q_row_stride * 2,
            stride2_bytes=head_stride * 2,
            tile_dim0=kTmaTileCols,
            tile_dim1=kBlockM,
        )
        tma_K = create_tma_3d_descriptor(
            k,
            dim0=headdim,
            dim1=total_k,
            dim2=nkv_heads,
            stride1_bytes=kv_row_stride * 2,
            stride2_bytes=head_stride * 2,
            tile_dim0=kTmaTileCols,
            tile_dim1=kBlockN,
        )
        tma_V = create_tma_3d_descriptor(
            v,
            dim0=headdim,
            dim1=total_k,
            dim2=nkv_heads,
            stride1_bytes=kv_row_stride * 2,
            stride2_bytes=head_stride * 2,
            tile_dim0=kTmaTileCols,
            tile_dim1=kBlockN,
        )

        # Compute FastDivmod for GQA
        qhead_per_khead = nheads // nkv_heads
        fd = FastDivmod(qhead_per_khead)

        # Pre-compile DSL arguments
        dsl_args = (
            tma_Q, tma_K, tma_V, out_dsl, cu_q, cu_k, total_q, total_k, max_q, max_k, batch, nheads, nkv_heads, headdim,
            1,  # is_causal
            softmax_scale_log2, nheads * headdim,  # stride_O_row
            sem_dsl, fd.multiplier,  # qhead_per_khead_multiplier
            fd.shift_right,  # qhead_per_khead_shift
        )
        kernel_params, arg_data = precompile_dsl_args(dsl_args)

        # Warmup both kernels
        for _ in range(warmup):
            lib.flash_attn_fwd_varlen(
                q.data_ptr(),
                k.data_ptr(),
                v.data_ptr(),
                out_cuda.data_ptr(),
                cu_q.data_ptr(),
                cu_k.data_ptr(),
                total_q,
                total_q,
                max_q,
                max_k,
                batch,
                nheads,
                nkv_heads,
                headdim,
                scale,
                True,
                stream,
                lse.data_ptr(),
                sem_cuda.data_ptr(),
                None,
                None,
                None,
            )
            cudart.cudaMemsetAsync(ct.c_void_p(sem_dsl.data_ptr()), 0, 4, ct.c_void_p(0))
            runtime._libcuda.cuLaunchKernel(
                func,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_mem_bytes,
                ctypes.c_void_p(0),
                kernel_params,
                None,
            )
        torch.cuda.synchronize()

        # Interleaved benchmark: alternate CUDA and DSL to eliminate frequency drift
        cuda_times = []
        dsl_times = []

        for _ in range(rounds):
            # Benchmark CUDA
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                lib.flash_attn_fwd_varlen(
                    q.data_ptr(),
                    k.data_ptr(),
                    v.data_ptr(),
                    out_cuda.data_ptr(),
                    cu_q.data_ptr(),
                    cu_k.data_ptr(),
                    total_q,
                    total_q,
                    max_q,
                    max_k,
                    batch,
                    nheads,
                    nkv_heads,
                    headdim,
                    scale,
                    True,
                    stream,
                    lse.data_ptr(),
                    sem_cuda.data_ptr(),
                    None,
                    None,
                    None,
                )
            end.record()
            torch.cuda.synchronize()
            cuda_times.append(start.elapsed_time(end) / iters)

            # Benchmark DSL
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                cudart.cudaMemsetAsync(ct.c_void_p(sem_dsl.data_ptr()), 0, 4, ct.c_void_p(0))
                runtime._libcuda.cuLaunchKernel(
                    func,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared_mem_bytes,
                    ctypes.c_void_p(0),
                    kernel_params,
                    None,
                )
            end.record()
            torch.cuda.synchronize()
            dsl_times.append(start.elapsed_time(end) / iters)

        # Use median to reduce noise
        import statistics
        ms_cuda = statistics.median(cuda_times)
        ms_dsl = statistics.median(dsl_times)
        cuda_tflops = flops / (ms_cuda * 1e-3) / 1e12
        dsl_tflops = flops / (ms_dsl * 1e-3) / 1e12

        ratio = dsl_tflops / cuda_tflops if cuda_tflops > 0 else 0
        print("  %-45s %10.1f %10.1f %7.2fx" % (shape_str(shape), cuda_tflops, dsl_tflops, ratio))

    print()


if __name__ == "__main__":
    print("=" * 70)
    print("Flash Attention 3 - little_kernel DSL Implementation")
    print("=" * 70)

    # Test CUDA kernel first
    if test_cuda_correctness():
        print("CUDA kernel test PASSED\n")
    else:
        print("CUDA kernel test FAILED\n")

    # Build DSL kernel
    print("Building DSL kernel...")
    try:
        kernel = build_kernel()
        print("DSL kernel built successfully!\n")
    except Exception as e:
        print(f"DSL kernel build failed: {e}")
        sys.exit(1)

    # Test DSL vs CUDA
    if not test_dsl_vs_cuda():
        print("DSL vs CUDA correctness test FAILED!")
        sys.exit(1)

    # Benchmark
    benchmark_dsl_vs_cuda()
