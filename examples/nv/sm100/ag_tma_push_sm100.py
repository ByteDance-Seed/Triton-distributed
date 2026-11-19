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
"""Persistent intra-node TMA push all-gather for SM100."""

import little_kernel as lk
import little_kernel.language as ll

CHUNK_ROWS = 64
COPY_BYTES = 16 * 1024
BF16_BYTES = 2
COPY_ELEMS = COPY_BYTES // BF16_BYTES
NUM_STAGES = 4
NUM_THREADS = 32
SHARED_MEM_BYTES = NUM_STAGES * COPY_BYTES + 1024


@lk.ll_kernel(backend="cuda", is_entry=True)
def ag_tma_push_sm100_kernel(
    local_a: ll.Tensor[ll.bfloat16],
    workspace_ptrs: ll.ptr[ll.ptr[ll.bfloat16]],
    ready_peers: ll.ptr[ll.ptr[ll.int32]],
    rank: ll.int32,
    world_size: ll.int32,
    m_per_rank: ll.int32,
    k: ll.int32,
    chunks_per_rank: ll.int32,
    epoch: ll.int32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")
    full_barriers = ll.empty([NUM_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    staging = ll.empty([NUM_STAGES * COPY_ELEMS], dtype=ll.bfloat16, scope="dynamic_shared")

    issuer: ll.int32 = ll.elect_one()
    if issuer == 1:
        for stage_init in ll.unroll(range(NUM_STAGES)):
            ll.init_smem_barrier(full_barriers + stage_init, 1)
    ll.__syncthreads()

    grid_size: ll.int32 = ll.gridDim_x()

    # local_a aliases this rank's workspace slice, so it is ready without a copy.
    local_chunk: ll.int32 = ll.blockIdx_x()
    if issuer == 1:
        while local_chunk < chunks_per_rank:
            local_signal_idx: ll.int32 = rank * chunks_per_rank + local_chunk
            ll.st_release_system(ready_peers[rank] + local_signal_idx, epoch)
            local_chunk = local_chunk + grid_size
    ll.__syncwarp()

    stage: ll.int32 = 0
    full_phase: ll.int32 = 0
    chunk: ll.int32 = ll.blockIdx_x()
    while chunk < chunks_per_rank:
        row: ll.int32 = chunk * CHUNK_ROWS
        rows: ll.int32 = ll.min_val(CHUNK_ROWS, m_per_rank - row)
        chunk_bytes: ll.int32 = rows * k * BF16_BYTES
        byte_offset: ll.int32 = 0

        while byte_offset < chunk_bytes:
            transfer_bytes: ll.int32 = ll.min_val(COPY_BYTES, chunk_bytes - byte_offset)

            # Reuse the oldest stage while keeping up to NUM_STAGES stores in flight.
            ll.tma_store_wait_group(NUM_STAGES - 1)
            ll.__syncwarp()
            stage_offset: ll.int32 = stage * COPY_ELEMS
            if issuer == 1:
                ll.tma_copy_1d_g2s(
                    local_a + row * k + byte_offset // BF16_BYTES,
                    full_barriers + stage,
                    staging + stage_offset,
                    transfer_bytes,
                )
                ll.mbarrier_arrive_and_expect_tx(full_barriers + stage, transfer_bytes)
            ll.mbarrier_wait(full_barriers + stage, full_phase)

            dst_offset: ll.int32 = (rank * m_per_rank + row) * k + byte_offset // BF16_BYTES
            peer_step: ll.int32 = 1
            while peer_step < world_size:
                dst_rank: ll.int32 = (rank + peer_step) % world_size
                dst: ll.ptr[ll.bfloat16] = workspace_ptrs[dst_rank]
                if issuer == 1:
                    ll.tma_copy_1d_s2g(staging + stage_offset, dst + dst_offset, transfer_bytes)
                peer_step = peer_step + 1
            if issuer == 1:
                ll.tma_store_commit()
            ll.__syncwarp()

            stage = stage + 1
            if stage == NUM_STAGES:
                # A ring wrap starts the next mbarrier parity.
                stage = 0
                full_phase = full_phase ^ 1
            byte_offset = byte_offset + transfer_bytes

        # Publish only after every store for this chunk is globally visible.
        ll.tma_store_wait_group(0)
        ll.__syncwarp()
        signal_idx: ll.int32 = rank * chunks_per_rank + chunk
        if issuer == 1:
            signal_step: ll.int32 = 1
            while signal_step < world_size:
                signal_rank: ll.int32 = (rank + signal_step) % world_size
                ll.st_release_system(ready_peers[signal_rank] + signal_idx, epoch)
                signal_step = signal_step + 1
        chunk = chunk + grid_size
