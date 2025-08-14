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
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import tid, atomic_add, ld_acquire, __syncthreads, ld_b32, atomic_add_per_warp, st, ld
from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE
from triton_dist.kernels.nvidia.common_ops import (barrier_on_this_grid)


########## triton kernels ##########
@triton.jit
def kernel_dispatch_token(
    send_reqs_for_nodes,
    signals_for_nodes,
    counter_ptr,
    recv_buf_offset_per_expert,
    input_buf,  # recv token from other nodes
    output_buf,
    weight_send_buf,
    weight_recv_buf,
    topk_indices_tensor,  # [nnodes, max_tokens, topk]
    token_dst_scatter_idx,  # [self.nnodes, self.max_tokens, self.topk]
    num_input_tokens_per_rank,  # [world_size]
    max_tokens: int,
    topk: int,
    hidden_size: int,
    bytes_per_token: int,
    experts_per_rank: tl.constexpr,
    local_world_size: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WITH_SCATTER_INDICES: tl.constexpr,
    num_warps: tl.constexpr,
):
    WARP_SIZE = 32
    rank = dl.rank()
    world_size = dl.num_ranks()
    node_id = rank // local_world_size
    nnodes = world_size // local_world_size
    local_rank = rank % local_world_size
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    thread_idx = tid(0)
    warp_id = thread_idx // WARP_SIZE
    total_warps = num_warps * num_pid
    global_warp_id = pid * num_warps + warp_id
    weight_elem_size = tl.constexpr(weight_send_buf.dtype.element_ty.primitive_bitwidth) // 8
    num_tokens = tl.load(num_input_tokens_per_rank + rank)
    for node_offset in range(0, nnodes):
        if node_offset != nnodes - 1:
            target_node = (node_id + node_offset + 1) % nnodes
            target_rank = local_rank + target_node * local_world_size
            for req_id in range(global_warp_id, num_tokens, total_warps):
                start_index = ld_b32(send_reqs_for_nodes + target_node * max_tokens * 2 + req_id)
                end_index = ld_b32(send_reqs_for_nodes + target_node * max_tokens * 2 + max_tokens + req_id)
                msg_size = (end_index - start_index) * bytes_per_token
                src_ptr = input_buf + node_id * max_tokens * hidden_size + start_index * hidden_size
                if end_index > start_index:
                    libshmem_device.putmem_nbi_warp(src_ptr, src_ptr, msg_size, target_rank)

            if pid == 0:
                if HAS_WEIGHT:
                    weight_send_buf_ptr = weight_send_buf + node_id * max_tokens * topk
                    libshmem_device.putmem_nbi_block(
                        weight_send_buf_ptr,
                        weight_send_buf_ptr,
                        weight_elem_size * num_tokens * topk,
                        target_rank,
                    )

            __syncthreads()

            count = tl.atomic_add(counter_ptr + target_node, 1, scope="gpu", sem="release")  # noqa: F841
            libshmem_device.fence()

            while ld_acquire(counter_ptr + target_node, "gpu") != num_pid:
                pass


#            __syncthreads()
            if pid == 0:
                if thread_idx == 0:
                    libshmem_device.signal_op(
                        signals_for_nodes + node_id,
                        1,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        target_rank,
                    )

        __syncthreads()
        src_send_node = (node_id - node_offset + nnodes) % nnodes
        if node_offset > 0:
            if thread_idx == 0:
                libshmem_device.signal_wait_until(signals_for_nodes + src_send_node, libshmem_device.NVSHMEM_CMP_EQ, 1)
            __syncthreads()
        src_rank = local_rank + src_send_node * local_world_size
        token_num = tl.load(num_input_tokens_per_rank + src_rank)
        for token_offset in range(global_warp_id, token_num, total_warps):
            for j in range(topk):
                expert_idx = ld(topk_indices_tensor + (src_send_node * max_tokens + token_offset) * topk + j)
                expert_rank = expert_idx // experts_per_rank
                expert_node_idx = expert_rank // local_world_size
                expert_idx_intra_rank = expert_idx % experts_per_rank
                if expert_node_idx == node_id:
                    # TODO(zhengxuegui.0): use warp level put mem
                    if not WITH_SCATTER_INDICES:
                        store_idx = atomic_add_per_warp(
                            recv_buf_offset_per_expert + expert_rank * experts_per_rank * world_size +
                            expert_idx_intra_rank * world_size + src_rank, 1, scope="gpu", semantic="relaxed")
                    else:
                        store_idx = ld(token_dst_scatter_idx + (src_send_node * max_tokens + token_offset) * topk + j)

                    src_ptr = input_buf + src_send_node * max_tokens * hidden_size + token_offset * hidden_size
                    dst_ptr = output_buf + store_idx * hidden_size
                    libshmem_device.putmem_warp(dst_ptr, src_ptr, bytes_per_token, expert_rank)

                    if not WITH_SCATTER_INDICES:
                        st(token_dst_scatter_idx + (src_send_node * max_tokens + token_offset) * topk + j, store_idx)

                    if HAS_WEIGHT:
                        libshmem_device.putmem_warp(
                            weight_recv_buf + store_idx,
                            weight_send_buf + (src_send_node * max_tokens + token_offset) * topk + j, weight_elem_size,
                            expert_rank)


@triton.jit
def kernel_combine_token(
    counter_ptr,
    num_input_tokens_per_rank,  # [world_size]
    send_reqs_in_dispatch,  #[nnodes, 2, max_tokens]
    intra_node_reduce_buf,  # symm buffer (recv token in dispatch stage)
    input_buf,  # symm buffer (recv token in dispatch stage)
    send_buf,  #[nnode, max_tokens, hidden]
    topk_indices_buf,  # [nnodes, max_tokens, topk]
    token_dst_scatter_idx,  # [self.nnodes, self.max_tokens, self.topk]
    max_tokens: int,
    topk: int,
    hidden_size: tl.constexpr,
    bytes_per_token: tl.constexpr,
    expert_per_rank: tl.constexpr,
    local_world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    tl.static_assert(hidden_size <= BLOCK_SIZE, "BLOCK_SIZE must be larger than hidden_size")
    WARP_SIZE = 32

    rank = dl.rank()
    world_size = dl.num_ranks()
    node_id = rank // local_world_size
    nnodes = world_size // local_world_size
    local_rank = rank % local_world_size
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    token_block_offs = tl.arange(0, BLOCK_SIZE)
    token_mask = (token_block_offs[:] < hidden_size)
    thread_idx = tid(0)
    total_warps = num_warps * num_pid
    warp_id = thread_idx // WARP_SIZE
    global_warp_id = pid * num_warps + warp_id

    for node_offset in range(1, nnodes):
        target_node = (node_id + node_offset) % nnodes
        target_rank = local_rank + target_node * local_world_size
        token_num = tl.load(num_input_tokens_per_rank + target_rank)
        for token_idx in range(pid, token_num, num_pid):
            token_accum = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
            for j in range(topk):
                expert_idx = tl.load(topk_indices_buf + (target_node * max_tokens + token_idx) * topk + j)
                expert_rank = expert_idx // expert_per_rank
                expert_node_idx = expert_rank // local_world_size
                if expert_node_idx == node_id:
                    token_scatter_idx = tl.load(token_dst_scatter_idx + (target_node * max_tokens + token_idx) * topk +
                                                j)
                    remote_input_ptr = dl.symm_at(input_buf, expert_rank)
                    remote_input_ptr = tl.multiple_of(remote_input_ptr, 32)
                    token = tl.load(remote_input_ptr + token_scatter_idx * hidden_size + token_block_offs,
                                    mask=token_mask).to(tl.float32)
                    token_accum = token_accum + token
            tl.store(intra_node_reduce_buf + (target_node * max_tokens + token_idx) * hidden_size + token_block_offs,
                     token_accum.to(intra_node_reduce_buf.dtype.element_ty), mask=token_mask)
        # ensure that all threads have finished data writing
        __syncthreads()
        # grid sync
        count = tl.atomic_add(counter_ptr + target_node, 1, scope="gpu", sem="release")  # noqa: F841
        while ld_acquire(counter_ptr + target_node, "gpu") != num_pid:
            pass

        for req_id in range(global_warp_id, token_num, total_warps):
            start_index = ld_b32(send_reqs_in_dispatch + target_node * max_tokens * 2 + req_id)
            end_index = ld_b32(send_reqs_in_dispatch + target_node * max_tokens * 2 + max_tokens + req_id)
            msg_size = (end_index - start_index) * bytes_per_token
            if end_index > start_index:
                src_ptr = intra_node_reduce_buf + target_node * max_tokens * hidden_size + start_index * hidden_size
                dst_ptr = send_buf + node_id * max_tokens * hidden_size + start_index * hidden_size
                libshmem_device.putmem_nbi_warp(dst_ptr, src_ptr, msg_size, target_rank)

    num_dispatch_token_cur_rank = tl.load(num_input_tokens_per_rank + rank)
    # for current node
    for token_idx in range(pid, num_dispatch_token_cur_rank, num_pid):
        token_accum = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        for j in range(topk):
            expert_idx = tl.load(topk_indices_buf + (node_id * max_tokens + token_idx) * topk + j)
            expert_rank = expert_idx // expert_per_rank
            expert_node_idx = expert_rank // local_world_size

            if expert_node_idx == node_id:
                token_scatter_idx = tl.load(token_dst_scatter_idx + (node_id * max_tokens + token_idx) * topk + j)
                remote_input_ptr = dl.symm_at(input_buf, expert_rank)
                remote_input_ptr = tl.multiple_of(remote_input_ptr, 32)
                token = tl.load(remote_input_ptr + token_scatter_idx * hidden_size + token_block_offs,
                                mask=token_mask).to(tl.float32)
                token_accum = token_accum + token

        tl.store(send_buf + (node_id * max_tokens + token_idx) * hidden_size + token_block_offs,
                 token_accum.to(send_buf.dtype.element_ty), token_mask)


@triton.jit(do_not_specialize=["nelems"])
def copy_1d_tilewise_kernel(dst_ptr, src_ptr,  #
                            nelems,  #
                            BLOCK_SIZE: tl.constexpr,  #
                            ):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles = nelems // BLOCK_SIZE

    for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        data = tl.load(src_ptr + offs, volatile=True)
        tl.store(dst_ptr + offs, data)

    if nelems % BLOCK_SIZE:
        if pid == NUM_COPY_SMS - 1:
            offs = num_tiles * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < nelems
            data = tl.load(src_ptr + offs, mask=mask, volatile=True)
            tl.store(dst_ptr + offs, data, mask=mask)


@triton.jit(do_not_specialize=["num_tokens"])
def kernel_get_ag_splits_and_recv_offset(num_tokens,  # int
                                         send_reqs_for_nodes, send_reqs_recv_bufs,
                                         send_reqs_recv_bufs_copy,  # torch tensor, [nnodes, 2, max_tokens]
                                         exp_indices,  # [num_tokens, topk],
                                         topk_indices_buf,  # symm buf, [nnodes, max_tokens, topk]
                                         topk_indices_buf_copy,  # torch tensor, [nnodes, max_tokens, topk]
                                         expert_indices_signal_buf,
                                         local_splits_buf,  # symm buf, [num_experts + 1, ] (with drop token)
                                         full_splits_buf,  # symm buf, [world_size, num_experts + 1]
                                         splits_signal_buf,  # symm buf, [world_size, ]
                                         num_input_tokens_per_rank,  # [world_size, ]
                                         cumsum_input_tokens_per_rank,  # [world_size, ]
                                         num_recv_tokens_per_rank_cpu,  # pin memory, [world_size, ]
                                         cumsum_recv_tokens_per_rank,  # [world_size, ]
                                         recv_buf_offset_per_expert,  # [world_size, experts_per_rank, world_size]
                                         grid_sync_counter,  #[1,] zero init
                                         full_scatter_indices,  # [num_total_tokens, topk]
                                         token_dst_scatter_idx,  #[nnodes, max_tokens, topk]
                                         full_splits_buf_expert_stride, local_world_size, max_tokens, experts_per_rank,
                                         topk: int, BLOCK_SIZE: tl.constexpr,  # larger than num_experts
                                         ):
    # TODO: get topk_indices_buf
    rank = dl.rank()
    world_size = dl.num_ranks()
    nnodes = world_size // local_world_size
    local_rank = rank % local_world_size
    node_id = rank // local_world_size
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_experts = experts_per_rank * world_size
    elem_size = tl.constexpr(local_splits_buf.dtype.element_ty.primitive_bitwidth) // 8
    nbytes = full_splits_buf_expert_stride * elem_size  # num_drop_token is counted in position `num_experts`
    index_elem_size = tl.constexpr(topk_indices_buf.dtype.element_ty.primitive_bitwidth) // 8
    n_warps: tl.constexpr = tl.extra.cuda.num_warps()
    threads_per_block = n_warps * 32
    thread_idx = tid(0)

    for remote_rank in range(pid, world_size, num_pid):
        libshmem_device.putmem_signal_nbi_block(
            full_splits_buf + rank * full_splits_buf_expert_stride,
            local_splits_buf,
            nbytes,
            splits_signal_buf + rank,
            1,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            remote_rank,
        )

    # send expert indices and send_reqs
    if pid == 0:
        for node_offset in range(1, nnodes):
            target_node = (node_id + node_offset) % nnodes
            target_rank = local_rank + target_node * local_world_size
            indices_ptr = topk_indices_buf + node_id * max_tokens * topk
            send_req_src_ptr = send_reqs_for_nodes + target_node * max_tokens * 2
            send_req_dst_ptr = send_reqs_recv_bufs + node_id * max_tokens * 2
            msg_size = max_tokens * 2 * index_elem_size
            libshmem_device.putmem_nbi_block(send_req_dst_ptr, send_req_src_ptr, msg_size, target_rank)
            libshmem_device.fence()
            libshmem_device.putmem_signal_nbi_block(
                indices_ptr,
                indices_ptr,
                index_elem_size * max_tokens * topk,
                expert_indices_signal_buf + rank,
                1,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                target_rank,
            )

    # Ensure that all communication has been completed
    barrier_on_this_grid(grid_sync_counter, False)
    if pid == 0:
        libshmem_device.barrier_all_block()
    barrier_on_this_grid(grid_sync_counter, False)

    offs = tl.arange(0, BLOCK_SIZE)
    full_splits_mask = (offs[:] < full_splits_buf_expert_stride)
    recv_buf_offset_mask = (offs[:] < num_experts)  # do not count drop token

    for target_rank in range(pid, world_size, num_pid):
        # libshmem_device.signal_wait_until(splits_signal_buf + target_rank, libshmem_device.NVSHMEM_CMP_EQ, 1)
        token = dl.wait(splits_signal_buf + target_rank, 1, "sys", "acquire")
        full_splits_buf = dl.consume_token(full_splits_buf, token)
        __syncthreads()
        for expert_idx in range(thread_idx, num_experts, threads_per_block):
            val = ld(full_splits_buf + target_rank * full_splits_buf_expert_stride + expert_idx, semantic="acquire")
            ep_rank = expert_idx // experts_per_rank
            expert_idx_intra_rank = expert_idx % experts_per_rank
            st(
                recv_buf_offset_per_expert + ep_rank * experts_per_rank * world_size +
                expert_idx_intra_rank * world_size + target_rank, val, semantic="release")
        __syncthreads()
        splits_cur_rank = tl.load(full_splits_buf + target_rank * full_splits_buf_expert_stride + offs,
                                  mask=full_splits_mask, other=0, volatile=True)
        total_topk_token_cur_rank = tl.sum(splits_cur_rank)
        num_input_tokens_cur_rank = total_topk_token_cur_rank // topk
        tl.store(num_input_tokens_per_rank + target_rank, num_input_tokens_cur_rank)
        tl.store(cumsum_input_tokens_per_rank + target_rank, num_input_tokens_cur_rank)
        __syncthreads()

    # recv full expert indices and send_reqs
    for node_offset in range(1, nnodes):
        src_node = (node_id + node_offset) % nnodes
        src_rank = local_rank + src_node * local_world_size
        # libshmem_device.signal_wait_until(expert_indices_signal_buf + src_rank, libshmem_device.NVSHMEM_CMP_EQ, 1)
        token = dl.wait(expert_indices_signal_buf + src_rank, 1, "sys", "acquire")
        send_reqs_recv_bufs = dl.consume_token(send_reqs_recv_bufs, token)
        topk_indices_buf = dl.consume_token(topk_indices_buf, token)

        __syncthreads()

    COPY_BLOCK_SIZE: tl.constexpr = 8192
    nelems_send_reqs_recv_bufs = nnodes * 2 * max_tokens
    copy_1d_tilewise_kernel(send_reqs_recv_bufs_copy, send_reqs_recv_bufs, nelems_send_reqs_recv_bufs, COPY_BLOCK_SIZE)
    nelems_topk_indices_buf = nnodes * max_tokens * topk
    copy_1d_tilewise_kernel(topk_indices_buf_copy, topk_indices_buf, nelems_topk_indices_buf, COPY_BLOCK_SIZE)

    # # grid sync
    barrier_on_this_grid(grid_sync_counter, False)

    for ep_rank in range(pid, world_size, num_pid):
        splits_cur_rank = tl.load(recv_buf_offset_per_expert + ep_rank * num_experts + offs, mask=recv_buf_offset_mask,
                                  other=0, volatile=True)
        recv_tokens = tl.sum(splits_cur_rank)
        cusum_splits_cur_rank = tl.cumsum(splits_cur_rank)
        cusum_splits_exclude = cusum_splits_cur_rank - splits_cur_rank
        tl.store(recv_buf_offset_per_expert + ep_rank * num_experts + offs, cusum_splits_exclude,
                 mask=recv_buf_offset_mask)
        tl.store(num_recv_tokens_per_rank_cpu + ep_rank, recv_tokens)
        tl.store(cumsum_recv_tokens_per_rank + ep_rank, recv_tokens)
    __syncthreads()

    # grid sync: wait all pid to save recv_tokens to cumsum_recv_tokens_per_rank
    barrier_on_this_grid(grid_sync_counter, False)

    # # compute cumsum of num_recv_tokens_per_rank_cpu
    if pid == 0:
        # BLOCK_SIZE is larger than num_experts
        rank_mask = (offs[:] < world_size)
        recv_tokens_per_ranks = tl.load(cumsum_recv_tokens_per_rank + offs, mask=rank_mask, other=0, volatile=True)
        cusum_recv_tokens = tl.cumsum(recv_tokens_per_ranks)
        cusum_recv_tokens_exclude = cusum_recv_tokens - recv_tokens_per_ranks
        tl.store(cumsum_recv_tokens_per_rank + offs, cusum_recv_tokens_exclude, mask=rank_mask)

        input_tokens_per_ranks = tl.load(cumsum_input_tokens_per_rank + offs, mask=rank_mask, other=0, volatile=True)
        cusum_input_tokens = tl.cumsum(input_tokens_per_ranks)
        cusum_input_tokens_exclude = cusum_input_tokens - input_tokens_per_ranks
        tl.store(cumsum_input_tokens_per_rank + offs, cusum_input_tokens_exclude, mask=rank_mask)
        __syncthreads()

    barrier_on_this_grid(grid_sync_counter, False)

    if full_scatter_indices:
        tl.static_assert(token_dst_scatter_idx is not None)

        # grid sync: wait cumsum_recv_tokens_per_rank computation
        barrier_on_this_grid(grid_sync_counter, False)

        for target_node_id in range(0, nnodes):
            target_rank = local_rank + target_node_id * local_world_size
            # tokens after topk gate
            tokens_start = tl.load(cumsum_input_tokens_per_rank + target_rank, volatile=True) * topk
            num_tokens_target_rank = tl.load(num_input_tokens_per_rank + target_rank, volatile=True) * topk
            token_dst_scatter_idx_base_ptr = token_dst_scatter_idx + target_node_id * max_tokens * topk
            __syncthreads()
            for token_idx in range(thread_idx + pid * threads_per_block, num_tokens_target_rank,
                                   threads_per_block * num_pid):
                scatter_idx = ld(full_scatter_indices + tokens_start + token_idx)
                expert_idx = ld(topk_indices_buf + (target_node_id * max_tokens * topk) + token_idx)
                expert_rank = expert_idx // experts_per_rank
                # skip drop token
                if expert_rank < world_size:
                    global_out_rank_offset = ld(cumsum_recv_tokens_per_rank + expert_rank)
                    """
                        scatter_idx is a global offset (outputs from all ranks are view as a single flatten buffer).
                        needs to be sub cumsum_recv_tokens_per_rank[expert_rank]
                        to get the index of local output buffer in expert_rank
                    """
                    st(token_dst_scatter_idx_base_ptr + token_idx, scatter_idx - global_out_rank_offset)
            __syncthreads()


@triton.jit(do_not_specialize=["length"])
def kernel_bincount(
    n,
    input,
    output,
    length,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    with dl.simt_exec_region() as (thread_idx, threads_per_block):
        for i in range(pid * threads_per_block + thread_idx, n, num_pid * threads_per_block):
            val = ld(input + i)
            if val < length:
                atomic_add(output + val, 1, scope="gpu", semantic="relaxed")


########################################


def bincount(input, length, output=None, output_dtype=torch.int32, num_sm=16):
    if output is None:
        output = torch.zeros(length, dtype=output_dtype, device=input.device)
    assert input.dim() == 1
    assert output.size(0) >= length
    assert input.is_contiguous()
    n = input.size(0)
    num_warps = 8
    kernel_bincount[(num_sm, )](n, input, output, length, num_warps=num_warps)
    return output


def get_dispatch_send_reqs_for_target_node(token_node_idx, traget_node_id, index_type=torch.int32):
    send_token_mask = (token_node_idx == traget_node_id).any(dim=1).to(torch.int32)
    # padded_mask = F.pad(send_token_mask, (1, 1))
    padded_mask = torch.zeros((send_token_mask.shape[0] + 2, ), dtype=send_token_mask.dtype,
                              device=send_token_mask.device)
    padded_mask[1:-1].copy_(send_token_mask)
    diff = padded_mask[1:] - padded_mask[:-1]

    diff_st = (diff == 1).to(torch.int32)
    diff_ed = (diff == -1).to(torch.int32)
    start_indices = torch.argsort(diff_st, descending=True, stable=True).to(index_type)
    end_indices = torch.argsort(diff_ed, descending=True, stable=True).to(index_type)
    return start_indices[:-1], end_indices[:-1]


def get_ag_splits_and_recv_offset_for_dispatch(send_reqs_for_nodes, send_reqs_recv_bufs, exp_indices, topk_indices_buf,
                                               expert_indices_signal_buf, local_splits, full_splits_buf,
                                               splits_signal_buf, topk, local_world_size, world_size, max_tokens,
                                               experts_per_rank, full_scatter_indices=None, cpu_default_val=-1,
                                               offset_dtype=torch.int32, num_sm=20):
    num_recv_tokens_per_rank_cpu = torch.empty((world_size, ), dtype=torch.int32, device="cpu", pin_memory=True)
    num_recv_tokens_per_rank_cpu.fill_(cpu_default_val)
    # gpu tensor
    num_input_tokens_per_rank = torch.empty((world_size, ), dtype=torch.int32, device=torch.cuda.current_device())
    cumsum_recv_tokens_per_rank = torch.empty((world_size, ), dtype=torch.int32, device=torch.cuda.current_device())
    cumsum_input_tokens_per_rank = torch.empty((world_size, ), dtype=torch.int32, device=torch.cuda.current_device())
    topk_indices_buf_copy = torch.zeros(topk_indices_buf.size(), dtype=topk_indices_buf.dtype,
                                        device=torch.cuda.current_device())
    send_reqs_recv_bufs_copy = torch.zeros(send_reqs_recv_bufs.size(), dtype=send_reqs_recv_bufs.dtype,
                                           device=torch.cuda.current_device())

    token_dst_scatter_idx = None
    if full_scatter_indices is not None:
        assert len(full_scatter_indices.shape) == 2  # [num_total_tokens, topk]
        nnodes = world_size // local_world_size
        assert full_scatter_indices.dtype == offset_dtype
        token_dst_scatter_idx = torch.empty((nnodes, max_tokens, topk), dtype=full_scatter_indices.dtype,
                                            device=full_scatter_indices.device)
    """
    recv_buf_offset_per_expert:
        recv_buf_offset_per_expert[i, j, k] represents the starting offset in the output of all tokens sent by `rank k` to `expert j` on `rank i`.
        This ensures that the tokens sent from all ranks to each expert are continuous in the output,
        which meet the layout requirements of group gemm and avoid post-processing.
    """
    recv_buf_offset_per_expert = torch.zeros((world_size, experts_per_rank, world_size), dtype=offset_dtype,
                                             device=torch.cuda.current_device())
    grid = (num_sm, )
    num_grid_sync = 8
    counter_workspace = torch.zeros((num_grid_sync, ), dtype=torch.int32, device=torch.cuda.current_device())
    assert splits_signal_buf.dtype == NVSHMEM_SIGNAL_DTYPE
    assert len(full_splits_buf.shape) == 2 and full_splits_buf.shape[1] == local_splits.shape[0]
    assert full_splits_buf.shape[0] == world_size

    BLOCK_SIZE = 1 << (full_splits_buf.shape[1]).bit_length()  # the extra one is for drop token
    assert BLOCK_SIZE >= (full_splits_buf.shape[1])
    num_tokens = exp_indices.shape[0]

    kernel_get_ag_splits_and_recv_offset[grid](
        num_tokens,
        send_reqs_for_nodes,
        send_reqs_recv_bufs,
        send_reqs_recv_bufs_copy,
        exp_indices,
        topk_indices_buf,
        topk_indices_buf_copy,
        expert_indices_signal_buf,
        local_splits,
        full_splits_buf,
        splits_signal_buf,
        num_input_tokens_per_rank,
        cumsum_input_tokens_per_rank,
        num_recv_tokens_per_rank_cpu,
        cumsum_recv_tokens_per_rank,
        recv_buf_offset_per_expert,
        counter_workspace,
        full_scatter_indices,
        token_dst_scatter_idx,
        full_splits_buf.shape[1],
        local_world_size,
        max_tokens,
        experts_per_rank,
        topk,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )
    return recv_buf_offset_per_expert, num_recv_tokens_per_rank_cpu, num_input_tokens_per_rank, token_dst_scatter_idx, send_reqs_recv_bufs_copy, topk_indices_buf_copy
