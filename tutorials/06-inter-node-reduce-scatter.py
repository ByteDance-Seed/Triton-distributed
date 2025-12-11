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
"""
Inter-node ReduceScatter
========================
In this tutorial, you will write a multi-node reduce-scatter operation.

.. code-block:: bash

    # To run this tutorial
    source ./scripts/setenv.sh
    bash ./scripts/launch.sh tutorials/06-inter-node-reduce-scatter.py

"""

import torch
import dataclasses
import triton
import triton.language as tl
import triton_dist.language as dl

from typing import Optional, List
import nvshmem.core
from triton_dist.kernels.common_ops import wait_eq
from triton_dist.kernels.nvidia.common_ops import BarrierAllContext, barrier_all_on_stream
from triton_dist.kernels.nvidia.reduce_scatter import ring_reduce
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import initialize_distributed, nvshmem_barrier_all_on_stream, NVSHMEM_SIGNAL_DTYPE, nvshmem_create_tensors, nvshmem_free_tensor_sync

import os


# reduce-scatter context. Stores comm info (dtype, nnodes, etc), symm buffers, and signals.
################### GEMM RS context ###################
@dataclasses.dataclass
class ReduceScatter2DContext:
    max_M: int
    N: int
    rank: int
    world_size: int
    local_world_size: int
    dtype: torch.dtype
    overlap_with_gemm: bool

    # comm buffer
    symm_scatter_bufs: List[torch.Tensor]
    symm_rs_per_node_bufs: List[torch.Tensor]
    symm_p2p_bufs: List[torch.Tensor]

    # barrier bufs
    symm_signal_bufs: List[
        torch.
        Tensor]  # need reset: signal_buf =  scatter_signal | rs_per_node_signal
    barrier: BarrierAllContext

    # stream
    reduction_stream: torch.cuda.Stream
    p2p_stream: torch.cuda.Stream

    # sms
    num_sync_sms: int
    num_p2p_sms: int
    num_reduction_sms: int

    # preprocess to reduce cpu overhead
    # comm barriers
    symm_scatter_signal_bufs: List[torch.Tensor] = dataclasses.field(
        init=False)
    symm_rs_per_node_signal_bufs: List[torch.Tensor] = dataclasses.field(
        init=False)

    local_rank: int = dataclasses.field(init=False)
    node_id: int = dataclasses.field(init=False)
    nnodes: int = dataclasses.field(init=False)

    scatter_signal_buf_list_for_each_node: List[
        torch.Tensor] = dataclasses.field(init=False)

    def __post_init__(self):
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        self.nnodes = self.world_size // self.local_world_size
        self.scatter_signal_buf_list_for_each_node = []
        for buf in self.symm_signal_bufs:
            assert buf.shape[0] >= 2 * self.world_size

        self.symm_scatter_signal_bufs = [
            buf[:self.world_size] for buf in self.symm_signal_bufs
        ]
        self.symm_rs_per_node_signal_bufs = [
            buf[self.world_size:self.world_size * 2]
            for buf in self.symm_signal_bufs
        ]

        for node_id in range(self.nnodes):
            self.scatter_signal_buf_list_for_each_node.append(
                self.symm_scatter_signal_bufs[self.local_rank]
                [node_id * self.local_world_size:(node_id + 1) *
                 self.local_world_size])

    def finalize(self):
        nvshmem_free_tensor_sync(self.symm_scatter_bufs[self.local_rank])
        nvshmem_free_tensor_sync(self.symm_rs_per_node_bufs[self.local_rank])
        nvshmem_free_tensor_sync(self.symm_p2p_bufs[self.local_rank])
        nvshmem_free_tensor_sync(self.symm_signal_bufs[self.local_rank])

    def reset_barriers(self):
        self.symm_signal_bufs[self.local_rank].fill_(0)

    def get_scatter_bufs_and_signal_for_each_node(self, input, node_id):
        M = input.shape[0]
        M_per_rank = M // self.world_size
        M_per_node = M_per_rank * self.local_world_size
        scatter_bufs_intra_node = [
            self.symm_scatter_bufs[i][node_id * M_per_node:(node_id + 1) *
                                      M_per_node]
            for i in range(self.local_world_size)
        ]
        return scatter_bufs_intra_node, self.scatter_signal_buf_list_for_each_node[
            node_id]

    @property
    def symm_rs_per_node_buf(self) -> torch.Tensor:
        return self.symm_rs_per_node_bufs[self.local_rank]

    @property
    def symm_rs_per_node_signal_buf(self) -> torch.Tensor:
        return self.symm_rs_per_node_signal_bufs[self.local_rank]

    @property
    def symm_p2p_buf(self) -> torch.Tensor:
        return self.symm_p2p_bufs[self.local_rank]

    @property
    def num_rs_sms(self) -> int:
        if self.nnodes > 1:
            return self.num_sync_sms + self.num_p2p_sms + self.num_reduction_sms
        else:
            # for intra node rs, no sm required.
            return 0

    @property
    def symm_scatter_signal_buf(self) -> torch.Tensor:
        return self.symm_scatter_signal_bufs[self.local_rank]


def create_reduce_scater_2d_ctx(
        max_M,
        N,
        rank,
        world_size,
        local_world_size,
        dtype,
        overlap_with_gemm=True,
        num_reduction_sms=15) -> ReduceScatter2DContext:
    """
        for num_reduction_sms: tunable param, 16 sms are enough for H800
            For H800, we overlap local reduce and inter-node p2p with intra-node scatter.
            The reduction kernel bandwidth is not a bottleneck if it exceeds 450GB, so only a few SMs are needed.
            For machines with higher intra_node bandwidth(e.g. H100), we may need to increase the number of SMs or redesign overlapping.
    """
    assert world_size % local_world_size == 0
    assert max_M % world_size == 0

    scatter_bufs = nvshmem_create_tensors([max_M, N], dtype, rank,
                                          local_world_size)

    rs_per_node_bufs = nvshmem_create_tensors([max_M // local_world_size, N],
                                              dtype, rank, local_world_size)

    p2p_bufs = nvshmem_create_tensors([max_M // local_world_size, N], dtype,
                                      rank, local_world_size)

    # signal_buf: scatter_signal | rs_per_node_signal
    num_signal_bufs = 2
    signal_bufs = nvshmem_create_tensors([world_size * num_signal_bufs],
                                         NVSHMEM_SIGNAL_DTYPE, rank,
                                         local_world_size)
    signal_buf = signal_bufs[rank]
    signal_buf.zero_()

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    p2p_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)
    reduction_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)

    num_sync_sms = 0
    num_p2p_sms = 1
    ctx = ReduceScatter2DContext(max_M=max_M,
                                 N=N,
                                 rank=rank,
                                 world_size=world_size,
                                 local_world_size=local_world_size,
                                 dtype=dtype,
                                 overlap_with_gemm=overlap_with_gemm,
                                 symm_scatter_bufs=scatter_bufs,
                                 symm_rs_per_node_bufs=rs_per_node_bufs,
                                 symm_p2p_bufs=p2p_bufs,
                                 symm_signal_bufs=signal_bufs,
                                 barrier=BarrierAllContext(True),
                                 reduction_stream=reduction_stream,
                                 p2p_stream=p2p_stream,
                                 num_sync_sms=num_sync_sms,
                                 num_p2p_sms=num_p2p_sms,
                                 num_reduction_sms=num_reduction_sms)
    return ctx


################### triton kernel ###################
@triton.jit
def kernel_inter_node_p2p_for_same_local_rank(
        offset,
        local_world_size,
        M_per_rank,
        N,
        input,  # [M, N]
        output,  # [M, N]
):
    """
    This kernel performs P2P communication, sending corresponding data
    to the GPU with the same local rank on node `cur_node_id + offset + 1`.
    """
    rank = dl.rank()
    world_size = dl.num_ranks()
    node_id = rank // local_world_size
    nnodes = world_size // local_world_size
    local_rank = rank % local_world_size
    nelem_per_rank = M_per_rank * N

    remote_node_id = (offset + 1 + node_id) % nnodes
    remote_rank = local_rank + remote_node_id * local_world_size
    elem_size = tl.constexpr(input.dtype.element_ty.primitive_bitwidth) // 8
    libshmem_device.putmem_block(
        output + node_id * nelem_per_rank,
        input + remote_node_id * nelem_per_rank,
        nelem_per_rank * elem_size,
        remote_rank,
    )


def intra_node_scatter(input_intra_node,
                       scatter_bufs_intra_node: List[torch.Tensor],
                       scatter_signal_buf_intra_node: torch.Tensor,
                       local_rank,
                       stream,
                       overlap_with_gemm=True):
    M, N = input_intra_node.shape
    local_world_size = len(scatter_bufs_intra_node)
    M_per_rank = M // local_world_size
    # send input_intra_node[remote_rank * M_per_rank : (remote_rank + 1) * M_per_rank] on the current rank to
    # input_intra_node[rank * M_per_rank : (rank + 1)] on the remote rank.
    with torch.cuda.stream(stream):
        for i in range(0, local_world_size):
            # Rank level swizzle: Each rank perform scatter start from the next rank of the current.
            # In this way, the send/recv communication volume of each rank is balanced.
            remote_local_rank = (local_rank + i + 1) % local_world_size
            if overlap_with_gemm:
                # wait for the corresponding barrier to be set to 1, indicate the corresponding GEMM tile computation is complete.
                wait_eq(
                    scatter_signal_buf_intra_node[remote_local_rank].data_ptr(
                    ),
                    1,  # signal
                    stream,
                    True)
            remote_buf = scatter_bufs_intra_node[remote_local_rank][
                local_rank * M_per_rank:(local_rank + 1) * M_per_rank, :]
            local_buf = input_intra_node[remote_local_rank *
                                         M_per_rank:(remote_local_rank + 1) *
                                         M_per_rank, :]
            # use copy engine to perform scatter(torch will use `cudamemcpy` to copy continuous data)
            remote_buf.copy_(local_buf)


def reduce_scatter_for_each_node(input, stream, ctx: ReduceScatter2DContext):
    world_size = ctx.world_size
    local_world_size = ctx.local_world_size
    local_rank = ctx.local_rank
    reduction_stream = ctx.reduction_stream
    num_reduction_sms = ctx.num_reduction_sms
    M, N = input.shape
    M_per_rank = M // world_size
    M_per_node = M_per_rank * local_world_size
    nnodes = ctx.nnodes
    node_id = ctx.node_id
    rs_per_node_buf = ctx.symm_rs_per_node_buf
    p2p_buf = ctx.symm_p2p_buf
    # For `target_node_id` in the range [0, nnodes - 1], perform an intra-node reduce-scatter on the
    # input[target_node_id * M_per_node: (target_node_id + 1) * M_per_node]. Then send the result via
    # P2P to the same local rank on the node 'target_node_id'.
    with torch.cuda.stream(stream):
        for n in range(0, nnodes):
            """
            Node-level swizzle: Each node perform intra-node reduce-scatter start from the next node of the current.
            In this way, the P2P communication volume of each node is balanced, and the last intra-node reduce-scatter operation
            is performed on the current node without the need for inter-node communication.

            For time start from 0 to nnode - 1, the communication order between nodes:
                time 0: 0->1, 1->2, 2->3, 3->0
                time 1: 0->2, 1->3, 2->0, 3->1
                time 2: 0->3, 1->0, 2->1, 3->2
                time 3: 0->0, 1->1, 2->2, 3->3
            """
            cur_node_id = (node_id + n + 1) % nnodes
            input_intra_node = input[cur_node_id *
                                     M_per_node:(cur_node_id + 1) * M_per_node]
            scatter_bufs_intra_node, scatter_signal_buf_intra_node = ctx.get_scatter_bufs_and_signal_for_each_node(
                input, cur_node_id)
            # step1: intra node reduce-scatter
            # step1-1: intra node scatter, the corresponding data has been computed by GEMM.
            intra_node_scatter(input_intra_node,
                               scatter_bufs_intra_node,
                               scatter_signal_buf_intra_node,
                               local_rank,
                               stream,
                               overlap_with_gemm=ctx.overlap_with_gemm)

            rs_buf_cur_node = rs_per_node_buf[M_per_rank *
                                              cur_node_id:(cur_node_id + 1) *
                                              M_per_rank]
            # step1-2: perform barrier_all, wait for all peers within the node to complete the scatter operation
            barrier_all_on_stream(ctx.barrier, stream)

            reduction_stream.wait_stream(stream)
            with torch.cuda.stream(reduction_stream):
                # step1-3: perform reduction operation to get the result of the intra-node reduce-scatter.
                ring_reduce(scatter_bufs_intra_node[local_rank],
                            rs_buf_cur_node,
                            local_rank,
                            local_world_size,
                            num_sms=-1 if n == nnodes -
                            1 else num_reduction_sms)

                # step2: inter node p2p, send result to the same local rank on the node `(n + 1 + node_id) % nnodes`.
                if nnodes > 1:
                    if n == nnodes - 1:
                        p2p_buf[M_per_rank * node_id:M_per_rank *
                                (node_id + 1)].copy_(
                                    rs_per_node_buf[M_per_rank *
                                                    node_id:M_per_rank *
                                                    (node_id + 1)])
                    else:
                        grid = lambda META: (ctx.num_p2p_sms, )
                        kernel_inter_node_p2p_for_same_local_rank[grid](
                            n,
                            local_world_size,
                            M_per_rank,
                            N,
                            rs_per_node_buf,
                            p2p_buf,
                            num_warps=16,
                        )

    stream.wait_stream(reduction_stream)
    if nnodes == 1:
        return rs_per_node_buf[:M_per_rank * nnodes]
    return p2p_buf[:M_per_rank * nnodes]


def reduce_scatter_multi_node(input, stream, ctx: ReduceScatter2DContext):
    """
    A hierarchical reduce-scatter implementation that overlaps the intra-node scatter
    with the local reduce and the inter-node p2p(after reduce). It also provides a rank-wise
    signal and supports overlap with gemm.
    """
    M, N = input.shape
    M_per_rank = M // ctx.world_size
    ctx.p2p_stream.wait_stream(stream)
    """
    Step 1: Leveraging the characteristics of reduce-scatter, we first partition the input data according to the target nodes for communication.
            For the data send to each node, we perform an intra-node reduce-scatter operation within the current node.
            Finally, we use P2P communication to send the data to the same local rank on the target node.
            This can reduce the inter-node communication volume by a factor of local_world_size.
    """
    rs_resutl_per_node = reduce_scatter_for_each_node(input, stream, ctx)
    nvshmem_barrier_all_on_stream(stream)
    output = torch.empty((M_per_rank, N),
                         dtype=input.dtype,
                         device=input.device)
    """
    Step 2: After receiving data sent via P2P from all nodes, perform a reduction to get the final result.
    """
    with torch.cuda.stream(stream):
        ring_reduce(rs_resutl_per_node, output, ctx.node_id, ctx.nnodes)
    return output


def reduce_scatter_2d_op(input, ctx: ReduceScatter2DContext):
    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    reduction_stream = ctx.reduction_stream
    M, N = input.shape
    assert input.dtype == ctx.dtype
    assert ctx.max_M >= M and ctx.N == N
    assert M % ctx.world_size == 0

    current_stream = torch.cuda.current_stream()
    reduction_stream.wait_stream(current_stream)
    # Wait for the completion of the previous iteration.
    nvshmem_barrier_all_on_stream(current_stream)

    # perform reduce-scatter
    output = reduce_scatter_multi_node(input, current_stream, ctx)

    # Reset the barriers for the next iteration.
    ctx.reset_barriers()
    return output


def torch_rs(
    input: torch.Tensor,  # [M, N]
    TP_GROUP,
):
    M, N = input.shape
    rs_output = torch.empty((M // WORLD_SIZE, N),
                            dtype=input.dtype,
                            device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, input, group=TP_GROUP)
    return rs_output


if __name__ == "__main__":
    # init
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    TP_GROUP = initialize_distributed()
    torch.cuda.synchronize()

    output_dtype = torch.bfloat16
    M, N = 8192, 16384
    rs_ctx = create_reduce_scater_2d_ctx(M,
                                         N,
                                         RANK,
                                         WORLD_SIZE,
                                         LOCAL_WORLD_SIZE,
                                         output_dtype,
                                         overlap_with_gemm=False)

    # gen input
    input = torch.rand((M, N), dtype=output_dtype).cuda()

    # torch impl
    torch_output = torch_rs(input, TP_GROUP)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    # dist triton impl
    dist_triton_output = reduce_scatter_2d_op(input, rs_ctx)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    # check
    atol, rtol = 6e-2, 6e-2
    torch.testing.assert_close(torch_output,
                               dist_triton_output,
                               atol=atol,
                               rtol=rtol)
    torch.cuda.synchronize()
    print(f"RANK {RANK}: pass!")
    rs_ctx.finalize()
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
