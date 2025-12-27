.. _sphx_glr_getting-started_tutorials_06-inter-node-reduce-scatter.rst:

Inter-node ReduceScatter
========================

In this tutorial, you will write a multi-node reduce-scatter operation.

.. code-block:: bash

    # To run this tutorial
    source ./scripts/setenv.sh
    bash ./scripts/launch.sh tutorials/06-inter-node-reduce-scatter.py


Kernel
------

.. code-block:: Python

    @triton_dist.jit
    def kernel_inter_node_p2p_for_same_local_rank(offset, local_world_size, M_per_rank, N, input,  # [M, N]
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


    def intra_node_scatter(input_intra_node, scatter_bufs_intra_node: List[torch.Tensor],
                       scatter_signal_buf_intra_node: torch.Tensor, local_rank, stream, overlap_with_gemm=True):
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
                    wait_eq(scatter_signal_buf_intra_node[remote_local_rank].data_ptr(), 1,  # signal
                            stream, True)
                remote_buf = scatter_bufs_intra_node[remote_local_rank][local_rank * M_per_rank:(local_rank + 1) *
                                                                        M_per_rank, :]
                local_buf = input_intra_node[remote_local_rank * M_per_rank:(remote_local_rank + 1) * M_per_rank, :]
                # use copy engine to perform scatter(torch will use `cudamemcpy` to copy continuous data)
                remote_buf.copy_(local_buf)

We perform node-level swizzle in the reduce-scatter kernel. Each node perform intra-node reduce-scatter start from the next node of the current.
In this way, the P2P communication volume of each node is balanced, and the last intra-node reduce-scatter operation
is performed on the current node without the need for inter-node communication.

For time start from 0 to nnode - 1, the communication order between nodes:

    time 0: 0->1, 1->2, 2->3, 3->0

    time 1: 0->2, 1->3, 2->0, 3->1

    time 2: 0->3, 1->0, 2->1, 3->2

    time 3: 0->0, 1->1, 2->2, 3->3

.. code-block:: Python

    def reducer_scatter_for_each_node(input, stream, ctx: ReduceScatter2DContext):
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
                cur_node_id = (node_id + n + 1) % nnodes
                input_intra_node = input[cur_node_id * M_per_node:(cur_node_id + 1) * M_per_node]
                scatter_bufs_intra_node, scatter_signal_buf_intra_node = ctx.get_scatter_bufs_and_signal_for_each_node(
                    input, cur_node_id)
                # step1: intra node reduce-scatter
                # step1-1: intra node scatter, the corresponding data has been computed by GEMM.
                intra_node_scatter(input_intra_node, scatter_bufs_intra_node, scatter_signal_buf_intra_node, local_rank,
                                stream, overlap_with_gemm=ctx.overlap_with_gemm)

                rs_buf_cur_node = rs_per_node_buf[M_per_rank * cur_node_id:(cur_node_id + 1) * M_per_rank]
                # step1-2: perform barrier_all, wait for all peers within the node to complete the scatter operation
                barrier_all_on_stream(ctx.barrier, stream)

                reduction_stream.wait_stream(stream)
                with torch.cuda.stream(reduction_stream):
                    # step1-3: perform reduction operation to get the result of the intra-node reduce-scatter.
                    ring_reduce(scatter_bufs_intra_node[local_rank], rs_buf_cur_node, local_rank, local_world_size,
                                num_sms=-1 if n == nnodes - 1 else num_reduction_sms)

                    # step2: inter node p2p, send result to the same local rank on the node `(n + 1 + node_id) % nnodes`.
                    if nnodes > 1:
                        if n == nnodes - 1:
                            p2p_buf[M_per_rank * node_id:M_per_rank * (node_id + 1)].copy_(
                                rs_per_node_buf[M_per_rank * node_id:M_per_rank * (node_id + 1)])
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


A hierarchical reduce-scatter implementation that overlaps the intra-node scatter with the local reduce and the inter-node p2p(after reduce). It also provides a rank-wise signal and supports overlap with gemm.
    
.. code-block:: Python

    def reduce_scatter_multi_node(input, stream, ctx: ReduceScatter2DContext):
        M, N = input.shape
        M_per_rank = M // ctx.world_size
        ctx.p2p_stream.wait_stream(stream)
        """
        Step 1: Leveraging the characteristics of reduce-scatter, we first partition the input data according to the target nodes for communication.
                For the data send to each node, we perform an intra-node reduce-scatter operation within the current node.
                Finally, we use P2P communication to send the data to the same local rank on the target node.
                This can reduce the inter-node communication volume by a factor of local_world_size.
        """
        rs_resutl_per_node = reducer_scatter_for_each_node(input, stream, ctx)
        nvshmem_barrier_all_on_stream(stream)
        output = torch.empty((M_per_rank, N), dtype=input.dtype, device=input.device)
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


Benchmark
---------

.. code-block:: Python

    def torch_rs(
        input: torch.Tensor,  # [M, N]
        TP_GROUP,
    ):
        M, N = input.shape
        rs_output = torch.empty((M // WORLD_SIZE, N), dtype=input.dtype, device=input.device)
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
        rs_ctx = create_reduce_scater_2d_ctx(M, N, RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, output_dtype,
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
        torch.testing.assert_close(torch_output, dist_triton_output, atol=atol, rtol=rtol)
        torch.cuda.synchronize()
        print(f"RANK {RANK}: pass!")
        rs_ctx.finalize()
        nvshmem.core.finalize()
        torch.distributed.destroy_process_group()

