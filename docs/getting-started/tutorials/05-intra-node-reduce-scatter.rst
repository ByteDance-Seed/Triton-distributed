.. _sphx_glr_getting-started_tutorials_05-intra-node-reduce-scatter.rst:

Intra-node ReduceScatter
========================

In this tutorial, you will write a intra-node reduce-scatter operation.

In doing so, you will learn about:

* use copy engine to communicate data within a node
* How to write an intra-node reduce-scatter kernel.

.. code-block:: bash

    # To run this tutorial
    source ./scripts/setenv.sh
    bash ./scripts/launch.sh tutorials/05-intra-node-reduce-scatter.py


Reduce Kernel
-------------

The difference between this kernel and a normal reduction operation lies in the accumulation order.
In a normal reduction, the accumulation starts from the 0th row.
This kernel starts the accumulation from the `offset`.

.. code-block:: Python

    @triton.jit
    def kernel_ring_reduce(
        c_ptr,  # [M, N]
        out_ptr,  # [M_per_split, N]
        # shape of matrix
        M_per_rank,
        N,
        begin_idx,
        num_splits: tl.constexpr,
        # reduce tile shape
        BLOCK_SIZE_M: tl.constexpr = 256,
        BLOCK_SIZE_N: tl.constexpr = 64,
    ):
        c_desc = tl.make_tensor_descriptor(
            c_ptr,
            shape=[M_per_rank * num_splits, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )
        output_desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[M_per_rank, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        pid = tl.program_id(axis=0)
        num_pid = tl.num_programs(axis=0)
        num_tiles_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
        num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
        total_tiles = num_tiles_m * num_tiles_n
        for tile_id in range(pid, total_tiles, num_pid):
            tile_id_m = tile_id // num_tiles_n
            tile_id_n = tile_id % num_tiles_n
            # accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=out_ptr.dtype.element_ty)
            cur_rank = (begin_idx + 1) % num_splits
            accum = c_desc.load([tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N])
            for i in range(1, num_splits):
                cur_rank = (i + begin_idx + 1) % num_splits
                data = c_desc.load([tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N])
                accum += data

            output_desc.store([tile_id_m * BLOCK_SIZE_M, tile_id_n * BLOCK_SIZE_N], accum)

    
.. code-block:: Python

    def ring_reduce(
        input,  # [M_per_node, N]
        output,  # [M_per_rank, N]
        begin_idx,
        num_splits,
        stream,
        num_sms=-1,
    ):
        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)
        total_M, N = input.shape
        M_per_split = total_M // num_splits
        assert output.shape[0] == M_per_split and total_M % num_splits == 0
        if num_sms == -1:
            grid = lambda META: (triton.cdiv(M_per_split, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
            with torch.cuda.stream(stream):
                kernel_ring_reduce[grid](
                    input,
                    output,
                    M_per_split,
                    N,
                    begin_idx,
                    num_splits,
                    BLOCK_SIZE_M=256,
                    BLOCK_SIZE_N=64,
                    num_warps=4,
                )
        else:
            grid = lambda META: (min(
                triton.cdiv(M_per_split, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), num_sms), )
            with torch.cuda.stream(stream):
                kernel_ring_reduce[grid](
                    input,
                    output,
                    M_per_split,
                    N,
                    begin_idx,
                    num_splits,
                    BLOCK_SIZE_M=256,
                    BLOCK_SIZE_N=128,
                    num_warps=8,
                )

        return output


Scatter Kernel
--------------

We perform rank level swizzle in the scatter kernel. Each rank perform scatter start from the next rank of the current. In this way, the send/recv communication volume of each rank is balanced.
For time start from 0 to local_world_size, the communication order between ranks:

- time 0: 0->1, 1->2, 2->3, 3->0
- time 1: 0->2, 1->3, 2->0, 3->1
- time 2: 0->3, 1->0, 2->1, 3->2
- time 3: 0->0, 1->1, 2->2, 3->3  

.. code-block:: Python

    def intra_node_scatter(input_intra_node, scatter_bufs_intra_node: List[torch.Tensor], local_rank, stream):
        M, N = input_intra_node.shape
        local_world_size = len(scatter_bufs_intra_node)
        M_per_rank = M // local_world_size

        # send input_intra_node[remote_rank * M_per_rank : (remote_rank + 1) * M_per_rank] on the current rank to
        # input_intra_node[rank * M_per_rank : (rank + 1)] on the remote rank.
        with torch.cuda.stream(stream):
            for i in range(0, local_world_size):
                remote_local_rank = (local_rank + i + 1) % local_world_size

                remote_buf = scatter_bufs_intra_node[remote_local_rank][local_rank * M_per_rank:(local_rank + 1) *
                                                                        M_per_rank, :]
                local_buf = input_intra_node[remote_local_rank * M_per_rank:(remote_local_rank + 1) * M_per_rank, :]
                # use copy engine to perform scatter(torch will use `cudamemcpy` to copy continuous data)
                remote_buf.copy_(local_buf)


Reduce-Scatter
--------------

.. code-block:: Python

    @p2p_native_atomic_required
    def reducer_scatter_intra_node(input, scatter_bufs, sync_buf, local_rank, local_world_size):

        stream = torch.cuda.current_stream()
        M, N = input.shape
        M_per_rank = M // local_world_size

        output = torch.empty((M_per_rank, N), dtype=dtype, device=input.device)
        # step 1: intra node reduce-scatter
        intra_node_scatter(input, scatter_bufs, local_rank, stream)

        # step 2: waits for all ranks to complete the scatter.
        barrier_all_intra_node_atomic_cas_block[(1, )](local_rank, local_rank, local_world_size, sync_buf)
        # step 3: perform reduction to get the result of the intra-node reduce-scatter.
        ring_reduce(scatter_bufs[local_rank], output, local_rank, local_world_size, stream)
        return output

Benckmark
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
        TP_GROUP = triton_dist.utils.initialize_distributed()
        torch.cuda.synchronize()

        assert LOCAL_WORLD_SIZE == WORLD_SIZE, "runs on 1 node expected."

        dtype = torch.bfloat16
        M, N = 8192, 16384

        input = torch.rand((M, N), dtype=dtype).cuda()

        symm_scatter_bufs = nvshmem_create_tensors([M, N], dtype, RANK, LOCAL_WORLD_SIZE)
        symm_sync_buf = nvshmem_create_tensor((LOCAL_WORLD_SIZE, ), dtype=torch.int32)
        symm_sync_buf.fill_(0)

        torch_output = torch_rs(input, TP_GROUP)

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        dist_triton_output = reducer_scatter_intra_node(input, symm_scatter_bufs, symm_sync_buf, LOCAL_RANK,
                                                        LOCAL_WORLD_SIZE)

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

        atol, rtol = 6e-2, 6e-2
        torch.testing.assert_close(torch_output, dist_triton_output, atol=atol, rtol=rtol)
        torch.cuda.synchronize()
        print(f"RANK {LOCAL_RANK}: pass!")

        nvshmem_free_tensor_sync(symm_sync_buf)
        nvshmem_free_tensor_sync(symm_scatter_bufs[LOCAL_RANK])
        nvshmem.core.finalize()
        torch.distributed.destroy_process_group()