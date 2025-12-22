.. _sphx_glr_getting-started_tutorials_08-overlapping-gemm-reduce-scatter.rst:

Overlapping GEMM ReduceScatter
==============================

In this tutorial, you will write a Multi-node Gemm reduce-scatter operation that is significantly faster
than PyTorch's native op.

In doing so, you will learn about:

* How to overlap reduce-scatter with gemm operations to hide communication.

.. code-block:: bash

    # To run this tutorial
    source ./scripts/setenv.sh
    bash ./scripts/launch.sh tutorials/08-overlapping-gemm-reduce-scatter.py


Overlapping GEMM ReduceScatter Kernel
-------------------------------------

The 'kernel_gemm_rs_producer_persistent' kernel is almost identical to a regular Triton GEMM kernel, with only two minor differences:

    1. The computation order of tiles is swizzled according to the rank.
    2. There is an additional operation to set the barrier in the epilogue.

.. code-block:: Python

    @triton_dist.jit
    def kernel_gemm_rs_producer_persistent(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        barrier_ptr,
        counter_ptr,
        local_world_size,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        EPILOGUE_SUBTILE: tl.constexpr,
        NUM_SMS: tl.constexpr,
    ):
        rank = dl.rank()
        num_ranks = dl.num_ranks()
        dtype = c_ptr.dtype.element_ty
        start_pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
        num_tiles = num_pid_m * num_pid_n
        node_id = rank // local_world_size
        nnodes = num_ranks // local_world_size

        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[N, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
        c_desc = tl.make_tensor_descriptor(
            c_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[
                BLOCK_SIZE_M,
                BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
            ],
        )

        tiles_per_SM = num_tiles // NUM_SMS
        if start_pid < num_tiles % NUM_SMS:
            tiles_per_SM += 1

        tile_id = start_pid - NUM_SMS
        ki = -1

        pid_m = 0
        pid_n = 0
        offs_am = 0
        offs_bn = 0

        M_per_rank = M // num_ranks
        # M_per_rank % BLOCK_SIZE_M == 0 is guaranteed by the caller
        num_pid_m_per_rank = M_per_rank // BLOCK_SIZE_M

        num_pid_in_group = GROUP_SIZE_M * num_pid_n

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for _ in range(0, k_tiles * tiles_per_SM):
            ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
            if ki == 0:
                tile_id += NUM_SMS
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (tile_id % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m

                m_rank = pid_m // num_pid_m_per_rank
                pid_m_intra_rank = pid_m - m_rank * num_pid_m_per_rank
                """
                Difference 1: Based on the m dimension, calculate the target rank where the output data will be scattered to.
                Then, perform a swizzle operation according to the local rank and the node_id of the current GPU.
                This ensures that during communication, the data sent and received by each rank is balanced, maximizing the utilization of all communication bandwidth.
                """
                # original rank and node_id
                m_node_id = m_rank // local_world_size
                m_local_rank = m_rank % local_world_size
                swizzle_m_node_id = (m_node_id + node_id + 1) % nnodes
                swizzle_m_local_rank = (m_local_rank + rank + 1) % local_world_size
                swizzle_m_rank = swizzle_m_node_id * local_world_size + swizzle_m_local_rank

                # perform swizzle
                pid_m = swizzle_m_rank * num_pid_m_per_rank + pid_m_intra_rank

                offs_am = pid_m * BLOCK_SIZE_M
                offs_bn = pid_n * BLOCK_SIZE_N

            offs_k = ki * BLOCK_SIZE_K

            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

            if ki == k_tiles - 1:
                if EPILOGUE_SUBTILE:
                    acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                    acc = tl.permute(acc, (0, 2, 1))
                    acc0, acc1 = tl.split(acc)
                    c0 = acc0.to(dtype)
                    c_desc.store([offs_am, offs_bn], c0)
                    c1 = acc1.to(dtype)
                    c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
                else:
                    c = accumulator.to(dtype)
                    c_desc.store([offs_am, offs_bn], c)
                """
                Difference 2: # Compute the rank that the current tile will be sent
                If the current tile is the last one to complete for that rank, set its barrier to 1 (indicating the ready state).
                the reduce-scatter on another stream waits for the barrier to be ready and then performs the scatter operation.
                """
                counter_start = offs_am // M_per_rank
                counter_end = (offs_am + BLOCK_SIZE_M - 1) // M_per_rank
                counter_end = min(counter_end, num_ranks - 1)
                for counter_id in range(counter_start, counter_end + 1):
                    m_start = M_per_rank * counter_id
                    m_end = M_per_rank * (counter_id + 1) - 1
                    tiled_m_start = m_start // BLOCK_SIZE_M
                    tiled_m_end = m_end // BLOCK_SIZE_M
                    tiled_m_size = tiled_m_end - tiled_m_start + 1
                    tiled_n = tl.cdiv(N, BLOCK_SIZE_N)
                    # `tiled_m_size * tiled_n` represents the total number of tiles within the rank.
                    val = tl.atomic_add(counter_ptr + counter_id, 1, sem="release", scope="gpu")
                    # If `val` is equal to `tiled_m_size * tiled_n - 1`, it means the current tile
                    # is the last one to be completed for this rank.
                    if val == tiled_m_size * tiled_n - 1:
                        dl.notify(barrier_ptr + counter_id, rank, signal=1, comm_scope="gpu")
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    def gemm_rs_producer_persistent(a, b, c, barrier, workspace, world_size, local_world_size, num_gemm_sms,
                                    BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, STAGES=3):
        # Check constraints.
        assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
        assert a.dtype == b.dtype, "Incompatible dtypes"

        M, local_K = a.shape
        N, local_K = b.shape

        M_per_rank = M // world_size

        assert M_per_rank % BLOCK_SIZE_M == 0

        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        grid = lambda META: (min(
            num_gemm_sms,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        # Launch the Triton GEMM kernel. Once the kernel has completed the computation of the output tiles
        # that send to a specific rank, will set the corresponding barrier to 1.
        compiled = kernel_gemm_rs_producer_persistent[grid](
            a,
            b,
            c,
            M,
            N,
            local_K,
            barrier,
            workspace,
            local_world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            False,
            NUM_SMS=num_gemm_sms,  #
            num_stages=STAGES,
            num_warps=8,
        )

        return compiled


Pad the `input` tensor so that all data within a output tile is associated with a single rank. It enables the scatter operation to wait and send data to only one rank at a time, which significantly enhances communication efficiency and simplifies control logic.

.. code-block:: Python

    def padded_to_BLOCK_M(input, world_size, BLOCK_SIZE_M):
        M, local_K = input.shape
        M_per_rank = M // world_size
        pad_size = (M_per_rank + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * BLOCK_SIZE_M
        if pad_size == M_per_rank:
            return input
        input = input.reshape(world_size, M_per_rank, local_K)
        pad_input = torch.empty((world_size, pad_size, local_K), dtype=input.dtype, device=input.device)
        pad_input[:, :M_per_rank].copy_(input)
        pad_input = pad_input.reshape(-1, local_K)
        return pad_input

    def gemm_rs_multi_node_persistent_op(input, weight, ctx: GEMMReduceScatterTensorParallelContext):
        world_size = ctx.rs_ctx.world_size
        local_world_size = ctx.rs_ctx.local_world_size
        rs_stream = ctx.rs_stream
        output_dtype = ctx.output_dtype
        num_gemm_sms = ctx.num_gemm_sms

        orig_M = input.shape[0]
        orig_M_per_rank = orig_M // world_size

        input = padded_to_BLOCK_M(input, world_size, ctx.BLOCK_M)
        M, local_K = input.shape
        N = weight.shape[0]
        assert N == ctx.rs_ctx.N

        assert M % world_size == 0
        assert weight.shape[1] == local_K
        local_M = M // world_size
        current_stream = torch.cuda.current_stream()
        rs_stream.wait_stream(current_stream)

        output = torch.empty((local_M, N), dtype=output_dtype, device=input.device)
        workspace = torch.zeros((world_size, ), dtype=torch.int32, device=input.device)
        gemm_out = ctx.get_gemm_out_buf(input)
        scatter_signal = ctx.rs_ctx.scatter_signal_buf
        """
        Perform the GEMM operation. The output tiles sent to different ranks each correspond to a barrier.
        If the computation of the corresponding tiles is completed, set the barrier to 1.
        """
        gemm_rs_producer_persistent(input, weight, gemm_out, scatter_signal, workspace, world_size, local_world_size,
                                    num_gemm_sms, BLOCK_SIZE_M=ctx.BLOCK_M, BLOCK_SIZE_N=ctx.BLOCK_N,
                                    BLOCK_SIZE_K=ctx.BLOCK_K, GROUP_SIZE_M=ctx.GROUP_M, STAGES=ctx.stages)
        """
        Perform reduce-scatter on the rs_stream, overlapping with the gemm operation.
        This implementation is based on tile level barriers, enabling the overlap of
        communication and computation. Once the data corresponding to each barrier is
        computed(barrier[wait_rank] = 1), the corresponding reduce-scatterwill be perform.
        """
        with torch.cuda.stream(rs_stream):
            output = reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output=output)
        current_stream.wait_stream(rs_stream)

        return output[:orig_M_per_rank]

    def gemm_rs_multi_node(a, b, ctx):
        """GEMM Reduce-Scatter for Multi-Node

        computes local GEMM (a x b) to generate partial results, followed by `reduce_scatter` to produce c

        Args:
            a (torch.Tensor<bfloat16/float16>): local matmul A matrix. shape: [M, local_K]
            b (torch.Tensor<bfloat16/float16>): local matmul B matrix. shape: [N, local_K]
            ctx(GEMMReduceScatterTensorParallelContext): context

        Returns:
            c (torch.Tensor<bfloat16/float16>): local matmul C matrix. shape: [M // world_size, N]
        """
        c = gemm_rs_multi_node_persistent_op(a, b, ctx)
        return c


Benchmark
---------

.. code-block:: Python

    def torch_gemm_rs(
        input: torch.Tensor,  # [M, local_k]
        weight: torch.Tensor,  # [N, local_K]
        TP_GROUP,
    ):
        M, local_K = input.shape
        N = weight.shape[0]
        output = torch.matmul(input, weight.T)
        rs_output = torch.empty((M // WORLD_SIZE, N), dtype=output.dtype, device=input.device)
        torch.distributed.reduce_scatter_tensor(rs_output, output, group=TP_GROUP)
        return rs_output


    if __name__ == "__main__":
        if torch.cuda.get_device_capability()[0] < 9:
            print("Skip the test because the device is not sm90 or higher")
            import sys
            sys.exit()

        # init
        RANK = int(os.environ.get("RANK", 0))
        LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
        WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
        LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        TP_GROUP = triton_dist.utils.initialize_distributed()
        torch.cuda.synchronize()
        M, N, K = 16384, 12288, 49152
        local_K = K // TP_GROUP.size()

        # gen input
        input_dtype = torch.bfloat16
        output_dtype = input_dtype
        scale = TP_GROUP.rank() + 1

        def _make_data(M):
            data_config = [((M, local_K), input_dtype, (0.01 * scale, 0)),  # A
                        ((N, local_K), input_dtype, (0.01 * scale, 0)),  # B
                        ]
            generator = generate_data(data_config)
            input, weight = next(generator)
            return input, weight

        input, weight = _make_data(M)

        # create context for dist triton
        rs_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)
        dist_gemm_rs_ctx = create_gemm_rs_context(M, N, RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, output_dtype, rs_stream)

        # torch impl
        torch_output, torch_perf = perf_func(partial(torch_gemm_rs, input, weight, TP_GROUP), iters=100, warmup_iters=20)

        nvshmem_barrier_all_on_stream()
        torch.cuda.synchronize()

        # dist triton impl
        dist_triton_output, dist_triton_perf = perf_func(partial(gemm_rs_multi_node, input, weight, dist_gemm_rs_ctx),
                                                        iters=100, warmup_iters=20)

        nvshmem_barrier_all_on_stream()
        torch.cuda.synchronize()

        # check
        atol, rtol = 6e-2, 6e-2
        torch.testing.assert_close(torch_output, dist_triton_output, atol=atol, rtol=rtol)
        torch.cuda.synchronize()

        # perf
        dist_print(f"dist-triton #{RANK}", dist_triton_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        dist_print(f"torch #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

        dist_gemm_rs_ctx.finalize()
        nvshmem.core.finalize()
        torch.distributed.destroy_process_group()