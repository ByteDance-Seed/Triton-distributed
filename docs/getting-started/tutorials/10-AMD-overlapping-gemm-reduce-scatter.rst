.. _sphx_glr_getting-started_tutorials_10-AMD-overlapping-gemm-reduce-scatter.rst:

Overlapping GEMM ReduceScatter on AMD GPU
=========================================

In this tutorial, you will write a fused Gemm and ReduceScatter Op using Triton-distributed.

In doing so, you will learn about:

* How to overlap reduce-scatter with gemm operations to hide communication on AMD GPUs.

.. code-block:: bash

    # To run this tutorial
    bash ./scripts/launch_amd.sh ./tutorials/10-AMD-overlapping-gemm-reduce-scatter.py


Kernel
------

.. code-block:: Python

    @triton.jit
    def kernel_gemm_rs_producer_fuse_scatter(
            # Pointers to matrices
            a_ptr, b_ptr, scatter_bufs_ptr, rank, num_ranks,
            # Matrix dimensions
            M, N, K,
            # The stride variables represent how much to increase the ptr by when moving by 1
            # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
            # by to get the element one row down (A has M rows).
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,
            # Meta-parameters
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
            GROUP_SIZE_M: tl.constexpr, M_PER_COPY_CHUNK: tl.constexpr  #
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.

        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_bn > 0)
        tl.assume(stride_cm > 0)
        tl.assume(stride_cn > 0)

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # rank swizzle
        M_per_rank = M // num_ranks
        num_pid_m_per_copy_chunk = M_PER_COPY_CHUNK // BLOCK_SIZE_M
        chunk_offset = pid_m // (num_pid_m_per_copy_chunk * num_ranks)
        rank_offset = pid_m % (num_pid_m_per_copy_chunk * num_ranks) // num_pid_m_per_copy_chunk
        block_offset = pid_m % num_pid_m_per_copy_chunk

        rank_offset = (rank_offset + rank + 1) % num_ranks
        pid_m = (rank_offset * M_per_rank + chunk_offset * M_PER_COPY_CHUNK + block_offset * BLOCK_SIZE_M) // BLOCK_SIZE_M

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            # We accumulate along the K dimension.
            accumulator = tl.dot(a, b, accumulator)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        # You can fuse arbitrary activation functions here
        # while the accumulator is still in FP32!
        dtype = a_ptr.dtype.element_ty
        c = accumulator.to(dtype)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        target_m = ((pid_m * BLOCK_SIZE_M % M_per_rank) + M_per_rank * rank)
        offs_cm = target_m + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptr = tl.load(scatter_bufs_ptr + rank_offset).to(tl.pointer_type(dtype))
        c_ptr = tl.multiple_of(c_ptr, 16)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


.. code-block:: Python

    @triton.jit
    def kernel_consumer_reduce(
        c_ptr,  # [M, N]
        out_ptr,  # [M_per_rank, N]
        # shape of matrix
        M_per_rank,
        N,
        rank,
        num_ranks: tl.constexpr,
        # tile size
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs = tl.where(offs < M_per_rank * N, offs, 0)
        out_ptrs = out_ptr + offs

        accum = tl.zeros((BLOCK_SIZE, ), dtype=out_ptr.dtype.element_ty)
        for i in range(0, num_ranks):
            cur_rank = (i + rank + 1) % num_ranks
            c_ptrs = c_ptr + offs + cur_rank * M_per_rank * N
            data = tl.load(c_ptrs)
            accum += data

        tl.store(out_ptrs, accum)


    def ring_reduce_after_scatter(
        rank,
        num_ranks,
        scatter_out,  # [M, N]
        stream,
    ):
        M, N = scatter_out.shape
        M_per_rank = M // num_ranks
        output = torch.empty((M_per_rank, N), dtype=scatter_out.dtype, device=scatter_out.device)
        grid = lambda META: (triton.cdiv(M_per_rank * N, META["BLOCK_SIZE"]), )
        with torch.cuda.stream(stream):
            kernel_consumer_reduce[grid](
                scatter_out,
                output,
                M_per_rank,
                N,
                rank=rank,
                num_ranks=num_ranks,
                BLOCK_SIZE=2048,
                num_warps=2,
            )

        return output


GEMM RS Class
-------------

.. code-block:: Python

    class triton_gemm_rs_intra_node(torch.nn.Module):

        def __init__(
            self,
            tp_group: torch.distributed.ProcessGroup,
            max_M: int,
            N: int,
            K: int,
            input_dtype: torch.dtype,
            output_dtype: torch.dtype,
            fuse_scatter: bool = True,
        ):
            self.tp_group = tp_group
            self.rank: int = tp_group.rank()
            self.world_size = tp_group.size()
            self.max_M: int = max_M
            self.N = N
            self.K = K
            self.input_dtype = input_dtype
            self.output_dtype = output_dtype
            self.fuse_scatter = fuse_scatter

            # Use the auxiliary functions provided by Triton-distributed to construct the context required for GEMM-RS.
            # This simplifies the code logic. The context mainly includes:
            # (1) The globally symmetric memory required;
            # (2) The signals used for communication between prodcuer and consumer;
            # (3) Scatter streams.
            self.ctx = create_gemm_rs_intra_node_context(
                self.max_M,
                self.N,
                self.output_dtype,
                self.rank,
                self.world_size,
                self.tp_group,
                self.fuse_scatter,
            )

        def forward(self, input: torch.Tensor,  # [M, local_K]
                    weight: torch.Tensor,  # [N, local_K]
                    ):

            ctx = self.ctx
            M, local_K = input.shape
            N, K = weight.shape
            stride_bk, stride_bn = weight.stride(1), weight.stride(0)
            assert K == local_K
            M_per_rank = M // ctx.num_ranks

            current_stream = torch.cuda.current_stream()
            barrier_all_on_stream(ctx.rank, ctx.num_ranks, ctx.sync_bufs_ptr, current_stream)

            output = torch.empty((M_per_rank, N), dtype=output_dtype, device=input.device)
            alignment = 256
            assert M % alignment == 0 and N % alignment == 0 and K % alignment == 0

            # producer gemm fused scatter
            grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
            kernel_gemm_rs_producer_fuse_scatter[grid](
                input,
                weight,
                ctx.scatter_bufs_ptr,
                ctx.rank,
                ctx.num_ranks,
                M,
                N,
                K,
                input.stride(0),
                input.stride(1),
                stride_bk,
                stride_bn,
                N,
                1,
            )

            scatter_out = ctx.scatter_bufs[ctx.rank][:M]

            # barrier all to wait for gemm finish
            barrier_all_on_stream(ctx.rank, ctx.num_ranks, ctx.sync_bufs_ptr, current_stream)

            # consumer reduction
            output = ring_reduce_after_scatter(ctx.rank, ctx.num_ranks, scatter_out, current_stream)

            return output

Benchmark
---------

.. code-block:: Python

    def torch_gemm_rs(
        input: torch.Tensor,  # [M, local_k]
        weight: torch.Tensor,  # [N, local_K]
        bias: Optional[torch.Tensor],
        TP_GROUP,
    ):
        M, local_K = input.shape
        world_size = TP_GROUP.size()
        N, _ = weight.shape
        output = torch.matmul(input, weight.T)
        if bias:
            output = output + bias
        rs_output = torch.empty((M // world_size, N), dtype=output.dtype, device=input.device)
        torch.distributed.reduce_scatter_tensor(rs_output, output, group=TP_GROUP)
        return rs_output


    def init():
        RANK = int(os.environ.get("RANK", 0))
        LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
        WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(LOCAL_RANK)
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=WORLD_SIZE,
            rank=RANK,
            timeout=datetime.timedelta(seconds=1800),
        )
        assert torch.distributed.is_initialized()
        TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
        torch.distributed.barrier(TP_GROUP)

        torch.manual_seed(3 + RANK)
        torch.cuda.manual_seed_all(3 + RANK)

        torch.cuda.synchronize()
        torch.distributed.barrier()

        return RANK, LOCAL_RANK, WORLD_SIZE, TP_GROUP


    def destroy():
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


    if __name__ == "__main__":
        # init
        RANK, LOCAL_RANK, WORLD_SIZE, TP_GROUP = init()

        # NOTE: We should get device after process group init.
        DEVICE = triton.runtime.driver.active.get_active_torch_device()

        dtype = torch.float16
        M = 8192
        N = 4096
        K = 12288
        local_K = K // WORLD_SIZE
        input_dtype = dtype
        output_dtype = input_dtype
        atol = 1e-2
        rtol = 1e-2

        # Generate input and weight.
        scale = TP_GROUP.rank() + 1
        data_config = [((M, local_K), dtype, (0.01 * scale, 0), DEVICE),  # input
                    ((N, local_K), dtype, (0.01 * scale, 0), DEVICE),  # weight
                    (None),  # bias
                    ]
        generator = generate_data(data_config)
        input, weight, bias = next(generator)

        # torch impl
        ref_out = torch_gemm_rs(input, weight, False, bias, TP_GROUP)
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # dist triton impl
        dist_gemm_rs_op = triton_gemm_rs_intra_node(TP_GROUP, M, N, K, input_dtype, output_dtype)
        tri_out = dist_gemm_rs_op.forward(input, weight)

        if torch.allclose(tri_out, ref_out, atol=atol, rtol=rtol):
            dist_print("✅ Triton and Torch match")
        else:
            dist_print(f"The maximum difference between torch and triton is {torch.max(torch.abs(tri_out - ref_out))}")
            dist_print("❌ Triton and Torch differ")

        # Finally destroy distributed process group.
        destroy()
