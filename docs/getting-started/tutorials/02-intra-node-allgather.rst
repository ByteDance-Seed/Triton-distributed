.. _sphx_glr_getting-started_tutorials_02-intra-node-allgather.rst:

Intra-node AllGather
====================

In this tutorial, you will write a distributed AllGather kernel using Triton-distributed.

In doing so, you will learn about:

* Writing the AllGather kernel with symmetric pointers directly.

* Writing the AllGather kernel with NVSHMEM device functions.

.. code-block:: bash

    # To run this tutorial
    source ./scripts/setenv.sh
    bash ./scripts/launch.sh tutorials/02-intra-node-allgather.py


Kernel
------

There are several communication methods available for intra-node communication: it can be achieved directly through Memory Copy interfaces (utilizing the copy engine), via kernel ld/st operations (using SMs), or with NVSHMEM primitives (also using SMs). We recommend using either the Memory Copy interface or NVSHMEM primitives. In terms of performance, both methods can achieve comparable results. The key difference is that Memory Copy does not occupy SM resources.

Let's introduce the Memory Copy interface first. The only difference from a regular PyTorch program is the ``remote_tensor_buffers`` parameter. This parameter is a list of Tensors, where each element corresponds to the Tensor at the respective rank position. This parameter is obtained through NVSHMEM's host interface:


.. code-block:: Python

    from triton_dist.utils import nvshmem_create_tensors
    
    # Create symmetric tensors accessible from all ranks
    symm_ag_buffers = nvshmem_create_tensors((M, N), dtype, rank, LOCAL_WORLD_SIZE)


.. code-block:: Python

    import os
    from typing import List

    import nvshmem.core
    import torch
    from triton_dist.utils import cuda

    import triton_dist
    import triton.language as tl
    from triton_dist.language.extra import libshmem_device
    from triton_dist.utils import (
        CUDA_CHECK,
        dist_print,
        initialize_distributed,
        nvshmem_barrier_all_on_stream,
        NVSHMEM_SIGNAL_DTYPE,
        nvshmem_create_tensors,
        nvshmem_free_tensor_sync,
    )

    def cp_engine_producer_all_gather_full_mesh_pull(
        rank,
        num_ranks,
        local_tensor: torch.Tensor,
        remote_tensor_buffers: List[torch.Tensor],
        ag_stream: torch.cuda.Stream,
        barrier_buffers: List[torch.Tensor],
    ):
        M_per_rank, N = local_tensor.shape

        rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

        with torch.cuda.stream(ag_stream):
            for src_rank in rank_orders:
                if src_rank == rank:
                    continue
                # peer: src_rank, offset src_rank[src_rank] -> rank[src_rank]
                dst = remote_tensor_buffers[rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
                src = remote_tensor_buffers[src_rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
                dst.copy_(src)
                (err, ) = cuda.cuStreamWriteValue32(
                    ag_stream.cuda_stream,
                    barrier_buffers[rank][src_rank].data_ptr(),
                    1,
                    cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
                )
                CUDA_CHECK(err)


For kernels with NVSHMEM primitives, we use the ``triton_dist.jit`` decorator. In this example, we use 8 SMs (``DISPATCH_BLOCK_NUM = 8``) to perform AllGather. Each block is responsible for sending its local data to the other 7 GPUs.


.. code-block:: Python

    @triton_dist.jit
    def nvshmem_device_producer_all_gather_2d_put_block_kernel(
        remote_tensor_ptr,
        signal_buffer_ptr,
        elem_per_rank,
        size_per_elem,
        signal_target,
        local_rank,
        world_size,
        DISPATCH_BLOCK_NUM: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)

        if pid < DISPATCH_BLOCK_NUM:  # intra dispatch block
            peer = (local_rank + pid + 1) % world_size
            segment = local_rank
            libshmem_device.putmem_signal_block(  # send the segment to the peer and notify the segment is ready
                remote_tensor_ptr + segment * elem_per_rank,
                remote_tensor_ptr + segment * elem_per_rank,
                elem_per_rank * size_per_elem,
                signal_buffer_ptr + segment,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )


Test the Correctness
--------------------


.. code-block:: Python
    
    if __name__ == "__main__":
        TP_GROUP = initialize_distributed()
        rank = TP_GROUP.rank()
        num_ranks = TP_GROUP.size()
        LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE"))
        assert num_ranks == LOCAL_WORLD_SIZE, "This tutorial is designed for intra-node"

        M = 8192
        N = 12288
        M_per_rank = M // num_ranks
        dtype = torch.float16

        local_data = torch.randn([M_per_rank, N], dtype=dtype, device="cuda")
        # Create symmetric tensors using the new API
        symm_ag_buffers = nvshmem_create_tensors((M, N), dtype, rank, LOCAL_WORLD_SIZE)
        symm_ag_buffer = symm_ag_buffers[rank]
        symm_signals = nvshmem_create_tensors((num_ranks, ), NVSHMEM_SIGNAL_DTYPE, rank, LOCAL_WORLD_SIZE)
        symm_signal = symm_signals[rank]
        
        # Calculate golden
        golden = torch.empty([M, N], dtype=dtype, device="cuda")
        torch.distributed.all_gather_into_tensor(golden, local_data, group=TP_GROUP)

        #####################
        # Copy Engine
        symm_ag_buffer.fill_(-1)  # reset buffer
        symm_ag_buffer[
            rank * M_per_rank:(rank + 1) * M_per_rank,
        ].copy_(local_data)  # copy local data to symmetric memory for communication
        symm_signal.fill_(0)  # The initial value of signal should be 0s
        # We need barrier all to make sure the above initialization visible to other ranks
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        cp_engine_producer_all_gather_full_mesh_pull(
            rank, num_ranks, local_data, symm_ag_buffers, torch.cuda.current_stream(),
            symm_signals)  # Here we use current stream for allgather, we can pass any other stream for comm-comp fusion.

        # Check results. Pull mode doesn't need sync after communication
        dist_print(f"Rank {rank} CpEngine Result:\n", symm_ag_buffer, need_sync=True, allowed_ranks="all")
        dist_print(f"Rank {rank} CpEngine Signal:\n", symm_signal, need_sync=True, allowed_ranks="all")
        assert torch.allclose(golden, symm_ag_buffer, atol=1e-5, rtol=1e-5)
        dist_print(f"Rank {rank}", "Pass!✅", need_sync=True, allowed_ranks="all")

        #####################
        # NVSHMEM Primitives
        symm_ag_buffer.fill_(-1)  # reset buffer
        symm_ag_buffer[
            rank * M_per_rank:(rank + 1) * M_per_rank,
        ].copy_(local_data)  # copy local data to symmetric memory for communication
        symm_signal.fill_(0)  # The initial value of signal should be 0s
        # We need barrier all to make sure the above initialization visible to other ranks
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        grid = lambda META: (int(num_ranks), )
        nvshmem_device_producer_all_gather_2d_put_block_kernel[grid](
            symm_ag_buffer, symm_signal, M_per_rank * N,  # No. of elems of local data
            local_data.element_size(),  # element size
            1,  # signal target, can be any other value in practice
            rank, num_ranks, num_ranks)
        # Need to sync all to guarantee the completion of communication
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        # Check results. Pull mode doesn't need sync after communication
        dist_print(f"Rank {rank} NVSHMEM Result:\n", symm_ag_buffer, need_sync=True, allowed_ranks="all")
        dist_print(f"Rank {rank} NVSHMEM Signal:\n", symm_signal, need_sync=True, allowed_ranks="all")
        assert torch.allclose(golden, symm_ag_buffer, atol=1e-5, rtol=1e-5)
        dist_print(f"Rank {rank}", "Pass!✅", need_sync=True, allowed_ranks="all")

        # Clean up symmetric memory
        nvshmem_free_tensor_sync(symm_ag_buffer)
        nvshmem_free_tensor_sync(symm_signal)
        nvshmem.core.finalize()
        torch.distributed.destroy_process_group()
