Distributed Notify and Wait
===========================

In this tutorial, you will write a simple notify and wait example using Triton-distributed.

In doing so, you will learn about:

* The signal exchange concept within a single node using Triton-distributed.

* The `wait`, `consume_token`, `notify` primitives, which is used to do signal exchange.

* The distributed runtime initialization and symmetric tensor management.

* How to write producer-consumer data transfer through a small queue

.. code-block:: bash

    # To run this tutorial
    bash ./launch.sh ./tutorials/01-distributed-notify-wait.py


Kernel
------

In this example, the kernel is divided into two parts: one part acts as a producer and the other as a consumer. They transfer data between each other via a small, symmetric memory buffer, requiring handshaking to ensure the buffer's positions are available for the producer to write to and for the consumer to read from.


.. code-block:: Python

    import torch
    from triton_dist import pynvshmem
    import os
    import datetime

    import triton
    import triton.language as tl
    import triton_dist.language as dl
    from triton_dist.utils import dist_print
    from triton.language.extra.cuda.language_extra import (__syncthreads)


    @triton.jit
    def producer_consumer_kernel(
        rank: tl.constexpr,
        num_ranks: tl.constexpr,
        input_ptr,
        output_ptr,
        num_inputs: int,
        queue_ptr,
        signal_ptr,  # *Pointer* to signals.
        queue_size: tl.constexpr,  # The length of queue in unit of BLOCKs
        BLOCK_SIZE: tl.constexpr,  # The size of each BLOCK
        NUM_PRODUCER_SMS: tl.constexpr,
        NUM_CONSUMER_SMS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        # This kernel issues async-tasks to two group of blocks
        if pid < NUM_PRODUCER_SMS:
            #
            # Producer
            #
            peer_rank = (rank + 1) % num_ranks  # Peer is the next rank
            offs = tl.arange(0, BLOCK_SIZE)
            for i in range(pid, num_inputs, NUM_PRODUCER_SMS):
                queue_offset = i % queue_size
                queue_repeat = i // queue_size
                token = dl.wait(
                    # Use `symm_at` to map the data pointer to remote peer rank
                    dl.symm_at(signal_ptr, peer_rank) + queue_offset, 1,  # The number of signals to wait
                    "sys",  # The scope of the barrier, `gpu` or `sys`
                    "acquire",  # The semantic of the wait
                    waitValue=queue_repeat * 2,  # The value expected, should conform to certain order
                )  # This wait ensures that the corresponding position is empty
                input_ptr = dl.consume_token(input_ptr, token)  # consume the token to make sure the `wait` is needed
                data = tl.load(input_ptr + i * BLOCK_SIZE + offs)
                # Use `symm_at` to map the data pointer to remote peer rank
                tl.store(dl.symm_at(queue_ptr, peer_rank) + queue_offset * BLOCK_SIZE + offs, data)
                # need a syncthreads to make sure all the data has been sent
                __syncthreads()
                # `notify` is also single thread scope, we need to use a different thread from `wait`
                dl.notify(signal_ptr + queue_offset, peer_rank,  # Notify the signal object on the peer rank
                        signal=queue_repeat * 2 + 1,  # Write a value to the signal
                        sig_op="set",  # Set the value. Another choice is `add`
                        comm_scope="intra_node",  # This example is intra-node, another choice is `inter_node`
                        )  # This notifies the consumer that the data is ready
        elif pid < NUM_PRODUCER_SMS + NUM_CONSUMER_SMS:
            #
            # Consumer
            #
            pid = pid - NUM_PRODUCER_SMS
            offs = tl.arange(0, BLOCK_SIZE)
            for i in range(pid, num_inputs, NUM_CONSUMER_SMS):
                queue_offset = i % queue_size
                queue_repeat = i // queue_size
                token = dl.wait(signal_ptr + queue_offset,  # The base *Pointer* of signals at the current rank
                                1,  # The number of signals to wait
                                "sys",  # The scope of the barrier
                                "acquire",  # The semantic of the wait
                                waitValue=queue_repeat * 2 + 1,  # The value expected
                                )  # This wait ensures that the corresponding position is full
                queue_ptr = dl.consume_token(queue_ptr, token)
                data = tl.load(queue_ptr + queue_offset * BLOCK_SIZE + offs)
                tl.store(output_ptr + i * BLOCK_SIZE + offs, data)
                __syncthreads()
                dl.notify(signal_ptr + queue_offset, rank,  # Notify the signal object on the current rank
                        signal=queue_repeat * 2 + 2,  # Write a value to the signal
                        sig_op="set",  # Set the value. Another choice is `add`
                        comm_scope="intra_node",  # This example is intra-node, another choice is `inter_node`
                        )  # This notifies the consumer that the data is ready
        else:
            pass


Initialize the Distributed System
---------------------------------

Here, we show you how to initialize a distributed system for our example.

.. code-block:: Python

    def initialize_distributed():
        RANK = int(os.environ.get("RANK", 0))
        LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
        WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
        assert WORLD_SIZE <= 8  # This example only runs on a single node
        torch.cuda.set_device(LOCAL_RANK)
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=WORLD_SIZE,
            rank=RANK,
            timeout=datetime.timedelta(seconds=1800),
        )
        assert torch.distributed.is_initialized()
        TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

        torch.cuda.synchronize()
        # You need to use `init_nvshmem_by_uniqueid` to initialize
        # the distributed system
        pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)
        return TP_GROUP


Test the Correctness
----------------

Let's now check our notify and wait kernel for correctness.

.. code-block:: Python

    INPUT_SIZE = 2025  # A large input size
    QUEUE_SIZE = 32  # Queue is smaller than input size
    BLOCK_SIZE = 128


    def main(TP_GROUP):
        stream = torch.cuda.current_stream()
        # The created tensor is by-default on current cuda device
        queue = pynvshmem.nvshmem_create_tensor((QUEUE_SIZE * BLOCK_SIZE, ),  # Shape on each device
                                                torch.float32)
        signal = pynvshmem.nvshmem_create_tensor((QUEUE_SIZE, ), torch.uint64  # Notify requires 64bit unsigned signal type
                                                )
        queue.fill_(-1)
        signal.fill_(0)  # The initial value of signal should be 0s
        # You need a barrier all to make sure the above initialization
        # is visible to all the other ranks.
        # This is usually used for intra-node.
        pynvshmem.nvshmemx_barrier_all_on_stream(stream.cuda_stream)

        # Distributed info
        rank = TP_GROUP.rank()
        num_ranks = TP_GROUP.size()

        # Prepare torch local data
        input_data = torch.randn((INPUT_SIZE * BLOCK_SIZE, ), dtype=torch.float32).cuda()
        output_data = torch.empty_like(input_data)

        NUM_REPEAS = 20
        # For distributed programming, you have to run it multiple times to ensure
        # your program is correct, including reseting signals, avoiding racing, etc.
        for iters in range(NUM_REPEAS):
            input_data = torch.randn((INPUT_SIZE * BLOCK_SIZE, ), dtype=torch.float32).cuda()
            # Need to reset the barrier every time, you may also omit this step for better performance
            # by using flipping barriers. We will cover this optimization in future tutorial.
            # TODO: tutorial for flipping barriers.
            signal.fill_(0)
            pynvshmem.nvshmemx_barrier_all_on_stream(stream.cuda_stream)

            producer_consumer_kernel[(20, )](  # use 20 SMs
                rank,
                num_ranks,
                input_data,
                output_data,
                INPUT_SIZE,
                queue,
                signal,
                QUEUE_SIZE,
                BLOCK_SIZE,
                16,  # 16 SMs for producer
                4,  # 4 SMs for consumer
                num_warps=4,
            )

            # Check results
            inputs_all_ranks = [torch.empty_like(input_data) for _ in range(num_ranks)]
            torch.distributed.all_gather(inputs_all_ranks, input_data, group=TP_GROUP)
            golden = inputs_all_ranks[(rank - 1 + num_ranks) % num_ranks]
            if iters == NUM_REPEAS - 1:
                dist_print(f"rank{rank}", output_data, need_sync=True, allowed_ranks=list(range(num_ranks)))
                dist_print(f"rank{rank}", golden, need_sync=True, allowed_ranks=list(range(num_ranks)))
            assert torch.allclose(output_data, golden, atol=1e-5, rtol=1e-5)
            if iters == NUM_REPEAS - 1:
                dist_print(f"rank{rank} Passed✅!", need_sync=True, allowed_ranks=list(range(num_ranks)))


    # Initialize the distributed system
    TP_GROUP = initialize_distributed()
    # The main function
    main(TP_GROUP)
    # Finalize
    torch.distributed.destroy_process_group()