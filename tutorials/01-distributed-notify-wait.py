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
    source ./scripts/setenv.sh
    bash ./scripts/launch.sh tutorials/01-distributed-notify-wait.py

"""

# %%
# Kernel
# --------------
import torch
import nvshmem.core

import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE, dist_print, initialize_distributed, nvshmem_barrier_all_on_stream, nvshmem_free_tensor_sync, nvshmem_create_tensor
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
                dl.symm_at(signal_ptr, peer_rank) + queue_offset,
                1,  # The number of signals to wait
                "sys",  # The scope of the barrier, `gpu` or `sys`
                "acquire",  # The semantic of the wait
                waitValue=queue_repeat *
                2,  # The value expected, should conform to certain order
            )  # This wait ensures that the corresponding position is empty
            input_ptr = dl.consume_token(
                input_ptr,
                token)  # consume the token to make sure the `wait` is needed
            data = tl.load(input_ptr + i * BLOCK_SIZE + offs)
            # Use `symm_at` to map the data pointer to remote peer rank
            tl.store(
                dl.symm_at(queue_ptr, peer_rank) + queue_offset * BLOCK_SIZE +
                offs, data)
            # need a syncthreads to make sure all the data has been sent
            __syncthreads()
            # `notify` is also single thread scope, we need to use a different thread from `wait`
            dl.notify(
                signal_ptr + queue_offset,
                peer_rank,  # Notify the signal object on the peer rank
                signal=queue_repeat * 2 + 1,  # Write a value to the signal
                sig_op="set",  # Set the value. Another choice is `add`
                comm_scope=
                "intra_node",  # This example is intra-node, another choice is `inter_node`
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
            token = dl.wait(
                signal_ptr +
                queue_offset,  # The base *Pointer* of signals at the current rank
                1,  # The number of signals to wait
                "sys",  # The scope of the barrier
                "acquire",  # The semantic of the wait
                waitValue=queue_repeat * 2 + 1,  # The value expected
            )  # This wait ensures that the corresponding position is full
            queue_ptr = dl.consume_token(queue_ptr, token)
            data = tl.load(queue_ptr + queue_offset * BLOCK_SIZE + offs)
            tl.store(output_ptr + i * BLOCK_SIZE + offs, data)
            __syncthreads()
            dl.notify(
                signal_ptr + queue_offset,
                rank,  # Notify the signal object on the current rank
                signal=queue_repeat * 2 + 2,  # Write a value to the signal
                sig_op="set",  # Set the value. Another choice is `add`
                comm_scope=
                "intra_node",  # This example is intra-node, another choice is `inter_node`
            )  # This notifies the consumer that the data is ready
    else:
        pass


INPUT_SIZE = 2025  # A large input size
QUEUE_SIZE = 32  # Queue is smaller than input size
BLOCK_SIZE = 128


def main(TP_GROUP):
    stream = torch.cuda.current_stream()
    # The created tensor is by-default on current cuda device
    queue = nvshmem_create_tensor(
        (QUEUE_SIZE * BLOCK_SIZE, ),  # Shape on each device
        torch.float32)
    signal = nvshmem_create_tensor((QUEUE_SIZE, ), NVSHMEM_SIGNAL_DTYPE)
    queue.fill_(-1)
    signal.fill_(0)  # The initial value of signal should be 0s
    # You need a barrier all to make sure the above initialization
    # is visible to all the other ranks.
    # This is usually used for intra-node.
    nvshmem_barrier_all_on_stream(stream)

    # Distributed info
    rank = TP_GROUP.rank()
    num_ranks = TP_GROUP.size()

    # Prepare torch local data
    input_data = torch.randn((INPUT_SIZE * BLOCK_SIZE, ),
                             dtype=torch.float32).cuda()
    output_data = torch.empty_like(input_data)

    NUM_REPEAS = 20
    # For distributed programming, you have to run it multiple times to ensure
    # your program is correct, including reseting signals, avoiding racing, etc.
    for iters in range(NUM_REPEAS):
        input_data = torch.randn((INPUT_SIZE * BLOCK_SIZE, ),
                                 dtype=torch.float32).cuda()
        # Need to reset the barrier every time, you may also omit this step for better performance
        # by using flipping barriers. We will cover this optimization in future tutorial.
        # TODO: tutorial for flipping barriers.
        signal.fill_(0)
        nvshmem_barrier_all_on_stream(stream)

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
        inputs_all_ranks = [
            torch.empty_like(input_data) for _ in range(num_ranks)
        ]
        torch.distributed.all_gather(inputs_all_ranks,
                                     input_data,
                                     group=TP_GROUP)
        golden = inputs_all_ranks[(rank - 1 + num_ranks) % num_ranks]
        if iters == NUM_REPEAS - 1:
            dist_print(f"rank{rank}",
                       output_data,
                       need_sync=True,
                       allowed_ranks=list(range(num_ranks)))
            dist_print(f"rank{rank}",
                       golden,
                       need_sync=True,
                       allowed_ranks=list(range(num_ranks)))
        assert torch.allclose(output_data, golden, atol=1e-5, rtol=1e-5)
        if iters == NUM_REPEAS - 1:
            dist_print(f"rank{rank} Passed✅!",
                       need_sync=True,
                       allowed_ranks=list(range(num_ranks)))

    nvshmem_free_tensor_sync(queue)
    nvshmem_free_tensor_sync(signal)


# Initialize the distributed system
TP_GROUP = initialize_distributed()
# The main function
main(TP_GROUP)

nvshmem.core.finalize()
torch.distributed.destroy_process_group()
