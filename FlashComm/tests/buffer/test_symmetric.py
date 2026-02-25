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
Test command:
torchrun --nproc_per_node=8 tests/buffer/test_symmetric.py
"""
import torch
import torch.distributed as dist
from flash_comm.buffer import SymmetricTensor


def test_symmetric_buffer_ring_put():
    """
    Test symmetric buffer correctness using ring put pattern.
    Each rank writes its rank value to the next rank's buffer.
    Then each rank verifies it received the expected value from the previous rank.
    """
    dist.init_process_group("nccl")
    group = dist.distributed_c10d._get_default_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Create symmetric tensor (16 int32 elements)
    shape = (16, )
    dtype = torch.int32
    st = SymmetricTensor(shape, dtype, group=group, backend="auto")
    print(f"Rank {rank}: SymmetricTensor created, backend={st.buffer.backend}")

    # Initialize local buffer with -1
    local_tensor = st.get_local_tensor()
    local_tensor.fill_(-1)
    torch.cuda.synchronize()

    dist.barrier()

    # Ring put: write to next rank's buffer
    next_rank = (rank + 1) % world_size
    peer_tensor = st.get_peer_tensor(next_rank)

    # Write rank value to peer's buffer
    peer_tensor.fill_(rank)
    torch.cuda.synchronize()
    print(f"Rank {rank}: Wrote value {rank} to rank {next_rank}'s buffer")

    dist.barrier()

    # Verify: local buffer should contain value from previous rank
    prev_rank = (rank - 1 + world_size) % world_size
    expected_value = prev_rank

    local_tensor = st.get_local_tensor()
    actual_values = local_tensor.tolist()

    # Check all elements
    for i, val in enumerate(actual_values):
        assert val == expected_value, \
            f"Rank {rank}: Element {i} mismatch! Expected {expected_value}, got {val}"

    print(f"Rank {rank}: Verification PASSED - received {expected_value} from rank {prev_rank}")

    dist.barrier()
    dist.destroy_process_group()
    print(f"Rank {rank}: TEST PASSED")


if __name__ == "__main__":
    test_symmetric_buffer_ring_put()
