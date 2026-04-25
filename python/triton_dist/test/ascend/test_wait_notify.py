# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
import torch
import shmem as ash
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
from triton.language.extra.cann.extension import sub_vec_id

g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024
G_IP_PORT = "tcp://127.0.0.1:8666"


@triton.jit
def _producer_kernel(data_ptr, signal_ptr, rank, world_size):
    """Producer: write data and notify consumer"""
    subblock_idx = sub_vec_id()
    if subblock_idx == 0:
        # Write data
        offset = tl.program_id(0)
        value = rank * 100 + offset
        data_ptr = dl.symm_at(data_ptr, 0)
        tl.store(data_ptr + offset, value, None)

        # Notify consumer (rank 0) that data is ready
        target_rank = 0
        dl.notify(
            signal_ptr, target_rank, signal=1, sig_op="set", comm_scope="intra_node"
        )


@triton.jit
def _consumer_kernel(data_ptr, signal_ptr, output_ptr, rank, world_size):
    """Consumer: wait for signal, then read data"""
    subblock_idx = sub_vec_id()
    if subblock_idx == 0:
        # Wait for notification from all other ranks
        num_barriers = world_size - 1  # Wait for all producers
        barrier_ptrs = signal_ptr
        token = dl.wait(
            barrier_ptrs, num_barriers, scope="gpu", semantic="acquire", waitValue=1
        )
        data_ptr = dl.consume_token(data_ptr, token)
        # Read data from all producers
        offset = tl.program_id(0)
        data = tl.load(data_ptr + offset, None)
        tl.store(output_ptr + offset, data, None)


def run_test_distributed(rank, world_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed")
    attributes = ash.InitAttr()
    attributes.my_rank = rank
    attributes.n_ranks = world_size
    attributes.local_mem_size = g_ash_size
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    ret = ash.aclshmem_init(attributes)
    if ret != 0:
        raise ValueError("[ERROR] aclshmem_init failed")

    # Create shared memory for data and signals
    data_mem = ash.aclshmem_create_tensor([32], dtype=torch.int64, device_id=rank)
    signal_mem = ash.aclshmem_create_tensor([1], dtype=torch.int64, device_id=rank)

    # Initialize signal to 0
    signal_mem.zero_()

    if rank == 0:
        # Consumer (rank 0)
        output = torch.zeros(32, dtype=torch.int64).npu()

        # Run consumer kernel
        _consumer_kernel[1, 1, 1](data_mem, signal_mem, output, rank, world_size)

        # Verify: output should contain data from the last producer
        # (since all producers write to same location, last write wins)
        last_producer = world_size - 1
        expected = torch.zeros(32, dtype=torch.int64).npu()
        expected[0] = last_producer * 100
        assert torch.equal(output.cpu(), expected.cpu()), "Consumer: output mismatch"
    else:
        # Producers (rank 1, 2, ...)
        # Run producer kernel
        _producer_kernel[1, 1, 1](data_mem, signal_mem, rank, world_size)

    # Synchronize all ranks
    dist.barrier()

    # Cleanup
    ash.aclshmem_free_tensor(data_mem)
    ash.aclshmem_free_tensor(signal_mem)
    _ = ash.aclshmem_finialize()


@pytest.mark.dist
def test_wait_notify(dist_test):
    dist_test(run_test_distributed, world_size=2)
