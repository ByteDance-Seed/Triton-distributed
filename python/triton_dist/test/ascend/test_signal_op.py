# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
import torch
import shmem as ash
import torch.distributed as dist
import triton.language as tl
import triton
from triton_dist.language.extra import libshmem_device

G_ASH_SIZE = 1024 * 1024 * 1024


def _init_aclshmem(rank, world_size):
    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed")
    attributes = ash.InitAttr()
    attributes.my_rank = rank
    attributes.n_ranks = world_size
    attributes.local_mem_size = G_ASH_SIZE
    attributes.ip_port = "tcp://127.0.0.1:8666"
    attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    ret = ash.aclshmem_init(attributes)
    if ret != 0:
        raise ValueError("[ERROR] aclshmem_init failed")


@triton.jit
def signal_set_ring_kernel(sig_ptr, my_rank: tl.constexpr, world_size: tl.constexpr):
    """Each rank sets the next rank's signal, then waits for its own."""
    if tl.program_id(axis=0) == 0:
        next_pe = (my_rank + 1) % world_size
        expected = my_rank + 1
        signal_val = next_pe + 1
        libshmem_device.signal_op(
            sig_ptr,
            signal_val,
            libshmem_device.ACLSHMEM_SIGNAL_SET,
            next_pe,
        )
        libshmem_device.signal_wait_until(
            sig_ptr,
            libshmem_device.ACLSHMEM_CMP_EQ,
            expected,
        )


@triton.jit
def signal_wait_until_retval_kernel(sig_ptr, ret_ptr, wait_val: tl.constexpr, my_rank: tl.constexpr,
                                    world_size: tl.constexpr):
    """Rank 0 waits for a known signal value; rank 1 sets it.
    The return value of signal_wait_until (the satisfied signal value) is
    written to ret_ptr so the host can verify it."""
    if tl.program_id(axis=0) == 0:
        if my_rank == 0:
            ret = libshmem_device.signal_wait_until(
                sig_ptr,
                libshmem_device.ACLSHMEM_CMP_EQ,
                wait_val,
            )
            tl.store(ret_ptr, ret)
        else:
            # Give rank 0 time to enter the wait before we signal it.
            peer = 0
            libshmem_device.signal_op(
                sig_ptr,
                wait_val,
                libshmem_device.ACLSHMEM_SIGNAL_SET,
                peer,
            )


@triton.jit
def signal_pingpong_kernel(sig_ptr, iters: tl.constexpr, my_rank: tl.constexpr):
    """Ping-pong signal_op / signal_wait_until between rank 0 and rank 1."""
    if tl.program_id(axis=0) == 0:
        peer = 1 - my_rank
        n = 0
        while n < iters:
            val = 1 + n
            if my_rank == 0:
                libshmem_device.signal_wait_until(
                    sig_ptr,
                    libshmem_device.ACLSHMEM_CMP_EQ,
                    val,
                )
                libshmem_device.signal_op(
                    sig_ptr,
                    val,
                    libshmem_device.ACLSHMEM_SIGNAL_SET,
                    peer,
                )
            else:
                libshmem_device.signal_op(
                    sig_ptr,
                    val,
                    libshmem_device.ACLSHMEM_SIGNAL_SET,
                    peer,
                )
                libshmem_device.signal_wait_until(
                    sig_ptr,
                    libshmem_device.ACLSHMEM_CMP_EQ,
                    val,
                )
            n += 1


def run_test_distributed(rank, world_size):
    _init_aclshmem(rank, world_size)

    sig_ptr = ash.aclshmem_create_tensor([1], dtype=torch.int32, device_id=rank)

    # --- signal_wait_until return value: rank 0 waits, rank 1 signals ---
    if world_size >= 2:
        sig_ptr.zero_()
        ret_ptr = ash.aclshmem_create_tensor([1], dtype=torch.int32, device_id=rank)
        ret_ptr.zero_()
        wait_val = 42
        dist.barrier()
        signal_wait_until_retval_kernel[(1, 1, 1)](sig_ptr, ret_ptr, wait_val, rank, world_size)
        dist.barrier()
        if rank == 0:
            got = ret_ptr.cpu().item()
            assert got == wait_val, (f"Rank 0: signal_wait_until return value expected {wait_val}, got {got}")
            print(f"Rank {rank}: signal_wait_until return value OK (got {got})")
        ash.aclshmem_free_tensor(ret_ptr)

    dist.barrier()

    # --- signal_op + signal_wait_until with ACLSHMEM_SIGNAL_SET (ring) ---
    sig_ptr.zero_()
    dist.barrier()
    signal_set_ring_kernel[(1, 1, 1)](sig_ptr, rank, world_size)
    expected = rank + 1
    assert sig_ptr.cpu().item() == expected, (
        f"Rank {rank}: signal_set_ring expected {expected}, got {sig_ptr.cpu().item()}")
    print(f"Rank {rank}: signal_set_ring OK")

    # --- ping-pong between rank 0 and rank 1 ---
    if world_size >= 2:
        sig_ptr.zero_()
        dist.barrier()
        pingpong_iters = 100
        if rank <= 1:
            signal_pingpong_kernel[(1, 1, 1)](sig_ptr, pingpong_iters, rank)
        dist.barrier()
        if rank == 0:
            assert sig_ptr.cpu().item() == pingpong_iters, (
                f"Rank 0: signal_pingpong expected {pingpong_iters}, got {sig_ptr.cpu().item()}")
            print(f"Rank {rank}: signal_pingpong OK")

    dist.barrier()

    ash.aclshmem_free_tensor(sig_ptr)
    _ = ash.aclshmem_finalize()


@pytest.mark.dist
def test_signal_op(dist_test):
    dist_test(run_test_distributed, world_size=2)
