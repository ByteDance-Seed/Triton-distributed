# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
import torch
import shmem as ash
import torch.distributed as dist
import triton.language as tl
import triton
from triton_dist.language.extra import libshmem_device

G_ASH_SIZE = 1024 * 1024 * 1024
TEST_ELEMS = 32


@triton.jit
def barrier_all_putmem_signal_kernel(sync_ptr, epoch, world_size: tl.constexpr, my_rank: tl.constexpr):
    """Inter-node barrier using putmem + signal_wait_until."""
    if tl.program_id(axis=0) == 0:
        tl.store(sync_ptr + my_rank, epoch)  # source buffer for putmem
        peer = 0
        while peer < world_size:
            if peer != my_rank:
                libshmem_device.putmem(sync_ptr + my_rank, sync_ptr + my_rank, 4, peer)
            peer += 1
        peer = 0
        while peer < world_size:
            if peer != my_rank:
                libshmem_device.signal_wait_until(
                    sync_ptr + peer,
                    libshmem_device.ACLSHMEM_CMP_GE,
                    epoch,
                )
            peer += 1


def barrier_all_putmem_signal(sync_ptr, epoch, world_size: tl.constexpr, my_rank: tl.constexpr):
    epoch += 1
    barrier_all_putmem_signal_kernel[(1, 1, 1)](sync_ptr, epoch, world_size, my_rank)
    return epoch


@triton.jit
def putmem_kernel(src, dst, nbytes: tl.constexpr, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.putmem(dst, src, nbytes, peer_rank)


@triton.jit
def getmem_kernel(src, dst, nbytes: tl.constexpr, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.getmem(dst, src, nbytes, peer_rank)


@triton.jit
def putmem_nbi_kernel(src, dst, nbytes: tl.constexpr, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.putmem_nbi(dst, src, nbytes, peer_rank)
        libshmem_device.quiet()


@triton.jit
def getmem_nbi_kernel(src, dst, nbytes: tl.constexpr, peer: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.getmem_nbi(dst, src, nbytes, peer)
        libshmem_device.quiet()


def _init_aclshmem(rank, world_size):
    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed")
    attributes = ash.InitAttr()
    attributes.my_rank = rank
    attributes.n_ranks = world_size
    attributes.local_mem_size = G_ASH_SIZE
    attributes.ip_port = "tcp://127.0.0.1:8666"
    attributes.option_attr.data_op_engine_type = ash.OpEngineType.ROCE
    ret = ash.aclshmem_init(attributes)
    if ret != 0:
        raise ValueError("[ERROR] aclshmem_init failed")


def _build_expected(rank, world_size, elems, value_stride=1000):
    src_rank = (rank - 1 + world_size) % world_size
    return torch.arange(elems, dtype=torch.int32) + src_rank * value_stride


def run_test_distributed(rank, world_size):
    _init_aclshmem(rank, world_size)

    send_peer = int((rank + 1) % world_size)
    recv_peer = int((rank - 1 + world_size) % world_size)

    src = ash.aclshmem_create_tensor([TEST_ELEMS], dtype=torch.int32, device_id=rank)
    dst = ash.aclshmem_create_tensor([TEST_ELEMS], dtype=torch.int32, device_id=rank)
    signal_ptr = ash.aclshmem_create_tensor([world_size], dtype=torch.int32, device_id=rank)

    signal_ptr.zero_()
    nbytes = TEST_ELEMS * src.element_size()
    barrier_epoch = 0
    src.copy_(torch.arange(TEST_ELEMS, dtype=torch.int32, device=src.device) + rank * 1000)
    expected = _build_expected(rank, world_size, TEST_ELEMS)

    def _reset_buffer():
        dst.fill_(-1)

    _reset_buffer()

    putmem_kernel[1, 1, 1](src, dst, nbytes, send_peer)
    barrier_epoch = barrier_all_putmem_signal(signal_ptr, barrier_epoch, world_size,
                                              rank)  # dist.barrier() also works here.
    assert torch.equal(dst.cpu(), expected), f"Rank {rank}: putmem mismatch"
    print(f"Rank {rank}: putmem OK")

    _reset_buffer()

    putmem_nbi_kernel[1, 1, 1](src, dst, nbytes, send_peer)
    dist.barrier()  # We can also use barrier_all_putmem_signal kernel here.
    assert torch.equal(dst.cpu(), expected), f"Rank {rank}: putmem_nbi mismatch"
    print(f"Rank {rank}: putmem_nbi OK")

    _reset_buffer()

    getmem_kernel[1, 1, 1](src, dst, nbytes, recv_peer)
    assert torch.equal(dst.cpu(), expected), f"Rank {rank}: getmem mismatch"
    print(f"Rank {rank}: getmem OK")

    _reset_buffer()

    getmem_nbi_kernel[1, 1, 1](src, dst, nbytes, recv_peer)
    assert torch.equal(dst.cpu(), expected), f"Rank {rank}: getmem_nbi mismatch"
    print(f"Rank {rank}: getmem_nbi OK")

    dist.barrier()  # sync before freeing tensor to avoid race condition

    ash.aclshmem_free_tensor(src)
    ash.aclshmem_free_tensor(dst)
    ash.aclshmem_free_tensor(signal_ptr)
    _ = ash.aclshmem_finalize()


@pytest.mark.dist
def test_put_get_mem(dist_test):
    dist_test(run_test_distributed, world_size=2)
