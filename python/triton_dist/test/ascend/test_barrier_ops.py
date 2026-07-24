# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
import torch
import shmem as ash
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra.cann.extension import sub_vec_id
from triton_dist.language.extra import libshmem_device

G_ASH_SIZE = 1024 * 1024 * 1024
G_IP_PORT = "tcp://127.0.0.1:8666"


def _init_aclshmem(rank, world_size):
    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed")
    attributes = ash.InitAttr()
    attributes.my_rank = rank
    attributes.n_ranks = world_size
    attributes.local_mem_size = G_ASH_SIZE
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    ret = ash.aclshmem_init(attributes)
    if ret != 0:
        raise ValueError("[ERROR] aclshmem_init failed")


@triton.jit
def _increase_with_barrier_kernel(counter_ptr, team):
    team_rank = libshmem_device.team_my_pe(team)

    if team_rank >= 0:
        team_size = libshmem_device.team_n_pes(team)
        val = tl.load(counter_ptr)

        libshmem_device.barrier(team)

        if sub_vec_id() == 0:
            next_team_rank = (team_rank + 1) % team_size
            next_pe = libshmem_device.team_translate_pe(
                team,
                next_team_rank,
                libshmem_device.ACLSHMEM_TEAM_WORLD,
            )
            remote = libshmem_device.remote_ptr(counter_ptr, next_pe)
            tl.store(remote, val + 1)

        libshmem_device.barrier(team)


@triton.jit
def _increase_with_barrier_all_kernel(counter_ptr, my_rank: tl.constexpr, world_size: tl.constexpr):
    val = tl.load(counter_ptr)

    libshmem_device.barrier_all()

    if sub_vec_id() == 0:
        next_pe = (my_rank + 1) % world_size
        remote = libshmem_device.remote_ptr(counter_ptr, next_pe)
        tl.store(remote, val + 1)

    libshmem_device.barrier_all()


@triton.jit
def _increase_with_barrier_vec_kernel(counter_ptr, team):
    team_rank = libshmem_device.team_my_pe(team)

    if team_rank >= 0:
        team_size = libshmem_device.team_n_pes(team)
        val = tl.load(counter_ptr)

        libshmem_device.barrier_vec(team)

        if sub_vec_id() == 0:
            next_team_rank = (team_rank + 1) % team_size
            next_pe = libshmem_device.team_translate_pe(
                team,
                next_team_rank,
                libshmem_device.ACLSHMEM_TEAM_WORLD,
            )
            remote = libshmem_device.remote_ptr(counter_ptr, next_pe)
            tl.store(remote, val + 1)

        libshmem_device.barrier_vec(team)


@triton.jit
def _increase_with_barrier_all_vec_kernel(counter_ptr, my_rank: tl.constexpr, world_size: tl.constexpr):
    val = tl.load(counter_ptr)

    libshmem_device.barrier_all_vec()

    if sub_vec_id() == 0:
        next_pe = (my_rank + 1) % world_size
        remote = libshmem_device.remote_ptr(counter_ptr, next_pe)
        tl.store(remote, val + 1)

    libshmem_device.barrier_all_vec()


def _run_team_increase_test(rank, kernel_fn, team, expected):
    counter = ash.aclshmem_create_tensor([1], dtype=torch.int64, device_id=rank)
    try:
        counter.zero_()
        dist.barrier()
        kernel_fn[(1, 1, 1)](counter, team)
        dist.barrier()
        actual = counter.cpu().item()
        assert actual == expected, (f"Rank {rank}: expected counter {expected}, got {actual}")
    finally:
        ash.aclshmem_free_tensor(counter)


def _run_world_increase_test(rank, world_size, kernel_fn, *kernel_args):
    counter = ash.aclshmem_create_tensor([1], dtype=torch.int64, device_id=rank)
    try:
        counter.zero_()
        dist.barrier()
        kernel_fn[(1, 1, 1)](counter, *kernel_args, rank, world_size)
        dist.barrier()
        actual = counter.cpu().item()
        assert actual == 1, f"Rank {rank}: expected counter 1, got {actual}"
    finally:
        ash.aclshmem_free_tensor(counter)


def _create_odd_sub_team(world_size):
    return ash.team_split_strided(
        libshmem_device.ACLSHMEM_TEAM_WORLD,
        1,
        2,
        world_size // 2,
    )


def _test_team_world_barriers(rank, world_size):
    team = libshmem_device.ACLSHMEM_TEAM_WORLD

    _run_team_increase_test(rank, _increase_with_barrier_kernel, team, expected=1)
    print(f"Rank {rank}: aclshmem_barrier (TEAM_WORLD) OK")

    _run_world_increase_test(rank, world_size, _increase_with_barrier_all_kernel)
    print(f"Rank {rank}: aclshmem_barrier_all OK")

    _run_team_increase_test(rank, _increase_with_barrier_vec_kernel, team, expected=1)
    print(f"Rank {rank}: aclshmemx_barrier_vec (TEAM_WORLD) OK")

    _run_world_increase_test(rank, world_size, _increase_with_barrier_all_vec_kernel)
    print(f"Rank {rank}: aclshmemx_barrier_all_vec OK")


def _test_odd_sub_team_barriers(rank, world_size):
    team_odd = _create_odd_sub_team(world_size)
    try:
        # Odd ranks are members of team_split_strided(TEAM_WORLD, 1, 2, world_size // 2).
        expected = 1 if rank % 2 == 1 else 0

        _run_team_increase_test(rank, _increase_with_barrier_kernel, team_odd, expected)
        print(f"Rank {rank}: aclshmem_barrier (odd sub-team) "
              f"{'OK' if expected else 'skipped (not in team)'}")

        _run_team_increase_test(rank, _increase_with_barrier_vec_kernel, team_odd, expected)
        print(f"Rank {rank}: aclshmemx_barrier_vec (odd sub-team) "
              f"{'OK' if expected else 'skipped (not in team)'}")
    finally:
        ash.team_destroy(team_odd)


def run_test_team_world(rank, world_size):
    _init_aclshmem(rank, world_size)
    _test_team_world_barriers(rank, world_size)
    _ = ash.aclshmem_finalize()


def run_test_sub_team(rank, world_size):
    _init_aclshmem(rank, world_size)
    _test_team_world_barriers(rank, world_size)
    dist.barrier()
    _test_odd_sub_team_barriers(rank, world_size)
    _ = ash.aclshmem_finalize()


@pytest.mark.dist
def test_barrier_ops_team_world(dist_test):
    dist_test(run_test_team_world, world_size=2)


@pytest.mark.dist
def test_barrier_ops_sub_team(dist_test):
    dist_test(run_test_sub_team, world_size=4)
