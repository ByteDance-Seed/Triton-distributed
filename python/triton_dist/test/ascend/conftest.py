# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def pytest_configure(config):
    config.addinivalue_line("markers", "dist: mark distributed Ascend tests")


def _worker_wrapper(rank, world_size, backend, fn, args, error_queue):
    """
    Common entry point for each worker process.
    Exits normally on success; on failure, pushes the exception into error_queue.
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        torch.npu.set_device(rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"npu:{torch.npu.current_device()}"),
        )
        dist.barrier()
        fn(rank, world_size, *args)
    except Exception as e:
        error_queue.put((rank, e))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_dist_test(fn, world_size=2, backend="hccl", args=()):
    """
    Executes fn(rank, world_size, *args) across world_size processes.
    If any rank raises an exception, the test fails and the original error is shown.
    """
    ctx = mp.get_context("spawn")
    error_queue = ctx.Queue()

    mp.spawn(
        _worker_wrapper,
        args=(world_size, backend, fn, args, error_queue),
        nprocs=world_size,
        join=True,
    )

    # Collect errors from all ranks
    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())

    if errors:
        msgs = "\n".join(f"[rank {r}] {e}" for r, e in errors)
        pytest.fail(f"Distributed worker error:\n{msgs}")


@pytest.fixture
def dist_test():
    """
    Usage:
        def test_xxx(dist_test):
            dist_test(my_worker_fn, world_size=2)
    """
    return run_dist_test
