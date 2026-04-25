# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker_wrapper(rank, world_size, backend, fn, args, error_queue):
    """
    每个 worker 进程的通用入口。
    成功时正常退出，失败时把异常推入 error_queue。
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    try:
        torch.npu.set_device(rank)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        fn(rank, world_size, *args)
    except Exception as e:
        error_queue.put((rank, e))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_dist_test(fn, world_size=2, backend="hccl", args=()):
    """
    在 world_size 个进程里执行 fn(rank, world_size, *args)。
    任意 rank 抛异常，测试失败并展示原始错误。
    """
    ctx = mp.get_context("spawn")
    error_queue = ctx.Queue()

    mp.spawn(
        _worker_wrapper,
        args=(world_size, backend, fn, args, error_queue),
        nprocs=world_size,
        join=True,
    )

    # 收集所有 rank 的错误
    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())

    if errors:
        msgs = "\n".join(f"[rank {r}] {e}" for r, e in errors)
        pytest.fail(f"分布式 worker 报错:\n{msgs}")


@pytest.fixture
def dist_test():
    """
    用法:
        def test_xxx(dist_test):
            dist_test(my_worker_fn, world_size=2)
    """
    return run_dist_test
