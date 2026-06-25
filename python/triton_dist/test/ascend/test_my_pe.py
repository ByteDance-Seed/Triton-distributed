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


def torch_mype(x0, rank, ncore):
    y_ref = x0.cpu()
    for i in range(ncore):
        y_ref[i] += rank
    return y_ref


@triton.jit
def _shmemx_get_my_pe(ptr):
    subblock_idx = sub_vec_id()
    if subblock_idx == 0:
        mype = dl.rank()
        offset = tl.program_id(0)
        tmp0 = tl.load(ptr + offset, None)
        tmp2 = tmp0 + mype
        tl.store(ptr + offset, tmp2, None)


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
    x0 = torch.zeros([32], dtype=torch.int64).npu()
    y_ref = torch_mype(x0, rank, 3)
    _shmemx_get_my_pe[3, 1, 1](x0)
    assert torch.equal(x0.cpu(), y_ref.cpu())
    _ = ash.aclshmem_finialize()


@pytest.mark.dist
def test_mype(dist_test):
    dist_test(run_test_distributed, world_size=2)
