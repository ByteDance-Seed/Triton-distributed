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
import os
import datetime
import torch
import nvshmem
import nvshmem.core
import triton
import triton.language as tl
from triton_dist.utils import init_nvshmem_by_torch_process_group, init_seed
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import tid


@triton.jit
def team_translate_pe_kernel(
    src_team: int,
    pe_in_src_team: int,
    dest_team: int,
    out_ptr,
):
    thread_id = tid(0)
    if thread_id == 0:
        my_pe = libshmem_device.team_translate_pe(src_team, pe_in_src_team, dest_team)
        tl.store(out_ptr, my_pe)


def split_torch_process_group(pg: torch.distributed.ProcessGroup, num_groups: int) -> torch.distributed.ProcessGroup:
    size = pg.size()
    rank = pg.rank()
    group_size = size // num_groups
    group_id = rank // group_size
    if size % group_size != 0:
        raise ValueError(f"Process group size {size} is not divisible by group size {group_size}.")
    # Create list of ranks per group
    for n in range(num_groups):
        subgroup_ranks = [i + n * group_size for i in range(group_size)]
        # Create new NCCL group
        print("subgroup_ranks", subgroup_ranks)
        subgroup_ = torch.distributed.new_group(ranks=subgroup_ranks, backend="nccl")
        if n == group_id:
            subgroup = subgroup_
    return subgroup


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    # use all ranks as tp group
    WORLD_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(WORLD_GROUP)

    seed = 2025
    init_seed(seed=seed if seed is not None else RANK)

    init_nvshmem_by_torch_process_group(WORLD_GROUP)

    ep_size = 4
    pp_size = WORLD_SIZE // ep_size

    ep_group = split_torch_process_group(WORLD_GROUP, pp_size)

    config = nvshmem.core.team_get_config(libshmem_device.NVSHMEM_TEAM_WORLD)
    print("config", config)

    ep_team, pp_team = nvshmem.core.team_split_2d(libshmem_device.NVSHMEM_TEAM_WORLD, ep_size, config, 0, config, 0)

    mype_pp = nvshmem.core.team_my_pe(pp_team)
    mype_ep = nvshmem.core.team_my_pe(ep_team)
    npes_pp = nvshmem.core.team_n_pes(pp_team)
    npes_ep = nvshmem.core.team_n_pes(ep_team)
    print(f"PP group: {mype_pp}/{npes_pp}, EP group: {mype_ep}/{npes_ep}")

    mype = nvshmem.bindings.team_translate_pe(pp_team, mype_pp, libshmem_device.NVSHMEM_TEAM_WORLD)
    assert mype == RANK, f"mype: {mype} should be equal to RANK: {RANK}"

    print(f"type of team: {type(pp_team)}")
    print(f"type of pe: {type(mype_pp)}")
    out = torch.empty([1], dtype=torch.int32, device="cuda")
    team_translate_pe_kernel[(1, )](int(pp_team), int(mype_pp), libshmem_device.NVSHMEM_TEAM_WORLD, out)
    assert out[0].cpu().item() == RANK, f"out[0].cpu().item(): {out[0].cpu().item()} should be equal to RANK: {RANK}"

    torch.distributed.destroy_process_group()
