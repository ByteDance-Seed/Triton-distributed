# torchrun --nproc_per_node=8 --nnodes=1 run_ring_put.py
import datetime
import os
import pymxshmem
import torch
import torch.distributed
import time

def ring_put():
    t = pymxshmem.mxshmem_create_tensor([1024], torch.int)
    t[0] = 3
    if RANK == 1:
        print(f"create torch tensor with mxshmem in rank:{RANK}", t)
    torch.cuda.synchronize()
    
    pymxshmem.mxshmem_int_p(t.data_ptr(), 7, (RANK + 1) % WORLD_SIZE)
    pymxshmem.mxshmem_barrier_all()
    time.sleep(5)
    if RANK == 1:
        print(f"after put_rank_to_next rank:{RANK}", t)

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
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

torch.cuda.synchronize()
pymxshmem.init_mxshmem_by_uniqueid(TP_GROUP)
ring_put()

pymxshmem.mxshmem_finalize()
torch.distributed.destroy_process_group()