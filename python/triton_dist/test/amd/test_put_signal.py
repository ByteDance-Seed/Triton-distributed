
# This file mimics the C++ rocshmem test 3rdparty/rocshmem/examples/rocshmem_put_signal_test.cc in triton.
import os
import torch

import triton_dist
import triton.language as tl
import pyrocshmem
from triton_dist.language.extra.language_extra import __syncthreads, tid
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import finalize_distributed, initialize_distributed, NVSHMEM_SIGNAL_DTYPE

@triton_dist.jit(do_not_specialize=["my_pe", "dst_pe"])
def simple_put_signal_test(data, message, nelem, sig_addr, my_pe, dst_pe):
    pid = tl.program_id(0)
    thread_id = tid(0)

    if thread_id == 0 and pid == 0:
        if my_pe == 0:
            libshmem_device.ulong_put_signal(data, message, nelem, sig_addr, 1, libshmem_device.ROCSHMEM_SIGNAL_SET, dst_pe)
        else:
            libshmem_device.signal_wait_until(sig_addr, libshmem_device.ROCSHMEM_CMP_EQ, 1)
            libshmem_device.ulong_put_signal(data, data, nelem, sig_addr, 1, libshmem_device.ROCSHMEM_SIGNAL_SET, dst_pe)

    __syncthreads()

# get all distributed arguments from environment. which is set by torchrun
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE
TP_GROUP = initialize_distributed()

elems = 256

dst_pe = (RANK + 1) % WORLD_SIZE

message = pyrocshmem.rocshmem_create_tensor((elems,), torch.uint64)
data = pyrocshmem.rocshmem_create_tensor((elems,), torch.uint64)
sig_addr = pyrocshmem.rocshmem_create_tensor((1,), NVSHMEM_SIGNAL_DTYPE)

for i in range(elems):
    message[i] = RANK
    data[i] = RANK

sig_addr.zero_()
torch.cuda.synchronize()

print(RANK, '->', dst_pe)

simple_put_signal_test[(1, )](data, message, elems, sig_addr, RANK, dst_pe)
pyrocshmem.rocshmem_barrier_all()
torch.cuda.synchronize()

passed = True
for i in range(elems):
    if data[i] != 0:
        passed = False

print(data[0], sig_addr)
print("Test pass: ", passed)

torch.distributed.barrier()

del message
del data
del sig_addr
finalize_distributed()
