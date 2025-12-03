
import os
import torch

import triton
import triton.language as tl
import pyrocshmem
from triton.language.extra.hip.libdevice import (thread_idx, __syncthreads)
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import finalize_distributed, initialize_distributed, NVSHMEM_SIGNAL_DTYPE

@triton.jit(do_not_specialize=["my_pe", "dst_pe"])
def simple_put_signal_test(data, message, nelem, sig_addr, my_pe, dst_pe, ctx):
    libshmem_device.set_rocshmem_ctx(ctx)
    pid = tl.program_id(0)
    tid = thread_idx(0);

    if tid == 0 and pid == 0:
        if my_pe == 0:
            libshmem_device.ulong_put_signal(data, message, nelem, sig_addr, 1, libshmem_device.ROCSHMEM_SIGNAL_SET, dst_pe)
            #libshmem_device.putmem_signal(data, message, nelem * torch.uint64.itemsize, sig_addr, 1, libshmem_device.ROCSHMEM_SIGNAL_SET, dst_pe)
        else:
            libshmem_device.signal_wait_until(sig_addr, libshmem_device.ROCSHMEM_CMP_EQ, 1)
            libshmem_device.ulong_put_signal(data, data, nelem, sig_addr, 1, libshmem_device.ROCSHMEM_SIGNAL_SET, dst_pe)
            #libshmem_device.putmem_signal(data, data, nelem * torch.uint64.itemsize, sig_addr, 1, libshmem_device.ROCSHMEM_SIGNAL_SET, dst_pe)

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
sig_addr = pyrocshmem.rocshmem_malloc(NVSHMEM_SIGNAL_DTYPE.itemsize)

for i in range(elems):
    message[i] = RANK

message.zero_()
torch.cuda.synchronize()

print(RANK, '->', dst_pe)
ctx = pyrocshmem.rocshmem_get_device_ctx()

simple_put_signal_test[(1, )](data, message, elems, sig_addr, RANK, dst_pe, ctx)
pyrocshmem.rocshmem_barrier_all()
torch.cuda.synchronize()

passed = True
for i in range(elems):
    if data[i] != 0:
        passed = False

print(message[0], sig_addr)
print("Test pass: ", passed)

finalize_distributed()
