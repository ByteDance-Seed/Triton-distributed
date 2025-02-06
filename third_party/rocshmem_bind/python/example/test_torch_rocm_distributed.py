################################################################################
# Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
################################################################################
# ref to: https://rocm.blogs.amd.com/artificial-intelligence/ddp-training-pytorch/README.html

# Single node and single GPU
# torchrun --nnodes 1 --nproc_per_node 1 ./test_torch_rocm_distributed.py

# Single node and multi GPUs
# torchrun --nnodes 1 --nproc_per_node 4 ./test_torch_rocm_distributed.py

import torch

import torch.distributed as dist

import os

import pyrocshmem

GLOBAL_RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
print(
    f"Hello from LOCAL_RANK {LOCAL_RANK}, GLOBAL_RANK {GLOBAL_RANK}. WORLD_SIZE {WORLD_SIZE}, LOCAL_WORLD_SIZE {LOCAL_WORLD_SIZE}"
)

torch.cuda.set_device(LOCAL_RANK)

# nccl is recommended by pytorch for distributed GPU training
dist.init_process_group(backend="nccl")

# use all ranks as tp group
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend='nccl')


@torch.no_grad()
def ag_gemm_torch(input: torch.Tensor, weight: torch.Tensor, warmup: int, iters: int):
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()

    torch.distributed.barrier()
    ## All gather input tensors from all gpus
    full_input = torch.zeros((M, input.size(1)), dtype=input.dtype, device=torch.cuda.current_device(),
                             requires_grad=False)
    torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)

    torch.distributed.barrier()
    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    allgather_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)
        allgather_end_events[i].record()
        output = torch.matmul(full_input, weight.t())
        end_events[i].record()

    comm_times = []  ## all gather
    gemm_times = []  ## gemm
    for i in range(total_iters):
        allgather_end_events[i].synchronize()
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(allgather_end_events[i]) / 1000)
            gemm_times.append(allgather_end_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times) / iters * 1000
    gemm_time = sum(gemm_times) / iters * 1000

    print(
        f"name=rank:{TP_GROUP.rank()}, output={output}, total_ms={gemm_time + comm_time}, gemm_time_ms={gemm_time}, comm_time_ms={comm_time}"
    )


def test_ipc_handle():
    M = 1024
    N = 2048  # noqa: F841
    K = 4096
    dtype = torch.int
    device = "cuda"
    shape = [M, K]

    # test pass pg to cpp
    pyrocshmem.test_ipc_handle(TP_GROUP, shape, dtype)

    # test create tensor
    local_tensor = pyrocshmem.ipc_create_tensor_and_handle(shape, dtype)
    print(f"local_tensor:{local_tensor}")

    # test ipc handle
    input_buffer = torch.zeros((M, K), dtype=dtype, device=device, requires_grad=False)
    input_buffer_offset = input_buffer.storage_offset()
    shm_handle = input_buffer._typed_storage()._share_cuda_()[1]  # cudaIpcMemHandle_t
    shm_handle = shm_handle[2:]  # skip first two bytes for rocm backend
    shm_offset = input_buffer._typed_storage()._share_cuda_()[3]
    shm_handle_ts_cuda = torch.ByteTensor(torch.ByteStorage._from_buffer(shm_handle)).cuda()
    shm_handles = [torch.empty_like(shm_handle_ts_cuda) for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather(shm_handles, shm_handle_ts_cuda, group=TP_GROUP)
    print(f"currank={TP_GROUP.rank()}, local_handle:{shm_handle_ts_cuda}")
    print(f"currank={TP_GROUP.rank()}, shm_handles:{shm_handles}")

    offset_value = shm_offset + input_buffer_offset
    offset_list = [None for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather_object(offset_list, offset_value, group=TP_GROUP)
    print(f"currank={TP_GROUP.rank()}, offset_list:{offset_list}")

    num_stages = TP_GROUP.size()
    signal = torch.zeros((num_stages), dtype=torch.int64, device=device, requires_grad=False)
    signal_tensor_offset = signal.storage_offset()

    signal_handle = signal._typed_storage()._share_cuda_()[1]
    signal_offset = signal._typed_storage()._share_cuda_()[3]
    signal_handle_ts_cuda = torch.ByteTensor(torch.ByteStorage._from_buffer(signal_handle)).cuda()
    signal_handles = [torch.empty_like(signal_handle_ts_cuda) for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather(signal_handles, signal_handle_ts_cuda, group=TP_GROUP)
    signal_handles = [handle.cpu() for handle in signal_handles]

    signal_total_offset = signal_tensor_offset + signal_offset
    signal_offset_list = [None for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather_object(signal_offset_list, signal_total_offset, group=TP_GROUP)
    print(f"currank={TP_GROUP.rank()}, signal_handles:{signal_handles}")
    print(f"currank={TP_GROUP.rank()}, signal_offset_list:{signal_offset_list}")

    shm_buffers = pyrocshmem.rocshmem_get_tensors_from_ipchandle(LOCAL_RANK, WORLD_SIZE, shm_handles, offset_list,
                                                                 input_buffer.shape, dtype)
    shm_buffers.insert(LOCAL_RANK, input_buffer)

    ipcshm_buffers = pyrocshmem.hipipc_create_tensor_list(TP_GROUP, input_buffer.shape, dtype)

    remote_tensor = ipcshm_buffers[(LOCAL_RANK + 1) % WORLD_SIZE]
    local_tensor = ipcshm_buffers[LOCAL_RANK]
    print(f"before currank={TP_GROUP.rank()}, remote_tensor:{remote_tensor}")
    remote_tensor.copy_(remote_tensor * 0 + LOCAL_RANK)
    torch.distributed.barrier()
    print(f"after currank={TP_GROUP.rank()}, local_tensor:{ipcshm_buffers[LOCAL_RANK % WORLD_SIZE]}")


if __name__ == '__main__':
    torch.cuda.set_device(LOCAL_RANK)

    # from flux test case: 4096 12288 6144
    dtype = torch.float16
    M = 4096
    N = 12288
    K = 6144

    local_M = M // TP_GROUP.size()
    local_N = N // TP_GROUP.size()

    # input: [M, K], weight: [N, K]
    input = (torch.rand((local_M, K), dtype=dtype).cuda() / 100 * ((TP_GROUP.rank() + 1)**2))
    weight = (torch.rand((local_N, K), dtype=dtype).cuda() / 100 * ((TP_GROUP.rank() + 1)**2))

    ag_gemm_torch(input, weight, 3, 10)

    pyrocshmem.rocshmem_init()
    test_ipc_handle()
    pyrocshmem.rocshmem_finalize()

    # post process
    dist.barrier()
    dist.destroy_process_group()
