
import os
import time
import torch
import torch.distributed as dist
import torch_npu
# torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=job123 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 allgather_matmul.py

def all_gather_matmul(world_size, input, weight, all_gather_buffer, num_blocks, device, streams):
    input_chunk_size = input.size(0) // num_blocks
    handles = []
    for i in range(num_blocks):
        with torch.npu.stream(streams[i]):
            input_chunk = input[i * input_chunk_size: (i+1) * input_chunk_size]
            gather_start_idx = i * input_chunk_size * world_size
            handle = dist.all_gather_into_tensor(all_gather_buffer[gather_start_idx:gather_start_idx+input_chunk_size * world_size], 
                                                 input_chunk, async_op=True)
            handles.append((handle, gather_start_idx))
    outputs = torch.zeros(input.size(0) * world_size, weight.size(1), dtype=torch.bfloat16, device=device)
    for i in range(num_blocks):
        with torch.npu.stream(streams[i]):
            handle, gather_start_idx = handles[i]
            handle.wait()
            gathered_input = all_gather_buffer[gather_start_idx:gather_start_idx + input_chunk_size * world_size]
            outputs[i*input_chunk_size*world_size: (i+1)*input_chunk_size*world_size, :] = torch.matmul(gathered_input, weight)
    torch.npu.synchronize(device)
    return outputs


# 主函数
def main(world_size, rank, local_rank):
    torch.manual_seed(42)
    device = torch.device('npu', local_rank)
    
    matrix_size = 32 * 1024
    n = 4096
    num_blocks = 8  # 输入切分数
    streams = [torch.npu.Stream(device=device) for _ in range(num_blocks)]
    input = torch.randn(matrix_size, matrix_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(matrix_size, n, dtype=torch.bfloat16, device=device)
    all_gather_buffer = torch.zeros(matrix_size*world_size, matrix_size, dtype=torch.bfloat16, device=device)

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    ) as prof:
        for i in range(5):
            dist.barrier()
            t0 = time.time()
            output = all_gather_matmul(world_size, input, weight, all_gather_buffer, num_blocks, device, streams)
            torch.npu.synchronize()
            t1 = time.time()
            if rank == 0:
                print(f"iter:{i} e2e:{t1-t0:.5f} data:{output.mean()}")
    prof.export_chrome_trace(f"./trace_allgather_matmul_{time.time_ns()}.json")


if __name__ == "__main__":
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f'world_size={world_size}, rank={rank}, local_rank={local_rank}')
    dist.init_process_group(backend="hccl", world_size=world_size, rank=rank) 
    torch.npu.set_device(local_rank)
    main(world_size, rank, local_rank)