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
from hip import hip
import torch


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


M = 1
N = 8
world_size = 4
input_shape = (M, N)
output_shape = (M * world_size, N)

src_tensor_list = []
dst_tensor_list = []
signal_tensor_list = []

for rank in range(world_size):
    input = torch.empty(input_shape, dtype=torch.float32, device=f"cuda:{rank}")
    input.fill_(rank)
    src_tensor_list.append(input)
    output = torch.empty(output_shape, dtype=torch.float32, device=f"cuda:{rank}")
    dst_tensor_list.append(output)
    signal = torch.zeros((world_size, ), dtype=torch.int32, device=f"cuda:{rank}")
    signal_tensor_list.append(signal)


def all_gather_cp_engine(src_tensor_list, signal, dst_tensor, rank, stream):

    for offset in range(world_size):
        src_rank = (rank + offset) % world_size
        M = input_shape[0]
        src_tensor = src_tensor_list[src_rank]
        nbytes = src_tensor.numel() * src_tensor.element_size()
        call_result = hip.hipMemcpyAsync(
            dst_tensor[M * src_rank:M * (src_rank + 1)].data_ptr(),
            src_tensor.data_ptr(),
            nbytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
            stream.cuda_stream,
        )
        hip_check(call_result)

        call_result = hip.hipStreamWriteValue32(
            stream.cuda_stream,
            signal[src_rank].data_ptr(),
            1,
            0,
        )
        hip_check(call_result)


def consumer(signal, dst_tensor, stream):
    mask = 0xFFFFFFFF
    result = torch.zeros((input_shape), dtype=torch.float32, device=dst_tensor.device)
    for rank in range(world_size):
        call_result = hip.hipStreamWaitValue32(stream.cuda_stream, signal[rank].data_ptr(), 1, hip.hipStreamWaitValueEq,
                                               mask)
        hip_check(call_result)
        result = result + dst_tensor[M * rank:M * (rank + 1)]
    return result


if __name__ == "__main__":

    current_stream = torch.cuda.current_stream()
    ag_stream = torch.cuda.Stream()
    for rank in range(world_size):
        all_gather_cp_engine(src_tensor_list, signal_tensor_list[rank], dst_tensor_list[rank], rank, ag_stream)
    for rank in range(world_size):
        result = consumer(signal, dst_tensor_list[rank], current_stream)
        print(result)
    current_stream.wait_stream(ag_stream)
