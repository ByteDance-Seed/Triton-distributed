# Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
import torch
from maca import maca

src_tensor = torch.randint(100, (3, 4), device="cuda")
stream = torch.cuda.current_stream().cuda_stream
src_tensor[0] = 1
print("src", src_tensor)
maca.mcStreamWaitValue32(
        stream,
        src_tensor.data_ptr(),
        1, 
        maca.mcStreamWaitValue_flags.MC_STREAM_WAIT_VALUE_EQ
)

print("final src: ", src_tensor)
