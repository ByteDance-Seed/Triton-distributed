# Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
import torch
from maca import maca, macart

src_tensor = torch.randint(100, (3, 4), device="cuda")
stream = torch.cuda.current_stream().cuda_stream
print("src", src_tensor)
(err, ) = maca.mcStreamWriteValue32(
        stream,
        src_tensor.data_ptr(),
        1, 
        maca.mcStreamWriteValue_flags.MC_STREAM_WRITE_VALUE_DEFAULT
)

if isinstance(err, macart.mcError_t):
    if err != macart.mcError_t.mcSuccess:
        raise RuntimeError(f"MACA Error: {err}: {maca.mcGetErrorName(err)}")
print("final src: ", src_tensor)
