# Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
import torch
from maca import macart, maca

src_tensor = torch.randn(100, device="cuda", dtype=torch.float)
dst_tensor = torch.randn(100, device="cuda", dtype=torch.float)
size = 100*4
stream = torch.cuda.current_stream().cuda_stream
src_ptr = src_tensor
print("origin dst: ", dst_tensor)
print("src", src_tensor)
(err,) = macart.mcMemcpyAsync(
        dst_tensor.data_ptr(),
        src_tensor.data_ptr(),
        size,
        macart.mcMemcpyKind.mcMemcpyDefault,
        stream
)
print(err)

if isinstance(err, macart.mcError_t):
        if err != macart.mcError_t.mcSuccess:
                raise RuntimeError(f"maca Error: {err}: {macart.mcGetErrorString(err)}")
else:
        raise RuntimeError(f"Unknown error type: {err}")

print("final dst: ", dst_tensor)
