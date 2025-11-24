# Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
import torch
import triton
import triton.language as tl
from triton.language.extra.maca.language_extra import tid

import time
import argparse
import os
import sys
from typing import Optional



@triton.jit
def tid_store(ptr):
    tid_ = tid(0)
    st_ptr = ptr + tl.arange(0, 64)
    tl.store(st_ptr, tid_)
    

def test_tid():
    tensor = torch.ones((128), device = "cuda", dtype = torch.int32)
    tensor_golden = torch.ones((128), device = "cuda", dtype = torch.int32)
    tid_store[(64, )](tensor)
    tensor_golden[0:64] = torch.arange(64, dtype=torch.int32)
    torch.cuda.synchronize()
    assert torch.allclose(tensor, tensor_golden)


if __name__ == "__main__":
    test_tid()
