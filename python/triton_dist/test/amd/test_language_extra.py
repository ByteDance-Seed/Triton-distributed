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
import torch
import triton
import triton.language as tl
from triton_dist.language.extra.hip.language_extra import (
    ld,
    st,
    laneid,
    tid,
    __shfl_sync_i32,
    __shfl_up_sync_i32,
    __shfl_down_sync_i32,
)

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def test_laneid(device):

    @triton.jit
    def store_laneid_kernel(inp_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        lid = laneid()
        tid = pid * 64 + lid
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a = tl.load(inp_ptr + offsets)
        res = a + tid
        tl.store(out_ptr + offsets, res)

    SIZE = 8 * 64
    dtype = torch.int32
    inp = torch.ones((SIZE, ), dtype=dtype, device=device)
    tri_out = torch.empty_like(inp)
    tids = torch.arange(SIZE).to(dtype).to(device)
    ref_out = inp + tids
    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']), )
    store_laneid_kernel[grid](
        inp,
        tri_out,
        BLOCK_SIZE=64,
    )
    torch.testing.assert_close(tri_out, ref_out, equal_nan=True)
    print("✅ [laneid] passed")


def test_shfl_sync(device):

    @triton.jit
    def shfl_sync_kernel(input, output, index, width):
        thread_idx = tid(0)
        x = ld(input + thread_idx, scope="agent", semantic="relaxed")
        y = __shfl_sync_i32(x, index)
        st(output + thread_idx, y, scope="agent", semantic="relaxed")

    WARP_SIZE = 64
    num_warps = 2
    size = WARP_SIZE * num_warps

    output = torch.zeros(size, device=device, dtype=torch.int32)
    delta = 5

    golden = torch.cat((torch.ones(WARP_SIZE, dtype=torch.int32) * delta,
                        torch.ones(WARP_SIZE, dtype=torch.int32) * (delta + WARP_SIZE))).to(DEVICE)

    shfl_sync_kernel[(1, )](
        torch.arange(size, device=device, dtype=torch.int32),
        output,
        delta,
        64,
        num_warps=num_warps,
    )

    assert torch.allclose(output, golden), output
    print("✅ [shfl_sync] passed.")


def test_shfl_up_sync(device):

    @triton.jit
    def shfl_up_sync_kernel(input, output, index, width):
        thread_idx = tid(0)
        x = ld(input + thread_idx, scope="agent", semantic="relaxed")
        y = __shfl_up_sync_i32(x, index)
        st(output + thread_idx, y, scope="agent", semantic="relaxed")

    WARP_SIZE = 64
    num_warps = 2
    size = WARP_SIZE * num_warps

    output = torch.zeros(size, device=device, dtype=torch.int32)
    delta = 1

    golden = []
    for i in range(num_warps):
        golden.append(
            torch.min((i + 1) * WARP_SIZE * torch.ones(WARP_SIZE, dtype=torch.int32) - 1,
                      torch.arange(i * WARP_SIZE, (i + 1) * WARP_SIZE, dtype=torch.int32) + delta))
    golden = torch.cat(golden).to(DEVICE)

    shfl_up_sync_kernel[(1, )](
        torch.arange(size, device=device, dtype=torch.int32),
        output,
        delta,
        64,
        num_warps=num_warps,
    )

    assert torch.allclose(output, golden), output
    print("✅ [shfl_up_sync] passed.")


def test_shfl_down_sync(device):

    @triton.jit
    def shfl_down_sync_kernel(input, output, index, width):
        thread_idx = tid(0)
        x = ld(input + thread_idx, scope="agent", semantic="relaxed")
        y = __shfl_down_sync_i32(x, index)
        st(output + thread_idx, y, scope="agent", semantic="relaxed")

    WARP_SIZE = 64
    num_warps = 2
    size = WARP_SIZE * num_warps

    output = torch.zeros(size, device=device, dtype=torch.int32)
    delta = 1

    assert delta >= 0

    golden = []
    for i in range(num_warps):
        arr = torch.arange(i * WARP_SIZE, (i + 1) * WARP_SIZE, dtype=torch.int32)
        left_arr = arr[:WARP_SIZE - delta]
        right_arr = arr[WARP_SIZE - delta:]
        golden.append(torch.cat((
            right_arr,
            left_arr,
        )))
    golden = torch.cat(golden).to(DEVICE)

    shfl_down_sync_kernel[(1, )](
        torch.arange(size, device=device, dtype=torch.int32),
        output,
        delta,
        64,
        num_warps=num_warps,
    )

    assert torch.allclose(output, golden), output
    print("✅ [shfl_down_sync] passed.")


if __name__ == "__main__":
    test_laneid(DEVICE)
    test_shfl_sync(DEVICE)
    test_shfl_up_sync(DEVICE)
    test_shfl_down_sync(DEVICE)
