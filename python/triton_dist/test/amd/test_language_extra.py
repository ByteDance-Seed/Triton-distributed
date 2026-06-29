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

    # CUDA semantics: lane i reads from lane max(i - delta, i_if_clamped)
    # i.e., lane i reads val[i - delta] when i >= delta, else val[i]
    golden = []
    for i in range(num_warps):
        arr = torch.arange(i * WARP_SIZE, (i + 1) * WARP_SIZE, dtype=torch.int32)
        shifted = torch.cat([arr[:delta], arr[:WARP_SIZE - delta]])
        golden.append(shifted)
    golden = torch.cat(golden).to(DEVICE)

    shfl_up_sync_kernel[(1, )](
        torch.arange(size, device=device, dtype=torch.int32),
        output,
        delta,
        64,
        num_warps=num_warps,
    )

    assert torch.allclose(output, golden), f"shfl_up mismatch:\n  got:    {output}\n  expect: {golden}"
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

    # CUDA semantics: lane i reads from lane min(i + delta, i_if_clamped)
    # i.e., lane i reads val[i + delta] when (i + delta) < WARP_SIZE, else val[i]
    golden = []
    for i in range(num_warps):
        arr = torch.arange(i * WARP_SIZE, (i + 1) * WARP_SIZE, dtype=torch.int32)
        shifted = torch.cat([arr[delta:], arr[WARP_SIZE - delta:]])
        golden.append(shifted)
    golden = torch.cat(golden).to(DEVICE)

    shfl_down_sync_kernel[(1, )](
        torch.arange(size, device=device, dtype=torch.int32),
        output,
        delta,
        64,
        num_warps=num_warps,
    )

    assert torch.allclose(output, golden), f"shfl_down mismatch:\n  got:    {output}\n  expect: {golden}"
    print("✅ [shfl_down_sync] passed.")


def test_shfl_up_down_multi_delta(device):
    """Test shuffle up/down with multiple delta values to verify clamping."""

    @triton.jit
    def shfl_up_kernel(input, output, delta, width):
        thread_idx = tid(0)
        x = ld(input + thread_idx, scope="agent", semantic="relaxed")
        y = __shfl_up_sync_i32(x, delta)
        st(output + thread_idx, y, scope="agent", semantic="relaxed")

    @triton.jit
    def shfl_down_kernel(input, output, delta, width):
        thread_idx = tid(0)
        x = ld(input + thread_idx, scope="agent", semantic="relaxed")
        y = __shfl_down_sync_i32(x, delta)
        st(output + thread_idx, y, scope="agent", semantic="relaxed")

    WARP_SIZE = 64
    num_warps = 1
    size = WARP_SIZE * num_warps

    for delta in [1, 2, 4, 16, 32, 63]:
        inp = torch.arange(size, device=device, dtype=torch.int32)

        # shfl_up: lane i reads from lane max(i - delta, clamp_to_self)
        out_up = torch.zeros(size, device=device, dtype=torch.int32)
        arr = torch.arange(WARP_SIZE, dtype=torch.int32)
        golden_up = torch.tensor([arr[max(i - delta, 0)] if i >= delta else arr[i] for i in range(WARP_SIZE)],
                                 dtype=torch.int32).to(device)
        shfl_up_kernel[(1, )](inp, out_up, delta, 64, num_warps=num_warps)
        assert torch.allclose(out_up,
                              golden_up), f"shfl_up delta={delta} mismatch:\n  got:    {out_up}\n  expect: {golden_up}"

        # shfl_down: lane i reads from lane min(i + delta, clamp_to_self)
        out_down = torch.zeros(size, device=device, dtype=torch.int32)
        golden_down = torch.tensor([arr[i + delta] if (i + delta) < WARP_SIZE else arr[i] for i in range(WARP_SIZE)],
                                   dtype=torch.int32).to(device)
        shfl_down_kernel[(1, )](inp, out_down, delta, 64, num_warps=num_warps)
        assert torch.allclose(out_down, golden_down), \
            f"shfl_down delta={delta} mismatch:\n  got:    {out_down}\n  expect: {golden_down}"

    print("✅ [shfl_up_down_multi_delta] passed.")


if __name__ == "__main__":
    test_laneid(DEVICE)
    test_shfl_sync(DEVICE)
    test_shfl_up_sync(DEVICE)
    test_shfl_down_sync(DEVICE)
    test_shfl_up_down_multi_delta(DEVICE)
