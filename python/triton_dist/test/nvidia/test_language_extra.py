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
import triton
import triton.language as tl
import torch
from triton_dist.language.extra.cuda.language_extra import (
    tid,
    ntid,
    laneid,
    ld,
    st,
    atomic_add,
    atomic_add_per_warp,
    __shfl_up_sync_i32,
    __shfl_down_sync_i32,
    __shfl_sync_i32,
    __shfl_xor_sync_i32,
    __ballot_sync,
    __syncthreads,
)


def test_sreg():

    @triton.jit
    def _kernel(ptr):
        st(ptr + tid(0), tid(0) * ntid(0) + laneid())

    print("[test_sreg] start...")
    result = torch.zeros((64, ), device="cuda", dtype=torch.int32)
    _kernel[(1, )](result, num_warps=2)
    torch.testing.assert_close(
        result,
        torch.arange(64, device="cuda", dtype=torch.int32) * 64 + torch.cat((
            torch.arange(32, device="cuda", dtype=torch.int32),
            torch.arange(32, device="cuda", dtype=torch.int32),
        )),
    )
    print("✅ [test_sreg] done...")


def test_ld_st():

    @triton.jit
    def _ld_st_kernel(
        input_tensor,
        output_tensor,
        scope: tl.constexpr,
        ld_semantic: tl.constexpr,
        st_semantic: tl.constexpr,
    ):
        value = ld(input_tensor + tid(0), scope=scope, semantic=ld_semantic)
        st(output_tensor + tid(0), value + 1, scope=scope, semantic=st_semantic)

    for dtype in [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
    ]:
        for scope in ["gpu", "sys"]:
            for ld_semantic, st_semantic in [
                ("relaxed", "relaxed"),
                ("acquire", "release"),
            ]:
                print(f"[ld_st] with dtype {dtype} scope {scope} semantic {ld_semantic} {st_semantic}")
                tensor_input = torch.arange(64, device="cuda", dtype=torch.int32).to(dtype)
                tensor_output = torch.zeros(64, device="cuda", dtype=dtype)
                compiled_kernel = _ld_st_kernel[(1, )](
                    tensor_input,
                    tensor_output,
                    scope=scope,
                    ld_semantic=ld_semantic,
                    st_semantic=st_semantic,
                    num_warps=2,
                )
                torch.testing.assert_close(
                    (tensor_input.to(torch.int32) + 1),
                    tensor_output.to(torch.int32),
                )
                try:
                    assert (f"st.global.{st_semantic}.{scope}.b{dtype.itemsize * 8}" in compiled_kernel.asm["ptx"])
                except AssertionError:
                    print(dtype)
                    print(compiled_kernel.asm["ptx"])
                    raise
                print(f"✅ [ld_st] with dtype {dtype} scope {scope} semantic {ld_semantic} {st_semantic} done")


def test_atomic_add():

    @triton.jit
    def _atomic_add_kernel(inptr, outptr, scope: tl.constexpr, semantic: tl.constexpr):
        value = atomic_add(inptr + tid(0), 1, scope=scope, semantic=semantic)
        st(outptr + tid(0), value, scope="gpu", semantic="relaxed")

    for dtype in [
            torch.int32,
            torch.uint32,
            torch.uint64,
    ]:
        for scope in ["gpu", "sys"]:
            for semantic in ("acquire", "release", "relaxed", "acq_rel"):
                print(f"[atomic_add] with dtype {dtype} scope {scope} semantic {semantic}")
                tensor_input = torch.arange(64, device="cuda", dtype=torch.int32).to(dtype)
                tensor_output = torch.zeros(64, device="cuda", dtype=dtype)
                compiled_kernel = _atomic_add_kernel[(1, )](
                    tensor_input,
                    tensor_output,
                    scope=scope,
                    semantic=semantic,
                    num_warps=2,
                )
                torch.testing.assert_close(
                    (tensor_input.to(torch.int32) - 1),
                    tensor_output.to(torch.int32),
                )
                sign = "s" if dtype.is_signed else "u"
                assert (f"atom.{semantic}.{scope}.global.add.{sign}{dtype.itemsize * 8}" in compiled_kernel.asm["ptx"])
                print(f"✅ [atomic_add] with dtype {dtype} scope {scope} semantic {semantic} done")


def test_shfl_sync():

    @triton.jit
    def shfl_sync_kernel(input, output, index, width):
        thread_idx = tid(0)
        x = ld(input + thread_idx, scope="gpu", semantic="relaxed")
        y = __shfl_sync_i32(0xFFFFFFFF, x, index)
        st(output + thread_idx, y, scope="gpu", semantic="relaxed")

    output = torch.zeros(64, device="cuda", dtype=torch.int32)
    delta = 5
    shfl_sync_kernel[(1, )](
        torch.arange(64, device="cuda", dtype=torch.int32),
        output,
        delta,
        32,
        num_warps=2,
    )

    assert torch.allclose(
        output,
        torch.cat((
            torch.ones(32, dtype=torch.int32) * delta,
            torch.ones(32, dtype=torch.int32) * (delta + 32),
        )).cuda()), output
    print("✅ [shfl_sync] passed.")


def test_shfl_up_sync():

    @triton.jit
    def shfl_up_sync_kernel(input, output, delta, width):
        thread_idx = tid(0)
        x = ld(input + thread_idx, scope="gpu", semantic="relaxed")
        y = __shfl_up_sync_i32(0xFFFFFFFF, x, delta)
        st(output + thread_idx, y, scope="gpu", semantic="relaxed")

    output = torch.zeros(64, device="cuda", dtype=torch.int32)
    shfl_up_sync_kernel[(1, )](
        torch.arange(64, device="cuda", dtype=torch.int32),
        output,
        1,
        32,
        num_warps=2,
    )

    assert torch.allclose(
        output,
        torch.cat((torch.max(0 * torch.ones(32, dtype=torch.int32),
                             torch.arange(32, dtype=torch.int32) - 1),
                   torch.max(32 * torch.ones(32, dtype=torch.int32),
                             torch.arange(32, 64, dtype=torch.int32) - 1))).cuda())
    print("✅ [shfl_up_sync] passed.")


def test_shfl_down_sync():

    @triton.jit
    def shfl_down_sync_kernel(input, output, delta, width):
        thread_idx = tid(0)
        x = ld(input + thread_idx, scope="gpu", semantic="relaxed")
        y = __shfl_down_sync_i32(0xFFFFFFFF, x, delta)
        st(output + thread_idx, y, scope="gpu", semantic="relaxed")

    output = torch.zeros(64, device="cuda", dtype=torch.int32)
    shfl_down_sync_kernel[(1, )](
        torch.arange(64, device="cuda", dtype=torch.int32),
        output,
        1,
        32,
        num_warps=2,
    )

    assert torch.allclose(
        output,
        torch.cat((
            torch.min(torch.arange(1, 33, dtype=torch.int32),
                      torch.ones(32, dtype=torch.int32) * 31),
            torch.min(torch.arange(33, 65, dtype=torch.int32),
                      torch.ones(32, dtype=torch.int32) * 63),
        )).cuda(),
    )
    print("✅ [shfl_down_sync] passed.")


def test_shfl_xor_sync():

    @triton.jit
    def shfl_xor_sync_kernel(input, output, delta, width):
        thread_idx = tid(0)
        x = ld(input + thread_idx, scope="gpu", semantic="relaxed")
        y = __shfl_xor_sync_i32(0xFFFFFFFF, x, delta)
        st(output + thread_idx, y, scope="gpu", semantic="relaxed")

    input = torch.arange(64, device="cuda", dtype=torch.int32)
    output = torch.zeros(64, device="cuda", dtype=torch.int32)
    shfl_xor_sync_kernel[(1, )](
        input,
        output,
        1,
        32,
        num_warps=2,
    )

    assert torch.allclose(output, input.reshape((32, 2)).flip(1).reshape((64, )))
    print("✅ [shfl_xor_sync] passed.")


def test_ballot_sync():

    @triton.jit
    def ballot_vote_kernel(input, output, value):
        thread_idx = tid(0)
        x = ld(input + thread_idx, scope="gpu", semantic="relaxed")
        y = __ballot_sync(0xffffffff, x < 32)
        st(output + thread_idx, y, scope="gpu", semantic="relaxed")

    print("[ballot_vote] start...")
    input = torch.arange(64, device="cuda", dtype=torch.int32)
    output = torch.zeros(64, device="cuda", dtype=torch.int32)
    ballot_vote_kernel[(1, )](
        input,
        output,
        1,
        num_warps=2,
    )
    assert torch.allclose(
        output,
        torch.cat((
            torch.ones(32, dtype=torch.int32) * 0xFFFFFFFF,
            torch.ones(32, dtype=torch.int32) * 0x00000000,
        )).cuda(),
    )
    print("✅ [ballot_vote] passed.")


def test_warp_prefix():

    @triton.jit
    def warp_prefix_sum_kernel(input, output):
        i = 1
        thread_idx = tl.cast(tid(axis=0), tl.int32)
        value = tl.load(input + thread_idx)
        laneid = thread_idx % 32
        while i < 32:
            val = __shfl_up_sync_i32(0xFFFFFFFF, value, i)
            if laneid >= i:
                value += val
            i = i * 2

        atomic_add(output + thread_idx, value, "gpu", "relaxed")

    print("[shfl_up_sync] start...")
    N = 1024
    in_tensor = torch.randint(0, 255, (N, ), dtype=torch.int32, device="cuda")
    out_tensor = torch.zeros_like(in_tensor)
    gt_tensor = torch.zeros_like(in_tensor)
    torch.cumsum(in_tensor.reshape(-1, 32), dim=1, out=gt_tensor)
    gt_tensor = gt_tensor.reshape(N)
    grid = (1, 1, 1)
    warp_prefix_sum_kernel[grid](in_tensor, out_tensor, num_warps=N // 32)
    assert torch.allclose(gt_tensor, out_tensor), (gt_tensor, out_tensor, in_tensor)
    print("✅ [warp_prefix] done.")


def test_atomic_add_per_warp():

    @triton.jit
    def _atomic_add_per_warp_kernel(inptr, outptr, scope: tl.constexpr, semantic: tl.constexpr):
        value = atomic_add_per_warp(inptr + tid(0), 1, scope=scope, semantic=semantic)
        st(outptr + tid(0), value, scope="gpu", semantic="relaxed")

    for dtype in [
            torch.int32,
            torch.uint32,
    ]:
        for scope in ["gpu", "sys"]:
            for semantic in ("acquire", "release", "relaxed", "acq_rel"):
                print(f"[atomic_add_per_warp] with dtype {dtype} scope {scope} semantic {semantic}")
                tensor_input = torch.arange(64, device="cuda", dtype=torch.int32).to(dtype)
                tensor_output = torch.zeros(64, device="cuda", dtype=dtype)
                _atomic_add_per_warp_kernel[(1, )](
                    tensor_input,
                    tensor_output,
                    scope=scope,
                    semantic=semantic,
                    num_warps=2,
                )
                torch.testing.assert_close(
                    tensor_input,
                    (torch.arange(64, device="cuda", dtype=torch.int32) +
                     torch.zeros(64, device="cuda", dtype=torch.int32).index_fill_(
                         0, torch.arange(0, 64, 32, device="cuda"), 1)).to(dtype),
                )
                torch.testing.assert_close(
                    tensor_output,
                    torch.zeros(64, device="cuda").index_fill_(0, torch.arange(32, 64, device="cuda"), 32).to(dtype),
                )
                print(f"✅ [atomic_add_per_warp] with dtype {dtype} scope {scope} semantic {semantic} passed.")


def test_may_hang():

    @triton.jit
    def _may_hang_kernel(ptr):
        thread_idx = tid(0)
        if thread_idx == 0:
            while tl.load(ptr, volatile=True) == 0:
                pass
        __syncthreads()
        if thread_idx == 0:
            while tl.load(ptr, volatile=True) == 0:
                pass
        __syncthreads()

    t = torch.ones((1, ), device="cuda")
    print("test may hang...")
    _may_hang_kernel[(1, )](t)
    print("✅ may hang passed.")


if __name__ == "__main__":
    test_sreg()
    test_ld_st()
    test_atomic_add()
    test_warp_prefix()
    test_shfl_sync()
    test_shfl_up_sync()
    test_shfl_down_sync()
    test_shfl_xor_sync()
    test_ballot_sync()
    test_atomic_add_per_warp()
    test_may_hang()
