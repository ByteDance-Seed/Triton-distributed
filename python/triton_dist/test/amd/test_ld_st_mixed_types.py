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
    tid,
)

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def test_ld_st_mixed_types(device):
    """Test ld/st with mixed int32 and int64 types in the same kernel."""

    @triton.jit
    def mixed_ld_st_kernel(
        in32_ptr,
        in64_ptr,
        out32_ptr,
        out64_ptr,
        size,
    ):
        thread_idx = tid(0)
        pid = tl.program_id(axis=0)
        global_tid = pid * 256 + thread_idx

        if global_tid < size:
            val32 = ld(in32_ptr + global_tid, scope="agent", semantic="relaxed")
            val64 = ld(in64_ptr + global_tid, scope="agent", semantic="relaxed")
            st(out32_ptr + global_tid, val32 + 1, scope="agent", semantic="relaxed")
            st(out64_ptr + global_tid, val64 + 1, scope="agent", semantic="relaxed")

    SIZE = 256
    in32 = torch.arange(SIZE, device=device, dtype=torch.int32)
    in64 = torch.arange(SIZE, device=device, dtype=torch.int64)
    out32 = torch.zeros(SIZE, device=device, dtype=torch.int32)
    out64 = torch.zeros(SIZE, device=device, dtype=torch.int64)

    mixed_ld_st_kernel[(1, )](in32, in64, out32, out64, SIZE, num_warps=4)

    expected32 = in32 + 1
    expected64 = in64 + 1

    torch.testing.assert_close(out32, expected32, atol=0, rtol=0)
    torch.testing.assert_close(out64, expected64, atol=0, rtol=0)
    print("✅ [ld_st_mixed_types] passed.")


if __name__ == "__main__":
    test_ld_st_mixed_types(DEVICE)
