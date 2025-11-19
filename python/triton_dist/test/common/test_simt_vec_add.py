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
import triton_dist.language as tl_dist
from triton_dist.profiler_utils import perf_func
from functools import partial


@triton.jit
def simt_add(x, y, out, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_tiles = tl.cdiv(n, BLOCK_SIZE)
    vec_size: tl.constexpr = 128 // x.dtype.element_ty.primitive_bitwidth
    tl.static_assert(BLOCK_SIZE % vec_size == 0)
    with tl_dist.simt_exec_region() as (thread_idx, num_threads):
        for tile_id in range(pid, num_tiles, num_pids):
            start_vec = tile_id * BLOCK_SIZE
            end_vec = min(n, start_vec + BLOCK_SIZE) // vec_size * vec_size
            # perfect case
            for j in range(start_vec + thread_idx * vec_size, end_vec, num_threads * vec_size):
                vec_x = tl_dist.ld_vector(x + j, vec_size=vec_size)
                vec_y = tl_dist.ld_vector(y + j, vec_size=vec_size)
                vec_z = (vec_x + vec_y + 1) * 2
                tl_dist.st_vector(out + j, vec_z)

            # non-perfect case: the last vector in the last tile
            if start_vec + BLOCK_SIZE >= n:
                for j in range(end_vec + thread_idx, n, num_threads):
                    vec_x = tl_dist.ld_vector(x + j, vec_size=1)
                    vec_y = tl_dist.ld_vector(y + j, vec_size=1)
                    vec_z = (vec_x + vec_y + 1) * 2
                    tl_dist.st_vector(out + j, vec_z)


def test_simt_add_vec(n, dtype):
    print(f"testing with n = {n}, dtype = {dtype}")
    if dtype.is_floating_point:
        x = torch.randn((n, ), dtype=dtype, device=torch.cuda.current_device())
        y = torch.randn((n, ), dtype=dtype, device=torch.cuda.current_device())
    else:
        x = torch.randint(low=0, high=32768, size=(n, ), dtype=dtype, device=torch.cuda.current_device())
        y = torch.randint(low=0, high=32768, size=(n, ), dtype=dtype, device=torch.cuda.current_device())
    out_ref = (x + y + 1) * 2

    def _add(x, y):
        n = x.shape[0]
        out_triton = torch.empty((n, ), dtype=dtype, device=torch.cuda.current_device())
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
        simt_add[grid](x, y, out_triton, n, BLOCK_SIZE=1024)
        return out_triton

    out_triton, perf_triton = perf_func(partial(_add, x, y), 100, 20)
    print(f"perf_triton = {perf_triton}, BW = {n * dtype.itemsize * 3 / 1e12 / (perf_triton / 1e3)} TB/s")
    torch.testing.assert_close(out_ref, out_triton, atol=0, rtol=0, equal_nan=False)


if __name__ == "__main__":
    torch.cuda.set_device(0)

    n = 8192 * 7168
    for dtype in [torch.int16, torch.int32, torch.bfloat16, torch.float16, torch.float32]:
        test_simt_add_vec(n, dtype)
        test_simt_add_vec(n - 1, dtype)
    print("pass!")
