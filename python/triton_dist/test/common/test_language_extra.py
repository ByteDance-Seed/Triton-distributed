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

import triton_dist.language.extra.language_extra as language_extra


def _test_atomic_cas(semantic, scope, dtype):
    print(f"Testing atomic_cas with dtype={dtype}, scope={scope}, semantic={semantic}")
    device = torch.cuda.current_device()

    @triton.jit
    def _atomic_cas_kernel(ptr_p, cmp_val_p, new_val_p, out_p, semantic: tl.constexpr, scope: tl.constexpr, size):
        pid = tl.program_id(axis=0)
        block_start = pid
        ptr_plus_offsets = ptr_p + block_start
        cmp_val_plus_offsets = cmp_val_p + block_start
        new_val_plus_offsets = new_val_p + block_start

        new_val = tl.load(new_val_plus_offsets)
        cmp_val = tl.load(cmp_val_plus_offsets)
        if language_extra.tid(0) == 0:
            res = language_extra.atomic_cas(ptr_plus_offsets, cmp_val, new_val, semantic=semantic, scope=scope)
            tl.store(out_p + block_start, res)
        language_extra.__syncthreads()

    SIZE = 128 * 4
    x = torch.randint(0, 100, (SIZE, ), dtype=torch.int64, device=device)
    # "compare_cuda" not implemented for usigned int
    cmp_val = torch.randint(0, 100, (SIZE, ), dtype=torch.int64, device=device)
    cmp_val = torch.where(cmp_val < 50, cmp_val, x)

    new_val = torch.randint(0, 100, (SIZE, ), dtype=torch.int64, device=device)
    z_old = torch.empty((SIZE, ), dtype=dtype, device=device)
    x_new_ref = torch.where(x == cmp_val, new_val, x)

    x = x.to(dtype)
    cmp_val = cmp_val.to(dtype)

    new_val = new_val.to(dtype)
    x_new_ref = x_new_ref.to(dtype)

    grid = lambda meta: (SIZE, )
    x_new = x.clone().detach()
    _atomic_cas_kernel[grid](
        x_new,
        cmp_val,
        new_val,
        z_old,
        semantic=semantic,
        scope=scope,
        size=SIZE,
    )
    # Check maybe swaped result.
    torch.testing.assert_close(x_new_ref, x_new, atol=0, rtol=0, equal_nan=False)
    # Check old value
    torch.testing.assert_close(x, z_old, atol=0, rtol=0, equal_nan=False)
    print(f"✅ atom_cas_{semantic}_{scope}_{dtype} Triton and Torch match")


def _test_atomic_add(semantic, scope, dtype):

    print(f"Testing atomic_add with dtype={dtype}, scope={scope}, semantic={semantic}")
    device = torch.cuda.current_device()

    @triton.jit
    def _bincount_kernel(
        indices,
        out,
        size,
        semantic: tl.constexpr,
        scope: tl.constexpr,
    ):
        threads_per_block = language_extra.num_threads()
        tid = language_extra.tid(0)
        pid = tl.program_id(axis=0)
        num_pid = tl.num_programs(axis=0)
        total_threads = threads_per_block * num_pid
        global_tid = pid * threads_per_block + tid

        for i in range(global_tid, size, total_threads):
            idx = language_extra.ld(indices + i)
            language_extra.atomic_add(out + idx, 1, semantic=semantic, scope=scope)

    SIZE = 1024
    max_lens = 32
    indices = torch.randint(0, max_lens, (SIZE, ), dtype=torch.int64, device=device)
    num_warps = 4
    out = torch.zeros((max_lens, ), dtype=dtype, device=device)
    out_ref = torch.bincount(indices.flatten(), minlength=max_lens).to(dtype)
    indices = indices.to(dtype)

    num_sms = 16
    grid = lambda meta: (num_sms, )
    _bincount_kernel[grid](
        indices,
        out,
        SIZE,
        semantic=semantic,
        scope=scope,
        num_warps=num_warps,
    )

    torch.testing.assert_close(out_ref, out, atol=0, rtol=0, equal_nan=False)
    print(f"✅ atom_add_{semantic}_{scope}_{dtype} Triton and Torch match")


def _test_ld_st(ld_semantic, st_semantic, scope, dtype):
    print(
        f"Testing ld/store with dtype={dtype}, load_semantic={ld_semantic}, store_semantic={st_semantic}, scope={scope}"
    )
    device = torch.cuda.current_device()

    @triton.jit
    def _ld_st_kernel(
        input,
        out,
        size,
        ld_semantic: tl.constexpr,
        st_semantic: tl.constexpr,
        scope: tl.constexpr,
    ):
        threads_per_block = language_extra.num_threads()
        tid = language_extra.tid(0)
        pid = tl.program_id(axis=0)
        num_pid = tl.num_programs(axis=0)
        total_threads = threads_per_block * num_pid
        global_tid = pid * threads_per_block + tid

        for i in range(global_tid, size, total_threads):
            val = language_extra.ld(input + i, semantic=ld_semantic, scope=scope)
            language_extra.st(out + i, val + val, semantic=st_semantic, scope=scope)

    SIZE = 128 * 1024
    x = torch.randint(0, 100, (SIZE, ), dtype=dtype, device=device)
    z_ref = x + x
    z = torch.empty((SIZE, ), dtype=dtype, device=device)
    num_warps = 4
    grid = lambda meta: (SIZE, )
    _ld_st_kernel[grid](
        x,
        z,
        SIZE,
        ld_semantic=ld_semantic,
        st_semantic=st_semantic,
        scope=scope,
        num_warps=num_warps,
    )
    torch.testing.assert_close(z_ref, z, atol=0, rtol=0, equal_nan=False)
    print(f"✅ ld_{ld_semantic}_st_{st_semantic}_{scope}_{dtype} Triton and Torch match")


def test_atomic_cas_main():
    dtype_list = [torch.int32, torch.uint32, torch.int64, torch.uint64]
    scope_list = ["cta", "gpu", "sys"]
    semantic_list = ["acquire", "release", "relaxed"]
    for dtype in dtype_list:
        for scope in scope_list:
            for semantic in semantic_list:
                _test_atomic_cas(semantic, scope, dtype)


def test_atomic_add_main():
    dtype_list = [torch.int32, torch.uint32, torch.uint64]
    scope_list = ["gpu", "sys"]
    semantic_list = ["acquire", "release", "relaxed"]
    for dtype in dtype_list:
        for scope in scope_list:
            for semantic in semantic_list:
                _test_atomic_add(semantic, scope, dtype)


def test_ld_st_main():
    dtype_list = [torch.int32, torch.int64]
    scope_list = ["cta", "gpu", "sys"]
    ld_semantic_list = ["acquire", "relaxed"]
    st_semantic_list = ["release", "relaxed"]
    for dtype in dtype_list:
        for scope in scope_list:
            for ld_semantic in ld_semantic_list:
                for st_semantic in st_semantic_list:
                    _test_ld_st(ld_semantic, st_semantic, scope, dtype)


if __name__ == "__main__":

    torch.cuda.set_device(0)
    test_atomic_cas_main()
    test_atomic_add_main()
    test_ld_st_main()
    print("All tests passed!")
