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
import pytest
import torch

import triton
import triton.language as tl

from triton.language.extra.hip import libdevice as libdevice

DEVICE = triton.runtime.driver.active.get_active_torch_device()

## distributed program functions

## distributed load-line functions


@pytest.mark.parametrize("libdevice_load_fn", [
    "load_acquire_workgroup",
    "load_relaxed_workgroup",
    "load_acquire_agent",
    "load_relaxed_agent",
    "load_acquire_system",
    "load_relaxed_system",
])
@pytest.mark.parametrize("device", [DEVICE])
def test_load(libdevice_load_fn, device):

    @triton.jit
    def load_kernel(in_p, out_p, size, fn: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        in_ptr_plus_offsets = in_p + offsets
        x = getattr(libdevice, fn)(in_ptr_plus_offsets)
        out_ptr_plus_offsets = out_p + offsets
        tl.store(out_ptr_plus_offsets, x)

    SIZE = 128 * 64
    dtype = torch.int32

    x = torch.randint(0, 100, (SIZE, ), dtype=dtype, device=device)
    y_exp = torch.empty((SIZE, ), dtype=dtype, device=device)
    y_ref = x.clone()

    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']), )
    load_kernel[grid](
        x,
        y_exp,
        SIZE,
        fn=libdevice_load_fn,
        BLOCK_SIZE=128,
    )
    torch.testing.assert_close(y_ref, y_exp, equal_nan=True)
    print("✅ Triton and Torch match")


## distributed store-line functions


@pytest.mark.parametrize("libdevice_store_fn", [
    #"store_release_workgroup",
    #"store_relaxed_workgroup",
    "store_release_agent",
    #"store_relaxed_agent",
    "store_release_system",
    "store_relaxed_system",
])
@pytest.mark.parametrize("device", [DEVICE])
def test_store(libdevice_store_fn, device):

    @triton.jit
    def store_kernel(in_p, out_p, size, fn: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < size
        in_ptr_plus_offsets = in_p + offsets
        x = tl.load(in_ptr_plus_offsets, mask=mask)
        out_ptr_plus_offsets = out_p + offsets
        getattr(libdevice, fn)(out_ptr_plus_offsets, x)

    SIZE = 128 * 64
    dtype = torch.int32

    x = torch.randint(0, 100, (SIZE, ), dtype=dtype, device=device)
    y_exp = torch.empty((SIZE, ), dtype=dtype, device=device)
    y_ref = x.clone()

    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']), )
    store_kernel[grid](
        x,
        y_exp,
        SIZE,
        fn=libdevice_store_fn,
        BLOCK_SIZE=128,
    )
    torch.testing.assert_close(y_ref, y_exp, equal_nan=True)
    print("✅ Triton and Torch match")


# NOTE: The following store_xxx functions are used to set signal to constant one only.
@pytest.mark.parametrize("libdevice_store_fn", [
    "store_release_workgroup",
    "store_relaxed_workgroup",
    "store_relaxed_agent",
])
@pytest.mark.parametrize("device", [DEVICE])
def test_store_one(libdevice_store_fn, device):

    @triton.jit
    def store_kernel(out_p, size, fn: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        out_ptr_plus_offsets = out_p + offsets
        getattr(libdevice, fn)(out_ptr_plus_offsets)

    SIZE = 128 * 64
    dtype = torch.int32

    y_exp = torch.empty((SIZE, ), dtype=dtype, device=device)
    y_ref = torch.ones((SIZE, ), dtype=dtype, device=device)

    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']), )
    store_kernel[grid](
        y_exp,
        SIZE,
        fn=libdevice_store_fn,
        BLOCK_SIZE=128,
    )
    torch.testing.assert_close(y_ref, y_exp, equal_nan=True)
    print("✅ Triton and Torch match")


## distributed atomic functions


@pytest.mark.parametrize("libdevice_atomic_add_fn", [
    "atom_add_acquire_agent",
    "atom_add_relaxed_agent",
    "atom_add_acqrel_agent",
    "atom_add_acquire_system",
    "atom_add_relaxed_system",
    "atom_add_acqrel_system",
])
@pytest.mark.parametrize("device", [DEVICE])
def test_atomic_add(libdevice_atomic_add_fn, device):

    @triton.jit
    def atomic_add_kernel(
        lhs_p,
        rhs_p,
        out_p,
        size,
        fn: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < size
        lhs_ptr_plus_offsets = lhs_p + offsets
        rhs_ptr_plus_offsets = rhs_p + offsets
        rhs = tl.load(rhs_ptr_plus_offsets)

        res = getattr(libdevice, fn)(lhs_ptr_plus_offsets, rhs)
        #res = libdevice.atom_add_acquire_agent(lhs_ptr_plus_offsets, rhs)
        tl.store(out_p + offsets, res, mask=mask)

    num_warps = 1
    thread_per_warp = 64  # AMD specified
    thread_per_cta = num_warps * thread_per_warp

    SIZE = 128 * 4
    dtype = torch.int32

    for BLOCK_SIZE in [32, 64, 128]:
        x = torch.randint(0, 100, (SIZE, ), dtype=dtype, device=device)
        y = torch.randint(0, 100, (SIZE, ), dtype=dtype, device=device)
        z_old = torch.empty((SIZE, ), dtype=dtype, device=device)
        if BLOCK_SIZE < thread_per_cta:
            z_ref = x + y * (thread_per_warp // BLOCK_SIZE)
        else:
            z_ref = x + y
        grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']), )
        atomic_add_kernel[grid](
            x,
            y,
            z_old,
            SIZE,
            libdevice_atomic_add_fn,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        torch.testing.assert_close(z_ref, x, equal_nan=True)
        print("✅ Triton and Torch match")


@pytest.mark.parametrize("libdevice_atomic_cas_fn", [
    "atom_cas_acquire_relaxed_agent",
    "atom_cas_release_relaxed_agent",
    "atom_cas_relaxed_relaxed_agent",
    "atom_cas_acqrel_relaxed_agent",
    "atom_cas_acquire_relaxed_system",
    "atom_cas_release_relaxed_system",
    "atom_cas_relaxed_relaxed_system",
    "atom_cas_acqrel_relaxed_system",
])
@pytest.mark.parametrize("device", [DEVICE])
def test_atomic_cas(libdevice_atomic_cas_fn, device):

    @triton.jit
    def atomic_cas_kernel(ptr_p, cmp_val_p, new_val_p, out_p, fn: tl.constexpr, size, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < size
        ptr_plus_offsets = ptr_p + offsets
        cmp_val_plus_offsets = cmp_val_p + offsets
        new_val_plus_offsets = new_val_p + offsets

        new_val = tl.load(new_val_plus_offsets, mask=mask)

        res = getattr(libdevice, fn)(ptr_plus_offsets, cmp_val_plus_offsets, new_val)
        tl.store(out_p + offsets, res)

    SIZE = 128 * 4
    dtype = torch.int32
    x = torch.randint(0, 100, (SIZE, ), dtype=dtype, device=device)
    cmp_val = torch.randint(0, 100, (SIZE, ), dtype=dtype, device=device)
    cmp_val = torch.where(cmp_val < 50, cmp_val, x)
    new_val = torch.randint(0, 100, (SIZE, ), dtype=dtype, device=device)
    z_old = torch.empty((SIZE, ), dtype=dtype, device=device)
    x_new_ref = torch.where(x == cmp_val, new_val, x)

    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']), )
    x_new = x.clone().detach()
    atomic_cas_kernel[grid](
        x_new,
        cmp_val,
        new_val,
        z_old,
        fn=libdevice_atomic_cas_fn,
        size=SIZE,
        BLOCK_SIZE=128,
    )
    # Check maybe swaped result.
    torch.testing.assert_close(x_new_ref, x_new, equal_nan=True)
    # Check old value
    torch.testing.assert_close(x, z_old, equal_nan=True)
    print("✅ Triton and Torch match")
