################################################################################
# Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
################################################################################
"""
Libdevice (`tl.extra.libdevice`) function"
==============================
Triton can invoke a custom function from an external library.

"""

import numpy as np

import triton
import torch
import triton.language as tl
from triton.language.extra import libdevice
from triton.language.extra.hip import libdevice  # noqa: F811

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def test_extern_function_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # noqa: F841
    # x = tl.load(x_ptr + offsets, mask=mask)
    # x = libdevice.asin(x)
    x_ptr_plus_offsets = x_ptr + offsets
    x = libdevice.load_acquire_workgroup(x_ptr_plus_offsets)
    x = libdevice.load_relaxed_workgroup(x_ptr_plus_offsets)
    x = libdevice.load_acquire_agent(x_ptr_plus_offsets)
    x = libdevice.load_relaxed_agent(x_ptr_plus_offsets)
    x = libdevice.load_acquire_system(x_ptr_plus_offsets)
    x = libdevice.load_relaxed_system(x_ptr_plus_offsets)

    x = libdevice.store_release_workgroup(x_ptr_plus_offsets)
    x = libdevice.store_relaxed_workgroup(x_ptr_plus_offsets)
    x = libdevice.store_release_agent(x_ptr_plus_offsets, x)
    x = libdevice.store_relaxed_agent(x_ptr_plus_offsets)
    x = libdevice.store_release_system(x_ptr_plus_offsets, x)
    x = libdevice.store_relaxed_system(x_ptr_plus_offsets, x)

    x = libdevice.syncthreads().to(x.dtype)

    y_ptr_plus_offsets = y_ptr + offsets
    x = libdevice.red_add_release_agent(y_ptr_plus_offsets, x).to(x.dtype)
    x = libdevice.red_add_release_system(y_ptr_plus_offsets, x).to(x.dtype)

    x = libdevice.atom_add_acquire_agent(y_ptr_plus_offsets, x)
    x = libdevice.atom_add_relaxed_agent(y_ptr_plus_offsets, x)
    x = libdevice.atom_add_acqrel_agent(y_ptr_plus_offsets, x)

    x = libdevice.atom_add_acquire_system(y_ptr_plus_offsets, x)
    x = libdevice.atom_add_relaxed_system(y_ptr_plus_offsets, x)
    x = libdevice.atom_add_acqrel_system(y_ptr_plus_offsets, x)

    x = libdevice.atom_cas_acquire_relaxed_agent(y_ptr_plus_offsets, y_ptr_plus_offsets, x).to(x.dtype)
    x = libdevice.atom_cas_release_relaxed_agent(y_ptr_plus_offsets, y_ptr_plus_offsets, x).to(x.dtype)
    x = libdevice.atom_cas_relaxed_relaxed_agent(y_ptr_plus_offsets, y_ptr_plus_offsets, x).to(x.dtype)

    x = libdevice.atom_cas_acquire_relaxed_system(y_ptr_plus_offsets, y_ptr_plus_offsets, x).to(x.dtype)
    x = libdevice.atom_cas_release_relaxed_system(y_ptr_plus_offsets, y_ptr_plus_offsets, x).to(x.dtype)
    x = libdevice.atom_cas_release_relaxed_system(y_ptr_plus_offsets, y_ptr_plus_offsets, x).to(x.dtype)
    x = libdevice.atom_cas_relaxed_relaxed_system(y_ptr_plus_offsets, y_ptr_plus_offsets, x).to(x.dtype)

    tl.store(y_ptr_plus_offsets, x)
    #return x


# %%
#  Using the default libdevice library path
# -----------------------------------------
# We can use the default libdevice library path encoded in `triton/language/math.py`

torch.manual_seed(0)
size = 98432
x = torch.randint(low=0, high=2, size=(size, ), dtype=torch.int32, device=DEVICE)
output_triton = torch.zeros(size, device=DEVICE, dtype=torch.int32)
output_torch = torch.asin(x).to(dtype=torch.uint64)
assert x.is_cuda and output_triton.is_cuda
n_elements = output_torch.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
test_extern_function_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{np.max(np.abs(output_torch.cpu().numpy() - output_triton.cpu().numpy()))}')
