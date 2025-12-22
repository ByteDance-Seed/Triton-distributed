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

import os
import sys
import time
import shutil
import torch
import torch.distributed as dist


# Set default environment variables
os.environ.setdefault('TRITON_DIST_SHMEM_BACKEND', 'mori_shmem')
os.environ["MORI_SHMEM_HEAP_SIZE"] = "1G"

_test_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.abspath(os.path.join(_test_dir, "../../../.."))
_triton_dist_python_path = os.path.join(_workspace_root, "python")
if _triton_dist_python_path not in sys.path:
    sys.path.insert(0, _triton_dist_python_path)
current_pythonpath = os.environ.get('PYTHONPATH', '')
if _triton_dist_python_path not in current_pythonpath:
    os.environ['PYTHONPATH'] = f"{_triton_dist_python_path}:{current_pythonpath}" if current_pythonpath else _triton_dist_python_path

# Add upstream Triton python path
_triton_python_path = os.path.join(_workspace_root, "3rdparty/triton/python")
if os.path.exists(_triton_python_path):
    sys.path.insert(0, _triton_python_path)

from triton_dist.utils import HIP_CHECK, initialize_distributed, finalize_distributed, get_triton_dist_world
import mori.shmem as mori_shmem


# Simple helper to wrap mori shmem pointer as torch tensor
class MoriShmemBuffer:
    def __init__(self, ptr, nbytes, dtype: torch.dtype):
        self.ptr = ptr
        self.nbytes = nbytes
        self.dtype = dtype
        self.__cuda_array_interface__ = {
            "data": (self.ptr, False),
            "shape": (self.nbytes,),
            "typestr": "<i1",  # uint8
            "strides": None,
            "version": 3,
        }

def mori_shmem_create_tensor(shape, dtype):
    """Create a torch tensor backed by mori shmem memory"""
    nbytes = torch.tensor(shape).prod().item() * dtype.itemsize
    torch.cuda.synchronize()
    ptr = mori_shmem.shmem_malloc(nbytes)
    assert ptr != 0, "mori_shmem.shmem_malloc failed"
    buffer = MoriShmemBuffer(ptr, nbytes, dtype)
    tensor = torch.as_tensor(buffer, device="cuda").view(dtype).view(shape)
    tensor._mori_ptr = ptr  # Keep reference to free later
    return tensor


def test_mori_shmem_basic():
    """Run all mori shmem basic tests"""
    import triton
    import triton.language as tl
    
    @triton.jit
    def _mori_shmem_basic(ptr, mype, npes):
        tl.store(ptr, mype)
        tl.store(ptr + 1, npes)

    print("**mori_shmem basic start!")

    mype = mori_shmem.shmem_mype()
    npes = mori_shmem.shmem_npes()
    assert npes == get_triton_dist_world().size(), f"nPEs mismatch: expected {get_triton_dist_world().size()}, got {npes}"
    
    # Create tensor backed by mori shmem memory
    comm_buf = mori_shmem_create_tensor((2,), torch.int32)

    # Launch kernel
    _mori_shmem_basic[(1, )](comm_buf, mype, npes)
    mori_shmem.shmem_barrier_all()
    torch.cuda.synchronize()

    print(f"mype#{mype} comm_buf: {comm_buf}")

    try:
        torch.testing.assert_close(comm_buf, torch.tensor([mype, npes], dtype=torch.int32, device="cuda"))
    except Exception as e:
        print(f"❌ _mori_shmem_basic #{mype} failed")
        raise e
    else:
        print(f"✅ _mori_shmem_basic #{mype} pass")
    
    # Cleanup
    if hasattr(comm_buf, '_mori_ptr'):
        mori_shmem.shmem_free(comm_buf._mori_ptr)

def test_mori_shmem_device():
    import triton
    import triton.language as tl
    import triton_dist
    import triton_dist.language as dl
    from triton_dist.language.extra import libshmem_device

    @triton_dist.jit
    def _mori_shmem_device(comm_buf):
        mype = dl.rank()
        npes = dl.num_ranks()

        mype = libshmem_device.my_pe()
        npes = libshmem_device.n_pes()
        tl.store(comm_buf, mype)
        comm_buf += 1
        tl.store(comm_buf, npes)

    @triton_dist.jit
    def _mori_shmem_ring_put(ptr):
        mype = libshmem_device.my_pe()
        npes = libshmem_device.n_pes()
        peer = (mype + 1) % npes

        libshmem_device.int_p(ptr, mype, peer)


    print("**test_mori_shmem_device start!")

    mype = mori_shmem.shmem_mype()
    npes = mori_shmem.shmem_npes()
    comm_buf = mori_shmem_create_tensor((2, ), torch.int32)

    torch.distributed.barrier()
    _mori_shmem_device[(1, )](comm_buf)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    print(f"mype#{mype} comm_buf: {comm_buf}")

    try:
        torch.testing.assert_close(comm_buf, torch.tensor([mype, npes], dtype=torch.int32, device="cuda"))
    except Exception as e:
        print(f"❌ _mori_shmem_device #{mype} failed")
        raise e
    else:
        print(f"✅ _mori_shmem_device #{mype} pass")

    # Cleanup first tensor
    if hasattr(comm_buf, '_mori_ptr'):
        mori_shmem.shmem_free(comm_buf._mori_ptr)

    # Test ring put
    print("**test_mori_shmem_ring_put start!")
    
    put_buf = mori_shmem_create_tensor((1,), torch.int32)
    put_buf.fill_(-1)  # Initialize with -1
    
    torch.distributed.barrier()
    mori_shmem.shmem_barrier_all()
    
    # Each PE puts its rank to next PE in ring
    _mori_shmem_ring_put[(1,)](put_buf)
    
    torch.distributed.barrier()
    mori_shmem.shmem_barrier_all()
    torch.cuda.synchronize()
    
    # Verify: should receive from previous PE in ring
    expected_value = (mype - 1 + npes) % npes
    actual_value = put_buf[0].item()
    
    print(f"mype#{mype} ring_put result: received={actual_value}, expected={expected_value}")
    
    try:
        assert actual_value == expected_value, f"Ring put failed: expected {expected_value}, got {actual_value}"
    except Exception as e:
        print(f"❌ _mori_shmem_ring_put #{mype} failed")
        raise e
    else:
        print(f"✅ _mori_shmem_ring_put #{mype} pass")
    
    # Cleanup
    if hasattr(put_buf, '_mori_ptr'):
        mori_shmem.shmem_free(put_buf._mori_ptr)

if __name__ == "__main__":

    TP_GROUP = initialize_distributed()
    test_mori_shmem_basic()
    test_mori_shmem_device()

    finalize_distributed()
    print("All tests passed!")
