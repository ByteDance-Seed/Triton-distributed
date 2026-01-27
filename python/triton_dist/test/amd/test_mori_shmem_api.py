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
import torch

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
    os.environ[
        'PYTHONPATH'] = f"{_triton_dist_python_path}:{current_pythonpath}" if current_pythonpath else _triton_dist_python_path

# Add upstream Triton python path
_triton_python_path = os.path.join(_workspace_root, "3rdparty/triton/python")
if os.path.exists(_triton_python_path):
    sys.path.insert(0, _triton_python_path)

from triton_dist.utils import initialize_distributed, finalize_distributed, get_triton_dist_world
import mori.shmem as mori_shmem
from mori.shmem import mori_shmem_create_tensor


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
    assert npes == get_triton_dist_world().size(
    ), f"nPEs mismatch: expected {get_triton_dist_world().size()}, got {npes}"

    # Create tensor backed by mori shmem memory
    comm_buf = mori_shmem_create_tensor((2, ), torch.int32)

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
    import triton.language as tl
    import triton_dist
    import triton_dist.language as dl
    from triton_dist.language.extra import libshmem_device

    @triton_dist.jit
    def _mori_shmem_device(comm_buf):
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

    @triton_dist.jit
    def _mori_shmem_get_put_symm_at(local_ptr):
        """Test dl.symm_at() - get remote pointer and read/write data"""
        mype = dl.rank()
        npes = dl.num_ranks()
        pid = tl.program_id(axis=0)
        boffset = pid + tl.arange(0, 4)

        # Read from all other ranks using dl.symm_at()
        for i in range(1, npes):
            src_rank = (mype + i) % npes
            # Use dl.symm_at() to get remote pointer
            remote_ptr = dl.symm_at(local_ptr, src_rank)
            rank_offset = src_rank * 4
            # Load from remote memory
            val = tl.load(remote_ptr + rank_offset + boffset)
            # Store to local memory
            tl.store(local_ptr + rank_offset + boffset, val)

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

    put_buf = mori_shmem_create_tensor((1, ), torch.int32)
    put_buf.fill_(-1)  # Initialize with -1

    torch.distributed.barrier()
    mori_shmem.shmem_barrier_all()

    # Each PE puts its rank to next PE in ring
    _mori_shmem_ring_put[(1, )](put_buf)

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

    # Test dl.symm_at() for remote memory access
    print("**test_mori_shmem_symm_at start!")

    # Check if running in multi-node environment
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    is_multinode = world_size > local_world_size

    if is_multinode:
        if mype % local_world_size == 0:
            print(
                f"   Skipping symm_at test in multi-node environment (world_size={world_size}, local_world_size={local_world_size})"
            )
            print("   symm_at() is only supported for intra-node communication")
        return

    nelems_per_rank = 4
    n_elements = npes * nelems_per_rank

    # Create symmetric buffer for all ranks
    symm_buf = mori_shmem_create_tensor((n_elements, ), torch.int32)
    ref_tensor = torch.arange(n_elements, dtype=torch.int32).cuda()

    # Each rank initializes its own portion
    symm_buf[nelems_per_rank * mype:nelems_per_rank * (mype + 1)].copy_(ref_tensor[nelems_per_rank *
                                                                                   mype:nelems_per_rank * (mype + 1)])

    torch.distributed.barrier()
    mori_shmem.shmem_barrier_all()

    # Use dl.symm_at() to read from all other ranks
    _mori_shmem_get_put_symm_at[(1, )](symm_buf)

    torch.distributed.barrier()
    mori_shmem.shmem_barrier_all()
    torch.cuda.synchronize()

    print(f"mype#{mype} symm_at result: {symm_buf}")

    try:
        torch.testing.assert_close(symm_buf, ref_tensor, atol=0, rtol=0)
    except Exception as e:
        print(f"❌ _mori_shmem_get_put_symm_at #{mype} failed")
        print(f"   Expected: {ref_tensor}")
        print(f"   Got:      {symm_buf}")
        raise e
    else:
        print(f"✅ _mori_shmem_get_put_symm_at #{mype} pass - dl.symm_at() works correctly!")

    # Cleanup
    if hasattr(symm_buf, '_mori_ptr'):
        mori_shmem.shmem_free(symm_buf._mori_ptr)


if __name__ == "__main__":

    TP_GROUP = initialize_distributed()
    test_mori_shmem_basic()
    test_mori_shmem_device()

    finalize_distributed()
    print("All tests passed!")
