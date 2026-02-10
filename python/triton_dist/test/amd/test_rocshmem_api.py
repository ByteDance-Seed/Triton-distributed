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
import argparse
import os
from functools import partial

import triton
import torch
import pyrocshmem
from hip import hip
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.profiler_utils import get_torch_prof_ctx, perf_func
from triton_dist.utils import HIP_CHECK, initialize_distributed, finalize_distributed

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def test_rocshmem_device():

    @triton_dist.jit
    def _rocshmem_device(comm_buf, ptr):
        mype = dl.rank()
        npes = dl.num_ranks()

        mype = libshmem_device.my_pe()
        npes = libshmem_device.n_pes()
        tl.store(comm_buf, mype)
        comm_buf += 1
        tl.store(comm_buf, npes)

    @triton_dist.jit
    def _rocshmem_put(ptr):
        mype = libshmem_device.my_pe()
        npes = libshmem_device.n_pes()
        peer = (mype + 1) % npes

        libshmem_device.int_p(ptr, mype, peer)

    @triton_dist.jit
    def _rocshmem_get_put_symm_at(local_ptr):
        mype = libshmem_device.my_pe()
        npes = libshmem_device.n_pes()
        pid = tl.program_id(axis=0)
        boffset = pid + tl.arange(0, 4)

        for i in range(1, npes):
            src_rank = (mype + i) % npes
            remote_ptr = dl.symm_at(local_ptr, src_rank)
            rank_offset = src_rank * 4
            val = tl.load(remote_ptr + rank_offset + boffset)
            tl.store(local_ptr + rank_offset + boffset, val)

    print("**test_rocshmem_device start!")

    mype = pyrocshmem.rocshmem_my_pe()
    npes = pyrocshmem.rocshmem_n_pes()
    comm_buf = pyrocshmem.rocshmem_create_tensor((2, ), torch.int32)

    torch.distributed.barrier()
    _rocshmem_device[(1, )](comm_buf, comm_buf.data_ptr())
    torch.distributed.barrier()
    torch.cuda.synchronize()

    print(f"mype#: {mype} comm_buffs: {comm_buf}")
    try:
        torch.testing.assert_close(comm_buf, torch.tensor([mype, npes], dtype=torch.int32, device="cuda"))
    except Exception as e:
        print(f" _rocshmem_device #{mype} failed")
        raise (e)
    else:
        print(f"✅ _rocshmem_device #{mype} pass")

    put_buf = pyrocshmem.rocshmem_create_tensor((1, ), torch.int32)
    torch.distributed.barrier()
    _rocshmem_put[(1, )](put_buf)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    print(f"put_buf from pe#{mype}: {put_buf}")

    nelems_per_rank = 4
    n_elements = npes * nelems_per_rank

    put_bufs = pyrocshmem.rocshmem_create_tensor((n_elements, ), torch.int32)
    ref_tensor = torch.arange(n_elements, dtype=torch.int32).cuda()
    put_bufs[nelems_per_rank * mype:nelems_per_rank * (mype + 1)].copy_(ref_tensor[nelems_per_rank *
                                                                                   mype:nelems_per_rank * (mype + 1)])

    torch.distributed.barrier()
    _rocshmem_get_put_symm_at[(1, )](put_bufs)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    print(f"put_buf remote_ptr from pe#{mype}: {put_bufs}")

    try:
        torch.testing.assert_close(put_bufs, ref_tensor, atol=0, rtol=0)
    except Exception as e:
        print(f"❌ RANK[{mype}] check failed")
        raise e
    else:
        print(f"✅ RANK[{mype}] check passed")


def test_rocshmem_basic():

    @triton.jit
    def _rocshmem_basic(comm_buf, mype, npes):
        tl.store(comm_buf, mype)
        comm_buf += 1
        tl.store(comm_buf, npes)

    print("**rocshmem basic start!")

    mype = pyrocshmem.rocshmem_my_pe()
    npes = pyrocshmem.rocshmem_n_pes()

    comm_buf = pyrocshmem.rocshmem_create_tensor((2, ), torch.int32)

    # torch.distributed.barrier()
    _rocshmem_basic[(1, )](comm_buf, mype, npes)
    # torch.distributed.barrier()
    pyrocshmem.rocshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    print(f"mype#: {mype} comm_buffs: {comm_buf}")

    try:
        torch.testing.assert_close(comm_buf, torch.tensor([mype, npes], dtype=torch.int32, device="cuda"))
    except Exception as e:
        print(f" _rocshmem_basic #{mype} failed")
        raise (e)
    else:
        print(f"✅ _rocshmem_basic #{mype} pass")


def test_rocshmem_graph_capture():
    """
    Test CUDA/HIP graph capture compatibility with rocshmem_hipmodule_init.

    This test validates that rocshmem_hipmodule_init() works correctly within
    torch.cuda.graph() capture, which requires device-to-device copy operations
    instead of legacy stream operations (Issue #2741).
    """
    print("**test_rocshmem_graph_capture start!")

    # IMPORTANT: Define the kernel INSIDE this function to ensure it's NOT cached
    # This forces rocshmem_hipmodule_init to be called during graph capture
    @triton_dist.jit
    def _graph_test_kernel(result_buf):
        """Kernel for graph capture test - must be unique to avoid caching"""
        mype = libshmem_device.my_pe()
        npes = libshmem_device.n_pes()
        # Write rank-specific value to verify execution
        tl.store(result_buf, mype + 42)

    mype = pyrocshmem.rocshmem_my_pe()
    npes = pyrocshmem.rocshmem_n_pes()

    # Create result buffer
    result_buf = pyrocshmem.rocshmem_create_tensor((1,), torch.int32)
    result_buf.fill_(0)  # Initialize to 0

    print(f"[Rank {mype}] Creating CUDA graph for rocshmem_hipmodule_init test...")

    # Create a CUDA graph
    graph = torch.cuda.CUDAGraph()

    # Use a dedicated stream for graph capture
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        # DO NOT warm up - we want rocshmem_hipmodule_init to be called
        # DURING graph capture to test if it's graph compatible

        # Begin graph capture IMMEDIATELY
        print(f"[Rank {mype}] Beginning graph capture (first kernel launch will call rocshmem_hipmodule_init)...")
        graph.capture_begin()

        try:
            # Launch kernel within graph capture
            # This is the FIRST launch, so rocshmem_hipmodule_init will be called NOW
            # With OLD implementation: hipMemcpyFromSymbol uses legacy stream -> SHOULD FAIL
            # With NEW implementation: device-to-device copy -> SHOULD SUCCEED
            _graph_test_kernel[(1,)](result_buf)

            # End graph capture
            graph.capture_end()
            print(f"[Rank {mype}] ✅ Graph capture completed successfully")
            print(f"[Rank {mype}] ✅ rocshmem_hipmodule_init worked within graph capture!")

        except Exception as e:
            print(f"[Rank {mype}] ❌ Graph capture FAILED: {e}")
            print(f"[Rank {mype}] ❌ This proves rocshmem_hipmodule_init is NOT graph compatible")
            try:
                graph.capture_end()
            except:
                pass
            raise

    # Synchronize before replay
    torch.cuda.synchronize()

    # Reset buffer before graph replay
    result_buf.fill_(0)

    # Replay the captured graph
    print(f"[Rank {mype}] Replaying captured graph...")
    graph.replay()
    torch.cuda.synchronize()

    # Verify result
    expected_value = mype + 42
    actual_value = result_buf.item()

    print(f"[Rank {mype}] Graph replay result: expected={expected_value}, actual={actual_value}")

    try:
        torch.testing.assert_close(
            result_buf,
            torch.tensor([expected_value], dtype=torch.int32, device="cuda"),
            atol=0,
            rtol=0
        )
    except Exception as e:
        print(f"[Rank {mype}] ❌ Graph capture test FAILED - result mismatch")
        raise e
    else:
        print(f"[Rank {mype}] ✅ Graph replay verification PASSED")
        print(f"[Rank {mype}] ✅ CONFIRMED: rocshmem_hipmodule_init is CUDA graph compatible!")


def test_rocshmem_memcpy():
    print("**rocshmem memcpy start!")

    mype = pyrocshmem.rocshmem_my_pe()
    npes = pyrocshmem.rocshmem_n_pes()
    peer = (mype + 1) % npes

    nelems_per_rank = 1024 * 1024

    comm_buffs = pyrocshmem.rocshmem_create_tensor_list_intra_node([nelems_per_rank], torch.int32)
    comm_buffs[mype].fill_(0)

    torch.cuda.synchronize()

    one = torch.arange(nelems_per_rank, dtype=torch.int32, device=torch.cuda.current_device())
    cur_stream = torch.cuda.current_stream()
    ag_streams = [torch.cuda.Stream(priority=-1) for i in range(npes)]

    torch.cuda.synchronize()
    pyrocshmem.rocshmem_barrier_all_on_stream(cur_stream.cuda_stream)

    with torch.cuda.stream(ag_streams[mype]):
        for i in range(npes):
            remote_rank = (i + mype) % npes
            if remote_rank == i:
                continue
            stream = ag_streams[i]
            dst_ptr = comm_buffs[remote_rank].data_ptr()
            src_ptr = one.data_ptr()
            nbytes = nelems_per_rank * one.element_size()
            cp_res = hip.hipMemcpyAsync(
                dst_ptr,
                src_ptr,
                nbytes,
                hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                stream.cuda_stream,
            )

            HIP_CHECK(cp_res)

    pyrocshmem.rocshmem_barrier_all_on_stream(cur_stream.cuda_stream)

    torch.cuda.synchronize()

    try:
        torch.testing.assert_close(comm_buffs[peer], one)
    except Exception as e:
        print(f" _rocshmem_memcpy #{mype} - Check tensor_list failed")
        raise (e)
    else:
        print(f"✅ _rocshmem_memcpy #{mype} - Check tensor_list pass")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")

    return parser.parse_args()


if __name__ == "__main__":
    # init
    args = parse_args()
    nbytes = 1024 * 1024 * 4

    TP_GROUP = initialize_distributed()

    test_rocshmem_basic()

    test_rocshmem_device()

    test_rocshmem_graph_capture()

    test_rocshmem_memcpy()

    ctx = get_torch_prof_ctx(args.profile)

    with ctx:
        perf = perf_func(partial(test_rocshmem_memcpy), iters=10, warmup_iters=5)

    torch.cuda.synchronize()
    torch.distributed.barrier()

    print(f"rocSHMEM #{RANK} ", perf)

    finalize_distributed()
