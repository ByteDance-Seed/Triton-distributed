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
"""
Multi-GPU correctness test for FlashComm dispatch + combine pipeline.

End-to-end test:
  1. compute_offset + dispatch_layout  (per-rank, single GPU)
  2. dispatch_intranode_v1             (cross-rank TMA copies)
  3. Verify received tokens match reference

Tests 2-GPU, 4-GPU, and 8-GPU configurations using peer-accessible memory.
Cross-GPU buffers are allocated via raw cudaMalloc (not PyTorch caching allocator)
to ensure proper P2P accessibility.

Usage:
  python design/test_flashcomm_multi_gpu.py                 # auto: test 2,4,8 GPUs
  python design/test_flashcomm_multi_gpu.py --num-ranks 2   # force 2 GPUs
"""

import os
import sys
import time
import threading
import argparse
import ctypes

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.cuda

from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

from flashcomm_compute import (
    kernel_compute_offset,
    kernel_compute_dispatch_layout,
    NUM_WARPS,
    MAX_EXPERTS_PLUS_1,
    BLOCK_SIZE,
)
from flashcomm_dispatch import (
    kernel_dispatch_intranode_v1,
    TOTAL_SMEM as DISPATCH_SMEM,
    V1_TOTAL_THREADS,
)
from flashcomm_barrier import (
    kernel_barrier_all_on_stream,
    NUM_THREADS as BARRIER_THREADS,
)

# ============================================================================
# CUDA Runtime API bindings for raw memory allocation
# ============================================================================
_libcudart = ctypes.CDLL('libcudart.so')

_libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_libcudart.cudaMalloc.restype = ctypes.c_int
_libcudart.cudaFree.argtypes = [ctypes.c_void_p]
_libcudart.cudaFree.restype = ctypes.c_int
_libcudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
_libcudart.cudaMemset.restype = ctypes.c_int
_libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libcudart.cudaMemcpy.restype = ctypes.c_int
_libcudart.cudaSetDevice.argtypes = [ctypes.c_int]
_libcudart.cudaSetDevice.restype = ctypes.c_int
_libcudart.cudaDeviceSynchronize.restype = ctypes.c_int
_libcudart.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
_libcudart.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
_libcudart.cudaDeviceCanAccessPeer.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
_libcudart.cudaDeviceCanAccessPeer.restype = ctypes.c_int
_libcudart.cudaGetLastError.argtypes = []
_libcudart.cudaGetLastError.restype = ctypes.c_int

cudaMemcpyDeviceToHost = 2
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToDevice = 3


class CudaMallocBuffer:
    """Raw cudaMalloc buffer that is P2P accessible (unlike PyTorch tensors)."""

    def __init__(self, size_bytes: int, gpu_id: int):
        self.gpu_id = gpu_id
        self.size_bytes = size_bytes
        self.ptr = ctypes.c_void_p()
        _libcudart.cudaSetDevice(gpu_id)
        err = _libcudart.cudaMalloc(ctypes.byref(self.ptr), size_bytes)
        assert err == 0, f"cudaMalloc failed on GPU {gpu_id}: err={err}"
        _libcudart.cudaMemset(self.ptr, 0, size_bytes)

    def data_ptr(self):
        return self.ptr.value

    def zero_(self):
        _libcudart.cudaSetDevice(self.gpu_id)
        _libcudart.cudaMemset(self.ptr, 0, self.size_bytes)

    def to_torch(self, dtype=torch.int32, shape=None):
        """Copy contents to a PyTorch tensor for inspection."""
        _libcudart.cudaSetDevice(self.gpu_id)
        if dtype == torch.int32:
            elem_size = 4
        elif dtype == torch.float32:
            elem_size = 4
        elif dtype == torch.bfloat16:
            elem_size = 2
        elif dtype == torch.int64:
            elem_size = 8
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        n_elems = self.size_bytes // elem_size
        if shape is None:
            shape = (n_elems, )
        t = torch.zeros(shape, dtype=dtype, device=f'cuda:{self.gpu_id}')
        _libcudart.cudaMemcpy(ctypes.c_void_p(t.data_ptr()), self.ptr, self.size_bytes, cudaMemcpyDeviceToDevice)
        return t

    def from_torch(self, tensor: torch.Tensor):
        """Copy from a PyTorch tensor to this buffer."""
        _libcudart.cudaSetDevice(self.gpu_id)
        _libcudart.cudaMemcpy(self.ptr, ctypes.c_void_p(tensor.data_ptr()), self.size_bytes, cudaMemcpyDeviceToDevice)

    def free(self):
        if self.ptr.value:
            _libcudart.cudaSetDevice(self.gpu_id)
            _libcudart.cudaFree(self.ptr)
            self.ptr = ctypes.c_void_p()

    def __del__(self):
        self.free()


# ============================================================================
# Helpers
# ============================================================================
def find_free_gpus(n_needed: int) -> list:
    """Find GPUs with least memory used."""
    all_gpus = []
    for i in range(torch.cuda.device_count()):
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        used_mb = (total_mem - free_mem) / (1024 * 1024)
        all_gpus.append((used_mb, i))
    all_gpus.sort()
    return [g[1] for g in all_gpus[:n_needed]]


_peer_access_enabled = set()


def enable_peer_access(gpu_ids: list):
    """Enable P2P memory access between all GPU pairs (idempotent)."""
    for i in gpu_ids:
        _libcudart.cudaSetDevice(i)
        for j in gpu_ids:
            if i != j and (i, j) not in _peer_access_enabled:
                err = _libcudart.cudaDeviceEnablePeerAccess(j, 0)
                if err == 0:
                    _peer_access_enabled.add((i, j))
                elif err == 704:  # already enabled
                    _peer_access_enabled.add((i, j))
                    # Clear the error state
                    _libcudart.cudaGetLastError()
                else:
                    print(f"  Warning: cudaDeviceEnablePeerAccess({i}->{j}) returned {err}")
                    _libcudart.cudaGetLastError()


def make_ptr_tensor(buffers_or_ptrs, target_device: int) -> torch.Tensor:
    """Create int64 tensor of data_ptr() values on target_device."""
    ptrs = []
    for b in buffers_or_ptrs:
        if isinstance(b, CudaMallocBuffer):
            ptrs.append(b.data_ptr())
        elif isinstance(b, torch.Tensor):
            ptrs.append(b.data_ptr())
        else:
            ptrs.append(int(b))
    return torch.tensor(ptrs, dtype=torch.int64, device=f"cuda:{target_device}")


# ============================================================================
# Reference implementation
# ============================================================================
def reference_global_rank(flat_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Stable within-expert offset."""
    M = flat_idx.numel()
    perm = torch.argsort(flat_idx, stable=True)
    k = flat_idx[perm]
    i = torch.arange(M, device=flat_idx.device, dtype=torch.int64)
    head = torch.empty(M, device=flat_idx.device, dtype=torch.bool)
    head[0] = True
    head[1:] = k[1:] != k[:-1]
    boundary = torch.where(head, i, torch.full_like(i, -1))
    run_start = torch.cummax(boundary, dim=0).values
    rank_sorted = (i - run_start).to(torch.int32)
    out = torch.empty(M, device=flat_idx.device, dtype=torch.int32)
    out[perm] = rank_sorted
    return out


def reference_expert_counts(flat_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    return torch.bincount(flat_idx, minlength=num_experts + 1).to(torch.int32)


def reference_dispatch(
    all_topk_indices: list,
    all_x: list,
    all_weights: list,
    num_experts: int,
    num_ranks: int,
):
    """
    Pure-Python reference for dispatch. Returns expected recv_x, recv_weights,
    recv_token_count for each rank.
    """
    experts_per_rank = num_experts // num_ranks
    device = all_x[0].device
    hidden = all_x[0].shape[1]
    topk = all_topk_indices[0].shape[1]

    all_offsets = []
    all_counts = []
    for r in range(num_ranks):
        flat = all_topk_indices[r].flatten().to(device)
        off = reference_global_rank(flat, num_experts).view(-1, topk)
        cnt = reference_expert_counts(flat, num_experts)
        all_offsets.append(off)
        all_counts.append(cnt)

    recv_counts = torch.zeros(num_ranks, dtype=torch.int32, device=device)
    for target_rank in range(num_ranks):
        total = 0
        for src_rank in range(num_ranks):
            for e in range(target_rank * experts_per_rank, (target_rank + 1) * experts_per_rank):
                total += all_counts[src_rank][e].item()
        recv_counts[target_rank] = total

    recv_base = {}
    for target_rank in range(num_ranks):
        offset = 0
        recv_base[target_rank] = {}
        for src_rank in range(num_ranks):
            recv_base[target_rank][src_rank] = {}
            for local_e in range(experts_per_rank):
                global_e = target_rank * experts_per_rank + local_e
                recv_base[target_rank][src_rank][local_e] = offset
                offset += all_counts[src_rank][global_e].item()

    max_recv = int(recv_counts.max().item()) + 1024
    ref_recv_x = [torch.zeros(max_recv, hidden, dtype=torch.bfloat16, device=device) for _ in range(num_ranks)]
    ref_recv_weights = [torch.zeros(max_recv, dtype=torch.float32, device=device) for _ in range(num_ranks)]

    for src_rank in range(num_ranks):
        num_token = all_topk_indices[src_rank].shape[0]
        for t in range(num_token):
            for k_slot in range(topk):
                expert_idx = all_topk_indices[src_rank][t, k_slot].item()
                if expert_idx >= num_experts:
                    continue
                target_rank = expert_idx // experts_per_rank
                local_expert = expert_idx % experts_per_rank
                within_offset = all_offsets[src_rank][t, k_slot].item()
                base = recv_base[target_rank][src_rank][local_expert]
                dst_idx = base + within_offset
                ref_recv_x[target_rank][dst_idx] = all_x[src_rank][t].to(torch.bfloat16)
                ref_recv_weights[target_rank][dst_idx] = all_weights[src_rank][t, k_slot]

    return ref_recv_x, ref_recv_weights, recv_counts


# ============================================================================
# Kernel builders (cached per GPU)
# ============================================================================
_kernel_cache = {}


def _build_on_gpu(gpu_id, build_fn):
    """Build a kernel ensuring it's loaded on the specified GPU."""
    torch.cuda.set_device(gpu_id)
    kernel = build_fn()
    _ = kernel.launcher  # Force eager module loading
    return kernel


def get_compute_offset_kernel(num_experts, gpu_id=0):
    key = ("compute_offset", num_experts, gpu_id)
    if key not in _kernel_cache:
        actual_experts_plus_1 = max(num_experts + 1, MAX_EXPERTS_PLUS_1)
        smem = NUM_WARPS * actual_experts_plus_1 * 4
        _kernel_cache[key] = _build_on_gpu(
            gpu_id, lambda: kernel_compute_offset.build(
                passes=PASSES["cuda"],
                codegen_func=codegen_cuda,
                grid=(4, 1, 1),
                block=(BLOCK_SIZE, 1, 1),
                shared_mem_bytes=smem,
            ))
    return _kernel_cache[key]


def get_dispatch_layout_kernel(gpu_id=0):
    key = ("dispatch_layout", gpu_id)
    if key not in _kernel_cache:
        _kernel_cache[key] = _build_on_gpu(
            gpu_id, lambda: kernel_compute_dispatch_layout.build(
                passes=PASSES["cuda"],
                codegen_func=codegen_cuda,
                grid=(4, 1, 1),
                block=(BLOCK_SIZE, 1, 1),
                shared_mem_bytes=NUM_WARPS * 4,
            ))
    return _kernel_cache[key]


def get_dispatch_v1_kernel(gpu_id=0):
    key = ("dispatch_v1", gpu_id)
    if key not in _kernel_cache:
        _kernel_cache[key] = _build_on_gpu(
            gpu_id, lambda: kernel_dispatch_intranode_v1.build(
                passes=PASSES["cuda"],
                codegen_func=codegen_cuda,
                grid=(128, 1, 1),
                block=(V1_TOTAL_THREADS, 1, 1),
                shared_mem_bytes=DISPATCH_SMEM + 4096,
            ))
    return _kernel_cache[key]


def get_barrier_kernel(gpu_id=0):
    key = ("barrier", gpu_id)
    if key not in _kernel_cache:
        _kernel_cache[key] = _build_on_gpu(
            gpu_id, lambda: kernel_barrier_all_on_stream.build(
                passes=PASSES["cuda"],
                codegen_func=codegen_cuda,
                grid=(1, 1, 1),
                block=(BARRIER_THREADS, 1, 1),
                shared_mem_bytes=0,
            ))
    return _kernel_cache[key]


# ============================================================================
# Main multi-GPU test
# ============================================================================
def test_multi_gpu_dispatch(
    num_ranks: int,
    num_experts: int = 64,
    topk: int = 8,
    num_token: int = 256,
    hidden_size: int = 128,
    seed: int = 42,
):
    print(f"\n{'='*70}")
    print(f"Multi-GPU Dispatch Test: {num_ranks} ranks")
    print(f"  num_experts={num_experts}, topk={topk}, num_token={num_token}, "
          f"hidden={hidden_size}, seed={seed}")
    print(f"{'='*70}")

    assert num_experts % num_ranks == 0
    experts_per_rank = num_experts // num_ranks

    torch.manual_seed(seed)

    gpu_ids = find_free_gpus(num_ranks)
    if len(gpu_ids) < num_ranks:
        print(f"  SKIP: need {num_ranks} GPUs but only {len(gpu_ids)} available")
        return None
    print(f"  Using GPUs: {gpu_ids}")

    # Create PyTorch contexts and enable peer access
    for gid in gpu_ids:
        torch.cuda.set_device(gid)
        _ = torch.zeros(1, device=f'cuda:{gid}')
    enable_peer_access(gpu_ids)

    # Build kernels
    print("  Building kernels...")
    t_build = time.time()
    for gid in gpu_ids:
        get_compute_offset_kernel(num_experts, gid)
        get_dispatch_layout_kernel(gid)
        get_dispatch_v1_kernel(gid)
        get_barrier_kernel(gid)
    print(f"    Built in {time.time()-t_build:.1f}s")

    # ---- Allocate per-rank data ----
    max_recv = num_token * num_ranks * topk + 1024

    all_topk_indices = []
    all_x = []
    all_weights = []

    # Cross-GPU buffers: allocated via raw cudaMalloc for P2P accessibility
    recv_x_bufs = []  # [num_ranks] of CudaMallocBuffer
    recv_weights_bufs = []  # [num_ranks] of CudaMallocBuffer
    recv_scatter_bufs = []  # [num_ranks] of CudaMallocBuffer
    barrier_bufs = []  # [num_ranks] of CudaMallocBuffer
    full_splits_bufs = []  # [num_ranks] of CudaMallocBuffer

    for r, gpu_id in enumerate(gpu_ids):
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

        # Input data: regular PyTorch tensors (single-GPU access only)
        ti = torch.randint(0, num_experts + 1, (num_token, topk), device=device, dtype=torch.int32)
        all_topk_indices.append(ti)

        x_data = torch.arange(num_token, device=device, dtype=torch.float32).unsqueeze(1)
        x_data = (x_data + r * 1000) * 0.01
        x_data = x_data.expand(-1, hidden_size).contiguous().to(torch.bfloat16)
        all_x.append(x_data)

        w_data = torch.rand(num_token, topk, device=device, dtype=torch.float32)
        all_weights.append(w_data)

        # Cross-GPU buffers: raw cudaMalloc for P2P access
        recv_x_bufs.append(CudaMallocBuffer(max_recv * hidden_size * 2, gpu_id))  # bf16
        recv_weights_bufs.append(CudaMallocBuffer(max_recv * topk * 4, gpu_id))  # f32
        recv_scatter_bufs.append(CudaMallocBuffer(max_recv * topk * 4, gpu_id))  # i32
        barrier_bufs.append(CudaMallocBuffer(num_ranks * 4, gpu_id))  # i32
        full_splits_bufs.append(CudaMallocBuffer(num_ranks * (num_experts + 1) * 4, gpu_id))  # i32

    for gid in gpu_ids:
        _libcudart.cudaSetDevice(gid)
        _libcudart.cudaDeviceSynchronize()

    # ---- Phase 1a: Compute offset (per-rank, no cross-rank deps) ----
    print("  Phase 1a: compute_offset (per-rank) ...")
    t0 = time.time()

    results = {}
    for r, gpu_id in enumerate(gpu_ids):
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        num_token_r = all_topk_indices[r].shape[0]
        total_elements = num_token_r * topk
        num_tiles = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

        block_cumsum_hist = torch.zeros((num_tiles, num_experts + 1), device=device, dtype=torch.int32)
        token_within_expert_offset = torch.zeros((num_token_r, topk), device=device, dtype=torch.int32)
        expert_counts = torch.zeros((num_experts + 1, ), device=device, dtype=torch.int32)

        get_compute_offset_kernel(num_experts, gpu_id)(
            all_topk_indices[r],
            num_token_r,
            topk,
            num_experts,
            block_cumsum_hist,
            token_within_expert_offset,
            expert_counts,
        )
        torch.cuda.synchronize(gpu_id)

        results[r] = {
            "token_within_expert_offset": token_within_expert_offset,
            "expert_counts": expert_counts,
        }
    print(f"    Done ({time.time()-t0:.2f}s)")

    # ---- Phase 1b: Dispatch layout (cross-rank barrier, must run concurrently) ----
    print("  Phase 1b: dispatch_layout (concurrent, cross-rank barrier) ...")
    t0b = time.time()
    errors = []

    def layout_worker(rank, gpu_id):
        try:
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
            num_token_r = all_topk_indices[rank].shape[0]

            full_splits_ptrs = make_ptr_tensor(full_splits_bufs, gpu_id)
            barrier_ptrs = make_ptr_tensor(barrier_bufs, gpu_id)

            recv_base_offset = torch.zeros(num_ranks * experts_per_rank * num_ranks, device=device, dtype=torch.int32)
            token_dst_scatter_indices = torch.full((num_token_r, topk), -1, device=device, dtype=torch.int32)
            token_topk_send_mask = torch.zeros((num_token_r, topk), device=device, dtype=torch.int32)
            recv_token_count = torch.zeros(num_ranks, device=device, dtype=torch.int32)

            get_dispatch_layout_kernel(gpu_id)(
                all_topk_indices[rank],
                results[rank]["token_within_expert_offset"],
                results[rank]["expert_counts"],
                full_splits_ptrs,
                barrier_ptrs,
                recv_base_offset,
                token_dst_scatter_indices,
                token_topk_send_mask,
                0,  # recv_token_count_cpu
                recv_token_count,
                num_token_r,
                topk,
                num_experts,
                rank,
                num_ranks,
            )
            torch.cuda.synchronize(gpu_id)

            results[rank].update({
                "recv_base_offset": recv_base_offset,
                "token_dst_scatter_indices": token_dst_scatter_indices,
                "token_topk_send_mask": token_topk_send_mask,
                "recv_token_count": recv_token_count,
            })
        except Exception:
            import traceback
            errors.append((rank, traceback.format_exc()))

    threads = []
    for r, gpu_id in enumerate(gpu_ids):
        t = threading.Thread(target=layout_worker, args=(r, gpu_id))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)  # 60s timeout for deadlock detection

    if errors:
        for rank, err in errors:
            print(f"  ERROR on rank {rank}:\n{err}")
        return False

    # Check if any thread is still alive (deadlock)
    for i, t in enumerate(threads):
        if t.is_alive():
            print(f"  DEADLOCK: rank {i} thread still alive after 60s")
            return False

    print(f"    Done ({time.time()-t0b:.2f}s)")
    for r in range(num_ranks):
        rtc = results[r]["recv_token_count"].cpu().tolist()
        sm = results[r]["token_topk_send_mask"].sum().item()
        vs = (results[r]["token_dst_scatter_indices"] >= 0).sum().item()
        print(f"    Rank {r} (GPU {gpu_ids[r]}) recv_count={rtc}, send_mask_sum={sm}, valid_scatter={vs}")

    # Reset barrier buffers
    for buf in barrier_bufs:
        buf.zero_()
    for gid in gpu_ids:
        _libcudart.cudaSetDevice(gid)
        _libcudart.cudaDeviceSynchronize()

    # ---- Phase 2: Dispatch (cross-rank) ----
    print("  Phase 2: dispatch_intranode_v1 (cross-rank) ...")
    t1 = time.time()
    errors = []

    def dispatch_worker(rank, gpu_id):
        try:
            torch.cuda.set_device(gpu_id)
            r = results[rank]
            recv_x_ptrs = make_ptr_tensor(recv_x_bufs, gpu_id)
            recv_w_ptrs = make_ptr_tensor(recv_weights_bufs, gpu_id)
            recv_s_ptrs = make_ptr_tensor(recv_scatter_bufs, gpu_id)

            get_dispatch_v1_kernel(gpu_id)(
                all_x[rank],
                r["token_topk_send_mask"],
                all_weights[rank],
                all_topk_indices[rank],
                r["token_dst_scatter_indices"],
                num_token,
                hidden_size,
                experts_per_rank,
                rank,
                num_ranks,
                recv_x_ptrs,
                recv_w_ptrs,
                recv_s_ptrs,
            )
        except Exception:
            import traceback
            errors.append((rank, traceback.format_exc()))

    # Launch dispatch on all ranks concurrently
    threads = []
    for r, gpu_id in enumerate(gpu_ids):
        t = threading.Thread(target=dispatch_worker, args=(r, gpu_id))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        for rank, err in errors:
            print(f"  ERROR on rank {rank}:\n{err}")
        return False

    for gid in gpu_ids:
        torch.cuda.synchronize(gid)
    print(f"    Done ({time.time()-t1:.2f}s)")

    # Quick check: did dispatch write any non-zero data?
    for r, gpu_id in enumerate(gpu_ids):
        rx = recv_x_bufs[r].to_torch(dtype=torch.bfloat16, shape=(max_recv, hidden_size))
        nonzero_rows = (rx != 0).any(dim=1).sum().item()
        print(f"    Rank {r} recv_x: {nonzero_rows} non-zero rows (of {max_recv})")

    # ---- Phase 3: Barrier ----
    print("  Phase 3: barrier ...")
    for buf in barrier_bufs:
        buf.zero_()
    for gid in gpu_ids:
        _libcudart.cudaSetDevice(gid)
        _libcudart.cudaDeviceSynchronize()

    t2 = time.time()
    errors = []

    def barrier_worker(rank, gpu_id):
        try:
            torch.cuda.set_device(gpu_id)
            barrier_ptrs = make_ptr_tensor(barrier_bufs, gpu_id)
            get_barrier_kernel(gpu_id)(barrier_ptrs, rank, num_ranks)
        except Exception:
            import traceback
            errors.append((rank, traceback.format_exc()))

    threads = []
    for r, gpu_id in enumerate(gpu_ids):
        t = threading.Thread(target=barrier_worker, args=(r, gpu_id))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        for rank, err in errors:
            print(f"  ERROR on rank {rank}:\n{err}")
        return False

    for gid in gpu_ids:
        torch.cuda.synchronize(gid)
    print(f"    Done ({time.time()-t2:.2f}s)")

    # ---- Verification ----
    print("  Verifying results ...")
    ref_device = f"cuda:{gpu_ids[0]}"
    torch.cuda.set_device(gpu_ids[0])

    all_pass = True

    # Check recv_token_count via reference
    all_topk_ref = [t.to(ref_device) for t in all_topk_indices]
    all_x_ref = [t.to(ref_device) for t in all_x]
    all_w_ref = [t.to(ref_device) for t in all_weights]
    _, _, ref_recv_counts = reference_dispatch(all_topk_ref, all_x_ref, all_w_ref, num_experts, num_ranks)

    for r in range(num_ranks):
        actual = results[r]["recv_token_count"].to(ref_device)
        expected = ref_recv_counts[r].item()
        got = actual[r].item()
        if got != expected:
            print(f"    FAIL: rank {r} recv_token_count: got {got}, expected {expected}")
            all_pass = False
        else:
            print(f"    PASS: rank {r} recv_token_count = {got}")

    # Check recv_x at positions where dispatch actually wrote data.
    # The dispatch kernel sends x data only for topk slots with send_mask==1.
    # For each (src_rank, token, topk_slot) with send_mask==1:
    #   target_rank = topk_indices[token, slot] // experts_per_rank
    #   dst_idx = token_dst_scatter_indices[token, slot]
    #   recv_x[target_rank][dst_idx] should == x[src_rank][token]

    # Collect actual recv_x for each target rank
    actual_recv_x_all = {}
    for r, gpu_id in enumerate(gpu_ids):
        actual_recv_x_all[r] = recv_x_bufs[r].to_torch(dtype=torch.bfloat16,
                                                       shape=(max_recv, hidden_size)).to(ref_device)

    total_checks = 0
    total_matched = 0
    for src_rank in range(num_ranks):
        send_mask = results[src_rank]["token_topk_send_mask"].to(ref_device)
        scatter_idx = results[src_rank]["token_dst_scatter_indices"].to(ref_device)
        topk_idx = all_topk_indices[src_rank].to(ref_device)
        src_x = all_x[src_rank].to(ref_device)

        for t in range(num_token):
            for k in range(topk):
                if send_mask[t, k].item() == 1:
                    expert = topk_idx[t, k].item()
                    if expert >= num_experts:
                        continue
                    target_rank = expert // experts_per_rank
                    dst = scatter_idx[t, k].item()
                    if dst < 0:
                        continue

                    actual_val = actual_recv_x_all[target_rank][dst]
                    expected_val = src_x[t]
                    total_checks += 1
                    if torch.allclose(actual_val.float(), expected_val.float(), atol=1e-2, rtol=1e-2):
                        total_matched += 1
                    elif total_checks - total_matched <= 5:
                        print(f"      MISMATCH: src_rank={src_rank} token={t} slot={k} "
                              f"-> rank={target_rank} idx={dst}: "
                              f"actual[0]={actual_val[0].item():.4f} "
                              f"expected[0]={expected_val[0].item():.4f}")

    if total_checks == 0:
        print("    WARNING: no dispatch checks performed")
        all_pass = False
    elif total_matched == total_checks:
        print(f"    PASS: recv_x matches for all {total_checks} dispatched tokens")
    else:
        print(f"    FAIL: recv_x mismatch: {total_matched}/{total_checks} tokens matched")
        all_pass = False

    if all_pass:
        print(f"\n  ALL PASS for {num_ranks}-GPU dispatch test!")
    else:
        print(f"\n  SOME TESTS FAILED for {num_ranks}-GPU test!")

    # Cleanup
    for bufs in [recv_x_bufs, recv_weights_bufs, recv_scatter_bufs, barrier_bufs, full_splits_bufs]:
        for buf in bufs:
            buf.free()

    return all_pass


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-ranks", type=int, default=0, help="0 = auto test 2,4,8")
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-token", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("FlashComm Multi-GPU Dispatch Pipeline Test")
    print("=" * 70)

    if args.num_ranks > 0:
        rank_configs = [args.num_ranks]
    else:
        rank_configs = [2, 4, 8]

    all_results = []
    for nr in rank_configs:
        if nr > torch.cuda.device_count():
            print(f"\nSKIP: {nr}-GPU test (only {torch.cuda.device_count()} GPUs)")
            continue

        passed = test_multi_gpu_dispatch(
            num_ranks=nr,
            num_experts=args.num_experts,
            topk=args.topk,
            num_token=args.num_token,
            hidden_size=args.hidden,
            seed=args.seed,
        )
        all_results.append((nr, passed))

    print(f"\n{'='*70}")
    print("Summary:")
    for nr, passed in all_results:
        status = "PASS" if passed else ("FAIL" if passed is False else "SKIP")
        print(f"  {status}: {nr}-GPU dispatch test")
    total_pass = sum(1 for _, p in all_results if p)
    total = sum(1 for _, p in all_results if p is not None)
    print(f"\n  {total_pass}/{total} tests passed")
    print("=" * 70)

    if total_pass < total:
        sys.exit(1)
