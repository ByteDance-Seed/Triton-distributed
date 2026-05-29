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
import shutil
import torch
import triton
import triton.language as tl
import triton_dist.language.simt_ops as simt_ops
import triton_dist.language.extra.hip.language_extra as dl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ---------------------------------------------------------------------------
# Kernel: simt vector add  —  mirrors common/test_simt_vec_add.py::simt_add
# ---------------------------------------------------------------------------
@triton.jit
def simt_add(x, y, out, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_tiles = tl.cdiv(n, BLOCK_SIZE)
    vec_size: tl.constexpr = 128 // x.dtype.element_ty.primitive_bitwidth
    tl.static_assert(BLOCK_SIZE % vec_size == 0)
    with simt_ops.simt_exec_region() as (thread_idx, num_threads):
        for tile_id in range(pid, num_tiles, num_pids):
            start_vec = tile_id * BLOCK_SIZE
            end_vec = min(n, start_vec + BLOCK_SIZE) // vec_size * vec_size
            for j in range(start_vec + thread_idx * vec_size, end_vec, num_threads * vec_size):
                vec_x = dl.ld_vector(x + j, vec_size=vec_size)
                vec_y = dl.ld_vector(y + j, vec_size=vec_size)
                vec_z = (vec_x + vec_y + 1) * 2
                dl.st_vector(out + j, vec_z)

            if start_vec + BLOCK_SIZE >= n:
                for j in range(end_vec + thread_idx, n, num_threads):
                    vec_x = dl.ld_vector(x + j, vec_size=1)
                    vec_y = dl.ld_vector(y + j, vec_size=1)
                    vec_z = (vec_x + vec_y + 1) * 2
                    dl.st_vector(out + j, vec_z)


# ---------------------------------------------------------------------------
# Simple copy kernel  —  used for basic vec copy correctness checks
# ---------------------------------------------------------------------------
@triton.jit
def simt_copy(src, dst, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_tiles = tl.cdiv(n, BLOCK_SIZE)
    vec_size: tl.constexpr = 128 // src.dtype.element_ty.primitive_bitwidth
    tl.static_assert(BLOCK_SIZE % vec_size == 0)
    with simt_ops.simt_exec_region() as (thread_idx, num_threads):
        for tile_id in range(pid, num_tiles, num_pids):
            start_vec = tile_id * BLOCK_SIZE
            end_vec = min(n, start_vec + BLOCK_SIZE) // vec_size * vec_size
            for j in range(start_vec + thread_idx * vec_size, end_vec, num_threads * vec_size):
                vec = dl.ld_vector(src + j, vec_size=vec_size)
                dl.st_vector(dst + j, vec)

            if start_vec + BLOCK_SIZE >= n:
                for j in range(end_vec + thread_idx, n, num_threads):
                    vec = dl.ld_vector(src + j, vec_size=1)
                    dl.st_vector(dst + j, vec)


# ---------------------------------------------------------------------------
# test helpers
# ---------------------------------------------------------------------------
def _make_data(n, dtype, device):
    if dtype.is_floating_point:
        x = torch.randn((n, ), dtype=dtype, device=device)
        y = torch.randn((n, ), dtype=dtype, device=device)
    else:
        x = torch.randint(0, 32768, (n, ), dtype=dtype, device=device)
        y = torch.randint(0, 32768, (n, ), dtype=dtype, device=device)
    return x, y


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------
def test_simt_add_vec(n, dtype, device):
    x, y = _make_data(n, dtype, device)
    out_ref = (x + y + 1) * 2
    out_triton = torch.empty_like(out_ref)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
    simt_add[grid](x, y, out_triton, n, BLOCK_SIZE=1024)
    if dtype.is_floating_point:
        torch.testing.assert_close(out_ref, out_triton, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(out_ref, out_triton, atol=0, rtol=0)


def test_simt_copy(n, dtype, device):
    x, _ = _make_data(n, dtype, device)
    out_triton = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
    simt_copy[grid](x, out_triton, n, BLOCK_SIZE=1024)
    torch.testing.assert_close(x, out_triton, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Assembly verification — compile a small kernel and inspect its IR/asm
# ---------------------------------------------------------------------------
@triton.jit
def _vec_copy_kernel(src, dst, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    vec_size: tl.constexpr = 128 // src.dtype.element_ty.primitive_bitwidth
    with simt_ops.simt_exec_region() as (thread_idx, num_threads):
        start = pid * BLOCK_SIZE
        end = min(n, start + BLOCK_SIZE) // vec_size * vec_size
        for j in range(start + thread_idx * vec_size, end, num_threads * vec_size):
            vec = dl.ld_vector(src + j, vec_size=vec_size)
            dl.st_vector(dst + j, vec)


def check_asm_vectorization(device):
    """Compile a small vec-copy kernel and verify LLIR + GCN assembly."""
    import re
    import subprocess
    import tempfile

    n = 1024
    x = torch.ones(n, dtype=torch.int32, device=device)
    y = torch.empty_like(x)
    compiled = _vec_copy_kernel[(1, )](x, y, n, BLOCK_SIZE=1024)

    llir = compiled.asm.get("llir", "")
    vec_loads = len(re.findall(r"load <4 x i32>", llir))
    vec_stores = len(re.findall(r"store <4 x i32>", llir))
    print(f"  LLIR  : {vec_loads} vector loads, {vec_stores} vector stores")
    assert vec_loads > 0, "LLIR: no <4 x i32> loads found"
    assert vec_stores > 0, "LLIR: no <4 x i32> stores found"

    hsaco = compiled.asm.get("hsaco", None)
    if hsaco is not None:
        rocm_objdump = "/opt/rocm/llvm/bin/llvm-objdump"
        objdump = rocm_objdump if os.path.isfile(rocm_objdump) \
            else shutil.which("llvm-objdump")
        if not objdump:
            print("  AMDGCN: llvm-objdump not found, skipping GCN assembly check")
            return False
        with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as tmp:
            tmp.write(hsaco)
            tmp_path = tmp.name
        result = subprocess.run([objdump, "-d", tmp_path], capture_output=True, text=True)
        os.unlink(tmp_path)
        asm = result.stdout
        dwordx4_loads = [l.strip() for l in asm.split("\n") if "global_load_dwordx4" in l]
        dwordx4_stores = [l.strip() for l in asm.split("\n") if "global_store_dwordx4" in l]
        print(f"  AMDGCN: {len(dwordx4_loads)} global_load_dwordx4, "
              f"{len(dwordx4_stores)} global_store_dwordx4")
        for inst in dwordx4_loads[:3]:
            print(f"    {inst}")
        for inst in dwordx4_stores[:3]:
            print(f"    {inst}")
        assert len(dwordx4_loads) > 0, "AMDGCN: no global_load_dwordx4 found"
        assert len(dwordx4_stores) > 0, "AMDGCN: no global_store_dwordx4 found"
        return True
    else:
        print("  AMDGCN: hsaco not available in compiled.asm")
        return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Use a smaller n than NVIDIA (no perf measurement needed, just correctness)
    N_PERFECT = 8192
    N_NONPERFECT = 8192 - 1

    dtypes_to_test = [
        torch.int16,
        torch.int32,
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ]

    print("=" * 60)
    print("  AMD ld_vector / st_vector — aligned with NVIDIA tests")
    print("=" * 60)

    # 1) Simple copy tests (all dtypes, perfect + non-perfect)
    print("\n--- simt_copy (pure vector copy) ---")
    for dtype in dtypes_to_test:
        for n in [N_PERFECT, N_NONPERFECT]:
            test_simt_copy(n, dtype, DEVICE)
            tag = "perfect" if n == N_PERFECT else "tail"
            print(f"  [copy] dtype={str(dtype):20s}  n={n:6d} ({tag})  PASS")

    # 2) Vector add tests (all dtypes, perfect + non-perfect)
    print("\n--- simt_add (vector arithmetic) ---")
    for dtype in dtypes_to_test:
        for n in [N_PERFECT, N_NONPERFECT]:
            test_simt_add_vec(n, dtype, DEVICE)
            tag = "perfect" if n == N_PERFECT else "tail"
            print(f"  [add]  dtype={str(dtype):20s}  n={n:6d} ({tag})  PASS")

    # 3) Assembly verification — compile a dedicated kernel and inspect
    print("\n--- Assembly vectorization check (dedicated kernel) ---")
    has_v4 = check_asm_vectorization(DEVICE)
    if has_v4:
        print("  ==> TRUE VECTORIZATION CONFIRMED (dwordx4)")
    else:
        print("  ==> WARNING: dwordx4 not found")

    print("\nAll tests passed!")
