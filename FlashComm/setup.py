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

from __future__ import annotations

import os
import platform
from pathlib import Path

import setuptools
from setuptools import setup


def get_extension():
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Building flash_comm requires PyTorch with CUDA available. "
                           "Install torch first, then run `python setup.py bdist_wheel`.") from e

    custom_arch = os.getenv('CUSTOM_CUDA_ARCH')
    if custom_arch:
        cuda_arch = custom_arch
    else:
        cuda_arch = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')
    os.environ['TORCH_CUDA_ARCH_LIST'] = cuda_arch
    print(f"Using CUDA architecture: {cuda_arch}")

    # Derive max dynamic shared memory per block (bytes) from target arch.
    # Reference: NVIDIA Programming Guide per-arch shared memory limits.
    #   SM 8.x (Ampere):   164 KB per SM → max per block = 163 KB
    #   SM 9.0 (Hopper):   228 KB per SM → max per block = 227 KB
    #   SM 10.0 (Blackwell): 228 KB per SM → max per block = 227 KB
    _smem_per_block_kb = {8: 163, 9: 227, 10: 227}
    arch_tokens = [a.strip() for a in cuda_arch.replace('+PTX', '').replace('a', '').split(';') if a.strip()]
    max_major = max(int(float(a)) for a in arch_tokens) if arch_tokens else 9
    max_smem_kb = _smem_per_block_kb.get(max_major, 227)
    max_smem_bytes = max_smem_kb * 1024
    print(f"Max dynamic shared memory per block: {max_smem_kb} KB ({max_smem_bytes} bytes) for SM major={max_major}")

    sources = [
        os.path.join("csrc", "ep", "kernels", "intranode_cuda.cu"),
        os.path.join("csrc", "ep", "intranode.cpp"),
        os.path.join("csrc", "bindings.cpp"),
        os.path.join("csrc", "buffer", "pybind.cpp"),
        os.path.join("csrc", "buffer", "shareable_block.cpp"),
        os.path.join("csrc", "buffer", "symmetric_memory.cpp")
    ]
    cur_dir = Path(__file__).resolve().parent
    include_dirs = [str(cur_dir / "include")]
    # Make `-lcuda` resolvable on both x86_64 and aarch64.
    # On many systems only the CUDA stub has `libcuda.so`, while the driver ships `libcuda.so.1`.
    cuda_home = Path(os.getenv("CUDA_HOME", "/usr/local/cuda"))
    arch = platform.machine()
    # CUDA targets naming differs across distros:
    # - x86_64:   targets/x86_64-linux
    # - aarch64:  targets/sbsa-linux (common on ARM servers) or targets/aarch64-linux
    if arch in ("aarch64", "arm64"):
        if (cuda_home / "targets" / "sbsa-linux").exists():
            target = "sbsa-linux"
        else:
            target = "aarch64-linux"
    else:
        target = "x86_64-linux"

    candidate_lib_dirs = [
        cuda_home / "lib64",
        cuda_home / "compat",
        cuda_home / "targets" / target / "lib" / "stubs",
        cuda_home / "targets" / target / "lib",
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/lib/x86_64-linux-gnu"),
    ]
    library_dirs = [str(p) for p in candidate_lib_dirs if p and p.exists()]
    extra_compile_args = {}
    extra_link_args = ['-lcuda', '-lcudart']
    smem_define = f"-DFLASH_COMM_MAX_SMEM_BYTES={max_smem_bytes}"
    nvcc_flags = [
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-Xcompiler",
        "-fPIC,-fvisibility=hidden",
        "-O3",
        smem_define,
    ]
    cxx_flags = [
        "-std=c++17",
        "-fvisibility=hidden",
        smem_define,
    ]
    extra_compile_args["nvcc"] = nvcc_flags
    extra_compile_args["cxx"] = cxx_flags

    print(f"include_dirs={include_dirs}")
    print(f"library_dirs={library_dirs}")
    print(f"extra_link_args={extra_link_args}")
    print(f"sources={sources}")
    print(f"extra_compile_args={extra_compile_args}")

    extension = CUDAExtension(name="flash_comm._C", include_dirs=include_dirs, library_dirs=library_dirs,
                              sources=sources, extra_compile_args=extra_compile_args, extra_link_args=extra_link_args)

    return extension


def _lazy_build_ext():
    from torch.utils.cpp_extension import BuildExtension
    return BuildExtension


def _get_ext_modules():
    return [get_extension()]


def _get_cmdclass():
    return {"build_ext": _lazy_build_ext()}


setup(
    name="flash_comm",
    version="0.0.1",
    description="FlashComm: CUDA communication library",
    package_dir={"": "python"},
    packages=setuptools.find_packages(where="python"),
    include_package_data=True,
    zip_safe=False,
    ext_modules=_get_ext_modules(),
    cmdclass=_get_cmdclass(),
)
