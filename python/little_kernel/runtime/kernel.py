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

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple, Any, List, Union
from .compiler import compile_cuda
from .cuda_runtime import CUDARuntime, KernelLauncher


class CompiledKernel:
    """Compiled CUDA kernel that can be launched from Python."""

    def __init__(
        self,
        cuda_code: str,
        kernel_name: str,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        shared_mem_bytes: int = 0,
        arch: Optional[str] = None,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
        include_paths: Optional[List[Union[str, Path]]] = None,
        cluster_dim: Optional[Tuple[int, int, int]] = None,
    ):
        self.cuda_code = cuda_code
        self.kernel_name = kernel_name
        self.grid = grid
        self.block = block
        self.shared_mem_bytes = shared_mem_bytes
        self.arch = arch
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.include_paths = include_paths
        self.cluster_dim = cluster_dim

        self._runtime = None
        self._module = None
        self._launcher = None
        self._cubin_data = None

    @property
    def runtime(self) -> CUDARuntime:
        if self._runtime is None:
            self._runtime = CUDARuntime()
        return self._runtime

    def _get_cache_key(self) -> str:
        code_hash = hashlib.sha256(self.cuda_code.encode()).hexdigest()[:16]
        arch_str = self.arch or "auto"
        return f"{self.kernel_name}_{code_hash}_{arch_str}"

    def _compile(self) -> bytes:
        if self.cache_dir:
            cache_key = self._get_cache_key()
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.cubin")
            if os.path.exists(cache_file):
                if self.verbose:
                    print(f"Loading cached binary: {cache_file}")
                with open(cache_file, "rb") as f:
                    return f.read()

        if self.verbose:
            print(f"Compiling kernel: {self.kernel_name}")

        cubin = compile_cuda(
            self.cuda_code,
            self.kernel_name,
            arch=self.arch,
            target_format="cubin",
            verbose=self.verbose,
            cache_dir=self.cache_dir,
            include_paths=self.include_paths,
        )

        if self.cache_dir:
            cache_key = self._get_cache_key()
            os.makedirs(self.cache_dir, exist_ok=True)
            cubin_file = os.path.join(self.cache_dir, f"{cache_key}.cubin")
            with open(cubin_file, "wb") as f:
                f.write(cubin)
            # Also cache CUDA source for debugging and reproducibility
            cu_file = os.path.join(self.cache_dir, f"{cache_key}.cu")
            with open(cu_file, "w", encoding="utf-8") as f:
                f.write(self.cuda_code)
            if self.verbose:
                print(f"Cached binary: {cubin_file}")
                print(f"Cached source: {cu_file}")

        return cubin

    @property
    def module(self):
        if self._module is None:
            cubin = self._compile()
            self._cubin_data = cubin
            self._module = self.runtime.load_module(cubin)
        return self._module

    @property
    def launcher(self) -> KernelLauncher:
        if self._launcher is None:
            _ = self.module
            self._launcher = KernelLauncher(
                self.runtime,
                self.module,
                self.kernel_name,
                self.shared_mem_bytes,
                self._cubin_data,
            )
        return self._launcher

    def __call__(self, *args, stream: Optional[Any] = None, grid: Optional[Tuple[int, int, int]] = None):
        g = grid if grid is not None else self.grid
        self.launcher(g, self.block, *args, stream=stream, cluster_dim=self.cluster_dim)

    def synchronize(self, stream: Optional[Any] = None):
        self.runtime.synchronize(stream)


def compile_and_load_kernel(
    cuda_code: str,
    kernel_name: str,
    grid: Tuple[int, int, int],
    block: Tuple[int, int, int],
    shared_mem_bytes: int = 0,
    arch: Optional[str] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = False,
    include_paths: Optional[List[Union[str, Path]]] = None,
) -> CompiledKernel:
    return CompiledKernel(
        cuda_code,
        kernel_name,
        grid,
        block,
        shared_mem_bytes,
        arch,
        cache_dir,
        verbose,
        include_paths,
    )
