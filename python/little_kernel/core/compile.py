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

import ast
import inspect
import textwrap
from pathlib import Path
from typing import Optional, Tuple, List, Union
from .internal import __LITTLE_KERNEL_ENTRY__


def _is_little_kernel_related(value) -> bool:
    """True if value is safe to add to ctx (whitelist for closure vars)."""
    if value is None:
        return False
    # little_kernel.language module (e.g. user's "ll")
    if hasattr(value, "__name__") and getattr(value, "__name__", "").startswith("little_kernel."):
        return True
    # Type or instance: allow if type's module is little_kernel
    mod = getattr(type(value), "__module__", None)
    if mod is not None and mod.startswith("little_kernel."):
        return True
    # Value itself may be a module
    mod = getattr(value, "__module__", None)
    if mod is not None and mod.startswith("little_kernel."):
        return True
    # Allow classes (e.g. struct stub classes) so type inference can resolve SpecialStruct(), etc.
    if isinstance(value, type):
        return True
    return False


class LLKernel:

    def __init__(self, py_func, backend, is_entry):
        self.py_func = py_func
        self.ctx = py_func.__globals__.copy()
        self.backend = backend

        # Add closure variables to context only if little_kernel-related (whitelist).
        # Only ValueError is caught: cell.cell_contents raises ValueError when the cell is
        # empty/unbound. We do not catch other exceptions (explicit catch only).
        if py_func.__closure__ is not None:
            code = py_func.__code__
            for var_name, cell in zip(code.co_freevars, py_func.__closure__):
                try:
                    val = cell.cell_contents
                    if _is_little_kernel_related(val):
                        self.ctx[var_name] = val
                except ValueError:
                    # Cell is empty or unbound (e.g. not yet closed over); skip this freevar.
                    pass

        self.is_entry = is_entry
        if is_entry:
            self.ctx[__LITTLE_KERNEL_ENTRY__] = self.py_func

    def compile(self, passes, codegen_func, need_header=True, num_threads=None):
        """Compile kernel to CUDA code.
        
        Parameters
        ----------
        passes : list
            List of passes to apply
        codegen_func : callable
            Code generation function (e.g., codegen_cuda)
        need_header : bool
            Whether to emit header includes
        num_threads : int, optional
            Number of threads per block for __launch_bounds__ attribute.
            If None, no __launch_bounds__ will be generated.
            Note: If codegen_func is created with functools.partial and already has
            num_threads set, this parameter will override it.
        
        Returns
        -------
        str
            Generated CUDA code
        """
        tree = self.lower(passes)
        # Only pass num_threads if explicitly provided (not None)
        # This allows functools.partial to set default num_threads
        if num_threads is not None:
            return codegen_func(tree, self.ctx, emit_header=need_header, num_threads=num_threads)
        else:
            # Don't pass num_threads, let codegen_func use its default (which may be set by partial)
            return codegen_func(tree, self.ctx, emit_header=need_header)

    def lower(self, passes):
        source = inspect.getsource(self.py_func)
        # Remove indentation for functions defined in non-global scope (e.g., inside classes)
        source = textwrap.dedent(source)

        tree = ast.parse(source)
        for p in passes:
            tree = p(tree, self.ctx)
        return tree

    def __call__(self, *args, **kwargs):
        self.ctx[__LITTLE_KERNEL_ENTRY__] = self.py_func

    def build(
        self,
        passes,
        codegen_func,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        shared_mem_bytes: int = 0,
        arch: Optional[str] = None,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
        include_paths: Optional[List[Union[str, Path]]] = None,
        cluster_dim: Optional[Tuple[int, int, int]] = None,
    ):
        """
        Compile and build a kernel that can be launched.
        
        Parameters
        ----------
        passes : list
            List of passes to apply.
        codegen_func : callable
            Code generation function (e.g., codegen_cuda).
        grid : Tuple[int, int, int]
            Grid dimensions (gridDimX, gridDimY, gridDimZ).
        block : Tuple[int, int, int]
            Block dimensions (blockDimX, blockDimY, blockDimZ).
        shared_mem_bytes : int
            Dynamic shared memory size in bytes.
        arch : Optional[str]
            Target architecture (e.g., "sm_90a").
        cache_dir : Optional[str]
            Directory to cache compiled binaries.
        verbose : bool
            Print compilation output.
        include_paths : Optional[List[Union[str, Path]]]
            List of include paths to add to nvcc command.
        cluster_dim : Optional[Tuple[int, int, int]]
            Cluster dimensions (x, y, z) for SM90+ cluster launch.
        
        Returns
        -------
        CompiledKernel
            Compiled kernel that can be launched.
        """
        from little_kernel.runtime import CompiledKernel

        # Calculate total number of threads per block for __launch_bounds__
        num_threads = block[0] * block[1] * block[2]

        # Generate CUDA code with __launch_bounds__ if num_threads > 0
        cuda_code = self.compile(passes, codegen_func, need_header=True,
                                 num_threads=num_threads if num_threads > 0 else None)

        # Get kernel name
        kernel_name = self.py_func.__name__

        # Create compiled kernel
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
            cluster_dim,
        )


def ll_kernel(backend="cuda", is_entry=False):

    def _compile_helper(func):
        compiled_func = LLKernel(func, backend, is_entry)

        compiled_func.__name__ = func.__name__
        compiled_func.__doc__ = func.__doc__
        compiled_func.__module__ = func.__module__
        return compiled_func

    return _compile_helper
