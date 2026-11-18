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
import hashlib
import shutil
import subprocess
import threading
from pathlib import Path

from .compiler import compile_hip, get_hip_arch
from .hip_runtime import get_hip_runtime, pack_kernel_args, parse_kernel_params
from little_kernel.runtime.abi import KernelSignature, pack_contiguous_arguments
from little_kernel.runtime.cache import (AtomicBinaryCache, BuildIdentity, normalize_options, normalize_include_paths,
                                         scan_user_dependencies, expand_option_dependencies)


class CompiledHipKernel:
    """HIP executable using shared typed arguments, identity and atomic storage."""

    def __init__(self, hip_code, kernel_name, grid, block, shared_mem_bytes=0, arch=None, cache_dir=None, verbose=False,
                 include_paths=None, extra_hipcc_options=None, signature=None, runtime=None, compiler=None,
                 toolchain_id=None):
        if signature is not None and not isinstance(signature, KernelSignature):
            raise TypeError("signature must be a KernelSignature")
        self.hip_code, self.kernel_name = hip_code, kernel_name
        self.grid, self.block = tuple(grid), tuple(block)
        self.shared_mem_bytes = shared_mem_bytes
        self.arch = arch or get_hip_arch()
        self.cache_dir, self.verbose = cache_dir, verbose
        self.include_paths = tuple(include_paths or ())
        self.extra_hipcc_options = tuple(extra_hipcc_options or ())
        self.signature = signature
        self.param_formats = None if signature is not None else parse_kernel_params(hip_code, kernel_name)
        self._runtime, self._compiler = runtime, compiler or compile_hip
        self._toolchain_id = toolchain_id
        self._code_object = self._module = self._function = None
        self._closed = False
        self._lock = threading.RLock()

    def _check_open(self):
        if self._closed:
            raise RuntimeError("CompiledHipKernel is closed")

    @property
    def runtime(self):
        self._check_open()
        return self._runtime if self._runtime is not None else get_hip_runtime()

    @property
    def build_identity(self):
        toolchain = self._toolchain_id
        if toolchain is None:
            compiler = shutil.which("hipcc")
            if compiler is None:
                toolchain = "hipcc:unavailable"
            else:
                result = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=False)
                toolchain = "hipcc:" + hashlib.sha256(
                    (str(Path(compiler).resolve()) + result.stdout + result.stderr).encode()).hexdigest()
        options = normalize_options(self.extra_hipcc_options)
        includes = normalize_include_paths(self.include_paths)
        system, quoted, forced, responses = expand_option_dependencies(options)
        dependencies = scan_user_dependencies(self.hip_code, includes, (), option_system_dirs=system,
                                              option_quote_dirs=quoted, forced_includes=forced)
        dependencies = tuple(sorted(set(dependencies) | {(f"response:{h}", "response") for h in responses}))
        return BuildIdentity(self.kernel_name,
                             hashlib.sha256(self.hip_code.encode()).hexdigest(), self.arch, options, includes,
                             dependencies, toolchain)

    def _get_cache_key(self):
        return self.build_identity.cache_key

    @property
    def code_object(self):
        with self._lock:
            self._check_open()
            if self._code_object is None:

                def compile_binary():
                    return self._compiler(self.hip_code, self.kernel_name, arch=self.arch,
                                          options=list(self.extra_hipcc_options), verbose=self.verbose,
                                          include_paths=list(self.include_paths) or None)

                if self.cache_dir:
                    cache = AtomicBinaryCache(self.cache_dir, binary_suffix=".hsaco", source_suffix=".hip")
                    self._code_object = cache.get_or_build(self.build_identity, compile_binary, self.hip_code)
                else:
                    self._code_object = compile_binary()
            return self._code_object

    @property
    def function(self):
        with self._lock:
            self._check_open()
            if self._module is None:
                self._module = self.runtime.load_module(self.code_object)
            if self._function is None:
                self._function = self.runtime.get_function(self._module, self.kernel_name)
            return self._function

    def __call__(self, *args, stream=None, grid=None):
        with self._lock:
            self._check_open()
            packed = (pack_contiguous_arguments(self.signature, args)
                      if self.signature is not None else pack_kernel_args(self.param_formats, args))
            self.runtime.launch(self.function, grid or self.grid, self.block, packed, self.shared_mem_bytes, stream)

    def synchronize(self, stream=None):
        self.runtime.synchronize(stream)

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._module is not None:
                self.runtime.unload_module(self._module)
                self._module = self._function = None
            self._closed = True
