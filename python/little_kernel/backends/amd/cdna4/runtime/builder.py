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
"""CDNA4 HIP kernel construction from a shared build request."""
from little_kernel.core.compile import _lk_cache_dir as resolve_cache_dir


def build_kernel(request):
    self = request.kernel
    passes = request.passes
    codegen_func = request.codegen
    grid, block = request.grid, request.block
    shared_mem_bytes, arch = request.shared_mem_bytes, request.architecture
    cache_dir, verbose, include_paths = request.cache_dir, request.verbose, request.include_paths
    opts = request.options
    disable_var_hoisting = opts.get("disable_var_hoisting", False)
    frontend, signature = opts.get("frontend"), opts.get("signature")
    extra_hipcc_options = opts.get("extra_hipcc_options")
    waves_per_eu = opts.get("waves_per_eu")
    if codegen_func is None:
        from ..codegen.emitter import codegen_hip
        codegen_func = codegen_hip
    from .hip_kernel import CompiledHipKernel
    num_threads = block[0] * block[1] * block[2]
    options = {"waves_per_eu": waves_per_eu} if waves_per_eu is not None else None
    artifact = self.compile_artifact(passes, codegen_func, target=arch, need_header=True, num_threads=num_threads
                                     or None, disable_var_hoisting=disable_var_hoisting, cache_dir=cache_dir,
                                     codegen_options=options, frontend=frontend,
                                     require_abi=(signature is None or bool(self.py_func.__annotations__)))
    source = artifact.source
    launch_signature = artifact.signature if artifact.signature is not None else signature
    return CompiledHipKernel(source, self.py_func.__name__, grid, block, shared_mem_bytes, arch,
                             resolve_cache_dir(cache_dir), verbose, include_paths, extra_hipcc_options,
                             signature=launch_signature)
