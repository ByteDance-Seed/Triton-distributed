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
"""NVIDIA kernel construction from a shared build request."""
from little_kernel.core.compile import _lk_cache_dir as resolve_cache_dir, _normalize_cluster_shape as normalize_cluster_shape


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
    cluster_shape = opts.get("cluster_shape")
    extra_nvcc_options = opts.get("extra_nvcc_options")
    allow_non_portable_cluster = opts.get("allow_non_portable_cluster", False)
    if codegen_func is None:
        from ..codegen.emitter import codegen_cuda
        codegen_func = codegen_cuda
    from .kernel import CompiledKernel

    requested_cluster_shape = normalize_cluster_shape(cluster_shape)
    resolved_cluster_shape = self.cluster_shape
    if (requested_cluster_shape is not None and resolved_cluster_shape is not None
            and requested_cluster_shape != resolved_cluster_shape):
        raise ValueError(f"cluster_shape {requested_cluster_shape} does not match the kernel "
                         f"declaration {resolved_cluster_shape}")
    if requested_cluster_shape is not None:
        resolved_cluster_shape = requested_cluster_shape

    # Calculate total number of threads per block for __launch_bounds__
    num_threads = block[0] * block[1] * block[2]

    # disable_var_hoisting is now a named parameter

    # Generate CUDA code with __launch_bounds__ if num_threads > 0, and
    # carry the actual post-constexpr KernelSignature to the launcher.
    # Resolve the cache dir locally: another compilation must never be
    # able to overwrite it before CompiledKernel is constructed.
    resolved_cache_dir = resolve_cache_dir(cache_dir)
    # Require a typed host ABI unless this is the explicit legacy
    # untyped/raw-source path carrying a caller-supplied signature:
    # an untyped kernel with no override must fail honestly (never a
    # silently downgraded launch), while a typed kernel propagates any
    # real ABI construction error regardless of this flag.
    artifact = self.compile_artifact(passes, codegen_func, target=arch, need_header=True,
                                     num_threads=num_threads if num_threads > 0 else None,
                                     disable_var_hoisting=disable_var_hoisting, cache_dir=resolved_cache_dir,
                                     frontend=frontend, require_abi=(signature is None
                                                                     or bool(self.py_func.__annotations__)))
    cuda_code = artifact.source

    kernel_name = self.py_func.__name__
    # The frontend-derived signature is authoritative. A raw-source
    # caller may supply one only for the explicitly untyped legacy path.
    launch_signature = artifact.signature
    if launch_signature is None and signature is not None:
        launch_signature = signature
    kwargs = dict(
        cluster_shape=resolved_cluster_shape,
        cta_group_shape=self.cta_group_shape,
        extra_nvcc_options=extra_nvcc_options,
        allow_non_portable_cluster=allow_non_portable_cluster,
        signature=launch_signature,
    )
    return CompiledKernel(
        cuda_code,
        kernel_name,
        grid,
        block,
        shared_mem_bytes,
        arch,
        resolved_cache_dir,
        verbose,
        include_paths,
        **kwargs,
    )
